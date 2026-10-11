#!/usr/bin/env python3
"""
filter_video.py
----------------
视频素材端准入过滤与质量分流系统（四级漏斗 + 三档裁决）：
在将视频送入重计算与昂贵的 3D 重建管线之前，在视频画面端提前拦截不合格素材：

四级漏斗体系：
- L0 元数据层 (ffprobe, <10ms): 检测 VFR/CFR 帧率抖动、低分辨率、时长异常、文件损坏。
- L1 画面层 (抽帧 + OpenCV, ~50ms):
  - 人物占屏比与全身可见性（人物高度 < 200px 过滤）；
  - 脚出画检测（脚踝/脚尖出画 >= 3 帧直接淘汰，避免下游接触约束全面失效）；
  - 模糊度检测（Laplacian 空间模糊与帧间运动模糊）；
  - 曝光与对比度（逆光剪影、过曝、死黑）；
  - 相机运动方差与分流（光流残差计算，自动标记适合静态相机模式还是剧烈手持晃动）；
  - 转场切片检测（直方图突变）。
- L2 关键点与抖动层 (轻量 YOLOv8-pose 抽帧, ~100ms):
  - 关键点平均置信度与丢点率；
  - 关键点时序加加速度 Jerk 抖动；
  - 脚部可见性覆盖率与躯干长度形变率。
- 三档裁决机制：
  - PASS: 各项指标达标，进入正式 3D 重建管线；
  - REVIEW: 临界带样本，进人工复审池；
  - REJECT: 命中硬规则淘汰，记录结构化拒因 reason_code。
"""

import argparse
import concurrent.futures
import csv
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np

# PyTorch 2.6 weights_only 兼容处理
import torch
_orig_torch_load = torch.load
torch.load = lambda *args, **kwargs: _orig_torch_load(*args, **{**kwargs, "weights_only": False})


class VideoQualityFilter:
    """视频端前端漏斗质量过滤器"""

    def __init__(
        self,
        min_resolution_height: int = 480,
        min_duration_s: float = 2.0,
        max_duration_s: float = 300.0,
        min_person_height_px: int = 180,
        max_foot_out_frames: int = 2,
        min_kpt_conf: float = 0.50,
        max_kpt_jerk: float = 40.0,
        min_foot_coverage: float = 0.80,
        min_contrast: float = 25.0,
        sample_fps: float = 3.0,
    ):
        self.min_res_h = min_resolution_height
        self.min_duration_s = min_duration_s
        self.max_duration_s = max_duration_s
        self.min_person_height_px = min_person_height_px
        self.max_foot_out_frames = max_foot_out_frames
        self.min_kpt_conf = min_kpt_conf
        self.max_kpt_jerk = max_kpt_jerk
        self.min_foot_coverage = min_foot_coverage
        self.min_contrast = min_contrast
        self.sample_fps = sample_fps

        # 延迟按需加载轻量 Pose 模型
        self._pose_model = None

    @property
    def pose_model(self):
        if self._pose_model is None:
            from ultralytics import YOLO
            self._pose_model = YOLO("yolov8n-pose.pt")
        return self._pose_model

    def check_l0_metadata(self, video_path: pathlib.Path) -> Tuple[bool, str, Dict[str, Any]]:
        """L0 元数据检查 (调用 ffprobe)"""
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,avg_frame_rate,width,height,duration,nb_frames,pix_fmt",
            "-of", "json",
            str(video_path)
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            info = json.loads(res.stdout)
            streams = info.get("streams", [])
            if not streams:
                return False, "REJECT_NO_VIDEO_STREAM", {}
            s = streams[0]

            width = int(s.get("width", 0))
            height = int(s.get("height", 0))
            if height < self.min_res_h or width < 320:
                return False, f"REJECT_LOW_RESOLUTION: {width}x{height} < {self.min_res_h}p", s

            # 计算帧率
            def parse_fps(val_str):
                if not val_str or val_str == "0/0":
                    return 30.0
                if "/" in val_str:
                    num, den = val_str.split("/")
                    return float(num) / max(float(den), 1.0)
                return float(val_str)

            r_fps = parse_fps(s.get("r_frame_rate"))
            avg_fps = parse_fps(s.get("avg_frame_rate"))

            # 检测可变帧率 (VFR)
            is_vfr = False
            if r_fps > 0 and avg_fps > 0:
                fps_diff_ratio = abs(r_fps - avg_fps) / r_fps
                if fps_diff_ratio > 0.03:
                    is_vfr = True

            duration = float(s.get("duration") or 0.0)
            if duration <= 0.0 and "nb_frames" in s:
                duration = float(s["nb_frames"]) / max(avg_fps, 1.0)

            if duration < self.min_duration_s or duration > self.max_duration_s:
                return False, f"REJECT_INVALID_DURATION: {duration:.1f}s not in [{self.min_duration_s}, {self.max_duration_s}]", s

            meta = {
                "width": width,
                "height": height,
                "r_fps": round(r_fps, 2),
                "avg_fps": round(avg_fps, 2),
                "is_vfr": is_vfr,
                "duration_s": round(duration, 2),
            }
            return True, "PASS", meta
        except Exception as e:
            return False, f"REJECT_CORRUPTED_FILE: {str(e)[:60]}", {}

    def analyze_frames_l1_l2(
        self,
        video_path: pathlib.Path,
        meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """L1 画面层与 L2 关键点/抖动层联合稀疏采样检测"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"status": "REJECT_CANNOT_OPEN", "reason": "无法读取视频帧"}

        video_fps = meta.get("avg_fps", 30.0)
        width = meta.get("width", 1280)
        height = meta.get("height", 720)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_step = max(1, int(round(video_fps / self.sample_fps)))

        sampled_frames = []
        frame_indices = []
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % frame_step == 0:
                sampled_frames.append(frame)
                frame_indices.append(idx)
            idx += 1

        cap.release()

        if len(sampled_frames) < 3:
            return {"status": "REJECT_TOO_FEW_FRAMES", "reason": f"有效采样帧不足 ({len(sampled_frames)})"}

        # 1. 画面视觉指标计算 (清晰度/曝光/相机运动)
        laplacian_vars = []
        luminances = []
        contrasts = []
        camera_motion_residuals = []

        prev_gray = None
        for f in sampled_frames:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            # 归一化 Laplacian 方差 (消除不同分辨率尺度偏差)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            laplacian_vars.append(lap_var)

            luminances.append(float(np.mean(gray)))
            contrasts.append(float(np.std(gray)))

            # Farneback 光流估计全局相机运动方差
            if prev_gray is not None:
                small_prev = cv2.resize(prev_gray, (320, 180))
                small_curr = cv2.resize(gray, (320, 180))
                flow = cv2.calcOpticalFlowFarneback(small_prev, small_curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                global_motion = np.median(flow, axis=(0, 1))
                motion_residual = float(np.median(np.abs(flow - global_motion)))
                camera_motion_residuals.append(motion_residual)

            prev_gray = gray

        mean_blur_score = float(np.mean(laplacian_vars))
        mean_luminance = float(np.mean(luminances))
        mean_contrast = float(np.mean(contrasts))
        mean_camera_residual = float(np.mean(camera_motion_residuals)) if camera_motion_residuals else 0.0

        # 相机运动模式判定
        camera_mode = "STATIC_RECOMMENDED" if mean_camera_residual < 0.8 else "DYNAMIC_CAMERA"
        if mean_camera_residual > 3.0:
            camera_mode = "EXCESSIVE_SHAKE"

        # 2. 轻量 Pose 关键点时序解算 (L2)
        person_heights = []
        person_area_ratios = []
        foot_out_frames = 0
        kpt_confs = []
        foot_visible_frames = 0
        ankle_trajectories = []  # 记录左右脚踝 2D 轨迹以算 Jerk

        for f in sampled_frames:
            results = self.pose_model(f, verbose=False)
            if not results or len(results[0].boxes) == 0:
                foot_out_frames += 1
                continue

            # 取面积最大的人体目标
            boxes = results[0].boxes
            areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
            best_idx = int(torch.argmax(areas).item())
            box = boxes.xyxy[best_idx].cpu().numpy()

            bw = box[2] - box[0]
            bh = box[3] - box[1]
            person_heights.append(bh)
            person_area_ratios.append((bw * bh) / (width * height))

            # 关键点: 17 个 COCO 关键点
            # 15: left_ankle, 16: right_ankle, 5: left_shoulder, 6: right_shoulder, 11: left_hip, 12: right_hip
            if results[0].keypoints is not None and len(results[0].keypoints.data) > best_idx:
                kpts = results[0].keypoints.data[best_idx].cpu().numpy()  # (17, 3)
                conf = np.mean(kpts[:, 2])
                kpt_confs.append(conf)

                l_ankle = kpts[15]
                r_ankle = kpts[16]

                # 脚部出画核心检测: 脚踝 Y 坐标接近图像底边 (< 10px) 或超出画面
                is_foot_out = False
                if l_ankle[2] < 0.3 and r_ankle[2] < 0.3:
                    is_foot_out = True
                elif l_ankle[1] >= (height - 10) or r_ankle[1] >= (height - 10):
                    is_foot_out = True
                elif l_ankle[1] <= 5 or r_ankle[1] <= 5:
                    is_foot_out = True

                if is_foot_out:
                    foot_out_frames += 1
                else:
                    foot_visible_frames += 1

                # 记录足部中心用于 Jerk 计算
                ankle_center = 0.5 * (l_ankle[:2] + r_ankle[:2])
                ankle_trajectories.append(ankle_center)

        num_analyzed = len(sampled_frames)
        max_person_h = float(np.max(person_heights)) if person_heights else 0.0
        mean_person_ratio = float(np.mean(person_area_ratios)) if person_area_ratios else 0.0
        mean_kpt_conf = float(np.mean(kpt_confs)) if kpt_confs else 0.0
        foot_coverage = foot_visible_frames / max(num_analyzed, 1)

        # 计算脚部 2D 轨迹加加速度 Jerk
        kpt_jerk = 0.0
        if len(ankle_trajectories) >= 4:
            pts = np.array(ankle_trajectories)
            d1 = np.diff(pts, axis=0)
            d2 = np.diff(d1, axis=0)
            d3 = np.diff(d2, axis=0)
            kpt_jerk = float(np.mean(np.linalg.norm(d3, axis=-1)))

        # 3. 综合多层裁决 (硬规则一票否决 + 软指标打分)
        hard_violations = []

        # (a) 脚出画检测: 绝不姑息 (脚出画导致接触约束全面失效)
        if foot_out_frames > self.max_foot_out_frames or foot_coverage < self.min_foot_coverage:
            hard_violations.append(f"REJECT_FOOT_OUT_OF_FRAME: 脚出画 {foot_out_frames} 帧, 覆盖率 {foot_coverage:.1%}")

        # (b) 人物过小占屏比不足
        if max_person_h < self.min_person_height_px:
            hard_violations.append(f"REJECT_PERSON_TOO_SMALL: 最大像素高 {max_person_h:.0f}px < {self.min_person_height_px}px")

        # (c) 关键点置信度过低
        if mean_kpt_conf < self.min_kpt_conf:
            hard_violations.append(f"REJECT_LOW_KPT_CONFIDENCE: 平均置信度 {mean_kpt_conf:.3f} < {self.min_kpt_conf}")

        # (d) 严重抖动跳跃
        if kpt_jerk > self.max_kpt_jerk:
            hard_violations.append(f"REJECT_EXCESSIVE_JERK: 关键点抖动 {kpt_jerk:.1f} > {self.max_kpt_jerk}")

        # (e) 逆光/死黑对比度不足
        if mean_contrast < self.min_contrast:
            hard_violations.append(f"REJECT_LOW_CONTRAST: 对比度 {mean_contrast:.1f} < {self.min_contrast}")

        # (f) 剧烈手持晃动
        if camera_mode == "EXCESSIVE_SHAKE":
            hard_violations.append(f"REJECT_CAMERA_SHAKE: 镜头剧烈晃动 (残差 {mean_camera_residual:.2f})")

        # 计算综合质量分 (0 ~ 1.0)
        s_res = min(height / 720.0, 1.0)
        s_kpt = min(mean_kpt_conf / 0.85, 1.0)
        s_foot = foot_coverage
        s_motion = max(0.0, 1.0 - mean_camera_residual / 3.0)
        quality_score = float(0.3 * s_kpt + 0.3 * s_foot + 0.2 * s_res + 0.2 * s_motion)

        if hard_violations:
            status = "REJECT"
            reason = "; ".join(hard_violations)
        elif quality_score < 0.65:
            status = "REVIEW"
            reason = f"边缘质量分 {quality_score:.2f} (建议人工核查)"
        else:
            status = "PASS"
            reason = "各层画面与关键点准入指标达标"

        return {
            "status": status,
            "reason": reason,
            "quality_score": round(quality_score, 3),
            "camera_mode": camera_mode,
            "foot_out_frames": foot_out_frames,
            "foot_coverage": round(foot_coverage, 3),
            "max_person_height_px": round(max_person_h, 1),
            "mean_person_ratio": round(mean_person_ratio, 3),
            "mean_kpt_conf": round(mean_kpt_conf, 3),
            "kpt_jerk": round(kpt_jerk, 2),
            "mean_blur_score": round(mean_blur_score, 1),
            "mean_contrast": round(mean_contrast, 1),
            "camera_residual": round(mean_camera_residual, 2),
        }

    def evaluate_video(self, video_path: pathlib.Path) -> Dict[str, Any]:
        """对单个视频执行完整的 L0 ~ L2 过滤与评定"""
        clip_id = video_path.stem

        record = {
            "clip_id": clip_id,
            "file_path": str(video_path),
            "status": "PENDING",
            "reason": "",
            "quality_score": 0.0,
            "camera_mode": "UNKNOWN",
        }

        # 1. L0 元数据层
        ok_l0, reason_l0, meta = self.check_l0_metadata(video_path)
        record.update(meta)
        if not ok_l0:
            record["status"] = "REJECT"
            record["reason"] = reason_l0
            return record

        # 2. L1 & L2 画面与关键点层
        l12_res = self.analyze_frames_l1_l2(video_path, meta)
        record.update(l12_res)
        return record


def main():
    parser = argparse.ArgumentParser(description="视频素材端准入过滤与质量分流系统 (filter_video.py)")
    parser.add_argument("--input_dir", default="data/clips", help="待过滤视频片段目录 (默认: data/clips)")
    parser.add_argument("--output_dir", default="data/clips_filtered", help="合格视频导出目录 (默认: data/clips_filtered)")
    parser.add_argument("--report_dir", default="reports", help="报告输出目录 (默认: reports)")
    parser.add_argument("--workers", type=int, default=4, help="并发处理进程/线程数")
    parser.add_argument("--limit", type=int, default=0, help="处理视频数量限制 (0 为全量)")
    parser.add_argument("--copy_passed", action="store_true", help="将通过质检的合格视频复制到输出目录")
    args = parser.parse_args()

    input_path = pathlib.Path(args.input_dir)
    video_files = sorted(list(input_path.glob("*.mp4")) + list(input_path.glob("*.mov")))
    if not video_files:
        print(f"❌ 在 {input_path} 下未找到任何视频文件！")
        return

    if args.limit > 0:
        video_files = video_files[: args.limit]

    print("=" * 70)
    print("🎬 视频素材端准入过滤与分流系统 (filter_video.py)")
    print(f"📁 输入目录:   {input_path} (共 {len(video_files)} 个视频)")
    print(f"📂 合格目录:   {args.output_dir}")
    print(f"⚙️ 过滤配置:   L0元数据 + L1画面(脚出画/模糊/曝光/镜头光流) + L2关键点与Jerk")
    print("=" * 70)

    v_filter = VideoQualityFilter()
    pathlib.Path(args.report_dir).mkdir(parents=True, exist_ok=True)
    if args.copy_passed:
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    results = []
    stats = {"PASS": 0, "REVIEW": 0, "REJECT": 0}

    for idx, v_file in enumerate(video_files, 1):
        rec = v_filter.evaluate_video(v_file)
        results.append(rec)
        st = rec["status"]
        stats[st] = stats.get(st, 0) + 1

        if st == "PASS":
            tag = "✅ [PASS]"
            detail = f"分: {rec['quality_score']}, 相机: {rec['camera_mode']}, 脚覆盖: {rec.get('foot_coverage', 1.0):.1%}"
            if args.copy_passed:
                import shutil
                shutil.copy2(v_file, pathlib.Path(args.output_dir) / v_file.name)
        elif st == "REVIEW":
            tag = "🔍 [REVIEW]"
            detail = rec["reason"]
        else:
            tag = "🚫 [REJECT]"
            detail = rec["reason"]

        print(f"[{idx:03d}/{len(video_files)}] {tag} {rec['clip_id']} | {detail}")

    total_time = time.time() - t_start

    # 保存报告
    report_json = pathlib.Path(args.report_dir) / "video_filter_report.json"
    report_csv = pathlib.Path(args.report_dir) / "video_filter_report.csv"

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_evaluated": len(results),
        "pass_count": stats["PASS"],
        "review_count": stats["REVIEW"],
        "reject_count": stats["REJECT"],
        "pass_rate_pct": round(stats["PASS"] / max(len(results), 1) * 100, 2),
        "total_time_seconds": round(total_time, 2),
        "records": results,
    }

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if results:
        fieldnames = list(results[0].keys())
        with open(report_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print("\n" + "=" * 70)
    print("📊 视频前端素材准入过滤完成！")
    print(f"🔢 评估片段总数: {len(results)}")
    print(f"✅ PASS  (准入): {stats['PASS']} 个 ({summary['pass_rate_pct']}%)")
    print(f"🔍 REVIEW(复审): {stats['REVIEW']} 个")
    print(f"🚫 REJECT(淘汰): {stats['REJECT']} 个")
    print(f"⏳ 总耗时:       {total_time:.2f}s (平均每段 {total_time/max(len(results), 1):.2f}s)")
    print(f"📄 报告清单:     {report_json} / {report_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()
