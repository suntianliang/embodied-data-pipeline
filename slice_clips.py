#!/usr/bin/env python3
"""
slice_clips.py
--------------
调用 ffmpeg 对本地下载的原始视频（data/raw/*.mp4）进行预处理切片：
1. 保存原始数据：原始视频文件完全不修改，处理产物输出至 data/clips/
2. 去除片头片尾静止帧：通过帧差感知分析（Motion Energy / Inter-frame Diff）精准截除静态片头与片尾
3. 统一重采样为 30fps：调用 ffmpeg 进行帧率归一化（Mac 支持 VideoToolbox 硬件硬解硬编加速）
4. 严格切片为 3-10 秒有效运动片段：支持智能等分或滑动窗口切片
5. 多进程并发处理与详尽元数据清册导出（JSON & CSV）
"""

import argparse
import concurrent.futures
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

# 读取基础配置
CONFIG_PATH = Path("config.yaml")
if CONFIG_PATH.exists():
    try:
        CFG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        CFG = {}
else:
    CFG = {}

DEFAULT_RAW_DIR = Path(CFG.get("paths", {}).get("raw_root", "data/raw"))
DEFAULT_CLIPS_DIR = Path(CFG.get("paths", {}).get("clips", "data/clips"))
REPORTS_DIR = Path("reports")


def detect_motion_boundaries(
    video_path: Path,
    motion_threshold: float = 1.0,
    max_check_sec: float = 3.5,
    min_motion_run: int = 3,
) -> Tuple[float, float, float, int, int]:
    """
    快速检测片头和片尾的静止帧区间，返回 (start_sec, end_sec, total_duration, width, height)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_dur = n_frames / max(fps, 1.0)

    check_frames = int(fps * max_check_sec)

    # 1. 扫描片头静止帧 (从第 0 帧向后扫描)
    start_sec = 0.0
    prev_gray = None
    motion_run = 0

    scan_head_limit = min(check_frames, n_frames)
    for i in range(scan_head_limit):
        ret, frame = cap.read()
        if not ret:
            break
        # 缩小至 160x90 加速差分计算
        small = cv2.resize(frame, (160, 90))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))))
            if diff >= motion_threshold:
                motion_run += 1
                if motion_run >= min_motion_run:
                    # 确定动作真正开始时间点
                    start_sec = max(0.0, (i - min_motion_run + 1) / fps)
                    break
            else:
                motion_run = 0
        prev_gray = gray

    # 2. 扫描片尾静止帧 (从倒数 check_frames 顺序读取并反向检测)
    end_sec = total_dur
    tail_start_frame = max(0, n_frames - check_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, tail_start_frame)

    tail_diffs = []
    prev_gray = None
    curr_frame_idx = tail_start_frame
    while curr_frame_idx < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (160, 90))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))))
            tail_diffs.append((curr_frame_idx / fps, diff))
        prev_gray = gray
        curr_frame_idx += 1

    cap.release()

    # 从尾部倒序寻找动作停止点
    tail_motion_run = 0
    for t_sec, diff in reversed(tail_diffs):
        if diff >= motion_threshold:
            tail_motion_run += 1
            if tail_motion_run >= min_motion_run:
                end_sec = min(total_dur, t_sec + (min_motion_run / fps))
                break
        else:
            tail_motion_run = 0

    # 边界保护
    if start_sec >= end_sec:
        start_sec = 0.0
        end_sec = total_dur

    return start_sec, end_sec, total_dur, width, height


def calculate_segments(
    start_sec: float,
    end_sec: float,
    min_clip_s: float = 3.0,
    max_clip_s: float = 10.0,
    target_clip_s: float = 6.0,
    mode: str = "partition",
    stride_s: float = 5.0,
) -> List[Tuple[float, float]]:
    """
    计算有效动作区间内的切片起止时间，确保每片均在 [min_clip_s, max_clip_s] 之间
    """
    active_dur = end_sec - start_sec
    if active_dur < min_clip_s:
        return []

    # 动作区间在 3~10 秒内，直接保留完整段
    if active_dur <= max_clip_s:
        return [(start_sec, end_sec)]

    segments = []
    if mode == "partition":
        # 智能均匀划分：将长视频无缝等分成 K 个长度处于 [min_clip_s, max_clip_s] 的片断
        num_clips = max(1, int(np.ceil(active_dur / target_clip_s)))
        clip_dur = active_dur / num_clips

        # 若等分后单片超出范围，调整片段数
        while clip_dur > max_clip_s:
            num_clips += 1
            clip_dur = active_dur / num_clips
        while clip_dur < min_clip_s and num_clips > 1:
            num_clips -= 1
            clip_dur = active_dur / num_clips

        for i in range(num_clips):
            seg_start = start_sec + i * clip_dur
            seg_end = start_sec + (i + 1) * clip_dur if i < num_clips - 1 else end_sec
            segments.append((round(seg_start, 3), round(seg_end, 3)))

    elif mode == "sliding":
        # 滑动窗口模式（用于强化学习/时序生成）
        curr = start_sec
        while curr + min_clip_s <= end_sec:
            seg_end = min(curr + target_clip_s, end_sec)
            if seg_end - curr < min_clip_s:
                break
            segments.append((round(curr, 3), round(seg_end, 3)))
            curr += stride_s

    return segments


def run_ffmpeg_slice(
    src_path: Path,
    dst_path: Path,
    start_sec: float,
    end_sec: float,
    target_fps: int = 30,
) -> bool:
    """
    调用 ffmpeg 执行切片与 30fps 重采样，支持 Mac VideoToolbox 硬件加速
    """
    # 优先使用 Mac 硬件加速 h264_videotoolbox
    is_mac = sys.platform == "darwin"
    vcodec = "h264_videotoolbox" if is_mac else "libx264"
    codec_args = ["-b:v", "5M"] if is_mac else ["-crf", "19", "-preset", "fast"]

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-to", str(end_sec),
        "-i", str(src_path),
        "-vf", f"fps={target_fps}",
        "-c:v", vcodec,
        *codec_args,
        "-an",  # 机器人动作训练不需要音轨
        "-movflags", "+faststart",
        str(dst_path),
    ]

    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0 and is_mac:
        # 若 videotoolbox 不支持该分辨率，自动降级为 libx264
        fallback_cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-to", str(end_sec),
            "-i", str(src_path),
            "-vf", f"fps={target_fps}",
            "-c:v", "libx264", "-crf", "19", "-preset", "fast",
            "-an",
            "-movflags", "+faststart",
            str(dst_path),
        ]
        res_fb = subprocess.run(fallback_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        return res_fb.returncode == 0

    return res.returncode == 0


def process_single_video(
    video_path: Path,
    out_dir: Path,
    min_clip_s: float = 3.0,
    max_clip_s: float = 10.0,
    target_clip_s: float = 6.0,
    mode: str = "partition",
    motion_thresh: float = 1.0,
    target_fps: int = 30,
) -> List[Dict]:
    """单视频处理流程：静止帧检测 -> 分段 -> ffmpeg 30fps 输出 -> 返回片段元数据"""
    stem = video_path.stem
    try:
        start_sec, end_sec, total_dur, w, h = detect_motion_boundaries(
            video_path, motion_threshold=motion_thresh
        )
    except Exception as e:
        print(f"⚠️ 无法分析边界 {video_path.name}: {e}")
        return []

    segments = calculate_segments(
        start_sec,
        end_sec,
        min_clip_s=min_clip_s,
        max_clip_s=max_clip_s,
        target_clip_s=target_clip_s,
        mode=mode,
    )

    if not segments:
        print(f"  ⏭️ 跳过 {video_path.name}: 去除静止帧后有效时长不足 {min_clip_s}s ({end_sec - start_sec:.2f}s)")
        return []

    trimmed_head = round(start_sec, 2)
    trimmed_tail = round(total_dur - end_sec, 2)

    clip_records = []
    for idx, (seg_s, seg_e) in enumerate(segments, 1):
        clip_name = f"{stem}_clip{idx:02d}.mp4"
        clip_path = out_dir / clip_name
        dur = round(seg_e - seg_s, 2)

        # 检查是否已存在
        if clip_path.exists():
            clip_records.append({
                "clip_id": clip_path.stem,
                "clip_path": str(clip_path),
                "source_video": str(video_path),
                "source_id": stem,
                "start_sec": seg_s,
                "end_sec": seg_e,
                "duration": dur,
                "fps": float(target_fps),
                "width": w,
                "height": h,
                "trimmed_head_s": trimmed_head,
                "trimmed_tail_s": trimmed_tail,
                "status": "already_exists",
            })
            continue

        ok = run_ffmpeg_slice(
            video_path,
            clip_path,
            seg_s,
            seg_e,
            target_fps=target_fps,
        )

        if ok and clip_path.exists():
            clip_records.append({
                "clip_id": clip_path.stem,
                "clip_path": str(clip_path),
                "source_video": str(video_path),
                "source_id": stem,
                "start_sec": seg_s,
                "end_sec": seg_e,
                "duration": dur,
                "fps": float(target_fps),
                "width": w,
                "height": h,
                "trimmed_head_s": trimmed_head,
                "trimmed_tail_s": trimmed_tail,
                "status": "success",
            })
        else:
            print(f"  ❌ ffmpeg 切片失败: {clip_name}")

    return clip_records


def main():
    parser = argparse.ArgumentParser(
        description="ffmpeg 视频预处理切片工具：统一 30fps 重采样、去除静止片头片尾、切片成 3-10 秒训练片段"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(DEFAULT_RAW_DIR),
        help=f"原始视频目录 (默认: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_CLIPS_DIR),
        help=f"切片输出目录 (默认: {DEFAULT_CLIPS_DIR})",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="目标统一采样帧率 (默认: 30fps)",
    )
    parser.add_argument(
        "--min-clip-s",
        type=float,
        default=3.0,
        help="有效片段最短时长（秒，默认: 3.0）",
    )
    parser.add_argument(
        "--max-clip-s",
        type=float,
        default=10.0,
        help="有效片段最长时长（秒，默认: 10.0）",
    )
    parser.add_argument(
        "--target-clip-s",
        type=float,
        default=6.0,
        help="目标单切片理想时长（秒，默认: 6.0）",
    )
    parser.add_argument(
        "--mode",
        choices=["partition", "sliding"],
        default="partition",
        help="切片模式：partition (智能无缝等分，默认) / sliding (滑动重叠窗口)",
    )
    parser.add_argument(
        "--motion-thresh",
        type=float,
        default=1.0,
        help="静止帧差分阈值 (默认: 1.0，值越小判定越敏感)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 4),
        help="并行处理进程数 (默认: CPU 核心数)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制处理原视频数量 (用于快速测试，如 --limit 5)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_videos = sorted(list(input_dir.glob("*.mp4")))
    if not raw_videos:
        print(f"❌ 未在 {input_dir} 下找到任何 mp4 视频文件！")
        return

    if args.limit:
        raw_videos = raw_videos[: args.limit]

    print("=" * 65)
    print("🎬 视频预处理切片任务启动")
    print(f"📁 原始视频输入目录: {input_dir} (待处理: {len(raw_videos)} 个)")
    print(f"📂 切片产物输出目录: {out_dir} (原数据完整保留)")
    print(f"🎯 统一目标帧率:     {args.fps} fps")
    print(f"⏱️ 片段时长约束:     [{args.min_clip_s}s, {args.max_clip_s}s] (理想值: {args.target_clip_s}s)")
    print(f"✂️ 切片策略:         {args.mode} 模式")
    print(f"⚡ 并发进程数:       {args.workers}")
    print("=" * 65)

    all_manifest = []
    t_start = time.time()

    # 多进程并发处理
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(
                process_single_video,
                p,
                out_dir,
                args.min_clip_s,
                args.max_clip_s,
                args.target_clip_s,
                args.mode,
                args.motion_thresh,
                args.fps,
            ): p
            for p in raw_videos
        }

        total_done = 0
        for future in concurrent.futures.as_completed(future_map):
            v_path = future_map[future]
            total_done += 1
            try:
                records = future.result()
                all_manifest.extend(records)
                print(
                    f"[{total_done:02d}/{len(raw_videos)}] ✅ {v_path.name} -> 生成 {len(records)} 个 30fps 切片"
                )
            except Exception as e:
                print(f"[{total_done:02d}/{len(raw_videos)}] ❌ 处理失败 {v_path.name}: {e}")

    # 导出清册 JSON
    manifest_json = REPORTS_DIR / "clips_manifest.json"
    manifest_json.write_text(
        json.dumps(
            {
                "total_source_videos": len(raw_videos),
                "total_clips": len(all_manifest),
                "target_fps": args.fps,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "clips": all_manifest,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # 导出清册 CSV
    manifest_csv = REPORTS_DIR / "clips_manifest.csv"
    if all_manifest:
        fieldnames = list(all_manifest[0].keys())
        with manifest_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_manifest)

    # 统计信息
    total_time = time.time() - t_start
    total_duration = sum(r["duration"] for r in all_manifest)
    avg_duration = total_duration / max(len(all_manifest), 1)

    print("\n" + "=" * 65)
    print("🎉 预处理切片全部完成！")
    print(f"📊 原始视频总数: {len(raw_videos)}")
    print(f"📦 生成切片总数: {len(all_manifest)} 个")
    print(f"⏱️ 切片总有效时长: {total_duration:.1f} 秒 (平均每片 {avg_duration:.2f} 秒)")
    print(f"⏳ 总耗时: {total_time:.2f} 秒 (平均每个视频 {total_time/len(raw_videos):.2f} 秒)")
    print(f"📄 元数据清册:")
    print(f"   - JSON 报告: {manifest_json}")
    print(f"   - CSV  清单: {manifest_csv}")
    print("=" * 65)


if __name__ == "__main__":
    main()
