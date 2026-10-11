#!/usr/bin/env python3
"""
filter_pose.py
--------------
具身机器人动作学习专项——姿态估计质量检查与批处理过滤脚本：
针对 GVHMR / HMR4D 输出的 hmr4d_results.pt 进行全方位机器人仿真可用性质检：

质检关卡体系：
1. 置信度质检 (Confidence Gate): 关键点平均置信度低于 0.6 直接丢弃。
2. 运动量质检 (Motion Rate Gate): 肢体平均位移变化率接近 0（呆站着待机）或方差过小过滤掉。
3. 悬空不着地质检 (Floating Feet Gate): 动作整体悬空、脚掌未触碰地面（离地过高）的片段过滤掉。
4. 肢体抖动质检与时序低通优化 (Jitter Gate & Butterworth Smoothing):
   - 剧烈不可逆抽搐抖动直接剔除；
   - 对合格动作中存在的轻中度高频抖动，自动进行 6Hz 双向零相位 Butterworth 低通滤波平滑优化。
5. 动作突变/首尾跳变质检 (Discontinuity Gate): 单帧角位移或平移发生物理不可行的瞬间跳跃（切镜头/断层）过滤掉。
6. 前后主体切换质检 (Subject Switching Gate): 体型参数 betas 突变或空间位置瞬移（多人群体追踪 ID 切换）过滤掉。

产出：
- 自动导出质检合格（且平滑优化）的 .pt 与 .npz 文件至目标目录；
- 生成结构化报告 reports/pose_filter_report.json 与 CSV 清单。
"""

import argparse
import concurrent.futures
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from scipy.signal import butter, filtfilt

from robot_feasibility import RobotFeasibilityChecker

# 读取配置
CONFIG_PATH = Path("config.yaml")
if CONFIG_PATH.exists():
    try:
        CFG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        CFG = {}
else:
    CFG = {}

DEFAULT_INPUT_DIR = Path("/Users/dawn/Downloads/GVHMR_results")
DEFAULT_OUT_DIR = Path("data/poses_filtered")
REPORTS_DIR = Path("reports")
SMPLX_BODY_MODEL_PATH = Path("GMR/assets/body_models")


def butter_lowpass_filter(data: np.ndarray, cutoff_hz: float = 6.0, fps: float = 30.0, order: int = 2) -> np.ndarray:
    """
    双向零相位 Butterworth 低通滤波器（无相位延迟），平滑肢体高频抖动
    """
    if data.shape[0] <= 12:
        return data  # 序列过短直接返回
    nyq = 0.5 * fps
    normal_cutoff = min(0.99, max(0.01, cutoff_hz / nyq))
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    try:
        filtered = filtfilt(b, a, data, axis=0)
        return filtered.astype(np.float32)
    except Exception:
        return data


def extract_confidence(data: Union[Dict, np.lib.npyio.NpzFile], default_conf: float = 1.0) -> Tuple[float, str]:
    """自适应提取关键点置信度"""
    conf_keys = [
        "confidence", "conf", "scores", "score",
        "keypoints_conf", "joint_confidence", "joint_conf",
        "pred_conf", "pred_confidence", "bbox_scores", "kpt_conf"
    ]

    for k in conf_keys:
        if k in data:
            val = data[k]
            if isinstance(val, torch.Tensor):
                val = val.detach().cpu().numpy()
            arr = np.asarray(val, dtype=np.float32)
            if arr.size > 0:
                return float(np.mean(arr)), f"field:{k}"

    for k in ["keypoints", "keypoints_2d", "pred_keypoints_2d", "joints_2d", "joints"]:
        if k in data:
            val = data[k]
            if isinstance(val, torch.Tensor):
                val = val.detach().cpu().numpy()
            arr = np.asarray(val, dtype=np.float32)
            if arr.ndim >= 3 and arr.shape[-1] in (3, 4):
                conf_channel = arr[..., -1]
                if np.max(conf_channel) <= 1.0 and np.min(conf_channel) >= 0.0:
                    return float(np.mean(conf_channel)), f"channel:{k}"

    return default_conf, "default_fallback"


def extract_pose_tensors(data: Union[Dict, np.lib.npyio.NpzFile]) -> Dict[str, np.ndarray]:
    """提取 smpl 姿态与轨迹张量"""
    tensors = {}
    if "smpl_params_global" in data and isinstance(data["smpl_params_global"], dict):
        source_dict = data["smpl_params_global"]
    elif "smpl_params_incam" in data and isinstance(data["smpl_params_incam"], dict):
        source_dict = data["smpl_params_incam"]
    else:
        source_dict = data

    for k in ["body_pose", "pose_body", "poses"]:
        if k in source_dict:
            v = source_dict[k]
            if isinstance(v, torch.Tensor): v = v.detach().cpu().numpy()
            v = np.asarray(v, dtype=np.float32)
            if v.ndim == 3: v = v.reshape(v.shape[0], -1)
            tensors["body_pose"] = v
            break

    for k in ["transl", "trans", "translation"]:
        if k in source_dict:
            v = source_dict[k]
            if isinstance(v, torch.Tensor): v = v.detach().cpu().numpy()
            tensors["transl"] = np.asarray(v, dtype=np.float32)
            break

    for k in ["global_orient", "root_orient", "orient"]:
        if k in source_dict:
            v = source_dict[k]
            if isinstance(v, torch.Tensor): v = v.detach().cpu().numpy()
            v = np.asarray(v, dtype=np.float32)
            if v.ndim == 3: v = v.reshape(v.shape[0], -1)
            tensors["global_orient"] = v
            break

    for k in ["betas", "shape"]:
        if k in source_dict:
            v = source_dict[k]
            if isinstance(v, torch.Tensor): v = v.detach().cpu().numpy()
            tensors["betas"] = np.asarray(v, dtype=np.float32)
            break

    if "body_pose" in tensors:
        T = tensors["body_pose"].shape[0]
        if "transl" not in tensors:
            tensors["transl"] = np.zeros((T, 3), dtype=np.float32)
        if "global_orient" not in tensors:
            tensors["global_orient"] = np.zeros((T, 3), dtype=np.float32)
        if "betas" not in tensors:
            tensors["betas"] = np.zeros((T, 10), dtype=np.float32)

    return tensors


def evaluate_motion_quality(
    tensors: Dict[str, np.ndarray],
    fps: float = 30.0,
    body_model=None,
) -> Dict[str, float]:
    """
    全方位评估人体运动质检指标：
    1. 动态变化率与方差 (是否呆站着)
    2. 悬空/离地距离 (是否脚掌不着地)
    3. 肢体抖动度与加加速度 Jerk (是否剧烈抽搐)
    4. 帧间瞬时跳变与首尾跳变 (是否物理不可行)
    5. 体型与空间突变 (是否主体切换)
    """
    body_pose = tensors["body_pose"]          # (T, 63)
    transl = tensors["transl"]                # (T, 3)
    global_orient = tensors["global_orient"]  # (T, 3)
    betas = tensors["betas"]                  # (T, 10) 或 (10,)

    T = len(body_pose)
    dt = 1.0 / max(fps, 1.0)

    # 1. 动态变化率与时域方差
    pose_delta = np.linalg.norm(body_pose[1:] - body_pose[:-1], axis=-1)  # (T-1,)
    trans_delta = np.linalg.norm(transl[1:] - transl[:-1], axis=-1)       # (T-1,)
    mean_motion_rate = float(np.mean(pose_delta) + 2.0 * np.mean(trans_delta))
    pose_std = float(np.mean(np.std(body_pose, axis=0)))
    trans_std = float(np.linalg.norm(np.std(transl, axis=0)))

    # 2. 肢体抖动与 Jerk (加加速度)
    d1 = np.diff(body_pose, axis=0)
    d2 = np.diff(d1, axis=0)
    d3 = np.diff(d2, axis=0)
    mean_jerk = float(np.mean(np.linalg.norm(d3, axis=-1))) * 100.0 if len(d3) > 0 else 0.0
    mean_acc = float(np.mean(np.linalg.norm(d2, axis=-1))) * 30.0 if len(d2) > 0 else 0.0

    # 3. 帧间最大瞬态跳变与首尾变化
    max_pose_jump = float(np.max(pose_delta)) if len(pose_delta) > 0 else 0.0
    max_trans_jump = float(np.max(trans_delta)) if len(trans_delta) > 0 else 0.0
    boundary_pose_dist = float(np.linalg.norm(body_pose[-1] - body_pose[0]))
    boundary_trans_dist = float(np.linalg.norm(transl[-1] - transl[0]))

    # 4. 前后主体切换指标 (betas 时域波动与瞬移)
    if betas.ndim > 1 and len(betas) > 1:
        beta_std = float(np.mean(np.std(betas, axis=0)))
        max_beta_jump = float(np.max(np.linalg.norm(betas[1:] - betas[:-1], axis=-1)))
    else:
        beta_std = 0.0
        max_beta_jump = 0.0

    # 5. 悬空与双脚触地分析 (Ground Contact)
    min_foot_y = 0.0
    ground_contact_ratio = 1.0

    if body_model is not None:
        try:
            # 运行 SMPL-X 正向运动学获取左右脚踝/脚尖 3D 轨迹 (关节 10: 左脚, 11: 右脚)
            num_frames = T
            betas_raw = torch.tensor(betas if betas.ndim == 2 else betas[None, :].repeat(num_frames, 1)).float()
            if betas_raw.shape[1] != body_model.num_betas:
                if betas_raw.shape[1] < body_model.num_betas:
                    pad = torch.zeros(num_frames, body_model.num_betas - betas_raw.shape[1])
                    betas_raw = torch.cat([betas_raw, pad], dim=-1)
                else:
                    betas_raw = betas_raw[:, :body_model.num_betas]

            smpl_out = body_model(
                betas=betas_raw,
                global_orient=torch.tensor(global_orient).float(),
                body_pose=torch.tensor(body_pose).float(),
                transl=torch.tensor(transl).float(),
                left_hand_pose=torch.zeros(num_frames, 45).float(),
                right_hand_pose=torch.zeros(num_frames, 45).float(),
                jaw_pose=torch.zeros(num_frames, 3).float(),
                leye_pose=torch.zeros(num_frames, 3).float(),
                reye_pose=torch.zeros(num_frames, 3).float(),
                expression=torch.zeros(num_frames, getattr(body_model, "num_expression_coeffs", 10)).float(),
                return_full_pose=False,
            )
            joints = smpl_out.joints  # (T, 127, 3)
            # 提取脚踝/脚掌高度 (SMPL-X 关节 10, 11)
            l_foot_y = joints[:, 10, 1].detach().cpu().numpy()
            r_foot_y = joints[:, 11, 1].detach().cpu().numpy()
            min_feet_per_frame = np.minimum(l_foot_y, r_foot_y)
            min_foot_y = float(np.min(min_feet_per_frame))
            # 脚触地判定: 离地 <= 0.09m 视为着地
            ground_contact_ratio = float(np.mean(min_feet_per_frame <= 0.09))
        except Exception:
            min_hip_y = float(np.min(transl[:, 1]))
            min_foot_y = max(0.0, min_hip_y - 0.90)
            ground_contact_ratio = 1.0 if min_foot_y <= 0.09 else 0.0
    else:
        min_hip_y = float(np.min(transl[:, 1]))
        min_foot_y = max(0.0, min_hip_y - 0.90)
        ground_contact_ratio = 1.0 if min_foot_y <= 0.09 else 0.0

    return {
        "mean_motion_rate": mean_motion_rate,
        "pose_std": pose_std,
        "trans_std": trans_std,
        "mean_jerk": mean_jerk,
        "mean_acc": mean_acc,
        "max_pose_jump": max_pose_jump,
        "max_trans_jump": max_trans_jump,
        "boundary_pose_dist": boundary_pose_dist,
        "boundary_trans_dist": boundary_trans_dist,
        "beta_std": beta_std,
        "max_beta_jump": max_beta_jump,
        "min_foot_y": min_foot_y,
        "ground_contact_ratio": ground_contact_ratio,
        "num_frames": T,
        "duration_s": float(T * dt),
    }


def inspect_and_filter_file(
    file_path: Path,
    out_dir: Path,
    body_model,
    feasibility_checker: Optional[RobotFeasibilityChecker] = None,
    min_conf: float = 0.6,
    min_motion_rate: float = 0.08,
    min_pose_std: float = 0.04,
    max_ground_clearance: float = 0.12,
    max_jerk: float = 240.0,
    max_pose_jump: float = 1.25,
    max_trans_jump: float = 0.40,
    max_beta_std: float = 0.35,
    smooth: bool = True,
    smooth_cutoff: float = 6.0,
    out_format: str = "both",
    dry_run: bool = False,
    strict_conf: bool = False,
) -> Dict:
    """对单个位姿文件执行 6 维质检 + Unitree G1 机器人物理可复现性（Feasibility）综合质检与合格导出"""
    clip_id = file_path.parent.name if file_path.name == "hmr4d_results.pt" else file_path.stem

    record = {
        "clip_id": clip_id,
        "source_file": str(file_path),
        "status": "PENDING",
        "mean_conf": 0.0,
        "mean_motion_rate": 0.0,
        "pose_std": 0.0,
        "min_foot_y": 0.0,
        "ground_contact_ratio": 0.0,
        "mean_jerk": 0.0,
        "max_pose_jump": 0.0,
        "max_trans_jump": 0.0,
        "beta_std": 0.0,
        "num_frames": 0,
        "duration_s": 0.0,
        "rejection_reason": "",
        "smoothed": False,
        "feasibility_metrics": {},
        "out_pt": "",
        "out_npz": "",
    }

    try:
        if file_path.suffix == ".npz":
            data = dict(np.load(file_path, allow_pickle=True))
        else:
            data = torch.load(file_path, map_location="cpu")
    except Exception as e:
        record["status"] = "ERROR_LOAD"
        record["rejection_reason"] = f"文件损坏或加载失败: {str(e)[:60]}"
        return record

    # 1. 关卡 1: 置信度质检 (Confidence Gate)
    mean_conf, _ = extract_confidence(data, default_conf=1.0 if not strict_conf else 0.0)
    record["mean_conf"] = round(mean_conf, 4)
    if mean_conf < min_conf:
        record["status"] = "REJECT_LOW_CONFIDENCE"
        record["rejection_reason"] = f"平均置信度 {mean_conf:.3f} < {min_conf}"
        return record

    tensors = extract_pose_tensors(data)
    if "body_pose" not in tensors or len(tensors["body_pose"]) < 4:
        record["status"] = "REJECT_INVALID_DATA"
        record["rejection_reason"] = "缺少有效 body_pose 数据或帧数过短"
        return record

    # 计算质检全维指标
    m = evaluate_motion_quality(tensors, fps=30.0, body_model=body_model)
    record.update({
        "mean_motion_rate": round(m["mean_motion_rate"], 4),
        "pose_std": round(m["pose_std"], 4),
        "min_foot_y": round(m["min_foot_y"], 3),
        "ground_contact_ratio": round(m["ground_contact_ratio"], 3),
        "mean_jerk": round(m["mean_jerk"], 1),
        "max_pose_jump": round(m["max_pose_jump"], 3),
        "max_trans_jump": round(m["max_trans_jump"], 3),
        "beta_std": round(m["beta_std"], 4),
        "num_frames": m["num_frames"],
        "duration_s": round(m["duration_s"], 2),
    })

    # 2. 关卡 2: 运动量与方差质检 (呆站着/待机)
    if m["mean_motion_rate"] < min_motion_rate and m["pose_std"] < min_pose_std:
        record["status"] = "REJECT_STATIC_POSE"
        record["rejection_reason"] = f"运动量过低/呆站着 (变化率: {m['mean_motion_rate']:.3f}, 标准差: {m['pose_std']:.3f})"
        return record

    # 3. 关卡 3: 整体悬空/脚掌不着地质检 (Floating Feet Gate)
    if m["min_foot_y"] > max_ground_clearance or m["ground_contact_ratio"] < 0.03:
        record["status"] = "REJECT_FLOATING_FEET"
        record["rejection_reason"] = (
            f"整体悬空不着地: 双脚最低高度 {m['min_foot_y']:.2f}m > {max_ground_clearance}m, "
            f"触地帧率 {m['ground_contact_ratio']:.1%}"
        )
        return record

    # 4. 关卡 4: 前后主体切换质检 (Subject Switching / Identity Swap)
    if (m["beta_std"] > max_beta_std and m["max_beta_jump"] > 0.4) or (m["max_trans_jump"] > 0.65 and m["max_pose_jump"] > 1.5):
        record["status"] = "REJECT_SUBJECT_SWITCH"
        record["rejection_reason"] = (
            f"前后主体切换: beta方差 {m['beta_std']:.3f} 或单帧突变 "
            f"(平移跳跃 {m['max_trans_jump']:.2f}m, 姿态跳跃 {m['max_pose_jump']:.2f}rad)"
        )
        return record

    # 5. 关卡 5: 帧间突变与首尾剧烈跳变质检 (Discontinuity Gate)
    if m["max_pose_jump"] > max_pose_jump or m["max_trans_jump"] > max_trans_jump:
        record["status"] = "REJECT_UNPHYSICAL_JUMP"
        record["rejection_reason"] = (
            f"动作突变/不可复现: 单帧最大角位移 {m['max_pose_jump']:.2f}rad > {max_pose_jump} "
            f"或平移跳变 {m['max_trans_jump']:.2f}m > {max_trans_jump}"
        )
        return record

    # 6. 关卡 6: 肢体持续严重抽搐抖动质检 (Jitter Gate)
    if m["mean_jerk"] > max_jerk:
        record["status"] = "REJECT_SEVERE_JITTER"
        record["rejection_reason"] = f"肢体持续剧烈抽搐抖动: 平均 Jerk {m['mean_jerk']:.1f} > {max_jerk}"
        return record

    # 执行 6Hz Butterworth 低通时序平滑优化 (去除轻微高频毛刺)
    final_body_pose = tensors["body_pose"]
    final_transl = tensors["transl"]
    if smooth:
        final_body_pose = butter_lowpass_filter(tensors["body_pose"], cutoff_hz=smooth_cutoff, fps=30.0)
        final_transl = butter_lowpass_filter(tensors["transl"], cutoff_hz=smooth_cutoff, fps=30.0)
        record["smoothed"] = True

    # 7. 关卡 7: Unitree G1 机器人物理可复现性（Feasibility）门控
    retargeted_qpos = None
    if feasibility_checker is not None:
        # 获取候选 G1 关节轨迹 qpos (优先读取 data/retargeted/{clip_id}.pkl)
        retarget_pkl = Path("data/retargeted") / f"{clip_id}.pkl"
        if retarget_pkl.exists():
            try:
                import pickle
                with open(retarget_pkl, "rb") as fp:
                    m_data = pickle.load(fp)
                    retargeted_qpos = m_data.get("qpos")
            except Exception:
                retargeted_qpos = None

        if retargeted_qpos is None:
            # 现场进行快速 GMR Retargeting 求解
            try:
                from general_motion_retargeting import GeneralMotionRetargeting as GMR
                from GMR.general_motion_retargeting.utils.smpl import load_gvhmr_pred_file, get_gvhmr_data_offline_fast
                smplx_d, b_model, smplx_out, h_height = load_gvhmr_pred_file(file_path, SMPLX_BODY_MODEL_PATH)
                frames, aligned_fps = get_gvhmr_data_offline_fast(smplx_d, b_model, smplx_out, tgt_fps=30)
                retarget = GMR(actual_human_height=h_height, src_human="smplx", tgt_robot="unitree_g1")
                qpos_list = [retarget.retarget(f) for f in frames]
                retargeted_qpos = np.array(qpos_list, dtype=np.float32)
            except Exception as e:
                retargeted_qpos = None

        if retargeted_qpos is not None:
            # 应用时序平滑与舵机极限限速截断优化
            if smooth:
                retargeted_qpos = feasibility_checker.smooth_motion(retargeted_qpos, fps=30.0, cutoff_hz=smooth_cutoff)

            # 综合执行 4 项机器人物理约束质检
            feasibility_res = feasibility_checker.evaluate_feasibility(retargeted_qpos, fps=30.0)
            record["feasibility_metrics"] = feasibility_res.get("metrics", {})

            if not feasibility_res["passed"]:
                record["status"] = feasibility_res["status"]
                record["rejection_reason"] = feasibility_res["reason"]
                return record

    # ✅ 通过全部质检关卡
    record["status"] = "PASS"

    # 产出合格文件
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        pt_dst = out_dir / f"{clip_id}.pt"
        npz_dst = out_dir / f"{clip_id}.npz"

        # 导出 .pt
        if out_format in ("pt", "both"):
            # 保持原始字典框架并注入平滑优化后的位姿
            save_dict = dict(data)
            if "smpl_params_global" in save_dict and isinstance(save_dict["smpl_params_global"], dict):
                save_dict["smpl_params_global"]["body_pose"] = torch.tensor(final_body_pose)
                save_dict["smpl_params_global"]["transl"] = torch.tensor(final_transl)
            torch.save(save_dict, pt_dst)
            record["out_pt"] = str(pt_dst)

        # 导出标准 GMR / AMASS .npz
        if out_format in ("npz", "both"):
            betas_export = tensors["betas"][0] if tensors["betas"].ndim == 2 else tensors["betas"]
            npz_dict = {
                "pose_body": final_body_pose,
                "root_orient": tensors["global_orient"],
                "trans": final_transl,
                "betas": betas_export,
                "gender": "neutral",
                "mocap_frame_rate": 30,
                "clip_id": clip_id,
                "mean_conf": record["mean_conf"],
                "mean_motion_rate": record["mean_motion_rate"],
                "min_foot_y": record["min_foot_y"],
                "smoothed": record["smoothed"],
            }
            np.savez_compressed(npz_dst, **npz_dict)
            record["out_npz"] = str(npz_dst)

    return record


def find_pose_files(search_dir: Path) -> List[Path]:
    """递归检索所有 GVHMR 位姿文件"""
    if not search_dir.exists():
        return []
    hmr_files = list(search_dir.glob("**/hmr4d_results.pt"))
    if hmr_files:
        return sorted(hmr_files)
    pt_files = list(search_dir.glob("**/*.pt"))
    if pt_files:
        return sorted(pt_files)
    return sorted(list(search_dir.glob("**/*.npz")))


def main():
    parser = argparse.ArgumentParser(
        description="GVHMR 机器人动作姿态全维质检与平滑过滤批处理工具"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help=f"GVHMR 位姿结果目录 (默认: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help=f"质检合格动作导出目录 (默认: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.6,
        help="关键点置信度最低阈值 (默认: 0.6)",
    )
    parser.add_argument(
        "--min-motion-rate",
        type=float,
        default=0.08,
        help="肢体平均位移变化率最低阈值 (过滤呆站待机，默认: 0.08)",
    )
    parser.add_argument(
        "--min-pose-std",
        type=float,
        default=0.04,
        help="姿态时域标准差最低阈值 (默认: 0.04)",
    )
    parser.add_argument(
        "--max-ground-clearance",
        type=float,
        default=0.12,
        help="双脚最低离地高度上限（米），超过此值判定为悬空不着地 (默认: 0.12m)",
    )
    parser.add_argument(
        "--max-jerk",
        type=float,
        default=35.0,
        help="肢体加加速度 Jerk 阈值上限，超过此值判定为剧烈不可逆抽搐抖动 (默认: 35.0)",
    )
    parser.add_argument(
        "--max-pose-jump",
        type=float,
        default=1.25,
        help="单帧关节角位移最大跳变 (rad)，超过此值判定为动作突变/镜头切换 (默认: 1.25 rad = 72度)",
    )
    parser.add_argument(
        "--max-trans-jump",
        type=float,
        default=0.40,
        help="单帧根节点平移最大跳跃 (m)，超过此值判定为瞬移/主体切换 (默认: 0.40m)",
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="禁用 Butterworth 时序平滑优化 (默认启用平滑优化)",
    )
    parser.add_argument(
        "--smooth-cutoff",
        type=float,
        default=6.0,
        help="Butterworth 低通滤波器截止频率 (Hz，默认: 6.0Hz)",
    )
    parser.add_argument(
        "--format",
        choices=["both", "pt", "npz"],
        default="both",
        help="合格位姿输出格式 (默认: both)",
    )
    parser.add_argument(
        "--check-feasibility",
        action="store_true",
        default=True,
        help="开启 Unitree G1 机器人物理可复现性（Feasibility）综合质检 (默认: 开启)",
    )
    parser.add_argument(
        "--no-check-feasibility",
        dest="check_feasibility",
        action="store_false",
        help="关闭机器人物理可行性质检",
    )
    parser.add_argument(
        "--robot-xml",
        type=str,
        default="assets/unitree_g1/g1.xml",
        help="Unitree G1 机器人 MuJoCo XML 模型路径",
    )
    parser.add_argument(
        "--max-joint-vel",
        type=float,
        default=25.0,
        help="G1 舵机物理最大响应角速度 (rad/s，默认: 25.0)",
    )
    parser.add_argument(
        "--max-joint-acc",
        type=float,
        default=400.0,
        help="G1 舵机物理最大角加速度 (rad/s²，默认: 400.0)",
    )
    parser.add_argument(
        "--max-ground-penetration",
        type=float,
        default=0.04,
        help="脚底陷入地面最大允许深度 (米，默认: 0.04m)",
    )
    parser.add_argument(
        "--min-hand-dist",
        type=float,
        default=0.11,
        help="双手手腕中心最小安全间距，小于视为自碰撞互穿 (米，默认: 0.11m)",
    )
    parser.add_argument(
        "--max-arm-sat",
        type=float,
        default=0.20,
        help="手臂关节死贴在限位硬停止位最大允许帧率 (针对反手摸后背反关节，默认: 0.20)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 4),
        help="并发线程数 (默认: CPU 核心数)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="试运行模式：仅进行质检打分并生成统计报告，不输出姿态文件",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🤖 具身机器人动作数据全维质检过滤工具 (filter_pose.py)")
    print(f"📁 检索输入目录:         {input_dir}")
    print(f"📂 合格姿态输出目录:     {out_dir} ({args.format})")
    print(f"🎯 1. 关键点置信度阈值:  >= {args.min_conf}")
    print(f"🏃 2. 运动动态变化率:    >= {args.min_motion_rate} (过滤'呆站着')")
    print(f"🦶 3. 脚掌着地离地高度:  <= {args.max_ground_clearance}m (过滤'悬空不着地')")
    print(f"〰️ 4. 肢体持续抖动Jerk:  <= {args.max_jerk} (时序低通平滑: {'开启(6Hz)' if not args.no_smooth else '关闭'})")
    print(f"⚡ 5. 帧间瞬时动作跳跃:  <= {args.max_pose_jump}rad / {args.max_trans_jump}m (过滤'动作突变/切镜头')")
    print(f"👥 6. 前后主体切换监控:  开启 (监控体型 betas 与空间瞬移)")
    if args.check_feasibility:
        print(f"🦾 7. 机器人物理可复现性: 开启 (Unitree G1 约束)")
        print(f"   - 速度/加速度限值:   <= {args.max_joint_vel} rad/s / <= {args.max_joint_acc} rad/s²")
        print(f"   - 关节越界/死锁门控: 越界 <= 0.08 rad, 手臂限位贴合 <= {args.max_arm_sat:.0%}")
        print(f"   - 地面穿透严查阈值: 陷地深度 <= {args.max_ground_penetration}m")
        print(f"   - 自碰撞互穿安全间距: 双手间距 >= {args.min_hand_dist}m, 网格穿透容差 8mm")
    print("=" * 70)

    # 预加载 SMPL-X 骨架模型用于精确足底高度解算
    body_model = None
    if SMPLX_BODY_MODEL_PATH.exists():
        try:
            from GMR.general_motion_retargeting.utils.smpl import _create_smplx_model
            body_model = _create_smplx_model(SMPLX_BODY_MODEL_PATH, gender="neutral")
            print("✅ SMPL-X 骨骼运动学模型加载成功 (启用毫米级足底高度解算)")
        except Exception as e:
            print(f"⚠️ SMPL-X 模型加载跳过 (使用根节点估算足底高度): {e}")

    # 预初始化 Unitree G1 机器人物理可复现性质检器
    feasibility_checker = None
    if args.check_feasibility:
        try:
            feasibility_checker = RobotFeasibilityChecker(
                xml_path=args.robot_xml,
                max_joint_vel=args.max_joint_vel,
                max_joint_acc=args.max_joint_acc,
                max_ground_penetration=args.max_ground_penetration,
                max_ground_clearance=args.max_ground_clearance,
                min_hand_dist=args.min_hand_dist,
                max_saturation_ratio=args.max_arm_sat,
            )
            print("✅ Unitree G1 机器人物理可行性质检器加载就绪")
        except Exception as e:
            print(f"⚠️ 机器人可行性质检器初始化跳过: {e}")

    pose_files = find_pose_files(input_dir)
    if not pose_files:
        print(f"❌ 在 {input_dir} 下未找到任何位姿文件！")
        return

    print(f"🔍 检索到位姿片段共 {len(pose_files)} 个，开始并行质检与平滑...\n")

    t_start = time.time()
    results = []
    stats = {
        "PASS": 0,
        "REJECT_LOW_CONFIDENCE": 0,
        "REJECT_STATIC_POSE": 0,
        "REJECT_FLOATING_FEET": 0,
        "REJECT_SEVERE_JITTER": 0,
        "REJECT_UNPHYSICAL_JUMP": 0,
        "REJECT_SUBJECT_SWITCH": 0,
        # 机器人物理约束门控
        "REJECT_GROUND_PENETRATION": 0,
        "REJECT_JOINT_LIMIT": 0,
        "REJECT_SELF_COLLISION": 0,
        "REJECT_VELOCITY_LIMIT": 0,
        "REJECT_ACCEL_LIMIT": 0,
        "OTHER_REJECT": 0,
    }

    # 执行并行质检
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(
                inspect_and_filter_file,
                f,
                out_dir,
                body_model,
                feasibility_checker=feasibility_checker,
                min_conf=args.min_conf,
                min_motion_rate=args.min_motion_rate,
                min_pose_std=args.min_pose_std,
                max_ground_clearance=args.max_ground_clearance,
                max_jerk=args.max_jerk,
                max_pose_jump=args.max_pose_jump,
                max_trans_jump=args.max_trans_jump,
                smooth=not args.no_smooth,
                smooth_cutoff=args.smooth_cutoff,
                out_format=args.format,
                dry_run=args.dry_run,
            ): f
            for f in pose_files
        }

        total_done = 0
        for future in concurrent.futures.as_completed(future_map):
            total_done += 1
            f_path = future_map[future]
            try:
                rec = future.result()
                results.append(rec)
                st = rec["status"]
                cid = rec["clip_id"]

                if st == "PASS":
                    stats["PASS"] += 1
                    smooth_tag = " (已时序平滑)" if rec["smoothed"] else ""
                    print(f"[{total_done:03d}/{len(pose_files)}] ✅ [PASS] {cid}{smooth_tag} "
                          f"| 变化率: {rec['mean_motion_rate']:.3f}, 足底: {rec['min_foot_y']:.2f}m, Jerk: {rec['mean_jerk']:.1f}")
                elif st in stats:
                    stats[st] += 1
                    tag_map = {
                        "REJECT_LOW_CONFIDENCE": "❌ 低置信度",
                        "REJECT_STATIC_POSE": "⏸️ 呆站着/微动",
                        "REJECT_FLOATING_FEET": "🪁 悬空不着地",
                        "REJECT_SEVERE_JITTER": "⚡ 肢体剧烈抖动",
                        "REJECT_UNPHYSICAL_JUMP": "💥 首尾动作突变/跳帧",
                        "REJECT_SUBJECT_SWITCH": "👥 前后主体切换",
                        "REJECT_GROUND_PENETRATION": "🕳️ 地面穿透超限",
                        "REJECT_JOINT_LIMIT": "🔒 关节越界/卡死",
                        "REJECT_SELF_COLLISION": "💥 自碰撞/肢体穿模",
                        "REJECT_VELOCITY_LIMIT": "🏃 角速度超限",
                        "REJECT_ACCEL_LIMIT": "⚡ 角加速度超限",
                    }
                    print(f"[{total_done:03d}/{len(pose_files)}] 🚫 [{tag_map.get(st, st)}] {cid} -> {rec['rejection_reason']}")
                else:
                    stats["OTHER_REJECT"] += 1
                    print(f"[{total_done:03d}/{len(pose_files)}] ⚠️ [REJECT] {cid} -> {rec['rejection_reason']}")

            except Exception as e:
                stats["OTHER_REJECT"] += 1
                print(f"[{total_done:03d}/{len(pose_files)}] ❌ 处理异常 {f_path.name}: {e}")

    # 导出报告
    report_json = REPORTS_DIR / "pose_filter_report.json"
    report_csv = REPORTS_DIR / "pose_filter_report.csv"

    summary_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_evaluated": len(results),
        "passed_count": stats["PASS"],
        "pass_rate_pct": round(stats["PASS"] / max(len(results), 1) * 100, 2),
        "rejection_statistics": stats,
        "criteria": {
            "min_conf": args.min_conf,
            "min_motion_rate": args.min_motion_rate,
            "max_ground_clearance": args.max_ground_clearance,
            "max_jerk": args.max_jerk,
            "max_pose_jump": args.max_pose_jump,
            "max_trans_jump": args.max_trans_jump,
            "check_robot_feasibility": args.check_feasibility,
            "max_joint_vel": args.max_joint_vel,
            "max_joint_acc": args.max_joint_acc,
            "max_ground_penetration": args.max_ground_penetration,
            "min_hand_dist": args.min_hand_dist,
            "max_arm_saturation_ratio": args.max_arm_sat,
            "smoothed_with_butterworth_6hz": not args.no_smooth,
        },
        "records": results,
    }

    report_json.write_text(json.dumps(summary_data, indent=2, ensure_ascii=False), encoding="utf-8")

    if results:
        fieldnames = [k for k in results[0].keys() if k != "feasibility_metrics"]
        with report_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

    total_time = time.time() - t_start

    print("\n" + "=" * 70)
    print("📊 姿态全维质检与平滑批处理完成！")
    print(f"🔢 评估片段总数:         {len(results)} 个")
    print(f"✅ 质检合格并优化:       {stats['PASS']} 个 (合格率: {summary_data['pass_rate_pct']}%)")
    print("----------------------------------------------------------------------")
    print(f"🚫 剔除原因统计:")
    print(f"   【人体质量关卡】")
    print(f"   - 🪁 整体悬空/脚不着地:     {stats['REJECT_FLOATING_FEET']} 个")
    print(f"   - ⚡ 肢体剧烈不可逆抖动:     {stats['REJECT_SEVERE_JITTER']} 个")
    print(f"   - 💥 动作突变/首尾断层跳跃: {stats['REJECT_UNPHYSICAL_JUMP']} 个")
    print(f"   - 👥 前后主体切换/ID混淆:  {stats['REJECT_SUBJECT_SWITCH']} 个")
    print(f"   - ⏸️ 呆站着待机/运动量不足: {stats['REJECT_STATIC_POSE']} 个")
    print(f"   - ❌ 关键点置信度过低:     {stats['REJECT_LOW_CONFIDENCE']} 个")
    print(f"   【机器人物理约束关卡】")
    print(f"   - 🕳️ 地面穿透超限 (>4cm):   {stats['REJECT_GROUND_PENETRATION']} 个")
    print(f"   - 🔒 关节越界/反手摸后背卡死: {stats['REJECT_JOINT_LIMIT']} 个")
    print(f"   - 💥 自碰撞/双手穿模互撞:   {stats['REJECT_SELF_COLLISION']} 个")
    print(f"   - 🏃 舵机角速度超限 (>25rad/s): {stats['REJECT_VELOCITY_LIMIT']} 个")
    print(f"   - ⚡ 舵机角加速度超限 (>400rad/s²): {stats['REJECT_ACCEL_LIMIT']} 个")
    print("----------------------------------------------------------------------")
    print(f"📁 合格动作产出路径:     {out_dir.resolve()} ({args.format} 格式)")
    print(f"⏳ 总质检耗时:           {total_time:.2f} 秒 (平均每片 {total_time/max(len(results), 1):.3f}s)")
    print(f"📄 质检报告清册:")
    print(f"   - JSON 报告: {report_json}")
    print(f"   - CSV  清单: {report_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()
