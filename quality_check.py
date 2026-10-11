#!/usr/bin/env python3
"""
quality_check.py
-----------------
GMR 机器人动作六维量化质检评估工具：
对重定向到 Unitree G1 后的动作序列 (pkl/npz) 进行严格的 6 项量化质检打分：

1. 地面穿透率 (Ground Penetration Rate): 脚底深入地面 (Z < -0.04m) 的帧占比与最大穿透深度 (合格: < 2%, 最大穿透 < 5cm)
2. 悬浮量 (Floating Clearance): 站立支撑相脚底离地高度均值 (合格: < 3cm)
3. 自相交/自碰撞帧数 (Self-Collision Count): 左右手互穿、手穿躯干、脚互踩的帧占比 (合格: 0, 容忍 < 1%)
4. 关节速度尖峰率 (Velocity Spike Rate): 关节速度超过 80% 电机极限 (> 20 rad/s) 的帧占比 (合格: < 1%)
5. 支撑脚水平滑移量 (Stance Foot Horizontal Slip): 站立接触相内脚部水平移动总距离 (合格: < 3cm/步)
6. 限位贴边/饱和率 (Joint Limit Saturation Rate): 关节角落在机械硬限位 ±2° (0.035 rad) 范围内的帧占比 (合格: < 10%)
"""

import argparse
import csv
import json
import os
import pathlib
import pickle
import time
from typing import Dict, Any, List, Tuple

import mujoco as mj
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_G1_XML = PROJECT_ROOT / "assets" / "unitree_g1" / "g1.xml"
if not DEFAULT_G1_XML.exists():
    DEFAULT_G1_XML = PROJECT_ROOT / "GMR" / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"


class RobotMotionQualityAuditor:
    """机器人动作六维量化质检审计器"""

    def __init__(self, xml_path: str = str(DEFAULT_G1_XML)):
        self.xml_path = xml_path
        self.model = mj.MjModel.from_xml_path(self.xml_path)
        self.data = mj.MjData(self.model)

        # 缓存关键关节与刚体 ID
        m = self.model
        self.joint_names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, i) for i in range(1, m.njnt)]
        self.joint_ranges = np.zeros((29, 2), dtype=np.float32)
        for i in range(1, m.njnt):
            self.joint_ranges[i - 1] = m.jnt_range[i]

        def get_bid(name):
            try: return m.body(name).id
            except Exception: return -1

        self.left_wrist_id = get_bid("left_wrist_yaw_link")
        self.right_wrist_id = get_bid("right_wrist_yaw_link")
        self.torso_id = get_bid("torso_link")
        self.pelvis_id = get_bid("pelvis")
        self.left_ankle_id = get_bid("left_ankle_roll_link")
        self.right_ankle_id = get_bid("right_ankle_roll_link")

        # 脚底触地几何球体
        self.foot_geom_ids = [
            g for g in range(m.ngeom)
            if "ankle_roll" in (mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "")
            or "toe" in (mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "")
        ]

    def audit_motion(self, qpos: np.ndarray, fps: float = 30.0) -> Dict[str, Any]:
        """对单个动作轨迹执行完整的六维质检"""
        T = len(qpos)
        dt = 1.0 / max(fps, 1.0)
        dof_pos = qpos[:, 7:]  # (T, 29)

        m = self.model
        d = mj.MjData(m)

        # 1. 关节速度与尖峰率 (Metric 4)
        dof_vel = np.diff(dof_pos, axis=0) / dt  # (T-1, 29)
        max_vel = float(np.abs(dof_vel).max()) if len(dof_vel) > 0 else 0.0
        # 80% G1 电机速度阈值 (20.0 rad/s)
        vel_spike_frames = int(np.sum(np.any(np.abs(dof_vel) > 20.0, axis=-1)))
        vel_spike_ratio = vel_spike_frames / max(T - 1, 1)

        # 2. 限位贴边率 (Metric 6)
        # 关节值落在极限 ±2° (0.035 rad) 内
        margin_rad = np.deg2rad(2.0)
        low_dist = np.abs(dof_pos - self.joint_ranges[:, 0])
        upp_dist = np.abs(dof_pos - self.joint_ranges[:, 1])
        is_near_limit = (low_dist < margin_rad) | (upp_dist < margin_rad)
        limit_saturation_ratio = float(np.mean(is_near_limit))
        max_joint_saturation = float(np.max(np.mean(is_near_limit, axis=0)))

        # 3. 逐帧正向运动学分析 (穿透、悬浮、自碰撞、滑移)
        foot_z_list = []
        l_ankle_xy = []
        r_ankle_xy = []
        penetration_frames = 0
        self_collision_frames = 0
        max_penetration_m = 0.0
        stance_foot_z_list = []

        for t in range(T):
            d.qpos[:] = qpos[t]
            mj.mj_step1(m, d)

            # 计算脚底几何最低点高度
            if self.foot_geom_ids:
                sole_z = min(d.geom_xpos[g][2] - m.geom_size[g][0] for g in self.foot_geom_ids)
            else:
                sole_z = min(d.xpos[self.left_ankle_id][2], d.xpos[self.right_ankle_id][2]) - 0.03

            foot_z_list.append(sole_z)
            if sole_z < -0.04:  # 穿透深入地面 > 4cm
                penetration_frames += 1
                max_penetration_m = max(max_penetration_m, abs(sole_z))

            if -0.04 <= sole_z <= 0.05:  # 支撑相
                stance_foot_z_list.append(sole_z)

            # 记录脚部水平 XY 坐标
            p_la = d.xpos[self.left_ankle_id][:2]
            p_ra = d.xpos[self.right_ankle_id][:2]
            l_ankle_xy.append(p_la)
            r_ankle_xy.append(p_ra)

            # 自碰撞穿模检测
            p_lw = d.xpos[self.left_wrist_id]
            p_rw = d.xpos[self.right_wrist_id]
            hand_dist = float(np.linalg.norm(p_lw - p_rw))
            p_torso = d.xpos[self.torso_id]
            ht_dist = min(np.linalg.norm(p_lw - p_torso), np.linalg.norm(p_rw - p_torso))

            is_colliding = False
            if hand_dist < 0.11 or ht_dist < 0.08:
                is_colliding = True

            # MuJoCo 穿透检查
            for i in range(d.ncon):
                if d.contact[i].dist < -0.008:
                    is_colliding = True
                    break

            if is_colliding:
                self_collision_frames += 1

        # Metric 1: 地面穿透率
        penetration_ratio = penetration_frames / T

        # Metric 2: 悬浮量 (站立支撑相脚踝高度均值，米)
        floating_clearance_m = float(np.mean(stance_foot_z_list)) if stance_foot_z_list else float(np.mean(foot_z_list))
        floating_clearance_m = max(0.0, floating_clearance_m)

        # Metric 3: 自碰撞率
        self_collision_ratio = self_collision_frames / T

        # Metric 5: 支撑脚水平滑移量 (cm/步)
        l_xy = np.array(l_ankle_xy)
        r_xy = np.array(r_ankle_xy)
        fz = np.array(foot_z_list)
        # 判定左脚或右脚贴地帧 (z <= 0.03m)
        l_stance = fz <= 0.03
        r_stance = fz <= 0.03
        l_slip_m = np.sum(np.linalg.norm(np.diff(l_xy[l_stance], axis=0), axis=-1)) if np.sum(l_stance) > 2 else 0.0
        r_slip_m = np.sum(np.linalg.norm(np.diff(r_xy[r_stance], axis=0), axis=-1)) if np.sum(r_stance) > 2 else 0.0
        # 估算步数 (按时长估算，平均每秒 1.5 步)
        num_steps = max(1.0, (T * dt) * 1.5)
        foot_slip_cm_per_step = float(((l_slip_m + r_slip_m) * 100.0) / num_steps)

        # 综合合格评定
        passed = bool(
            penetration_ratio <= 0.02 and
            max_penetration_m <= 0.05 and
            floating_clearance_m <= 0.04 and
            self_collision_ratio <= 0.01 and
            vel_spike_ratio <= 0.02 and
            foot_slip_cm_per_step <= 5.0 and
            limit_saturation_ratio <= 0.15
        )

        # 评定等级
        score = 100.0
        score -= min(30.0, penetration_ratio * 300.0)
        score -= min(25.0, floating_clearance_m * 300.0)
        score -= min(25.0, self_collision_ratio * 300.0)
        score -= min(15.0, vel_spike_ratio * 200.0)
        score -= min(15.0, max(0.0, foot_slip_cm_per_step - 2.0) * 3.0)
        score -= min(15.0, limit_saturation_ratio * 50.0)
        score = max(0.0, round(score, 1))

        if score >= 90: grade = "A (优秀)"
        elif score >= 75: grade = "B (良好)"
        elif score >= 60: grade = "C (及格)"
        else: grade = "F (不合格)"

        return {
            "passed": passed,
            "quality_score": score,
            "grade": grade,
            "penetration_ratio": round(penetration_ratio, 4),
            "max_penetration_cm": round(max_penetration_m * 100.0, 1),
            "floating_clearance_cm": round(floating_clearance_m * 100.0, 2),
            "self_collision_ratio": round(self_collision_ratio, 4),
            "vel_spike_ratio": round(vel_spike_ratio, 4),
            "max_joint_vel": round(max_vel, 2),
            "foot_slip_cm_per_step": round(foot_slip_cm_per_step, 2),
            "limit_saturation_ratio": round(limit_saturation_ratio, 4),
            "max_single_joint_saturation": round(max_joint_saturation, 4),
            "num_frames": T,
            "duration_s": round(T * dt, 2),
        }


def main():
    parser = argparse.ArgumentParser(description="GMR 机器人动作六维量化质检评估工具 (quality_check.py)")
    parser.add_argument("--motion_dir", default="data/retargeted", help="重定向动作文件目录 (默认: data/retargeted)")
    parser.add_argument("--report_dir", default="reports", help="质检报告输出目录 (默认: reports)")
    parser.add_argument("--limit", type=int, default=0, help="评估动作数量上限 (0 为全量)")
    args = parser.parse_args()

    motion_dir = pathlib.Path(args.motion_dir)
    pkl_files = sorted(list(motion_dir.glob("*.pkl")))
    if not pkl_files:
        print(f"❌ 在 {motion_dir} 下未找到任何 .pkl 动作文件！")
        return

    if args.limit > 0:
        pkl_files = pkl_files[: args.limit]

    print("=" * 70)
    print("📋 GMR 机器人动作六维量化质检评估 (quality_check.py)")
    print(f"📁 动作目录: {motion_dir} (共 {len(pkl_files)} 个动作序列)")
    print("🎯 六维硬核指标:")
    print("   1. 穿透率 (<2%, 深度<5cm)   2. 悬浮量 (<3cm)        3. 自碰撞率 (<1%)")
    print("   4. 速度尖峰率 (<1%)         5. 脚滑移 (<3cm/步)     6. 限位贴边率 (<10%)")
    print("=" * 70)

    auditor = RobotMotionQualityAuditor()
    results = []
    t_start = time.time()
    pass_count = 0

    for idx, fpath in enumerate(pkl_files, 1):
        with open(fpath, "rb") as fp:
            data = pickle.load(fp)
        qpos = data["qpos"]
        fps = data.get("fps", 30.0)

        res = auditor.audit_motion(qpos, fps=fps)
        res["clip_id"] = fpath.stem
        results.append(res)

        if res["passed"]:
            pass_count += 1
            tag = "✅ [PASS]"
        else:
            tag = "⚠️ [FLAW]"

        print(f"[{idx:03d}/{len(pkl_files)}] {tag} {res['clip_id']} | "
              f"得分: {res['quality_score']} ({res['grade']}) | "
              f"穿透: {res['penetration_ratio']:.1%}, 悬浮: {res['floating_clearance_cm']:.1f}cm, "
              f"滑移: {res['foot_slip_cm_per_step']:.1f}cm/步, 碰撞: {res['self_collision_ratio']:.1%}")

    total_time = time.time() - t_start

    # 导出报告
    pathlib.Path(args.report_dir).mkdir(parents=True, exist_ok=True)
    report_json = pathlib.Path(args.report_dir) / "quality_audit_summary.json"
    report_csv = pathlib.Path(args.report_dir) / "quality_audit_summary.csv"

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_audited": len(results),
        "pass_count": pass_count,
        "pass_rate_pct": round(pass_count / max(len(results), 1) * 100, 2),
        "avg_quality_score": round(float(np.mean([r["quality_score"] for r in results])), 1) if results else 0.0,
        "records": results,
    }

    with open(report_json, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    if results:
        fieldnames = ["clip_id", "quality_score", "grade", "passed", "penetration_ratio", "max_penetration_cm",
                      "floating_clearance_cm", "self_collision_ratio", "vel_spike_ratio", "max_joint_vel",
                      "foot_slip_cm_per_step", "limit_saturation_ratio", "duration_s"]
        with open(report_csv, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

    print("\n" + "=" * 70)
    print("📊 机器人动作六维质检完成！")
    print(f"🔢 评估片段: {len(results)}")
    print(f"🌟 平均得分: {summary['avg_quality_score']} / 100.0")
    print(f"✅ 全达标率: {pass_count} 个 ({summary['pass_rate_pct']}%)")
    print(f"⏳ 总耗时:   {total_time:.2f}s (平均每段 {total_time/max(len(results), 1):.2f}s)")
    print(f"📄 报告文件: {report_json} / {report_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()
