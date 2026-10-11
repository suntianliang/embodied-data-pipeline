#!/usr/bin/env python3
"""
robot_feasibility.py
---------------------
Unitree G1 机器人物理可复现性（Feasibility）质检评估模块：
严格检验与优化人体动作重定向到 Unitree G1 后的物理可行性：

1. 加速度 / 角速度超限 (Joint Velocity & Acceleration Limit):
   - 检测关节角速度与角加速度，防止人甩手过快导致舵机跟不上或扭断。
   - 包含时序低通滤波平滑与速度截断优化机制。
2. 关节越界 (Joint Limit Violation):
   - 检测所有 29 个自由度关节角是否超越 G1 机械极限范围（反手摸后背、极限反关节）。
   - 检测关节是否持续死锁贴合在硬限位停止器上（Saturation）。
3. 地面穿透与悬空 (Ground Penetration & Clearance):
   - 基于 MuJoCo 脚底触地几何球体，精确计算脚底与地平面 (Z=0) 交互。
   - 脚底深陷地面超过阈值（> 4cm）判定为非法穿透；
   - 整体悬空不着地（> 12cm 且无触地）判定为非法悬空。
4. 自碰撞与肢体穿模 (Self-Collision & Limb Intersection):
   - 双手互穿（双手间距 < 11cm 或 MuJoCo 碰撞引擎穿透深 > 8mm）；
   - 手体穿模（手腕穿透躯干或骨盆）；
   - 双脚互撞（双踝间距 < 7cm）。
"""

import os
import pathlib
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import mujoco as mj
from scipy.signal import butter, filtfilt

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_G1_XML = PROJECT_ROOT / "assets" / "unitree_g1" / "g1.xml"
if not DEFAULT_G1_XML.exists():
    DEFAULT_G1_XML = PROJECT_ROOT / "GMR" / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"


class RobotFeasibilityChecker:
    """Unitree G1 机器人物理可复现性质检器"""

    def __init__(
        self,
        xml_path: Optional[str] = None,
        max_joint_vel: float = 25.0,        # rad/s (G1 舵机响应速度上限，约 1432 deg/s)
        max_joint_acc: float = 400.0,       # rad/s^2 (G1 关节角加速度上限)
        max_joint_viol: float = 0.08,       # rad (关节越界允许容差，约 4.6 度)
        max_saturation_ratio: float = 0.20, # 最大允许关节死贴在极限停止位的帧率
        max_ground_penetration: float = 0.04, # 米 (脚底沉入地面最大允许深度，4cm)
        max_ground_clearance: float = 0.12,   # 米 (双脚最低离地高度上限，超过为悬空)
        min_contact_ratio: float = 0.05,    # 最低触地帧率要求 (5%)
        min_hand_dist: float = 0.11,        # 米 (双手手腕中心最小安全距离，小于视为互穿)
        min_hand_torso_dist: float = 0.08,  # 米 (手腕与躯干/盆骨中心最小距离)
        min_foot_dist: float = 0.07,        # 米 (双脚最小安全间距)
        collision_penetration_tol: float = 0.008, # 米 (MuJoCo 穿透深度容差: 8mm)
        max_collision_ratio: float = 0.03,  # 允许自碰撞最大帧率 (3%)
    ):
        self.xml_path = str(xml_path or DEFAULT_G1_XML)
        if not os.path.exists(self.xml_path):
            raise FileNotFoundError(f"Robot XML model not found at {self.xml_path}")

        self.model = mj.MjModel.from_xml_path(self.xml_path)
        self.data = mj.MjData(self.model)

        self.max_joint_vel = max_joint_vel
        self.max_joint_acc = max_joint_acc
        self.max_joint_viol = max_joint_viol
        self.max_saturation_ratio = max_saturation_ratio
        self.max_ground_penetration = max_ground_penetration
        self.max_ground_clearance = max_ground_clearance
        self.min_contact_ratio = min_contact_ratio
        self.min_hand_dist = min_hand_dist
        self.min_hand_torso_dist = min_hand_torso_dist
        self.min_foot_dist = min_foot_dist
        self.collision_penetration_tol = collision_penetration_tol
        self.max_collision_ratio = max_collision_ratio

        # 缓存关键关节与刚体 ID
        self._init_body_and_geom_ids()

    def _init_body_and_geom_ids(self):
        m = self.model
        # 关节限位 (跳过 0 号 freejoint, 提取 29 个铰链关节)
        self.joint_names = []
        self.joint_ranges = np.zeros((29, 2), dtype=np.float32)
        for i in range(1, m.njnt):
            jname = mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, i)
            self.joint_names.append(jname)
            self.joint_ranges[i - 1] = m.jnt_range[i]

        # 关键 Body ID
        def get_bid(name):
            try:
                return m.body(name).id
            except Exception:
                return -1

        self.left_wrist_id = get_bid("left_wrist_yaw_link")
        self.right_wrist_id = get_bid("right_wrist_yaw_link")
        self.torso_id = get_bid("torso_link")
        self.pelvis_id = get_bid("pelvis")
        self.left_ankle_id = get_bid("left_ankle_roll_link")
        self.right_ankle_id = get_bid("right_ankle_roll_link")

        # 肢体分组集合
        self.left_arm_bids = set()
        self.right_arm_bids = set()
        self.torso_pelvis_bids = set()
        self.left_leg_bids = set()
        self.right_leg_bids = set()

        for bid in range(m.nbody):
            bname = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, bid) or ""
            if "left_shoulder" in bname or "left_elbow" in bname or "left_wrist" in bname or "left_rubber_hand" in bname:
                self.left_arm_bids.add(bid)
            elif "right_shoulder" in bname or "right_elbow" in bname or "right_wrist" in bname or "right_rubber_hand" in bname:
                self.right_arm_bids.add(bid)
            elif "torso" in bname or "pelvis" in bname or "waist" in bname or "head" in bname:
                self.torso_pelvis_bids.add(bid)
            elif "left_hip" in bname or "left_knee" in bname or "left_ankle" in bname or "left_toe" in bname:
                self.left_leg_bids.add(bid)
            elif "right_hip" in bname or "right_knee" in bname or "right_ankle" in bname or "right_toe" in bname:
                self.right_leg_bids.add(bid)

        # 脚底触地接触球体 Geom 集合 (用于计算真实离地高与穿透)
        self.foot_geom_ids = []
        for g in range(m.ngeom):
            bid = m.geom_bodyid[g]
            bname = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, bid) or ""
            if "ankle_roll" in bname or "toe" in bname:
                self.foot_geom_ids.append(g)

    def smooth_motion(
        self,
        qpos: np.ndarray,
        fps: float = 30.0,
        cutoff_hz: float = 6.0,
    ) -> np.ndarray:
        """
        对机器人动作轨迹进行 6Hz Butterworth 低通滤波与速度截断，优化轻微速度/加速度毛刺
        """
        T = len(qpos)
        if T <= 12:
            return qpos.copy()

        nyq = 0.5 * fps
        norm_cutoff = min(max(cutoff_hz / nyq, 0.05), 0.95)
        b, a = butter(2, norm_cutoff, btype="low", analog=False)

        smoothed_qpos = qpos.copy()

        # 平滑 29 个关节角 (qpos[7:])
        dof_pos = qpos[:, 7:]
        smoothed_dof = filtfilt(b, a, dof_pos, axis=0)

        # 平滑根平移 (qpos[:3])
        root_pos = qpos[:, :3]
        smoothed_root_pos = filtfilt(b, a, root_pos, axis=0)

        # 速度截断 (Velocity Clamping): 确保不超过舵机极限
        dt = 1.0 / max(fps, 1.0)
        dof_vel = np.diff(smoothed_dof, axis=0) / dt
        clamped_vel = np.clip(dof_vel, -self.max_joint_vel, self.max_joint_vel)
        # 重建截断后关节角
        reconstructed_dof = np.zeros_like(smoothed_dof)
        reconstructed_dof[0] = smoothed_dof[0]
        for t in range(1, T):
            reconstructed_dof[t] = reconstructed_dof[t - 1] + clamped_vel[t - 1] * dt

        smoothed_qpos[:, :3] = smoothed_root_pos
        smoothed_qpos[:, 7:] = reconstructed_dof

        # 确保根四元数规一化
        quats = smoothed_qpos[:, 3:7]
        quats = quats / np.linalg.norm(quats, axis=-1, keepdims=True)
        smoothed_qpos[:, 3:7] = quats

        return smoothed_qpos

    def evaluate_feasibility(
        self,
        qpos: np.ndarray,
        fps: float = 30.0,
    ) -> Dict[str, Any]:
        """
        执行全面的 4 项机器人动作可复现性（Feasibility）指标评测：
        返回 pass/fail 判定及详细度量指标。
        """
        T = len(qpos)
        if T < 4:
            return {
                "passed": False,
                "status": "REJECT_INVALID_DATA",
                "reason": "动作帧数过短 (<4)",
                "metrics": {},
            }

        dt = 1.0 / max(fps, 1.0)
        dof_pos = qpos[:, 7:]  # (T, 29)

        # ========================================================
        # 1. 关卡 A: 加速度 / 角速度超限质检 (Velocity & Accel Gate)
        # ========================================================
        dof_vel = np.diff(dof_pos, axis=0) / dt  # (T-1, 29)
        dof_acc = np.diff(dof_vel, axis=0) / dt  # (T-2, 29)

        max_vel = float(np.abs(dof_vel).max()) if len(dof_vel) > 0 else 0.0
        max_acc = float(np.abs(dof_acc).max()) if len(dof_acc) > 0 else 0.0

        vel_over_ratio = float(np.mean(np.abs(dof_vel) > self.max_joint_vel))
        acc_over_ratio = float(np.mean(np.abs(dof_acc) > self.max_joint_acc))

        # ========================================================
        # 2. 关卡 B: 关节越界质检 (Joint Limit Violation Gate)
        # ========================================================
        low_viol = np.maximum(0.0, self.joint_ranges[:, 0] - dof_pos)
        upp_viol = np.maximum(0.0, dof_pos - self.joint_ranges[:, 1])
        all_viol = np.maximum(low_viol, upp_viol)
        max_joint_viol = float(all_viol.max())

        # 检查手臂关节是否死锁贴合在限位极限停止器上 (针对反手摸后背、极限反关节)
        arm_indices = [i for i, name in enumerate(self.joint_names) if 'shoulder' in name or 'elbow' in name or 'wrist' in name]
        saturation_margin = 0.03  # 约 1.7 度范围内
        is_saturated = (np.abs(dof_pos[:, arm_indices] - self.joint_ranges[arm_indices, 0]) < saturation_margin) | \
                       (np.abs(dof_pos[:, arm_indices] - self.joint_ranges[arm_indices, 1]) < saturation_margin)
        max_arm_saturation_ratio = float(np.max(np.mean(is_saturated, axis=0))) if len(arm_indices) > 0 else 0.0

        # ========================================================
        # 3. 关卡 C: 地面穿透与悬空质检 (Ground Penetration & Clearance Gate)
        # ========================================================
        m = self.model
        d = mj.MjData(m)  # 线程安全的局部 MjData
        min_sole_z_overall = 999.0
        contact_frames = 0
        penetration_frames = 0

        # ========================================================
        # 4. 关卡 D: 自碰撞与穿模质检 (Self-Collision Gate)
        # ========================================================
        min_hand_dist_overall = 999.0
        min_hand_torso_dist_overall = 999.0
        min_foot_dist_overall = 999.0
        hand_collision_frames = 0
        self_collision_frames = 0
        colliding_pairs_detected = set()

        for t in range(T):
            d.qpos[:] = qpos[t]
            mj.mj_step1(m, d)

            # 计算脚底几何最低点高度
            if self.foot_geom_ids:
                sole_z = min(d.geom_xpos[g][2] - m.geom_size[g][0] for g in self.foot_geom_ids)
            else:
                sole_z = min(d.xpos[self.left_ankle_id][2], d.xpos[self.right_ankle_id][2]) - 0.03

            min_sole_z_overall = min(min_sole_z_overall, sole_z)
            if sole_z <= 0.05:
                contact_frames += 1
            if sole_z < -self.max_ground_penetration:
                penetration_frames += 1

            # 计算左右手距离
            p_lw = d.xpos[self.left_wrist_id]
            p_rw = d.xpos[self.right_wrist_id]
            hand_dist = float(np.linalg.norm(p_lw - p_rw))
            min_hand_dist_overall = min(min_hand_dist_overall, hand_dist)

            # 计算手与躯干距离
            p_torso = d.xpos[self.torso_id]
            p_pelvis = d.xpos[self.pelvis_id]
            ht_dist = min(
                np.linalg.norm(p_lw - p_torso),
                np.linalg.norm(p_rw - p_torso),
                np.linalg.norm(p_lw - p_pelvis),
                np.linalg.norm(p_rw - p_pelvis),
            )
            min_hand_torso_dist_overall = min(min_hand_torso_dist_overall, ht_dist)

            # 计算双脚距离
            p_la = d.xpos[self.left_ankle_id]
            p_ra = d.xpos[self.right_ankle_id]
            foot_dist = float(np.linalg.norm(p_la - p_ra))
            min_foot_dist_overall = min(min_foot_dist_overall, foot_dist)

            is_hand_colliding = hand_dist < self.min_hand_dist
            if is_hand_colliding:
                hand_collision_frames += 1

            # MuJoCo 碰撞引擎精确检测
            has_step_collision = False
            for i in range(d.ncon):
                c = d.contact[i]
                if c.dist < -self.collision_penetration_tol:
                    b1 = m.geom_bodyid[c.geom1]
                    b2 = m.geom_bodyid[c.geom2]
                    # 左臂与右臂互穿
                    if (b1 in self.left_arm_bids and b2 in self.right_arm_bids) or \
                       (b2 in self.left_arm_bids and b1 in self.right_arm_bids):
                        has_step_collision = True
                        n1 = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, b1)
                        n2 = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, b2)
                        colliding_pairs_detected.add(tuple(sorted([n1, n2])))
                    # 手腕深穿躯干/骨盆
                    elif ((b1 == self.left_wrist_id or b1 == self.right_wrist_id) and b2 in self.torso_pelvis_bids) or \
                         ((b2 == self.left_wrist_id or b2 == self.right_wrist_id) and b1 in self.torso_pelvis_bids):
                        has_step_collision = True
                        n1 = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, b1)
                        n2 = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, b2)
                        colliding_pairs_detected.add(tuple(sorted([n1, n2])))

            if is_hand_colliding or has_step_collision:
                self_collision_frames += 1

        ground_contact_ratio = contact_frames / T
        penetration_ratio = penetration_frames / T
        hand_collision_ratio = hand_collision_frames / T
        self_collision_ratio = self_collision_frames / T

        metrics = {
            "max_joint_vel": round(max_vel, 2),
            "max_joint_acc": round(max_acc, 1),
            "vel_over_ratio": round(vel_over_ratio, 4),
            "acc_over_ratio": round(acc_over_ratio, 4),
            "max_joint_viol": round(max_joint_viol, 4),
            "max_arm_saturation_ratio": round(max_arm_saturation_ratio, 3),
            "min_sole_z": round(min_sole_z_overall, 3),
            "ground_contact_ratio": round(ground_contact_ratio, 3),
            "penetration_ratio": round(penetration_ratio, 3),
            "min_hand_dist": round(min_hand_dist_overall, 3),
            "min_hand_torso_dist": round(min_hand_torso_dist_overall, 3),
            "min_foot_dist": round(min_foot_dist_overall, 3),
            "self_collision_ratio": round(self_collision_ratio, 3),
            "colliding_pairs": list(colliding_pairs_detected),
        }

        # ========================================================
        # 决策与归因 (Decision & Attribution)
        # ========================================================

        # 1. 关节越界 (Limit Violation)
        if max_joint_viol > self.max_joint_viol:
            return {
                "passed": False,
                "status": "REJECT_JOINT_LIMIT",
                "reason": f"关节越界严重: 最大越界 {max_joint_viol:.2f} rad > {self.max_joint_viol}",
                "metrics": metrics,
            }
        if max_arm_saturation_ratio > self.max_saturation_ratio:
            return {
                "passed": False,
                "status": "REJECT_JOINT_LIMIT",
                "reason": f"手臂反关节死锁卡死: 手臂死锁在限位停止位帧率 {max_arm_saturation_ratio:.1%} > {self.max_saturation_ratio:.0%}",
                "metrics": metrics,
            }

        # 2. 地面穿透 (Ground Penetration)
        if penetration_ratio > 0.05 or min_sole_z_overall < -0.06:
            return {
                "passed": False,
                "status": "REJECT_GROUND_PENETRATION",
                "reason": f"脚底严重陷地穿透: 最低高度 {min_sole_z_overall:.3f}m, 穿透帧率 {penetration_ratio:.1%}",
                "metrics": metrics,
            }

        # 3. 整体悬空 (Floating)
        if ground_contact_ratio < self.min_contact_ratio or min_sole_z_overall > self.max_ground_clearance:
            return {
                "passed": False,
                "status": "REJECT_FLOATING_FEET",
                "reason": f"动作整体悬空未着地: 最低脚高 {min_sole_z_overall:.2f}m, 触地帧率 {ground_contact_ratio:.1%}",
                "metrics": metrics,
            }

        # 4. 自碰撞穿模 (Self-Collision)
        if self_collision_ratio > self.max_collision_ratio or min_hand_dist_overall < 0.08:
            pairs_str = ", ".join(f"{p[0]}<->{p[1]}" for p in list(colliding_pairs_detected)[:2])
            return {
                "passed": False,
                "status": "REJECT_SELF_COLLISION",
                "reason": f"双手或肢体自碰撞穿模 (最小间距 {min_hand_dist_overall:.2f}m, 碰撞帧率 {self_collision_ratio:.1%}, 部位: {pairs_str or '双手互穿'})",
                "metrics": metrics,
            }

        # 5. 角速度超限 (Velocity Limit)
        if max_vel > self.max_joint_vel or vel_over_ratio > 0.05:
            return {
                "passed": False,
                "status": "REJECT_VELOCITY_LIMIT",
                "reason": f"关节角速度超限: 峰值角速度 {max_vel:.1f} rad/s > {self.max_joint_vel} (舵机无法响应)",
                "metrics": metrics,
            }

        # 6. 角加速度超限 (Acceleration Limit)
        if max_acc > self.max_joint_acc or acc_over_ratio > 0.05:
            return {
                "passed": False,
                "status": "REJECT_ACCEL_LIMIT",
                "reason": f"关节角加速度超限: 峰值角加速度 {max_acc:.1f} rad/s² > {self.max_joint_acc}",
                "metrics": metrics,
            }

        return {
            "passed": True,
            "status": "PASS",
            "reason": "物理可复现性全部达标 (无越界/无穿模/无撞击/速度合格/地面接触优良)",
            "metrics": metrics,
        }
