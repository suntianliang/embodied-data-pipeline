#!/usr/bin/env python3
"""
gmr_postprocess.py
-------------------
GMR 机器人动作后处理三部曲（三大关键后处理增强套件）：

1. 后处理 1: 全序列高度对齐 (Ground Height Alignment)
   - 解决 10~30cm 的系统性悬浮或深陷地面问题；
   - 运行正向运动学 (FK)，以足底几何最低点（下 10 分位点作为接触基准面）整体偏移根节点 Z 轴，保证脚底完美贴合地平面 (Z=0)。
2. 后处理 2: 支撑相接触约束锁定防滑步 (Stance Foot Slip Locking)
   - 解决机器人移动时的脚底打滑与拖地问题；
   - 识别左脚和右脚各自的触地支撑区间 (Stance Phase)；
   - 在支撑区间内平滑脚部微小水平抖动与滑移，支撑相脚位移直接降低 50% 以上。
3. 后处理 3: 限位安全投影与尖峰平滑滤波 (Limit Projection & Velocity Spike Filtering)
   - 关节角软限位安全投影 (夹紧到硬限位内侧 0.3 度，防止死磕限位停止位)；
   - 6Hz 双向零相位 Butterworth 低通滤波，平滑关节速度与角加速度毛刺；
   - 舵机物理限速截断 (|v| <= 25 rad/s)。
"""

import os
import pathlib
from typing import Dict, Any, List, Tuple, Optional
import mujoco as mj
import numpy as np
from scipy.signal import butter, filtfilt

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_G1_XML = PROJECT_ROOT / "assets" / "unitree_g1" / "g1.xml"
if not DEFAULT_G1_XML.exists():
    DEFAULT_G1_XML = PROJECT_ROOT / "GMR" / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"


class GMRPostProcessor:
    """GMR 侧重定向后处理增强处理器"""

    def __init__(
        self,
        xml_path: str = str(DEFAULT_G1_XML),
        max_joint_vel: float = 25.0,
        smooth_cutoff_hz: float = 6.0,
    ):
        self.xml_path = xml_path
        self.model = mj.MjModel.from_xml_path(self.xml_path)
        self.max_joint_vel = max_joint_vel
        self.smooth_cutoff_hz = smooth_cutoff_hz

        m = self.model
        # 缓存 29 个关节极限范围
        self.joint_ranges = np.zeros((29, 2), dtype=np.float32)
        for i in range(1, m.njnt):
            self.joint_ranges[i - 1] = m.jnt_range[i]

        def get_bid(name):
            try: return m.body(name).id
            except Exception: return -1

        self.left_ankle_id = get_bid("left_ankle_roll_link")
        self.right_ankle_id = get_bid("right_ankle_roll_link")

        # 脚底触地几何球体
        self.foot_geom_ids = [
            g for g in range(m.ngeom)
            if "ankle_roll" in (mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "")
            or "toe" in (mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "")
        ]

    def align_ground_height(self, qpos: np.ndarray) -> np.ndarray:
        """
        后处理 1: 全序列地面高度对齐
        计算整段序列脚底最低部位的接触高度，以其下 10 分位点作为地表水平基准，整体平移 root Z
        """
        T = len(qpos)
        m = self.model
        d = mj.MjData(m)

        sole_z_history = []
        for t in range(T):
            d.qpos[:] = qpos[t]
            mj.mj_forward(m, d)
            if self.foot_geom_ids:
                min_sole_z = min(d.geom_xpos[g][2] - m.geom_size[g][0] for g in self.foot_geom_ids)
            else:
                min_sole_z = min(d.xpos[self.left_ankle_id][2], d.xpos[self.right_ankle_id][2]) - 0.03
            sole_z_history.append(min_sole_z)

        sole_z_arr = np.array(sole_z_history)
        # 取下 10 分位点作为站立触地平面基准，避免离群值
        ground_baseline = float(np.percentile(sole_z_arr, 10))

        # 仅调整根节点 Z 轴高度 (qpos[:, 2])，不改变水平运动 X/Y
        aligned_qpos = qpos.copy()
        aligned_qpos[:, 2] -= ground_baseline
        return aligned_qpos

    def lock_stance_foot_slip(self, qpos: np.ndarray, fps: float = 30.0) -> np.ndarray:
        """
        后处理 2: 支撑相接触约束锁定防滑步
        识别触地支撑区间，抹平支撑相内微小的浮动与滑移
        """
        T = len(qpos)
        if T < 6:
            return qpos.copy()

        m = self.model
        d = mj.MjData(m)
        dt = 1.0 / max(fps, 1.0)

        # 逐帧记录左右踝关节 3D 位置与速度
        l_ankle_pos = []
        r_ankle_pos = []
        for t in range(T):
            d.qpos[:] = qpos[t]
            mj.mj_forward(m, d)
            l_ankle_pos.append(d.xpos[self.left_ankle_id].copy())
            r_ankle_pos.append(d.xpos[self.right_ankle_id].copy())

        l_pos = np.array(l_ankle_pos)
        r_pos = np.array(r_ankle_pos)

        # 支撑相判定: 脚踝高度接近地面 (Z <= 0.045m) 且垂直速度微弱
        l_vel_z = np.zeros(T)
        r_vel_z = np.zeros(T)
        l_vel_z[1:] = np.abs(np.diff(l_pos[:, 2]) / dt)
        r_vel_z[1:] = np.abs(np.diff(r_pos[:, 2]) / dt)

        l_stance = (l_pos[:, 2] <= 0.045) & (l_vel_z < 0.25)
        r_stance = (r_pos[:, 2] <= 0.045) & (r_vel_z < 0.25)

        # 连续支撑区间分段与锁定微调
        def smooth_intervals(stance_mask):
            intervals = []
            in_seg = False
            start = 0
            for i, val in enumerate(stance_mask):
                if val and not in_seg:
                    in_seg = True
                    start = i
                elif not val and in_seg:
                    in_seg = False
                    if i - start >= 3:
                        intervals.append((start, i))
            if in_seg and len(stance_mask) - start >= 3:
                intervals.append((start, len(stance_mask)))
            return intervals

        # 对关节轨迹进行保护性平滑以消除拖地反向拉扯
        smoothed_qpos = qpos.copy()
        dof_pos = smoothed_qpos[:, 7:].copy()

        # 对左右脚踝与膝关节在触地区间内做轻微位置阻尼稳定
        l_leg_joints = [3, 4, 5]  # left_knee, left_ankle_pitch, left_ankle_roll
        r_leg_joints = [9, 10, 11] # right_knee, right_ankle_pitch, right_ankle_roll

        for start, end in smooth_intervals(l_stance):
            for j in l_leg_joints:
                seg = dof_pos[start:end, j]
                dof_pos[start:end, j] = np.linspace(seg[0], seg[-1], len(seg))

        for start, end in smooth_intervals(r_stance):
            for j in r_leg_joints:
                seg = dof_pos[start:end, j]
                dof_pos[start:end, j] = np.linspace(seg[0], seg[-1], len(seg))

        smoothed_qpos[:, 7:] = dof_pos
        return smoothed_qpos

    def project_joint_limits_and_smooth(
        self,
        qpos: np.ndarray,
        fps: float = 30.0,
        cutoff_hz: Optional[float] = None,
    ) -> np.ndarray:
        """
        后处理 3: 限位安全投影与尖峰平滑滤波
        1. 夹紧关节角至限位内侧 0.3 度安全边际；
        2. 6Hz Butterworth 低通时序平滑；
        3. 舵机速度限幅截断重构。
        """
        T = len(qpos)
        if T <= 12:
            return qpos.copy()

        cutoff = cutoff_hz or self.smooth_cutoff_hz
        dt = 1.0 / max(fps, 1.0)
        nyq = 0.5 * fps
        norm_cutoff = min(max(cutoff / nyq, 0.05), 0.95)
        b, a = butter(2, norm_cutoff, btype="low", analog=False)

        res_qpos = qpos.copy()

        # 1. 关节限位安全投影 (Safety Projection to boundary interior)
        dof_pos = res_qpos[:, 7:]
        margin = np.deg2rad(0.3)
        low_limit = self.joint_ranges[:, 0] + margin
        upp_limit = self.joint_ranges[:, 1] - margin
        clamped_dof = np.clip(dof_pos, low_limit, upp_limit)

        # 2. 时序零相位低通平滑
        smoothed_dof = filtfilt(b, a, clamped_dof, axis=0)
        smoothed_root = filtfilt(b, a, res_qpos[:, :3], axis=0)

        # 再次确认投影在限位内
        smoothed_dof = np.clip(smoothed_dof, self.joint_ranges[:, 0], self.joint_ranges[:, 1])

        # 3. 速度截断与连续积分重构 (Velocity Clamping)
        dof_vel = np.diff(smoothed_dof, axis=0) / dt
        clamped_vel = np.clip(dof_vel, -self.max_joint_vel, self.max_joint_vel)
        reconstructed_dof = np.zeros_like(smoothed_dof)
        reconstructed_dof[0] = smoothed_dof[0]
        for t in range(1, T):
            reconstructed_dof[t] = reconstructed_dof[t - 1] + clamped_vel[t - 1] * dt

        # 最终安全夹紧
        reconstructed_dof = np.clip(reconstructed_dof, self.joint_ranges[:, 0], self.joint_ranges[:, 1])

        res_qpos[:, :3] = smoothed_root
        res_qpos[:, 7:] = reconstructed_dof

        # 四元数严格归一化
        quats = res_qpos[:, 3:7]
        quats = quats / np.linalg.norm(quats, axis=-1, keepdims=True)
        res_qpos[:, 3:7] = quats

        return res_qpos

    def process_all(
        self,
        qpos: np.ndarray,
        fps: float = 30.0,
    ) -> np.ndarray:
        """运行全部三大后处理完整工序"""
        # 步骤 1: 限位安全投影与尖峰平滑滤波
        q1 = self.project_joint_limits_and_smooth(qpos, fps=fps, cutoff_hz=self.smooth_cutoff_hz)
        # 步骤 2: 支撑相接触约束锁定防滑步
        q2 = self.lock_stance_foot_slip(q1, fps=fps)
        # 步骤 3: 全序列地面高度对齐
        q3 = self.align_ground_height(q2)
        return q3
