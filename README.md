# Embodied Data Pipeline (具身机器人运动数据全流程管线)

<p align="center">
  <b>从野外单目 RGB 视频到 Unitree G1 (29 DoF) 人形机器人全身模仿学习轨迹的全自动化离线生产管线</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Robot-Unitree%20G1%20(29DoF)-blue?style=flat-square" alt="Robot">
  <img src="https://img.shields.io/badge/Python-3.10%2B-brightgreen?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/Kinematics-GMR%20IK-orange?style=flat-square" alt="GMR">
  <img src="https://img.shields.io/badge/Physics%20Engine-MuJoCo-red?style=flat-square" alt="MuJoCo">
  <img src="https://img.shields.io/badge/Pose-GVHMR%20%2F%20SMPL--X-purple?style=flat-square" alt="Pose">
</p>

---

## 📖 项目简介 (Overview)

**Embodied Data Pipeline** 是一套端到端的具身人形机器人动作数据提炼与重定向系统。针对真实互联网视频中普遍存在的**视角抖动、出画遮挡、脚底打滑、悬空下陷、关节越界及自碰撞**等问题，构建了全自动的四级漏斗过滤、多维物理可行性校验与运动学后处理优化，为下游模仿学习（Imitation Learning / BeyondMimic）与强化学习策略训练提供高质量、高物理逼真度的运动轨迹库。

---

## 🚀 核心架构与处理流程 (Pipeline Architecture)

```
[野外视频采集] (YouTube / Pixabay)
       │  collect_videos.py
       ▼
[智能切片与重采样] (CFR 30fps, 3-10s 动态片段)
       │  slice_clips.py
       ▼
[前端质量漏斗过滤] (L0 元数据 + L1 视觉质量/脚出画 + L2 YOLOv8-pose Jerk)
       │  filter_video.py (PASS / REVIEW / REJECT)
       ▼
[3D 人体运动恢复] (GVHMR 重力视角 SMPL-X 姿态与世界轨迹)
       │
       ▼
[姿态与物理可行性质检] (置信度、滤波去抖、G1 速度/加速度/限位/MuJoCo 自碰撞)
       │  filter_pose.py + robot_feasibility.py
       ▼
[GMR 动作重定向] (SMPL-X 到 Unitree G1 29DoF 逆运动学求解)
       │  batch_retarget_g1.py
       ▼
[运动学后处理三部曲] (地面高度对齐 + 支撑相零滑移锁定 + 限位内投影)
       │  gmr_postprocess.py
       ▼
[六维动作质量审计] (穿地率、浮空距离、自碰撞、速度突刺、足底滑移、限位饱和度)
       │  quality_check.py (输出 A/B/C 评级与报告)
       ▼
[合格运动轨迹库] (.pkl / .npz) ──► 供 BeyondMimic / RL Policy 训练使用
```

---

## 🛠️ 功能模块索引 (Modules)

| 阶段 / 模块 | 核心脚本 | 核心功能说明 |
| :--- | :--- | :--- |
| **视频采集** | [`collect_videos.py`](collect_videos.py) | 基于 `yt-dlp` 的多源运动视频自动下载，覆盖行走、跑步、深蹲、踢腿、跑酷等 10 维动作 |
| **视频预处理** | [`slice_clips.py`](slice_clips.py) | 基于 `ffmpeg` 将视频统一重采样为恒定 30fps，去除头尾静止帧并切片为 3~10 秒有效片段 |
| **前端漏斗过滤** | [`filter_video.py`](filter_video.py) | L0 容器元数据 + L1 视觉质量（**脚出画检测**、光流相机抖动）+ L2 轻量姿态 Jerk 抖动三档裁决 |
| **姿态与物理质检** | [`filter_pose.py`](filter_pose.py) | 校验 SMPL 关键点置信度、运动活跃度、时域突变/主体切换，执行 6Hz Butterworth 低通滤波 |
| **机器人物理准入** | [`robot_feasibility.py`](robot_feasibility.py) | Unitree G1 电机角速度/加速度限值、手臂极限饱和度（≤20% 拦截反手摸背）、MuJoCo 自碰撞检测 |
| **动作重定向** | [`batch_retarget_g1.py`](batch_retarget_g1.py) | 基于 GMR IK 求解器批量生成 Unitree G1 关节轨迹，并录制 30fps MuJoCo Review 视频 |
| **GMR 后处理** | [`gmr_postprocess.py`](gmr_postprocess.py) | 地面高度绝对对齐 (根骨骼 Z 轴重偏置) + 支撑相脚滑锁定 (Zero-Slip Lock) + 关节限位内投影 |
| **六维质量审计** | [`quality_check.py`](quality_check.py) | 6 维量化综合审计（穿地、浮空、自碰撞、速度突刺、滑移、限位饱和），输出诊断报告 |
| **全局配置** | [`config.yaml`](config.yaml) | 统一配置参数中心（路径、阈值、物理常数、多进程设置） |

---

## 💻 快速上手 (Quick Start)

### 1. 环境安装

推荐使用 Conda 创建独立 Python 3.10 环境：

```bash
conda create -n edp python=3.10 -y
conda activate edp

# 安装 PyTorch (建议根据平台选择合适版本)
pip install torch torchvision

# 安装 MuJoCo 与数据处理依赖
pip install mujoco opencv-python ultralytics yt-dlp pyyaml rich tqdm scipy numpy

# 本地以可编辑模式安装内置 GMR 库
pip install -e ./GMR
```

### 2. 视频采集与切片

```bash
# 1. 采集运动视频 (YouTube + Pixabay)
python collect_videos.py --download-all

# 2. 30fps CFR 重采样与切片
python slice_clips.py --workers 4
```

### 3. 前端视频质量漏斗过滤

```bash
python filter_video.py --input-dir data/clips --output-dir reports
```

### 4. 姿态质检与机器人物理可行性过滤

```bash
# 针对 GVHMR 结果进行 6 重姿态门限 + G1 机器人物理可行性准入筛选
python filter_pose.py --results_dir /path/to/GVHMR_results --out_dir data/poses_filtered
```

### 5. GMR 批量重定向与 MuJoCo Review 视频生成

```bash
python batch_retarget_g1.py --input-dir data/poses_filtered --record-video --workers 4
```

### 6. 动作质量审计

```bash
python quality_check.py --input-dir data/retargeted --output-dir reports
```

---

## 📊 质量审计指标体系 (Quality Metrics)

系统通过 6 大核心物理维度对重定向后的动作进行量化评分：

1. **地面穿透率 (Penetration Rate)**: 足底深入地面超过 $4\text{cm}$ 的帧占比（合格：$<2\%$）。
2. **浮空距离 (Floating Clearance)**: 站立支撑相足底距离地面平均高度（合格：$<3\text{cm}$）。
3. **自碰撞率 (Self-collision Rate)**: 左右手腕及躯干网格穿模发生帧率（合格：$0\%$）。
4. **速度突刺率 (Velocity Spike Rate)**: 关节角速度超过电机极限 $25\text{ rad/s}$ 的尖峰率（合格：$<1\%$）。
5. **支撑相滑移 (Stance Slip)**: 支撑相足底水平滑移量（合格：$<3\text{cm/步}$）。
6. **限位贴边率 (ROM Saturation)**: 关节角度落在极限边界 $\pm 2^\circ$ 内的占比（合格：$<20\%$）。

---

## 📄 对话与技术演进全书

项目中包含完整的开发与设计全记录：
* [conversation_history.md](conversation_history.md): 记录了从最初需求沟通、算法选型、多级过滤设计、到物理引擎冲突调优的完整技术问答与工程交付全书。

---

## ⚖️ License & Acknowledgements

- 核心管线代码采用 MIT 许可证。
- GMR 运动重定向模块遵循其原开源协议。
- 机器人模型与仿真资产来源于 Unitree Robotics 与 MuJoCo Menagerie。
