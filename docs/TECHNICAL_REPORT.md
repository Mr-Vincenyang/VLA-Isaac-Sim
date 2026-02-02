# VLA Isaac Sim 仿真平台技术报告

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [VLA模型详解](#3-vla模型详解)
4. [Isaac Sim使用指南](#4-isaac-sim使用指南)
5. [运动控制方法](#5-运动控制方法)
6. [代码结构详解](#6-代码结构详解)
7. [训练流程](#7-训练流程)
8. [部署指南](#8-部署指南)

---

## 1. 项目概述

### 1.1 什么是VLA？

**VLA (Vision-Language-Action)** 是一类将视觉感知、语言理解和机器人动作生成统一的端到端模型。

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   图像输入   │ ──▶ │   VLA模型   │ ──▶ │  动作输出   │
│  (Vision)   │     │             │     │  (Action)   │
└─────────────┘     │  语言指令   │     └─────────────┘
                    │ (Language)  │
                    └─────────────┘
```

### 1.2 项目目标

本项目实现一个完整的VLA仿真复现平台，包括：

1. **OpenVLA模型复现** - 理解和本地部署7B参数VLA模型
2. **Isaac Sim集成** - 高保真机器人仿真环境
3. **RL微调** - 使用PPO+LoRA进行强化学习微调
4. **运动控制** - 阻抗控制、轨迹规划等高级控制方法

### 1.3 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| GPU (本地推理) | 8GB VRAM (INT4) | 16GB+ VRAM |
| GPU (训练) | 24GB VRAM | 48GB+ VRAM (A100) |
| CPU | 8核 | 16核+ |
| RAM | 32GB | 64GB+ |
| 存储 | 50GB SSD | 200GB+ NVMe |

---

## 2. 系统架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        VLA Platform                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────┐          ┌─────────────────────┐       │
│  │    本地 (Isaac Sim)  │          │    远程 (GPU Server) │       │
│  │                     │  REST/   │                     │       │
│  │  ┌───────────────┐  │  gRPC    │  ┌───────────────┐  │       │
│  │  │ Simulation    │  │ ◀──────▶ │  │ VLA Model     │  │       │
│  │  │ - Franka Env  │  │          │  │ - OpenVLA-7B  │  │       │
│  │  │ - Sensors     │  │          │  │ - RT-2        │  │       │
│  │  └───────────────┘  │          │  └───────────────┘  │       │
│  │                     │          │                     │       │
│  │  ┌───────────────┐  │          │  ┌───────────────┐  │       │
│  │  │ Control       │  │          │  │ Training      │  │       │
│  │  │ - PD Control  │  │          │  │ - PPO         │  │       │
│  │  │ - Impedance   │  │          │  │ - LoRA        │  │       │
│  │  │ - Trajectory  │  │          │  │ - Data        │  │       │
│  │  └───────────────┘  │          │  └───────────────┘  │       │
│  │                     │          │                     │       │
│  └─────────────────────┘          └─────────────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流

```
1. 观测采集
   Camera ─▶ RGB Image (224x224) ─▶ 预处理 ─▶ VLA Model

2. 动作生成
   VLA Model ─▶ Action Tokens ─▶ 解码 ─▶ 7-DoF Delta Action

3. 运动执行
   Delta Action ─▶ IK求解 ─▶ 关节控制 ─▶ 机器人执行
```

### 2.3 核心数据结构

```python
# 观测 (Observation)
@dataclass
class Observation:
    image: np.ndarray           # [H, W, 3] RGB图像
    depth: Optional[np.ndarray] # [H, W] 深度图
    joint_positions: np.ndarray # [7] 关节位置
    ee_position: np.ndarray     # [3] 末端位置
    gripper_state: float        # [0, 1] 夹爪状态

# 动作 (Action)
@dataclass
class Action:
    values: np.ndarray          # [7] 动作值
    action_type: str            # "delta_ee", "absolute_ee", "joint"
    
    # 分解
    position_delta: np.ndarray  # [3] 位置增量 (x, y, z)
    rotation_delta: np.ndarray  # [3] 旋转增量 (rx, ry, rz)
    gripper_action: float       # 夹爪动作
```

---

## 3. VLA模型详解

### 3.1 VLA发展历史

```
2022: RT-1 (Google)
  ↓ - Transformer架构用于机器人控制
2023: RT-2 (Google)
  ↓ - 视觉-语言预训练 + 动作微调
2024: OpenVLA (Stanford/Berkeley)
  ↓ - 开源7B参数VLA模型
2024+: 各种改进变体
```

### 3.2 OpenVLA架构

```
┌─────────────────────────────────────────────────────────────┐
│                       OpenVLA-7B                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐     │
│  │ Vision       │   │ Projection   │   │ LLM          │     │
│  │ Encoder      │──▶│ Layer        │──▶│ (Llama-2-7B) │     │
│  │ (SigLIP+     │   │              │   │              │     │
│  │  DinoV2)     │   │              │   │              │     │
│  └──────────────┘   └──────────────┘   └──────────────┘     │
│        ▲                                       │             │
│        │                                       ▼             │
│   ┌─────────┐                          ┌──────────────┐     │
│   │  Image  │                          │ Action Head  │     │
│   │ 224×224 │                          │ (256 bins)   │     │
│   └─────────┘                          └──────────────┘     │
│                                               │              │
│                                               ▼              │
│                                        7-DoF Actions        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### 关键组件：

**1. 视觉编码器 (Vision Encoder)**
- **SigLIP**: 用于语义理解
- **DinoV2**: 用于空间理解
- 融合两者特征，输出视觉tokens

**2. 语言模型 (LLM)**
- 基于 Llama-2-7B
- 处理融合后的视觉-语言tokens
- 自回归生成动作tokens

**3. 动作离散化 (Action Tokenization)**
```python
# 将连续动作离散化为256个bins
def encode_action(action: float, bins: int = 256) -> int:
    # action范围: [-1, 1]
    normalized = (action + 1) / 2  # [0, 1]
    token = int(normalized * (bins - 1))
    return np.clip(token, 0, bins - 1)

def decode_action(token: int, bins: int = 256) -> float:
    normalized = token / (bins - 1)  # [0, 1]
    action = normalized * 2 - 1  # [-1, 1]
    return action
```

### 3.3 RT-2模型

RT-2是Google的VLA模型，与OpenVLA的主要区别：

| 特性 | OpenVLA | RT-2 |
|------|---------|------|
| 基础模型 | Llama-2-7B | PaLI-X / PaLM-E |
| 视觉编码器 | SigLIP + DinoV2 | ViT |
| 动作表示 | 256 bins | 512 tokens (字符串) |
| 开源 | ✅ | ❌ |

### 3.4 动作空间设计

```python
# 7-DoF 动作空间
action = [
    dx,    # x方向位移增量 (±5cm)
    dy,    # y方向位移增量 (±5cm)
    dz,    # z方向位移增量 (±5cm)
    drx,   # 绕x轴旋转增量 (±15°)
    dry,   # 绕y轴旋转增量 (±15°)
    drz,   # 绕z轴旋转增量 (±15°)
    grip,  # 夹爪动作 (0: 闭合, 1: 张开)
]

# 归一化范围
ACTION_LOW = [-0.05, -0.05, -0.05, -0.25, -0.25, -0.25, 0]
ACTION_HIGH = [0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 1]
```

---

## 4. Isaac Sim使用指南

### 4.1 Isaac Sim简介

**NVIDIA Isaac Sim** 是基于Omniverse平台的机器人仿真器，提供：

- 物理真实的机器人仿真
- 高质量渲染
- 合成数据生成
- ROS/ROS2集成

### 4.2 核心概念

#### Stage和Prim

```python
# Stage是场景的最高级别容器
# Prim是场景中的基本对象

# 添加一个立方体
from omni.isaac.core.objects import DynamicCuboid

cube = DynamicCuboid(
    prim_path="/World/Cube",     # Prim路径
    name="red_cube",              # 名称
    position=np.array([0.5, 0, 0.1]),
    size=np.array([0.05, 0.05, 0.05]),
    color=np.array([1, 0, 0]),    # RGB
    mass=0.1
)
```

#### World和Scene

```python
from omni.isaac.core import World

# 创建仿真世界
world = World(
    stage_units_in_meters=1.0,   # 单位：米
    physics_dt=1/120,            # 物理步长
    rendering_dt=1/60,           # 渲染步长
)

# 添加默认地面
world.scene.add_default_ground_plane()

# 步进仿真
world.step(render=True)
```

### 4.3 Franka Panda机器人

```python
from omni.isaac.franka import Franka

# 添加Franka机器人
robot = Franka(
    prim_path="/World/Franka",
    name="franka",
    position=np.array([0, 0, 0]),
)

world.scene.add(robot)
world.reset()

# 获取关节状态
joint_positions = robot.get_joint_positions()  # [9] 7臂+2夹爪
joint_velocities = robot.get_joint_velocities()

# 设置关节位置
robot.set_joint_positions(target_positions)

# 控制夹爪
robot.gripper.open()   # 张开
robot.gripper.close()  # 闭合
```

#### Franka关节配置

| 关节索引 | 名称 | 范围 (rad) |
|---------|------|-----------|
| 0 | panda_joint1 | [-2.90, 2.90] |
| 1 | panda_joint2 | [-1.76, 1.76] |
| 2 | panda_joint3 | [-2.90, 2.90] |
| 3 | panda_joint4 | [-3.07, -0.07] |
| 4 | panda_joint5 | [-2.90, 2.90] |
| 5 | panda_joint6 | [-0.02, 3.75] |
| 6 | panda_joint7 | [-2.90, 2.90] |
| 7-8 | 夹爪 | [0, 0.04] |

### 4.4 传感器

#### RGB相机

```python
from omni.isaac.sensor import Camera

camera = Camera(
    prim_path="/World/Camera",
    position=np.array([0.5, 0, 0.5]),
    frequency=30,
    resolution=(224, 224),
)
camera.initialize()

# 获取图像
rgba = camera.get_rgba()  # [H, W, 4]
rgb = rgba[:, :, :3]

# 获取深度
depth = camera.get_depth()  # [H, W]
```

### 4.5 仿真循环

```python
# 基本仿真循环
world.reset()

for step in range(1000):
    # 1. 获取观测
    observation = get_observation()
    
    # 2. 计算动作
    action = policy(observation)
    
    # 3. 应用动作
    apply_action(action)
    
    # 4. 步进仿真
    world.step(render=True)
```

### 4.6 Isaac Lab (IsaacLab)

Isaac Lab是基于Isaac Sim的强化学习框架：

```python
# Isaac Lab环境示例
from omni.isaac.lab.envs import ManagerBasedRLEnv

class FrankaGraspEnv(ManagerBasedRLEnv):
    def _setup_scene(self):
        # 设置场景
        pass
    
    def _get_observations(self):
        # 返回观测
        pass
    
    def _compute_rewards(self):
        # 计算奖励
        pass
```

---

## 5. 运动控制方法

### 5.1 控制层次

```
┌─────────────────────────────────────────┐
│          任务级控制                      │
│  (VLA模型输出: "抓取红色方块")            │
└─────────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────┐
│          运动规划                        │
│  (轨迹生成: 起点 → 目标点)                │
└─────────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────┐
│          关节级控制                      │
│  (PD控制、阻抗控制)                      │
└─────────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────┐
│          执行器                          │
│  (电机驱动)                              │
└─────────────────────────────────────────┘
```

### 5.2 PD控制器

**基本原理**：根据位置误差和速度误差计算控制力矩

```python
class PDController:
    def __init__(self, kp: np.ndarray, kd: np.ndarray):
        self.kp = kp  # 比例增益 [7]
        self.kd = kd  # 微分增益 [7]
    
    def compute(
        self,
        current_pos: np.ndarray,
        target_pos: np.ndarray,
        current_vel: np.ndarray,
        target_vel: np.ndarray = None
    ) -> np.ndarray:
        """
        τ = Kp * (q_d - q) + Kd * (q̇_d - q̇)
        """
        if target_vel is None:
            target_vel = np.zeros_like(current_vel)
        
        pos_error = target_pos - current_pos
        vel_error = target_vel - current_vel
        
        torque = self.kp * pos_error + self.kd * vel_error
        return torque
```

**增益调节指南**：

| 关节 | Kp建议 | Kd建议 | 说明 |
|------|--------|--------|------|
| 1-3 | 600 | 50 | 肩部关节，需要较大力矩 |
| 4 | 600 | 50 | 肘部关节 |
| 5-6 | 250 | 30 | 腕部关节，惯量较小 |
| 7 | 50 | 15 | 末端关节 |

### 5.3 阻抗控制

**原理**：将机器人末端建模为弹簧-阻尼系统

```python
class ImpedanceController:
    """
    末端力 = K * (x_d - x) + D * (ẋ_d - ẋ)
    
    K: 刚度矩阵 (6x6)
    D: 阻尼矩阵 (6x6)
    """
    
    def __init__(self, stiffness: np.ndarray, damping: np.ndarray):
        # stiffness: [Kx, Ky, Kz, Krx, Kry, Krz]
        self.K = np.diag(stiffness)
        self.D = np.diag(damping)
    
    def compute(
        self,
        current_position: np.ndarray,
        target_position: np.ndarray,
        current_velocity: np.ndarray
    ) -> np.ndarray:
        """计算笛卡尔空间力"""
        pos_error = target_position - current_position
        
        force = self.K @ pos_error - self.D @ current_velocity
        return force
```

**应用场景**：
- 接触任务（擦拭、插入）
- 安全人机交互
- 力控制抓取

### 5.4 轨迹规划

#### 线性插值

```python
def linear_interpolate(start, end, num_points):
    """简单线性插值"""
    t = np.linspace(0, 1, num_points)
    trajectory = []
    for ti in t:
        point = start + ti * (end - start)
        trajectory.append(point)
    return trajectory
```

#### 最小急动度轨迹

```python
def min_jerk_trajectory(start, end, duration, dt):
    """
    五次多项式轨迹，最小化加加速度(jerk)
    
    s(t) = 10(t/T)³ - 15(t/T)⁴ + 6(t/T)⁵
    """
    t = np.arange(0, duration, dt)
    T = duration
    
    # 归一化时间
    tau = t / T
    
    # 五次多项式
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    
    # 速度 (一阶导数)
    s_dot = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T
    
    # 加速度 (二阶导数)
    s_ddot = (60 * tau - 180 * tau**2 + 120 * tau**3) / T**2
    
    trajectory = []
    for i in range(len(t)):
        point = TrajectoryPoint(
            position=start + s[i] * (end - start),
            velocity=s_dot[i] * (end - start),
            acceleration=s_ddot[i] * (end - start),
            time=t[i]
        )
        trajectory.append(point)
    
    return trajectory
```

**特点**：
- 平滑的速度曲线
- 零起始和结束速度/加速度
- 减少机械振动

### 5.5 逆运动学 (IK)

**问题**：给定末端目标位姿，求关节角度

```python
def jacobian_ik(
    current_joints: np.ndarray,
    target_position: np.ndarray,
    jacobian_func: Callable,
    max_iterations: int = 100,
    tolerance: float = 0.001
) -> np.ndarray:
    """
    基于雅可比矩阵的迭代IK
    
    Δq = J⁺ * Δx
    """
    joints = current_joints.copy()
    
    for iteration in range(max_iterations):
        # 计算当前末端位置 (正运动学)
        current_pos = forward_kinematics(joints)
        
        # 位置误差
        error = target_position - current_pos
        
        if np.linalg.norm(error) < tolerance:
            break
        
        # 计算雅可比矩阵
        J = jacobian_func(joints)
        
        # 阻尼最小二乘 (防止奇异)
        damping = 0.01
        J_pinv = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(3))
        
        # 更新关节
        delta_joints = J_pinv @ error
        joints = joints + 0.5 * delta_joints  # 步长因子
    
    return joints
```

---

## 6. 代码结构详解

### 6.1 目录结构

```
VLA/
├── vla_platform/                    # 核心平台代码
│   ├── __init__.py
│   │
│   ├── core/                        # 核心模块
│   │   ├── config.py               # 配置管理
│   │   └── base_interfaces.py      # 抽象接口定义
│   │
│   ├── remote/                      # 远程通信
│   │   ├── rest_client.py          # REST API客户端
│   │   ├── grpc_client.py          # gRPC客户端
│   │   └── protos/                 # Protocol Buffers定义
│   │
│   ├── models/                      # VLA模型客户端
│   │   ├── openvla_client.py       # OpenVLA封装
│   │   ├── rt2_client.py           # RT-2风格客户端
│   │   └── action_tokenizer.py     # 动作离散化
│   │
│   ├── simulation/                  # Isaac Sim仿真
│   │   ├── sim_manager.py          # 仿真管理器
│   │   ├── envs/
│   │   │   └── franka_env.py       # Franka抓取环境
│   │   └── sensors/
│   │       └── sensor_manager.py   # 传感器管理
│   │
│   ├── control/                     # 运动控制
│   │   ├── motion_controller.py    # PD控制、IK
│   │   ├── trajectory_planner.py   # 轨迹规划
│   │   └── impedance_controller.py # 阻抗控制
│   │
│   └── training/                    # 训练模块
│       ├── openvla_model.py        # OpenVLA本地复现
│       ├── rl/
│       │   └── ppo_trainer.py      # PPO训练器
│       └── data/
│           └── dataset.py          # 数据收集和Dataset
│
├── server/                          # 远程服务器
│   ├── server_deploy.py            # 模型部署脚本
│   ├── requirements_server.txt     # 服务器依赖
│   └── requirements_training.txt   # 训练依赖
│
├── scripts/                         # 脚本
│   ├── train_openvla.py            # 训练脚本
│   └── collect_data.py             # 数据收集
│
├── demos/                           # 演示
│   ├── demo_vla_grasp.py           # VLA抓取演示
│   └── demo_motion_control.py      # 运动控制演示
│
├── configs/                         # 配置文件
│   ├── default.yaml                # 默认配置
│   └── train_config.yaml           # 训练配置
│
├── tests/                           # 测试
│   └── test_vla_platform.py
│
├── docs/                            # 文档
├── requirements.txt                 # 本地依赖
└── README.md
```

### 6.2 核心类关系

```
                    ┌─────────────────┐
                    │ PlatformConfig  │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ RemoteServerConfig│ │ SimulationConfig│ │ ControlConfig   │
└─────────────────┘ └─────────────────┘ └─────────────────┘
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ RESTClient      │ │ SimulationManager│ │ MotionController │
│ GRPCClient      │ │ FrankaGraspEnv  │ │ TrajectoryPlanner│
└─────────────────┘ └─────────────────┘ └─────────────────┘
         │                   ▲                   │
         ▼                   │                   ▼
┌─────────────────┐          │          ┌─────────────────┐
│ OpenVLAClient   │──────────┘          │ ImpedanceControl│
│ RT2Client       │                     └─────────────────┘
└─────────────────┘
```

### 6.3 关键方法说明

#### 配置加载

```python
# 从YAML加载配置
from vla_platform.core.config import PlatformConfig

config = PlatformConfig.from_yaml("configs/default.yaml")

# 访问配置
server_host = config.remote.host
action_dim = config.model.action_dim
physics_dt = config.simulation.physics_dt
```

#### VLA推理

```python
from vla_platform.models import OpenVLAClient

# 创建客户端
client = OpenVLAClient(config.remote, config.model)
client.connect()

# 推理
action = client.predict(observation, "pick up the red block")

# 应用动作
env.apply_action(action)
```

#### 运动控制

```python
from vla_platform.control import MotionController, TrajectoryPlanner

# 创建控制器
controller = MotionController(config.control)

# 生成轨迹
planner = TrajectoryPlanner(max_velocity, max_acceleration)
trajectory = planner.interpolate(waypoints, dt=0.01)

# 跟踪轨迹
for point in trajectory:
    torque = controller.compute_joint_control(
        current_pos, current_vel, point.position
    )
    robot.apply_torque(torque)
```

---

## 7. 训练流程

### 7.1 监督学习微调 (SFT)

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ 演示数据    │ ──▶ │ 预处理      │ ──▶ │ 模型微调    │
│ (Image,     │     │ (Tokenize)  │     │ (LoRA)      │
│  Action)    │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
```

```python
# SFT训练
python scripts/train_openvla.py \
    --mode sft \
    --use_lora \
    --lora_r 16 \
    --data_dir data/demos \
    --batch_size 4 \
    --learning_rate 1e-5
```

### 7.2 强化学习微调 (RL)

```
┌───────────────────────────────────────────────────────────┐
│                     PPO训练循环                            │
│                                                           │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐ │
│  │ 收集    │ ──▶│ 计算    │ ──▶│ 更新    │ ──▶│ 评估    │ │
│  │ Rollout │    │ 优势    │    │ 策略    │    │         │ │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘ │
│       ▲                                            │      │
│       └────────────────────────────────────────────┘      │
└───────────────────────────────────────────────────────────┘
```

**PPO算法关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| clip_epsilon | 0.2 | PPO裁剪范围 |
| gamma | 0.99 | 折扣因子 |
| gae_lambda | 0.95 | GAE参数 |
| ppo_epochs | 4 | 每次更新的epoch数 |
| batch_size | 32 | 批量大小 |

### 7.3 LoRA高效微调

```python
# LoRA将权重分解为低秩矩阵
# W' = W + ΔW = W + A·B
# 其中 A: [d, r], B: [r, k], r << min(d, k)

from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                    # 秩
    lora_alpha=32,           # 缩放因子
    target_modules=[         # 目标模块
        "q_proj", "v_proj", 
        "k_proj", "o_proj"
    ],
    lora_dropout=0.05,
)

model = get_peft_model(base_model, lora_config)

# 只训练LoRA参数 (约2-4% of原参数)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

**LoRA优势**：
- 显存节省: ~70%
- 训练速度: 快2-3倍
- 不降低性能 (通常)

---

## 8. 部署指南

### 8.1 本地环境设置

```bash
# 1. 激活Isaac Lab环境
conda activate isaaclab

# 2. 安装依赖
cd /home/vincent/Desktop/code/VLA
pip install -r requirements.txt

# 3. 验证安装
python -c "from vla_platform.training import OpenVLAModel; print('OK')"
```

### 8.2 服务器训练

```bash
# 1. 复制代码到服务器
scp -r VLA/ user@server:/path/to/

# 2. 安装训练依赖
pip install -r server/requirements_training.txt

# 3. 启动训练
python scripts/train_openvla.py \
    --mode rl \
    --use_lora \
    --headless \
    --output_dir checkpoints/
```

### 8.3 模型部署

```bash
# 在GPU服务器上启动推理服务
python server/server_deploy.py \
    --model openvla/openvla-7b \
    --quantization int4 \
    --port 8000
```

### 8.4 Isaac Sim运行

```bash
# 通过Isaac Sim启动器运行
./isaac-sim.sh --/isaac/startup/pip_packages="requests,h5py"

# 或在Python环境中
python demos/demo_vla_grasp.py --server http://your-server:8000
```

---

## 附录

### A. 常见问题

**Q: Isaac Sim显示为不可用？**
A: 需要通过Isaac Sim启动器运行，或设置正确的环境变量。

**Q: GPU显存不足？**
A: 使用INT4量化（4GB足够）或远程服务器。

**Q: 训练不收敛？**
A: 检查学习率、增加数据量、调整奖励函数。

### B. 参考资料

1. [OpenVLA Paper](https://openvla.github.io/)
2. [RT-2 Paper](https://arxiv.org/abs/2307.15818)
3. [Isaac Sim Documentation](https://docs.omniverse.nvidia.com/isaacsim/)
4. [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
5. [PEFT/LoRA](https://github.com/huggingface/peft)

### C. 术语表

| 术语 | 解释 |
|------|------|
| VLA | Vision-Language-Action模型 |
| IK | 逆运动学 (Inverse Kinematics) |
| FK | 正运动学 (Forward Kinematics) |
| EE | 末端执行器 (End-Effector) |
| DoF | 自由度 (Degrees of Freedom) |
| SFT | 监督微调 (Supervised Fine-Tuning) |
| RL | 强化学习 (Reinforcement Learning) |
| PPO | 近端策略优化 (Proximal Policy Optimization) |
| LoRA | 低秩适应 (Low-Rank Adaptation) |
| GAE | 广义优势估计 (Generalized Advantage Estimation) |
