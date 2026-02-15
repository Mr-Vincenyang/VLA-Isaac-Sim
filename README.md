# VLA Isaac Sim 仿真复现平台

基于Isaac Sim的VLA (Vision-Language-Action) 模型仿真复现平台，支持远程推理和运动控制。

## 项目特点

- 🤖 **VLA模型复现**: 支持OpenVLA和RT-2风格模型的远程推理
- 🎮 **Isaac Sim集成**: 完整的Franka Panda机械臂仿真环境
- 🌐 **分布式架构**: 本地仿真 + 远程GPU推理
- ⚙️ **运动控制**: PD控制、轨迹规划、阻抗控制

## 系统架构

```
本地 (Isaac Sim)                 远程服务器 (GPU)
┌────────────────┐              ┌────────────────┐
│  仿真环境       │   REST/gRPC  │  VLA模型       │
│  传感器数据  ──────────────────▶   OpenVLA     │
│  运动控制   ◀──────────────────  Action预测    │
└────────────────┘              └────────────────┘
```

## 环境要求

### 系统要求

- **操作系统**: Ubuntu 20.04/22.04 LTS
- **GPU**: NVIDIA GPU (RTX 3070或更高，显存8GB+)
- **内存**: 32GB RAM 推荐
- **Python**: 3.11 (与Isaac Sim内置Python版本一致)
- **Isaac Sim**: 5.1.0

### Isaac Sim 安装验证

在开始使用本项目前，请确保您的Isaac Sim 5.1.0已正确安装。

#### 1. 检查Isaac Sim版本

```bash
# 查看安装的Isaac Sim版本
cat /home/vincent/isaac-sim/VERSION
# 预期输出: 5.1.0-rc.19+release.26219.9c81211b.gl
```

#### 2. 运行安装验证脚本

我们提供了验证脚本来检查Isaac Sim是否安装正确：

```bash
# 方法1: 使用系统Python检查安装状态（仅检查文件）
python3 check_isaac_sim.py

# 方法2: 使用Isaac Sim Python运行完整功能测试（推荐）
cd /home/vincent/isaac-sim
./python.sh /home/vincent/Desktop/code/VLA/verify_isaac_sim.py
```

#### 3. 验证测试内容

验证脚本将检查以下内容：

| 检查项目 | 说明 |
|---------|------|
| SimulationApp | Isaac Sim应用启动 |
| World模块 | 仿真世界创建与管理 |
| Franka机器人 | 机械臂模型加载与控制 |
| Camera相机 | RGB/深度相机传感器 |
| DynamicCuboid | 动态物体创建 |

**预期输出**:
```
============================================================
Isaac Sim 5.1.0 基础验证
============================================================

[1/5] 检查 SimulationApp...
✓ SimulationApp 可导入

[2/5] 启动 SimulationApp...
✓ SimulationApp 启动成功

[3/5] 检查核心模块...
✓ isaacsim.core.api.World
✓ isaacsim.sensors.camera.Camera
✓ isaacsim.robot.manipulators.examples.franka.Franka
✓ isaacsim.core.api.objects.DynamicCuboid

模块检查: 4/4 通过

[4/5] 测试 World 创建...
✓ World 创建成功

[5/5] 测试 Franka 机器人...
✓ Franka 创建成功 (DOF: 9)

🎉 所有测试通过！Isaac Sim 5.1.0 工作正常
```

#### 4. 常见问题

**问题1**: 模块导入失败 (`No module named 'isaacsim'`)

**解决**: 必须使用Isaac Sim自带的Python运行
```bash
# 错误
python3 your_script.py

# 正确
cd /home/vincent/isaac-sim
./python.sh your_script.py
```

**问题2**: 无法启动SimulationApp

**解决**: 检查GPU驱动和CUDA
```bash
# 检查NVIDIA驱动
nvidia-smi

# 检查CUDA
nvcc --version
```

## 快速开始

### 1. 安装依赖

**本地环境 (Isaac Sim conda)**:
```bash
cd /home/vincent/Desktop/code/VLA
pip install -r requirements.txt
```

**远程服务器**:
```bash
cd server
pip install -r requirements_server.txt
```

### 2. 启动远程服务器

在GPU服务器上:
```bash
python server/server_deploy.py --model openvla/openvla-7b --port 8000
```

可选量化以减少显存:
```bash
python server/server_deploy.py --model openvla/openvla-7b --quantization int8
```

### 3. 运行演示

⚠️ **重要**: 所有演示脚本必须使用Isaac Sim自带的Python运行！

**VLA抓取演示**:
```bash
# 切换到Isaac Sim目录并运行
cd /home/vincent/isaac-sim
./python.sh /home/vincent/Desktop/code/VLA/demos/demo_vla_grasp.py --server http://your-server:8000
```

**运动控制演示**:
```bash
cd /home/vincent/isaac-sim
./python.sh /home/vincent/Desktop/code/VLA/demos/demo_motion_control.py --demo all
```

**使用Conda环境（可选）**:
```bash
# 创建Python 3.11环境
conda create -n isaacsim python=3.11 -y
conda activate isaacsim

# 设置Isaac Sim环境
cd /home/vincent/isaac-sim
source setup_conda_env.sh

# 现在可以直接使用python命令
python /home/vincent/Desktop/code/VLA/demos/demo_vla_grasp.py --server http://your-server:8000
```

## 项目结构

```
VLA/
├── vla_platform/           # 核心包
│   ├── core/               # 配置和接口定义
│   ├── remote/             # 远程通信 (REST/gRPC)
│   ├── models/             # VLA模型客户端
│   ├── simulation/         # Isaac Sim仿真
│   └── control/            # 运动控制器
├── server/                 # 远程服务器脚本
├── demos/                  # 演示脚本
├── configs/                # 配置文件
└── tests/                  # 单元测试
```

## Isaac Sim 5.1.0 API 兼容性

本项目已针对 **Isaac Sim 5.1.0** 进行优化和修复，使用了新的 `isaacsim` 命名空间。

### 主要API变更

| 旧API (Isaac Sim 4.x) | 新API (Isaac Sim 5.1.0) | 说明 |
|---------------------|-----------------------|------|
| `omni.isaac.core.World` | `isaacsim.core.api.World` | 仿真世界管理 |
| `omni.isaac.sensor.Camera` | `isaacsim.sensors.camera.Camera` | 相机传感器 |
| `omni.isaac.franka.Franka` | `isaacsim.robot.manipulators.examples.franka.Franka` | Franka机器人 |
| `omni.isaac.core.objects.DynamicCuboid` | `isaacsim.core.api.objects.DynamicCuboid` | 动态立方体 |

### 向后兼容

代码中包含了新旧命名空间的双重导入支持：

```python
# 尝试新的 isaacsim 命名空间 (Isaac Sim 5.x)
try:
    from isaacsim.core.api import World
    from isaacsim.sensors.camera import Camera
    from isaacsim.robot.manipulators.examples.franka import Franka
    ISAAC_SIM_AVAILABLE = True
except ImportError:
    pass

# 尝试旧的 omni.isaac 命名空间 (兼容性)
if not ISAAC_SIM_AVAILABLE:
    try:
        from omni.isaac.core import World
        from omni.isaac.sensor import Camera
        from omni.isaac.franka import Franka
        ISAAC_SIM_AVAILABLE = True
    except ImportError:
        pass
```

## 核心模块

### VLA模型客户端

```python
from vla_platform.models import OpenVLAClient
from vla_platform.core import RemoteServerConfig

# 连接远程服务器
config = RemoteServerConfig(host="your-server", port=8000)
client = OpenVLAClient(config)
client.connect()

# 推理
action = client.predict(observation, "pick up the red block")
```

### 运动控制

```python
from vla_platform.control import MotionController, TrajectoryPlanner

# 轨迹规划
planner = TrajectoryPlanner(max_velocity, max_acceleration)
trajectory = planner.interpolate(waypoints, dt=0.01)

# 阻抗控制
from vla_platform.control import ImpedanceController
controller = ImpedanceController()
wrench = controller.compute(current_pose, target_pose, ...)
```

## 测试脚本

本项目包含以下测试脚本，用于验证环境和功能：

### 1. Isaac Sim 安装验证

**文件**: `check_isaac_sim.py`

用途：检查Isaac Sim是否已正确安装，不需要启动GUI

```bash
python3 check_isaac_sim.py
```

### 2. Isaac Sim 功能测试

**文件**: `verify_isaac_sim.py`

用途：运行完整的Isaac Sim功能测试（包括World创建、Franka机器人、相机传感器）

```bash
cd /home/vincent/isaac-sim
./python.sh /home/vincent/Desktop/code/VLA/verify_isaac_sim.py
```

### 3. 项目单元测试

```bash
cd /home/vincent/isaac-sim
./python.sh -m pytest /home/vincent/Desktop/code/VLA/tests/ -v
```

## 面试项目亮点

1. **分布式系统设计**: 解决本地算力不足问题
2. **VLA模型理解**: 深入OpenVLA/RT-2架构
3. **机器人学知识**: 阻抗控制、轨迹规划
4. **工程能力**: 模块化设计、接口抽象

## 已知问题和解决方案

### 1. VLA远程服务器500错误

**问题描述**: 远程服务器返回 `500 Internal Server Error`，错误信息为 `Input type (float) and bias type (c10::BFloat16) should be the same`

**原因**: 模型默认使用 `bfloat16` 格式，但输入数据为 `float32`，导致数据类型不匹配

**解决方案**: 
- ✅ 已在 `server/server_deploy.py` 中修复，将模型加载时的 `torch_dtype` 从 `torch.bfloat16` 改为 `torch.float32`
- 如需使用更高精度，可手动修改为 `torch.float16` 或 `torch.bfloat16`，并确保输入数据类型匹配

**更新步骤**:
```bash
# 1. 在本地提交修复
git add server/server_deploy.py
git commit -m "Fix: Change model dtype from bfloat16 to float32 to avoid type mismatch"
git push

# 2. 在远程服务器拉取更新
cd ~/VLA
git pull

# 3. 重启VLA服务
pkill -f server_deploy
cd server
python server_deploy.py --model openvla/openvla-7b --port 8000
```

### 2. 没有画面显示

**问题描述**: 运行demo时没有GUI窗口显示

**原因**: 使用了 `--headless` 参数

**解决方案**: 移除 `--headless` 参数，或改为 `--no-window`
```bash
# 显示GUI
./python.sh demos/demo_vla_grasp.py --server http://localhost:8080

# 无头模式（不显示GUI，用于服务器）
./python.sh demos/demo_vla_grasp.py --server http://localhost:8080 --headless
```

### 3. 相机深度图警告

**问题描述**: 日志中出现 `Annotator 'distance_to_image_plane' not attached`

**原因**: 深度传感器annotator未正确配置

**解决方案**: 这是警告信息，不影响基本功能。如需深度图，确保在CameraConfig中启用 `enable_depth=True`

## 更新日志

### 2026-02-14
- ✅ 修复Isaac Sim 5.1.0 API兼容性问题
- ✅ 修复Franka导入路径 (`isaacsim.robot.manipulators.examples.franka`)
- ✅ 修复Camera传感器初始化问题
- ✅ 修复远程VLA服务器数据类型错误
- ✅ 添加完整的SSH远程部署指南
- ✅ 添加Isaac Sim安装验证脚本

## License

MIT
