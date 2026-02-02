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

**VLA抓取演示**:
```bash
# 在Isaac Sim Python环境中
python demos/demo_vla_grasp.py --server http://your-server:8000
```

**运动控制演示**:
```bash
python demos/demo_motion_control.py --demo all
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

## 面试项目亮点

1. **分布式系统设计**: 解决本地算力不足问题
2. **VLA模型理解**: 深入OpenVLA/RT-2架构
3. **机器人学知识**: 阻抗控制、轨迹规划
4. **工程能力**: 模块化设计、接口抽象

## License

MIT
