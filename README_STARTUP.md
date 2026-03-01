# VLA 一键启动脚本

本项目提供了一键启动脚本，方便快速运行 VLA 演示。

## 快速开始

### 1. 启动 VLA 服务器 (GPU 服务器)

```bash
# 默认配置启动 (openvla/openvla-7b, port 8000)
./start_server.sh

# 自定义配置
./start_server.sh --model openvla/openvla-7b --port 8000 --quantization int8

# 使用环境变量
export VLA_MODEL=openvla/openvla-7b
export VLA_PORT=8000
./start_server.sh
```

### 2. 启动本地演示 (Isaac Sim)

```bash
# 运动控制演示 (自动录制视频)
./start_local.sh --demo motion --record

# 抓取演示 (连接 VLA 服务器)
./start_local.sh --demo grasp --server http://localhost:8000 --record

# 无头模式运行 (不显示 GUI，纯视频录制)
./start_local.sh --demo grasp --record --headless

# 自定义 Isaac Sim 路径
./start_local.sh --demo motion --isaac-path /path/to/isaac-sim
```

## 脚本说明

### start_server.sh

启动 VLA 推理服务器。

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | VLA 模型名称 | openvla/openvla-7b |
| `--port` | 服务器端口 | 8000 |
| `--quantization` | 量化选项 (int4/int8) | 无 |

### start_local.sh

启动本地 Isaac Sim 演示。

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--demo` | 演示类型 (motion/grasp) | (必填) |
| `--record` | 启用视频录制 | false |
| `--server` | VLA 服务器地址 | 无 |
| `--episodes` | 抓取演示回合数 | 3 |
| `--headless` | 无头模式运行 | false |
| `--isaac-path` | Isaac Sim 安装路径 | ~/isaac-sim |

## 输出文件

视频文件将保存到 `output/` 目录：

- `motion_demo_YYYYMMDD_HHMMSS.mp4` - 运动控制演示
- `grasp_demo_YYYYMMDD_HHMMSS.mp4` - 抓取演示

## 使用流程

### 完整演示流程 (需要 VLA 服务器)

1. **启动 VLA 服务器** (在 GPU 机器上)
   ```bash
   ./start_server.sh
   ```

2. **启动本地演示** (在 Isaac Sim 机器上)
   ```bash
   ./start_local.sh --demo grasp --server http://<server-ip>:8000 --record
   ```

### 本地演示流程 (不需要 VLA 服务器)

1. **运行运动控制演示**
   ```bash
   ./start_local.sh --demo motion --record
   ```

2. **运行抓取演示 (启发式控制)**
   ```bash
   ./start_local.sh --demo grasp --record
   ```

## 注意事项

- 确保 Isaac Sim 已正确安装
- VLA 服务器需要 GPU (建议 24GB+ 显存)
- 视频录制需要确保 `output/` 目录存在
- 无头模式 (`--headless`) 用于服务器环境，不显示 GUI
