# 远程服务器VLA服务部署与启动指南

本指南介绍如何通过SSH连接远程GPU服务器，并在远程服务器上部署和启动VLA（Vision-Language-Action）服务。

## 📋 前提条件

### 远程服务器要求

- **操作系统**: Ubuntu 20.04/22.04 LTS
- **GPU**: NVIDIA GPU (显存16GB+ 推荐，如RTX 3090/4090/A100)
- **CUDA**: 11.8 或 12.x
- **Python**: 3.10 或 3.11
- **网络**: 开放的端口用于服务通信（如8000端口）

### 本地环境要求

- SSH客户端（Linux/Mac自带，Windows可使用PowerShell或Git Bash）
- 了解服务器的SSH连接信息（IP、端口、用户名）

---

## 🔑 第一步：SSH连接远程服务器

### 1.1 基本连接

使用提供的SSH指令连接服务器：

```bash
ssh root@sh01-ssh.gpuhome.cc -p 30046
```

**参数说明**:
- `root`: 用户名
- `sh01-ssh.gpuhome.cc`: 服务器地址
- `-p 30046`: SSH端口（非默认22端口）

### 1.2 使用SSH密钥（推荐）

如果服务器配置了SSH密钥认证：

```bash
# 使用私钥连接
ssh -i ~/.ssh/your_key.pem root@sh01-ssh.gpuhome.cc -p 30046

# 或者将密钥添加到ssh-agent
ssh-add ~/.ssh/your_key.pem
ssh root@sh01-ssh.gpuhome.cc -p 30046
```

### 1.3 保持连接（防止断开）

```bash
# 使用tmux或screen保持会话
ssh root@sh01-ssh.gpuhome.cc -p 30046 -t "tmux new -s vla_session"

# 重新连接到会话
ssh root@sh01-ssh.gpuhome.cc -p 30046 -t "tmux attach -t vla_session"
```

---

## 📦 第二步：环境准备

### 2.1 检查GPU状态

连接成功后，首先检查GPU状态：

```bash
# 检查NVIDIA驱动和GPU
nvidia-smi

# 预期输出示例：
# +---------------------------------------------------------------------------------------+
# | NVIDIA-SMI 535.104.05             Driver Version: 535.104.05   CUDA Version: 12.2     |
# |-----------------------------------------+----------------------+----------------------+
# | GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
# |                                         |                      |               MIG M. |
# |=========================================+======================+======================|
# |   0  NVIDIA GeForce RTX 4090        Off | 00000000:01:00.0 Off |                  Off |
# |  0%   35C    P8              15W / 450W |    100MiB / 24564MiB |      0%      Default |
# +-----------------------------------------+----------------------+----------------------+
```

### 2.2 检查Python环境

```bash
# 检查Python版本
python3 --version
# 应该显示 Python 3.10.x 或 3.11.x

# 检查pip
pip3 --version
```

### 2.3 克隆VLA项目（如未克隆）

```bash
# 进入工作目录
cd ~

# 克隆项目（如果还没有）
git clone https://github.com/your-repo/VLA.git
# 或者上传本地代码

# 进入项目目录
cd VLA
```

### 2.4 安装Python依赖

```bash
# 安装项目依赖
pip install -r requirements.txt

# 安装远程服务器依赖
pip install -r server/requirements_server.txt
```

---

## 🚀 第三步：启动VLA推理服务

### 3.1 方式一：启动OpenVLA推理服务器

#### 基础启动

```bash
cd ~/VLA/server

# 启动VLA推理服务（使用OpenVLA-7B模型）
python server_deploy.py --model openvla/openvla-7b --port 8000
```

#### 使用量化减少显存（推荐）

如果GPU显存有限（如12GB），使用INT8量化：

```bash
python server_deploy.py \
    --model openvla/openvla-7b \
    --port 8000 \
    --quantization int8
```

#### 后台运行（使用nohup）

```bash
# 后台运行并将日志保存到文件
nohup python server_deploy.py \
    --model openvla/openvla-7b \
    --port 8000 \
    > vla_server.log 2>&1 &

# 查看日志
tail -f vla_server.log

# 查看进程
ps aux | grep server_deploy
```

#### 使用tmux后台运行（推荐）

```bash
# 创建tmux会话
tmux new -s vla_server

# 在tmux中启动服务
cd ~/VLA/server
python server_deploy.py --model openvla/openvla-7b --port 8000

# 分离tmux会话（按 Ctrl+B，然后按 D）
# 重新连接：tmux attach -t vla_server
```

### 3.2 方式二：启动Isaac Sim仿真（如果远程服务器支持）

如果远程服务器有图形界面或支持headless模式：

```bash
# 检查Isaac Sim是否安装
ls ~/isaac-sim/

# 启动仿真（headless模式）
cd ~/isaac-sim
./python.sh ~/VLA/demos/demo_vla_grasp.py \
    --server http://localhost:8000 \
    --headless
```

---

## 🌐 第四步：端口转发（本地访问远程服务）

如果远程服务器的端口没有开放给公网，可以使用SSH端口转发：

### 4.1 在本地终端执行（不要关闭）

```bash
# 将远程服务器的8000端口转发到本地的8000端口
ssh -L 8000:localhost:8000 root@sh01-ssh.gpuhome.cc -p 30046
```

**参数说明**:
- `-L 8000:localhost:8000`: 本地端口:远程地址:远程端口
- 这样访问本地的 `http://localhost:8000` 就相当于访问远程服务器的8000端口

### 4.2 测试服务

在**本地**打开新的终端窗口测试：

```bash
# 测试服务是否运行
curl http://localhost:8000/health

# 预期输出：
# {"status": "healthy", "model": "openvla/openvla-7b"}
```

---

## 🔄 第五步：本地Isaac Sim连接远程服务

### 5.1 修改本地配置

编辑本地VLA项目的配置，指向远程服务器：

```python
# 在本地编辑 vla_platform/core/config.py
# 或者通过命令行参数指定

# 如果使用端口转发（推荐）
python demo_vla_grasp.py --server http://localhost:8000

# 如果远程端口开放给公网
python demo_vla_grasp.py --server http://sh01-ssh.gpuhome.cc:8000
```

### 5.2 启动本地仿真

在**本地**运行Isaac Sim：

```bash
cd /home/vincent/isaac-sim
./python.sh /home/vincent/Desktop/code/VLA/demos/demo_vla_grasp.py \
    --server http://localhost:8000
```

---

## 📊 第六步：监控和管理

### 6.1 监控GPU使用情况

在远程服务器上：

```bash
# 实时监控GPU（每1秒刷新）
watch -n 1 nvidia-smi

# 或者使用nvitop（更美观）
pip install nvitop
nvitop
```

### 6.2 查看服务日志

```bash
# 如果使用nohup
tail -f ~/VLA/server/vla_server.log

# 如果使用tmux
tmux attach -t vla_server
```

### 6.3 停止服务

```bash
# 查找Python进程
ps aux | grep server_deploy

# 杀掉进程（替换<PID>为实际进程ID）
kill <PID>

# 或者强制停止
kill -9 <PID>

# 如果使用tmux
tmux kill-session -t vla_server
```

---

## 🔧 常见问题

### Q1: SSH连接超时断开

**解决**: 配置SSH保持连接

```bash
# 编辑本地 ~/.ssh/config
Host gpu-server
    HostName sh01-ssh.gpuhome.cc
    Port 30046
    User root
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

### Q2: 端口被占用

```bash
# 查找占用8000端口的进程
lsof -i :8000

# 杀掉进程
kill -9 <PID>

# 或者使用其他端口
python server_deploy.py --port 8080
```

### Q3: 显存不足

```bash
# 使用更小的模型或量化
python server_deploy.py --model openvla/openvla-7b --quantization int8

# 或者使用4-bit量化
python server_deploy.py --model openvla/openvla-7b --quantization int4
```

### Q4: 模型下载慢

```bash
# 设置HuggingFace镜像（在国内服务器）
export HF_ENDPOINT=https://hf-mirror.com

# 然后启动服务
python server_deploy.py --model openvla/openvla-7b
```

### Q5: 连接被拒绝

检查防火墙和端口开放情况：

```bash
# 在远程服务器上检查端口监听
netstat -tlnp | grep 8000

# 或者
ss -tlnp | grep 8000

# 检查防火墙
ufw status
```

---

## 📝 完整操作示例

### 场景：在远程服务器启动VLA服务，本地连接

**步骤1**: 连接远程服务器
```bash
ssh root@sh01-ssh.gpuhome.cc -p 30046
```

**步骤2**: 在远程服务器上启动服务（使用tmux）
```bash
tmux new -s vla_server
cd ~/VLA/server
python server_deploy.py --model openvla/openvla-7b --port 8000 --quantization int8
# 按 Ctrl+B, 然后 D 分离会话
```

**步骤3**: 在本地设置端口转发
```bash
ssh -L 8000:localhost:8000 root@sh01-ssh.gpuhome.cc -p 30046
```

**步骤4**: 在本地启动Isaac Sim仿真
```bash
cd /home/vincent/isaac-sim
./python.sh /home/vincent/Desktop/code/VLA/demos/demo_vla_grasp.py --server http://localhost:8000
```

---

## 🔐 安全建议

1. **不要使用root用户**: 创建普通用户运行服务
   ```bash
   useradd -m vlauser
   usermod -aG sudo vlauser
   su - vlauser
   ```

2. **使用防火墙限制端口**: 只开放必要的端口
   ```bash
   ufw allow from <your-ip> to any port 8000
   ```

3. **使用SSH密钥**: 禁用密码登录
   ```bash
   # 编辑 /etc/ssh/sshd_config
   PasswordAuthentication no
   PubkeyAuthentication yes
   ```

4. **使用HTTPS**: 生产环境应配置SSL证书

---

## 📞 技术支持

如果遇到问题，请检查：

1. 远程服务器GPU状态：`nvidia-smi`
2. 服务日志：`tail -f vla_server.log`
3. 端口监听：`netstat -tlnp | grep 8000`
4. 防火墙设置：`ufw status`

---

**最后更新**: 2026-02-14
**适用版本**: Isaac Sim 5.1.0, VLA Platform v1.0
