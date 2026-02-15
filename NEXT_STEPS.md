# VLA 仿真平台 - 下一步操作指南

## 概述
所有关键修复已完成提交。现在需要：
1. 修复CPU性能问题（解决卡顿）
2. 推送代码到远程仓库
3. 更新远程服务器
4. 运行测试

---

## 步骤 1: 修复CPU性能问题（关键）

Isaac Sim 日志显示 `CPU performance profile is set to powersave`，这会导致严重卡顿。

### 安装 cpufrequtils
```bash
sudo apt update
sudo apt install cpufrequtils -y
sudo systemctl enable cpufrequtils
```

### 设置为性能模式
```bash
echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils
sudo systemctl restart cpufrequtils
```

### 验证设置
```bash
# 检查当前CPU频率策略
cpufreq-info | grep "current policy"
# 应该显示 "performance"
```

---

## 步骤 2: 推送代码到远程仓库

在本地机器执行：

```bash
cd /home/vincent/Desktop/code/VLA

# 检查提交状态
git log --oneline -3

# 推送到远程（需要输入GitHub用户名和Token/密码）
git push origin main
```

**注意**: 如果推送失败需要认证，可以使用以下方式之一：

### 方式A: 使用GitHub Personal Access Token
```bash
# 在GitHub设置中生成Token后：
git remote set-url origin https://YOUR_TOKEN@github.com/YOUR_USERNAME/VLA.git
git push origin main
```

### 方式B: 使用SSH（如果已配置）
```bash
git remote set-url origin git@github.com:YOUR_USERNAME/VLA.git
git push origin main
```

---

## 步骤 3: 更新远程服务器

SSH到远程服务器后执行：

```bash
# 进入项目目录
cd ~/VLA

# 拉取最新代码
git pull origin main

# 停止旧的服务器进程
pkill -f server_deploy

# 等待进程完全停止
sleep 2

# 启动新的VLA服务器
cd server
python server_deploy.py --model openvla/openvla-7b --port 8000 --device cuda
```

**验证服务器运行**：
```bash
# 在另一个终端中检查
ps aux | grep server_deploy
curl http://localhost:8000/health
```

---

## 步骤 4: 本地运行测试

### 启动SSH隧道（如果尚未建立）
```bash
# 在本地机器（非远程服务器）执行
ssh -N -L 8080:localhost:8000 YOUR_REMOTE_USER@YOUR_REMOTE_IP
```

保持此终端运行，然后在另一个终端中：

### 运行VLA抓取演示
```bash
cd ~/isaac-sim
./python.sh /home/vincent/Desktop/code/VLA/demos/demo_vla_grasp.py --server http://localhost:8080
```

### 参数说明
- `--server http://localhost:8080`: VLA服务器地址（通过SSH隧道）
- `--episodes 3`: 运行3个episode（可选）
- `--headless`: 无头模式（不显示GUI，可选）

---

## 预期结果

### 正常输出示例
```
✓ SimulationApp started successfully
Setting up VLA Grasp Demo...
✓ Isaac Sim environment verified
Creating simulation manager...
Setting up camera...
Creating Franka environment...
Added Franka robot
Added table
Added 1 objects
Franka grasp environment setup complete
Camera added to world scene at /World/Camera/overhead
Initializing simulation...
Connecting to VLA server: http://localhost:8080
Connected to VLA server!
Viewport camera set to position [1.2 -0.8 0.9], looking at [0.4 0.0 0.1]
Setup complete!
Starting episode with instruction: 'pick up the red block'
...
```

### 警告信息（可忽略）
- `Annotator 'distance_to_image_plane' not attached`: 深度传感器警告，不影响RGB图像
- 各种deprecation警告：API兼容性警告，不影响功能
- `CPU performance profile is set to powersave`: **必须在步骤1修复**

---

## 故障排除

### 问题1: 服务器连接失败
```
Could not connect to VLA server. Running in simulation-only mode.
```
**解决**: 
- 检查SSH隧道是否建立
- 检查远程服务器是否运行：`curl http://localhost:8000/health`
- 检查防火墙设置

### 问题2: 仍然看不到机器人
- 按 `F` 键聚焦选中物体
- 或手动调整视口相机：右上角视口菜单 → Camera → 选择相机

### 问题3: 程序崩溃/段错误
- 确保Isaac Sim完全关闭后重新运行
- 检查NVIDIA驱动：`nvidia-smi`
- 清理缓存：`rm -rf ~/.cache/isaac-sim/`

### 问题4: 性能仍然卡顿
```bash
# 检查CPU是否已切换到performance模式
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | head -1
# 应该显示 "performance"

# 如果没有，手动设置
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance | sudo tee $cpu
done
```

---

## 一键执行脚本（可选）

创建文件 `start_vla_demo.sh`：

```bash
#!/bin/bash

# 1. 修复CPU性能
echo "Setting CPU to performance mode..."
echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils
sudo systemctl restart cpufrequtils 2>/dev/null || true

# 2. 推送代码
echo "Pushing code to remote..."
cd /home/vincent/Desktop/code/VLA
git push origin main

echo "Done! Now manually update remote server and run the demo."
```

赋予执行权限：
```bash
chmod +x start_vla_demo.sh
./start_vla_demo.sh
```

---

## 完成检查清单

- [ ] CPU性能模式已设置为performance
- [ ] 代码已推送到GitHub
- [ ] 远程服务器已拉取最新代码
- [ ] VLA服务器已在远程启动
- [ ] SSH隧道已建立
- [ ] 本地演示成功运行并能看到机器人
- [ ] VLA服务器返回正确的动作预测

---

**最后更新**: 2026-02-15  
**修复内容**: 视口相机API、物理初始化顺序、空值处理
