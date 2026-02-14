# NVIDIA Isaac Sim 安装指南

## ✅ 安装状态: 已完成

Isaac Sim 5.1.0 已成功安装到 `~/isaac-sim/`

---

## 使用 Isaac Sim Python 运行脚本

```bash
# 数据收集 (需要先退出 conda 环境)
conda deactivate
~/isaac-sim/python.sh /home/vincent/Desktop/code/VLA/scripts/collect_data.py \
    --num_episodes 500 \
    --output_dir /home/vincent/Desktop/code/VLA/data/demos \
    --headless

# 训练 (使用 Isaac Sim Python)
~/isaac-sim/python.sh /home/vincent/Desktop/code/VLA/scripts/train_openvla.py \
    --mode sft \
    --data_dir data/demos \
    --batch_size 16
```

---

## 启动 Isaac Sim GUI

```bash
conda deactivate
~/isaac-sim/isaac-sim.selector.sh
```

---

## 系统要求

| 要求 | 你的配置 |
|------|----------|
| GPU显存 | 48 GB ✅ |
| Python | 3.11.13 ✅ |

---

## 注意事项

⚠️ **运行 Isaac Sim 脚本前需要退出 conda 环境**：
```bash
conda deactivate
```

