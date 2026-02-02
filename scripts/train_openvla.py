#!/usr/bin/env python3
# OpenVLA Training Script
"""
OpenVLA训练脚本

支持:
1. 监督学习微调 (SFT)
2. 强化学习微调 (RL/PPO)
3. LoRA高效微调

使用方法:
    # 监督学习微调
    python scripts/train_openvla.py --mode sft --data_dir data/demos

    # RL微调
    python scripts/train_openvla.py --mode rl --env isaac_sim

    # 在服务器上训练（无GUI）
    python scripts/train_openvla.py --mode sft --headless
"""
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from vla_platform.training.openvla_model import (
    OpenVLAModel, 
    OpenVLAConfig,
    load_pretrained_openvla,
    estimate_gpu_memory_for_openvla
)
from vla_platform.training.rl import PPOTrainer, RLConfig, LoRAWrapper
from vla_platform.training.data import VLADataset, TrajectoryCollector, create_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="OpenVLA Training Script")
    
    # 训练模式
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["sft", "rl", "eval"],
        default="sft",
        help="Training mode: sft (supervised), rl (reinforcement), eval (evaluation)"
    )
    
    # 模型参数
    parser.add_argument(
        "--model_name",
        type=str,
        default="openvla/openvla-7b",
        help="Pretrained model name"
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA for efficient fine-tuning"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["none", "4bit", "8bit"],
        default="none",
        help="Quantization for inference"
    )
    
    # 数据参数
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/demos",
        help="Directory containing training data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints"
    )
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    
    # RL参数
    parser.add_argument("--rollout_steps", type=int, default=2048)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    
    # 其他
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--local_model",
        action="store_true",
        help="Use simplified local model instead of pretrained"
    )
    
    return parser.parse_args()


def check_gpu():
    """检查GPU可用性和显存"""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Training will be slow on CPU.")
        return None, 0
    
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    logger.info(f"GPU: {gpu_name}")
    logger.info(f"Total memory: {total_memory:.1f} GB")
    
    # 检查是否足够运行OpenVLA
    memory_requirements = estimate_gpu_memory_for_openvla()
    logger.info(f"Memory requirements: {memory_requirements}")
    
    if total_memory < memory_requirements["int4_7b"]:
        logger.warning(
            f"GPU memory ({total_memory:.1f}GB) may not be sufficient for OpenVLA. "
            "Consider using INT4 quantization or remote server."
        )
    
    return device, total_memory


def train_sft(args, model, train_loader, val_loader, device):
    """监督学习微调"""
    logger.info("Starting supervised fine-tuning...")
    
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader) * args.num_epochs
    )
    
    model.to(device)
    model.train()
    
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # 准备输入
            images = batch["image"].to(device)
            actions = batch["action"].to(device)
            
            # 简化：使用虚拟input_ids
            input_ids = torch.zeros(images.size(0), 10, dtype=torch.long, device=device)
            
            # 前向传播
            outputs = model(images, input_ids, labels=actions.long())
            loss = outputs.get("loss", torch.tensor(0.0))
            
            # 梯度累积
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            epoch_loss += loss.item() * args.gradient_accumulation_steps
            num_batches += 1
            
            if global_step % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}, Step {global_step}, "
                    f"Loss: {epoch_loss/num_batches:.4f}, "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )
            
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        
        # Epoch结束
        avg_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
        
        # 保存检查点
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = Path(args.output_dir) / f"best_model.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved best model to {save_path}")
    
    logger.info("Training completed!")


def train_rl(args, model, device):
    """强化学习微调"""
    logger.info("Starting RL fine-tuning...")
    
    # 检查Isaac Sim
    try:
        from vla_platform.simulation import SimulationManager, FrankaGraspEnv
    except ImportError:
        logger.error("Isaac Sim not available. Run in Isaac Sim environment.")
        return
    
    # 创建RL配置
    rl_config = RLConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.ppo_epochs,
        use_lora=args.use_lora,
    )
    
    # 创建训练器
    trainer = PPOTrainer(model, rl_config, device=device)
    
    # 创建环境
    from vla_platform.core.config import SimulationConfig
    sim_config = SimulationConfig(headless=args.headless)
    sim_manager = SimulationManager(sim_config)
    sim_manager.create_world()
    
    env = FrankaGraspEnv(sim_manager)
    env.setup()
    
    # 训练循环
    for iteration in range(args.num_epochs):
        logger.info(f"=== RL Iteration {iteration+1}/{args.num_epochs} ===")
        
        # 收集经验
        rollout_stats = trainer.collect_rollouts(
            env,
            vla_client=None,  # 使用本地模型
            num_steps=args.rollout_steps,
            instruction="pick up the red block"
        )
        logger.info(f"Rollout stats: {rollout_stats}")
        
        # 训练
        train_stats = trainer.train_step()
        logger.info(f"Training stats: {train_stats}")
        
        # 保存检查点
        if (iteration + 1) % 10 == 0:
            save_path = Path(args.output_dir) / f"rl_checkpoint_{iteration+1}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(str(save_path))
    
    # 清理
    sim_manager.cleanup()
    logger.info("RL training completed!")


def main():
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 检查GPU
    device, gpu_memory = check_gpu()
    if device is None:
        device = torch.device("cpu")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    if args.local_model:
        logger.info("Using simplified local model")
        config = OpenVLAConfig()
        model = OpenVLAModel(config)
    else:
        logger.info(f"Loading pretrained model: {args.model_name}")
        quantization = None if args.quantization == "none" else args.quantization
        
        try:
            model, processor = load_pretrained_openvla(
                args.model_name,
                device=str(device),
                quantization=quantization
            )
        except Exception as e:
            logger.warning(f"Failed to load pretrained model: {e}")
            logger.info("Falling back to local model")
            config = OpenVLAConfig()
            model = OpenVLAModel(config)
    
    # 应用LoRA
    if args.use_lora:
        logger.info("Applying LoRA...")
        model = LoRAWrapper.apply_lora(model, r=args.lora_r)
    
    # 根据模式训练
    if args.mode == "sft":
        # 加载数据
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            logger.warning(f"Data directory {data_dir} not found. Creating dummy data...")
            data_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            train_loader, val_loader = create_dataloaders(
                str(data_dir),
                batch_size=args.batch_size
            )
            train_sft(args, model, train_loader, val_loader, device)
        except Exception as e:
            logger.error(f"Failed to create dataloaders: {e}")
            logger.info("Please collect demonstration data first using data collection scripts")
    
    elif args.mode == "rl":
        train_rl(args, model, device)
    
    elif args.mode == "eval":
        logger.info("Evaluation mode - not implemented yet")


if __name__ == "__main__":
    main()
