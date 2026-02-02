#!/usr/bin/env python3
# Data Collection Script
"""
数据收集脚本

从Isaac Sim仿真中收集演示数据用于VLA训练

使用方法:
    # 使用专家策略收集数据
    python scripts/collect_data.py --num_episodes 100 --output_dir data/demos
    
    # 使用远程VLA策略收集数据
    python scripts/collect_data.py --policy vla --server http://server:8000
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from vla_platform.training.data import TrajectoryCollector
from vla_platform.simulation import SimulationManager, FrankaGraspEnv, check_isaac_sim_available
from vla_platform.core.config import SimulationConfig
from vla_platform.core.base_interfaces import Action

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def expert_policy(observation, instruction: str) -> np.ndarray:
    """
    专家策略：简单的启发式控制
    
    用于收集演示数据
    """
    # 获取物体位置（假设已知）
    # 实际应该从观测中推断
    target_pos = np.array([0.4, 0.0, 0.1])  # 假设的目标位置
    
    ee_pos = observation.ee_position if observation.ee_position is not None else np.array([0.3, 0.0, 0.3])
    
    # 计算动作
    delta = target_pos - ee_pos
    delta = np.clip(delta, -0.02, 0.02)
    
    # 夹爪控制
    distance = np.linalg.norm(delta[:2])
    gripper = 0.0 if distance < 0.02 else 1.0
    
    return np.concatenate([delta, np.zeros(3), [gripper]])


def main():
    parser = argparse.ArgumentParser(description="Data Collection Script")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="data/demos")
    parser.add_argument("--instruction", type=str, default="pick up the red block")
    parser.add_argument("--policy", type=str, choices=["expert", "vla", "random"], default="expert")
    parser.add_argument("--server", type=str, default=None, help="VLA server URL for vla policy")
    parser.add_argument("--headless", action="store_true")
    
    args = parser.parse_args()
    
    # 检查Isaac Sim
    if not check_isaac_sim_available():
        logger.error("Isaac Sim not available. Run in Isaac Sim environment.")
        return
    
    # 创建仿真环境
    logger.info("Creating simulation environment...")
    sim_config = SimulationConfig(headless=args.headless)
    sim_manager = SimulationManager(sim_config)
    sim_manager.create_world()
    
    env = FrankaGraspEnv(sim_manager)
    env.setup()
    
    # 创建数据收集器
    collector = TrajectoryCollector(
        save_dir=args.output_dir,
        save_format="hdf5"
    )
    
    # 选择策略
    if args.policy == "expert":
        policy = expert_policy
    elif args.policy == "vla":
        if args.server is None:
            logger.error("VLA server URL required for vla policy")
            return
        # 创建VLA客户端
        from vla_platform.models import OpenVLAClient
        from vla_platform.core.config import RemoteServerConfig
        
        config = RemoteServerConfig(host=args.server.split("//")[1].split(":")[0])
        vla_client = OpenVLAClient(config)
        vla_client.connect()
        
        def vla_policy(obs, inst):
            action = vla_client.predict(obs, inst)
            return action.values
        
        policy = vla_policy
    else:
        def random_policy(obs, inst):
            return np.random.uniform(-0.02, 0.02, size=7)
        policy = random_policy
    
    # 收集数据
    logger.info(f"Collecting {args.num_episodes} episodes...")
    stats = collector.collect_demonstrations(
        env=env,
        policy=policy,
        num_episodes=args.num_episodes,
        instruction=args.instruction,
        max_steps=args.max_steps
    )
    
    logger.info(f"Collection complete! Stats: {stats}")
    logger.info(f"Data saved to: {args.output_dir}")
    
    # 清理
    sim_manager.cleanup()


if __name__ == "__main__":
    main()
