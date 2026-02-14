#!/usr/bin/env python3
"""
VLA 数据收集脚本 - Isaac Sim 5.x
使用 Isaac Sim 原生相机和控制器收集演示数据
"""

import argparse
import sys
from pathlib import Path

# 首先解析参数
parser = argparse.ArgumentParser(description='Collect demonstration data')
parser.add_argument('--num_episodes', type=int, default=100)
parser.add_argument('--output_dir', type=str, default='data/demos')
parser.add_argument('--instruction', type=str, default='pick up the red block')
parser.add_argument('--max_steps', type=int, default=200)
parser.add_argument('--headless', action='store_true')
parser.add_argument('--render_interval', type=int, default=1, help='渲染间隔步数')
args = parser.parse_args()

# 必须在导入其他模块前启动 SimulationApp
from isaacsim import SimulationApp

config = {
    "headless": args.headless,
    "width": 640,
    "height": 480,
    "renderer": "RayTracedLighting",  # 使用光线追踪获取更好的图像
    "anti_aliasing": 3,
}

simulation_app = SimulationApp(config)
print("SimulationApp started successfully!")

# 现在可以导入 Isaac Sim 模块
import numpy as np
import logging
import h5py
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
from PIL import Image

# Isaac Sim 核心模块
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, GroundPlane
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.franka import Franka
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.prims import XFormPrim
from pxr import Gf

# 相机模块
try:
    from omni.isaac.sensor import Camera
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False
    print("Warning: omni.isaac.sensor not available, using synthetic camera")

import omni.replicator.core as rep

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCollector:
    """数据收集器 - 使用 Isaac Sim 原生功能"""
    
    def __init__(
        self,
        output_dir: str,
        image_size: Tuple[int, int] = (224, 224),
        headless: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_size = image_size
        self.headless = headless
        
        # 仿真世界
        self.world: Optional[World] = None
        self.robot: Optional[Franka] = None
        self.camera: Optional[Camera] = None
        self.target_object = None
        self.objects = []
        
        # 数据缓存
        self.episode_count = 0
        self.current_data = None
        
    def setup_scene(self) -> None:
        """设置仿真场景"""
        # 创建世界
        self.world = World(stage_units_in_meters=1.0)
        
        # 添加地面
        self.world.scene.add_default_ground_plane()
        
        # 添加 Franka 机械臂
        self.robot = self.world.scene.add(
            Franka(
                prim_path="/World/Franka",
                name="franka",
                position=np.array([0.0, 0.0, 0.0]),
            )
        )
        
        # 添加桌子 (使用 FixedCuboid 更稳定)
        from omni.isaac.core.objects import FixedCuboid
        table = FixedCuboid(
            prim_path="/World/Table",
            name="table",
            position=np.array([0.5, 0.0, 0.25]),
            size=1.0,
            scale=np.array([0.6, 0.8, 0.5]),
            color=np.array([0.5, 0.35, 0.2]),
        )
        self.world.scene.add(table)
        
        # 添加目标物体
        self._add_target_objects()
        
        # 设置相机
        self._setup_camera()
        
        # 重置世界
        self.world.reset()
        
        # 多步模拟让物体稳定
        for _ in range(50):
            self.world.step(render=True)
        
        logger.info("Scene setup complete")
    
    def _add_target_objects(self) -> None:
        """添加抓取目标物体"""
        colors = [
            [1.0, 0.0, 0.0],  # 红色
            [0.0, 1.0, 0.0],  # 绿色
            [0.0, 0.0, 1.0],  # 蓝色
        ]
        
        for i, color in enumerate(colors):
            # 随机位置在桌面上
            x = np.random.uniform(0.35, 0.55)
            y = np.random.uniform(-0.2, 0.2)
            z = 0.55  # 桌面上方
            
            obj = DynamicCuboid(
                prim_path=f"/World/Object_{i}",
                name=f"object_{i}",
                position=np.array([x, y, z]),
                size=0.04,  # 4cm 边长的方块
                color=np.array(color),
                mass=0.1,
            )
            self.world.scene.add(obj)
            self.objects.append(obj)
        
        # 设置目标物体为第一个（红色）
        self.target_object = self.objects[0]
        logger.info(f"Added {len(self.objects)} objects")
    
    def _setup_camera(self) -> None:
        """设置相机 - 使用 Replicator 进行 headless 渲染"""
        import omni.replicator.core as rep
        
        # 使用 Replicator 创建相机
        self.cam = rep.create.camera(
            position=(0.8, 0.5, 0.8),
            look_at=(0.4, 0.0, 0.5),
            name="data_collection_camera"
        )
        
        # 创建渲染产品
        self.render_product = rep.create.render_product(
            self.cam,
            resolution=(self.image_size[0], self.image_size[1])
        )
        
        # 创建 RGB 输出
        self.rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb_annot.attach([self.render_product])
        
        # 预热渲染管线
        logger.info("Warming up rendering pipeline...")
        for _ in range(30):
            rep.orchestrator.step()
        
        logger.info("Camera initialized with Replicator")
    
    def get_camera_image(self) -> np.ndarray:
        """获取相机图像"""
        import omni.replicator.core as rep
        
        if not hasattr(self, 'rgb_annot') or self.rgb_annot is None:
            return np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        
        try:
            # 执行一帧渲染
            rep.orchestrator.step(rt_subframes=4)
            
            # 获取 RGB 数据
            data = self.rgb_annot.get_data()
            
            if data is None or len(data) == 0:
                return np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
            
            # 如果是 RGBA，取前3通道
            if isinstance(data, np.ndarray):
                if data.ndim == 3 and data.shape[2] == 4:
                    return data[:, :, :3].astype(np.uint8)
                elif data.ndim == 3 and data.shape[2] == 3:
                    return data.astype(np.uint8)
            
            return np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
            
        except Exception as e:
            logger.warning(f"Camera capture error: {e}")
            return np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
    
    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取末端执行器位姿"""
        if self.robot is None:
            return np.zeros(3), np.array([1, 0, 0, 0])
        
        ee_pos, ee_quat = self.robot.gripper.get_world_pose()
        return ee_pos, ee_quat
    
    def get_target_position(self) -> np.ndarray:
        """获取目标物体位置"""
        if self.target_object is None:
            return np.array([0.4, 0.0, 0.55])
        
        pos, _ = self.target_object.get_world_pose()
        return pos
    
    def expert_policy(self) -> np.ndarray:
        """
        专家策略：基于当前状态计算动作
        
        Returns:
            7维动作: [dx, dy, dz, drx, dry, drz, gripper]
        """
        ee_pos, _ = self.get_ee_pose()
        target_pos = self.get_target_position()
        
        # 计算到目标的距离和方向
        delta = target_pos - ee_pos
        distance = np.linalg.norm(delta)
        
        # 阶段控制
        if distance > 0.15:  # 远距离：快速接近
            # 先抬高，再移动到目标上方
            approach_height = 0.3
            if ee_pos[2] < target_pos[2] + 0.15:
                # 先抬高
                action = np.array([0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0])
            else:
                # 水平移动到目标上方
                dx = np.clip(delta[0], -0.05, 0.05)
                dy = np.clip(delta[1], -0.05, 0.05)
                action = np.array([dx, dy, 0.0, 0.0, 0.0, 0.0, 1.0])
        elif distance > 0.05:  # 中距离：下降接近
            # 向目标移动
            dx = np.clip(delta[0], -0.02, 0.02)
            dy = np.clip(delta[1], -0.02, 0.02)
            dz = np.clip(delta[2] - 0.02, -0.02, 0.02)  # 稍微在上方
            action = np.array([dx, dy, dz, 0.0, 0.0, 0.0, 1.0])
        else:  # 近距离：抓取
            # 下降并闭合夹爪
            dz = np.clip(delta[2], -0.01, 0.01)
            gripper = 0.0  # 闭合夹爪
            action = np.array([0.0, 0.0, dz, 0.0, 0.0, 0.0, gripper])
        
        # 添加小的随机噪声使轨迹更自然
        noise = np.random.normal(0, 0.002, size=6)
        action[:6] += noise
        
        return action
    
    def apply_action(self, action: np.ndarray) -> None:
        """应用动作到机器人"""
        if self.robot is None:
            return
        
        # 获取当前末端位置
        ee_pos, ee_quat = self.get_ee_pose()
        
        # 计算目标位置
        target_pos = ee_pos + action[:3]
        
        # 使用 Franka 的控制器移动到目标位置
        # 简化：使用关节位置控制
        current_joint_positions = self.robot.get_joint_positions()
        
        # 使用简单的雅可比近似
        # 实际应用中应该使用 IK 控制器
        joint_delta = np.zeros(9)
        joint_delta[:3] = action[:3] * 10.0  # 放大系数 - 实际需要 IK
        
        # 夹爪控制
        gripper_action = action[6]
        if gripper_action < 0.5:
            # 闭合夹爪
            joint_delta[7] = -0.01
            joint_delta[8] = -0.01
        else:
            # 张开夹爪
            joint_delta[7] = 0.01
            joint_delta[8] = 0.01
        
        new_positions = current_joint_positions + joint_delta
        
        # 限制关节范围
        joint_limits_low = np.array([-2.8, -1.7, -2.8, -3.0, -2.8, -0.01, -2.8, 0.0, 0.0])
        joint_limits_high = np.array([2.8, 1.7, 2.8, -0.07, 2.8, 3.7, 2.8, 0.04, 0.04])
        new_positions = np.clip(new_positions, joint_limits_low, joint_limits_high)
        
        self.robot.set_joint_positions(new_positions)
    
    def reset_episode(self) -> Dict[str, Any]:
        """重置 episode"""
        # 重置机器人到初始位置
        if self.robot is not None:
            initial_joint_positions = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04])
            self.robot.set_joint_positions(initial_joint_positions)
        
        # 随机化物体位置
        for i, obj in enumerate(self.objects):
            x = np.random.uniform(0.35, 0.55)
            y = np.random.uniform(-0.2, 0.2)
            z = 0.55
            obj.set_world_pose(position=np.array([x, y, z]))
        
        # 运行几步让物体稳定
        for _ in range(20):
            self.world.step(render=True)
        
        # 获取初始观测
        return self.get_observation()
    
    def get_observation(self) -> Dict[str, Any]:
        """获取当前观测"""
        image = self.get_camera_image()
        ee_pos, ee_quat = self.get_ee_pose()
        joint_positions = self.robot.get_joint_positions() if self.robot else np.zeros(9)
        
        return {
            "image": image,
            "ee_position": ee_pos,
            "ee_orientation": ee_quat,
            "joint_positions": joint_positions[:7],
            "gripper_state": (joint_positions[7] + joint_positions[8]) / 0.08 if self.robot else 0.0,
        }
    
    def check_success(self) -> bool:
        """检查是否抓取成功"""
        if self.target_object is None:
            return False
        
        target_pos, _ = self.target_object.get_world_pose()
        ee_pos, _ = self.get_ee_pose()
        
        # 如果物体被抬起且靠近夹爪
        if target_pos[2] > 0.6 and np.linalg.norm(target_pos[:2] - ee_pos[:2]) < 0.1:
            return True
        
        return False
    
    def collect_episode(self, instruction: str) -> bool:
        """收集一个 episode 的数据"""
        logger.info(f"Collecting episode {self.episode_count}...")
        
        # 初始化数据存储
        images = []
        actions = []
        ee_positions = []
        joint_positions = []
        rewards = []
        
        # 重置
        obs = self.reset_episode()
        
        # 收集数据
        for step in range(args.max_steps):
            # 获取动作
            action = self.expert_policy()
            
            # 记录数据
            images.append(obs["image"])
            actions.append(action)
            ee_positions.append(obs["ee_position"])
            joint_positions.append(obs["joint_positions"])
            
            # 应用动作
            self.apply_action(action)
            
            # 仿真步进
            self.world.step(render=True)
            
            # 获取新观测
            obs = self.get_observation()
            
            # 计算奖励
            ee_pos = obs["ee_position"]
            target_pos = self.get_target_position()
            distance = np.linalg.norm(ee_pos - target_pos)
            reward = -distance  # 负距离作为奖励
            rewards.append(reward)
            
            # 检查成功
            if self.check_success():
                logger.info(f"  Success at step {step}!")
                break
        
        # 保存数据
        self._save_episode(
            images=np.array(images),
            actions=np.array(actions),
            ee_positions=np.array(ee_positions),
            joint_positions=np.array(joint_positions),
            rewards=np.array(rewards),
            instruction=instruction,
            success=self.check_success()
        )
        
        self.episode_count += 1
        return self.check_success()
    
    def _save_episode(
        self,
        images: np.ndarray,
        actions: np.ndarray,
        ee_positions: np.ndarray,
        joint_positions: np.ndarray,
        rewards: np.ndarray,
        instruction: str,
        success: bool
    ) -> None:
        """保存 episode 数据到 HDF5"""
        filepath = self.output_dir / f"episode_{self.episode_count:06d}.h5"
        
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('images', data=images, compression='gzip')
            f.create_dataset('actions', data=actions)
            f.create_dataset('ee_positions', data=ee_positions)
            f.create_dataset('joint_positions', data=joint_positions)
            f.create_dataset('rewards', data=rewards)
            
            # 元数据
            f.attrs['instruction'] = instruction
            f.attrs['success'] = success
            f.attrs['num_steps'] = len(images)
            f.attrs['timestamp'] = datetime.now().isoformat()
        
        logger.info(f"  Saved: {filepath} ({len(images)} steps)")
    
    def collect_all(self, num_episodes: int, instruction: str) -> Dict[str, Any]:
        """收集所有 episodes"""
        stats = {
            "total_episodes": 0,
            "successful_episodes": 0,
            "total_steps": 0,
        }
        
        for ep in range(num_episodes):
            success = self.collect_episode(instruction)
            
            stats["total_episodes"] += 1
            if success:
                stats["successful_episodes"] += 1
            
            if (ep + 1) % 10 == 0:
                logger.info(
                    f"Progress: {ep+1}/{num_episodes} "
                    f"(success rate: {stats['successful_episodes']/(ep+1)*100:.1f}%)"
                )
        
        return stats
    
    def cleanup(self) -> None:
        """清理资源"""
        if self.world is not None:
            self.world.clear()


def main():
    logger.info("Starting data collection...")
    
    collector = DataCollector(
        output_dir=args.output_dir,
        headless=args.headless
    )
    
    # 设置场景
    collector.setup_scene()
    
    # 收集数据
    stats = collector.collect_all(
        num_episodes=args.num_episodes,
        instruction=args.instruction
    )
    
    logger.info(f"Collection complete!")
    logger.info(f"Stats: {stats}")
    
    # 清理
    collector.cleanup()
    simulation_app.close()


if __name__ == "__main__":
    main()
