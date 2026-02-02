# VLA Data Collection and Dataset
"""
VLA训练数据收集和数据集模块

用于:
1. 从Isaac Sim收集演示数据
2. 在线RL数据收集
3. 数据预处理和增强
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import h5py
import pickle
from PIL import Image
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryData:
    """单条轨迹数据"""
    episode_id: str
    instruction: str
    images: List[np.ndarray]  # [T, H, W, 3]
    actions: List[np.ndarray]  # [T, action_dim]
    rewards: List[float]
    joint_positions: Optional[List[np.ndarray]] = None
    ee_positions: Optional[List[np.ndarray]] = None
    gripper_states: Optional[List[float]] = None
    success: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrajectoryCollector:
    """
    轨迹数据收集器
    
    从Isaac Sim仿真中收集演示数据
    """
    
    def __init__(
        self,
        save_dir: str,
        image_size: Tuple[int, int] = (224, 224),
        save_format: str = "hdf5"  # "hdf5" or "pickle"
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.image_size = image_size
        self.save_format = save_format
        
        self.current_trajectory: Optional[TrajectoryData] = None
        self.episode_count = 0
        
    def start_episode(self, instruction: str) -> None:
        """开始新的episode"""
        self.current_trajectory = TrajectoryData(
            episode_id=f"episode_{self.episode_count:06d}",
            instruction=instruction,
            images=[],
            actions=[],
            rewards=[],
            joint_positions=[],
            ee_positions=[],
            gripper_states=[],
            metadata={
                "timestamp": datetime.now().isoformat(),
                "image_size": self.image_size
            }
        )
        
    def add_step(
        self,
        observation,
        action: np.ndarray,
        reward: float
    ) -> None:
        """添加一步数据"""
        if self.current_trajectory is None:
            raise RuntimeError("No active episode. Call start_episode first.")
        
        # 保存图像
        image = observation.image
        if image.shape[:2] != self.image_size:
            pil_img = Image.fromarray(image)
            pil_img = pil_img.resize(self.image_size)
            image = np.array(pil_img)
        
        self.current_trajectory.images.append(image)
        self.current_trajectory.actions.append(action.copy())
        self.current_trajectory.rewards.append(reward)
        
        # 保存可选状态
        if observation.joint_positions is not None:
            self.current_trajectory.joint_positions.append(
                observation.joint_positions.copy()
            )
        if observation.ee_position is not None:
            self.current_trajectory.ee_positions.append(
                observation.ee_position.copy()
            )
        if observation.gripper_state is not None:
            self.current_trajectory.gripper_states.append(
                observation.gripper_state
            )
    
    def end_episode(self, success: bool = False) -> str:
        """结束episode并保存"""
        if self.current_trajectory is None:
            raise RuntimeError("No active episode")
        
        self.current_trajectory.success = success
        
        # 转换为数组
        self.current_trajectory.images = np.stack(self.current_trajectory.images)
        self.current_trajectory.actions = np.stack(self.current_trajectory.actions)
        
        # 保存
        filepath = self._save_trajectory(self.current_trajectory)
        
        self.episode_count += 1
        self.current_trajectory = None
        
        return filepath
    
    def _save_trajectory(self, trajectory: TrajectoryData) -> str:
        """保存轨迹到文件"""
        if self.save_format == "hdf5":
            return self._save_hdf5(trajectory)
        else:
            return self._save_pickle(trajectory)
    
    def _save_hdf5(self, trajectory: TrajectoryData) -> str:
        """保存为HDF5格式"""
        filepath = self.save_dir / f"{trajectory.episode_id}.h5"
        
        with h5py.File(filepath, 'w') as f:
            # 主数据
            f.create_dataset('images', data=trajectory.images, compression='gzip')
            f.create_dataset('actions', data=trajectory.actions)
            f.create_dataset('rewards', data=trajectory.rewards)
            
            # 属性
            f.attrs['episode_id'] = trajectory.episode_id
            f.attrs['instruction'] = trajectory.instruction
            f.attrs['success'] = trajectory.success
            f.attrs['metadata'] = json.dumps(trajectory.metadata)
            
            # 可选数据
            if trajectory.joint_positions:
                f.create_dataset(
                    'joint_positions', 
                    data=np.stack(trajectory.joint_positions)
                )
            if trajectory.ee_positions:
                f.create_dataset(
                    'ee_positions',
                    data=np.stack(trajectory.ee_positions)
                )
        
        logger.info(f"Saved trajectory to {filepath}")
        return str(filepath)
    
    def _save_pickle(self, trajectory: TrajectoryData) -> str:
        """保存为Pickle格式"""
        filepath = self.save_dir / f"{trajectory.episode_id}.pkl"
        
        with open(filepath, 'wb') as f:
            pickle.dump(trajectory, f)
        
        return str(filepath)
    
    def collect_demonstrations(
        self,
        env,
        policy: Callable,
        num_episodes: int,
        instruction: str,
        max_steps: int = 200
    ) -> Dict[str, Any]:
        """
        收集演示数据
        
        Args:
            env: Isaac Sim环境
            policy: 策略函数 (observation, instruction) -> action
            num_episodes: 收集的episode数量
            instruction: 任务指令
            max_steps: 每个episode的最大步数
            
        Returns:
            收集统计信息
        """
        stats = {
            "num_episodes": 0,
            "num_success": 0,
            "total_steps": 0,
            "total_rewards": 0.0
        }
        
        for ep in range(num_episodes):
            observation = env.reset()
            self.start_episode(instruction)
            
            episode_reward = 0
            
            for step in range(max_steps):
                # 获取动作
                action = policy(observation, instruction)
                
                # 执行
                from vla_platform.core.base_interfaces import Action
                action_obj = Action(values=action, action_type="delta_ee")
                env.apply_action(action_obj)
                env.sim_manager.step()
                
                next_observation = env.get_observation()
                
                # 计算奖励
                reward = self._compute_reward(env)
                episode_reward += reward
                
                # 保存
                self.add_step(observation, action, reward)
                
                observation = next_observation
                
                # 检查完成
                if env.check_grasp_success():
                    break
            
            success = env.check_grasp_success()
            self.end_episode(success)
            
            stats["num_episodes"] += 1
            stats["num_success"] += int(success)
            stats["total_steps"] += step + 1
            stats["total_rewards"] += episode_reward
            
            logger.info(
                f"Episode {ep+1}/{num_episodes}: "
                f"steps={step+1}, reward={episode_reward:.2f}, success={success}"
            )
        
        stats["success_rate"] = stats["num_success"] / stats["num_episodes"]
        stats["avg_steps"] = stats["total_steps"] / stats["num_episodes"]
        stats["avg_reward"] = stats["total_rewards"] / stats["num_episodes"]
        
        return stats
    
    def _compute_reward(self, env) -> float:
        """计算奖励"""
        if env.check_grasp_success():
            return 10.0
        return -0.01  # 小的时间惩罚


class VLADataset(Dataset):
    """
    VLA训练数据集
    
    从收集的轨迹数据创建训练数据集
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        max_length: Optional[int] = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # 加载数据文件列表
        self.data_files = self._load_file_list()
        
        if max_length:
            self.data_files = self.data_files[:max_length]
        
        logger.info(f"Loaded {len(self.data_files)} trajectories for {split}")
    
    def _load_file_list(self) -> List[Path]:
        """加载数据文件列表"""
        h5_files = list(self.data_dir.glob("*.h5"))
        pkl_files = list(self.data_dir.glob("*.pkl"))
        return sorted(h5_files + pkl_files)
    
    def __len__(self) -> int:
        return len(self.data_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        filepath = self.data_files[idx]
        
        if filepath.suffix == ".h5":
            data = self._load_hdf5(filepath)
        else:
            data = self._load_pickle(filepath)
        
        # 随机选择一个时间步
        T = len(data["images"])
        t = np.random.randint(0, T)
        
        sample = {
            "image": data["images"][t],
            "action": data["actions"][t],
            "instruction": data["instruction"],
        }
        
        # 转换为tensor
        sample["image"] = torch.from_numpy(sample["image"]).float().permute(2, 0, 1) / 255.0
        sample["action"] = torch.from_numpy(sample["action"]).float()
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def _load_hdf5(self, filepath: Path) -> Dict[str, Any]:
        """加载HDF5文件"""
        with h5py.File(filepath, 'r') as f:
            return {
                "images": f["images"][:],
                "actions": f["actions"][:],
                "instruction": f.attrs["instruction"],
                "success": f.attrs["success"],
            }
    
    def _load_pickle(self, filepath: Path) -> Dict[str, Any]:
        """加载Pickle文件"""
        with open(filepath, 'rb') as f:
            traj = pickle.load(f)
        return {
            "images": traj.images,
            "actions": traj.actions,
            "instruction": traj.instruction,
            "success": traj.success,
        }


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.9
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器
    """
    # 获取所有数据文件
    data_path = Path(data_dir)
    all_files = sorted(list(data_path.glob("*.h5")) + list(data_path.glob("*.pkl")))
    
    # 分割
    n_train = int(len(all_files) * train_ratio)
    
    train_dataset = VLADataset(data_dir, split="train", max_length=n_train)
    val_dataset = VLADataset(data_dir, split="val")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
