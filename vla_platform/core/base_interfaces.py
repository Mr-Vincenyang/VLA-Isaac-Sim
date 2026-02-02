# VLA Platform - Base Interfaces
"""
VLA仿真平台核心接口定义
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import numpy as np


@dataclass
class Observation:
    """机器人观测数据"""
    image: np.ndarray  # RGB图像 (H, W, 3)
    depth: Optional[np.ndarray] = None  # 深度图 (H, W)
    joint_positions: Optional[np.ndarray] = None  # 关节位置
    joint_velocities: Optional[np.ndarray] = None  # 关节速度
    ee_position: Optional[np.ndarray] = None  # 末端执行器位置 (x, y, z)
    ee_orientation: Optional[np.ndarray] = None  # 末端执行器姿态 (quaternion)
    gripper_state: Optional[float] = None  # 夹爪状态 [0, 1]


@dataclass
class Action:
    """机器人动作"""
    values: np.ndarray  # 动作值 (通常7维: dx, dy, dz, drx, dry, drz, gripper)
    action_type: str = "delta_ee"  # "delta_ee", "absolute_ee", "joint"
    
    @property
    def position_delta(self) -> np.ndarray:
        """位置增量 (dx, dy, dz)"""
        return self.values[:3]
    
    @property
    def rotation_delta(self) -> np.ndarray:
        """旋转增量 (drx, dry, drz)"""
        return self.values[3:6]
    
    @property
    def gripper_action(self) -> float:
        """夹爪动作"""
        return float(self.values[6]) if len(self.values) > 6 else 0.0


class VLAModelInterface(ABC):
    """VLA模型抽象接口"""
    
    @abstractmethod
    def predict(self, observation: Observation, instruction: str) -> Action:
        """
        根据观测和语言指令预测动作
        
        Args:
            observation: 当前观测
            instruction: 语言指令
            
        Returns:
            预测的动作
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """重置模型状态（如果有历史依赖）"""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """返回模型名称"""
        pass
    
    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """检查模型服务是否连接"""
        pass


class RobotController(ABC):
    """机器人控制器抽象接口"""
    
    @abstractmethod
    def apply_action(self, action: Action) -> bool:
        """
        应用动作到机器人
        
        Args:
            action: 要执行的动作
            
        Returns:
            是否成功应用
        """
        pass
    
    @abstractmethod
    def get_observation(self) -> Observation:
        """
        获取当前观测
        
        Returns:
            当前观测数据
        """
        pass
    
    @abstractmethod
    def reset(self, initial_config: Optional[Dict[str, Any]] = None) -> Observation:
        """
        重置机器人到初始状态
        
        Args:
            initial_config: 可选的初始配置
            
        Returns:
            重置后的观测
        """
        pass
    
    @abstractmethod
    def get_joint_positions(self) -> np.ndarray:
        """获取关节位置"""
        pass
    
    @abstractmethod
    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取末端执行器位姿 (position, quaternion)"""
        pass


class SensorInterface(ABC):
    """传感器接口"""
    
    @abstractmethod
    def capture(self) -> Dict[str, np.ndarray]:
        """
        采集传感器数据
        
        Returns:
            传感器数据字典
        """
        pass
    
    @abstractmethod
    def get_resolution(self) -> Tuple[int, int]:
        """获取传感器分辨率"""
        pass
    
    @property
    @abstractmethod
    def sensor_type(self) -> str:
        """传感器类型"""
        pass


class TrajectoryPlanner(ABC):
    """轨迹规划器抽象接口"""
    
    @abstractmethod
    def plan(
        self, 
        start: np.ndarray, 
        goal: np.ndarray,
        obstacles: Optional[List[Any]] = None
    ) -> Optional[List[np.ndarray]]:
        """
        规划从起点到终点的轨迹
        
        Args:
            start: 起始配置
            goal: 目标配置
            obstacles: 障碍物列表
            
        Returns:
            轨迹点列表，如果规划失败返回None
        """
        pass
    
    @abstractmethod
    def interpolate(
        self,
        waypoints: List[np.ndarray],
        dt: float
    ) -> List[np.ndarray]:
        """
        在路径点之间插值
        
        Args:
            waypoints: 路径点
            dt: 时间步长
            
        Returns:
            插值后的轨迹
        """
        pass
