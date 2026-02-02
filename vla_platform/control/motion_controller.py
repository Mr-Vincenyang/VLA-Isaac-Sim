# VLA Platform - Motion Controller
"""
运动控制器实现
支持关节空间和笛卡尔空间控制
"""
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
import logging

from ..core.config import ControlConfig
from ..core.base_interfaces import Action

logger = logging.getLogger(__name__)


@dataclass
class ControllerState:
    """控制器状态"""
    target_joint_positions: Optional[np.ndarray] = None
    target_ee_position: Optional[np.ndarray] = None
    target_ee_orientation: Optional[np.ndarray] = None
    error_integral: np.ndarray = field(default_factory=lambda: np.zeros(7))
    last_error: np.ndarray = field(default_factory=lambda: np.zeros(7))


class PDController:
    """
    PD控制器
    
    用于关节空间位置控制
    """
    
    def __init__(
        self,
        kp: np.ndarray,
        kd: np.ndarray,
        joint_limits_low: Optional[np.ndarray] = None,
        joint_limits_high: Optional[np.ndarray] = None
    ):
        """
        初始化PD控制器
        
        Args:
            kp: 比例增益
            kd: 微分增益
            joint_limits_low: 关节下限
            joint_limits_high: 关节上限
        """
        self.kp = np.array(kp)
        self.kd = np.array(kd)
        self.joint_limits_low = joint_limits_low
        self.joint_limits_high = joint_limits_high
        
    def compute(
        self,
        current_position: np.ndarray,
        target_position: np.ndarray,
        current_velocity: np.ndarray,
        target_velocity: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        计算控制输出（力矩/加速度）
        
        Args:
            current_position: 当前关节位置
            target_position: 目标关节位置
            current_velocity: 当前关节速度
            target_velocity: 目标关节速度（可选）
            
        Returns:
            控制输出
        """
        if target_velocity is None:
            target_velocity = np.zeros_like(current_velocity)
        
        # 位置误差
        position_error = target_position - current_position
        
        # 速度误差
        velocity_error = target_velocity - current_velocity
        
        # PD控制律
        output = self.kp * position_error + self.kd * velocity_error
        
        return output
    
    def clip_to_limits(self, positions: np.ndarray) -> np.ndarray:
        """裁剪到关节限制"""
        if self.joint_limits_low is not None and self.joint_limits_high is not None:
            return np.clip(positions, self.joint_limits_low, self.joint_limits_high)
        return positions


class MotionController:
    """
    运动控制器
    
    支持关节空间和笛卡尔空间控制
    """
    
    # Franka Panda DH参数（简化版）
    FRANKA_LINK_LENGTHS = [0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.107]
    
    def __init__(self, config: Optional[ControlConfig] = None):
        """
        初始化运动控制器
        
        Args:
            config: 控制配置
        """
        self.config = config or ControlConfig()
        
        # 创建PD控制器
        self.pd_controller = PDController(
            kp=np.array(self.config.kp),
            kd=np.array(self.config.kd)
        )
        
        self.state = ControllerState()
        
    def set_target_joint_positions(self, positions: np.ndarray) -> None:
        """设置目标关节位置"""
        self.state.target_joint_positions = positions.copy()
    
    def set_target_ee_pose(
        self,
        position: np.ndarray,
        orientation: Optional[np.ndarray] = None
    ) -> None:
        """设置目标末端执行器位姿"""
        self.state.target_ee_position = position.copy()
        if orientation is not None:
            self.state.target_ee_orientation = orientation.copy()
    
    def compute_joint_control(
        self,
        current_positions: np.ndarray,
        current_velocities: np.ndarray,
        target_positions: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        计算关节空间控制输出
        
        Args:
            current_positions: 当前关节位置
            current_velocities: 当前关节速度
            target_positions: 目标位置（可选，使用存储的目标）
            
        Returns:
            关节力矩/控制命令
        """
        if target_positions is None:
            target_positions = self.state.target_joint_positions
        
        if target_positions is None:
            return np.zeros(7)
        
        return self.pd_controller.compute(
            current_positions,
            target_positions,
            current_velocities
        )
    
    def delta_ee_to_joint_delta(
        self,
        delta_position: np.ndarray,
        delta_rotation: np.ndarray,
        current_joint_positions: np.ndarray,
        jacobian: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        将末端执行器增量转换为关节增量
        
        使用雅可比矩阵伪逆
        
        Args:
            delta_position: 位置增量 [dx, dy, dz]
            delta_rotation: 旋转增量 [drx, dry, drz]
            current_joint_positions: 当前关节位置
            jacobian: 雅可比矩阵（可选，如果不提供则计算）
            
        Returns:
            关节增量
        """
        # 组合笛卡尔增量
        delta_x = np.concatenate([delta_position, delta_rotation])
        
        # 计算或使用提供的雅可比矩阵
        if jacobian is None:
            jacobian = self.compute_jacobian(current_joint_positions)
        
        # 使用阻尼最小二乘法（DLS）计算伪逆
        damping = 0.05
        J_T = jacobian.T
        delta_q = J_T @ np.linalg.inv(jacobian @ J_T + damping**2 * np.eye(6)) @ delta_x
        
        return delta_q
    
    def compute_jacobian(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        计算雅可比矩阵（简化版本）
        
        完整实现应使用完整的DH参数
        """
        # 简化的数值雅可比
        # 实际应用中应使用解析雅可比或从Isaac Sim获取
        eps = 1e-6
        n_joints = len(joint_positions)
        J = np.zeros((6, n_joints))
        
        # 当前末端位置
        ee_pos_0 = self._forward_kinematics(joint_positions)
        
        for i in range(n_joints):
            q_plus = joint_positions.copy()
            q_plus[i] += eps
            ee_pos_plus = self._forward_kinematics(q_plus)
            
            # 位置雅可比
            J[:3, i] = (ee_pos_plus[:3] - ee_pos_0[:3]) / eps
            # 旋转雅可比（简化）
            J[3:, i] = (ee_pos_plus[3:] - ee_pos_0[3:]) / eps if len(ee_pos_plus) > 3 else 0
        
        return J
    
    def _forward_kinematics(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        正向运动学（简化版本）
        
        返回末端执行器位置和方向
        """
        # 简化的FK实现
        # 实际应使用完整的DH变换或从Isaac Sim获取
        x = 0.4 + 0.1 * np.sin(joint_positions[0])
        y = 0.1 * np.sin(joint_positions[1])
        z = 0.4 + 0.1 * np.cos(joint_positions[1])
        
        # 简化的欧拉角
        rx = joint_positions[3] * 0.5
        ry = joint_positions[4] * 0.5
        rz = joint_positions[5] * 0.5
        
        return np.array([x, y, z, rx, ry, rz])
    
    def action_to_joint_command(
        self,
        action: Action,
        current_joint_positions: np.ndarray,
        current_joint_velocities: np.ndarray
    ) -> np.ndarray:
        """
        将VLA动作转换为关节命令
        
        Args:
            action: VLA模型输出的动作
            current_joint_positions: 当前关节位置
            current_joint_velocities: 当前关节速度
            
        Returns:
            目标关节位置
        """
        if action.action_type == "joint":
            # 直接使用关节目标
            return action.values[:7]
        
        elif action.action_type == "delta_ee":
            # 使用雅可比转换
            delta_q = self.delta_ee_to_joint_delta(
                action.position_delta,
                action.rotation_delta,
                current_joint_positions
            )
            return current_joint_positions + delta_q
        
        elif action.action_type == "absolute_ee":
            # 需要逆运动学
            # 简化实现：返回当前位置（应使用IK）
            logger.warning("Absolute EE control not fully implemented, using current position")
            return current_joint_positions
        
        else:
            logger.warning(f"Unknown action type: {action.action_type}")
            return current_joint_positions
    
    def limit_velocity(
        self,
        current_positions: np.ndarray,
        target_positions: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        限制关节速度
        
        确保目标位置的变化不超过最大速度
        """
        delta = target_positions - current_positions
        max_delta = self.config.max_velocity * dt
        
        # 按比例缩放
        scale = np.abs(delta).max() / max_delta
        if scale > 1.0:
            delta = delta / scale
        
        return current_positions + delta
    
    def reset(self) -> None:
        """重置控制器状态"""
        self.state = ControllerState()
