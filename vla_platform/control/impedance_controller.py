# VLA Platform - Impedance Controller
"""
阻抗控制器实现
用于力/位混合控制
"""
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ImpedanceParams:
    """阻抗控制参数"""
    # 笛卡尔空间阻抗参数 [x, y, z, rx, ry, rz]
    stiffness: np.ndarray = field(
        default_factory=lambda: np.array([400, 400, 400, 40, 40, 40], dtype=np.float64)
    )
    damping: np.ndarray = field(
        default_factory=lambda: np.array([40, 40, 40, 4, 4, 4], dtype=np.float64)
    )
    inertia: np.ndarray = field(
        default_factory=lambda: np.array([1, 1, 1, 0.1, 0.1, 0.1], dtype=np.float64)
    )
    
    # 力限制
    max_force: np.ndarray = field(
        default_factory=lambda: np.array([50, 50, 50, 10, 10, 10], dtype=np.float64)
    )


class ImpedanceController:
    """
    笛卡尔空间阻抗控制器
    
    实现弹簧-阻尼-质量模型：
    F = M * a + D * v + K * (x - x_d)
    
    其中：
    - M: 惯性矩阵
    - D: 阻尼矩阵
    - K: 刚度矩阵
    - x: 当前位置
    - x_d: 目标位置
    - F: 期望力
    """
    
    def __init__(self, params: Optional[ImpedanceParams] = None):
        """
        初始化阻抗控制器
        
        Args:
            params: 阻抗参数
        """
        self.params = params or ImpedanceParams()
        
        # 状态变量
        self._target_position = np.zeros(3)
        self._target_orientation = np.array([1, 0, 0, 0])  # 四元数 [w, x, y, z]
        self._target_force = np.zeros(6)  # 可选的力偏移
        
        # 积分项（用于消除稳态误差）
        self._position_error_integral = np.zeros(6)
        self._integral_gain = 0.0  # 默认关闭积分项
        
    def set_target(
        self,
        position: np.ndarray,
        orientation: Optional[np.ndarray] = None,
        force_offset: Optional[np.ndarray] = None
    ) -> None:
        """
        设置目标位姿和力偏移
        
        Args:
            position: 目标位置 [x, y, z]
            orientation: 目标姿态四元数 [w, x, y, z]
            force_offset: 力偏移 [fx, fy, fz, tx, ty, tz]
        """
        self._target_position = np.array(position)
        if orientation is not None:
            self._target_orientation = np.array(orientation)
        if force_offset is not None:
            self._target_force = np.array(force_offset)
    
    def compute(
        self,
        current_position: np.ndarray,
        current_orientation: np.ndarray,
        current_velocity: np.ndarray,
        current_angular_velocity: np.ndarray,
        external_force: Optional[np.ndarray] = None,
        dt: float = 0.001
    ) -> np.ndarray:
        """
        计算阻抗控制力/力矩
        
        Args:
            current_position: 当前末端位置 [x, y, z]
            current_orientation: 当前姿态四元数 [w, x, y, z]
            current_velocity: 当前线速度 [vx, vy, vz]
            current_angular_velocity: 当前角速度 [wx, wy, wz]
            external_force: 外部力/力矩反馈 [fx, fy, fz, tx, ty, tz]
            dt: 时间步长
            
        Returns:
            期望力/力矩 [fx, fy, fz, tx, ty, tz]
        """
        # 计算位置误差
        position_error = current_position - self._target_position
        
        # 计算姿态误差（使用四元数）
        orientation_error = self._quaternion_error(
            current_orientation, 
            self._target_orientation
        )
        
        # 组合误差
        pose_error = np.concatenate([position_error, orientation_error])
        velocity = np.concatenate([current_velocity, current_angular_velocity])
        
        # 更新积分项
        self._position_error_integral += pose_error * dt
        self._position_error_integral = np.clip(
            self._position_error_integral, -0.1, 0.1
        )
        
        # 阻抗控制律
        # F = -K * (x - x_d) - D * v + F_d
        wrench = (
            -self.params.stiffness * pose_error
            - self.params.damping * velocity
            - self._integral_gain * self._position_error_integral
            + self._target_force
        )
        
        # 考虑外部力反馈
        if external_force is not None:
            wrench += external_force
        
        # 裁剪到力限制
        wrench = np.clip(wrench, -self.params.max_force, self.params.max_force)
        
        return wrench
    
    def _quaternion_error(
        self,
        q_current: np.ndarray,
        q_target: np.ndarray
    ) -> np.ndarray:
        """
        计算四元数姿态误差
        
        返回欧拉角误差 [ex, ey, ez]
        """
        # 四元数格式: [w, x, y, z]
        # 计算误差四元数: q_error = q_target^(-1) * q_current
        q_target_inv = np.array([q_target[0], -q_target[1], -q_target[2], -q_target[3]])
        q_error = self._quaternion_multiply(q_target_inv, q_current)
        
        # 提取误差轴角
        # 近似: 对于小角度，误差约等于2 * [qx, qy, qz]
        return 2.0 * q_error[1:4]
    
    def _quaternion_multiply(
        self,
        q1: np.ndarray,
        q2: np.ndarray
    ) -> np.ndarray:
        """四元数乘法"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def set_stiffness(self, stiffness: np.ndarray) -> None:
        """设置刚度"""
        self.params.stiffness = np.array(stiffness)
    
    def set_damping(self, damping: np.ndarray) -> None:
        """设置阻尼"""
        self.params.damping = np.array(damping)
    
    def set_integral_gain(self, gain: float) -> None:
        """设置积分增益"""
        self._integral_gain = gain
    
    def reset(self) -> None:
        """重置控制器状态"""
        self._position_error_integral = np.zeros(6)


class ForceController:
    """
    力控制器
    
    用于实现纯力控制或力/位混合控制
    """
    
    def __init__(
        self,
        kp: np.ndarray,
        ki: np.ndarray,
        max_force: np.ndarray
    ):
        """
        初始化力控制器
        
        Args:
            kp: 比例增益
            ki: 积分增益
            max_force: 最大力
        """
        self.kp = np.array(kp)
        self.ki = np.array(ki)
        self.max_force = np.array(max_force)
        
        self._force_error_integral = np.zeros(6)
        
    def compute(
        self,
        target_force: np.ndarray,
        current_force: np.ndarray,
        dt: float = 0.001
    ) -> np.ndarray:
        """
        计算力控制输出
        
        Args:
            target_force: 目标力/力矩
            current_force: 当前力/力矩（传感器反馈）
            dt: 时间步长
            
        Returns:
            控制输出
        """
        # 力误差
        force_error = target_force - current_force
        
        # 更新积分
        self._force_error_integral += force_error * dt
        self._force_error_integral = np.clip(
            self._force_error_integral, -1.0, 1.0
        )
        
        # PI控制
        output = self.kp * force_error + self.ki * self._force_error_integral
        
        # 裁剪
        output = np.clip(output, -self.max_force, self.max_force)
        
        return output
    
    def reset(self) -> None:
        """重置控制器"""
        self._force_error_integral = np.zeros(6)


class HybridForcePositionController:
    """
    力/位混合控制器
    
    在不同方向上分别实现力控制和位置控制
    """
    
    def __init__(
        self,
        impedance_params: Optional[ImpedanceParams] = None,
        force_kp: np.ndarray = np.array([1.0] * 6),
        force_ki: np.ndarray = np.array([0.1] * 6),
        selection_matrix: Optional[np.ndarray] = None
    ):
        """
        初始化混合控制器
        
        Args:
            impedance_params: 阻抗控制参数
            force_kp: 力控制比例增益
            force_ki: 力控制积分增益
            selection_matrix: 选择矩阵（1=力控制，0=位置控制）
        """
        self.impedance_controller = ImpedanceController(impedance_params)
        self.force_controller = ForceController(
            kp=force_kp,
            ki=force_ki,
            max_force=np.array([50, 50, 50, 10, 10, 10])
        )
        
        # 选择矩阵：决定每个方向使用力控制还是位置控制
        # 默认：z方向力控制，其他方向位置控制
        if selection_matrix is None:
            self.selection_matrix = np.array([0, 0, 1, 0, 0, 0])
        else:
            self.selection_matrix = np.array(selection_matrix)
    
    def compute(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        target_force: np.ndarray,
        current_position: np.ndarray,
        current_orientation: np.ndarray,
        current_velocity: np.ndarray,
        current_angular_velocity: np.ndarray,
        current_force: np.ndarray,
        dt: float = 0.001
    ) -> np.ndarray:
        """
        计算混合控制输出
        
        Returns:
            期望力/力矩
        """
        # 设置阻抗控制目标
        self.impedance_controller.set_target(target_position, target_orientation)
        
        # 计算阻抗控制输出
        impedance_output = self.impedance_controller.compute(
            current_position,
            current_orientation,
            current_velocity,
            current_angular_velocity,
            dt=dt
        )
        
        # 计算力控制输出
        force_output = self.force_controller.compute(
            target_force,
            current_force,
            dt=dt
        )
        
        # 根据选择矩阵混合输出
        output = (
            (1 - self.selection_matrix) * impedance_output +
            self.selection_matrix * force_output
        )
        
        return output
    
    def set_selection(self, selection: np.ndarray) -> None:
        """设置选择矩阵"""
        self.selection_matrix = np.array(selection)
    
    def reset(self) -> None:
        """重置控制器"""
        self.impedance_controller.reset()
        self.force_controller.reset()
