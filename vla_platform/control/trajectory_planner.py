# VLA Platform - Trajectory Planner
"""
轨迹规划模块
支持路径插值、时间最优规划等
"""
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass
import logging

from ..core.base_interfaces import TrajectoryPlanner as TrajectoryPlannerInterface

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryPoint:
    """轨迹点"""
    position: np.ndarray  # 关节位置
    velocity: Optional[np.ndarray] = None  # 关节速度
    acceleration: Optional[np.ndarray] = None  # 关节加速度
    time: float = 0.0  # 时间戳


class LinearInterpolator:
    """线性插值器"""
    
    @staticmethod
    def interpolate(
        start: np.ndarray,
        end: np.ndarray,
        num_points: int
    ) -> List[np.ndarray]:
        """
        线性插值
        
        Args:
            start: 起始配置
            end: 结束配置
            num_points: 插值点数
            
        Returns:
            插值路径点列表
        """
        t = np.linspace(0, 1, num_points)
        return [start + ti * (end - start) for ti in t]


class CubicSplineInterpolator:
    """三次样条插值器"""
    
    @staticmethod
    def interpolate(
        waypoints: List[np.ndarray],
        num_points_per_segment: int = 10,
        velocity_constraints: Optional[List[np.ndarray]] = None
    ) -> List[TrajectoryPoint]:
        """
        三次样条插值
        
        Args:
            waypoints: 路径点列表
            num_points_per_segment: 每段的插值点数
            velocity_constraints: 每个路径点的速度约束
            
        Returns:
            插值后的轨迹点列表
        """
        if len(waypoints) < 2:
            return [TrajectoryPoint(position=w) for w in waypoints]
        
        trajectory = []
        n_joints = len(waypoints[0])
        
        for i in range(len(waypoints) - 1):
            p0 = waypoints[i]
            p1 = waypoints[i + 1]
            
            # 计算边界速度
            if velocity_constraints is not None:
                v0 = velocity_constraints[i]
                v1 = velocity_constraints[i + 1]
            else:
                # 使用数值微分估计速度
                v0 = np.zeros(n_joints) if i == 0 else (p1 - waypoints[i - 1]) / 2
                v1 = np.zeros(n_joints) if i == len(waypoints) - 2 else (waypoints[i + 2] - p0) / 2
            
            # 三次Hermite样条系数
            for j in range(num_points_per_segment):
                t = j / num_points_per_segment
                
                # Hermite基函数
                h00 = 2*t**3 - 3*t**2 + 1
                h10 = t**3 - 2*t**2 + t
                h01 = -2*t**3 + 3*t**2
                h11 = t**3 - t**2
                
                position = h00*p0 + h10*v0 + h01*p1 + h11*v1
                
                # 计算速度
                dh00 = 6*t**2 - 6*t
                dh10 = 3*t**2 - 4*t + 1
                dh01 = -6*t**2 + 6*t
                dh11 = 3*t**2 - 2*t
                
                velocity = dh00*p0 + dh10*v0 + dh01*p1 + dh11*v1
                
                trajectory.append(TrajectoryPoint(
                    position=position,
                    velocity=velocity,
                    time=i + t
                ))
        
        # 添加最后一个点
        trajectory.append(TrajectoryPoint(
            position=waypoints[-1],
            velocity=np.zeros(n_joints) if velocity_constraints is None else velocity_constraints[-1],
            time=len(waypoints) - 1
        ))
        
        return trajectory


class TrajectoryPlanner(TrajectoryPlannerInterface):
    """
    轨迹规划器
    
    提供路径规划和轨迹生成功能
    """
    
    def __init__(
        self,
        max_velocity: np.ndarray,
        max_acceleration: np.ndarray,
        joint_limits_low: Optional[np.ndarray] = None,
        joint_limits_high: Optional[np.ndarray] = None
    ):
        """
        初始化轨迹规划器
        
        Args:
            max_velocity: 最大关节速度
            max_acceleration: 最大关节加速度
            joint_limits_low: 关节下限
            joint_limits_high: 关节上限
        """
        self.max_velocity = np.array(max_velocity)
        self.max_acceleration = np.array(max_acceleration)
        self.joint_limits_low = joint_limits_low
        self.joint_limits_high = joint_limits_high
        
        self.spline_interpolator = CubicSplineInterpolator()
        self.linear_interpolator = LinearInterpolator()
    
    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: Optional[List] = None
    ) -> Optional[List[np.ndarray]]:
        """
        规划从起点到终点的路径
        
        简单实现：直线路径
        高级实现可使用RRT、PRM等算法
        
        Args:
            start: 起始配置
            goal: 目标配置
            obstacles: 障碍物列表（用于碰撞检测）
            
        Returns:
            路径点列表，如果规划失败返回None
        """
        # 检查目标是否在限制范围内
        if not self._check_joint_limits(goal):
            logger.warning("Goal position exceeds joint limits")
            return None
        
        # 简单的直线路径
        # 实际应用中应该进行碰撞检测
        path = [start, goal]
        
        return path
    
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
        if len(waypoints) < 2:
            return waypoints
        
        # 计算每段需要的时间（基于最大速度）
        trajectory = []
        
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            end = waypoints[i + 1]
            
            # 计算该段的时间（基于最慢的关节）
            delta = np.abs(end - start)
            time_per_joint = delta / self.max_velocity
            segment_time = np.max(time_per_joint)
            
            # 计算需要的步数
            num_steps = max(int(segment_time / dt), 1)
            
            # 线性插值
            segment = self.linear_interpolator.interpolate(start, end, num_steps)
            trajectory.extend(segment[:-1])  # 避免重复点
        
        trajectory.append(waypoints[-1])
        
        return trajectory
    
    def generate_time_optimal_trajectory(
        self,
        waypoints: List[np.ndarray],
        dt: float = 0.01
    ) -> List[TrajectoryPoint]:
        """
        生成时间最优轨迹
        
        使用三阶时间最优轨迹生成
        
        Args:
            waypoints: 路径点
            dt: 时间步长
            
        Returns:
            时间最优轨迹点列表
        """
        if len(waypoints) < 2:
            return [TrajectoryPoint(position=w) for w in waypoints]
        
        # 使用样条插值生成平滑轨迹
        trajectory = self.spline_interpolator.interpolate(
            waypoints,
            num_points_per_segment=50
        )
        
        # 重新计算时间以满足速度/加速度约束
        self._retime_trajectory(trajectory, dt)
        
        return trajectory
    
    def _retime_trajectory(self, trajectory: List[TrajectoryPoint], dt: float) -> None:
        """
        重新计算轨迹时间以满足动力学约束
        """
        time = 0.0
        for i, point in enumerate(trajectory):
            point.time = time
            
            if i < len(trajectory) - 1:
                # 计算到下一点的时间
                delta = np.abs(trajectory[i + 1].position - point.position)
                time_needed = np.max(delta / self.max_velocity)
                time += max(time_needed, dt)
    
    def _check_joint_limits(self, positions: np.ndarray) -> bool:
        """检查关节位置是否在限制范围内"""
        if self.joint_limits_low is None or self.joint_limits_high is None:
            return True
        
        return np.all(positions >= self.joint_limits_low) and np.all(positions <= self.joint_limits_high)
    
    def clip_to_limits(self, positions: np.ndarray) -> np.ndarray:
        """裁剪到关节限制"""
        if self.joint_limits_low is not None and self.joint_limits_high is not None:
            return np.clip(positions, self.joint_limits_low, self.joint_limits_high)
        return positions


class MinJerkTrajectory:
    """
    最小急动度轨迹生成器
    
    生成平滑的五次多项式轨迹
    """
    
    @staticmethod
    def generate(
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        duration: float,
        start_vel: Optional[np.ndarray] = None,
        end_vel: Optional[np.ndarray] = None,
        dt: float = 0.01
    ) -> List[TrajectoryPoint]:
        """
        生成最小急动度轨迹
        
        Args:
            start_pos: 起始位置
            end_pos: 结束位置
            duration: 运动持续时间
            start_vel: 起始速度
            end_vel: 结束速度
            dt: 时间步长
            
        Returns:
            轨迹点列表
        """
        if start_vel is None:
            start_vel = np.zeros_like(start_pos)
        if end_vel is None:
            end_vel = np.zeros_like(end_pos)
        
        trajectory = []
        n_steps = int(duration / dt)
        
        for i in range(n_steps + 1):
            t = i * dt
            tau = t / duration  # 归一化时间
            
            # 五次多项式系数
            a0 = start_pos
            a1 = start_vel * duration
            a2 = np.zeros_like(start_pos)  # 起始加速度为0
            
            # 边界条件求解剩余系数
            a3 = 10 * (end_pos - start_pos) - 6 * start_vel * duration - 4 * end_vel * duration
            a4 = -15 * (end_pos - start_pos) + 8 * start_vel * duration + 7 * end_vel * duration
            a5 = 6 * (end_pos - start_pos) - 3 * start_vel * duration - 3 * end_vel * duration
            
            # 计算位置、速度、加速度
            position = a0 + a1*tau + a2*tau**2 + a3*tau**3 + a4*tau**4 + a5*tau**5
            velocity = (a1 + 2*a2*tau + 3*a3*tau**2 + 4*a4*tau**3 + 5*a5*tau**4) / duration
            acceleration = (2*a2 + 6*a3*tau + 12*a4*tau**2 + 20*a5*tau**3) / duration**2
            
            trajectory.append(TrajectoryPoint(
                position=position,
                velocity=velocity,
                acceleration=acceleration,
                time=t
            ))
        
        return trajectory
