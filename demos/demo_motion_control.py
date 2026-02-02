# Motion Control Demo
"""
运动控制功能演示

演示内容:
1. 轨迹跟踪控制
2. 阻抗控制
3. 最小急动度轨迹生成

使用方法:
    在Isaac Sim Python环境中运行:
    python demos/demo_motion_control.py
"""
import argparse
import time
import logging
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from vla_platform.simulation.sim_manager import SimulationManager, check_isaac_sim_available
from vla_platform.simulation.envs.franka_env import FrankaGraspEnv, FrankaEnvConfig
from vla_platform.control.motion_controller import MotionController, PDController
from vla_platform.control.trajectory_planner import (
    TrajectoryPlanner,
    MinJerkTrajectory,
    TrajectoryPoint
)
from vla_platform.control.impedance_controller import ImpedanceController, ImpedanceParams
from vla_platform.core.config import ControlConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MotionControlDemo:
    """运动控制演示"""
    
    def __init__(self, headless: bool = False):
        self.headless = headless
        self.sim_manager = None
        self.env = None
        
    def setup(self):
        """设置仿真环境"""
        if not check_isaac_sim_available():
            raise RuntimeError("Isaac Sim not available")
        
        # 创建仿真管理器
        from vla_platform.core.config import SimulationConfig
        sim_config = SimulationConfig(
            physics_dt=1.0/240.0,
            rendering_dt=1.0/60.0,
            headless=self.headless
        )
        
        self.sim_manager = SimulationManager(sim_config)
        self.sim_manager.create_world()
        
        # 创建Franka环境
        env_config = FrankaEnvConfig(num_objects=0)  # 不需要物体
        self.env = FrankaGraspEnv(
            sim_manager=self.sim_manager,
            config=env_config
        )
        self.env.setup()
        
        logger.info("Motion control demo setup complete")
    
    def demo_pd_control(self):
        """演示PD关节控制"""
        logger.info("\n=== PD Control Demo ===")
        
        # 创建PD控制器
        config = ControlConfig()
        controller = MotionController(config)
        
        # 目标关节位置序列
        targets = [
            np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.785]),
            np.array([0.5, -0.3, 0.3, -1.8, 0.2, 1.2, 0.5]),
            np.array([-0.5, -0.7, -0.3, -2.2, -0.2, 1.8, 1.0]),
            np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]),  # 回到初始
        ]
        
        for i, target in enumerate(targets):
            logger.info(f"Moving to target {i+1}/{len(targets)}")
            
            for step in range(200):
                current_pos = self.env.get_joint_positions()[:7]
                current_vel = self.env.get_joint_velocities()[:7]
                
                # 计算控制输出
                torque = controller.compute_joint_control(
                    current_pos, current_vel, target
                )
                
                # 使用位置控制（简化）
                new_pos = current_pos + 0.01 * (target - current_pos)
                
                # 设置关节位置
                full_pos = self.env.get_joint_positions()
                full_pos[:7] = new_pos
                self.env._robot.set_joint_positions(full_pos)
                
                self.sim_manager.step()
                
                # 检查是否到达
                if np.linalg.norm(current_pos - target) < 0.01:
                    break
            
            time.sleep(0.5)
        
        logger.info("PD control demo complete")
    
    def demo_trajectory_following(self):
        """演示轨迹跟踪"""
        logger.info("\n=== Trajectory Following Demo ===")
        
        # 创建轨迹规划器
        max_velocity = np.ones(7) * 1.5
        max_acceleration = np.ones(7) * 3.0
        
        planner = TrajectoryPlanner(
            max_velocity=max_velocity,
            max_acceleration=max_acceleration
        )
        
        # 定义路径点
        waypoints = [
            np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]),
            np.array([0.3, -0.5, 0.2, -2.0, 0.1, 1.3, 0.6]),
            np.array([0.5, -0.3, 0.4, -1.8, 0.3, 1.1, 0.4]),
            np.array([0.3, -0.5, 0.2, -2.0, 0.1, 1.3, 0.6]),
            np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]),
        ]
        
        # 生成轨迹
        trajectory = planner.interpolate(waypoints, dt=0.02)
        logger.info(f"Generated trajectory with {len(trajectory)} points")
        
        # 跟踪轨迹
        for i, target in enumerate(trajectory):
            if i % 10 == 0:
                logger.info(f"Trajectory point {i}/{len(trajectory)}")
            
            # 设置关节位置
            full_pos = self.env.get_joint_positions()
            full_pos[:7] = target
            self.env._robot.set_joint_positions(full_pos)
            
            # 步进仿真
            for _ in range(5):  # 5个物理步进每个轨迹点
                self.sim_manager.step()
        
        logger.info("Trajectory following demo complete")
    
    def demo_min_jerk_trajectory(self):
        """演示最小急动度轨迹"""
        logger.info("\n=== Minimum Jerk Trajectory Demo ===")
        
        # 起点和终点
        start_pos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        end_pos = np.array([0.5, -0.3, 0.5, -1.5, 0.5, 1.0, 0.3])
        
        # 生成最小急动度轨迹
        duration = 3.0
        trajectory = MinJerkTrajectory.generate(
            start_pos=start_pos,
            end_pos=end_pos,
            duration=duration,
            dt=0.01
        )
        
        logger.info(f"Generated min-jerk trajectory with {len(trajectory)} points")
        
        # 执行轨迹
        for i, point in enumerate(trajectory):
            if i % 50 == 0:
                logger.info(f"Position: {point.position[:3]}")
                logger.info(f"Velocity: {point.velocity[:3] if point.velocity is not None else 'N/A'}")
            
            full_pos = self.env.get_joint_positions()
            full_pos[:7] = point.position
            self.env._robot.set_joint_positions(full_pos)
            
            self.sim_manager.step()
        
        logger.info("Minimum jerk trajectory demo complete")
    
    def demo_impedance_control(self):
        """演示阻抗控制"""
        logger.info("\n=== Impedance Control Demo ===")
        
        # 创建阻抗控制器
        params = ImpedanceParams(
            stiffness=np.array([200, 200, 200, 20, 20, 20]),
            damping=np.array([20, 20, 20, 2, 2, 2])
        )
        controller = ImpedanceController(params)
        
        # 获取当前末端位置
        current_ee_pos, current_ee_quat = self.env.get_ee_pose()
        
        # 设置目标（当前位置上移0.1m）
        target_pos = current_ee_pos.copy()
        target_pos[2] += 0.1
        
        controller.set_target(
            position=target_pos,
            orientation=current_ee_quat
        )
        
        logger.info(f"Target position: {target_pos}")
        
        for step in range(500):
            # 获取当前状态
            ee_pos, ee_quat = self.env.get_ee_pose()
            joint_vel = self.env.get_joint_velocities()[:7]
            
            # 简化：假设末端速度
            ee_vel = np.zeros(3)  # 应该从机器人动力学计算
            ee_angular_vel = np.zeros(3)
            
            # 计算阻抗控制力
            wrench = controller.compute(
                current_position=ee_pos,
                current_orientation=ee_quat,
                current_velocity=ee_vel,
                current_angular_velocity=ee_angular_vel,
                dt=1.0/240.0
            )
            
            if step % 100 == 0:
                error = target_pos - ee_pos
                logger.info(f"Step {step}: Position error = {error}, Wrench = {wrench[:3]}")
            
            # 简化：直接移动末端执行器（实际应用需要逆运动学）
            delta = 0.001 * wrench[:3]
            # 这里应该转换为关节空间控制
            
            self.sim_manager.step()
        
        logger.info("Impedance control demo complete")
    
    def run_all_demos(self):
        """运行所有演示"""
        logger.info("\n" + "="*50)
        logger.info("Starting Motion Control Demos")
        logger.info("="*50)
        
        self.demo_pd_control()
        time.sleep(1)
        
        self.demo_min_jerk_trajectory()
        time.sleep(1)
        
        self.demo_trajectory_following()
        time.sleep(1)
        
        self.demo_impedance_control()
        
        logger.info("\n" + "="*50)
        logger.info("All demos complete!")
        logger.info("="*50)
    
    def cleanup(self):
        """清理资源"""
        if self.sim_manager:
            self.sim_manager.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Motion Control Demo")
    parser.add_argument(
        "--demo",
        type=str,
        choices=["all", "pd", "trajectory", "minjerk", "impedance"],
        default="all",
        help="Which demo to run"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode"
    )
    
    args = parser.parse_args()
    
    demo = MotionControlDemo(headless=args.headless)
    
    try:
        demo.setup()
        
        if args.demo == "all":
            demo.run_all_demos()
        elif args.demo == "pd":
            demo.demo_pd_control()
        elif args.demo == "trajectory":
            demo.demo_trajectory_following()
        elif args.demo == "minjerk":
            demo.demo_min_jerk_trajectory()
        elif args.demo == "impedance":
            demo.demo_impedance_control()
    finally:
        demo.cleanup()


if __name__ == "__main__":
    main()
