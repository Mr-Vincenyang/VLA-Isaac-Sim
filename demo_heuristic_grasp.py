#!/usr/bin/env python3
"""
VLA演示 - 启发式控制版本（不依赖远程VLA服务器）
使用简单的规则控制机械臂抓取
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# 启动SimulationApp
print("Starting Isaac Sim...")
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})
print("✓ Isaac Sim started")

import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入模块
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.sensors.camera import Camera

class SimpleGraspDemo:
    """简单抓取演示 - 使用启发式控制"""
    
    def __init__(self):
        self.world = None
        self.franka = None
        self.cube = None
        self.camera = None
        
    def setup(self):
        """设置场景"""
        logger.info("Setting up scene...")
        
        # 创建世界
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        
        # 添加Franka机器人
        self.franka = self.world.scene.add(
            Franka(prim_path="/World/Franka", name="franka")
        )
        
        # 添加红色方块
        self.cube = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/Cube",
                name="cube",
                position=np.array([0.5, 0.0, 0.05]),
                scale=np.array([0.05, 0.05, 0.05]),
                color=np.array([1.0, 0.0, 0.0])
            )
        )
        
        # 添加相机
        self.camera = Camera(
            prim_path="/World/Camera",
            position=np.array([0.8, 0.8, 0.8]),
            frequency=30,
            resolution=(512, 512)
        )
        self.world.scene.add(self.camera)
        self.camera.initialize()
        
        # 重置世界
        self.world.reset()
        
        logger.info("✓ Scene setup complete")
        
    def get_cube_position(self):
        """获取方块位置"""
        return self.cube.get_world_pose()[0]
    
    def get_ee_position(self):
        """获取末端执行器位置（简化为使用关节位置推算）"""
        # 对于演示目的，使用固定轨迹
        # 实际应用中应该使用正向运动学计算
        try:
            # 获取Franka的基座位置
            base_pos = self.franka.get_world_pose()[0]
            # 末端执行器大约在基座前方0.5m，上方0.4m
            return base_pos + np.array([0.5, 0.0, 0.4])
        except:
            return np.array([0.5, 0.0, 0.5])
    
    def move_to_position(self, target_pos, gripper_open=True):
        """移动到目标位置（简单的PD控制）"""
        ee_pos = self.get_ee_position()
        
        # 计算位置误差
        error = target_pos - ee_pos
        
        # 简单的比例控制
        action = np.zeros(7)
        action[0:3] = error * 0.5  # 位置控制（比例增益0.5）
        action[6] = 1.0 if gripper_open else 0.0  # 夹爪
        
        # 应用动作
        self.franka.apply_action(action)
        
        return np.linalg.norm(error)
    
    def run_grasp_sequence(self):
        """运行抓取序列"""
        logger.info("Starting grasp sequence...")
        
        # 获取方块位置
        cube_pos = self.get_cube_position()
        logger.info(f"Cube position: {cube_pos}")
        
        # 阶段1：移动到方块上方
        logger.info("Phase 1: Moving above cube...")
        above_pos = cube_pos + np.array([0, 0, 0.15])  # 方块上方15cm
        
        for step in range(100):
            error = self.move_to_position(above_pos, gripper_open=True)
            self.world.step(render=True)
            
            if error < 0.02:  # 到达目标
                logger.info(f"Reached above position at step {step}")
                break
        
        # 阶段2：下降并抓取
        logger.info("Phase 2: Descending and grasping...")
        grasp_pos = cube_pos + np.array([0, 0, 0.02])  # 方块上方2cm
        
        for step in range(100):
            error = self.move_to_position(grasp_pos, gripper_open=True)
            self.world.step(render=True)
            
            if error < 0.02:
                logger.info(f"Reached grasp position at step {step}")
                break
        
        # 阶段3：关闭夹爪
        logger.info("Phase 3: Closing gripper...")
        for step in range(50):
            action = np.zeros(7)
            action[6] = 0.0  # 关闭夹爪
            self.franka.apply_action(action)
            self.world.step(render=True)
        
        # 阶段4：提起
        logger.info("Phase 4: Lifting...")
        lift_pos = cube_pos + np.array([0, 0, 0.3])  # 提起30cm
        
        for step in range(150):
            error = self.move_to_position(lift_pos, gripper_open=False)
            self.world.step(render=True)
            
            if error < 0.02:
                logger.info(f"Reached lift position at step {step}")
                break
        
        logger.info("✓ Grasp sequence complete!")
        
    def run(self, num_episodes=1):
        """运行演示"""
        logger.info(f"Running {num_episodes} episode(s)...")
        
        for episode in range(num_episodes):
            logger.info(f"\n{'='*50}")
            logger.info(f"Episode {episode + 1}/{num_episodes}")
            logger.info(f"{'='*50}")
            
            # 随机化方块位置
            new_pos = np.array([
                0.4 + np.random.rand() * 0.2,  # x: 0.4-0.6
                -0.1 + np.random.rand() * 0.2,  # y: -0.1-0.1
                0.05
            ])
            self.cube.set_world_pose(position=new_pos)
            self.world.reset()
            
            # 等待稳定
            for _ in range(30):
                self.world.step(render=True)
            
            # 运行抓取
            self.run_grasp_sequence()
            
            # 等待观察
            for _ in range(60):
                self.world.step(render=True)
        
        logger.info("\n✓ All episodes complete!")

def main():
    demo = SimpleGraspDemo()
    demo.setup()
    demo.run(num_episodes=1)
    
    # 保持运行
    logger.info("\nDemo finished. Press Ctrl+C to exit.")
    try:
        while True:
            demo.world.step(render=True)
            time.sleep(0.01)
    except KeyboardInterrupt:
        logger.info("\nExiting...")
    
    simulation_app.close()

if __name__ == "__main__":
    main()
