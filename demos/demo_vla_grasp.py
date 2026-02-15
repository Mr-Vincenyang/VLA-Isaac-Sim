# VLA Grasp Demo
"""
完整的VLA控制抓取演示

演示流程:
1. 初始化Isaac Sim环境
2. 连接远程VLA服务器
3. 执行语言指令控制的抓取任务

使用方法:
    在Isaac Sim Python环境中运行:
    python demos/demo_vla_grasp.py --server http://your-server:8000
"""
import argparse
import time
import logging
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 首先启动Isaac Sim SimulationApp（必须在其他isaacsim模块之前）
print("Starting Isaac Sim SimulationApp...")
try:
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": False})
    print("✓ SimulationApp started successfully")
except Exception as e:
    print(f"✗ Failed to start SimulationApp: {e}")
    print("Please run this script in Isaac Sim Python environment:")
    print("  ./python.sh demos/demo_vla_grasp.py")
    sys.exit(1)

import numpy as np

from vla_platform.core.config import PlatformConfig, RemoteServerConfig
from vla_platform.models.openvla_client import OpenVLAClient
from vla_platform.simulation.sim_manager import SimulationManager, check_isaac_sim_available
from vla_platform.simulation.envs.franka_env import FrankaGraspEnv, FrankaEnvConfig
from vla_platform.simulation.sensors.sensor_manager import CameraManager, CameraConfig
from vla_platform.control.motion_controller import MotionController

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VLAGraspDemo:
    """VLA抓取演示"""
    
    def __init__(
        self,
        server_url: str,
        headless: bool = False
    ):
        """
        初始化演示
        
        Args:
            server_url: 远程VLA服务器地址
            headless: 是否无头模式运行
        """
        self.server_url = server_url
        self.headless = headless
        
        # 解析服务器URL
        if "://" in server_url:
            parts = server_url.split("://")[1].split(":")
            host = parts[0]
            port = int(parts[1]) if len(parts) > 1 else 8000
        else:
            host = "localhost"
            port = 8000
        
        # 配置
        self.config = PlatformConfig(
            remote=RemoteServerConfig(
                host=host,
                port=port,
                protocol="http"
            )
        )
        
        # 组件（延迟初始化）
        self.sim_manager = None
        self.env = None
        self.camera = None
        self.vla_client = None
        self.motion_controller = None
        
    def setup(self):
        """设置仿真环境和连接"""
        logger.info("Setting up VLA Grasp Demo...")
        
        # 检查Isaac Sim（尝试导入SimulationApp验证）
        try:
            from isaacsim import SimulationApp
            logger.info("✓ Isaac Sim environment verified")
        except ImportError:
            raise RuntimeError(
                "Isaac Sim not available. Please run this script in Isaac Sim environment.\n"
                "Use: ./python.sh demo_vla_grasp.py"
            )
        
        # 1. 创建仿真管理器
        logger.info("Creating simulation manager...")
        self.sim_manager = SimulationManager(self.config.simulation)
        self.sim_manager.create_world()
        
        # 2. 设置相机（在world创建后，传入world实例）
        logger.info("Setting up camera...")
        camera_config = CameraConfig(
            width=224,
            height=224,
            position=[0.5, 0.0, 0.8],
            target=[0.4, 0.0, 0.1]
        )
        self.camera = CameraManager(
            prim_path="/World/Camera/overhead",
            config=camera_config,
            world=self.sim_manager.world  # Fix: 传入world实例以便添加到scene
        )
        
        # 3. 创建Franka环境
        logger.info("Creating Franka environment...")
        env_config = FrankaEnvConfig(
            num_objects=1,
            object_colors=[[1.0, 0.0, 0.0]]  # 红色方块
        )
        self.env = FrankaGraspEnv(
            sim_manager=self.sim_manager,
            config=env_config,
            camera_manager=self.camera
        )
        self.env.setup()
        
        # 4. 设置相机（需要在环境setup后）
        self.camera.setup()
        
        # 5. 连接VLA服务器
        logger.info(f"Connecting to VLA server: {self.server_url}")
        self.vla_client = OpenVLAClient(
            self.config.remote,
            self.config.model
        )
        
        if not self.vla_client.connect():
            logger.warning("Could not connect to VLA server. Running in simulation-only mode.")
        else:
            logger.info("Connected to VLA server!")
        
        # 6. 创建运动控制器
        self.motion_controller = MotionController(self.config.control)
        
        logger.info("Setup complete!")
    
    def run_episode(
        self,
        instruction: str = "pick up the red block",
        max_steps: int = 100,
        render: bool = True
    ) -> bool:
        """
        运行一个抓取episode
        
        Args:
            instruction: 语言指令
            max_steps: 最大步数
            render: 是否渲染
            
        Returns:
            是否成功抓取
        """
        logger.info(f"Starting episode with instruction: '{instruction}'")
        
        # 重置环境
        observation = self.env.reset({"randomize_objects": True})
        
        for step in range(max_steps):
            # 1. 从VLA模型获取动作
            if self.vla_client.is_connected:
                try:
                    action = self.vla_client.predict(observation, instruction)
                    logger.debug(f"Step {step}: Action = {action.values}")
                except Exception as e:
                    logger.error(f"VLA prediction failed: {e}")
                    # 使用默认动作
                    action = self._get_default_action()
            else:
                # 使用简单的启发式控制作为后备
                action = self._get_heuristic_action(observation)
            
            # 2. 应用动作
            self.env.apply_action(action)
            
            # 3. 步进仿真
            self.sim_manager.step(render=render)
            
            # 4. 获取新的观测
            observation = self.env.get_observation()
            
            # 5. 检查是否成功
            if self.env.check_grasp_success():
                logger.info(f"Grasp successful at step {step}!")
                return True
            
            # 短暂延迟以便观察
            if render:
                time.sleep(0.01)
        
        logger.info("Episode ended without successful grasp")
        return False
    
    def _get_default_action(self):
        """获取默认动作"""
        from vla_platform.core.base_interfaces import Action
        return Action(
            values=np.zeros(7),
            action_type="delta_ee"
        )
    
    def _get_heuristic_action(self, observation):
        """
        启发式控制（当VLA不可用时）
        
        简单的向下移动并抓取
        """
        from vla_platform.core.base_interfaces import Action
        
        # 获取物体位置
        object_positions = self.env.get_object_positions()
        if not object_positions:
            return self._get_default_action()
        
        target_pos = object_positions[0]
        ee_pos, _ = self.env.get_ee_pose()
        
        # 计算增量
        delta = target_pos - ee_pos
        delta = np.clip(delta, -0.02, 0.02)  # 限制步长
        
        # 如果接近目标，尝试抓取
        gripper = 0.0 if np.linalg.norm(delta[:2]) < 0.02 else 1.0
        
        return Action(
            values=np.concatenate([delta, np.zeros(3), [gripper]]),
            action_type="delta_ee"
        )
    
    def run_demo(self, num_episodes: int = 3):
        """运行完整演示"""
        instructions = [
            "pick up the red block",
            "grasp the object on the table",
            "pick up the cube"
        ]
        
        successes = 0
        for i in range(num_episodes):
            instruction = instructions[i % len(instructions)]
            logger.info(f"\n=== Episode {i+1}/{num_episodes} ===")
            
            if self.run_episode(instruction):
                successes += 1
        
        logger.info(f"\n=== Demo Complete ===")
        logger.info(f"Success rate: {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)")
    
    def cleanup(self):
        """清理资源"""
        if self.vla_client:
            self.vla_client.close()
        if self.sim_manager:
            self.sim_manager.cleanup()


def main():
    parser = argparse.ArgumentParser(description="VLA Grasp Demo")
    parser.add_argument(
        "--server",
        type=str,
        default="http://localhost:8000",
        help="VLA server URL"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode"
    )
    
    args = parser.parse_args()
    
    demo = VLAGraspDemo(
        server_url=args.server,
        headless=args.headless
    )
    
    try:
        demo.setup()
        demo.run_demo(num_episodes=args.episodes)
    finally:
        demo.cleanup()


if __name__ == "__main__":
    main()
