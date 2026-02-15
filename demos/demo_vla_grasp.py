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
    
    # 性能优化配置
    simulation_config = {
        "headless": False,
        # 降低渲染分辨率以提高性能
        "width": 1280,
        "height": 720,
        # 禁用不必要的后处理
        "renderer": "RayTracedLighting",
        # 禁用实时阴影以提高性能（可选）
        "enable_shadows": False,
    }
    
    simulation_app = SimulationApp(simulation_config)
    print("✓ SimulationApp started successfully")
    print("  Note: If performance is slow, ensure CPU is not in powersave mode:")
    print("  sudo cpupower frequency-set -g performance")
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
        
        # 7. 设置视口相机位置以便查看机器人
        self._setup_viewport_camera()
        
        logger.info("Setup complete!")
    
    def _setup_viewport_camera(self):
        """设置视口相机以便查看机器人和场景"""
        try:
            import omni.kit.viewport.utility as viewport_utils
            import numpy as np
            
            # 获取视口
            viewport = viewport_utils.get_active_viewport()
            if viewport is not None:
                # 设置相机位置：从斜上方查看机器人和桌子
                eye_pos = np.array([1.2, -0.8, 0.9])  # 相机位置
                target_pos = np.array([0.4, 0.0, 0.1])  # 看桌子中心
                
                # 使用简单的look-at计算朝向四元数
                forward = target_pos - eye_pos
                forward = forward / np.linalg.norm(forward)
                
                # 默认up向量
                world_up = np.array([0.0, 0.0, 1.0])
                
                # 计算right向量
                right = np.cross(forward, world_up)
                if np.linalg.norm(right) < 1e-6:
                    # forward和up平行，使用不同的up
                    world_up = np.array([0.0, 1.0, 0.0])
                    right = np.cross(forward, world_up)
                right = right / np.linalg.norm(right)
                
                # 重新计算up
                up = np.cross(right, forward)
                up = up / np.linalg.norm(up)
                
                # 构建旋转矩阵（从相机空间到世界空间）
                # 相机看向 -Z，up是Y，right是X
                rot_mat = np.array([
                    [right[0], right[1], right[2]],
                    [up[0], up[1], up[2]],
                    [-forward[0], -forward[1], -forward[2]]
                ])
                
                # 从旋转矩阵计算四元数 [w, x, y, z]
                trace = np.trace(rot_mat)
                if trace > 0:
                    s = 0.5 / np.sqrt(trace + 1.0)
                    w = 0.25 / s
                    x = (rot_mat[2, 1] - rot_mat[1, 2]) * s
                    y = (rot_mat[0, 2] - rot_mat[2, 0]) * s
                    z = (rot_mat[1, 0] - rot_mat[0, 1]) * s
                else:
                    # 找到最大的对角线元素
                    i = 0
                    if rot_mat[1, 1] > rot_mat[0, 0]:
                        i = 1
                    if rot_mat[2, 2] > rot_mat[i, i]:
                        i = 2
                    
                    j = (i + 1) % 3
                    k = (i + 2) % 3
                    
                    s = np.sqrt(rot_mat[i, i] - rot_mat[j, j] - rot_mat[k, k] + 1.0)
                    q = [0.0, 0.0, 0.0]
                    q[i] = s * 0.5
                    s = 0.5 / s
                    q[j] = (rot_mat[j, i] + rot_mat[i, j]) * s
                    q[k] = (rot_mat[k, i] + rot_mat[i, k]) * s
                    
                    w = (rot_mat[k, j] - rot_mat[j, k]) * s
                    x, y, z = q[0], q[1], q[2]
                
                orientation = np.array([w, x, y, z])
                
                # 设置视口相机
                viewport.set_world_pose(eye_pos, orientation)
                logger.info(f"Viewport camera set to position {eye_pos}, looking at {target_pos}")
        except Exception as e:
            logger.warning(f"Could not set viewport camera: {e}")
    
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
