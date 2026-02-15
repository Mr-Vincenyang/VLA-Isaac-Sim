# VLA Grasp Demo - Video Recording Version
"""
VLA控制抓取演示 - 视频录制版本

这个版本会保存视频文件，而不是依赖视口显示

使用方法:
    在Isaac Sim Python环境中运行:
    python demos/demo_vla_grasp_video.py --server http://your-server:8000 --output grasp_video.mp4
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
    
    # 配置为headless模式（无GUI）但启用渲染用于视频录制
    simulation_config = {
        "headless": True,  # 无GUI模式，但可以进行渲染
        "width": 640,
        "height": 480,
    }
    
    simulation_app = SimulationApp(simulation_config)
    print("✓ SimulationApp started successfully (headless mode for video recording)")
except Exception as e:
    print(f"✗ Failed to start SimulationApp: {e}")
    print("Please run this script in Isaac Sim Python environment:")
    print("  ./python.sh demos/demo_vla_grasp_video.py")
    sys.exit(1)

import numpy as np
import cv2

from vla_platform.core.config import PlatformConfig, RemoteServerConfig
from vla_platform.models.openvla_client import OpenVLAClient
from vla_platform.simulation.sim_manager import SimulationManager
from vla_platform.simulation.envs.franka_env import FrankaGraspEnv, FrankaEnvConfig
from vla_platform.simulation.sensors.sensor_manager import CameraManager, CameraConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VLAGraspVideoDemo:
    """VLA抓取演示 - 视频录制版本"""
    
    def __init__(
        self,
        server_url: str,
        output_path: str = "grasp_video.mp4",
        fps: int = 30
    ):
        """
        初始化演示
        
        Args:
            server_url: 远程VLA服务器地址
            output_path: 输出视频文件路径
            fps: 视频帧率
        """
        self.server_url = server_url
        self.output_path = output_path
        self.fps = fps
        
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
        
        # 组件
        self.sim_manager = None
        self.env = None
        self.camera = None
        self.vla_client = None
        self.video_writer = None
        
    def setup(self):
        """设置仿真环境"""
        logger.info("Setting up VLA Grasp Video Demo...")
        
        # 1. 创建仿真管理器
        logger.info("Creating simulation manager...")
        self.sim_manager = SimulationManager(self.config.simulation)
        self.sim_manager.create_world()
        
        # 2. 设置相机
        logger.info("Setting up camera...")
        camera_config = CameraConfig(
            width=640,
            height=480,
            position=[0.5, 0.0, 0.8],
            target=[0.4, 0.0, 0.1]
        )
        self.camera = CameraManager(
            prim_path="/World/Camera/overhead",
            config=camera_config,
            world=self.sim_manager.world
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
        
        # 4. 设置相机
        self.camera.setup()
        
        # 5. 初始化视频录制
        logger.info(f"Initializing video writer: {self.output_path}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.output_path, 
            fourcc, 
            self.fps, 
            (640, 480)
        )
        
        # 6. 连接VLA服务器
        logger.info(f"Connecting to VLA server: {self.server_url}")
        self.vla_client = OpenVLAClient(
            self.config.remote,
            self.config.model
        )
        
        if not self.vla_client.connect():
            logger.warning("Could not connect to VLA server. Running in simulation-only mode.")
        else:
            logger.info("Connected to VLA server!")
        
        logger.info("Setup complete!")
    
    def capture_frame(self):
        """捕获一帧图像"""
        try:
            # 从相机获取图像
            camera_data = self.camera.capture()
            rgb = camera_data.get("rgb")
            
            if rgb is not None and rgb.size > 0:
                # OpenCV使用BGR格式，需要转换
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                return frame
        except Exception as e:
            logger.warning(f"Failed to capture frame: {e}")
        
        return None
    
    def run_episode(
        self,
        instruction: str = "pick up the red block",
        max_steps: int = 100,
        save_frames: bool = True
    ) -> bool:
        """
        运行一个抓取episode并录制视频
        
        Args:
            instruction: 语言指令
            max_steps: 最大步数
            save_frames: 是否保存每一帧
            
        Returns:
            是否成功抓取
        """
        logger.info(f"Starting episode with instruction: '{instruction}'")
        
        # 重置环境
        observation = self.env.reset({"randomize_objects": True})
        
        # 捕获初始帧
        for _ in range(5):  # 等待几帧让场景稳定
            self.sim_manager.step(render=False)
            frame = self.capture_frame()
            if frame is not None and save_frames:
                self.video_writer.write(frame)
        
        for step in range(max_steps):
            # 1. 从VLA模型获取动作
            if self.vla_client.is_connected:
                try:
                    action = self.vla_client.predict(observation, instruction)
                    logger.debug(f"Step {step}: Action = {action.values}")
                except Exception as e:
                    logger.error(f"VLA prediction failed: {e}")
                    action = self._get_default_action()
            else:
                # 使用简单的启发式控制
                action = self._get_heuristic_action(observation)
            
            # 2. 应用动作
            self.env.apply_action(action)
            
            # 3. 步进仿真
            self.sim_manager.step(render=False)
            
            # 4. 捕获帧
            frame = self.capture_frame()
            if frame is not None and save_frames:
                self.video_writer.write(frame)
            
            # 5. 获取新的观测
            observation = self.env.get_observation()
            
            # 6. 检查是否成功
            if self.env.check_grasp_success():
                logger.info(f"Grasp successful at step {step}!")
                # 再录制几秒成功后的画面
                for _ in range(self.fps * 2):
                    self.sim_manager.step(render=False)
                    frame = self.capture_frame()
                    if frame is not None:
                        self.video_writer.write(frame)
                return True
            
            # 每10步打印一次进度
            if step % 10 == 0:
                logger.info(f"Step {step}/{max_steps}...")
        
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
        """启发式控制"""
        from vla_platform.core.base_interfaces import Action
        
        object_positions = self.env.get_object_positions()
        if not object_positions:
            return self._get_default_action()
        
        target_pos = object_positions[0]
        ee_pos, _ = self.env.get_ee_pose()
        
        delta = target_pos - ee_pos
        delta = np.clip(delta, -0.02, 0.02)
        
        gripper = 0.0 if np.linalg.norm(delta[:2]) < 0.02 else 1.0
        
        return Action(
            values=np.concatenate([delta, np.zeros(3), [gripper]]),
            action_type="delta_ee"
        )
    
    def run_demo(self, num_episodes: int = 1):
        """运行完整演示并录制视频"""
        instructions = [
            "pick up the red block",
        ]
        
        successes = 0
        for i in range(num_episodes):
            instruction = instructions[i % len(instructions)]
            logger.info(f"\n=== Episode {i+1}/{num_episodes} ===")
            
            if self.run_episode(instruction):
                successes += 1
        
        logger.info(f"\n=== Demo Complete ===")
        logger.info(f"Success rate: {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)")
        logger.info(f"Video saved to: {self.output_path}")
    
    def cleanup(self):
        """清理资源"""
        if self.video_writer is not None:
            self.video_writer.release()
            logger.info(f"Video saved: {self.output_path}")
        
        if self.vla_client:
            self.vla_client.close()
        if self.sim_manager:
            self.sim_manager.cleanup()


def main():
    parser = argparse.ArgumentParser(description="VLA Grasp Video Demo")
    parser.add_argument(
        "--server",
        type=str,
        default="http://localhost:8000",
        help="VLA server URL"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="grasp_video.mp4",
        help="Output video file path"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video FPS"
    )
    
    args = parser.parse_args()
    
    demo = VLAGraspVideoDemo(
        server_url=args.server,
        output_path=args.output,
        fps=args.fps
    )
    
    try:
        demo.setup()
        demo.run_demo(num_episodes=args.episodes)
    finally:
        demo.cleanup()
        
    print(f"\n✓ Video saved to: {args.output}")
    print(f"  You can view it with: ffplay {args.output}  or  vlc {args.output}")


if __name__ == "__main__":
    main()
