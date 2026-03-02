# VLA Grasp Demo - Video Recording Version (New)
"""
VLA控制抓取演示 - 视频录制版本
使用 demo_motion_control.py 的成功视频录制模式

使用方法:
    # 1. 先启动VLA服务器 (在另一个终端):
    python server/server_deploy.py --model openvla/openvla-7b --port 8000
    
    # 2. 运行demo:
    cd isaac-sim && ./python.sh ../VLA/demos/demo_vla_grasp_video_new.py --server http://localhost:8000 --output video.mp4
    或者
    ./start_local.sh --demo grasp --record --headless --server http://localhost:8000
"""
import argparse
import time
import logging
import os
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 创建输出目录
output_dir = "/home/vincent/Desktop/code/VLA/output"
os.makedirs(output_dir, exist_ok=True)

# 首先启动Isaac Sim
print("Starting Isaac Sim SimulationApp...")
try:
    from isaacsim import SimulationApp
    
    simulation_config = {
        "headless": True,
        "width": 1280,
        "height": 720,
        "renderer": "RayTracedLighting",
        "enable_shadows": False,
    }
    
    simulation_app = SimulationApp(simulation_config)
    print("✓ SimulationApp started successfully")
except Exception as e:
    print(f"✗ Failed to start SimulationApp: {e}")
    print("Please run this script in Isaac Sim Python environment:")
    print("  ./python.sh demos/demo_vla_grasp_video_new.py")
    sys.exit(1)

import numpy as np
import cv2

# Isaac Sim 模块
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.robot.manipulators.examples.franka import Franka
import omni.replicator.core as rep
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.utils.nucleus import get_assets_root_path

# VLA imports
from vla_platform.core.base_interfaces import Observation, Action
from vla_platform.models.openvla_client import OpenVLAClient
from vla_platform.core.config import PlatformConfig, RemoteServerConfig, VLAModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_observation(rgb_annotator, franka):
    """创建VLA所需的观测"""
    # 获取相机图像
    try:
        data = rgb_annotator.get_data()
        if data is not None and len(data.shape) == 3 and data.shape[2] >= 3:
            image = data[:,:,:3].astype(np.uint8)
            # 调试：保存第一帧图像用于检查
            if not hasattr(create_observation, '_debug_saved'):
                create_observation._debug_saved = True
                debug_path = os.path.join(output_dir, "debug_first_frame.png")
                cv2.imwrite(debug_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                logger.info(f"Debug: First frame saved to {debug_path}, shape: {image.shape}, min: {image.min()}, max: {image.max()}")
        else:
            image = np.zeros((720, 1280, 3), dtype=np.uint8)
            logger.warning("Invalid RGB data shape, using blank image")
    except Exception as e:
        logger.warning(f"Error getting RGB data: {e}")
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # 获取机器人状态
    try:
        joint_positions = franka.get_joint_positions()
        if joint_positions is None:
            joint_positions = np.zeros(9)
    except Exception as e:
        logger.warning(f"Error getting joint positions: {e}")
        joint_positions = np.zeros(9)
    
    # 获取末端执行器位置
    try:
        if hasattr(franka, 'gripper') and franka.gripper is not None:
            ee_pos, ee_quat = franka.gripper.get_world_pose()
        else:
            ee_pos = np.zeros(3)
            ee_quat = np.array([1, 0, 0, 0])
    except Exception as e:
        logger.warning(f"Error getting EE pose: {e}")
        ee_pos = np.zeros(3)
        ee_quat = np.array([1, 0, 0, 0])
    
    return Observation(
        image=image,
        depth=None,
        joint_positions=joint_positions[:7] if len(joint_positions) >= 7 else joint_positions,
        joint_velocities=np.zeros(7),
        ee_position=ee_pos,
        ee_orientation=ee_quat,
        gripper_state=0.0
    )


def apply_action(franka, action: Action):
    """应用动作到机器人"""
    if action.action_type == "joint":
        # 关节空间控制
        if action.values is not None and len(action.values) >= 7:
            # 添加夹爪位置
            gripper = action.values[6] if len(action.values) > 6 else 0.04
            joint_pos = np.array(list(action.values[:7]) + [gripper, gripper])
            try:
                franka.set_joint_positions(joint_pos)
            except Exception as e:
                logger.warning(f"Error setting joint positions: {e}")
    elif action.action_type == "delta_ee":
        # 末端执行器增量控制 - 简化处理
        pass
    elif action.action_type == "absolute_ee":
        # 绝对末端位置控制 - 简化处理
        pass


def get_heuristic_action(franka, target_object, step, max_steps):
    """启发式抓取控制 - 当VLA服务器不可用时使用"""
    try:
        # 获取物体位置
        obj_pos, _ = target_object.get_world_pose()
        
        # 获取末端执行器位置
        if hasattr(franka, 'gripper') and franka.gripper is not None:
            ee_pos, _ = franka.gripper.get_world_pose()
        else:
            ee_pos = np.zeros(3)
        
        # 计算距离
        dist = np.linalg.norm(obj_pos - ee_pos)
        
        # 状态机 - 基于步数
        progress = step / max_steps
        
        if progress < 0.2:
            # 初始阶段 - 移动到物体上方
            target_joints = np.array([0.25, -0.4, 0.15, -1.6, 0.1, 1.4, 0.5, 0.04, 0.04])
        elif progress < 0.5:
            # 下降阶段 - 靠近物体
            target_joints = np.array([0.3, -0.3, 0.1, -1.4, 0.08, 1.2, 0.4, 0.02, 0.02])
        elif progress < 0.7:
            # 抓取阶段 - 闭合夹爪
            target_joints = np.array([0.32, -0.25, 0.08, -1.3, 0.06, 1.0, 0.35, 0.0, 0.0])
        elif progress < 0.85:
            # 抬起阶段
            target_joints = np.array([0.2, -0.5, 0.15, -1.5, 0.1, 1.3, 0.5, 0.0, 0.0])
        else:
            # 移动到展示位置
            target_joints = np.array([-0.3, -0.4, 0.2, -1.4, -0.1, 1.2, 0.6, 0.0, 0.0])
        
        return Action(values=target_joints[:7], action_type="joint")
    except Exception as e:
        logger.warning(f"Error in heuristic action: {e}")
        return Action(values=np.zeros(7), action_type="joint")


def run_grasp_episode(franka, target_object, rgb_annotator, video_writer, vla_client, instruction, max_steps=30):
    """运行一个抓取episode - 使用demo_motion_control.py的视频录制模式"""
    global world
    
    print(f"\n=== Episode: {instruction} ===")
    
    # 重置物体位置
    try:
        target_object.set_world_pose(position=np.array([0.4, 0.0, 0.08]))
    except Exception as e:
        logger.warning(f"Error resetting object position: {e}")
    
    # 等待几帧让场景稳定
    for _ in range(10):
        world.step(render=True)
        rep.orchestrator.step()
    
    frame_count = 0
    
    # 记录初始帧 - 确保视频写入器工作
    print("Recording initial frames...")
    for i in range(10):
        world.step(render=True)
        rep.orchestrator.step()
        
        # 录制视频 - 使用和demo_motion_control.py完全相同的模式
        try:
            data = rgb_annotator.get_data()
            if data is not None and len(data.shape) == 3 and data.shape[2] >= 3:
                frame = cv2.cvtColor(data[:,:,:3].astype(np.uint8), cv2.COLOR_RGB2BGR)
                if video_writer.isOpened():
                    video_writer.write(frame)
                    frame_count += 1
                else:
                    logger.warning("Video writer not opened!")
            else:
                logger.warning(f"Invalid data shape: {data.shape if data is not None else None}")
        except Exception as e:
            logger.warning(f"Error recording frame: {e}")
    
    print(f"Initial frames recorded: {frame_count}")
    
    # 主循环
    for step in range(max_steps):
        # 1. 获取观测
        observation = create_observation(rgb_annotator, franka)
        
        # 2. 调用VLA模型获取动作
        if vla_client is not None and hasattr(vla_client, 'is_connected') and vla_client.is_connected:
            try:
                action = vla_client.predict(observation, instruction)
                # 调试：打印图像信息
                if step % 20 == 0:
                    print(f"Step {step}: VLA action = {action.values[:7] if action.values is not None else 'None'}")
                    print(f"  -> Image shape: {observation.image.shape}, min: {observation.image.min()}, max: {observation.image.max()}")
            except Exception as e:
                logger.warning(f"Step {step}: VLA prediction failed: {e}")
                action = get_heuristic_action(franka, target_object, step, max_steps)
        else:
            # 无VLA服务器，使用启发式控制
            action = get_heuristic_action(franka, target_object, step, max_steps)
            if step % 20 == 0:
                print(f"Step {step}: Heuristic action")
        
        # 3. 应用动作
        apply_action(franka, action)
        
        # 4. 步进仿真
        world.step(render=True)
        rep.orchestrator.step()
        
        # 5. 录制视频 - 使用和demo_motion_control.py完全相同的模式
        try:
            data = rgb_annotator.get_data()
            if data is not None and len(data.shape) == 3 and data.shape[2] >= 3:
                frame = cv2.cvtColor(data[:,:,:3].astype(np.uint8), cv2.COLOR_RGB2BGR)
                if video_writer.isOpened():
                    video_writer.write(frame)
                    frame_count += 1
            else:
                logger.warning(f"Invalid data at step {step}")
        except Exception as e:
            logger.warning(f"Error recording frame at step {step}: {e}")
        
        # 6. 检查是否成功抓取
        try:
            obj_pos, _ = target_object.get_world_pose()
            if obj_pos[2] > 0.15:  # 物体被抬起
                print(f"✓ Grasp success at step {step}!")
                # 再录制几秒
                for _ in range(60):
                    world.step(render=True)
                    rep.orchestrator.step()
                    try:
                        data = rgb_annotator.get_data()
                        if data is not None and len(data.shape) == 3:
                            frame = cv2.cvtColor(data[:,:,:3].astype(np.uint8), cv2.COLOR_RGB2BGR)
                            if video_writer.isOpened():
                                video_writer.write(frame)
                                frame_count += 1
                    except Exception as e:
                        pass
                return True, frame_count
        except Exception as e:
            pass
        
        if step % 30 == 0:
            print(f"Step {step}/{max_steps}, frames: {frame_count}")
    
    return False, frame_count


def main():
    global world
    
    parser = argparse.ArgumentParser(description="VLA Grasp Video Demo - New Version")
    parser.add_argument("--server", type=str, default="http://localhost:8000", help="VLA server URL")
    parser.add_argument("--output", type=str, default="vla_grasp_demo.mp4", help="Output video file")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    args = parser.parse_args()
    
    video_path = os.path.join(output_dir, args.output)
    
    # ===== 1. 创建世界 =====
    print("\n[1/7] Creating world...")
    world = World(physics_dt=1/60, rendering_dt=1/60)
    world.scene.add_default_ground_plane()
    print("✓ World created")
    
    # ===== 2. 添加Franka机器人 =====
    print("\n[2/7] Adding Franka robot...")
    franka = Franka(
        prim_path="/World/Franka",
        name="franka",
        position=np.array([0.0, 0.0, 0.0])
    )
    world.scene.add(franka)
    print("✓ Franka robot added")
    
    # ===== 3. 添加桌子 =====
    print("\n[3/7] Adding table...")
    table_pos = [0.4, 0.0, 0.025]
    table = DynamicCuboid(
        prim_path="/World/Table",
        name="table",
        position=np.array(table_pos),
        size=1.0,
        scale=np.array([0.6, 1.0, 0.05]),
        color=np.array([0.5, 0.35, 0.2]),
        mass=1000.0
    )
    world.scene.add(table)
    print("✓ Table added")
    
    # ===== 4. 添加目标物体 =====
    print("\n[4/7] Adding target object...")
    object_pos = [0.4, 0.0, 0.08]
    target_object = DynamicCuboid(
        prim_path="/World/Object_0",
        name="object_0",
        position=np.array(object_pos),
        size=0.05,
        color=np.array([1.0, 0.0, 0.0]),  # 红色
        mass=0.1
    )
    world.scene.add(target_object)
    print("✓ Target object added")
    
    # ===== 5. 设置Replicator相机 - 使用demo_motion_control.py的模式 =====
    print("\n[5/7] Setting up camera...")
    camera_pos = (1.5, -1.0, 1.2)
    look_at = (0.4, 0.0, 0.2)
    
    # 使用和demo_motion_control.py完全相同的相机设置
    camera = rep.create.camera(position=camera_pos, look_at=look_at)
    render_product = rep.create.render_product(camera, resolution=(1280, 720))
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annotator.attach(render_product)
    print("✓ Camera setup complete")
    
    # 设置视口相机
    print("\n[5b/7] Setting viewport camera...")
    try:
        set_camera_view(camera_pos, look_at, camera_prim_path="/OmniverseKit_Persp")
        print("✓ Viewport camera set")
    except Exception as e:
        print(f"Warning: Could not set viewport camera: {e}")
    
    # ===== 6. 连接VLA服务器 =====
    print("\n[6/7] Connecting to VLA server...")
    vla_client = None
    try:
        # 解析服务器URL
        server_url = args.server
        if "://" in server_url:
            parts = server_url.split("://")[1].split(":")
            host = parts[0]
            port = int(parts[1]) if len(parts) > 1 else 8000
        else:
            host = "localhost"
            port = 8000
        
        remote_config = RemoteServerConfig(host=host, port=port, protocol="http")
        model_config = VLAModelConfig(model_name="openvla/openvla-7b")
        
        vla_client = OpenVLAClient(remote_config, model_config)
        
        if vla_client.connect():
            print("✓ Connected to VLA server!")
        else:
            print("⚠ Could not connect to VLA server, using heuristic control")
    except Exception as e:
        print(f"⚠ VLA client error: {e}, using heuristic control")
    
    # ===== 7. 初始化和预热 - 使用demo_motion_control.py的模式 =====
    print("\n[7/7] Initializing and warming up...")
    world.reset()
    rep.orchestrator.run()
    
    print("Warming up (60 steps)...")
    for i in range(60):
        world.step(render=True)
        rep.orchestrator.step()
        if i % 20 == 0:
            print(f"  {i}/60")
    print("✓ Warmup complete")
    
    # ===== 开始录制视频 - 使用demo_motion_control.py的模式 =====
    print("\n" + "="*60)
    print("Starting VLA Grasp Demo with Video Recording...")
    print("="*60)
    
    # 初始化VideoWriter - 确保正确初始化
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, 30, (1280, 720))
    
    if not video_writer.isOpened():
        print(f"✗ Failed to open video writer: {video_path}")
        # 尝试其他格式
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(video_path.replace('.mp4', '.avi'), fourcc, 30, (1280, 720))
    
    if video_writer.isOpened():
        print(f"✓ Video writer opened: {video_path}")
    else:
        print(f"✗ Could not open video writer!")
    
    # 运行 episodes
    instructions = [
        "pick up the red block",
    ]
    
    total_frames = 0
    successes = 0
    
    for episode in range(args.episodes):
        instruction = instructions[episode % len(instructions)]
        success, frames = run_grasp_episode(
            franka, target_object, rgb_annotator, video_writer, 
            vla_client, instruction, max_steps=30
        )
        total_frames += frames
        if success:
            successes += 1
    
    # 释放VideoWriter
    video_writer.release()
    
    print("\n" + "="*60)
    print(f"Demo Complete!")
    print(f"Success rate: {successes}/{args.episodes}")
    print(f"Total frames recorded: {total_frames}")
    print(f"Video saved: {video_path}")
    print("="*60)
    
    # 捕获一张静态图片
    print("\nCapturing static frame...")
    try:
        data = rgb_annotator.get_data()
        if data is not None and len(data.shape) == 3:
            frame = cv2.cvtColor(data[:,:,:3].astype(np.uint8), cv2.COLOR_RGB2BGR)
            frame_path = os.path.join(output_dir, args.output.replace('.mp4', '_frame.png').replace('.avi', '_frame.png'))
            cv2.imwrite(frame_path, frame)
            print(f"  ✓ Frame saved: {frame_path}")
    except Exception as e:
        print(f"  Warning: Could not capture frame: {e}")
    
    # 清理
    if vla_client:
        try:
            vla_client.close()
        except:
            pass
    
    print("\nCleaning up...")
    rep.orchestrator.stop()
    world.clear()
    simulation_app.close()
    
    print("\n" + "="*60)
    print("All resources cleaned up!")
    print("="*60)


if __name__ == "__main__":
    main()
