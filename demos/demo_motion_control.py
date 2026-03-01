# Motion Control Demo - Simple Version
"""
运动控制功能演示 - 简化版
直接复用 demo_simple_scene.py 的 Replicator 录制方式
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

# 首先启动Isaac Sim SimulationApp
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
    print("  ./python.sh demos/demo_motion_control.py")
    sys.exit(1)

import numpy as np
import cv2

# Isaac Sim 模块
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.robot.manipulators.examples.franka import Franka
import omni.replicator.core as rep
from isaacsim.core.utils.viewports import set_camera_view

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # 创建世界
    print("\n[1/6] Creating world...")
    world = World(physics_dt=1/60, rendering_dt=1/60)
    world.scene.add_default_ground_plane()
    print("✓ Ground plane added")
    
    # 添加Franka机器人
    print("\n[2/6] Adding Franka robot...")
    franka = Franka(
        prim_path="/World/Franka",
        name="franka",
        position=np.array([0.0, 0.0, 0.0])
    )
    world.scene.add(franka)
    print("✓ Franka robot added")
    
    # 添加桌子
    print("\n[3/6] Adding table...")
    table_pos = [0.4, 0.0, 0.025]  # 桌子中心位置
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
    
    # 设置Replicator相机
    print("\n[4/6] Setting up Replicator camera...")
    camera_pos = (1.5, -1.0, 1.2)
    look_at = (0.4, 0.0, 0.2)
    
    camera = rep.create.camera(position=camera_pos, look_at=look_at)
    render_product = rep.create.render_product(camera, resolution=(1280, 720))
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annotator.attach(render_product)
    print("✓ Camera setup complete")
    
    # 设置视口相机
    print("\n[5/6] Setting viewport camera...")
    try:
        set_camera_view(camera_pos, look_at, camera_prim_path="/OmniverseKit_Persp")
        print("✓ Viewport camera set")
    except Exception as e:
        print(f"Warning: Could not set viewport camera: {e}")
    
    # 初始化
    world.reset()
    rep.orchestrator.run()
    
    # 预热
    print("\n[6/6] Warming up (60 steps)...")
    for i in range(60):
        world.step(render=True)
        rep.orchestrator.step()
        if i % 20 == 0:
            print(f"  {i}/60")
    print("✓ Warmup complete")
    
    # 使用 set_joint_positions 直接设置关节位置
    default_joints = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])
    
    # 开始录制视频
    print("\n" + "="*60)
    print("Recording video...")
    print("="*60)
    
    video_path = os.path.join(output_dir, "motion_control_demo.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, 30, (1280, 720))
    
    frame_count = 0
    
    # 运动控制序列 - 使用set_joint_positions
    targets = [
        np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.785, 0.04, 0.04]),
        np.array([0.3, -0.4, 0.2, -1.8, 0.1, 1.3, 0.6, 0.02, 0.02]),
        np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]),
    ]
    
    for target_idx, target in enumerate(targets):
        print(f"\nMoving to target {target_idx + 1}/{len(targets)}: {target[:7]}")
        
        # 逐步移动到目标 - 使用简单的P控制器
        for step in range(80):
            # 获取当前位置
            try:
                current_pos = franka.get_joint_positions()
                if current_pos is None:
                    current_pos = default_joints.copy()
            except:
                current_pos = default_joints.copy()
            
            # 简单的P控制器
            new_pos = current_pos + 0.025 * (target - current_pos)
            
            # 使用set_joint_positions
            try:
                franka.set_joint_positions(new_pos)
            except Exception as e:
                # 如果失败，忽略
                pass
            
            # 步进
            world.step(render=True)
            rep.orchestrator.step()
            
            # 录制
            data = rgb_annotator.get_data()
            if data is not None and len(data.shape) == 3 and data.shape[2] >= 3:
                frame = cv2.cvtColor(data[:,:,:3].astype(np.uint8), cv2.COLOR_RGB2BGR)
                video_writer.write(frame)
                frame_count += 1
            
            if step % 20 == 0:
                print(f"  Step {step}/80, frames: {frame_count}")
            
            # 检查是否到达
            if np.linalg.norm(new_pos[:7] - target[:7]) < 0.02:
                print(f"  Reached target at step {step}")
                break
        
        time.sleep(0.3)
    
    video_writer.release()
    print(f"\n✓ Video saved: {video_path}")
    print(f"  Total frames: {frame_count}")
    
    # 捕获一张静态图片
    print("\nCapturing static frame...")
    data = rgb_annotator.get_data()
    frame_path = ""
    if data is not None and len(data.shape) == 3:
        frame = cv2.cvtColor(data[:,:,:3].astype(np.uint8), cv2.COLOR_RGB2BGR)
        frame_path = os.path.join(output_dir, "motion_control_demo_frame.png")
        cv2.imwrite(frame_path, frame)
        print(f"  ✓ Frame saved: {frame_path}")
    
    # 清理
    print("\nCleaning up...")
    rep.orchestrator.stop()
    world.clear()
    simulation_app.close()
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print(f"视频: {video_path}")
    print(f"截图: {frame_path}")
    print("="*60)


if __name__ == "__main__":
    main()
