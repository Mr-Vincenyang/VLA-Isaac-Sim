#!/usr/bin/env python3
"""
纯仿真测试 - 验证场景和机械臂是否正确加载
不依赖远程VLA服务器
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# 启动SimulationApp
print("Starting Isaac Sim...")
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})  # GUI模式
print("✓ Isaac Sim started")

import numpy as np
import time

# 导入模块
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.sensors.camera import Camera

print("\nCreating World...")
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
print("✓ World created with ground plane")

print("\nAdding Franka robot...")
franka = world.scene.add(
    Franka(prim_path="/World/Franka", name="franka")
)
print(f"✓ Franka added (DOF: {franka.num_dof})")

print("\nAdding test objects...")
cube = world.scene.add(
    DynamicCuboid(
        prim_path="/World/Cube",
        name="cube",
        position=np.array([0.5, 0.0, 0.05]),
        scale=np.array([0.1, 0.1, 0.1]),
        color=np.array([1.0, 0.0, 0.0])
    )
)
print("✓ Test cube added")

print("\nAdding camera...")
camera = Camera(
    prim_path="/World/Camera",
    position=np.array([1.0, 1.0, 1.0]),
    frequency=30,
    resolution=(512, 512)
)
world.scene.add(camera)
camera.initialize()
print("✓ Camera added")

print("\nResetting world...")
world.reset()
print("✓ World reset")

print("\n✓✓✓ Scene setup complete! ✓✓✓")
print("\nYou should now see:")
print("- A ground plane")
print("- A Franka robot arm")
print("- A red cube")
print("- Camera view")
print("\nRunning simulation for 5 seconds...")

# 运行仿真
for i in range(500):  # 500 steps at 100Hz = 5 seconds
    world.step(render=True)
    if i % 100 == 0:
        print(f"Step {i}/500")

print("\n✓ Simulation complete!")
print("Closing Isaac Sim...")
simulation_app.close()
