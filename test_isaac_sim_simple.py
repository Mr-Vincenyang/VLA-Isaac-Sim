#!/usr/bin/env python3
"""
Isaac Sim 5.1.0 基础功能测试 - 简化版
在激活Isaac Sim环境后运行此脚本
"""

import sys

# 将输出重定向到stdout
print("="*60, flush=True)
print("Isaac Sim 5.1.0 基础功能测试", flush=True)
print("="*60, flush=True)

print("\n📦 测试基本模块导入...", flush=True)

tests = [
    ("isaacsim.core.api.World", "from isaacsim.core.api import World"),
    ("isaacsim.sensors.camera.Camera", "from isaacsim.sensors.camera import Camera"),
    ("isaacsim.robot.manipulators.examples.franka.Franka", "from isaacsim.robot.manipulators.examples.franka import Franka"),
    ("isaacsim.core.api.objects.DynamicCuboid", "from isaacsim.core.api.objects import DynamicCuboid"),
]

passed = 0
for name, import_stmt in tests:
    try:
        exec(import_stmt)
        print(f"  ✓ {name}", flush=True)
        passed += 1
    except Exception as e:
        print(f"  ✗ {name}: {e}", flush=True)

print(f"\n导入测试通过: {passed}/{len(tests)}", flush=True)

if passed == len(tests):
    print("\n✅ 所有模块导入成功！", flush=True)
    print("\n继续测试World创建...", flush=True)
    
    try:
        from isaacsim import SimulationApp
        print("  启动SimulationApp (headless模式)...", flush=True)
        simulation_app = SimulationApp({"headless": True})
        
        from isaacsim.core.api import World
        print("  创建World...", flush=True)
        world = World(stage_units_in_meters=1.0)
        
        print("  添加默认地面...", flush=True)
        world.scene.add_default_ground_plane()
        
        print("  重置World...", flush=True)
        world.reset()
        
        print("  ✓ World创建成功", flush=True)
        
        # 测试Franka
        print("\n  测试Franka机器人...", flush=True)
        from isaacsim.robot.manipulators.examples.franka import Franka
        franka = world.scene.add(
            Franka(prim_path="/World/Franka", name="franka")
        )
        world.reset()
        print(f"  ✓ Franka机器人创建成功，关节数: {franka.num_dof}", flush=True)
        
        # 测试Camera
        print("\n  测试相机...", flush=True)
        from isaacsim.sensors.camera import Camera
        import numpy as np
        
        camera = Camera(
            prim_path="/World/Camera",
            position=np.array([0.0, 0.0, 1.0]),
            resolution=(256, 256),
            frequency=20,
        )
        world.scene.add(camera)
        camera.initialize()
        print("  ✓ 相机创建并初始化成功", flush=True)
        
        # 运行几步仿真
        print("\n  运行仿真步进...", flush=True)
        for i in range(5):
            world.step(render=False)
        print("  ✓ 仿真步进成功", flush=True)
        
        # 清理
        print("\n  清理资源...", flush=True)
        simulation_app.close()
        print("  ✓ 清理完成", flush=True)
        
        print("\n" + "="*60, flush=True)
        print("🎉 所有测试通过！Isaac Sim 5.1.0 运行正常", flush=True)
        print("="*60, flush=True)
        sys.exit(0)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
else:
    print("\n⚠️ 部分模块导入失败，请检查环境配置", flush=True)
    sys.exit(1)
