#!/usr/bin/env python3
"""
Isaac Sim 5.1.0 基础功能测试 - 使用SimulationApp
必须在Isaac Sim环境中运行
"""

import sys

# 必须在导入其他模块之前启动SimulationApp
print("="*60)
print("Isaac Sim 5.1.0 基础功能测试")
print("="*60)

print("\n🚀 启动SimulationApp...")
print("  (这可能需要几秒钟时间)")

try:
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": True})
    print("  ✓ SimulationApp启动成功")
except Exception as e:
    print(f"  ✗ SimulationApp启动失败: {e}")
    sys.exit(1)

print("\n📦 测试基本模块导入...")

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
        print(f"  ✓ {name}")
        passed += 1
    except Exception as e:
        print(f"  ✗ {name}: {e}")

print(f"\n导入测试通过: {passed}/{len(tests)}")

if passed == len(tests):
    print("\n✅ 所有模块导入成功！")
    print("\n继续测试World创建...")
    
    try:
        from isaacsim.core.api import World
        print("  创建World...")
        world = World(stage_units_in_meters=1.0)
        
        print("  添加默认地面...")
        world.scene.add_default_ground_plane()
        
        print("  重置World...")
        world.reset()
        
        print("  ✓ World创建成功")
        
        # 测试Franka
        print("\n  测试Franka机器人...")
        from isaacsim.robot.manipulators.examples.franka import Franka
        franka = world.scene.add(
            Franka(prim_path="/World/Franka", name="franka")
        )
        world.reset()
        print(f"  ✓ Franka机器人创建成功，关节数: {franka.num_dof}")
        
        # 测试Camera
        print("\n  测试相机...")
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
        print("  ✓ 相机创建并初始化成功")
        
        # 运行几步仿真
        print("\n  运行仿真步进...")
        for i in range(5):
            world.step(render=False)
        print("  ✓ 仿真步进成功")
        
        # 清理
        print("\n  清理资源...")
        simulation_app.close()
        print("  ✓ 清理完成")
        
        print("\n" + "="*60)
        print("🎉 所有测试通过！Isaac Sim 5.1.0 运行正常")
        print("="*60)
        sys.exit(0)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
else:
    print("\n⚠️ 部分模块导入失败，请检查环境配置")
    simulation_app.close()
    sys.exit(1)
