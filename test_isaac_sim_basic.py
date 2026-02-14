#!/usr/bin/env python3
"""
Isaac Sim 5.1.0 基础功能测试
在激活Isaac Sim环境后运行此脚本
"""

import sys

def test_basic_imports():
    """测试基本导入"""
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
            print(f"  ✅ {name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
    
    print(f"\n通过: {passed}/{len(tests)}")
    return passed == len(tests)

def test_world_creation():
    """测试创建World（无GUI模式）"""
    print("\n🌍 测试World创建...")
    
    try:
        from isaacsim import SimulationApp
        
        # 无头模式启动
        print("  启动SimulationApp (headless模式)...")
        simulation_app = SimulationApp({"headless": True})
        
        from isaacsim.core.api import World
        
        print("  创建World...")
        world = World(stage_units_in_meters=1.0)
        
        print("  添加默认地面...")
        world.scene.add_default_ground_plane()
        
        print("  重置World...")
        world.reset()
        
        print("  ✅ World创建成功")
        
        # 清理
        simulation_app.close()
        print("  ✅ 清理完成")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_franka_robot():
    """测试Franka机器人"""
    print("\n🤖 测试Franka机器人...")
    
    try:
        from isaacsim import SimulationApp
        simulation_app = SimulationApp({"headless": True})
        
        from isaacsim.core.api import World
        from isaacsim.robot.manipulators.examples.franka import Franka
        
        world = World()
        world.scene.add_default_ground_plane()
        
        print("  添加Franka机器人...")
        franka = world.scene.add(
            Franka(prim_path="/World/Franka", name="franka")
        )
        
        world.reset()
        
        print(f"  关节数量: {franka.num_dof}")
        print(f"  关节位置: {franka.get_joint_positions()}")
        
        print("  ✅ Franka机器人测试成功")
        
        simulation_app.close()
        return True
        
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_camera():
    """测试相机"""
    print("\n📷 测试相机...")
    
    try:
        from isaacsim import SimulationApp
        simulation_app = SimulationApp({"headless": True})
        
        from isaacsim.core.api import World
        from isaacsim.sensors.camera import Camera
        import numpy as np
        
        world = World()
        world.scene.add_default_ground_plane()
        
        print("  创建相机...")
        camera = Camera(
            prim_path="/World/Camera",
            position=np.array([0.0, 0.0, 1.0]),
            resolution=(256, 256),
            frequency=20,
        )
        world.scene.add(camera)
        camera.initialize()
        
        print("  运行几步仿真...")
        for _ in range(5):
            world.step(render=False)
        
        print("  采集图像...")
        rgb = camera.get_rgba()
        print(f"  图像尺寸: {rgb.shape if rgb is not None else 'None'}")
        
        print("  ✅ 相机测试成功")
        
        simulation_app.close()
        return True
        
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("Isaac Sim 5.1.0 基础功能测试")
    print("="*60)
    
    print("\n⚠️  此测试需要在Isaac Sim环境中运行")
    print("运行方式:")
    print("  cd /home/vincent/isaac-sim")
    print("  ./python.sh /home/vincent/Desktop/code/VLA/test_isaac_sim_basic.py")
    
    results = []
    
    # 测试1: 基本导入
    results.append(("基本导入", test_basic_imports()))
    
    # 测试2: World创建
    results.append(("World创建", test_world_creation()))
    
    # 测试3: Franka机器人
    results.append(("Franka机器人", test_franka_robot()))
    
    # 测试4: 相机
    results.append(("相机", test_camera()))
    
    # 总结
    print("\n" + "="*60)
    print("测试结果总结")
    print("="*60)
    
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{status} - {name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n🎉 所有测试通过！Isaac Sim 5.1.0 运行正常")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查环境配置")
        return 1

if __name__ == "__main__":
    sys.exit(main())
