#!/usr/bin/env python3
"""
简单的Isaac Sim验证 - 仅检查模块导入
"""
import sys

print("="*60, file=sys.stderr)
print("Isaac Sim 5.1.0 基础验证", file=sys.stderr)
print("="*60, file=sys.stderr)

# 检查SimulationApp
print("\n[1/5] 检查 SimulationApp...", file=sys.stderr)
try:
    from isaacsim import SimulationApp
    print("✓ SimulationApp 可导入", file=sys.stderr)
    has_sim_app = True
except Exception as e:
    print(f"✗ SimulationApp: {e}", file=sys.stderr)
    has_sim_app = False
    sys.exit(1)

# 启动SimulationApp来加载其他模块
print("\n[2/5] 启动 SimulationApp...", file=sys.stderr)
try:
    simulation_app = SimulationApp({"headless": True})
    print("✓ SimulationApp 启动成功", file=sys.stderr)
except Exception as e:
    print(f"✗ 启动失败: {e}", file=sys.stderr)
    sys.exit(1)

# 检查核心模块
print("\n[3/5] 检查核心模块...", file=sys.stderr)
checks = [
    ("isaacsim.core.api.World", "from isaacsim.core.api import World"),
    ("isaacsim.sensors.camera.Camera", "from isaacsim.sensors.camera import Camera"),
    ("isaacsim.robot.manipulators.examples.franka.Franka", "from isaacsim.robot.manipulators.examples.franka import Franka"),
    ("isaacsim.core.api.objects.DynamicCuboid", "from isaacsim.core.api.objects import DynamicCuboid"),
]

passed = 0
for name, stmt in checks:
    try:
        exec(stmt)
        print(f"✓ {name}", file=sys.stderr)
        passed += 1
    except Exception as e:
        print(f"✗ {name}: {e}", file=sys.stderr)

print(f"\n模块检查: {passed}/{len(checks)} 通过", file=sys.stderr)

# 测试World创建
print("\n[4/5] 测试 World 创建...", file=sys.stderr)
try:
    from isaacsim.core.api import World
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    world.reset()
    print("✓ World 创建成功", file=sys.stderr)
    world_test = True
except Exception as e:
    print(f"✗ World 创建失败: {e}", file=sys.stderr)
    world_test = False

# 测试Franka
print("\n[5/5] 测试 Franka 机器人...", file=sys.stderr)
try:
    from isaacsim.robot.manipulators.examples.franka import Franka
    franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka"))
    world.reset()
    print(f"✓ Franka 创建成功 (DOF: {franka.num_dof})", file=sys.stderr)
    franka_test = True
except Exception as e:
    print(f"✗ Franka 创建失败: {e}", file=sys.stderr)
    franka_test = False

# 清理
print("\n清理资源...", file=sys.stderr)
simulation_app.close()

# 总结
print("\n" + "="*60, file=sys.stderr)
if passed == len(checks) and world_test and franka_test:
    print("🎉 所有测试通过！Isaac Sim 5.1.0 工作正常", file=sys.stderr)
    print("="*60, file=sys.stderr)
    # 写入成功文件
    with open('/tmp/isaac_sim_test_success', 'w') as f:
        f.write('success')
    sys.exit(0)
else:
    print("⚠️ 部分测试失败", file=sys.stderr)
    print("="*60, file=sys.stderr)
    sys.exit(1)
