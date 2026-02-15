#!/usr/bin/env python3
"""
简化版VLA测试 - 验证基本连接
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# 启动SimulationApp
print("Starting Isaac Sim...")
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
print("✓ Isaac Sim started")

# 导入VLA模块
print("\nImporting VLA modules...")
from vla_platform.simulation.sim_manager import SimulationManager, check_isaac_sim_available
from vla_platform.simulation.envs.franka_env import FrankaGraspEnv, FrankaEnvConfig
from vla_platform.simulation.sensors.sensor_manager import CameraManager, CameraConfig
print("✓ VLA modules imported")

# 检查Isaac Sim可用性
print("\nChecking Isaac Sim availability...")
if check_isaac_sim_available():
    print("✓ Isaac Sim is available")
else:
    print("✗ Isaac Sim is NOT available")
    sys.exit(1)

# 创建SimulationManager
print("\nCreating SimulationManager...")
from vla_platform.core.config import SimulationConfig
config = SimulationConfig()
sim_manager = SimulationManager(config)
print("✓ SimulationManager created")

# 创建World
print("\nCreating World...")
world = sim_manager.create_world()
print("✓ World created")

# 创建相机
print("\nCreating Camera...")
camera_config = CameraConfig(width=224, height=224)
camera = CameraManager(
    prim_path="/World/Camera/test",
    config=camera_config,
    world=sim_manager.world
)
camera.setup()
print("✓ Camera created and initialized")

# 测试连接远程VLA服务
print("\nConnecting to VLA server at http://localhost:8080...")
import requests
try:
    response = requests.get("http://localhost:8080/health", timeout=5)
    if response.status_code == 200:
        print(f"✓ VLA server is healthy: {response.json()}")
    else:
        print(f"⚠ VLA server returned status {response.status_code}")
except Exception as e:
    print(f"✗ Cannot connect to VLA server: {e}")

# 清理
print("\nCleaning up...")
simulation_app.close()
print("✓ Test completed successfully!")
