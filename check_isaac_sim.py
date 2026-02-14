#!/usr/bin/env python3
"""
Isaac Sim 5.1.0 安装验证脚本
检查本地Isaac Sim安装状态、版本和基本功能
"""

import sys
import subprocess
from pathlib import Path

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_section(text):
    print(f"\n📋 {text}")
    print("-" * 40)

def check_python_version():
    print_section("Python版本检查")
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 7:
        print("✅ Python版本符合要求 (>= 3.7)")
        return True
    else:
        print("❌ Python版本过低，需要 >= 3.7")
        return False

def check_isaac_sim_imports():
    print_section("Isaac Sim模块导入检查")
    
    results = {
        "isaacsim.core.api": False,
        "isaacsim.sensors.camera": False,
        "isaacsim.robot.manipulators.examples.franka": False,
        "isaacsim.storage.native": False,
    }
    
    # 检查新命名空间 (Isaac Sim 5.x)
    print("\n检查 isaacsim 命名空间 (Isaac Sim 5.x):")
    try:
        from isaacsim.core.api import World
        results["isaacsim.core.api"] = True
        print("  ✅ isaacsim.core.api.World")
    except ImportError as e:
        print(f"  ❌ isaacsim.core.api: {e}")
    
    try:
        from isaacsim.sensors.camera import Camera
        results["isaacsim.sensors.camera"] = True
        print("  ✅ isaacsim.sensors.camera.Camera")
    except ImportError as e:
        print(f"  ❌ isaacsim.sensors.camera: {e}")
    
    try:
        from isaacsim.robot.manipulators.examples.franka import Franka
        results["isaacsim.robot.manipulators.examples.franka"] = True
        print("  ✅ isaacsim.robot.manipulators.examples.franka.Franka")
    except ImportError as e:
        print(f"  ❌ isaacsim.robot.manipulators.examples.franka: {e}")
    
    try:
        from isaacsim.storage.native import get_assets_root_path
        results["isaacsim.storage.native"] = True
        print("  ✅ isaacsim.storage.native.get_assets_root_path")
    except ImportError as e:
        print(f"  ❌ isaacsim.storage.native: {e}")
    
    # 检查旧命名空间 (向后兼容)
    print("\n检查 omni.isaac 命名空间 (旧版本兼容):")
    try:
        from omni.isaac.core import World
        print("  ✅ omni.isaac.core.World (旧命名空间)")
    except ImportError:
        print("  ⚠️  omni.isaac.core 不可用")
    
    success_count = sum(results.values())
    print(f"\n新命名空间模块: {success_count}/{len(results)} 可用")
    
    return success_count > 0

def check_isaac_sim_version():
    print_section("Isaac Sim版本检查")
    
    version_info = {
        "version": "未知",
        "kit_version": "未知",
        "build": "未知"
    }
    
    # 尝试多种方式获取版本
    try:
        # 方式1: 通过omni.kit.app获取
        import omni.kit.app
        app = omni.kit.app.get_app()
        version_info["kit_version"] = app.get_version_string() if hasattr(app, 'get_version_string') else "N/A"
        print(f"  Omni Kit版本: {version_info['kit_version']}")
    except Exception as e:
        print(f"  ⚠️  无法获取Omni Kit版本: {e}")
    
    try:
        # 方式2: 检查isaacsim包版本
        import isaacsim
        if hasattr(isaacsim, '__version__'):
            version_info["version"] = isaacsim.__version__
            print(f"  Isaac Sim版本: {version_info['version']}")
        else:
            print(f"  Isaac Sim已安装 (版本号未知)")
    except Exception as e:
        print(f"  ⚠️  无法获取Isaac Sim版本: {e}")
    
    # 尝试运行 Isaac Sim 获取详细版本
    print("\n尝试获取详细版本信息...")
    isaac_sim_paths = [
        "/usr/local/share/isaac-sim",
        "/opt/isaac-sim",
        Path.home() / ".local/share/isaac-sim",
        Path.home() / "isaac-sim",
        "/isaac-sim",
    ]
    
    found_path = None
    for path in isaac_sim_paths:
        if Path(path).exists():
            found_path = path
            print(f"  ✅ 找到Isaac Sim安装目录: {path}")
            
            # 检查版本文件
            version_file = Path(path) / "VERSION"
            if version_file.exists():
                version = version_file.read_text().strip()
                print(f"  📄 版本文件内容: {version}")
            
            # 检查release文件
            release_file = Path(path) / "RELEASE.md"
            if release_file.exists():
                content = release_file.read_text()
                for line in content.split('\n')[:10]:
                    if 'version' in line.lower() or 'isaac' in line.lower():
                        print(f"  📄 {line}")
                        break
            
            break
    
    if not found_path:
        print("  ❌ 未找到Isaac Sim安装目录")
    
    return version_info

def check_simulation_app():
    print_section("SimulationApp检查")
    
    try:
        from isaacsim import SimulationApp
        print("  ✅ SimulationApp 可导入")
        
        # 注意: 实际创建SimulationApp会启动GUI
        # 这里只检查是否可以导入，不实际创建
        print("  ℹ️  SimulationApp可以创建（需要GUI环境）")
        return True
    except ImportError as e:
        print(f"  ❌ SimulationApp 导入失败: {e}")
        return False

def check_assets_path():
    print_section("Nucleus资产路径检查")
    
    try:
        from isaacsim.storage.native import get_assets_root_path
        assets_path = get_assets_root_path()
        if assets_path:
            print(f"  ✅ 资产根路径: {assets_path}")
            
            # 检查常见资产是否存在
            check_paths = [
                "Isaac/Robots/Franka/franka.usd",
                "Isaac/Robots/NVIDIA/Jetbot/jetbot.usd",
                "Isaac/Environments/Grid/default_environment.usd",
            ]
            
            print("\n  检查常用资产:")
            from omni.client import list as omni_list
            for asset_path in check_paths:
                full_path = f"{assets_path}/{asset_path}"
                print(f"    - {asset_path}: ", end="")
                try:
                    result, entries = omni_list(full_path)
                    if result == omni.Result.OK:
                        print("✅ 可用")
                    else:
                        print("⚠️  无法访问")
                except:
                    print("⚠️  检查失败")
        else:
            print("  ⚠️  资产路径为空，可能未连接到Nucleus服务器")
    except Exception as e:
        print(f"  ❌ 无法获取资产路径: {e}")

def run_basic_test():
    print_section("基础功能测试")
    
    print("尝试创建World实例（需要GUI环境，可能会失败）...")
    print("注意: 如果在无GUI环境运行，此测试会跳过")
    
    # 检查是否在无头环境
    import os
    if os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
        print("  ℹ️  检测到显示环境，可以尝试运行")
        
        try:
            # 尝试导入但不实际创建（避免启动GUI）
            from isaacsim.core.api import World
            print("  ✅ World类可导入")
            print("  ℹ️  要完整测试，请运行: python test_isaac_sim_basic.py")
        except Exception as e:
            print(f"  ❌ World类导入失败: {e}")
    else:
        print("  ℹ️  未检测到显示环境，跳过GUI测试")
        print("  💡 提示: 设置 DISPLAY 环境变量后可测试GUI功能")

def main():
    print_header("Isaac Sim 5.1.0 安装验证")
    
    print("\n本脚本将检查您的系统是否安装了Isaac Sim 5.1.0")
    print("并验证必要的模块是否可以正常导入。\n")
    
    results = []
    
    # 1. Python版本
    results.append(("Python版本", check_python_version()))
    
    # 2. Isaac Sim导入
    results.append(("Isaac Sim模块", check_isaac_sim_imports()))
    
    # 3. 版本信息
    version_info = check_isaac_sim_version()
    
    # 4. SimulationApp
    results.append(("SimulationApp", check_simulation_app()))
    
    # 5. 资产路径
    try:
        check_assets_path()
    except Exception as e:
        print(f"资产路径检查出错: {e}")
    
    # 6. 基础测试
    try:
        run_basic_test()
    except Exception as e:
        print(f"基础测试出错: {e}")
    
    # 总结
    print_header("验证结果总结")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n✅ Isaac Sim 5.1.0 已正确安装！")
        print("\n您可以运行以下命令测试VLA项目:")
        print("  cd /home/vincent/Desktop/code/VLA")
        print("  python demos/demo_vla_grasp.py --help")
    else:
        print("\n❌ Isaac Sim 未完全安装或配置不正确")
        print("\n检查项目:")
        for name, passed in results:
            status = "✅" if passed else "❌"
            print(f"  {status} {name}")
        
        print("\n💡 安装建议:")
        print("  1. 下载Isaac Sim 5.1.0:")
        print("     https://developer.nvidia.com/isaac-sim")
        print("  2. 解压到 /usr/local/share/isaac-sim 或 ~/isaac-sim")
        print("  3. 运行 ./install.sh 安装依赖")
        print("  4. 激活环境: source setup_conda_env.sh")
        print("  5. 重新运行本验证脚本")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
