# VLA Platform - Isaac Sim Manager
"""
Isaac Sim仿真环境管理器
负责创建和管理仿真世界、物理步进、渲染等
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, List, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    pass  # Type hints handled below
from pathlib import Path
import logging

# Isaac Sim imports - 支持新旧两种命名空间
ISAAC_SIM_AVAILABLE = False
World = None
Scene = None
XFormPrim = None
add_reference_to_stage = None
get_assets_root_path = None

# 尝试新的 isaacsim 命名空间 (Isaac Sim 5.x)
try:
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.nucleus import get_assets_root_path
    from isaacsim.core.prims import SingleXFormPrim as XFormPrim
    from isaacsim.core.api.scenes import Scene
    import omni.usd
    ISAAC_SIM_AVAILABLE = True
except ImportError:
    pass

# 尝试旧的 omni.isaac 命名空间 (兼容性)
if not ISAAC_SIM_AVAILABLE:
    try:
        from omni.isaac.core import World
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        from omni.isaac.core.prims import XFormPrim
        from omni.isaac.core.scenes import Scene
        import omni.usd
        ISAAC_SIM_AVAILABLE = True
    except ImportError:
        pass

from ..core.config import SimulationConfig

logger = logging.getLogger(__name__)


class SimulationManager:
    """
    Isaac Sim仿真管理器
    
    管理仿真世界的创建、运行和销毁
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        初始化仿真管理器
        
        Args:
            config: 仿真配置
        """
        if not ISAAC_SIM_AVAILABLE:
            raise RuntimeError(
                "Isaac Sim is not available. Please run this in Isaac Sim environment."
            )
        
        self.config = config or SimulationConfig()
        self._world: Optional[World] = None
        self._scene: Optional[Scene] = None
        self._is_running = False
        self._step_callbacks: List[Callable] = []
        
    @property
    def world(self) -> Optional[World]:
        """获取仿真世界"""
        return self._world
    
    @property
    def is_running(self) -> bool:
        """检查仿真是否运行中"""
        return self._is_running
    
    def create_world(self, name: str = "vla_sim") -> World:
        """
        创建仿真世界
        
        Args:
            name: 世界名称
            
        Returns:
            创建的World实例
        """
        self._world = World(
            stage_units_in_meters=1.0,
            physics_dt=self.config.physics_dt,
            rendering_dt=self.config.rendering_dt,
        )
        
        # 获取场景
        self._scene = self._world.scene
        
        logger.info(f"Created simulation world: {name}")
        return self._world
    
    def load_scene(self, usd_path: Optional[str] = None) -> None:
        """
        加载场景USD文件
        
        Args:
            usd_path: USD文件路径，如果为None则使用配置中的路径
        """
        if self._world is None:
            raise RuntimeError("World not created. Call create_world() first.")
        
        scene_path = usd_path or self.config.scene_usd_path
        
        if scene_path:
            add_reference_to_stage(
                usd_path=scene_path,
                prim_path="/World/Scene"
            )
            logger.info(f"Loaded scene from: {scene_path}")
        else:
            # 创建默认的地面平面
            self._world.scene.add_default_ground_plane()
            logger.info("Created default ground plane")
    
    def add_prim(
        self,
        prim_path: str,
        usd_path: str,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None
    ) -> XFormPrim:
        """
        添加USD对象到场景
        
        Args:
            prim_path: 场景中的prim路径
            usd_path: USD文件路径
            position: 位置 [x, y, z]
            orientation: 四元数 [w, x, y, z]
            scale: 缩放 [sx, sy, sz]
            
        Returns:
            创建的XFormPrim
        """
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        
        xform = XFormPrim(
            prim_path=prim_path,
            position=position,
            orientation=orientation,
            scale=scale
        )
        
        return xform
    
    def register_step_callback(self, callback: Callable) -> None:
        """
        注册步进回调函数
        
        回调将在每个物理步进后调用
        """
        self._step_callbacks.append(callback)
    
    def step(self, render: bool = True) -> None:
        """
        执行一个仿真步进
        
        Args:
            render: 是否渲染
        """
        if self._world is None:
            raise RuntimeError("World not created")
        
        self._world.step(render=render)
        
        # 调用回调
        for callback in self._step_callbacks:
            callback()
    
    def reset(self) -> None:
        """重置仿真世界"""
        if self._world is not None:
            self._world.reset()
            logger.info("Simulation reset")
    
    def start(self) -> None:
        """开始仿真"""
        if self._world is None:
            raise RuntimeError("World not created")
        self._is_running = True
        logger.info("Simulation started")
    
    def stop(self) -> None:
        """停止仿真"""
        self._is_running = False
        logger.info("Simulation stopped")
    
    def run_for_steps(self, num_steps: int, render: bool = True) -> None:
        """
        运行指定数量的仿真步数
        
        Args:
            num_steps: 步数
            render: 是否渲染
        """
        self.start()
        for _ in range(num_steps):
            if not self._is_running:
                break
            self.step(render=render)
        self.stop()
    
    async def run_async(self, on_step: Optional[Callable] = None) -> None:
        """
        异步运行仿真（用于与Isaac Sim UI集成）
        """
        self.start()
        while self._is_running:
            self.step()
            if on_step:
                await on_step()
    
    def get_physics_context(self):
        """获取物理上下文"""
        if self._world is not None:
            return self._world.get_physics_context()
        return None
    
    def set_physics_dt(self, dt: float) -> None:
        """设置物理步进时间"""
        physics_context = self.get_physics_context()
        if physics_context:
            physics_context.set_physics_dt(dt)
            self.config.physics_dt = dt
    
    def cleanup(self) -> None:
        """清理仿真资源"""
        self.stop()
        if self._world is not None:
            self._world.clear()
            self._world = None
        self._scene = None
        self._step_callbacks.clear()
        logger.info("Simulation cleaned up")
    
    def __enter__(self):
        self.create_world()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def get_isaac_assets_path() -> Optional[str]:
    """获取Isaac Sim资产路径"""
    if ISAAC_SIM_AVAILABLE:
        return get_assets_root_path()
    return None


def check_isaac_sim_available() -> bool:
    """检查Isaac Sim是否可用"""
    return ISAAC_SIM_AVAILABLE
