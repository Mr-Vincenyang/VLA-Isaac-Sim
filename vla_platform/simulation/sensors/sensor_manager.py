# VLA Platform - Sensor Manager (Replicator API)
"""
传感器管理模块 - 使用Replicator API
管理RGB相机、深度相机等传感器
解决Isaac Sim 5.1.0 headless模式下的相机黑屏问题
"""
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass, field
import logging
import omni.replicator.core as rep

# Isaac Sim imports
ISAAC_SIM_AVAILABLE = False
try:
    from isaacsim.core.api import World
    ISAAC_SIM_AVAILABLE = True
except ImportError:
    pass

from vla_platform.core.base_interfaces import SensorInterface

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """相机配置"""
    width: int = 224
    height: int = 224
    position: List[float] = field(default_factory=lambda: [0.5, 0.0, 0.5])
    target: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    fov: float = 60.0  # 视场角（度）
    near_clip: float = 0.01
    far_clip: float = 10.0
    enable_depth: bool = True
    enable_semantic: bool = False


class CameraManager(SensorInterface):
    """
    相机管理器 - 使用Replicator API
    
    管理仿真中的RGB和深度相机
    解决Isaac Sim 5.1.0 headless模式下的相机黑屏问题
    """
    
    def __init__(
        self,
        prim_path: str = "/World/Camera",
        config: Optional[CameraConfig] = None,
        world: Optional[Any] = None
    ):
        """
        初始化相机管理器
        
        Args:
            prim_path: 相机prim路径
            config: 相机配置
            world: Isaac Sim World实例（用于将相机添加到场景）
        """
        if not ISAAC_SIM_AVAILABLE:
            raise RuntimeError("Isaac Sim not available")
        
        self.prim_path = prim_path
        self.config = config or CameraConfig()
        self._world = world
        self._camera = None
        self._render_product = None
        self._rgb_annotator = None
        self._depth_annotator = None
        self._is_initialized = False
        self._orchestrator_running = False
        
    @property
    def sensor_type(self) -> str:
        return "camera"
    
    @property
    def is_initialized(self) -> bool:
        return self._is_initialized
    
    def setup(self) -> None:
        """设置相机 - 使用Replicator API"""
        # 创建相机
        camera_pos = tuple(self.config.position)
        target_pos = tuple(self.config.target)
        
        # 使用Replicator创建相机
        self._camera = rep.create.camera(
            position=camera_pos,
            look_at=target_pos
        )
        logger.info(f"Created Replicator camera at {camera_pos}")
        
        # 创建渲染产品
        self._render_product = rep.create.render_product(
            self._camera,
            resolution=(self.config.width, self.config.height)
        )
        logger.info(f"Created render product at {self.config.width}x{self.config.height}")
        
        # 设置RGB annotator
        self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self._rgb_annotator.attach(self._render_product)
        logger.info("Attached RGB annotator")
        
        # 设置深度 annotator（如果启用）
        if self.config.enable_depth:
            try:
                self._depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
                self._depth_annotator.attach(self._render_product)
                logger.info("Attached depth annotator")
            except Exception as e:
                logger.warning(f"Could not attach depth annotator: {e}")
        
        # 启动Replicator orchestrator
        if not self._orchestrator_running:
            rep.orchestrator.run()
            self._orchestrator_running = True
        
        self._is_initialized = True
        logger.info(f"Camera setup complete at {self.prim_path}")
    
    def orchestrator_step(self) -> None:
        """调用Replicator orchestrator步进（必须在每帧渲染后调用）"""
        if self._orchestrator_running:
            rep.orchestrator.step()
    
    def _calculate_orientation(self) -> np.ndarray:
        """计算相机朝向四元数（兼容方法）"""
        position = np.array(self.config.position)
        target = np.array(self.config.target)
        
        forward = target - position
        forward_norm = np.linalg.norm(forward)
        
        if forward_norm < 1e-6:
            return np.array([1.0, 0.0, 0.0, 0.0])
        
        forward = forward / forward_norm
        default_forward = np.array([0.0, 0.0, -1.0])
        
        rotation_axis = np.cross(default_forward, forward)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        if rotation_axis_norm < 1e-6:
            if np.dot(default_forward, forward) > 0:
                return np.array([1.0, 0.0, 0.0, 0.0])
            else:
                return np.array([0.0, 0.0, 1.0, 0.0])
        
        rotation_axis = rotation_axis / rotation_axis_norm
        cos_angle = np.clip(np.dot(default_forward, forward), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        half_angle = angle / 2.0
        
        w = np.cos(half_angle)
        x = rotation_axis[0] * np.sin(half_angle)
        y = rotation_axis[1] * np.sin(half_angle)
        z = rotation_axis[2] * np.sin(half_angle)
        
        return np.array([w, x, y, z])
    
    def _look_at(self, target: List[float]) -> None:
        """设置相机朝向目标点"""
        if self._camera is None:
            return
        
        self.config.target = target
        
        camera_pos = tuple(self.config.position)
        target_pos = tuple(target)
        
        self._camera = rep.create.camera(
            position=camera_pos,
            look_at=target_pos,
            fstop=1.8,
            focal_length=24.0
        )
        
        self._render_product = rep.create.render_product(
            self._camera,
            resolution=(self.config.width, self.config.height)
        )
        
        self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self._rgb_annotator.attach(self._render_product)
        
        if self._depth_annotator:
            self._depth_annotator.attach(self._render_product)
        
        logger.debug(f"Camera oriented to look at {target}")
    
    def get_resolution(self) -> Tuple[int, int]:
        """获取相机分辨率"""
        return (self.config.width, self.config.height)
    
    def capture(self) -> Dict[str, np.ndarray]:
        """
        采集相机数据
        
        Returns:
            包含'rgb'和可选'depth'的字典
        """
        if not self._is_initialized or self._rgb_annotator is None:
            return {
                "rgb": np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
            }
        
        result = {}
        
        try:
            # 调用orchestrator获取最新数据
            self.orchestrator_step()
            
            # 从annotator获取数据
            rgb_data = self._rgb_annotator.get_data()
            
            if rgb_data is not None:
                if isinstance(rgb_data, np.ndarray):
                    if len(rgb_data.shape) == 3 and rgb_data.shape[2] >= 3:
                        # RGBA -> RGB
                        result["rgb"] = rgb_data[:, :, :3].astype(np.uint8)
                        logger.debug(f"Camera captured image: {result['rgb'].shape}")
                    elif len(rgb_data.shape) == 2:
                        # 灰度图 -> RGB
                        result["rgb"] = np.stack([rgb_data]*3, axis=-1).astype(np.uint8)
                    else:
                        logger.warning(f"Unexpected RGB array shape: {rgb_data.shape}")
                        result["rgb"] = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
                else:
                    result["rgb"] = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
            else:
                logger.debug("Camera returned None, returning zeros")
                result["rgb"] = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
                
        except Exception as e:
            logger.warning(f"Failed to get RGB image: {e}")
            result["rgb"] = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
        
        # 获取深度图
        if self.config.enable_depth and self._depth_annotator is not None:
            try:
                depth_data = self._depth_annotator.get_data()
                if depth_data is not None and isinstance(depth_data, np.ndarray):
                    result["depth"] = depth_data
            except Exception as e:
                logger.warning(f"Failed to get depth image: {e}")
        
        return result
    
    def get_rgb(self) -> np.ndarray:
        """获取RGB图像"""
        return self.capture().get("rgb", np.zeros((self.config.height, self.config.width, 3)))
    
    def get_depth(self) -> Optional[np.ndarray]:
        """获取深度图"""
        if not self.config.enable_depth:
            return None
        return self.capture().get("depth")
    
    def set_position(self, position: List[float]) -> None:
        """设置相机位置"""
        self.config.position = position
        self._rebuild_camera()
    
    def set_target(self, target: List[float]) -> None:
        """设置相机目标点"""
        self._look_at(target)
    
    def _rebuild_camera(self) -> None:
        """重建相机（当位置或目标改变时）"""
        if not self._is_initialized:
            return
            
        camera_pos = tuple(self.config.position)
        target_pos = tuple(self.config.target)
        
        self._camera = rep.create.camera(
            position=camera_pos,
            look_at=target_pos,
            fstop=1.8,
            focal_length=24.0
        )
        
        self._render_product = rep.create.render_product(
            self._camera,
            resolution=(self.config.width, self.config.height)
        )
        
        self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self._rgb_annotator.attach(self._render_product)
        
        if self._depth_annotator:
            self._depth_annotator.attach(self._render_product)
        
        logger.debug(f"Camera rebuilt at {camera_pos}")
    
    def set_world(self, world) -> None:
        """设置world引用"""
        self._world = world
    
    def cleanup(self) -> None:
        """清理相机资源"""
        try:
            if self._orchestrator_running:
                rep.orchestrator.stop()
                self._orchestrator_running = False
        except Exception as e:
            logger.debug(f"Error stopping orchestrator: {e}")
        
        self._camera = None
        self._render_product = None
        self._rgb_annotator = None
        self._depth_annotator = None
        self._is_initialized = False
        logger.info(f"Camera cleaned up: {self.prim_path}")


class MultiCameraManager:
    """多相机管理器 - 管理多个视角的相机"""
    
    def __init__(self):
        self.cameras: Dict[str, CameraManager] = {}
    
    def add_camera(
        self,
        name: str,
        prim_path: str,
        config: Optional[CameraConfig] = None,
        world: Optional[Any] = None
    ) -> CameraManager:
        """添加相机"""
        camera = CameraManager(prim_path=prim_path, config=config, world=world)
        self.cameras[name] = camera
        return camera
    
    def setup_all(self) -> None:
        """设置所有相机"""
        rep.orchestrator.run()
        for camera in self.cameras.values():
            camera.setup()
    
    def capture_all(self) -> Dict[str, Dict[str, np.ndarray]]:
        """采集所有相机数据"""
        for camera in self.cameras.values():
            camera.orchestrator_step()
        
        return {
            name: camera.capture()
            for name, camera in self.cameras.items()
        }
    
    def get_camera(self, name: str) -> Optional[CameraManager]:
        """获取指定相机"""
        return self.cameras.get(name)
    
    def cleanup_all(self) -> None:
        """清理所有相机"""
        for camera in self.cameras.values():
            camera.cleanup()
        self.cameras.clear()


# 预设相机配置
OVERHEAD_CAMERA_CONFIG = CameraConfig(
    width=224,
    height=224,
    position=[0.4, 0.0, 0.8],
    target=[0.4, 0.0, 0.0],
    fov=60.0
)

WRIST_CAMERA_CONFIG = CameraConfig(
    width=224,
    height=224,
    position=[0.0, 0.0, 0.1],
    target=[0.0, 0.0, 0.0],
    fov=90.0
)

SIDE_CAMERA_CONFIG = CameraConfig(
    width=224,
    height=224,
    position=[0.0, 0.8, 0.4],
    target=[0.4, 0.0, 0.2],
    fov=60.0
)
