# VLA Platform - Sensor Manager
"""
传感器管理模块
管理RGB相机、深度相机等传感器
"""
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field
import logging

try:
    from omni.isaac.sensor import Camera
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.prims import XFormPrim
    import omni.replicator.core as rep
    ISAAC_SIM_AVAILABLE = True
except ImportError:
    ISAAC_SIM_AVAILABLE = False
    Camera = None

from ..core.base_interfaces import SensorInterface

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
    相机管理器
    
    管理仿真中的RGB和深度相机
    """
    
    def __init__(
        self,
        prim_path: str = "/World/Camera",
        config: Optional[CameraConfig] = None
    ):
        """
        初始化相机管理器
        
        Args:
            prim_path: 相机prim路径
            config: 相机配置
        """
        if not ISAAC_SIM_AVAILABLE:
            raise RuntimeError("Isaac Sim not available")
        
        self.prim_path = prim_path
        self.config = config or CameraConfig()
        self._camera: Optional[Camera] = None
        self._is_initialized = False
        
    @property
    def sensor_type(self) -> str:
        return "camera"
    
    @property
    def is_initialized(self) -> bool:
        return self._is_initialized
    
    def setup(self) -> None:
        """设置相机"""
        # 创建相机
        self._camera = Camera(
            prim_path=self.prim_path,
            position=np.array(self.config.position),
            frequency=30,  # 30 Hz
            resolution=(self.config.width, self.config.height),
        )
        
        # 设置相机朝向目标
        self._look_at(self.config.target)
        
        # 初始化渲染
        self._camera.initialize()
        
        self._is_initialized = True
        logger.info(f"Camera setup at {self.prim_path}")
    
    def _look_at(self, target: List[float]) -> None:
        """设置相机朝向"""
        if self._camera is None:
            return
        
        # 计算朝向
        position = np.array(self.config.position)
        target = np.array(target)
        
        # 简化实现：使用set_world_pose
        # 实际应用中应计算正确的四元数
        forward = target - position
        forward = forward / np.linalg.norm(forward)
        
    def get_resolution(self) -> Tuple[int, int]:
        """获取相机分辨率"""
        return (self.config.width, self.config.height)
    
    def capture(self) -> Dict[str, np.ndarray]:
        """
        采集相机数据
        
        Returns:
            包含'rgb'和可选'depth'的字典
        """
        if not self._is_initialized or self._camera is None:
            return {
                "rgb": np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
            }
        
        result = {}
        
        # 获取RGB图像
        rgb = self._camera.get_rgba()
        if rgb is not None:
            # 转换为RGB（去掉alpha通道）
            result["rgb"] = rgb[:, :, :3].astype(np.uint8)
        else:
            result["rgb"] = np.zeros(
                (self.config.height, self.config.width, 3), 
                dtype=np.uint8
            )
        
        # 获取深度图
        if self.config.enable_depth:
            depth = self._camera.get_depth()
            if depth is not None:
                result["depth"] = depth
        
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
        if self._camera is not None:
            self._camera.set_world_pose(position=np.array(position))
            self.config.position = position
    
    def set_target(self, target: List[float]) -> None:
        """设置相机目标点"""
        self.config.target = target
        self._look_at(target)
    
    def cleanup(self) -> None:
        """清理相机资源"""
        self._camera = None
        self._is_initialized = False


class MultiCameraManager:
    """
    多相机管理器
    
    管理多个视角的相机
    """
    
    def __init__(self):
        self.cameras: Dict[str, CameraManager] = {}
    
    def add_camera(
        self,
        name: str,
        prim_path: str,
        config: Optional[CameraConfig] = None
    ) -> CameraManager:
        """添加相机"""
        camera = CameraManager(prim_path=prim_path, config=config)
        self.cameras[name] = camera
        return camera
    
    def setup_all(self) -> None:
        """设置所有相机"""
        for camera in self.cameras.values():
            camera.setup()
    
    def capture_all(self) -> Dict[str, Dict[str, np.ndarray]]:
        """采集所有相机数据"""
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
    position=[0.0, 0.0, 0.1],  # 相对于末端执行器
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
