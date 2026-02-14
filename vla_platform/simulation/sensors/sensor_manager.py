# VLA Platform - Sensor Manager
"""
传感器管理模块
管理RGB相机、深度相机等传感器
"""
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass, field
import logging

# Isaac Sim imports - 支持新旧两种命名空间
ISAAC_SIM_AVAILABLE = False
Camera = None
rot_utils = None

# 尝试新的 isaacsim 命名空间 (Isaac Sim 5.x)
try:
    from isaacsim.sensors.camera import Camera
    import isaacsim.core.utils.numpy.rotations as rot_utils
    from isaacsim.core.api import World
    ISAAC_SIM_AVAILABLE = True
except ImportError:
    pass

# 尝试旧的 omni.isaac 命名空间 (兼容性)
if not ISAAC_SIM_AVAILABLE:
    try:
        from omni.isaac.sensor import Camera
        import omni.isaac.core.utils.numpy.rotations as rot_utils
        from omni.isaac.core import World
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
    相机管理器
    
    管理仿真中的RGB和深度相机
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
        self._camera: Optional["Camera"] = None
        self._is_initialized = False
        
    @property
    def sensor_type(self) -> str:
        return "camera"
    
    @property
    def is_initialized(self) -> bool:
        return self._is_initialized
    
    def setup(self) -> None:
        """设置相机"""
        # 计算朝向四元数
        orientation = self._calculate_orientation()
        
        # 创建相机
        self._camera = Camera(
            prim_path=self.prim_path,
            position=np.array(self.config.position),
            frequency=30,  # 30 Hz
            resolution=(self.config.width, self.config.height),
            orientation=orientation,
        )
        
        # 将相机添加到world scene（如果提供了world）
        if self._world is not None and hasattr(self._world, 'scene'):
            self._world.scene.add(self._camera)
            logger.info(f"Camera added to world scene at {self.prim_path}")
        
        # 初始化渲染 - 必须在添加到scene之后调用
        self._camera.initialize()
        
        # 设置裁剪平面
        self._camera.set_clipping_range(
            near_plane=self.config.near_clip,
            far_plane=self.config.far_clip
        )
        
        self._is_initialized = True
        logger.info(f"Camera setup at {self.prim_path} with resolution {self.config.width}x{self.config.height}")
    
    def _calculate_orientation(self) -> np.ndarray:
        """
        计算相机朝向四元数，使相机朝向target点
        
        Returns:
            四元数 [w, x, y, z]
        """
        if rot_utils is None:
            # 如果无法导入rot_utils，返回默认朝向（指向负Z轴）
            return np.array([1.0, 0.0, 0.0, 0.0])
        
        position = np.array(self.config.position)
        target = np.array(self.config.target)
        
        # 计算前向向量（从position指向target）
        forward = target - position
        forward_norm = np.linalg.norm(forward)
        
        if forward_norm < 1e-6:
            # 如果position和target重合，使用默认朝向
            return np.array([1.0, 0.0, 0.0, 0.0])
        
        forward = forward / forward_norm
        
        # Isaac Sim中相机的默认朝向是指向负Z轴
        # 我们需要计算从 [0, 0, -1] 旋转到 forward 的旋转
        default_forward = np.array([0.0, 0.0, -1.0])
        
        # 计算旋转轴和角度
        rotation_axis = np.cross(default_forward, forward)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        if rotation_axis_norm < 1e-6:
            # 两个向量平行
            if np.dot(default_forward, forward) > 0:
                # 同向，无需旋转
                return np.array([1.0, 0.0, 0.0, 0.0])
            else:
                # 反向，旋转180度
                return np.array([0.0, 0.0, 1.0, 0.0])
        
        rotation_axis = rotation_axis / rotation_axis_norm
        
        # 计算旋转角度
        cos_angle = np.dot(default_forward, forward)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        # 将轴角转换为四元数
        half_angle = angle / 2.0
        w = np.cos(half_angle)
        x = rotation_axis[0] * np.sin(half_angle)
        y = rotation_axis[1] * np.sin(half_angle)
        z = rotation_axis[2] * np.sin(half_angle)
        
        return np.array([w, x, y, z])
    
    def _look_at(self, target: List[float]) -> None:
        """
        设置相机朝向目标点
        
        Args:
            target: 目标位置 [x, y, z]
        """
        if self._camera is None:
            return
        
        self.config.target = target
        
        # 重新计算朝向
        orientation = self._calculate_orientation()
        
        # 设置新的朝向
        position = np.array(self.config.position)
        self._camera.set_world_pose(position=position, orientation=orientation)
        
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
        if not self._is_initialized or self._camera is None:
            return {
                "rgb": np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
            }
        
        result = {}
        
        # 获取RGB图像
        try:
            rgb = self._camera.get_rgba()
            if rgb is not None:
                # 转换为RGB（去掉alpha通道）
                result["rgb"] = rgb[:, :, :3].astype(np.uint8)
            else:
                result["rgb"] = np.zeros(
                    (self.config.height, self.config.width, 3), 
                    dtype=np.uint8
                )
        except Exception as e:
            logger.warning(f"Failed to get RGB image: {e}")
            result["rgb"] = np.zeros(
                (self.config.height, self.config.width, 3), 
                dtype=np.uint8
            )
        
        # 获取深度图
        if self.config.enable_depth:
            try:
                depth = self._camera.get_depth()
                if depth is not None:
                    result["depth"] = depth
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
        if self._camera is not None:
            orientation = self._calculate_orientation()
            self._camera.set_world_pose(position=np.array(position), orientation=orientation)
            self.config.position = position
    
    def set_target(self, target: List[float]) -> None:
        """设置相机目标点"""
        self._look_at(target)
    
    def set_world(self, world) -> None:
        """
        设置world引用（用于后续添加到scene）
        
        Args:
            world: Isaac Sim World实例
        """
        self._world = world
    
    def cleanup(self) -> None:
        """清理相机资源"""
        if self._camera is not None:
            try:
                # 如果相机在scene中，移除它
                if self._world is not None and hasattr(self._world, 'scene'):
                    self._world.scene.remove_object(self._camera.name)
            except Exception as e:
                logger.debug(f"Error removing camera from scene: {e}")
        
        self._camera = None
        self._is_initialized = False
        logger.info(f"Camera cleaned up: {self.prim_path}")


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
        config: Optional[CameraConfig] = None,
        world: Optional[Any] = None
    ) -> CameraManager:
        """添加相机"""
        camera = CameraManager(prim_path=prim_path, config=config, world=world)
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
