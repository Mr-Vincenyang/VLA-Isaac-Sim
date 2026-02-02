# VLA Platform - Configuration Module
"""
VLA仿真平台全局配置管理
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import yaml


@dataclass
class RemoteServerConfig:
    """远程VLA服务器配置"""
    host: str = "localhost"
    port: int = 8000
    protocol: str = "http"  # http 或 grpc
    timeout: float = 30.0
    max_retries: int = 3
    
    @property
    def base_url(self) -> str:
        return f"{self.protocol}://{self.host}:{self.port}"


@dataclass
class VLAModelConfig:
    """VLA模型配置"""
    model_name: str = "openvla/openvla-7b"
    quantization: Optional[str] = None  # "int4", "int8", None
    max_tokens: int = 256
    temperature: float = 0.0
    action_dim: int = 7  # 默认7自由度 (xyz + rpy + gripper)
    action_bins: int = 256  # 动作离散化bin数


@dataclass 
class SimulationConfig:
    """Isaac Sim仿真配置"""
    physics_dt: float = 1.0 / 120.0  # 物理仿真步进
    rendering_dt: float = 1.0 / 60.0  # 渲染频率
    robot_name: str = "franka"
    robot_usd_path: Optional[str] = None
    scene_usd_path: Optional[str] = None
    headless: bool = False
    
    # 相机配置
    camera_width: int = 224
    camera_height: int = 224
    camera_position: List[float] = field(default_factory=lambda: [0.5, 0.0, 0.5])
    camera_target: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class ControlConfig:
    """运动控制配置"""
    control_mode: str = "joint"  # "joint" 或 "cartesian"
    control_frequency: float = 100.0  # Hz
    
    # PD控制器增益
    kp: List[float] = field(default_factory=lambda: [600, 600, 600, 600, 250, 150, 50])
    kd: List[float] = field(default_factory=lambda: [50, 50, 50, 50, 30, 25, 15])
    
    # 阻抗控制参数
    impedance_stiffness: List[float] = field(default_factory=lambda: [400, 400, 400, 40, 40, 40])
    impedance_damping: List[float] = field(default_factory=lambda: [40, 40, 40, 4, 4, 4])
    
    # 安全限制
    max_velocity: float = 1.0  # rad/s
    max_acceleration: float = 2.0  # rad/s^2


@dataclass
class PlatformConfig:
    """平台总配置"""
    remote: RemoteServerConfig = field(default_factory=RemoteServerConfig)
    model: VLAModelConfig = field(default_factory=VLAModelConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "PlatformConfig":
        """从YAML文件加载配置"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(
            remote=RemoteServerConfig(**data.get('remote', {})),
            model=VLAModelConfig(**data.get('model', {})),
            simulation=SimulationConfig(**data.get('simulation', {})),
            control=ControlConfig(**data.get('control', {}))
        )
    
    def to_yaml(self, path: str) -> None:
        """保存配置到YAML文件"""
        from dataclasses import asdict
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)


# 默认配置实例
DEFAULT_CONFIG = PlatformConfig()
