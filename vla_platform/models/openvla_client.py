# VLA Platform - OpenVLA Client
"""
OpenVLA模型远程推理封装
兼容官方OpenVLA API格式
"""
import numpy as np
from typing import Optional, Dict, Any
from PIL import Image
import logging

from ..core.config import VLAModelConfig, RemoteServerConfig
from ..core.base_interfaces import VLAModelInterface, Observation, Action
from ..remote.rest_client import RESTClient

logger = logging.getLogger(__name__)


class OpenVLAClient(VLAModelInterface):
    """OpenVLA模型客户端"""
    
    # OpenVLA默认动作范围（用于归一化/反归一化）
    ACTION_LOW = np.array([-0.05, -0.05, -0.05, -0.25, -0.25, -0.25, 0.0])
    ACTION_HIGH = np.array([0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 1.0])
    
    def __init__(
        self,
        remote_config: RemoteServerConfig,
        model_config: Optional[VLAModelConfig] = None
    ):
        """
        初始化OpenVLA客户端
        
        Args:
            remote_config: 远程服务器配置
            model_config: 模型配置
        """
        self._remote_config = remote_config
        self._model_config = model_config or VLAModelConfig()
        self._client = RESTClient(remote_config)
        self._is_connected = False
        
    @property
    def model_name(self) -> str:
        return self._model_config.model_name
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected and self._client.is_connected
    
    def connect(self) -> bool:
        """连接到远程服务器"""
        self._is_connected = self._client.connect()
        if self._is_connected:
            logger.info(f"Connected to OpenVLA server at {self._remote_config.base_url}")
        return self._is_connected
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像以符合OpenVLA输入要求
        
        OpenVLA需要224x224 RGB图像
        """
        # 确保是RGB格式
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:
            image = image[:, :, :3]
        
        # 调整尺寸到224x224
        pil_image = Image.fromarray(image.astype(np.uint8))
        pil_image = pil_image.resize((224, 224), Image.Resampling.LANCZOS)
        
        return np.array(pil_image)
    
    def postprocess_action(self, action: np.ndarray) -> np.ndarray:
        """
        后处理动作
        
        OpenVLA输出的动作可能需要裁剪到有效范围
        """
        # 裁剪到有效范围
        return np.clip(action, self.ACTION_LOW, self.ACTION_HIGH)
    
    def predict(self, observation: Observation, instruction: str) -> Action:
        """
        根据观测和语言指令预测动作
        
        Args:
            observation: 当前观测（需要包含图像）
            instruction: 语言指令，例如 "pick up the red block"
            
        Returns:
            预测的7维动作 (dx, dy, dz, drx, dry, drz, gripper)
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to server. Call connect() first.")
        
        # 预处理图像
        processed_image = self.preprocess_image(observation.image)
        processed_obs = Observation(
            image=processed_image,
            depth=observation.depth,
            joint_positions=observation.joint_positions,
            joint_velocities=observation.joint_velocities,
            ee_position=observation.ee_position,
            ee_orientation=observation.ee_orientation,
            gripper_state=observation.gripper_state
        )
        
        # 发送请求
        action = self._client.predict(
            processed_obs,
            instruction,
            model_name=self._model_config.model_name,
            temperature=self._model_config.temperature,
        )
        
        # 后处理
        action.values = self.postprocess_action(action.values)
        
        return action
    
    def predict_with_unnormalized_action(
        self,
        observation: Observation,
        instruction: str,
        action_low: np.ndarray,
        action_high: np.ndarray
    ) -> Action:
        """
        预测并将动作映射到自定义范围
        
        用于将OpenVLA输出映射到特定机器人的动作空间
        """
        action = self.predict(observation, instruction)
        
        # 将动作从[-1, 1]映射到[action_low, action_high]
        normalized = (action.values - self.ACTION_LOW) / (self.ACTION_HIGH - self.ACTION_LOW)
        unnormalized = normalized * (action_high - action_low) + action_low
        
        return Action(values=unnormalized, action_type=action.action_type)
    
    def reset(self) -> None:
        """重置模型状态（OpenVLA是无状态的，此方法为空）"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return self._client.get_model_info()
    
    def close(self):
        """关闭连接"""
        self._client.close()
        self._is_connected = False
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
