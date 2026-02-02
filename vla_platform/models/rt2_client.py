# VLA Platform - RT-2 Style Client
"""
RT-2风格模型远程推理封装
支持类似RT-2的token化输出格式
"""
import numpy as np
from typing import Optional, Dict, Any, List
from PIL import Image
import logging

from ..core.config import VLAModelConfig, RemoteServerConfig
from ..core.base_interfaces import VLAModelInterface, Observation, Action
from ..remote.rest_client import RESTClient
from .action_tokenizer import RT2ActionTokenizer

logger = logging.getLogger(__name__)


class RT2Client(VLAModelInterface):
    """
    RT-2风格模型客户端
    
    RT-2使用Vision-Language Model架构，将动作表示为文本token
    """
    
    def __init__(
        self,
        remote_config: RemoteServerConfig,
        model_config: Optional[VLAModelConfig] = None,
        use_token_output: bool = True
    ):
        """
        初始化RT-2客户端
        
        Args:
            remote_config: 远程服务器配置
            model_config: 模型配置
            use_token_output: 是否使用token格式输出（vs直接数值）
        """
        self._remote_config = remote_config
        self._model_config = model_config or VLAModelConfig(
            action_bins=512  # RT-2默认使用512 bins
        )
        self._client = RESTClient(remote_config)
        self._tokenizer = RT2ActionTokenizer(
            num_bins=self._model_config.action_bins,
            action_dim=self._model_config.action_dim
        )
        self._use_token_output = use_token_output
        self._is_connected = False
        
        # 动作历史（RT-2可能使用历史信息）
        self._action_history: List[Action] = []
        self._max_history = 5
        
    @property
    def model_name(self) -> str:
        return self._model_config.model_name
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected and self._client.is_connected
    
    def connect(self) -> bool:
        """连接到远程服务器"""
        self._is_connected = self._client.connect()
        return self._is_connected
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像
        
        RT-2使用320x320图像（PaLI-X输入尺寸）
        """
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:
            image = image[:, :, :3]
        
        pil_image = Image.fromarray(image.astype(np.uint8))
        pil_image = pil_image.resize((320, 320), Image.Resampling.LANCZOS)
        
        return np.array(pil_image)
    
    def build_prompt(self, instruction: str) -> str:
        """
        构建RT-2风格的prompt
        
        RT-2使用特定格式：
        "What action should the robot take to [instruction]?"
        """
        return f"What action should the robot take to {instruction}?"
    
    def parse_model_output(self, output: Dict[str, Any]) -> Action:
        """
        解析模型输出
        
        处理token格式或直接数值格式的输出
        """
        if self._use_token_output and "token_string" in output:
            # 从token字符串解码
            action_values = self._tokenizer.decode_from_string(output["token_string"])
        elif "action" in output:
            # 直接使用数值
            action_values = np.array(output["action"])
        else:
            raise ValueError(f"Unexpected model output format: {output}")
        
        return Action(
            values=action_values,
            action_type=output.get("action_type", "delta_ee")
        )
    
    def predict(self, observation: Observation, instruction: str) -> Action:
        """
        根据观测和语言指令预测动作
        
        Args:
            observation: 当前观测
            instruction: 语言指令
            
        Returns:
            预测的动作
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to server")
        
        # 预处理
        processed_image = self.preprocess_image(observation.image)
        prompt = self.build_prompt(instruction)
        
        processed_obs = Observation(
            image=processed_image,
            depth=observation.depth,
            joint_positions=observation.joint_positions,
            ee_position=observation.ee_position,
            gripper_state=observation.gripper_state
        )
        
        # 发送请求
        raw_action = self._client.predict(
            processed_obs,
            prompt,
            model_name=self._model_config.model_name,
            temperature=self._model_config.temperature,
            max_tokens=self._model_config.max_tokens
        )
        
        # 更新历史
        self._action_history.append(raw_action)
        if len(self._action_history) > self._max_history:
            self._action_history.pop(0)
        
        return raw_action
    
    def predict_with_history(
        self,
        observation: Observation,
        instruction: str,
        include_history: bool = True
    ) -> Action:
        """
        使用历史信息预测动作
        
        某些VLA模型可以利用历史动作来提高一致性
        """
        if not include_history or not self._action_history:
            return self.predict(observation, instruction)
        
        # 构建包含历史的上下文
        history_str = " ".join(
            f"[Step {i}: {a.values.tolist()}]"
            for i, a in enumerate(self._action_history[-3:])
        )
        
        enhanced_instruction = f"{instruction} Previous actions: {history_str}"
        return self.predict(observation, enhanced_instruction)
    
    def reset(self) -> None:
        """重置模型状态，清除历史"""
        self._action_history.clear()
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = self._client.get_model_info()
        info["tokenizer_config"] = {
            "num_bins": self._tokenizer.config.num_bins,
            "action_dim": self._tokenizer.config.action_dim,
        }
        return info
    
    def close(self):
        """关闭连接"""
        self._client.close()
        self._is_connected = False
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
