# VLA Platform - REST API Client
"""
REST API客户端实现，用于与远程VLA服务器通信
兼容OpenVLA官方API格式
"""
import requests
import base64
import numpy as np
from typing import Optional, Dict, Any
from io import BytesIO
from PIL import Image
import time
import logging

from ..core.config import RemoteServerConfig
from ..core.base_interfaces import Observation, Action

logger = logging.getLogger(__name__)


class RESTClient:
    """REST API客户端，用于远程VLA模型推理"""
    
    def __init__(self, config: RemoteServerConfig):
        """
        初始化REST客户端
        
        Args:
            config: 远程服务器配置
        """
        self.config = config
        self.session = requests.Session()
        self._connected = False
        
    @property
    def base_url(self) -> str:
        return self.config.base_url
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    def connect(self) -> bool:
        """测试与服务器的连接"""
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=self.config.timeout
            )
            self._connected = response.status_code == 200
            return self._connected
        except requests.RequestException as e:
            logger.error(f"Connection failed: {e}")
            self._connected = False
            return False
    
    def _encode_image(self, image: np.ndarray) -> str:
        """将numpy图像编码为base64字符串"""
        pil_image = Image.fromarray(image.astype(np.uint8))
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def _decode_action(self, response_data: Dict[str, Any]) -> Action:
        """解析服务器返回的动作数据"""
        # OpenVLA格式: {"action": [7个浮点数]}
        action_values = np.array(response_data.get("action", [0.0] * 7))
        action_type = response_data.get("action_type", "delta_ee")
        return Action(values=action_values, action_type=action_type)
    
    def predict(
        self,
        observation: Observation,
        instruction: str,
        model_name: Optional[str] = None,
        **kwargs
    ) -> Action:
        """
        发送推理请求到远程服务器
        
        Args:
            observation: 当前观测
            instruction: 语言指令
            model_name: 可选的模型名称
            **kwargs: 其他参数
            
        Returns:
            预测的动作
        """
        # 构建请求payload
        payload = {
            "image": self._encode_image(observation.image),
            "instruction": instruction,
        }
        
        if model_name:
            payload["model"] = model_name
        
        # 添加可选的机器人状态信息
        if observation.joint_positions is not None:
            payload["joint_positions"] = observation.joint_positions.tolist()
        if observation.ee_position is not None:
            payload["ee_position"] = observation.ee_position.tolist()
        if observation.gripper_state is not None:
            payload["gripper_state"] = observation.gripper_state
            
        # 添加其他参数
        payload.update(kwargs)
        
        # 发送请求（支持重试）
        for attempt in range(self.config.max_retries):
            try:
                start_time = time.time()
                response = self.session.post(
                    f"{self.base_url}/predict",
                    json=payload,
                    timeout=self.config.timeout
                )
                latency = time.time() - start_time
                logger.debug(f"Inference latency: {latency*1000:.1f}ms")
                
                response.raise_for_status()
                return self._decode_action(response.json())
                
            except requests.RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_retries - 1:
                    raise RuntimeError(f"All {self.config.max_retries} attempts failed") from e
                time.sleep(0.5 * (attempt + 1))  # 指数退避
        
        # 不应该到达这里
        raise RuntimeError("Unexpected error in predict")
    
    def batch_predict(
        self,
        observations: list,
        instructions: list,
        **kwargs
    ) -> list:
        """
        批量推理请求
        
        Args:
            observations: 观测列表
            instructions: 指令列表
            **kwargs: 其他参数
            
        Returns:
            动作列表
        """
        if len(observations) != len(instructions):
            raise ValueError("observations和instructions长度必须相同")
        
        payloads = []
        for obs, inst in zip(observations, instructions):
            payloads.append({
                "image": self._encode_image(obs.image),
                "instruction": inst,
            })
        
        try:
            response = self.session.post(
                f"{self.base_url}/batch_predict",
                json={"requests": payloads, **kwargs},
                timeout=self.config.timeout * len(payloads)
            )
            response.raise_for_status()
            
            results = response.json().get("results", [])
            return [self._decode_action(r) for r in results]
            
        except requests.RequestException as e:
            raise RuntimeError(f"Batch prediction failed: {e}") from e
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取远程模型信息"""
        try:
            response = self.session.get(
                f"{self.base_url}/model_info",
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to get model info: {e}")
            return {}
    
    def close(self):
        """关闭会话"""
        self.session.close()
        self._connected = False
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
