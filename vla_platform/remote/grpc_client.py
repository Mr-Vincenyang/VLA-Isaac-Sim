# VLA Platform - gRPC Client
"""
gRPC客户端实现，用于高性能远程VLA模型推理
"""
import grpc
import numpy as np
from typing import Optional, Dict, Any, Iterator
from io import BytesIO
from PIL import Image
import time
import logging

from ..core.config import RemoteServerConfig
from ..core.base_interfaces import Observation, Action

logger = logging.getLogger(__name__)

# 注意: 需要先编译proto文件生成Python模块
# python -m grpc_tools.protoc -I./protos --python_out=./protos --grpc_python_out=./protos ./protos/vla_service.proto
try:
    from .protos import vla_service_pb2
    from .protos import vla_service_pb2_grpc
    GRPC_AVAILABLE = True
except ImportError:
    logger.warning("gRPC proto modules not found. Run proto compilation first.")
    GRPC_AVAILABLE = False


class GRPCClient:
    """gRPC客户端，用于高性能远程VLA模型推理"""
    
    def __init__(self, config: RemoteServerConfig):
        """
        初始化gRPC客户端
        
        Args:
            config: 远程服务器配置
        """
        if not GRPC_AVAILABLE:
            raise RuntimeError("gRPC modules not available. Compile proto files first.")
            
        self.config = config
        self.channel: Optional[grpc.Channel] = None
        self.stub = None
        self._connected = False
        
    @property
    def server_address(self) -> str:
        return f"{self.config.host}:{self.config.port}"
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    def connect(self) -> bool:
        """建立gRPC连接"""
        try:
            # 创建grpc channel
            self.channel = grpc.insecure_channel(
                self.server_address,
                options=[
                    ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
                    ('grpc.max_receive_message_length', 50 * 1024 * 1024),
                ]
            )
            
            # 创建stub
            self.stub = vla_service_pb2_grpc.VLAServiceStub(self.channel)
            
            # 健康检查
            response = self.stub.HealthCheck(
                vla_service_pb2.Empty(),
                timeout=self.config.timeout
            )
            self._connected = response.is_healthy
            return self._connected
            
        except grpc.RpcError as e:
            logger.error(f"gRPC connection failed: {e}")
            self._connected = False
            return False
    
    def _encode_image(self, image: np.ndarray) -> bytes:
        """将numpy图像编码为PNG字节"""
        pil_image = Image.fromarray(image.astype(np.uint8))
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        return buffer.getvalue()
    
    def _build_request(
        self,
        observation: Observation,
        instruction: str,
        model_name: Optional[str] = None,
        **kwargs
    ) -> 'vla_service_pb2.PredictRequest':
        """构建gRPC请求"""
        request = vla_service_pb2.PredictRequest(
            image=self._encode_image(observation.image),
            instruction=instruction,
        )
        
        if model_name:
            request.model_name = model_name
            
        if observation.joint_positions is not None:
            request.joint_positions.extend(observation.joint_positions.tolist())
        if observation.ee_position is not None:
            request.ee_position.extend(observation.ee_position.tolist())
        if observation.gripper_state is not None:
            request.gripper_state = observation.gripper_state
            
        # 添加推理参数
        if 'temperature' in kwargs:
            request.temperature = kwargs['temperature']
        if 'max_tokens' in kwargs:
            request.max_tokens = kwargs['max_tokens']
            
        return request
    
    def predict(
        self,
        observation: Observation,
        instruction: str,
        model_name: Optional[str] = None,
        **kwargs
    ) -> Action:
        """
        发送推理请求
        
        Args:
            observation: 当前观测
            instruction: 语言指令
            model_name: 可选的模型名称
            **kwargs: 其他参数
            
        Returns:
            预测的动作
        """
        if not self._connected or self.stub is None:
            raise RuntimeError("Not connected to server")
        
        request = self._build_request(observation, instruction, model_name, **kwargs)
        
        try:
            start_time = time.time()
            response = self.stub.Predict(request, timeout=self.config.timeout)
            latency = time.time() - start_time
            logger.debug(f"gRPC inference latency: {latency*1000:.1f}ms")
            
            if response.error:
                raise RuntimeError(f"Server error: {response.error}")
            
            return Action(
                values=np.array(response.action),
                action_type=response.action_type or "delta_ee"
            )
            
        except grpc.RpcError as e:
            raise RuntimeError(f"gRPC call failed: {e}") from e
    
    def stream_predict(
        self,
        observations: Iterator[Observation],
        instructions: Iterator[str],
        **kwargs
    ) -> Iterator[Action]:
        """
        流式推理（用于低延迟场景）
        
        Args:
            observations: 观测迭代器
            instructions: 指令迭代器
            **kwargs: 其他参数
            
        Yields:
            预测的动作
        """
        if not self._connected or self.stub is None:
            raise RuntimeError("Not connected to server")
        
        def request_generator():
            for obs, inst in zip(observations, instructions):
                yield self._build_request(obs, inst, **kwargs)
        
        try:
            for response in self.stub.StreamPredict(request_generator()):
                if response.error:
                    logger.warning(f"Stream error: {response.error}")
                    continue
                yield Action(
                    values=np.array(response.action),
                    action_type=response.action_type or "delta_ee"
                )
        except grpc.RpcError as e:
            raise RuntimeError(f"Stream prediction failed: {e}") from e
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取远程模型信息"""
        if not self._connected or self.stub is None:
            return {}
        
        try:
            response = self.stub.GetModelInfo(
                vla_service_pb2.Empty(),
                timeout=self.config.timeout
            )
            return {
                "model_name": response.model_name,
                "model_version": response.model_version,
                "action_dim": response.action_dim,
                "action_bins": response.action_bins,
                "device": response.device,
                "quantization": response.quantization,
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to get model info: {e}")
            return {}
    
    def close(self):
        """关闭连接"""
        if self.channel:
            self.channel.close()
        self._connected = False
        self.stub = None
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
