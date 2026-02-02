# VLA Platform - Action Tokenizer
"""
动作离散化/连续化转换
兼容OpenVLA和RT-2的动作token格式
"""
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ActionTokenizerConfig:
    """动作分词器配置"""
    num_bins: int = 256  # 离散化bin数量
    action_dim: int = 7  # 动作维度
    action_min: float = -1.0  # 动作最小值
    action_max: float = 1.0  # 动作最大值
    vocab_offset: int = 0  # token偏移量（用于与语言模型词表合并）


class ActionTokenizer:
    """
    动作分词器
    
    将连续动作离散化为token，或将token解码为连续动作。
    与OpenVLA和RT-2的动作表示兼容。
    """
    
    def __init__(self, config: Optional[ActionTokenizerConfig] = None):
        """
        初始化动作分词器
        
        Args:
            config: 分词器配置
        """
        self.config = config or ActionTokenizerConfig()
        
        # 计算bin边界
        self._bin_edges = np.linspace(
            self.config.action_min,
            self.config.action_max,
            self.config.num_bins + 1
        )
        # bin中心值（用于解码）
        self._bin_centers = (self._bin_edges[:-1] + self._bin_edges[1:]) / 2
        
    @property
    def vocab_size(self) -> int:
        """动作词表大小"""
        return self.config.num_bins
    
    def encode(self, action: np.ndarray) -> np.ndarray:
        """
        将连续动作编码为token
        
        Args:
            action: 连续动作 shape (action_dim,) 或 (batch, action_dim)
            
        Returns:
            离散token shape (action_dim,) 或 (batch, action_dim)
        """
        # 裁剪到有效范围
        action = np.clip(action, self.config.action_min, self.config.action_max)
        
        # 离散化
        tokens = np.digitize(action, self._bin_edges[1:-1])
        
        # 添加偏移量
        tokens = tokens + self.config.vocab_offset
        
        return tokens.astype(np.int32)
    
    def decode(self, tokens: np.ndarray) -> np.ndarray:
        """
        将token解码为连续动作
        
        Args:
            tokens: 离散token shape (action_dim,) 或 (batch, action_dim)
            
        Returns:
            连续动作
        """
        # 移除偏移量
        tokens = tokens - self.config.vocab_offset
        
        # 裁剪到有效token范围
        tokens = np.clip(tokens, 0, self.config.num_bins - 1)
        
        # 映射到bin中心
        return self._bin_centers[tokens]
    
    def encode_batch(self, actions: np.ndarray) -> np.ndarray:
        """批量编码"""
        return self.encode(actions)
    
    def decode_batch(self, tokens: np.ndarray) -> np.ndarray:
        """批量解码"""
        return self.decode(tokens)
    
    def to_normalized(
        self,
        action: np.ndarray,
        low: np.ndarray,
        high: np.ndarray
    ) -> np.ndarray:
        """
        将动作从原始范围归一化到[-1, 1]
        
        Args:
            action: 原始动作
            low: 原始动作下界
            high: 原始动作上界
            
        Returns:
            归一化动作
        """
        return 2.0 * (action - low) / (high - low) - 1.0
    
    def from_normalized(
        self,
        action: np.ndarray,
        low: np.ndarray,
        high: np.ndarray
    ) -> np.ndarray:
        """
        将动作从[-1, 1]反归一化到原始范围
        
        Args:
            action: 归一化动作
            low: 原始动作下界
            high: 原始动作上界
            
        Returns:
            原始动作
        """
        return (action + 1.0) / 2.0 * (high - low) + low


class RT2ActionTokenizer(ActionTokenizer):
    """
    RT-2风格的动作分词器
    
    RT-2使用512个bin，并有特殊的token格式
    """
    
    # RT-2使用的特殊token
    ACTION_TOKEN_BEGIN = "<action>"
    ACTION_TOKEN_END = "</action>"
    
    def __init__(
        self,
        num_bins: int = 512,
        action_dim: int = 7,
        vocab_offset: int = 32000  # Llama词表大小
    ):
        """
        初始化RT-2动作分词器
        
        Args:
            num_bins: 离散化bin数量
            action_dim: 动作维度
            vocab_offset: 在语言模型词表后的偏移量
        """
        config = ActionTokenizerConfig(
            num_bins=num_bins,
            action_dim=action_dim,
            vocab_offset=vocab_offset
        )
        super().__init__(config)
    
    def encode_to_string(self, action: np.ndarray) -> str:
        """
        将动作编码为RT-2格式的token字符串
        
        例如: "<action>128 64 192 128 128 128 255</action>"
        """
        tokens = self.encode(action)
        token_str = " ".join(str(t) for t in tokens)
        return f"{self.ACTION_TOKEN_BEGIN}{token_str}{self.ACTION_TOKEN_END}"
    
    def decode_from_string(self, token_string: str) -> np.ndarray:
        """
        从RT-2格式的token字符串解码动作
        """
        # 提取token部分
        start = token_string.find(self.ACTION_TOKEN_BEGIN)
        end = token_string.find(self.ACTION_TOKEN_END)
        
        if start == -1 or end == -1:
            raise ValueError(f"Invalid token string: {token_string}")
        
        token_part = token_string[start + len(self.ACTION_TOKEN_BEGIN):end]
        tokens = np.array([int(t) for t in token_part.split()])
        
        return self.decode(tokens)


class OpenVLAActionTokenizer(ActionTokenizer):
    """
    OpenVLA风格的动作分词器
    
    OpenVLA使用256个bin
    """
    
    def __init__(
        self,
        num_bins: int = 256,
        action_dim: int = 7,
    ):
        """
        初始化OpenVLA动作分词器
        """
        config = ActionTokenizerConfig(
            num_bins=num_bins,
            action_dim=action_dim,
            vocab_offset=0
        )
        super().__init__(config)
    
    def encode_for_model(
        self,
        action: np.ndarray,
        action_low: np.ndarray,
        action_high: np.ndarray
    ) -> np.ndarray:
        """
        编码动作用于模型训练
        
        先归一化到[-1, 1]，再离散化
        """
        normalized = self.to_normalized(action, action_low, action_high)
        return self.encode(normalized)
    
    def decode_from_model(
        self,
        tokens: np.ndarray,
        action_low: np.ndarray,
        action_high: np.ndarray
    ) -> np.ndarray:
        """
        从模型输出解码动作
        
        先解码到[-1, 1]，再反归一化
        """
        normalized = self.decode(tokens)
        return self.from_normalized(normalized, action_low, action_high)
