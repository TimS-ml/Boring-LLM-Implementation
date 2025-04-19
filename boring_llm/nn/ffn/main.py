import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange

from torch import Tensor
from typing import Optional, Tuple, Union, List, Literal, Any

from boring_llm.nn.ffn.config import FeedForwardConfig, ActivationType, ActivationConfig
from boring_utils.utils import cprint
from boring_utils.helpers import DEBUG

from boring_llm.nn.ffn.base import FeedForward
from boring_llm.nn.ffn.factory import FeedForwardFactory


def get_activation(activation_type: Union[str, ActivationType], **kwargs: Any) -> nn.Module:
    """
    获取激活函数模块
    
    Args:
        activation_type: 激活函数类型（字符串或枚举）
        **kwargs: 激活函数的额外参数
    
    Returns:
        激活函数模块
    """
    # 如果是枚举类型，转换为字符串
    if isinstance(activation_type, ActivationType):
        activation_type = activation_type.value
    
    # 处理内置PyTorch激活函数    
    if activation_type == "relu":
        return nn.ReLU(**kwargs)
    elif activation_type == "gelu":
        return nn.GELU(**kwargs)
    elif activation_type == "silu" or activation_type == "swish":
        return nn.SiLU(**kwargs)
    elif activation_type == "sigmoid":
        return nn.Sigmoid(**kwargs)
    elif activation_type == "tanh":
        return nn.Tanh(**kwargs)
    elif activation_type == "relu_squared":
        from boring_llm.nn.ffn.strategies.activation import ReluSquared
        return ReluSquared()
    else:
        raise ValueError(f"Unknown activation type: {activation_type}")


def get_activation_from_config(config: ActivationConfig) -> nn.Module:
    """
    从配置对象创建激活函数
    
    Args:
        config: 激活函数配置
        
    Returns:
        激活函数模块
    """
    activation_type = config.get_type_value()
    kwargs = {}
    
    # 添加特定于激活函数的参数
    if hasattr(config, 'inplace'):
        kwargs['inplace'] = config.inplace
        
    return get_activation(activation_type, **kwargs)


class GLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, config: FeedForwardConfig):
        super().__init__()
        no_bias = config.no_bias
        # 使用get_activation_from_config直接从配置创建激活函数
        self.act = get_activation_from_config(config.activation)
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=not no_bias)
        self.mult_bias = nn.Parameter(torch.ones(dim_out)) if config.activation.mult_bias else 1.

    def forward(self, x: Tensor) -> Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.act(gate) * self.mult_bias


class BoringFeedForward(FeedForward):
    """
    Main feed-forward network module that uses strategy pattern
    to support different types of feed-forward implementations
    """
    def __init__(self, config: FeedForwardConfig):
        super().__init__()
        self.config = config
        
        # Determine the FFN type based on configuration
        if config.activation.use_glu:
            ffn_type = "glu"
        else:
            ffn_type = "standard"
        
        # Get dimensions
        dim = config.d_model
        dim_out = config.ffn_dim_out or dim
        
        # 获取激活函数类型的字符串值
        activation_type = config.activation.get_type_value()
        
        # Create the appropriate FFN implementation based on config
        self.ffn_strategy = FeedForwardFactory.create(
            ffn_type=ffn_type,
            dim=dim,
            dim_out=dim_out,
            mult=config.mult_dim,
            activation_type=activation_type,
            mult_bias=config.activation.mult_bias if ffn_type == "glu" else False,
            post_act_ln=config.post_act_ln,
            dropout=config.dropout,
            no_bias=config.no_bias,
            zero_init_output=config.zero_init_output
        )
    
    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """
        Apply feed-forward transformation to input tensor
        
        Args:
            x: Input tensor of shape [batch, seq_len, dim]
            **kwargs: Additional arguments passed to the strategy
            
        Returns:
            Transformed tensor
        """
        return self.ffn_strategy(x, **kwargs)

    @staticmethod
    def get_activation(activation_type: ActivationType):
        """
        Legacy method for getting activation function (for backward compatibility)
        
        Args:
            activation_type: Activation type enum
            
        Returns:
            Activation function module
        """
        return get_activation(activation_type)
