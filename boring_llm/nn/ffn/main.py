import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange

from torch import Tensor
from typing import Optional, Tuple, Union, List, Literal

from nn.ffn.core import FeedForwardConfig, ActivationType
from boring_utils.utils import cprint
from boring_utils.helpers import DEBUG

from boring_llm.nn.ffn.base import FeedForward
from boring_llm.nn.ffn.factory import FeedForwardFactory


class GLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, config: FeedForwardConfig):
        super().__init__()
        activation_type = config.activation.type
        no_bias = config.no_bias
        self.act = BoringFeedForward.get_activation(activation_type)
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=not no_bias)
        self.mult_bias = nn.Parameter(torch.ones(dim_out)) if config.activation.mult_bias else 1.

    def forward(self, x: Tensor) -> Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.act(gate) * self.mult_bias


class ReluSquared(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x)**2


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
        
        # Create the appropriate FFN implementation based on config
        self.ffn_strategy = FeedForwardFactory.create(
            ffn_type=ffn_type,
            dim=dim,
            dim_out=dim_out,
            mult=config.mult_dim,
            activation_type=config.activation.type,
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
        if activation_type == ActivationType.RELU:
            return nn.ReLU()
        elif activation_type == ActivationType.GELU:
            return nn.GELU()
        elif activation_type == ActivationType.SWISH:
            return nn.SiLU()
        elif activation_type == ActivationType.RELU_SQUARED:
            # return lambda x: F.relu(x)**2  # causes an error for not being a nn.Module
            return ReluSquared()
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")
