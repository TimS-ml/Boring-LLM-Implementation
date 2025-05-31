"""
FFN Transform Registry
Contains all FFN transformation strategies and their registry
"""
from typing import Callable
from pydantic import Field
import torch
import torch.nn as nn
from torch import Tensor

from boring_llm.base.component_registry import ComponentTransform, ComponentRegistry


# ============= FFN Transform Base =============
class FFNTransform(ComponentTransform):
    """Base class for FFN transformations"""
    
    @property
    def output_dim(self) -> int:
        """FFN transforms must specify output dimension"""
        raise NotImplementedError


# ============= Registry Setup =============
ffn_registry = ComponentRegistry[FFNTransform]("FFN")


# ============= FFN Strategies =============
@ffn_registry.register("standard")
class StandardFFN(FFNTransform):
    """Standard feed-forward transformation"""
    
    def __init__(self, dim_model: int, inner_dim: int, activation: Callable = nn.GELU, 
                 no_bias: bool = False, **kwargs):
        super().__init__()
        self.proj = nn.Linear(dim_model, inner_dim, bias=not no_bias)
        self.act = activation() if callable(activation) and not isinstance(activation, nn.Module) else activation
        
    def apply(self, x: Tensor, **kwargs) -> Tensor:
        return self.act(self.proj(x))
    
    @property
    def output_dim(self) -> int:
        return self.proj.out_features


@ffn_registry.register("glu", {
    "mult_bias": (bool, Field(default=True, description="Whether to use multiplicative bias in GLU"))
})
class GLUFFN(FFNTransform):
    """Feed-forward transformation with Gated Linear Unit"""
    
    def __init__(self, dim_model: int, inner_dim: int, activation: Callable = nn.SiLU, 
                 mult_bias: bool = True, no_bias: bool = False, **kwargs):
        super().__init__()
        self.proj = nn.Linear(dim_model, inner_dim * 2, bias=not no_bias)
        self.act = activation() if callable(activation) and not isinstance(activation, nn.Module) else activation
        self.mult_bias = nn.Parameter(torch.ones(inner_dim)) if mult_bias else 1.0
        self._inner_dim = inner_dim
        
    def apply(self, x: Tensor, **kwargs) -> Tensor:
        value, gate = self.proj(x).chunk(2, dim=-1)
        return value * self.act(gate) * self.mult_bias
    
    @property
    def output_dim(self) -> int:
        return self._inner_dim


@ffn_registry.register("post_standard")
class PostProcessor(FFNTransform):
    """Standard post-processor for feed-forward networks"""
    
    def __init__(self, dim_model: int, inner_dim: int, dropout: float = 0.0, 
                 post_act_ln: bool = False, no_bias: bool = False, 
                 zero_init_output: bool = False, **kwargs):
        super().__init__()
        layers = []
        
        if post_act_ln: 
            layers.append(nn.LayerNorm(inner_dim))
        if dropout > 0: 
            layers.append(nn.Dropout(dropout))
        
        self.proj = nn.Linear(inner_dim, dim_model, bias=not no_bias)
        layers.append(self.proj)
        
        if zero_init_output:
            nn.init.zeros_(self.proj.weight)
            if not no_bias:
                nn.init.zeros_(self.proj.bias)
                
        self.sequence = nn.Sequential(*layers)
    
    def apply(self, x: Tensor, **kwargs) -> Tensor:
        return self.sequence(x)
    
    @property
    def output_dim(self) -> int:
        return self.proj.out_features
