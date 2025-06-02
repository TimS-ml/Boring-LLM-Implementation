from typing import Callable
from pydantic import Field
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from boring_llm.base.component_registry import ComponentTransform, ComponentRegistry


class FFNTransform(ComponentTransform):
    """Base class for FFN transformations"""
    
    @property
    def output_dim(self) -> int:
        """FFN transforms must specify output dimension"""
        raise NotImplementedError


ffn_registry = ComponentRegistry[FFNTransform]("FFN")


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


@ffn_registry.register("router", {
    "num_experts": (int, Field(default=8, description="Number of experts")),
    "top_k": (int, Field(default=2, description="Number of experts to route to")),
    "noise_std": (float, Field(default=1.0, description="Noise standard deviation for load balancing"))
})
class RouterTransform(FFNTransform):
    """Router/Gating network for MOE"""
    
    def __init__(self, dim_model: int, num_experts: int = 8, top_k: int = 2, 
                 noise_std: float = 1.0, no_bias: bool = False, **kwargs):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        
        # Main routing layer
        self.gate = nn.Linear(dim_model, num_experts, bias=not no_bias)
        # Noise layer for load balancing (only used if noise_std > 0)
        if noise_std > 0:
            self.noise_gate = nn.Linear(dim_model, num_experts, bias=not no_bias)
        else:
            self.noise_gate = None
        
    def apply(self, x: Tensor, **kwargs) -> tuple:
        """Returns routing weights and indices"""
        logits = self.gate(x)
        
        # Add noise for load balancing during training
        if self.training and self.noise_std > 0 and self.noise_gate is not None:
            noise_logits = self.noise_gate(x)
            noise = torch.randn_like(logits) * F.softplus(noise_logits) * self.noise_std
            logits = logits + noise
            
        # Get top-k experts
        top_k_logits, indices = torch.topk(logits, self.top_k, dim=-1)
        
        # Create sparse representation
        zeros = torch.full_like(logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        routing_weights = F.softmax(sparse_logits, dim=-1)
        
        return routing_weights, indices
    
    @property
    def output_dim(self) -> int:
        return self.num_experts


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
