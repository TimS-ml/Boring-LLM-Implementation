from jaxtyping import Float
from pydantic import Field
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from boring_llm.nn.ffn.base import FeedForwardTransform
from boring_llm.nn.ffn.factory import FeedForwardFactory, FeedForwardConfigFactory


class GLULayer(nn.Module):
    """
    Gated Linear Unit layer
    
    As described in papers like "GLU Variants Improve Transformer" 
    https://arxiv.org/abs/2002.05202
    """
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        activation: callable = nn.Identity(),
        mult_bias: bool = True,
        no_bias: bool = False
    ):
        super().__init__()
        
        # we are going to chunk the output into two parts later
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=not no_bias)
        self.act = activation
        
        # Multiplicative bias parameter (ST-MOE's bias improvement)
        self.mult_bias = nn.Parameter(torch.ones(dim_out)) if mult_bias else 1.

    def forward(self, x: Float[Tensor, "batch ... dim"]) -> Float[Tensor, "batch ... dim"]:
        # Split projection into value and gate
        value, gate = self.proj(x).chunk(2, dim=-1)
        
        # Apply gate and multiplicative bias
        return value * self.act(gate) * self.mult_bias


FeedForwardConfigFactory.register("glu")({
    "mult_bias": (bool, Field(default=True, description="Whether to use multiplicative bias in GLU (ST-MOE bias improvement)"))
})
@FeedForwardFactory.register("glu")
class GLUTransformation(FeedForwardTransform):
    """Feed-forward transformation with Gated Linear Unit"""
    
    def __init__(
        self, 
        dim_model: int,
        inner_dim: int,
        activation: callable = nn.SiLU(),
        no_bias: bool = False,
        mult_bias: bool = True,
        **kwargs
    ):
        super().__init__()
        self.dim = dim_model
        self.inner_dim = inner_dim
        
        self.glu = GLULayer(
            dim_in=dim_model,
            dim_out=inner_dim,
            activation=activation,
            mult_bias=mult_bias,
            no_bias=no_bias
        )
    
    def apply(self, x: Float[Tensor, "batch ... dim"]) -> Float[Tensor, "batch ... dim"]:
        return self.glu(x)
    
    @property
    def output_dim(self) -> int:
        return self.inner_dim
