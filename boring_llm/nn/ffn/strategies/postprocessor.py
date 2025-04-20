from jaxtyping import Float
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from boring_llm.nn.ffn.base import FeedForwardTransform
from boring_llm.nn.ffn.factory import FeedForwardFactory, FeedForwardConfigFactory


FeedForwardConfigFactory.register("post_standard")({})
@FeedForwardFactory.register("post_standard")
class StandardPostProcessor(FeedForwardTransform):
    """Standard post-processor for feed-forward networks"""
    
    def __init__(
        self, 
        dim_model: int,
        inner_dim: int,
        dropout: float = 0.0,
        post_act_ln: bool = False,
        no_bias: bool = False,
        zero_init_output: bool = False,
        **kwargs
    ):
        super().__init__()
        self.dim = dim_model
        self.inner_dim = inner_dim
        self.output_dim_value = dim_model
        
        layers = []
        if post_act_ln: layers.append(nn.LayerNorm(inner_dim))
        if dropout > 0: layers.append(nn.Dropout(dropout))
        self.proj = nn.Linear(inner_dim, dim_model, bias=not no_bias)
        layers.append(self.proj)
        
        if zero_init_output:
            nn.init.zeros_(self.proj.weight)
            if not no_bias:
                nn.init.zeros_(self.proj.bias)
                
        self.sequence = nn.Sequential(*layers)
    
    def apply(self, x: Float[Tensor, "batch ... dim"]) -> Float[Tensor, "batch ... out_dim"]:
        return self.sequence(x)
    
    @property
    def output_dim(self) -> int:
        return self.output_dim_value