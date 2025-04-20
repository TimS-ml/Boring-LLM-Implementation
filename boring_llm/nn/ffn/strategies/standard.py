from jaxtyping import Float
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from boring_llm.nn.ffn.base import FeedForwardTransform
from boring_llm.nn.ffn.factory import FeedForwardFactory, FeedForwardConfigFactory


FeedForwardConfigFactory.register("standard")({})
@FeedForwardFactory.register("standard")
class StandardTransformation(FeedForwardTransform):
    """Standard feed-forward transformation"""
    def __init__(
        self, 
        dim_model: int,
        inner_dim: int,
        activation: callable = nn.Identity(),
        no_bias: bool = False,
        **kwargs
    ):
        super().__init__()
        self.dim = dim_model
        self.inner_dim = inner_dim
        
        self.proj = nn.Linear(dim_model, inner_dim, bias=not no_bias)
        self.act = activation
    
    def apply(self, x: Float[Tensor, "batch ... dim"]) -> Float[Tensor, "batch ... inner_dim"]:
        return self.act(self.proj(x))
    
    @property
    def output_dim(self) -> int:
        return self.inner_dim 