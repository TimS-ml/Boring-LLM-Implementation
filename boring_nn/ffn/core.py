from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange

from torch import Tensor
from typing import Optional, Tuple, Union

from boring_llm_base.base_config import BaseConfig
from boring_utils.utils import cprint
from boring_utils.helpers import DEBUG


class ActivationType(Enum):
    RELU    = "relu"
    GELU    = "gelu"
    SWISH   = "swish"
    GLU     = "glu"
    RELU_SQUARED = "relu_squared"


class ActivationConfig(BaseModel):
    type: ActivationType = Field(ActivationType.GELU, description="Type of activation function")
    use_glu: bool        = Field(False,               description="Whether to use GLU variant")


class FeedForwardConfig(BaseConfig):
    # ffn_dim: int                   = Field(2048,  description="Feed-forward network dimension")
    ffn_dim: int = Field(default=2048, gt=0, description="Feed-forward network dimension")
    ffn_dim_out: Optional[int]     = Field(None,  description="Output dimension (if None, same as input)")
    mult: int                      = Field(4,     description="Multiplier for inner dimension")
    post_act_ln: bool              = Field(False, description="Whether to use LayerNorm after activation")
    no_bias: bool                  = Field(False, description="Whether to remove bias from linear layers")
    zero_init_output: bool         = Field(False, description="Whether to initialize output layer to zero")
    activation: ActivationConfig   = Field(default_factory=ActivationConfig, 
                                                  description="Activation function configuration")


class BoringFeedForward(nn.Module):
    def __init__(self, config: FeedForwardConfig):
        super().__init__()
        self.config = config
        
        dim = config.d_model
        dim_out = config.ffn_dim_out or dim
        mult = config.mult
        inner_dim = int(dim * mult)

        activation_type = config.activation.type
        use_glu = config.activation.use_glu

        if use_glu:
            self.glu = GLU(dim, inner_dim, activation_type)
            project_in = self.glu
        else:
            activation = self.get_activation(activation_type)
            project_in = nn.Sequential(
                nn.Linear(dim, inner_dim, bias=not config.no_bias),
                activation
            )

        self.net = nn.Sequential(
            project_in,
            nn.LayerNorm(inner_dim) if config.post_act_ln else nn.Identity(),
            nn.Dropout(config.dropout),
            nn.Linear(inner_dim, dim_out, bias=not config.no_bias)
        )

        if config.zero_init_output:
            nn.init.zeros_(self.net[-1].weight)

    def get_activation(self, activation_type: ActivationType):
        if activation_type == ActivationType.RELU:
            return nn.ReLU()
        elif activation_type == ActivationType.GELU:
            return nn.GELU()
        elif activation_type == ActivationType.SWISH:
            return nn.SiLU()
        elif activation_type == ActivationType.RELU_SQUARED:
            return lambda x: F.relu(x)**2
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")

    def forward(self, x):
        return self.net(x)


class GLU(nn.Module):
    def __init__(self, dim_in, dim_out, activation_type):
        super().__init__()
        self.act = BoringFeedForward.get_activation(activation_type)
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.act(gate)
