from enum import Enum
from pydantic import BaseModel, Field

import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange

from torch import Tensor
from typing import Optional, Tuple, Union, List

from boring_llm_base.base_config import BaseConfig
from boring_utils.utils import cprint
from boring_utils.helpers import DEBUG


class ActivationType(Enum):
    RELU    = "relu"
    GELU    = "gelu"
    SWISH   = "swish"
    RELU_SQUARED = "relu_squared"


class ActivationConfig(BaseModel):
    type: ActivationType      = Field(default=ActivationType.GELU, description="Type of activation function")
    use_glu: bool             = Field(default=False,               description="Enable Gated Linear Unit (GLU) with activation")
    mult_bias: Optional[bool] = Field(default=True,                description="ST-MOE's bias improvement")


class FeedForwardConfig(BaseConfig):
    # input ffn_dim = d_model
    ffn_dim_out: Optional[int]       = Field(default=None,  description="Output dimension (if None, same as input)")
    mult_dim: Optional[int]          = Field(default=4,     description="Multiplier for inner dimension")
    post_act_ln: Optional[bool]      = Field(default=False, description="Whether to use LayerNorm after activation")
    no_bias: Optional[bool]          = Field(default=False, description="Whether to remove bias from linear layers")
    zero_init_output: Optional[bool] = Field(default=False, description="Whether to initialize output layer to zero")
    activation: ActivationConfig     = Field(default_factory=ActivationConfig, 
                                                    description="Activation function configuration")

