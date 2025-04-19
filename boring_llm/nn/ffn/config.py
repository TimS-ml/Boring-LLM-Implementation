from enum import Enum
from pydantic import Field
from typing import Optional

from boring_llm.base.base_config import BaseConfig
from boring_utils.utils import cprint
from boring_utils.helpers import DEBUG


class ActivationType(str, Enum):
    RELU = "relu"
    GELU = "gelu"
    SWISH = "swish"
    RELU_SQUARED = "relu_squared"


class ActivationConfig(BaseConfig):
    type: ActivationType = Field(
        default=ActivationType.GELU, 
        description="Type of activation function"
    )
    use_glu: bool = Field(
        default=False, 
        description="Enable Gated Linear Unit (GLU) with activation"
    )
    mult_bias: bool = Field(
        default=True, 
        description="Whether to use multiplicative bias in GLU (ST-MOE's bias improvement)"
    )


class FeedForwardConfig(BaseConfig):
    """Configuration for feed forward network"""
    # input ffn_dim = d_model
    ffn_dim_out: Optional[int] = Field(
        default=None,  
        description="Output dimension (if None, same as input)"
    )
    mult_dim: int = Field(
        default=4,     
        description="Multiplier for inner dimension"
    )
    post_act_ln: bool = Field(
        default=False, 
        description="Whether to use LayerNorm after activation"
    )
    no_bias: bool = Field(
        default=False, 
        description="Whether to remove bias from linear layers"
    )
    zero_init_output: bool = Field(
        default=False, 
        description="Whether to initialize output layer to zero"
    )
    activation: ActivationConfig = Field(
        default_factory=ActivationConfig, 
        description="Activation function configuration"
    )
