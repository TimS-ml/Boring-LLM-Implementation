from enum import Enum
from pydantic import Field, field_validator, create_model
from typing import Optional, Type, List, Union, Literal

from boring_llm.base.base_config import BaseConfig
from boring_utils.utils import cprint
from boring_utils.helpers import DEBUG
from boring_llm.nn.ffn.factory import FeedForwardFactory, FeedForwardConfigFactory


class ActivationType(str, Enum):
    """Activation function type enum"""
    RELU = "relu"
    GELU = "gelu"
    SWISH = "swish"
    SILU = "silu"  # equivalent to SWISH
    RELU_SQUARED = "relu_squared"
    SIGMOID = "sigmoid"
    TANH = "tanh"


class ActivationConfig(BaseConfig):
    """Activation function configuration"""
    type: Union[ActivationType, str] = Field(
        default=ActivationType.GELU, 
        description="Activation function type"
    )
    use_glu: bool = Field(
        default=False, 
        description="Whether to use gated linear units (GLU)"
    )
    mult_bias: bool = Field(
        default=True, 
        description="Whether to use multiplicative bias in GLU (ST-MOE bias improvement)"
    )
    
    # Additional activation function specific parameters can be added here
    inplace: bool = Field(
        default=False,
        description="inplace parameter for PyTorch activation functions"
    )
    
    def get_type_value(self) -> str:
        """Get type value, handle enum or string type"""
        if isinstance(self.type, str):
            return self.type
        return self.type.value


class FeedForwardConfig(BaseConfig):
    """Feed forward network configuration"""
    type: str = Field(default="normal")
    
    @field_validator('type')
    def validate_type(cls, v):
        if not FeedForwardFactory.is_valid_type(v):
            valid_types = FeedForwardFactory.get_available_types()
            raise ValueError(f"Invalid feed forward type: {v}. Valid types: {valid_types}")
        return v

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
        description="Whether to use LayerNorm after activation function"
    )
    no_bias: bool = Field(
        default=False, 
        description="Whether to remove the bias of the linear layer"
    )
    dropout: float = Field(
        default=0.0, 
        description="Dropout probability"
    )
    zero_init_output: bool = Field(
        default=False, 
        description="Whether to initialize the output layer to zero"
    )
    activation: ActivationConfig = Field(
        default_factory=ActivationConfig, 
        description="Activation function configuration"
    )


def create_ffn_config(ffn_type: str) -> Type[FeedForwardConfig]:
    fields = {
        "type": (str, Field(default=ffn_type, const=True)),
        "ffn_dim_out": (Optional[int], Field(default=None)),
        "mult_dim": (int, Field(default=4)),
        "post_act_ln": (bool, Field(default=False)),
        "no_bias": (bool, Field(default=False)),
        "dropout": (float, Field(default=0.0)),
        "zero_init_output": (bool, Field(default=False)),
        "activation": (ActivationConfig, Field(default_factory=ActivationConfig))
    }

    type_fields = FeedForwardConfigFactory.get_config_fields(ffn_type)
    fields.update(type_fields)
    
    return create_model(f"{ffn_type.capitalize()}FeedForwardConfig", 
                       __base__=FeedForwardConfig, 
                       **fields)