from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Type, List, Union, Literal


class ActivationType(str, Enum):
    """Activation function type enum"""
    RELU = "relu"
    GELU = "gelu"
    SWISH = "swish"
    SILU = "silu"  # equivalent to SWISH
    RELU_SQUARED = "relu_squared"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    CUSTOMIZED = "customized"


# TODO: be able to insert customized activation params
class ActivationConfig(BaseModel):
    """Activation function configuration"""
    type: Union[ActivationType, str] = Field(
        default=ActivationType.GELU, 
        description="Activation function type"
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