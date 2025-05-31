from enum import Enum
from pydantic import Field, field_validator, create_model
from typing import Optional, Type, List, Union, Literal, Callable, Any
import torch.nn as nn

from boring_llm.base.base_config import BaseConfig
from boring_llm.nn.ffn.factory import FeedForwardFactory, FeedForwardConfigFactory


class FeedForwardConfig(BaseConfig):
    """Feed forward network configuration"""
    type: str = Field(default="standard")
    post_type: str = Field(default="post_standard")
    
    @field_validator('type')
    def validate_type(cls, v):
        import boring_llm.nn.ffn.strategies
        if not FeedForwardFactory.is_valid_type(v):
            valid_types = FeedForwardFactory.get_available_types()
            raise ValueError(f"Invalid feed forward type: {v}. Valid types: {valid_types}")
        return v

    @field_validator('post_type')
    def validate_post_type(cls, v):
        import boring_llm.nn.ffn.strategies
        valid_types = [t for t in FeedForwardFactory.get_available_types() if t.startswith("post_")]
        if v not in valid_types:
            raise ValueError(f"Invalid post processor type: {v}. Valid types: {valid_types}")
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
    inner_dim: int = Field(
        default=None,
        description="Inner dimension (if None, input * mult_dim)"
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
    activation: Callable[..., Any] = Field(
        default=nn.Identity, 
        description="Activation function"
    )
    # TODO:
    # layer_dropout

    # TODO: for BoringTransformer
    # macaron
    # sandwich_coef

    def model_post_init(self, __context):
        # If a class is passed instead of an instance, instantiate it
        if callable(self.activation) and not isinstance(self.activation, nn.Module):
            self.activation = self.activation()


def create_ffn_config(ffn_type: str) -> Type[FeedForwardConfig]:
    fields = {
        # this could be any factory
        "type": (Literal[ffn_type], Field(default=ffn_type)),
        "ffn_dim_out": (Optional[int], Field(default=None)),
        "mult_dim": (int, Field(default=4)),
        "post_act_ln": (bool, Field(default=False)),
        "no_bias": (bool, Field(default=False)),
        "dropout": (float, Field(default=0.0)),
        "zero_init_output": (bool, Field(default=False)),
        "activation": (Callable[..., Any], Field(default=nn.Identity))
    }

    type_fields = FeedForwardConfigFactory.get_config_fields(ffn_type)
    fields.update(type_fields)

    return create_model(f"{ffn_type.capitalize()}FeedForwardConfig", 
                       __base__=FeedForwardConfig, 
                       **fields)