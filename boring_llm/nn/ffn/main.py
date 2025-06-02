from typing import Optional, Callable
from pydantic import Field
import torch
import torch.nn as nn
from torch import Tensor

from boring_llm.base.component_registry import ComponentConfig
from boring_llm.nn.ffn.registry import ffn_registry
from boring_llm.nn.activation import get_activation_by_name


class FFNConfig(ComponentConfig):
    """
    FFN Configuration - inherits dim_model etc. from BaseConfig
    Notice that the `activation` is Callable
    """
    ffn_dim_out: Optional[int] = Field(default=None, description="Output dimension (if None, same as input)")
    mult_dim: float = Field(default=4, description="Multiplier for inner dimension")
    inner_dim: Optional[int] = Field(default=None, description="Inner dimension (if None, input * mult_dim)")
    post_act_ln: bool = Field(default=False, description="Whether to use LayerNorm after activation")
    no_bias: bool = Field(default=False, description="Whether to remove bias")
    dropout: float = Field(default=0.0, description="Dropout probability")
    zero_init_output: bool = Field(default=False, description="Whether to initialize output to zero")
    activation: Callable = Field(default=nn.GELU, description="Activation function")
    
    # FFN specific fields
    post_type: str = Field(default="post_standard", description="Post-processor type")
    mult_bias: bool = Field(default=True, description="GLU multiplicative bias")
    
    def model_post_init(self, __context):
        """Post-init processing"""
        if self.inner_dim is None:
            self.inner_dim = int(self.dim_model * self.mult_dim)
        
        if self.ffn_dim_out is None:
            self.ffn_dim_out = self.dim_model


class BoringFeedForward(nn.Module):
    def __init__(self, config: FFNConfig = None, **kwargs):
        super().__init__()
        
        # Handle both config object and direct kwargs
        config_dict = config.model_dump() if config else {}
        config_dict.update(kwargs)
        config = FFNConfig(**config_dict)
        
        base_kwargs = {
            'no_bias': config.no_bias,
        }
        
        # Create main transform strategy
        transform_kwargs = {
            **base_kwargs,
            'dim_model': config.dim_model,
            'inner_dim': config.inner_dim,
            'activation': config.activation,
        }
        if config.type == "glu": transform_kwargs['mult_bias'] = config.mult_bias
        self.transform = ffn_registry.create_strategy(config.type, **transform_kwargs)
        
        # Create post-processor, notice that dim_model and inner_dim are different
        post_kwargs = {
            **base_kwargs,
            'dim_model': config.ffn_dim_out,
            'inner_dim': self.transform.output_dim,
            'dropout': config.dropout,
            'post_act_ln': config.post_act_ln,
            'zero_init_output': config.zero_init_output,
        }
        
        self.post_processor = ffn_registry.create_strategy(config.post_type, **post_kwargs)
    
    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Apply FFN transformation"""
        transformed = self.transform.apply(x, **kwargs)
        return self.post_processor.apply(transformed)


def create_ffn(ffn_type: str = "standard", **kwargs) -> BoringFeedForward:
    """Convenience function to create FFN"""
    # Extract type from kwargs if present to avoid duplicate parameter
    if 'type' in kwargs: ffn_type = kwargs.pop('type')

    if 'activation' in kwargs and isinstance(kwargs['activation'], str):
        try:
            kwargs['activation'] = get_activation_by_name(kwargs['activation'])
        except Exception as e:
            raise ValueError(f"Unknown activation: {kwargs['activation']}, error: {e}")
    
    config = FFNConfig(type=ffn_type, **kwargs)
    return BoringFeedForward(config) 


if __name__ == "__main__":
    # Example 1: Using config object
    config = FFNConfig(
        type="standard",
        dim_model=512,
        mult_dim=4,
        activation=nn.GELU
    )
    ffn1 = BoringFeedForward(config)
    
    # Example 2: Using convenience function with string activation
    ffn2 = create_ffn(
        ffn_type="glu",
        dim_model=512,
        mult_dim=2,
        activation="SiLU",  # Use PyTorch standard naming
        mult_bias=True
    )
    
    # Example 3: Direct kwargs
    ffn3 = BoringFeedForward(
        type="standard",
        dim_model=512,
        mult_dim=4
    )
    
    # Test
    x = torch.randn(2, 10, 512)
    y1 = ffn1(x)
    y2 = ffn2(x)
    y3 = ffn3(x)
    
    print(f"Standard FFN output: {y1.shape}")
    print(f"GLU FFN output: {y2.shape}")
    print(f"Direct kwargs FFN output: {y3.shape}") 