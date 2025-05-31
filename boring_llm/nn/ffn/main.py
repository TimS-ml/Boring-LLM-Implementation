"""
Simplified FFN implementation
Reduces 6 files (base.py, config.py, factory.py, main.py, strategies/) to 1 file
"""
from typing import Optional, Callable
from pydantic import Field
import torch
import torch.nn as nn
from torch import Tensor

from boring_llm.base.component_registry import ComponentConfig
from .registry import ffn_registry


# ============= Configuration =============
class FFNConfig(ComponentConfig):
    """FFN Configuration - inherits dim_model etc. from BaseConfig"""
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
        # Calculate inner_dim if not provided
        if self.inner_dim is None:
            self.inner_dim = int(self.dim_model * self.mult_dim)
        
        # Set output dimension
        if self.ffn_dim_out is None:
            self.ffn_dim_out = self.dim_model


# ============= Main FFN Module =============
class BoringFeedForward(nn.Module):
    """Simplified FFN that reduces complexity while keeping flexibility"""
    
    def __init__(self, config: FFNConfig = None, **kwargs):
        super().__init__()
        
        # Handle both config object and direct kwargs
        if config is None:
            config = FFNConfig(**kwargs)
        else:
            # Override config with any provided kwargs
            config_dict = config.model_dump()
            config_dict.update(kwargs)
            config = FFNConfig(**config_dict)
        
        self.config = config
        
        # Create transform strategy
        transform_kwargs = {
            'dim_model': config.dim_model,
            'inner_dim': config.inner_dim,
            'activation': config.activation,
            'no_bias': config.no_bias,
        }
        
        # Add type-specific kwargs
        if config.type == "glu":
            transform_kwargs['mult_bias'] = config.mult_bias
            
        self.transform = ffn_registry.create_strategy(config.type, **transform_kwargs)
        
        # Create post-processor
        post_kwargs = {
            'dim_model': config.ffn_dim_out,
            'inner_dim': self.transform.output_dim,
            'dropout': config.dropout,
            'post_act_ln': config.post_act_ln,
            'no_bias': config.no_bias,
            'zero_init_output': config.zero_init_output,
        }
        
        self.post_processor = ffn_registry.create_strategy(config.post_type, **post_kwargs)
    
    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Apply FFN transformation"""
        transformed = self.transform.apply(x, **kwargs)
        return self.post_processor.apply(transformed)


# ============= Convenience Functions =============
def create_ffn(ffn_type: str = "standard", **kwargs) -> BoringFeedForward:
    """Convenience function to create FFN"""
    # Extract type from kwargs if present to avoid duplicate parameter
    if 'type' in kwargs:
        ffn_type = kwargs.pop('type')
    
    # Handle string activation names
    if 'activation' in kwargs and isinstance(kwargs['activation'], str):
        activation_map = {
            'gelu': nn.GELU,
            'relu': nn.ReLU,
            'silu': nn.SiLU,
            'swish': nn.SiLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid
        }
        activation_name = kwargs['activation'].lower()
        if activation_name in activation_map:
            kwargs['activation'] = activation_map[activation_name]
        else:
            raise ValueError(f"Unknown activation: {kwargs['activation']}")
    
    config = FFNConfig(type=ffn_type, **kwargs)
    return BoringFeedForward(config)


# ============= Usage Examples =============
if __name__ == "__main__":
    # Example 1: Using config object
    config = FFNConfig(
        type="standard",
        dim_model=512,
        mult_dim=4,
        activation=nn.GELU
    )
    ffn1 = BoringFeedForward(config)
    
    # Example 2: Using convenience function
    ffn2 = create_ffn(
        ffn_type="glu",
        dim_model=512,
        mult_dim=2,
        activation=nn.SiLU,
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