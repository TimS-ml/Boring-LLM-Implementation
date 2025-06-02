from typing import Optional, Callable
from pydantic import Field
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from boring_llm.base.component_registry import ComponentConfig
from boring_llm.nn.ffn.registry import ffn_registry, RouterTransform
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


class MOEConfig(ComponentConfig):
    """MOE-specific configuration"""
    ffn_dim_out: Optional[int] = Field(default=None, description="Output dimension (if None, same as input)")
    mult_dim: float = Field(default=4, description="Multiplier for inner dimension")
    inner_dim: Optional[int] = Field(default=None, description="Inner dimension (if None, input * mult_dim)")
    activation: Callable = Field(default=nn.GELU, description="Activation function")
    no_bias: bool = Field(default=False, description="Whether to remove bias")
    dropout: float = Field(default=0.0, description="Dropout probability")
    
    # MOE specific fields
    num_experts: int = Field(default=8, description="Number of experts in MOE")
    expert_inner_dim: Optional[int] = Field(default=None, description="Inner dimension for each expert")
    top_k: int = Field(default=2, description="Number of experts to route to")
    capacity_factor: float = Field(default=1.0, description="Load balancing, 1.25 means each expert can handle up to 25% tokens")
    noise_std: float = Field(default=1.0, description="Router noise standard deviation")
    expert_type: str = Field(default="standard", description="Expert FFN type (standard, glu, etc.)")
    
    def model_post_init(self, __context):
        """Post-init processing"""
        if self.inner_dim is None:
            self.inner_dim = int(self.dim_model * self.mult_dim)
        
        if self.ffn_dim_out is None:
            self.ffn_dim_out = self.dim_model
            
        if self.expert_inner_dim is None:
            self.expert_inner_dim = self.inner_dim


class BoringFeedForward(nn.Module):
    """Standard Feed-Forward Network implementation"""
    
    def __init__(self, config: FFNConfig = None, **kwargs):
        super().__init__()
        
        # Handle both config object and direct kwargs
        config_dict = config.model_dump() if config else {}
        config_dict.update(kwargs)
        config = FFNConfig(**config_dict)
        
        # Create main transform strategy
        transform_kwargs = {
            'dim_model': config.dim_model,
            'inner_dim': config.inner_dim,
            'activation': config.activation,
            'no_bias': config.no_bias,
        }
        if config.type == "glu": 
            transform_kwargs['mult_bias'] = config.mult_bias
        self.transform = ffn_registry.create_strategy(config.type, **transform_kwargs)
        
        # Create post-processor
        post_kwargs = {
            'dim_model': config.ffn_dim_out,
            'inner_dim': self.transform.output_dim,
            'dropout': config.dropout,
            'post_act_ln': config.post_act_ln,
            'zero_init_output': config.zero_init_output,
            'no_bias': config.no_bias,
        }
        
        self.post_processor = ffn_registry.create_strategy(config.post_type, **post_kwargs)
    
    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Apply FFN transformation"""
        transformed = self.transform.apply(x, **kwargs)
        return self.post_processor.apply(transformed)


class BoringFeedForwardMOE(nn.Module):
    """Mixture of Experts Feed-Forward Network implementation"""
    
    def __init__(self, config: MOEConfig = None, **kwargs):
        super().__init__()
        
        # Handle both config object and direct kwargs
        config_dict = config.model_dump() if config else {}
        config_dict.update(kwargs)
        config = MOEConfig(**config_dict)
        
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.capacity_factor = config.capacity_factor
        self.dim_model = config.dim_model
        
        # Create router
        self.router = RouterTransform(
            dim_model=config.dim_model, 
            num_experts=config.num_experts, 
            top_k=config.top_k,
            noise_std=config.noise_std,
            no_bias=config.no_bias
        )
        
        # Create experts using BoringFeedForward
        self.experts = nn.ModuleList([
            BoringFeedForward(
                FFNConfig(
                    type=config.expert_type,
                    dim_model=config.dim_model,
                    inner_dim=config.expert_inner_dim,
                    ffn_dim_out=config.dim_model,  # Expert output should be dim_model
                    activation=config.activation,
                    no_bias=config.no_bias,
                    dropout=config.dropout,
                    mult_bias=getattr(config, 'mult_bias', True) if config.expert_type == "glu" else True
                )
            ) for i in range(config.num_experts)
        ])
        
    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Sparse MOE forward pass"""
        B, T, C = x.shape
        flat_x = x.view(-1, C)  # (B*T, C)
        
        # Get routing decisions
        routing_weights, indices = self.router.apply(flat_x)
        
        # Initialize output
        final_output = torch.zeros_like(flat_x)
        
        # Apply expert capacity if needed
        if self.capacity_factor < float('inf'):
            tokens_per_batch = B * T * self.top_k
            expert_capacity = int((tokens_per_batch / self.num_experts) * self.capacity_factor)
        else:
            expert_capacity = None
        
        # Process each expert
        for expert_idx, expert in enumerate(self.experts):
            # Find tokens routed to this expert
            expert_mask = (indices == expert_idx).any(dim=-1)
            token_indices = torch.nonzero(expert_mask).squeeze(-1)
            
            # Apply capacity limit
            if expert_capacity is not None and len(token_indices) > expert_capacity:
                token_indices = token_indices[:expert_capacity]
                
            if len(token_indices) > 0:
                # Get expert input
                expert_input = flat_x[token_indices]
                expert_output = expert(expert_input)
                
                # Get routing weights for this expert
                expert_weights = routing_weights[token_indices, expert_idx].unsqueeze(-1)
                weighted_output = expert_output * expert_weights
                
                # Add to final output
                final_output.index_add_(0, token_indices, weighted_output)
        
        return final_output.view(B, T, C)


def create_ffn(ffn_type: str = "standard", **kwargs) -> BoringFeedForward:
    """Convenience function to create standard FFN"""
    # Extract type from kwargs if present to avoid duplicate parameter
    if 'type' in kwargs: ffn_type = kwargs.pop('type')

    if 'activation' in kwargs and isinstance(kwargs['activation'], str):
        try:
            kwargs['activation'] = get_activation_by_name(kwargs['activation'])
        except Exception as e:
            raise ValueError(f"Unknown activation: {kwargs['activation']}, error: {e}")
    
    config = FFNConfig(type=ffn_type, **kwargs)
    return BoringFeedForward(config) 


def create_moe_ffn(num_experts: int = 8, top_k: int = 2, expert_type: str = "standard", **kwargs) -> BoringFeedForwardMOE:
    """Convenience function to create Sparse MOE FFN"""
    if 'activation' in kwargs and isinstance(kwargs['activation'], str):
        try:
            kwargs['activation'] = get_activation_by_name(kwargs['activation'])
        except Exception as e:
            raise ValueError(f"Unknown activation: {kwargs['activation']}, error: {e}")
    
    kwargs.update({
        'num_experts': num_experts,
        'top_k': top_k,
        'expert_type': expert_type,
    })
    config = MOEConfig(**kwargs)
    return BoringFeedForwardMOE(config)


if __name__ == "__main__":
    # Example 1: Using config object for standard FFN
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
    
    # Example 3: Direct kwargs for standard FFN
    ffn3 = BoringFeedForward(
        type="standard",
        dim_model=512,
        mult_dim=4
    )
    
    # Example 4: MOE FFN with standard experts
    moe_ffn = create_moe_ffn(
        num_experts=8,
        top_k=2,
        expert_type="standard",
        dim_model=512,
        mult_dim=4,
        activation="SiLU",
        capacity_factor=1.25,
        noise_std=0.1
    )
    
    # Example 5: MOE FFN with GLU experts
    moe_glu_ffn = create_moe_ffn(
        num_experts=8,
        top_k=2,
        expert_type="glu",
        dim_model=512,
        mult_dim=4,
        activation="SiLU",
        capacity_factor=1.25,
        noise_std=0.1
    )
    
    # Example 6: MOE FFN with config
    moe_config = MOEConfig(
        dim_model=512,
        mult_dim=4,
        num_experts=8,
        top_k=2,
        expert_type="standard",
        activation=nn.SiLU,
        capacity_factor=1.25,
        noise_std=0.1,
        dropout=0.1
    )
    moe_ffn_with_config = BoringFeedForwardMOE(moe_config)
    
    # Test
    x = torch.randn(2, 10, 512)
    print("Testing FFN implementations...")
    
    y1 = ffn1(x)
    print(f"Standard FFN output: {y1.shape}")
    
    y2 = ffn2(x)
    print(f"GLU FFN output: {y2.shape}")
    
    y3 = ffn3(x)
    print(f"Direct kwargs FFN output: {y3.shape}") 
    
    y_moe = moe_ffn(x)
    print(f"MOE FFN with standard experts output: {y_moe.shape}")
    
    y_moe_glu = moe_glu_ffn(x)
    print(f"MOE FFN with GLU experts output: {y_moe_glu.shape}")
    
    y_moe_config = moe_ffn_with_config(x)
    print(f"MOE FFN with config output: {y_moe_config.shape}") 