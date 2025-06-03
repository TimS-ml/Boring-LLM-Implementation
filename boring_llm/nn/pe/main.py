from typing import Optional
from pydantic import Field
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

from boring_llm.base.component_registry import ComponentConfig
from boring_llm.nn.pe.registry import pe_registry


class PEConfig(ComponentConfig):
    """Positional Encoding Configuration - inherits dim_model etc. from BaseConfig"""
    # Type-specific fields (will be validated based on type)
    l2norm_embed: bool = Field(default=False, description="Whether to L2 normalize embeddings")
    rotary_percentage: float = Field(default=1.0, description="Percentage of dimensions for rotary encoding")
    rope_base: int = Field(default=10000, description="Base for rotary encoding")
    alibi_num_heads: Optional[int] = Field(default=None, description="Number of heads for ALiBi")


class BoringPositionalEncoding(nn.Module):
    """Simplified Positional Encoding that reduces complexity while keeping flexibility"""
    
    def __init__(self, config: PEConfig = None, **kwargs):
        super().__init__()
        config = PEConfig(**kwargs) if not config else config.model_copy(update=kwargs)
        self.max_seq_len = config.max_seq_len
        
        # Create strategy
        strategy_kwargs = {
            'dim_model': config.dim_model,
            'max_seq_len': config.max_seq_len,
        }
        
        # Add type-specific kwargs
        if config.type == "absolute":
            strategy_kwargs['l2norm_embed'] = config.l2norm_embed
        elif config.type == "rotary":
            strategy_kwargs['rotary_percentage'] = config.rotary_percentage
            strategy_kwargs['rope_base'] = config.rope_base
        elif config.type == "alibi":
            if config.alibi_num_heads is None:
                raise ValueError("alibi_num_heads must be specified for ALiBi encoding")
            strategy_kwargs['alibi_num_heads'] = config.alibi_num_heads
        
        self.pe_strategy = pe_registry.create_strategy(config.type, **strategy_kwargs)
    
    def forward(self, x: Tensor, pos: Optional[Tensor] = None, **kwargs) -> Tensor:
        """
        Apply positional encoding to input tensor
        
        Args:
            x: Input tensor of shape [batch, seq_len, dim]
            pos: Optional position indices. If None, uses sequential positions
            **kwargs: Additional arguments passed to the strategy
            
        Returns:
            Tensor with positional information
        """
        seq_len, device = x.shape[1], x.device
        
        # Validation
        assert seq_len <= self.max_seq_len, f'Sequence length {seq_len} exceeds maximum {self.max_seq_len}'
        
        if pos is None:
            pos = torch.arange(seq_len, device=device)
            
        # Delegate to strategy implementation
        return self.pe_strategy.apply(pos=pos, **kwargs)


def create_pe(pe_type: str = "fixed", **kwargs) -> BoringPositionalEncoding:
    """Convenience function to create Positional Encoding"""
    # Extract type from kwargs if present to avoid duplicate parameter
    if 'type' in kwargs:
        pe_type = kwargs.pop('type')
    config = PEConfig(type=pe_type, **kwargs)
    return BoringPositionalEncoding(config)


if __name__ == "__main__":
    # Example 1: Fixed sinusoidal encoding
    pe1 = create_pe(
        pe_type="fixed",
        dim_model=512,
        max_seq_len=1024
    )
    
    # Example 2: Learnable absolute encoding
    pe2 = create_pe(
        pe_type="absolute",
        dim_model=512,
        max_seq_len=1024,
        l2norm_embed=True
    )
    
    # Example 3: Rotary encoding
    pe3 = create_pe(
        pe_type="rotary",
        dim_model=512,
        rotary_percentage=0.5,
        rope_base=10000
    )
    
    # Example 4: ALiBi encoding
    pe4 = create_pe(
        pe_type="alibi",
        dim_model=512,
        alibi_num_heads=8
    )
    
    # Test
    x = torch.randn(2, 128, 512)
    
    pos_emb1 = pe1(x)
    pos_emb2 = pe2(x)
    pos_emb3 = pe3(x)
    pos_emb4 = pe4(x)
    
    print(f"Fixed PE output: {pos_emb1.shape if pos_emb1 is not None else None}")
    print(f"Absolute PE output: {pos_emb2.shape if pos_emb2 is not None else None}")
    print(f"Rotary PE output: {pos_emb3.shape if pos_emb3 is not None else None}")
    print(f"ALiBi PE output: {pos_emb4.shape if pos_emb4 is not None else None}") 