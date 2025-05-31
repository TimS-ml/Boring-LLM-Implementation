"""
Positional Encoding Transform Registry
Contains all PE transformation strategies and their registry
"""
from typing import Optional
from pydantic import Field
import torch
import torch.nn as nn
from torch import Tensor

from boring_llm.base.component_registry import ComponentTransform, ComponentRegistry
from boring_llm.nn.norm.norm import l2norm


# ============= PE Transform Base =============
class PETransform(ComponentTransform):
    """Base class for positional encoding transformations"""
    
    def apply(self, pos: Tensor, **kwargs) -> Tensor:
        """Apply positional encoding to position indices"""
        raise NotImplementedError


# ============= Registry Setup =============
pe_registry = ComponentRegistry[PETransform]("PositionalEncoding")


# ============= PE Strategies =============
@pe_registry.register("fixed")
class FixedPositionalEncoding(PETransform):
    """
    Sinusoidal positional embeddings from "Attention Is All You Need"
    - Even indices (2i):   PE_(pos, 2i)   = sin(pos/10000^(2i/d))
    - Odd indices (2i+1):  PE_(pos, 2i+1) = cos(pos/10000^(2i/d))
    """
    
    def __init__(self, dim_model: int, **kwargs):
        super().__init__()
        # [0, 2, 4, ..., dim] / dim = [0, 2/dim, 4/dim, ..., 1]
        inv_freq = 1. / (10000 ** (torch.arange(0, dim_model, 2).float() / dim_model))  # [dim/2]
        self.register_buffer('inv_freq', inv_freq)

    def apply(self, pos: Tensor, offset: int = 0, **kwargs) -> Tensor:
        """Apply fixed sinusoidal positional encoding"""
        pos = pos.type_as(self.inv_freq) + offset  # [seq_len]
        sinusoid_inp = pos.unsqueeze(-1) * self.inv_freq  # [seq_len, 1] * [dim/2] -> [seq_len, dim/2]
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)  # [seq_len, dim]
        return emb


@pe_registry.register("absolute", {
    "l2norm_embed": (bool, Field(default=False, description="Whether to L2 normalize embeddings"))
})
class AbsolutePositionalEncoding(PETransform):
    """Learnable absolute positional embeddings"""
    
    def __init__(self, dim_model: int, max_seq_len: int = 1024, l2norm_embed: bool = False, **kwargs):
        super().__init__()
        self.l2norm_embed = l2norm_embed
        self.scale = dim_model ** -0.5 if not l2norm_embed else 1.
        self.emb = nn.Embedding(max_seq_len, dim_model)

    def apply(self, pos: Tensor, **kwargs) -> Tensor:
        """Apply absolute positional encoding"""
        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return l2norm(pos_emb) if self.l2norm_embed else pos_emb


@pe_registry.register("none")
class NonePositionalEncoding(PETransform):
    """No positional encoding - returns None"""
    
    def __init__(self, **kwargs):
        super().__init__()
        
    def apply(self, **kwargs) -> Tensor:
        """Return None for no positional encoding"""
        return None


@pe_registry.register("rotary", {
    "rotary_percentage": (float, Field(default=1.0, description="Percentage of dimensions to apply rotary encoding to")),
    "rope_base": (int, Field(default=10000, description="Base for rotary encoding"))
})
class RotaryPositionalEncoding(PETransform):
    """Rotary Positional Encoding (RoPE)"""
    
    def __init__(self, dim_model: int, rotary_percentage: float = 1.0, rope_base: int = 10000, 
                 max_seq_len: int = 1024, **kwargs):
        super().__init__()
        self.rotary_percentage = rotary_percentage
        
        # Calculate dimensions for rotation
        dim_rotary = int(dim_model * rotary_percentage)
        if dim_rotary % 2 != 0:
            dim_rotary -= 1  # Ensure even dimension for rotation
        
        # Create frequency basis
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, dim_rotary, 2).float() / dim_rotary))
        self.register_buffer('inv_freq', inv_freq)
        
        # Pre-compute rotary embeddings for efficiency
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def apply(self, pos: Tensor, **kwargs) -> Tensor:
        """Apply rotary positional encoding"""
        # Return the cos/sin values for the given positions
        cos = self.cos_cached[pos]
        sin = self.sin_cached[pos]
        return torch.stack([cos, sin], dim=-1)


@pe_registry.register("alibi", {
    "alibi_num_heads": (int, Field(default=None, description="Number of attention heads for ALiBi"))
})
class AlibiPositionalEncoding(PETransform):
    """ALiBi (Attention with Linear Biases) Positional Encoding"""
    
    def __init__(self, alibi_num_heads: int, **kwargs):
        super().__init__()
        self.num_heads = alibi_num_heads
        
        # Generate ALiBi slopes
        slopes = self._get_slopes(alibi_num_heads)
        self.register_buffer('slopes', slopes)

    def _get_slopes(self, num_heads: int) -> Tensor:
        """Generate ALiBi slopes"""
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(torch.arange(1, n + 1).log2().floor() + 1)))
            return start

        if num_heads & (num_heads - 1) == 0:  # Power of 2
            return get_slopes_power_of_2(num_heads)
        else:
            # Not power of 2, use closest power of 2 and interpolate
            closest_power_of_2 = 2 ** torch.arange(1, num_heads + 1).log2().floor().max()
            slopes = get_slopes_power_of_2(closest_power_of_2)[:num_heads]
            return slopes

    def apply(self, pos: Tensor, **kwargs) -> Tensor:
        """Apply ALiBi positional encoding"""
        seq_len = pos.size(0)
        # Create relative position matrix
        pos_matrix = pos.unsqueeze(0) - pos.unsqueeze(1)  # [seq_len, seq_len]
        
        # Apply slopes to get bias
        bias = pos_matrix.unsqueeze(0) * self.slopes.unsqueeze(-1).unsqueeze(-1)  # [num_heads, seq_len, seq_len]
        return bias
