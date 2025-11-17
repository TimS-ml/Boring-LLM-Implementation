from typing import Optional
from pydantic import Field
import torch
import torch.nn as nn
from torch import Tensor

from boring_llm.base.component_registry import ComponentTransform, ComponentRegistry
from boring_llm.nn.norm.norm import l2norm


class PETransform(ComponentTransform):
    """Base class for positional encoding transformations"""
    
    def apply(self, pos: Tensor, **kwargs) -> Tensor:
        """Apply positional encoding to position indices"""
        raise NotImplementedError


pe_registry = ComponentRegistry[PETransform]("PositionalEncoding")


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


# TODO: apply RoPE to attention
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


# ============================================================================
# Advanced Positional Encodings (from x-transformers)
# ============================================================================

@pe_registry.register("relative_position_bias", {
    "num_heads": (int, Field(default=8, description="Number of attention heads")),
    "scale": (float, Field(default=1.0, description="Scale factor for bias")),
    "causal": (bool, Field(default=False, description="Use causal bias")),
    "num_buckets": (int, Field(default=32, description="Number of position buckets")),
    "max_distance": (int, Field(default=128, description="Maximum distance"))
})
class RelativePositionBiasEncoding(PETransform):
    """
    T5-style relative positional bias

    Adds learned biases to attention logits based on relative distances.
    Positions are bucketed (nearby use exact buckets, distant use log buckets).
    """

    def __init__(
        self,
        dim_model: int,  # Not used but required by interface
        num_heads: int = 8,
        scale: float = 1.0,
        causal: bool = False,
        num_buckets: int = 32,
        max_distance: int = 128,
        **kwargs
    ):
        super().__init__()
        import math
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.math = math  # Store math module
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position: Tensor,
        causal: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
        math_module = None
    ) -> Tensor:
        """Translate relative positions to bucket indices"""
        import math as math_lib
        math_module = math_module or math_lib

        ret = 0
        n = -relative_position

        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        # Half buckets for exact, half for logarithmic
        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) /
            math_module.log(max_distance / max_exact) *
            (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(
            val_if_large,
            torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def apply(self, pos: Tensor, seq_len_q: int = None, seq_len_k: int = None, **kwargs) -> Tensor:
        """
        Apply relative position bias

        Args:
            pos: Position indices (not used, lengths from seq_len_q/k)
            seq_len_q: Query sequence length
            seq_len_k: Key sequence length

        Returns:
            Bias tensor of shape [num_heads, seq_len_q, seq_len_k]
        """
        device = self.relative_attention_bias.weight.device

        if seq_len_q is None or seq_len_k is None:
            seq_len = pos.size(0)
            seq_len_q = seq_len_k = seq_len

        q_pos = torch.arange(seq_len_k - seq_len_q, seq_len_k, dtype=torch.long, device=device)
        k_pos = torch.arange(seq_len_k, dtype=torch.long, device=device)

        # Compute relative positions
        rel_pos = k_pos.unsqueeze(0) - q_pos.unsqueeze(1)  # [seq_q, seq_k]

        rp_bucket = self._relative_position_bucket(
            rel_pos,
            causal=self.causal,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
            math_module=self.math
        )

        values = self.relative_attention_bias(rp_bucket)  # [seq_q, seq_k, heads]
        bias = values.permute(2, 0, 1)  # [heads, seq_q, seq_k]

        return bias * self.scale


@pe_registry.register("dynamic_position_bias", {
    "num_heads": (int, Field(default=8, description="Number of attention heads")),
    "depth": (int, Field(default=2, description="MLP depth")),
    "log_distance": (bool, Field(default=False, description="Use log distance")),
    "use_norm": (bool, Field(default=False, description="Use normalization in MLP"))
})
class DynamicPositionBiasEncoding(PETransform):
    """
    Dynamic position bias using MLP

    Learns position bias through a small MLP network.
    """

    def __init__(
        self,
        dim_model: int,
        num_heads: int = 8,
        depth: int = 2,
        log_distance: bool = False,
        use_norm: bool = False,
        **kwargs
    ):
        super().__init__()
        assert depth >= 1, 'MLP depth must be >= 1'

        self.log_distance = log_distance
        self.mlp = nn.ModuleList([])

        # First layer
        layers = [nn.Linear(1, dim_model)]
        if use_norm:
            layers.append(nn.LayerNorm(dim_model))
        layers.append(nn.SiLU())
        self.mlp.append(nn.Sequential(*layers))

        # Hidden layers
        for _ in range(depth - 1):
            layers = [nn.Linear(dim_model, dim_model)]
            if use_norm:
                layers.append(nn.LayerNorm(dim_model))
            layers.append(nn.SiLU())
            self.mlp.append(nn.Sequential(*layers))

        # Output layer
        self.mlp.append(nn.Linear(dim_model, num_heads))

    def apply(self, pos: Tensor, seq_len_q: int = None, seq_len_k: int = None, **kwargs) -> Tensor:
        """
        Apply dynamic position bias

        Returns:
            Bias tensor of shape [num_heads, seq_len_q, seq_len_k]
        """
        device = self.mlp[0][0].weight.device

        if seq_len_q is None or seq_len_k is None:
            seq_len = pos.size(0)
            seq_len_q = seq_len_k = seq_len

        # Create position matrix
        seq_arange = torch.arange(seq_len_k - seq_len_q, seq_len_k, device=device)
        context_arange = torch.arange(seq_len_k, device=device)
        indices = seq_arange.unsqueeze(1) - context_arange.unsqueeze(0)  # [seq_q, seq_k]
        indices += (seq_len_k - 1)

        # Create continuous positions
        pos_input = torch.arange(-seq_len_k + 1, seq_len_k, device=device).float()
        pos_input = pos_input.unsqueeze(-1)  # [2*seq_k-1, 1]

        if self.log_distance:
            pos_input = torch.sign(pos_input) * torch.log(pos_input.abs() + 1)

        # Pass through MLP
        for layer in self.mlp:
            pos_input = layer(pos_input)

        # Get position biases
        bias = pos_input[indices]  # [seq_q, seq_k, heads]
        bias = bias.permute(2, 0, 1)  # [heads, seq_q, seq_k]

        return bias


@pe_registry.register("cope", {
    "num_heads": (int, Field(default=8, description="Number of attention heads")),
    "max_pos": (int, Field(default=16, description="Maximum position value")),
    "soft_onehot": (bool, Field(default=False, description="Use soft one-hot encoding")),
    "talking_heads": (bool, Field(default=False, description="Use talking heads")),
    "soft_onehot_temp": (float, Field(default=5e-2, description="Temperature for soft one-hot"))
})
class CoPEEncoding(PETransform):
    """
    Contextual Position Encoding (CoPE)

    From: https://arxiv.org/abs/2405.18719

    Computes positions based on attention patterns, allowing the model to
    count contextually relevant tokens rather than absolute positions.

    Note: This is typically used within attention mechanism, not as standalone PE.
    """

    def __init__(
        self,
        dim_model: int,
        num_heads: int = 8,
        max_pos: int = 16,
        soft_onehot: bool = False,
        talking_heads: bool = False,
        soft_onehot_temp: float = 5e-2,
        **kwargs
    ):
        super().__init__()
        # dim_model here is actually dim_head when used in attention
        dim_head = dim_model // num_heads if dim_model > num_heads else dim_model

        self.max_pos = max_pos
        self.num_heads = num_heads
        self.pos_emb = nn.Parameter(torch.zeros(max_pos, dim_head))

        self.talking_heads = nn.Conv2d(num_heads, num_heads, 1, bias=False) if talking_heads else None
        self.soft_onehot = soft_onehot
        self.soft_onehot_temp = soft_onehot_temp

        if soft_onehot:
            self.register_buffer('positions', torch.arange(max_pos))

    def apply(self, pos: Tensor, query: Tensor = None, attn_logits: Tensor = None, **kwargs) -> Tensor:
        """
        Apply CoPE

        Args:
            pos: Position indices (not used)
            query: Query tensor [batch, heads, seq, dim_head]
            attn_logits: Attention logits before softmax [batch, heads, seq_q, seq_k]

        Returns:
            Position embeddings to add to attention logits
        """
        if query is None or attn_logits is None:
            raise ValueError("CoPE requires 'query' and 'attn_logits' arguments")

        if self.talking_heads is not None:
            i, j = attn_logits.shape[-2:]
            causal_mask = attn_logits.new_ones(i, j).triu(j - i + 1).bool()

            attn_logits = self.talking_heads(attn_logits)
            attn_logits = attn_logits.masked_fill(causal_mask, -torch.finfo(attn_logits.dtype).max)

        # Compute contextual positions using cumulative gating
        gates = attn_logits.sigmoid()

        # Count backward from each position
        pos_computed = gates.flip(-1).cumsum(dim=-1).flip(-1)
        pos_computed = pos_computed.clamp(max=self.max_pos - 1)

        # Project queries to position logits
        logits_int = torch.einsum('b h n d, p d -> b h n p', query, self.pos_emb)

        if self.soft_onehot:
            # Soft interpolation
            import torch.nn.functional as F
            diff_pos = (pos_computed.unsqueeze(-1) - self.positions.unsqueeze(0).unsqueeze(0).unsqueeze(0)).abs()
            soft_onehot_pos = F.softmax(-diff_pos / self.soft_onehot_temp, dim=-1)
            cope_pos_emb = torch.einsum('b h i j p, b h i p -> b h i j', soft_onehot_pos, logits_int)
        else:
            # Linear interpolation between integer positions
            pos_ceil = pos_computed.ceil().long()
            pos_floor = pos_computed.floor().long()
            logits_ceil = logits_int.gather(-1, pos_ceil)
            logits_floor = logits_int.gather(-1, pos_floor)

            w = pos_computed - pos_floor.float()
            cope_pos_emb = logits_ceil * w + logits_floor * (1 - w)

        return cope_pos_emb


@pe_registry.register("data_dependent_alibi", {
    "num_heads": (int, Field(default=8, description="Number of attention heads")),
    "causal": (bool, Field(default=True, description="Use causal masking")),
    "bias_init": (float, Field(default=5., description="Initial bias value")),
    "post_log_scale": (float, Field(default=1., description="Scale after log"))
})
class DataDependentAlibiEncoding(PETransform):
    """
    Data-Dependent ALiBi from https://openreview.net/forum?id=q2Lnyegkr8

    Learns position-dependent forget gates based on the input.
    """

    def __init__(
        self,
        dim_model: int,
        num_heads: int = 8,
        causal: bool = True,
        bias_init: float = 5.,
        post_log_scale: float = 1.,
        **kwargs
    ):
        super().__init__()
        from einops.layers.torch import Rearrange

        self.causal = causal
        self.post_log_scale = post_log_scale

        linear = nn.Linear(dim_model, num_heads * (1 if causal else 2))

        self.to_forget_gates = nn.Sequential(
            linear,
            Rearrange('b n h -> b h n'),
            nn.LogSigmoid()
        )

        nn.init.constant_(linear.bias, bias_init)

    def apply(self, pos: Tensor, x: Tensor = None, **kwargs) -> Tensor:
        """
        Apply data-dependent ALiBi

        Args:
            pos: Position indices (not used)
            x: Input tensor [batch, seq, dim]

        Returns:
            Position bias [batch, heads, seq, seq]
        """
        if x is None:
            raise ValueError("Data-dependent ALiBi requires 'x' argument")

        bidirectional = not self.causal

        forget_gates = self.to_forget_gates(x)  # [batch, heads, seq]

        if bidirectional:
            forget_gates_fwd, forget_gates_bwd = forget_gates.chunk(2, dim=1)
        else:
            forget_gates_fwd = forget_gates

        # Compute cumulative products (positions)
        log_cumsum_fwd = torch.logcumsumexp(forget_gates_fwd, dim=-1)

        # Create position bias matrix
        bias_fwd = log_cumsum_fwd.unsqueeze(-1) - log_cumsum_fwd.unsqueeze(-2)
        bias_fwd = bias_fwd * self.post_log_scale

        if bidirectional:
            log_cumsum_bwd = torch.logcumsumexp(forget_gates_bwd.flip(-1), dim=-1).flip(-1)
            bias_bwd = log_cumsum_bwd.unsqueeze(-1) - log_cumsum_bwd.unsqueeze(-2)
            bias_bwd = bias_bwd * self.post_log_scale

            # Combine forward and backward
            bias = torch.where(
                torch.arange(x.shape[1], device=x.device).unsqueeze(0) <
                torch.arange(x.shape[1], device=x.device).unsqueeze(1),
                bias_fwd,
                bias_bwd
            )
        else:
            bias = bias_fwd
            # Apply causal masking
            causal_mask = torch.ones_like(bias, dtype=torch.bool).triu(1)
            bias = bias.masked_fill(causal_mask, 0)

        return bias
