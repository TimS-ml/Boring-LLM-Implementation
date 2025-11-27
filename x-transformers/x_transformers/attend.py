"""
Attention mechanisms for transformer models.

This module provides various attention implementations including:
- Standard scaled dot-product attention
- Flash attention (efficient GPU implementation)
- Causal (autoregressive) attention with masking
- Talking heads (cross-head communication)
- Selective attention (tokens can prevent being attended to)
- Sparse topk attention (only attend to top-k most relevant tokens)
- L2 distance-based attention
- Sigmoid and hard attention variants
- Gumbel softmax attention
- CoG (negative weights) attention
- Contextual positional encoding (CoPE)
"""

from __future__ import annotations

from functools import partial
from typing import Tuple, Callable

import torch
from torch.nn import Module, Parameter
from torch import cat, nn, einsum, Tensor
import torch.nn.functional as F

from collections import namedtuple
from functools import wraps
from packaging import version
from dataclasses import dataclass

from einops import rearrange, repeat, pack, unpack

# constants

@dataclass
class Intermediates:
    """
    Container for intermediate attention computation values.

    These intermediate values are useful for:
    - Debugging attention patterns
    - Visualizing attention weights
    - Implementing auxiliary losses based on attention
    - Analysis and interpretability

    Attributes:
        qk_similarities: Raw query-key similarity scores before any masking or softmax.
                        Shape: (batch, heads, seq_len_q, seq_len_k)
        pre_softmax_attn: Attention logits after scaling and masking but before softmax.
                         Shape: (batch, heads, seq_len_q, seq_len_k)
        post_softmax_attn: Attention weights after softmax normalization.
                          Shape: (batch, heads, seq_len_q, seq_len_k)
        values: Value vectors used in attention computation.
               Shape: (batch, heads, seq_len, dim)
        cached_kv: Cached key and value tensors for efficient autoregressive generation.
                  Tuple of (keys, values) tensors.
        layer_type: String identifier for the type of layer (e.g., 'self', 'cross').
        hybrid_hidden: Hidden states for hybrid attention architectures.
    """
    qk_similarities:    Tensor | None = None
    pre_softmax_attn:   Tensor | None = None
    post_softmax_attn:  Tensor | None = None
    values:             Tensor | None = None
    cached_kv:          tuple[Tensor, Tensor] | None = None
    layer_type:         str | None = None
    hybrid_hidden:      Tensor | None = None

    def to_tuple(self):
        """
        Convert intermediates to a tuple of main attention values.

        Returns:
            Tuple of (qk_similarities, pre_softmax_attn, post_softmax_attn)
        """
        return (self.qk_similarities, self.pre_softmax_attn, self.post_softmax_attn)

# helpers

def exists(val):
    """
    Check if a value is not None.

    Args:
        val: Any value to check.

    Returns:
        bool: True if val is not None, False otherwise.
    """
    return val is not None

def default(val, d):
    """
    Return val if it exists (is not None), otherwise return default value d.

    Args:
        val: The value to check.
        d: The default value to return if val is None.

    Returns:
        val if val is not None, otherwise d.
    """
    return val if exists(val) else d

def at_most_one_of(*bools):
    """
    Check that at most one of the provided boolean values is True.

    This is used to ensure mutually exclusive options in attention configuration.

    Args:
        *bools: Variable number of boolean values to check.

    Returns:
        bool: True if at most one of the bools is True, False if two or more are True.
    """
    return sum([*map(int, bools)]) <= 1

def compact(arr):
    """
    Filter out None values from an array.

    Args:
        arr: Iterable containing values, some of which may be None.

    Returns:
        List with all None values removed.
    """
    return [*filter(exists, arr)]

@torch.jit.script
def softclamp(t: Tensor, value: float):
    """
    Soft clamp tensor values to the range [-value, value] using tanh.

    This is a differentiable alternative to hard clamping that smoothly
    saturates values approaching the boundaries.

    Args:
        t: Input tensor to clamp.
        value: The clamping range (outputs will be in [-value, value]).

    Returns:
        Tensor with values soft-clamped to [-value, value].
    """
    return (t / value).tanh() * value

def pack_one(t, pattern):
    """
    Pack a single tensor using einops pack with the given pattern.

    Args:
        t: Tensor to pack.
        pattern: Einops pattern string for packing.

    Returns:
        Packed tensor and shape information.
    """
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    """
    Unpack a single tensor using einops unpack with the given pattern.

    Args:
        t: Tensor to unpack.
        ps: Packed shape information from pack operation.
        pattern: Einops pattern string for unpacking.

    Returns:
        Unpacked tensor.
    """
    return unpack(t, ps, pattern)[0]

def once(fn):
    """
    Decorator that ensures a function is only called once.

    Subsequent calls to the decorated function will return None without
    executing the function body.

    Args:
        fn: Function to wrap.

    Returns:
        Wrapped function that only executes once.
    """
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

# Print function that only prints once (useful for warnings)
print_once = once(print)

# gumbel softmax attention related

def log_prob_from_hard_attend(intermeds: Intermediates):
    """
    Extract log probabilities from hard attention (e.g., Gumbel softmax).

    For hard attention mechanisms, this function computes the log probability
    of the selected tokens based on the attention logits before hard selection.

    Args:
        intermeds: Intermediates object containing pre_softmax_attn (attention logits)
                  and post_softmax_attn (hard one-hot or Gumbel softmax output).

    Returns:
        Tensor of shape (batch, heads, seq_len) containing log probabilities
        of the selected attention targets.
    """
    # Convert attention logits to log probabilities
    log_probs = intermeds.pre_softmax_attn.log_softmax(dim = -1)

    # Get the indices of the hard-selected tokens
    one_hot = intermeds.post_softmax_attn.argmax(dim = -1, keepdim = True)
    # Gather the log probabilities at the selected indices
    log_prob = log_probs.gather(-1, one_hot)
    return rearrange(log_prob, 'b h i 1 -> b h i')

# selective attention
# https://arxiv.org/abs/2410.02703 - section 3.3
# it is a technique to allow each token to prevent itself from being attended to by future tokens
# if sim_head_gate not supplied, will use the first head of the attention logits (sim in this framework)

def selective_attn(
    sim,
    sim_head_gate = None,
    no_mask_sos = True
):
    """
    Apply selective attention masking to attention logits.

    Selective attention allows each token to control whether future tokens can attend to it.
    This is based on https://arxiv.org/abs/2410.02703 (section 3.3).

    The mechanism works by:
    1. Computing a "gate" value for each token (higher = more masking of future attention)
    2. Using these gates to dynamically mask future tokens from attending to current tokens
    3. This gives tokens selective control over their visibility to future tokens

    Args:
        sim: Attention similarity/logit tensor of shape (batch, heads, seq_len_q, seq_len_k).
        sim_head_gate: Optional tensor to use for computing gates. If None, uses the first
                      attention head from sim. Shape: (batch, seq_len_q, seq_len_k).
        no_mask_sos: If True, prevents masking of the first token (start-of-sequence).
                    Default: True.

    Returns:
        Modified attention logits with selective masking applied.
        Shape: (batch, heads, seq_len_q, seq_len_k).
    """
    i, j, device = *sim.shape[-2:], sim.device
    # Use first attention head as gate if not provided
    sim_head_gate = default(sim_head_gate, sim[:, 0])

    # Only positive gate values (relu ensures non-negative)
    gate = F.relu(sim_head_gate) # only positive

    # Optionally preserve start-of-sequence token from being masked
    if no_mask_sos:
        gate = gate.clone()
        gate[..., -i] = 0.

    # Create identity matrix for self-attention masking
    eye = torch.eye(i, device = device)

    # Handle case where key sequence is longer than query sequence (e.g., with KV cache)
    if j > i:
        eye = F.pad(eye, (j - i, 0), value = 1.)

    # Zero out self-attention positions (token cannot mask itself)
    gate = (1. - eye) * gate
    # Shift gates so each position only affects future positions
    gate = F.pad(gate, (0, 0, 1, -1), value = 0.) # only allow for masking the future
    # Cumulative sum creates progressive masking effect
    gate = gate.cumsum(dim = -2)

    # Subtract gate values from attention logits (reduces attention to masked tokens)
    return sim - rearrange(gate, 'b i j -> b 1 i j')

# alternative distance functions

def qk_l2_dist_squared(q, k):
    """
    Compute squared L2 distance between query and key vectors.

    This provides an alternative to dot-product similarity for attention.
    Instead of measuring alignment (like dot product), it measures distance.
    Closer vectors have smaller distances and should receive higher attention.

    Args:
        q: Query tensor of shape (batch, heads, seq_len_q, dim).
        k: Key tensor of shape (batch, heads, seq_len_k, dim) or
           (batch, seq_len_k, dim) for multi-query attention.

    Returns:
        Squared L2 distances of shape (batch, heads, seq_len_q, seq_len_k).
    """
    # Handle multi-query attention case where k has no head dimension
    if k.ndim == 3:
        k = repeat(k, 'b j d -> b h j d', h = q.shape[1])

    # Pack tensors to combine batch and head dimensions for efficient cdist computation
    q, packed_shape = pack_one(q, '* i d')
    k, _ = pack_one(k, '* j d')

    # Compute pairwise L2 distances and square them
    l2_dist_squared = torch.cdist(q, k) ** 2
    # Restore original shape structure
    return unpack_one(l2_dist_squared, packed_shape, '* i j')

# one-hot straight through softmax

def one_hot_straight_through(logits, temperature = 1.):
    """
    One-hot attention with straight-through gradient estimator.

    Forward pass: Select a single token via argmax (hard attention).
    Backward pass: Use softmax gradients (soft attention).

    This allows discrete selection during inference while maintaining
    differentiability for training.

    Args:
        logits: Attention logits of shape (..., seq_len).
        temperature: Temperature for softmax in backward pass. Default: 1.0.

    Returns:
        One-hot tensor in forward pass, soft attention in backward pass.
        Shape: same as logits.
    """
    # Get indices of maximum logit values (hard selection)
    one_hot_indices = logits.argmax(dim = -1, keepdim = True)
    # Create one-hot vectors
    one_hot = torch.zeros_like(logits).scatter(-1, one_hot_indices, 1.)

    # Compute soft attention for gradient flow
    soft_attn = (logits / temperature).softmax(dim = -1)
    # Straight-through estimator: forward = one_hot, backward = soft_attn
    return one_hot + soft_attn - soft_attn.detach()

# sparse topk attention - only keep topk attn logits for softmax
# optional straight through with masked out logits by setting `attn_sparse_topk_straight_through = True`

def sparse_topk_attn(
    logits,
    sparse_topk,
    temperature = 1.,
    straight_through = False
):
    """
    Sparse top-k attention: only attend to the k tokens with highest similarity.

    This reduces computation and can improve performance by focusing attention
    on the most relevant tokens. Based on ideas from sparse attention papers.

    Args:
        logits: Attention logits of shape (..., seq_len).
        sparse_topk: Number of top tokens to keep (k).
        temperature: Temperature for softmax in straight-through mode. Default: 1.0.
        straight_through: If True, use straight-through gradient estimator
                         (topk in forward, full softmax in backward). Default: False.

    Returns:
        Attention weights with only top-k values preserved, others set to 0.
        Shape: same as logits.
    """
    orig_logits = logits

    # Mask value for entries not in top-k
    mask_value = -torch.finfo(logits.dtype).max
    # Find the top-k values
    top_values, _ = logits.topk(sparse_topk, dim = -1)
    # Create mask for values >= k-th largest value
    sparse_topk_mask = (logits >= top_values[..., -1:]) & (logits > mask_value)
    # Mask out non-top-k values
    logits = logits.masked_fill(~sparse_topk_mask, mask_value)
    # Apply softmax (only top-k values will have non-zero probabilities)
    topk_attn = logits.softmax(dim = -1)

    if not straight_through:
        return topk_attn

    # Straight-through mode: sparse in forward, full softmax in backward
    soft_attn = (orig_logits / temperature).softmax(dim = -1)
    return topk_attn.detach() + soft_attn - soft_attn.detach()

# functions for creating causal mask
# need a special one for onnx cpu (no support for .triu)

def create_causal_mask(i, j, device):
    """
    Create a causal (upper triangular) mask for autoregressive attention.

    The mask prevents tokens from attending to future positions. Position i can
    only attend to positions <= i.

    Args:
        i: Query sequence length.
        j: Key sequence length.
        device: Device to create the mask on.

    Returns:
        Boolean tensor of shape (i, j) where True indicates positions that
        should be masked out (future positions). Uses .triu() operation.
    """
    return torch.ones((i, j), device = device, dtype = torch.bool).triu(j - i + 1)

def onnx_create_causal_mask(i, j, device):
    """
    Create a causal mask compatible with ONNX export.

    This version doesn't use .triu() which is not supported on ONNX CPU runtime.
    Instead, it uses element-wise comparisons to create the same mask pattern.

    Args:
        i: Query sequence length.
        j: Key sequence length.
        device: Device to create the mask on.

    Returns:
        Boolean tensor of shape (i, j) where True indicates positions that
        should be masked out (future positions). ONNX-compatible implementation.
    """
    r = torch.arange(i, device = device)
    # Compare row indices with column indices to create lower triangular pattern
    causal_mask = rearrange(r, 'i -> i 1') < rearrange(r, 'j -> 1 j')
    # Pad if key sequence is longer than query sequence
    causal_mask = F.pad(causal_mask, (j - i, 0), value = False)
    return causal_mask

# main class

class Attend(Module):
    """
    Flexible attention mechanism with support for various attention variants.

    This class provides a unified interface for different types of attention:
    - Standard scaled dot-product attention
    - Flash attention (memory-efficient GPU implementation)
    - Causal/autoregressive attention
    - Talking heads (learned linear transformations between heads)
    - Selective attention (tokens control their visibility)
    - Sparse top-k attention
    - L2 distance-based attention
    - Sigmoid, hard, and Gumbel softmax attention
    - CoG attention (with negative weights)
    - Contextual positional encoding (CoPE)

    The class handles various technical details like:
    - Grouped multi-query attention (GQA/MQA)
    - KV caching for efficient generation
    - Attention masking and causal masking
    - Attention dropout
    - Numerical stability (softclamping, proper masking)
    """

    def __init__(
        self,
        *,
        dropout = 0.,
        causal = False,
        heads = None,
        pre_talking_heads = False,
        post_talking_heads = False,
        pre_scale_post_talking_heads = False,
        sparse_topk = None,
        sparse_topk_straight_through = False, # https://arxiv.org/abs/2505.22074
        scale = None,
        qk_norm = False,
        l2_distance = False,
        sigmoid = False,
        gumbel_softmax = False,
        gumbel_softmax_temp = 1.,
        gumbel_softmax_hard = True,
        cog_signed = False,
        custom_attn_fn: Callable | None = None,
        flash = False,
        softclamp_logits = False,
        logit_softclamp_value = 50.,
        add_zero_kv = False,
        head_learned_sink = False,
        selective = False,
        hard = False,
        cope = None,
        onnxable = False,
        sdp_kwargs: dict = dict(
            enable_flash = True,
            enable_math = True,
            enable_mem_efficient = True
        )
    ):
        """
        Initialize the Attend module.

        Args:
            dropout: Dropout probability for attention weights. Default: 0.
            causal: If True, apply causal masking (autoregressive). Default: False.
            heads: Number of attention heads (required for talking heads). Default: None.
            pre_talking_heads: Apply learned head mixing before softmax. Default: False.
            post_talking_heads: Apply learned head mixing after softmax. Default: False.
            pre_scale_post_talking_heads: Use pre-softmax head mixing to scale
                post-softmax attention. Default: False.
            sparse_topk: If set, only keep top-k attention logits. Default: None.
            sparse_topk_straight_through: Use straight-through estimator for
                sparse topk attention. Default: False.
            scale: Manual attention scale factor. If None, uses 1/sqrt(dim). Default: None.
            qk_norm: Whether queries and keys are normalized (affects dtype
                handling in softmax). Default: False.
            l2_distance: Use L2 distance instead of dot product for attention. Default: False.
            sigmoid: Use sigmoid instead of softmax for attention. Default: False.
            gumbel_softmax: Use Gumbel softmax for hard attention. Default: False.
            gumbel_softmax_temp: Temperature for Gumbel softmax. Default: 1.0.
            gumbel_softmax_hard: Whether to use hard Gumbel softmax. Default: True.
            cog_signed: Enable CoG attention with signed (negative) attention weights. Default: False.
            custom_attn_fn: Custom attention function to use instead of softmax. Default: None.
            flash: Use PyTorch's flash attention (scaled_dot_product_attention). Default: False.
            softclamp_logits: Apply soft clamping to attention logits. Default: False.
            logit_softclamp_value: Value for soft clamping attention logits. Default: 50.
            add_zero_kv: Add zero key/value pair (attention sink). Default: False.
            head_learned_sink: Add learned per-head attention sink. Default: False.
            selective: Enable selective attention (tokens control visibility). Default: False.
            hard: Use hard (one-hot) attention. Default: False.
            cope: Contextual positional encoding module. Default: None.
            onnxable: Use ONNX-compatible operations. Default: False.
            sdp_kwargs: Keyword arguments for scaled_dot_product_attention backend
                selection. Only used when flash=True. Default: enables all backends.
        """
        super().__init__()
        # Store custom scale factor (if None, will use default 1/sqrt(dim) later)
        self.scale = scale

        # causal related

        self.causal = causal
        # Use ONNX-compatible causal mask creation if needed
        self.create_causal_mask = onnx_create_causal_mask if onnxable else create_causal_mask

        # attention type - validate mutual exclusivity and set attention function

        is_sparse_topk_attn = exists(sparse_topk)

        # Flash attention incompatibilities - flash is optimized for standard softmax
        assert not (flash and sigmoid), 'sigmoid attention not available for flash'
        assert not (flash and hard), 'hard attention not available for flash'
        assert not (flash and is_sparse_topk_attn), 'topk attention not available for flash'

        # Ensure only one alternative attention type is specified
        assert at_most_one_of(sigmoid, hard, l2_distance, gumbel_softmax, is_sparse_topk_attn)

        # Select the appropriate attention function based on configuration
        if exists(custom_attn_fn):
            self.attn_fn = custom_attn_fn
        elif sigmoid:
            # Sigmoid attention: bounded attention weights in [0, 1]
            self.attn_fn = F.sigmoid
        elif hard:
            # Hard attention: select single token, straight-through gradients
            self.attn_fn = one_hot_straight_through
        elif is_sparse_topk_attn:
            # Sparse attention: only attend to top-k most similar tokens
            self.attn_fn = partial(sparse_topk_attn, sparse_topk = sparse_topk, straight_through = sparse_topk_straight_through)
        elif gumbel_softmax:
            # Gumbel softmax: differentiable sampling from categorical distribution
            self.attn_fn = partial(F.gumbel_softmax, dim = -1, tau = gumbel_softmax_temp, hard = gumbel_softmax_hard)
        else:
            # Standard softmax attention
            softmax_fn = partial(F.softmax, dim = -1)
            # Use float32 for numerical stability unless using normalized queries/keys
            self.attn_fn = partial(softmax_fn, dtype = torch.float32) if not qk_norm else softmax_fn

        # dropouts

        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        # talking heads - learned linear mixing of attention heads
        # see "Talking-Heads Attention" - https://arxiv.org/abs/2003.02436

        assert not (flash and (pre_talking_heads or post_talking_heads or pre_scale_post_talking_heads)), 'talking heads not compatible with flash attention'

        # Pre-softmax talking heads: mix heads before normalization
        self.pre_softmax_talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if pre_talking_heads else None
        # Post-softmax talking heads: mix heads after normalization
        self.post_softmax_talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if post_talking_heads else None
        # Pre-scale variant: compute scaling factors pre-softmax, apply post-softmax
        self.pre_scale_post_talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if pre_scale_post_talking_heads else None

        # Initialize talking heads weights to identity (Dirac delta) - starts as no-op
        if exists(self.pre_softmax_talking_heads):
            nn.init.dirac_(self.pre_softmax_talking_heads.weight)

        if exists(self.post_softmax_talking_heads):
            nn.init.dirac_(self.post_softmax_talking_heads.weight)

        if exists(self.pre_scale_post_talking_heads):
            # An improvisation where heads are combined pre-softmax, then used to scale post-softmax attention
            nn.init.dirac_(self.pre_scale_post_talking_heads.weight)

        # selective attention - tokens can control whether they're attended to
        # see https://arxiv.org/abs/2410.02703

        assert not (flash and selective), 'selective attention cannot work on flash attention'
        assert not (selective and not causal), 'selective attention is designed for autoregressive'
        self.selective = selective

        # cog attention - negative weights for expressiveness
        # allows negative attention weights for greater model expressiveness
        # https://openreview.net/forum?id=ezRrwwbxd0

        assert not (flash and cog_signed), 'cog attention not available for flash'
        self.cog_signed = cog_signed

        # l2 distance attention - use Euclidean distance instead of dot product

        self.l2_distance = l2_distance

        # add a key / value token composed of zeros
        # can help control outliers, proposed by https://www.evanmiller.org/attention-is-off-by-one.html
        # acts as a "null" token that can absorb unwanted attention

        self.add_zero_kv = add_zero_kv

        # learned sink concatenated pre-softmax, working solution from gpt-oss
        # provides a learned "sink" for attention to flow to

        assert not (head_learned_sink and flash), f'not supported for flash attention yet'

        self.head_learned_sink = head_learned_sink
        # Learnable attention sink parameter (one per head)
        self.head_attn_sink = Parameter(torch.zeros(heads)) if head_learned_sink else None

        # soft clamp attention logit value
        # prevents extreme logit values which can cause numerical instability

        if softclamp_logits:
            assert not flash, 'flash attention not compatible with logit softclamp value yet'
            assert logit_softclamp_value > 0.

        self.softclamp_logits = softclamp_logits
        self.logit_softclamp_value = logit_softclamp_value

        # contextual positional encoding (CoPE)
        # dynamic positional encoding based on context

        self.cope = cope

        # flash attention - PyTorch's optimized CUDA kernel for attention
        # significantly faster and more memory efficient than manual implementation

        self.flash = flash

        torch_version = version.parse(torch.__version__)
        assert not (flash and torch_version < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # torch 2.3 uses new backend selection API with context manager

        if self.flash:
            if torch_version >= version.parse('2.3'):
                # PyTorch 2.3+ uses SDPBackend enum for backend selection
                from torch.nn.attention import SDPBackend

                str_to_backend = dict(
                    enable_flash = SDPBackend.FLASH_ATTENTION,
                    enable_mem_efficient = SDPBackend.EFFICIENT_ATTENTION,
                    enable_math = SDPBackend.MATH,
                    enable_cudnn = SDPBackend.CUDNN_ATTENTION
                )

                # Convert string keys to backend enum values
                sdpa_backends = [str_to_backend[enable_str] for enable_str, enable in sdp_kwargs.items() if enable]

                # Create context manager with selected backends
                self.sdp_context_manager = partial(torch.nn.attention.sdpa_kernel, sdpa_backends)
            else:
                # PyTorch 2.0-2.2 uses CUDA backend kwargs
                self.sdp_context_manager = partial(torch.backends.cuda.sdp_kernel, **sdp_kwargs)

    def flash_attn(
        self,
        q, k, v,
        mask = None,
        attn_bias = None
    ):
        """
        Compute attention using PyTorch's optimized flash attention kernel.

        Flash attention is a memory-efficient and faster implementation of attention
        that uses kernel fusion and tiling to reduce memory access. This is the
        preferred method when available and compatible with the attention configuration.

        Args:
            q: Query tensor of shape (batch, heads, seq_len_q, dim).
            k: Key tensor of shape (batch, heads, seq_len_k, dim) or
               (batch, seq_len_k, dim) for multi-query attention.
            v: Value tensor of shape (batch, heads, seq_len_k, dim) or
               (batch, seq_len_k, dim) for multi-query attention.
            mask: Optional attention mask of shape (batch, 1, 1, seq_len_k) or
                 (batch, heads, seq_len_q, seq_len_k). True means attend, False means mask.
            attn_bias: Optional additive attention bias (e.g., ALiBi positional bias).

        Returns:
            Tuple of (output, intermediates) where:
            - output: Attention output of shape (batch, heads, seq_len_q, dim)
            - intermediates: Empty Intermediates object (flash attention doesn't expose internals)
        """
        batch, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        # Recommended for multi-query single-key-value attention by Tri Dao
        # Expand k and v to have head dimension if using multi-query attention
        # kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])

        if k.ndim == 3:
            k = repeat(k, 'b ... -> b h ...', h = q.shape[1])

        if v.ndim == 3:
            v = repeat(v, 'b ... -> b h ...', h = q.shape[1])

        # handle maybe l2 distance
        # Transform q and k to compute L2 distance via dot product
        # Uses the identity: ||q-k||^2 = ||q||^2 + ||k||^2 - 2*q·k

        if self.l2_distance:
            # Augment k with its squared norm
            k_norm_sq = k.norm(dim = -1, keepdim = True) ** 2
            k = F.pad(k, (0, 1), value = -1.)
            k = cat((k, k_norm_sq), dim = -1)

            # Augment q to compute -2*q·k + ||q||^2 in single dot product
            q_norm_sq = q.norm(dim = -1, keepdim = True) ** 2
            q = cat((2 * q, q_norm_sq), dim = -1)
            q = F.pad(q, (0, 1), value = -1.)

        # handle scale - by default flash attention scales by dim_head ** -0.5
        # Need to adjust if using a custom scale (e.g., for cosine similarity attention)

        if exists(self.scale):
            default_scale = q.shape[-1] ** -0.5
            # Modify q to achieve custom scale while letting flash attention apply default
            q = q * (self.scale / default_scale)

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        causal = self.causal

        # in the case of kv caching with one token (q_len == 1), just turn off causal masking
        # Single token can attend to all previous tokens, no future tokens to mask
        # in speculative decoding, this may go up to 5-6, so right aligned causal mask will be needed there

        if q_len == 1 and causal:
            causal = False

        # expand key padding mask to full attention matrix shape

        if exists(mask):
            assert mask.ndim == 4
            mask = mask.expand(batch, heads, q_len, k_len)

        # handle kv cache - this should be bypassable in updated flash attention 2
        # When using KV cache, k_len > q_len. Need explicit causal mask for the cached portion

        if k_len > q_len and causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device = device)
            if not exists(mask):
                mask = ~causal_mask
            else:
                mask = mask & ~causal_mask
            causal = False

        # manually handle causal mask, if another mask was given
        # Flash attention's is_causal flag can't combine with custom masks

        if exists(mask) and causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device = device)
            mask = mask & ~causal_mask
            causal = False

        # protect against an entire row being masked out
        # Track this to zero out outputs for fully masked positions later

        row_is_entirely_masked = None

        if exists(mask):
            row_is_entirely_masked = ~mask.any(dim = -1)

        # handle alibi positional bias
        # convert from bool to float
        # Flash attention accepts mask as either boolean or additive bias

        if exists(attn_bias):
            attn_bias = attn_bias.expand(batch, heads, -1, -1)

            # if mask given, the mask would already contain the causal mask from above logic
            # otherwise, if no mask given but still causal, mask out alibi positional bias to a large negative number

            mask_value = -torch.finfo(q.dtype).max

            if exists(mask):
                # Combine boolean mask with additive bias
                attn_bias = attn_bias.masked_fill(~mask, mask_value // 2)
            elif causal:
                # Apply causal mask to bias
                causal_mask = self.create_causal_mask(q_len, k_len, device = device)
                attn_bias = attn_bias.masked_fill(causal_mask, mask_value // 2)
                causal = False

            # scaled_dot_product_attention handles attn_mask either as bool or additive bias
            # make it an additive bias here

            mask = attn_bias

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale
        # Use context manager to select appropriate backend (flash/efficient/math)

        with self.sdp_context_manager():
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout if self.training else 0.,
                is_causal = causal
            )

        # for a row that is entirely masked out, should zero out the output of that row token
        # This prevents NaN/inf values from propagating

        if exists(row_is_entirely_masked) and row_is_entirely_masked.any():
            out = out.masked_fill(row_is_entirely_masked[..., None], 0.)

        return out, Intermediates()

    def forward(
        self,
        q, k, v,
        mask = None,
        attn_bias = None,
        prev_attn = None
    ):
        """
        Compute attention over queries, keys, and values.

        This is the main attention computation method that implements the complete
        attention mechanism with all configured features (talking heads, masking,
        selective attention, etc.).

        Einstein notation used in this method:
        - b: batch size
        - h: number of attention heads
        - n, i, j: sequence lengths (base, query, key respectively)
        - d: feature dimension per head

        Args:
            q: Query tensor of shape (batch, heads, seq_len_q, dim).
            k: Key tensor of shape (batch, kv_heads, seq_len_k, dim) or
               (batch, seq_len_k, dim) for multi-query attention.
            v: Value tensor of shape (batch, kv_heads, seq_len_k, dim) or
               (batch, seq_len_k, dim) for multi-query attention.
            mask: Optional key padding mask of shape (batch, seq_len_k) or
                 (batch, 1, 1, seq_len_k). True means attend, False means mask out.
            attn_bias: Optional additive attention bias (e.g., ALiBi, RoPE bias).
                      Shape: (batch, heads, seq_len_q, seq_len_k).
            prev_attn: Optional previous attention logits for residual attention.
                      Shape: (batch, heads, seq_len_q, seq_len_k).

        Returns:
            Tuple of (output, intermediates) where:
            - output: Attention output of shape (batch, heads, seq_len_q, dim)
            - intermediates: Intermediates object containing attention values for
                           debugging/analysis (qk_similarities, pre/post softmax attention)
        """

        n, heads, kv_heads, device = q.shape[-2], q.shape[1], k.shape[1], q.device

        # Determine attention scale factor (default: 1/sqrt(dim))
        scale = default(self.scale, q.shape[-1] ** -0.5)

        causal = self.causal

        # handle key padding mask - expand from (batch, seq_len) to attention matrix shape

        if exists(mask) and mask.ndim == 2:
            mask = rearrange(mask, 'b j -> b 1 1 j')

        # handle kv cached decoding
        # When generating one token at a time, no need for causal mask

        if n == 1 and causal:
            causal = False

        # handle grouped multi-query attention (GQA) and multi-query attention (MQA)
        # GQA: fewer KV heads than Q heads (e.g., 8 Q heads, 2 KV heads)
        # MQA: single KV head shared across all Q heads

        if kv_heads == 1:
            # Multi-query attention: remove head dimension from k and v
            k, v = tuple(rearrange(t, 'b 1 n d -> b n d') for t in (k, v))
        elif kv_heads < heads:
            # Grouped multi-query: repeat KV heads to match number of Q heads
            k, v = tuple(repeat(t, 'b kvh n d -> b (r kvh) n d', r = heads // kv_heads) for t in (k, v))

        # handle zero kv, as means for allowing network to attend to nothing
        # Adds a "null" token at the beginning that model can attend to if needed
        # Helps with outlier control and gives model option to "not attend" anywhere

        if self.add_zero_kv:
            # Prepend zero vector to key and value sequences
            k, v = tuple(F.pad(t, (0, 0, 1, 0), value = 0.) for t in (k, v))

            if exists(mask):
                # Allow attention to the zero key/value
                mask = F.pad(mask, (1, 0), value = True)

            if exists(attn_bias):
                # Zero bias for the zero key/value position
                attn_bias = F.pad(attn_bias, (1, 0), value = 0.)

        # Use flash attention if enabled
        if self.flash:
            assert not exists(prev_attn), 'residual attention not compatible with flash attention'
            return self.flash_attn(q, k, v, mask = mask, attn_bias = attn_bias)

        # Manual attention computation
        # Choose einsum equation based on whether k/v have head dimension

        kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'

        # Compute query-key similarities
        if not self.l2_distance:
            # Standard dot-product attention
            sim = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k)
        else:
            # L2 distance attention (negated so smaller distance = higher attention)
            sim = -qk_l2_dist_squared(q, k)

        # Apply attention scale
        sim = sim * scale

        # Add residual attention if provided (for architectures like ReLA)
        if exists(prev_attn):
            sim = sim + prev_attn

        # Store raw similarities for analysis
        qk_similarities = sim.clone()

        # Pre-scale post talking heads: compute scaling factors before softmax
        if exists(self.pre_scale_post_talking_heads):
            pre_to_post_scale = self.pre_scale_post_talking_heads(sim)

        # Pre-softmax talking heads: mix attention heads before normalization
        if exists(self.pre_softmax_talking_heads):
            sim = sim + self.pre_softmax_talking_heads(sim)

        # Add attention bias (e.g., ALiBi positional bias)
        if exists(attn_bias):
            sim = sim + attn_bias

        # Soft clamp logits to prevent extreme values
        if self.softclamp_logits:
            sim = softclamp(sim, self.logit_softclamp_value)

        # pre-masking - handle cog by storing sign
        # CoG attention allows negative attention weights for expressiveness

        if self.cog_signed:
            # Store signs and work with absolute values through softmax
            sim_sign = sim.sign()
            sim = sim.abs()

        # masking - apply attention masks

        i, j, dtype = *sim.shape[-2:], sim.dtype

        # Large negative value to mask out positions
        mask_value = -torch.finfo(sim.dtype).max

        # Apply key padding mask
        if exists(mask):
            sim = sim.masked_fill(~mask, mask_value)

        # Apply causal mask (prevent attending to future positions)
        if causal:
            causal_mask = self.create_causal_mask(i, j, device = device)
            sim = sim.masked_fill(causal_mask, mask_value)

        # Track rows that are entirely masked (needed for zeroing output later)
        row_is_entirely_masked = None

        if exists(mask):
            row_is_entirely_masked = ~mask.any(dim = -1)

        # Apply contextual positional encoding (CoPE) if configured
        if exists(self.cope):
            sim = sim + self.cope(q, sim)

        # Apply selective attention (tokens control their visibility to future)
        if self.selective:
            sim = selective_attn(sim)

        # Add learned attention sink if configured
        if self.head_learned_sink:
            # add learned attention sink - per-head learnable value that can absorb attention
            attn_sink = repeat(self.head_attn_sink, 'h -> b h i 1', b = sim.shape[0], i = sim.shape[2])

            # Handle sink for CoG attention
            if self.cog_signed:
                attn_sink, attn_sink_sign = attn_sink.abs(), attn_sink.sign()
                sim_sign = cat((attn_sink_sign, sim_sign), dim = -1)

            # Prepend sink to attention logits
            sim = cat((attn_sink, sim), dim = -1)

        # Store pre-softmax attention for analysis
        pre_softmax_attn = sim

        # Apply attention function (softmax, sigmoid, gumbel, etc.)
        attn = self.attn_fn(sim)

        # Restore original dtype (attn_fn may cast to float32)
        attn = attn.type(dtype)

        # add back the sign for CoG attention

        if self.cog_signed:
            # Multiply attention weights by stored signs (allows negative attention)
            attn = attn * sim_sign

        # Store post-softmax attention for analysis
        post_softmax_attn = attn

        # Remove attention sink if it was added
        if self.head_learned_sink:
            # remove attention sink - exclude first position
            attn = attn[..., 1:]

        # Apply dropout to attention weights
        attn = self.attn_dropout(attn)

        # Post-softmax talking heads: mix attention heads after normalization
        if exists(self.post_softmax_talking_heads):
            attn = self.post_softmax_talking_heads(attn)

        # Apply pre-computed scaling if using pre-scale post talking heads
        if exists(self.pre_scale_post_talking_heads):
            attn = attn * pre_to_post_scale

        # Compute attention output: weighted sum of values
        out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)

        # Package intermediate values for return
        intermediates = Intermediates(
            qk_similarities = qk_similarities,
            pre_softmax_attn = pre_softmax_attn,
            post_softmax_attn = post_softmax_attn
        )

        # Zero out outputs for fully masked positions (prevents NaN/inf propagation)
        if exists(row_is_entirely_masked) and row_is_entirely_masked.any():
            out = out.masked_fill(row_is_entirely_masked[..., None], 0.)

        return out, intermediates
