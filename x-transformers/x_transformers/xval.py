"""
XVal: Transformer with Continuous Numerical Values

This module implements a transformer that handles both discrete tokens and
continuous numerical values, as described in:
"XVal: A Continuous Number Encoding for Large Language Models"
https://arxiv.org/abs/2310.02989

Regular transformers use discrete tokens for everything, including numbers.
XVal improves arithmetic reasoning by:
1. Using a special numerical token ID to mark numerical values
2. Scaling token embeddings by the numerical value
3. Predicting both next token and numerical value simultaneously

This allows the model to generalize better for arithmetic and numerical tasks.
"""

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from typing import Callable
from collections import namedtuple

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

from x_transformers.x_transformers import (
    AttentionLayers,
    TokenEmbedding,
    ScaledSinusoidalEmbedding,
    AbsolutePositionalEmbedding,
    always
)

from x_transformers.autoregressive_wrapper import (
    top_k,
    top_p
)

# constants

# Named tuple for detailed loss breakdown
LossBreakdown = namedtuple('LossBreakdown', ['cross_entropy_loss', 'numerical_mse_loss'])

# Named tuple for generation outputs
GenerateReturn = namedtuple('GenerateReturn', ['sampled_token_ids', 'sampled_numbers', 'is_number_mask'])

# helper functions

def exists(val):
    """
    Check if a value is not None.

    Args:
        val: Any value to check

    Returns:
        bool: True if val is not None, False otherwise
    """
    return val is not None

def default(val, d):
    """
    Return the value val if it exists, otherwise return default value d.

    Args:
        val: The value to check
        d: The default value to return if val is None. Can be a callable.

    Returns:
        Either val or d (or d() if d is callable) depending on whether val exists
    """
    if exists(val):
        return val
    return d() if callable(d) else d

# main classes

class XValTransformerWrapper(nn.Module):
    """
    Transformer Wrapper for XVal (continuous numerical values).

    This wrapper extends a standard transformer to handle continuous numerical
    values alongside discrete tokens. When a token is marked as numerical
    (by numerical_token_id), its embedding is scaled by the numerical value.

    The model predicts both:
    1. Next token ID (via standard cross-entropy)
    2. Numerical value (via regression for numerical tokens)

    Args:
        num_tokens (int): Vocabulary size
        max_seq_len (int): Maximum sequence length
        numerical_token_id (int): Special token ID that marks numerical values
        attn_layers (AttentionLayers): The attention layers (encoder/decoder)
        emb_dim (int, optional): Embedding dimension. Defaults to attn_layers.dim
        logits_dim (int, optional): Output logits dimension. Defaults to num_tokens
        tie_embedding (bool): If True, tie input and output embeddings. Defaults to False.
        max_mem_len (int): Maximum memory length for transformer-xl style models.
            Defaults to 0.
        num_memory_tokens (int, optional): Number of memory tokens to prepend.
            Defaults to None (no memory tokens).
        emb_dropout (float): Embedding dropout rate. Defaults to 0.
        use_abs_pos_emb (bool): Whether to use absolute positional embeddings.
            Defaults to True.
        scaled_sinu_pos_emb (bool): If True, use scaled sinusoidal positional
            embeddings. Defaults to False.

    Attributes:
        token_emb: Token embedding layer
        numerical_token_id: Special token ID for numerical values
        pos_emb: Positional embedding
        memory_tokens: Optional learnable memory tokens
        attn_layers: Attention layers
        to_logits: Projection to vocabulary logits
        to_numerical_output: Projection to numerical value prediction
    """
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        numerical_token_id,
        attn_layers: AttentionLayers,
        emb_dim = None,
        logits_dim = None,
        tie_embedding = False,
        max_mem_len = 0,
        num_memory_tokens = None,
        emb_dropout = 0.,
        use_abs_pos_emb = True,
        scaled_sinu_pos_emb = False
    ):
        super().__init__()
        dim = attn_layers.dim
        emb_dim = default(emb_dim, dim)

        self.emb_dim = emb_dim
        # Standard token embedding layer
        self.token_emb = TokenEmbedding(emb_dim, num_tokens)

        # Special token ID that indicates a numerical value
        self.numerical_token_id = numerical_token_id

        self.max_seq_len = max_seq_len

        self.max_mem_len = max_mem_len

        # Configure positional embeddings
        if not (use_abs_pos_emb and not attn_layers.disable_abs_pos_emb):
            self.pos_emb = always(0)  # No positional embedding
        elif scaled_sinu_pos_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(dim)
        else:
            self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len)

        self.emb_dropout = nn.Dropout(emb_dropout)

        # memory tokens

        # Optional learnable memory tokens prepended to sequence
        num_memory_tokens = default(num_memory_tokens, 0)
        self.has_memory_tokens = num_memory_tokens > 0

        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

        # attention layers

        self.attn_layers = attn_layers

        # to logits

        # Output projection to vocabulary
        logits_dim = default(logits_dim, num_tokens)
        self.to_logits = nn.Linear(dim, logits_dim) if not tie_embedding else lambda t: t @ self.token_emb.emb.weight.t()

        # Output projection to numerical value (single scalar per position)
        self.to_numerical_output = nn.Sequential(
            nn.Linear(dim, 1),
            Rearrange('... 1 -> ...')  # Remove last dimension to get scalar
        )

    def forward(
        self,
        x: Tensor,
        x_num: Tensor,
        return_embeddings = False,
        return_intermediates = False,
        return_mems = False,
        mask = None,
        return_attn = False,
        mems = None,
        pos = None,
        prepend_embeds = None,
        **kwargs
    ):
        """
        Forward pass through the XVal transformer.

        The key XVal innovation:
        1. Embed tokens normally
        2. For positions where token == numerical_token_id, scale the embedding
           by the numerical value in x_num
        3. Process through transformer
        4. Predict both next token and numerical value

        Args:
            x (Tensor): Token IDs of shape (batch, seq_len)
            x_num (Tensor): Numerical values of shape (batch, seq_len).
                For non-numerical positions, values are ignored (usually set to 1 or nan)
            return_embeddings (bool): If True, return embeddings instead of logits.
                Defaults to False.
            return_intermediates (bool): If True, return intermediate layer outputs.
                Defaults to False.
            return_mems (bool): If True, return memory for transformer-xl. Defaults to False.
            mask (Tensor, optional): Attention mask
            return_attn (bool): If True, return attention maps. Defaults to False.
            mems (tuple, optional): Memory from previous forward pass (transformer-xl)
            pos (Tensor, optional): Custom positional indices
            prepend_embeds (Tensor, optional): Embeddings to prepend (e.g., image embeddings)
            **kwargs: Additional arguments passed to attention layers

        Returns:
            If return_embeddings is False:
                tuple: (logits, numerical_pred)
                    - logits: Token logits of shape (batch, seq_len, num_tokens)
                    - numerical_pred: Numerical predictions of shape (batch, seq_len)
            If return_embeddings is True:
                Tensor: Final embeddings of shape (batch, seq_len, dim)

            Additional returns based on flags:
            - return_intermediates: (out, intermediates)
            - return_mems: (out, new_mems)
            - return_attn: (out, attn_maps)
        """
        assert x.shape == x_num.shape

        batch = x.shape[0]

        # Identify positions that contain numerical values
        is_number_mask = x == self.numerical_token_id

        # Get token embeddings
        x = self.token_emb(x)

        # Scale embeddings by numerical value for numerical positions
        # For non-numerical positions, scale by 1 (no change)
        scale = torch.where(is_number_mask, x_num, 1.)
        scale = rearrange(scale, '... -> ... 1')

        # Apply scaling (this is the key XVal innovation)
        x = x * scale

        # Add positional embeddings
        x = x + self.pos_emb(x, pos = pos)

        # memory tokens

        # Prepend learnable memory tokens if configured
        if self.has_memory_tokens:
            m = repeat(self.memory_tokens, 'm d -> b m d', b = batch)
            x, mem_ps = pack([m, x], 'b * d')

            # Extend mask to include memory tokens (all visible)
            if exists(mask):
                num_mems = m.shape[-2]
                mask = pad_at_dim(mask, (num_mems, 0), dim = -1, value = True)

        # whether to append embeds, as in PaLI, for image embeddings

        # Optionally prepend additional embeddings (e.g., for multimodal models)
        if exists(prepend_embeds):
            _, prepend_dim = prepend_embeds.shape[1:]
            assert prepend_dim == x.shape[-1], 'prepended embeddings need to have same dimensions as model dimensions'

            x = torch.cat((prepend_embeds, x), dim = -2)

        # Apply embedding dropout
        x = self.emb_dropout(x)

        # attention layers

        # Process through transformer attention layers
        x, intermediates = self.attn_layers(x, mask = mask, mems = mems, return_hiddens = True, **kwargs)

        # splice out memory tokens

        # Remove memory tokens from output
        if self.has_memory_tokens:
            m, x = unpack(x, mem_ps, 'b * d')
            intermediates.memory_tokens = m

        # Project to outputs (logits and numerical predictions)
        if not return_embeddings:
            logits = self.to_logits(x)
            numerical_pred = self.to_numerical_output(x)
            out = (logits, numerical_pred)
        else:
            out = x

        # Return additional information based on flags
        if return_intermediates:
            return out, intermediates

        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = tuple(t[..., -self.max_mem_len:, :].detach() for t in hiddens)
            return out, new_mems

        if return_attn:
            attn_maps = tuple(t.post_softmax_attn for t in intermediates.attn_intermediates)
            return out, attn_maps

        return out

class XValAutoregressiveWrapper(nn.Module):
    """
    Autoregressive Wrapper for XVal Transformer.

    Wraps an XValTransformerWrapper to handle autoregressive training and generation.
    This wrapper:
    1. Shifts inputs for autoregressive prediction (predict next token/value)
    2. Computes combined loss: cross-entropy for tokens + MSE for numerical values
    3. Supports autoregressive generation with both token and numerical prediction

    Args:
        net (XValTransformerWrapper): The XVal transformer to wrap
        ignore_index (int): Index to ignore in loss computation (e.g., padding).
            Defaults to -100.
        pad_value (int): Padding value. Defaults to 0.
        numerical_loss_weight (float): Weight for numerical MSE loss relative to
            cross-entropy loss. Defaults to 1.0.

    Attributes:
        net: The wrapped XVal transformer
        max_seq_len: Maximum sequence length from the network
        numerical_loss_weight: Weight for numerical loss
        ignore_index: Index to ignore in loss
    """
    def __init__(
        self,
        net: XValTransformerWrapper,
        ignore_index = -100,
        pad_value = 0,
        numerical_loss_weight = 1.
    ):
        super().__init__()
        self.net = net
        self.max_seq_len = net.max_seq_len
        self.numerical_loss_weight = numerical_loss_weight
        self.ignore_index = ignore_index

    @torch.no_grad()
    def generate(
        self,
        start_tokens: Tensor,
        start_numbers: Tensor,
        seq_len,
        filter_logits_fn: Callable = top_k,
        filter_kwargs: dict = dict(),
        temperature = 1.,
        **kwargs
    ):
        """
        Generate sequences autoregressively with both tokens and numerical values.

        Generates sequences token by token, predicting both the next token ID
        and its numerical value (if it's a numerical token).

        Args:
            start_tokens (Tensor): Starting token IDs of shape (batch, start_len)
            start_numbers (Tensor): Starting numerical values of shape (batch, start_len)
            seq_len (int): Number of tokens to generate
            filter_logits_fn (Callable): Function to filter logits before sampling
                (e.g., top_k, top_p). Defaults to top_k.
            filter_kwargs (dict): Keyword arguments for filter_logits_fn. Defaults to {}.
            temperature (float): Sampling temperature. Higher = more random. Defaults to 1.0.
            **kwargs: Additional arguments passed to the network

        Returns:
            GenerateReturn: Named tuple containing:
                - sampled_token_ids: Generated token IDs of shape (batch, seq_len)
                - sampled_numbers: Generated numerical values of shape (batch, seq_len).
                  Non-numerical positions contain nan.
                - is_number_mask: Boolean mask indicating numerical positions

        Note:
            start_tokens and start_numbers must have the same shape and at least 2 dimensions
        """
        device = start_tokens.device
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        assert num_dims >= 2, 'number of dimensions of your start tokens must be greater or equal to 2'
        assert start_tokens.shape == start_numbers.shape

        b, t, device = *start_tokens.shape, start_tokens.device

        # Set to eval mode for generation
        self.net.eval()
        out = start_tokens
        num_out = start_numbers

        # Generate tokens one at a time
        for _ in range(seq_len):
            # Use only the last max_seq_len tokens as context
            x = out[:, -self.max_seq_len:]
            x_num = num_out[:, -self.max_seq_len:]

            # Get predictions for next token and numerical value
            logits, numerical_pred = self.net(x, x_num, **kwargs)

            # Extract predictions for the last position
            last_logits = logits[:, -1]
            last_num_pred = numerical_pred[:, -1:]

            # Apply filtering (e.g., top-k, top-p) to logits
            filtered_logits = filter_logits_fn(last_logits, **filter_kwargs)

            # Sample next token from filtered distribution
            probs = F.softmax(filtered_logits / temperature, dim=-1)

            sample = torch.multinomial(probs, 1)

            # Append sampled token and predicted numerical value
            out = torch.cat((out, sample), dim = -1)
            num_out = torch.cat((num_out, last_num_pred), dim = -1)

        # Remove the starting tokens, keep only generated ones
        out = out[:, t:]
        num_out = num_out[:, t:]

        # Set numerical values to nan for non-numerical tokens
        is_number = out == self.net.numerical_token_id
        num_out = torch.where(is_number, num_out, float('nan'))

        # Restore training mode
        self.net.train(was_training)
        return GenerateReturn(out, num_out, is_number)

    def forward(
        self,
        x: Tensor,
        x_num: Tensor,
        return_loss_breakdown = False,
        **kwargs
    ):
        """
        Forward pass for autoregressive training.

        Computes the combined loss:
        1. Cross-entropy loss for token prediction
        2. MSE loss for numerical value prediction (only for numerical tokens)

        The total loss is: cross_entropy + numerical_loss_weight * numerical_mse

        Args:
            x (Tensor): Token IDs of shape (batch, seq_len)
            x_num (Tensor): Numerical values of shape (batch, seq_len)
            return_loss_breakdown (bool): If True, return loss and breakdown.
                Defaults to False.
            **kwargs: Additional arguments passed to the network (e.g., mask)

        Returns:
            If return_loss_breakdown is False:
                Tensor: Combined loss (scalar)
            If return_loss_breakdown is True:
                tuple: (loss, LossBreakdown)
                    - loss: Combined loss (scalar)
                    - LossBreakdown: Named tuple with (cross_entropy_loss, numerical_mse_loss)
        """
        # Shift inputs for autoregressive prediction (predict next token)
        inp, target = x[:, :-1], x[:, 1:]
        x_num_inp, x_num_target = x_num[:, :-1], x_num[:, 1:]

        # ignore index

        # Create mask for valid targets (not padding/ignore)
        target_mask = target != self.ignore_index

        # key padding mask

        # Apply additional padding mask if provided
        mask = kwargs.get('mask', None)
        if exists(mask):
            target_mask &= mask

            # Adjust mask to match input length (shifted by 1)
            if mask.shape[1] == x.shape[1]:
                mask = mask[:, :-1]
                kwargs['mask'] = mask

        # Get predictions from network
        logits, numerical_pred = self.net(inp, x_num_inp, **kwargs)

        # Rearrange for cross_entropy (expects [batch, classes, seq])
        logits = rearrange(logits, 'b n c -> b c n')

        # Compute cross-entropy loss for token prediction
        cross_entropy_loss = F.cross_entropy(logits, target, reduction = 'none', ignore_index = self.ignore_index)

        # protect against nan in `x_num` input tensor

        # Identify which targets are numerical tokens
        target_is_number_mask = target == self.net.numerical_token_id
        # Replace nan values with 0 for numerical targets (to avoid nan in loss)
        x_num_target = x_num_target.masked_fill(~target_is_number_mask, 0.)

        # numerical mse loss

        # Compute MSE loss for numerical value prediction
        numerical_mse_loss = F.mse_loss(numerical_pred, x_num_target, reduction = 'none')

        # Only count MSE loss for valid positions
        numerical_mse_loss = numerical_mse_loss * target_mask
        # Only count MSE loss for positions that are actually numerical tokens
        numerical_mse_loss = numerical_mse_loss.masked_fill(~target_is_number_mask, 0.)

        # combine losses

        # Combine cross-entropy and MSE losses with weighting
        loss = cross_entropy_loss + numerical_mse_loss * self.numerical_loss_weight

        # Apply target mask and compute mean
        loss = loss[target_mask]
        loss = loss.mean()

        # Return just the loss or also the breakdown
        if not return_loss_breakdown:
            return loss

        return loss, LossBreakdown(cross_entropy_loss, numerical_mse_loss)
