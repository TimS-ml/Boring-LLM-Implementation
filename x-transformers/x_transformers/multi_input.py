from __future__ import annotations

import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleDict
import torch.nn.functional as F

from typing import Dict

from einops import pack, repeat, unpack

from x_transformers.x_transformers import (
    AttentionLayers,
    ScaledSinusoidalEmbedding,
    AbsolutePositionalEmbedding,
    LayerIntermediates,
    LayerNorm,
    always,
    pad_at_dim,
    is_empty,
)

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
    Return a default value if the given value doesn't exist (is None).

    Args:
        val: The value to check
        d: The default value or a callable that returns the default value

    Returns:
        The original value if it exists, otherwise the default value.
        If d is callable, it will be called to get the default value.
    """
    if exists(val):
        return val
    return d() if callable(d) else d


class MultiInputTransformerWrapper(Module):
    """
    A flexible transformer wrapper that can handle multiple input types with different embeddings.

    This class wraps attention layers and handles multiple types of categorical inputs, each with
    their own embedding tables. The embeddings from different input types are summed together
    before being processed by the attention layers. This is useful for scenarios where you have
    multiple input modalities or types (e.g., different token types, segment IDs, etc.) that need
    to be embedded separately and then combined.

    The class also supports various advanced features like:
    - Memory tokens (similar to [CLS] tokens in BERT)
    - Cached key-value decoding for efficient generation
    - Gradient scaling for embeddings
    - Positional embeddings (absolute or scaled sinusoidal)
    - Prepending embeddings (e.g., for vision-language models like PaLI)
    """

    def __init__(
        self,
        *,
        num_tokens: Dict[str, int] = dict(),
        max_seq_len,
        attn_layers: AttentionLayers,
        emb_dim = None,
        max_mem_len = 0,
        shift_mem_down = 0,
        emb_dropout = 0.,
        post_emb_norm = False,
        num_memory_tokens = None,
        memory_tokens_interspersed_every = None,
        return_only_embed = False,
        use_abs_pos_emb = True,
        scaled_sinu_pos_emb = False,
        emb_frac_gradient = 1., # GLM-130B and Cogview successfully used this, set at 0.1
        attn_z_loss_weight = 1e-4,
    ):
        """
        Initialize the MultiInputTransformerWrapper.

        Args:
            num_tokens (Dict[str, int]): Dictionary mapping input names to their vocabulary sizes.
                Each key represents an input type (e.g., 'token', 'segment'), and each value
                is the number of unique tokens for that input type.
            max_seq_len (int): Maximum sequence length for positional embeddings.
            attn_layers (AttentionLayers): The attention layers to wrap (encoder or decoder).
            emb_dim (int, optional): Embedding dimension. Defaults to the dimension of attn_layers.
            max_mem_len (int): Maximum length of memory to keep for recurrent processing.
                Defaults to 0 (no memory).
            shift_mem_down (int): Number of positions to shift memory down by. Useful for
                specific memory management strategies. Defaults to 0.
            emb_dropout (float): Dropout rate applied to embeddings. Defaults to 0.
            post_emb_norm (bool): Whether to apply layer normalization after embeddings.
                Can help with training stability. Defaults to False.
            num_memory_tokens (int, optional): Number of memory tokens to prepend to the sequence
                (like [CLS] tokens). Defaults to None.
            memory_tokens_interspersed_every (int, optional): If set, memory tokens are
                interspersed every N positions instead of being prepended. Defaults to None.
            return_only_embed (bool): If True, the model will only return embeddings without
                a logits projection head. Defaults to False.
            use_abs_pos_emb (bool): Whether to use absolute positional embeddings.
                Defaults to True.
            scaled_sinu_pos_emb (bool): Whether to use scaled sinusoidal positional embeddings
                instead of learned absolute positional embeddings. Defaults to False.
            emb_frac_gradient (float): Fraction of gradient to pass to embeddings. Values < 1
                reduce the gradient flow to embeddings. Used successfully in GLM-130B and
                CogView (set at 0.1). Defaults to 1.0 (full gradient).
            attn_z_loss_weight (float): Weight for attention z-loss (used for load balancing
                in some attention mechanisms). Defaults to 1e-4.
        """
        super().__init__()

        # Get the model dimension from attention layers and set embedding dimension
        dim = attn_layers.dim
        emb_dim = default(emb_dim, dim)
        self.emb_dim = emb_dim

        # Store sequence length and memory configuration
        self.max_seq_len = max_seq_len
        self.max_mem_len = max_mem_len
        self.shift_mem_down = shift_mem_down

        # Determine if absolute positional embeddings should be disabled
        # This happens when max_seq_len is 0 or when positional embeddings are explicitly disabled
        no_abs_pos_emb = max_seq_len == 0 or not (use_abs_pos_emb and not attn_layers.disable_abs_pos_emb)

        # Initialize positional embeddings based on configuration
        if no_abs_pos_emb:
            # No positional embeddings - always return 0
            self.pos_emb = always(0)
        elif scaled_sinu_pos_emb:
            # Use scaled sinusoidal positional embeddings (learnable scale parameter)
            self.pos_emb = ScaledSinusoidalEmbedding(emb_dim)
        else:
            # Use standard learned absolute positional embeddings
            self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len)

        # Create embedding layers for each input type
        # For example, if num_tokens = {'token': 50000, 'segment': 2}, this creates
        # two embedding tables: 'token_embed' and 'segment_embed'
        self.embeds = ModuleDict({f'{name}_embed': nn.Embedding(one_num_tokens, emb_dim) for name, one_num_tokens in num_tokens.items()})

        # Gradient scaling for embeddings - can help with training stability
        # Reference: https://arxiv.org/abs/2105.13290 (GLM-130B paper)
        self.emb_frac_gradient = emb_frac_gradient

        # Optional post-embedding layer normalization for training stability
        self.post_emb_norm = LayerNorm(emb_dim) if post_emb_norm else nn.Identity()

        # Dropout applied to embeddings before passing to attention layers
        self.emb_dropout = nn.Dropout(emb_dropout)

        # Project embeddings to model dimension if they differ
        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()

        # Store the attention layers module
        self.attn_layers = attn_layers

        # Create output projection heads for each input type
        # Each head projects from model dimension to the vocabulary size for that input type
        if return_only_embed:
            # No logits projection - model only returns embeddings
            self.to_logits = None
        else:
            # Create a separate projection head for each input type
            self.to_logits = ModuleDict({name: nn.Linear(dim, logits_dim, bias = False) for name, logits_dim in num_tokens.items()})

        # Initialize memory tokens (similar to [CLS] tokens in BERT or Memory Transformers)
        # These are learnable tokens that attend to the sequence and can aggregate information
        num_memory_tokens = default(num_memory_tokens, 0)
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            # Create learnable memory token parameters
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

        # Configuration for interspersed memory tokens
        # If set, memory tokens are inserted every N positions instead of being prepended
        self.memory_tokens_interspersed_every = memory_tokens_interspersed_every

        # Determine if cached key-value decoding is possible
        # KV caching is only possible when there are no memory tokens
        self.can_cache_kv = self.num_memory_tokens == 0

        # Can cache KV pairs beyond max_seq_len only if not using absolute positional embeddings
        self.can_cache_kv_outside_max_seq_len = no_abs_pos_emb

    def forward(
        self,
        x: Dict[str, Tensor],
        return_embeddings = False,
        return_logits_and_embeddings = False,
        return_intermediates = False,
        mask = None,
        return_mems = False,
        return_attn = False,
        mems = None,
        mem_masks = None,
        pos = None,
        prepend_embeds = None,
        prepend_mask = None,
        sum_embeds = None,
        return_attn_z_loss = False,
        attn_z_loss_weight = 1e-4,
        seq_start_pos = None,
        cache: LayerIntermediates | None = None,
        **kwargs
    ):
        """
        Forward pass of the MultiInputTransformerWrapper.

        This method processes multiple input types through their respective embeddings,
        combines them, applies positional encodings, processes through attention layers,
        and optionally projects to logits.

        Args:
            x (Dict[str, Tensor]): Dictionary of input tensors, where keys match the names
                in num_tokens from __init__. Each tensor should have shape (batch, seq_len)
                containing token indices.
            return_embeddings (bool): If True, return embeddings instead of logits.
                Defaults to False.
            return_logits_and_embeddings (bool): If True, return both logits and embeddings
                as a tuple. Defaults to False.
            return_intermediates (bool): If True, return intermediate layer outputs and
                attention information. Defaults to False.
            mask (Tensor, optional): Boolean mask of shape (batch, seq_len) where True
                indicates positions to attend to. Defaults to None (attend to all).
            return_mems (bool): If True, return memory states for recurrent processing.
                Defaults to False.
            return_attn (bool): If True, return attention maps. Defaults to False.
            mems (list, optional): List of memory tensors from previous segments for
                recurrent processing. Defaults to None.
            mem_masks (Tensor, optional): Mask for memory tokens. Defaults to None.
            pos (Tensor, optional): Custom positional encodings or position indices.
                If dtype is long, treated as position indices; otherwise as embeddings.
                Defaults to None (use default positional encoding).
            prepend_embeds (Tensor, optional): Embeddings to prepend to the sequence
                (e.g., image embeddings in vision-language models). Shape should be
                (batch, prepend_seq_len, emb_dim). Defaults to None.
            prepend_mask (Tensor, optional): Mask for prepended embeddings. Defaults to None.
            sum_embeds (Tensor, optional): Additional embeddings to sum with token embeddings.
                Useful for self-conditioning in non-autoregressive models. Defaults to None.
            return_attn_z_loss (bool): If True, calculate and return attention z-loss for
                load balancing. Defaults to False.
            attn_z_loss_weight (float): Weight for attention z-loss calculation.
                Defaults to 1e-4.
            seq_start_pos (int, optional): Starting position for positional encoding.
                Useful for continuing generation. Defaults to None (start at 0).
            cache (LayerIntermediates, optional): Cached key-value pairs for efficient
                autoregressive generation. Defaults to None.
            **kwargs: Additional keyword arguments passed to attention layers.

        Returns:
            The return value depends on the flags:
            - Default: Dict[str, Tensor] of logits for each input type
            - return_embeddings=True: Tensor of embeddings (batch, seq_len, dim)
            - return_logits_and_embeddings=True: Tuple of (logits_dict, embeddings)
            - return_intermediates=True: Tuple of (output, intermediates)
            - return_mems=True: Tuple of (output, new_mems) or (output, intermediates) if
              return_intermediates is also True
            - return_attn=True: Tuple of (output, attn_maps)
        """
        # Ensure input dictionary is not empty
        assert not is_empty(x)

        # Get the first input tensor to extract batch size, sequence length, and device
        first_input = list(x.values())[0]

        # Unpack shape and configuration information
        # b: batch size, n: sequence length, device: tensor device
        b, n, device, num_mems, has_memory_tokens, emb_frac_gradient = *first_input.shape, first_input.device, self.num_memory_tokens, self.num_memory_tokens > 0, self.emb_frac_gradient

        # Determine if we need to return hidden states from intermediate layers
        # Required when we need mems, attention maps, or z-loss computation
        return_hiddens = return_mems | return_attn | return_intermediates | return_attn_z_loss

        # Force returning embeddings if no logits projection head exists
        return_embeddings = return_embeddings | (not exists(self.to_logits))

        # === Token Embedding Stage ===
        # Process each input type through its corresponding embedding layer and sum them

        # Verify all inputs have corresponding embedding layers
        assert len(x) == len(self.embeds)

        # Initialize token embeddings as zero (will accumulate all input embeddings)
        token_emb = 0.

        # Sum embeddings from all input types
        for name, embed_id in x.items():
            # Construct the embedding layer key (e.g., 'token_embed', 'segment_embed')
            embed_key = f'{name}_embed'

            # Ensure the embedding layer exists for this input type
            assert embed_key in self.embeds

            # Look up embeddings for the input indices and accumulate
            embed = self.embeds[embed_key](embed_id)
            token_emb = token_emb + embed

        # === Positional Embedding Stage ===
        # Add positional information to token embeddings

        # Check if positional embeddings are provided externally (not as indices)
        external_pos_emb = exists(pos) and pos.dtype != torch.long

        # Get positional embeddings: either compute from position indices or use external embeddings
        pos_emb = self.pos_emb(first_input, pos = pos, seq_start_pos = seq_start_pos) if not external_pos_emb else pos

        # Add positional embeddings to token embeddings
        token_emb = token_emb + pos_emb

        # === Additional Embedding Summation ===
        # Add external embeddings if provided (used for self-conditioning in non-autoregressive training)
        if exists(sum_embeds):
            token_emb = token_emb + sum_embeds

        # Assign combined embeddings to x for further processing
        x = token_emb

        # === Post-Embedding Normalization ===
        # Apply layer normalization if configured (can improve training stability)
        x = self.post_emb_norm(x)

        # === Prepend Embeddings Stage ===
        # Concatenate additional embeddings at the beginning (e.g., image embeddings in PaLI)
        if exists(prepend_embeds):
            # Extract sequence length and dimension of prepended embeddings
            prepend_seq, prepend_dim = prepend_embeds.shape[1:]

            # Verify dimension compatibility
            assert prepend_dim == x.shape[-1], 'prepended embeddings need to have same dimensions as text model dimensions'

            # Concatenate prepended embeddings before the token embeddings
            x = torch.cat((prepend_embeds, x), dim = -2)

            # Update mask to include prepended positions
            if exists(prepend_mask) or exists(mask):
                # Create default mask (all True) if not provided
                mask = default(mask, lambda: torch.ones((b, n), device = device, dtype = torch.bool))
                prepend_mask = default(prepend_mask, lambda: torch.ones((b, prepend_seq), device = device, dtype = torch.bool))

                # Concatenate masks
                mask = torch.cat((prepend_mask, mask), dim = -1)

        # === Gradient Scaling Stage ===
        # Reduce gradient flow to embeddings if configured
        # Technique from CogView paper, validated by GLM-130B
        if emb_frac_gradient < 1:
            assert emb_frac_gradient > 0
            # Scale gradients: fraction flows to embeddings, rest is detached
            x = x * emb_frac_gradient + x.detach() * (1 - emb_frac_gradient)

        # === Embedding Dropout ===
        # Apply dropout to embeddings for regularization
        x = self.emb_dropout(x)

        # === Projection to Model Dimension ===
        # Project embeddings to attention layer dimension if different
        x = self.project_emb(x)

        # === Memory Tokens Stage ===
        # Add learnable memory tokens if configured
        if has_memory_tokens:
            mem_every = self.memory_tokens_interspersed_every

            # Handle interspersed memory tokens (inserted periodically throughout sequence)
            if exists(mem_every):
                assert mem_every > 0
                assert isinstance(self.attn_layers, Decoder), 'only for decoder'

                # Calculate padded sequence length to make it divisible by mem_every
                next_seq_len = math.ceil(n / mem_every) * mem_every

                # Pad sequence to next_seq_len
                x = pad_at_dim(x, (0, next_seq_len - n), dim = -2, value = 0.)

                # Reshape to separate chunks of size mem_every
                x = rearrange(x, 'b (n m) d -> (b n) m d', m = mem_every)

            # Repeat memory tokens for each batch element
            mem = repeat(self.memory_tokens, 'n d -> b n d', b = x.shape[0])

            # Pack memory tokens with sequence (prepend memory tokens to each chunk)
            x, mem_packed_shape = pack((mem, x), 'b * d')

            # Update mask to account for memory tokens
            if not exists(mem_every) and exists(mask):
                # Pad mask with True values for memory token positions
                mask = pad_at_dim(mask, (num_mems, 0), dim = -1, value = True)

            # Reshape back to full sequence if using interspersed memory
            if exists(mem_every):
                x = rearrange(x, '(b n) m d -> b (n m) d', b = b)

        # === Memory Shifting ===
        # Shift recurrent memory positions if configured
        if self.shift_mem_down and exists(mems):
            # Split memories and reorder them (shift down)
            mems_l, mems_r = mems[:self.shift_mem_down], mems[self.shift_mem_down:]
            mems = [*mems_r, *mems_l]

        # === Attention Layers Processing ===
        # Process through transformer attention layers
        x, intermediates = self.attn_layers(x, mask = mask, mems = mems, mem_masks = mem_masks, cache = cache, return_hiddens = True, seq_start_pos = seq_start_pos, **kwargs)

        # === Post-Attention Memory Token Handling ===
        # Extract and remove memory tokens after attention processing
        if has_memory_tokens:
            # Reshape if memory tokens were interspersed
            if exists(mem_every):
                # Separate chunks including memory tokens
                x = rearrange(x, 'b (n m) d -> (b n) m d', m = (mem_every + num_mems))

            # Unpack to separate memory tokens from sequence tokens
            mem, x = unpack(x, mem_packed_shape, 'b * d')

            # Store memory tokens in intermediates for potential use
            intermediates.memory_tokens = mem

            # Reshape back to full sequence if interspersed
            if exists(mem_every):
                x = rearrange(x, '(b n) m d -> b (n m) d', b = b)

            # Trim sequence back to original length (remove any padding)
            x = x[:, :n]

        # === Projection to Logits ===
        # Project final embeddings to vocabulary logits for each input type
        if not return_embeddings:
            # Apply each projection head to get logits for each input type
            logits = {name: fn(x) for name, fn in self.to_logits.items()}

        # === Prepare Output Based on Return Flags ===
        # Format output according to requested return type
        if return_logits_and_embeddings:
            # Return both logits and embeddings
            out = (logits, x)
        elif return_embeddings:
            # Return only embeddings
            out = x
        else:
            # Return only logits (default)
            out = logits

        # === Auxiliary Loss Computation ===
        # Calculate attention z-loss if requested (helps with load balancing)
        if return_attn_z_loss:
            # Extract pre-softmax attention logits from all layers
            pre_softmax_attns = [t.pre_softmax_attn for t in intermediates.attn_intermediates]

            # Compute z-loss and store in intermediates
            intermediates.attn_z_loss = calc_z_loss(pre_softmax_attns, weight = attn_z_loss_weight)

            # Force returning intermediates since we computed z-loss
            return_intermediates = True

        # === Memory State Management ===
        # Prepare memory states for next segment in recurrent processing
        if return_mems:
            # Extract hidden states from all layers
            hiddens = intermediates.hiddens

            # Concatenate with previous memories if they exist
            new_mems = [torch.cat(pair, dim = -2) for pair in zip(mems, hiddens)] if exists(mems) else hiddens

            # Trim to max memory length and detach from computation graph
            new_mems = [t[..., -self.max_mem_len:, :].detach() for t in new_mems]

            # Return output with memories if not returning other intermediates
            if not return_intermediates:
                return out, new_mems

            # Store memories in intermediates
            intermediates.mems = new_mems

        # === Return Outputs ===
        # Return based on requested flags

        # Return output with all intermediate values
        if return_intermediates:
            return out, intermediates

        # Return output with attention maps
        if return_attn:
            # Extract post-softmax attention maps from all layers
            attn_maps = [t.post_softmax_attn for t in intermediates.attn_intermediates]
            return out, attn_maps

        # Return only the output (default case)
        return out
