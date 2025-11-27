from __future__ import annotations

import torch
from torch import nn, cat, stack, arange
from torch.nn import Module
import torch.nn.functional as F
from torch.distributions import Normal

import einx
from einops import rearrange, reduce, pack, repeat, unpack

from x_transformers.autoregressive_wrapper import align_right

from x_transformers.x_transformers import (
    Attention,
    AttentionLayers,
    ScaledSinusoidalEmbedding,
    AbsolutePositionalEmbedding,
    LayerNorm,
    masked_mean,
    always,
    pad_at_dim
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
    Return val if it exists, otherwise return default value d.

    If d is a callable (but not a Module), it will be called to get the default value.
    Otherwise, d is returned as-is.

    Args:
        val: Value to check
        d: Default value or callable that returns a default value

    Returns:
        Either val if it exists, or the default value d
    """
    if exists(val):
        return val
    return d() if not isinstance(d, Module) and callable(d) else d

def sample_from_mean_variance(
    mean,
    variance,
    eps = 1e-5,
    temperature = 1.
):
    """
    Sample from a normal distribution given mean and variance.

    This function generates samples from a Gaussian distribution with the specified
    mean and variance. The variance is clamped to a minimum value to prevent
    numerical instability, and a temperature parameter allows controlling the
    randomness of samples.

    Args:
        mean: Mean of the distribution (tensor)
        variance: Variance of the distribution (tensor)
        eps (float): Minimum variance value to prevent numerical instability. Default: 1e-5
        temperature (float): Temperature scaling factor for the standard deviation.
                           Higher values increase randomness. Default: 1.0

    Returns:
        Tensor: Samples drawn from the normal distribution N(mean, (variance * temperature)^2)
    """
    # Clamp variance to prevent sqrt of negative or very small values
    std = variance.clamp(min = eps).sqrt()
    # Sample from normal distribution with scaled standard deviation
    return torch.normal(mean, std * temperature)

def masked_mean(t, mask):
    """
    Compute the mean of a tensor along masked dimensions.

    This function computes the average of tensor t, but only over positions
    where the mask is True. Positions where mask is False are set to 0 and
    excluded from the mean calculation.

    Args:
        t: Input tensor of shape (batch, seq_len, dim)
        mask: Boolean mask tensor of shape (batch, seq_len)

    Returns:
        Tensor of shape (batch,): Mean values computed only over masked positions
    """
    # Zero out values where mask is False
    t = einx.where('b n, b n d, -> b n d', mask, t, 0.)

    # Sum all values (numerator)
    num = reduce(t, 'b n d -> b', 'sum')
    # Count number of True values in mask (denominator)
    den = mask.sum(dim = -1)

    # Compute masked average, clamping denominator to avoid division by zero
    masked_average = num / den.clamp(min = 1.)
    return masked_average

# probabilistic loss fn

class GaussianNLL(Module):
    """
    Gaussian Negative Log Likelihood loss function.

    This loss function is used for probabilistic predictions where the model
    outputs both a mean and variance for each prediction. It computes the
    negative log likelihood of the target under a Gaussian distribution with
    the predicted mean and variance.
    """

    def forward(self, pred, target):
        """
        Compute the Gaussian NLL loss.

        Args:
            pred: Tuple of (mean, variance) predictions
            target: Ground truth target values

        Returns:
            Tensor: Gaussian negative log likelihood loss (unreduced)
        """
        mean, var = pred
        # Use PyTorch's built-in Gaussian NLL loss without reduction
        return F.gaussian_nll_loss(mean, target, var, reduction = 'none')

# main classes

class ContinuousTransformerWrapper(Module):
    """
    Wrapper for transformer models that process continuous (real-valued) sequences.

    This wrapper handles continuous input sequences (as opposed to discrete tokens),
    providing input/output projection, positional embeddings, memory tokens, and
    optional probabilistic outputs (mean and variance).

    The wrapper is designed for tasks like time series prediction, continuous signal
    processing, or any domain where inputs and outputs are continuous vectors rather
    than discrete tokens.
    """

    def __init__(
        self,
        *,
        max_seq_len,
        attn_layers: AttentionLayers,
        dim_in = None,
        dim_out = None,
        emb_dim = None,
        max_mem_len = 0,
        num_memory_tokens = None,
        post_emb_norm = False,
        emb_dropout = 0.,
        use_abs_pos_emb = True,
        scaled_sinu_pos_emb = False,
        average_pool_embed = False,
        probabilistic = False,
    ):
        """
        Initialize the ContinuousTransformerWrapper.

        Args:
            max_seq_len (int): Maximum sequence length for positional embeddings
            attn_layers (AttentionLayers): The attention layers module to wrap
            dim_in (int, optional): Input dimension. If provided, input will be projected to model dim
            dim_out (int, optional): Output dimension. If provided, output will be projected from model dim
            emb_dim (int, optional): Embedding dimension (deprecated/unused)
            max_mem_len (int): Maximum memory length for storing past hidden states. Default: 0
            num_memory_tokens (int, optional): Number of learnable memory tokens to prepend. Default: None
            post_emb_norm (bool): Whether to apply layer normalization after embeddings. Default: False
            emb_dropout (float): Dropout rate after embeddings. Default: 0.0
            use_abs_pos_emb (bool): Whether to use absolute positional embeddings. Default: True
            scaled_sinu_pos_emb (bool): Whether to use scaled sinusoidal positional embeddings. Default: False
            average_pool_embed (bool): Whether to average pool the output embeddings. Default: False
            probabilistic (bool): Whether to output mean and variance for probabilistic predictions. Default: False
        """
        super().__init__()
        dim = attn_layers.dim

        self.max_seq_len = max_seq_len

        self.max_mem_len = max_mem_len

        # Determine whether to use positional embeddings
        # No positional embeddings if max_seq_len is 0 or if disabled by config
        no_abs_pos_emb = max_seq_len == 0 or not (use_abs_pos_emb and not attn_layers.disable_abs_pos_emb)

        if no_abs_pos_emb:
            # No positional embedding - always returns 0
            self.pos_emb = always(0)
        elif scaled_sinu_pos_emb:
            # Use scaled sinusoidal positional embeddings
            self.pos_emb = ScaledSinusoidalEmbedding(dim)
        else:
            # Use standard absolute positional embeddings
            self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len)

        # Optional layer normalization after embeddings
        self.post_emb_norm = LayerNorm(dim) if post_emb_norm else nn.Identity()
        # Dropout applied after embeddings
        self.emb_dropout = nn.Dropout(emb_dropout)

        # memory tokens

        # Initialize learnable memory tokens if specified
        # Memory tokens are prepended to the sequence and can store global information
        num_memory_tokens = default(num_memory_tokens, 0)
        self.has_memory_tokens = num_memory_tokens > 0

        if num_memory_tokens > 0:
            # Learnable memory token parameters
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

        # attention layers

        # Store the attention layers module
        self.attn_layers = attn_layers

        # average pool

        # Whether to average pool the final embeddings across sequence dimension
        self.average_pool_embed = average_pool_embed

        # project in and out

        # Project input from dim_in to model dimension if needed
        self.project_in = nn.Linear(dim_in, dim, bias = False) if exists(dim_in) else nn.Identity()

        # output is multipled by 2 for outputting mean and log variance

        # Store probabilistic flag
        self.probabilistic = probabilistic

        # Project output from model dimension to dim_out
        # If probabilistic, output dimension is doubled to output both mean and log variance
        self.project_out = nn.Linear(dim, dim_out * (2 if probabilistic else 1), bias = False) if exists(dim_out) else nn.Identity()

        # can cache kv

        # Check if all attention modules support key-value caching
        self.can_cache_kv = all([module.can_cache_kv for module in self.modules() if isinstance(module, Attention)])

    def forward(
        self,
        x,
        return_embeddings = False,
        return_intermediates = False,
        return_mems = False,
        mask = None,
        lens = None,
        return_attn = False,
        mems = None,
        mem_masks = None,
        pos = None,
        sum_embeds = None,
        prepend_embeds = None,
        prepend_mask = None,
        cache: LayerIntermediates | None = None,
        input_not_include_cache = False,
        seq_start_pos = None,
        **kwargs
    ):
        """
        Forward pass through the continuous transformer.

        Args:
            x: Input tensor of shape (batch, seq_len, dim_in)
            return_embeddings (bool): If True, return embeddings before final projection. Default: False
            return_intermediates (bool): If True, return intermediate layer outputs. Default: False
            return_mems (bool): If True, return memory states for next iteration. Default: False
            mask: Boolean mask of shape (batch, seq_len) indicating valid positions. Default: None
            lens: Tensor of sequence lengths of shape (batch,). Alternative to mask. Default: None
            return_attn (bool): If True, return attention maps. Default: False
            mems: Previous memory states from earlier iterations. Default: None
            mem_masks: Masks for memory states. Default: None
            pos: Explicit position indices. Default: None
            sum_embeds: Additional embeddings to add to input embeddings. Default: None
            prepend_embeds: Embeddings to prepend to the sequence (e.g., image embeddings). Default: None
            prepend_mask: Mask for prepended embeddings. Default: None
            cache: Cached key-value pairs from previous forward passes. Default: None
            input_not_include_cache (bool): If True, input doesn't include cached portion. Default: False
            seq_start_pos: Starting position in the full sequence (for position embeddings). Default: None
            **kwargs: Additional arguments passed to attention layers

        Returns:
            Tensor or tuple: Model output, optionally with intermediates, memories, or attention maps
                - If probabilistic=True: returns (mean, variance) tuple
                - If return_intermediates=True: returns (output, intermediates)
                - If return_mems=True: returns (output, new_memories)
                - If return_attn=True: returns (output, attention_maps)
                - Otherwise: returns output tensor
        """
        # Extract batch size, sequence length, original mask, and device
        batch, seq, orig_mask, device = *x.shape[:2], mask, x.device

        # maybe seq lengths passed in

        # Convert sequence lengths to a boolean mask if provided
        # Cannot provide both mask and lens
        if exists(lens):
            assert not exists(mask), 'either `mask` or `lens` passed in, but not both'
            # Create position indices
            seq_arange = arange(seq, device = device)

            # Create mask: True where position < sequence length
            mask = einx.less('j, i -> i j', seq_arange, lens)

        # take care of position embedding offsets in the presence of cache and sequence is less than cache length (not full sequence)

        # Initialize position offset for handling cached sequences
        seq_pos_offset = 0

        # If using cache and input doesn't include cached portion,
        # offset positions by the cache length
        if exists(cache) and input_not_include_cache:
            seq_pos_offset = cache.cache_length

        # project in + positional embedding

        # Project input to model dimension
        x = self.project_in(x)
        # Add positional embeddings with appropriate offsets
        x = x + self.pos_emb(x, pos = pos, seq_start_pos = seq_start_pos, offset = seq_pos_offset)

        # Add any additional embeddings if provided (e.g., for conditioning)
        if exists(sum_embeds):
            x = x + sum_embeds

        # Apply post-embedding normalization
        x = self.post_emb_norm(x)

        # memory tokens

        # Prepend learnable memory tokens if enabled
        if self.has_memory_tokens:
            # Expand memory tokens for batch dimension
            m = repeat(self.memory_tokens, 'm d -> b m d', b = batch)
            # Concatenate memory tokens with input sequence
            x, mem_ps = pack([m, x], 'b * d')

            # Extend mask to include memory tokens (always attend to memory)
            if exists(mask):
                num_mems = m.shape[-2]
                # Pad mask with True values for memory tokens
                mask = pad_at_dim(mask, (num_mems, 0), dim = -1, value = True)

        # whether to append embeds, as in PaLI, for image embeddings

        # Optionally prepend embeddings to sequence (e.g., for multimodal models)
        if exists(prepend_embeds):
            prepend_seq, prepend_dim = prepend_embeds.shape[1:]

            # Verify prepended embeddings have correct dimension
            assert prepend_dim == x.shape[-1], 'prepended embeddings need to have same dimensions as model dimensions'

            # Concatenate prepended embeddings before the main sequence
            x = cat((prepend_embeds, x), dim = -2)

            # Update mask to include prepended embeddings
            if exists(prepend_mask) or exists(mask):
                # Create default masks if not provided
                mask = default(mask, lambda: torch.ones((batch, seq), device = device, dtype = torch.bool))
                prepend_mask = default(prepend_mask, lambda: torch.ones((batch, prepend_seq), device = device, dtype = torch.bool))

                # Concatenate masks
                mask = cat((prepend_mask, mask), dim = -1)

        # Apply dropout to embeddings
        x = self.emb_dropout(x)

        # attention layers

        # Pass through transformer attention layers
        # Returns both output and intermediate layer states
        x, intermediates = self.attn_layers(x, mask = mask, mems = mems, mem_masks = mem_masks, cache = cache, input_not_include_cache = input_not_include_cache, seq_pos_offset = seq_pos_offset, return_hiddens = True, **kwargs)

        # splice out memory tokens

        # Remove memory tokens from output if they were added
        if self.has_memory_tokens:
            # Unpack memory tokens and sequence
            m, x = unpack(x, mem_ps, 'b * d')
            # Store memory tokens in intermediates for potential use
            intermediates.memory_tokens = m

        # Optionally average pool the sequence embeddings
        if self.average_pool_embed:
            # Compute masked mean over sequence dimension
            x = masked_mean(x, mask = orig_mask)

        # maybe linear project out

        # Project to output dimension, or return embeddings if requested
        out = self.project_out(x) if not return_embeddings else x

        # For probabilistic mode, split output into mean and variance
        if not return_embeddings and self.probabilistic:
            # Rearrange output to separate mean and log variance
            mean, log_var = rearrange(out, '... (d mean_log_var) -> mean_log_var ... d', mean_log_var = 2)
            # Convert log variance to variance
            variance = log_var.exp()
            # Stack mean and variance as output
            out = stack((mean, variance))

        # Return with intermediate layer outputs if requested
        if return_intermediates:
            return out, intermediates

        # Return with memory states for next iteration if requested
        if return_mems:
            hiddens = intermediates.hiddens
            # Keep only the last max_mem_len tokens from each layer's hidden states
            new_mems = tuple(t[..., -self.max_mem_len:, :].detach() for t in hiddens)
            return out, new_mems

        # Return with attention maps if requested
        if return_attn:
            # Extract post-softmax attention weights from all layers
            attn_maps = tuple(t.post_softmax_attn for t in intermediates.attn_intermediates)
            return out, attn_maps

        # Default: return just the output
        return out

class ContinuousAutoregressiveWrapper(Module):
    """
    Autoregressive wrapper for continuous sequence modeling.

    This wrapper enables autoregressive training and generation for continuous
    transformers. It handles:
    - Next-step prediction during training
    - Autoregressive sequence generation
    - Multi-step rollout prediction
    - Appropriate loss computation (MSE, L1, or Gaussian NLL)
    - Variable-length sequence handling with masking
    """

    def __init__(
        self,
        net: ContinuousTransformerWrapper,
        loss_fn: Module | None = None,
        use_l1_loss = False,
        equal_loss_weight_batch = False,  # setting this to True, if the mask is passed in and sequences are variable in length, each sequence will be weighted the same (as opposed to each token)
    ):
        """
        Initialize the ContinuousAutoregressiveWrapper.

        Args:
            net (ContinuousTransformerWrapper): The continuous transformer model to wrap
            loss_fn (Module, optional): Custom loss function. If None, uses default based on settings
            use_l1_loss (bool): If True, use L1 loss instead of MSE. Default: False
            equal_loss_weight_batch (bool): If True and mask is provided, weight each sequence
                                           equally regardless of length (instead of weighting
                                           each token equally). Default: False
        """
        super().__init__()
        self.net = net
        self.max_seq_len = net.max_seq_len

        probabilistic = net.probabilistic
        self.probabilistic = probabilistic

        # default loss function

        # Select appropriate loss function based on model configuration
        if not exists(loss_fn):
            if probabilistic:
                # Use Gaussian NLL for probabilistic outputs
                loss_fn = GaussianNLL()
            elif use_l1_loss:
                # Use L1 loss (mean absolute error)
                loss_fn = nn.L1Loss(reduction = 'none')
            else:
                # Use MSE loss (default)
                loss_fn = nn.MSELoss(reduction = 'none')

        self.loss_fn = loss_fn
        self.equal_loss_weight_batch = equal_loss_weight_batch

    @torch.no_grad()
    def generate(
        self,
        start_tokens,
        seq_len,
        temperature = 1.,
        cache_kv = True,
        **kwargs
    ):
        """
        Generate a continuous sequence autoregressively.

        Starting from initial tokens, generates a sequence by iteratively
        predicting the next step and appending it to the sequence. Supports
        key-value caching for efficient generation.

        Args:
            start_tokens: Initial sequence tokens of shape (batch, initial_len, dim) or (initial_len, dim)
            seq_len (int): Number of new tokens to generate
            temperature (float): Temperature for sampling in probabilistic mode.
                               Higher values increase randomness. Default: 1.0
            cache_kv (bool): Whether to cache key-value pairs for efficiency. Default: True
            **kwargs: Additional arguments passed to the model

        Returns:
            Tensor: Generated sequence of shape (batch, seq_len, dim) or (seq_len, dim)
        """
        # Enable caching only if requested and model supports it
        should_cache_kv = cache_kv and self.net.can_cache_kv
        device = start_tokens.device

        # Store training mode to restore later
        was_training = self.net.training
        num_dims = start_tokens.ndim

        # Validate input dimensions
        assert num_dims >= 2, 'number of dimensions of your start tokens must be greater or equal to 2'
        # Check if batch dimension is missing
        no_batch = num_dims == 2

        # Add batch dimension if not present
        if no_batch:
            start_tokens = rearrange(start_tokens, 'n d -> 1 n d')

        b, t, _, device = *start_tokens.shape, start_tokens.device

        # Set model to evaluation mode
        self.net.eval()
        # Initialize output with start tokens
        out = start_tokens

        # Initialize cache for key-value pairs
        cache = None

        # Generate sequence autoregressively
        for _ in range(seq_len):
            # Keep only last max_seq_len tokens for context
            x = out[:, -self.max_seq_len:]

            # Forward pass through model, get intermediates for cache
            net_out, new_cache = self.net(x, cache = cache, return_intermediates = True, **kwargs)

            # Extract prediction for next step (last position)
            last_output = net_out[..., -1:, :]

            # For probabilistic models, sample from predicted distribution
            if self.probabilistic:
                mean, var = last_output
                last_output = sample_from_mean_variance(mean, var, temperature = temperature)

            # Append predicted token to sequence
            out = cat((out, last_output), dim = -2)

            # Update cache if enabled
            if should_cache_kv:
                cache = new_cache

        # Remove initial tokens, keep only generated portion
        out = out[:, t:]

        # Remove batch dimension if input didn't have one
        if no_batch:
            out = rearrange(out, '1 n d -> n d')

        # Restore original training mode
        self.net.train(was_training)
        return out

    def forward_rollout(
        self,
        x,
        rollout_steps = 2,
        **kwargs
    ):
        """
        Forward pass with multi-step rollout prediction.

        Instead of single-step next-token prediction, this method performs multi-step
        rollout where the model predicts multiple future steps, using its own predictions
        as input for subsequent predictions. This is useful for training world models
        and improving long-term prediction accuracy.

        The method randomly samples a starting position in each sequence and performs
        rollout prediction from that point.

        Args:
            x: Input sequence of shape (batch, seq_len, dim)
            rollout_steps (int): Number of steps to rollout. Must be > 1. Default: 2
            **kwargs: Additional arguments passed to the model

        Returns:
            Tensor: Mean loss over the rollout predictions
        """
        # Rollout requires at least 2 steps
        assert rollout_steps > 1

        steps = rollout_steps

        device = x.device

        # assert inputs

        # Prepend embeds not supported in rollout mode
        assert 'prepend_embeds' not in kwargs

        # lens

        # Extract and handle sequence lengths
        lens = kwargs.pop('lens', None)

        # Convert lens to mask if provided
        if exists(lens):
            assert 'mask' not in kwargs, 'either `mask` or `lens` passed in, but not both'
            seq_len, device = inp.shape[1], inp.device
            seq_arange = arange(seq_len, device = device)
            # Create mask based on sequence lengths
            mask = einx.less('j, i -> i j', seq_arange, lens)
            kwargs['mask'] = mask

        # If no lens provided, assume all sequences use full length
        if not exists(lens):
            batch, seq_len = x.shape[:2]
            lens = torch.full((batch,), seq_len, device = device)

        # handle mask manually

        # Extract mask (will be handled separately from kwargs)
        mask = kwargs.pop('mask', None)

        # pick a random range for each batch sample and aligh the sequence to the right for rollout loss

        # Calculate how many positions are valid for starting rollout
        # (sequence length minus rollout steps)
        valid_tokens_for_rollout = (lens - steps).clamp(min = 0)
        valid_sample = valid_tokens_for_rollout > 0

        # Remove sequences that are too short for rollout
        x = x[valid_sample] # remove invalid sequence (lens less than rollout steps)

        if exists(mask):
            mask = mask[valid_sample]

        batch = x.shape[0]
        # Randomly sample starting position for each sequence
        seq_start_pos = (torch.rand((batch,), device = device) * valid_tokens_for_rollout).floor().long()

        batch_arange = torch.arange(batch, device = device)
        batch_arange = rearrange(batch_arange, 'b -> b 1')

        # crop out sequence to use

        # Calculate end position for each sequence
        seq_end_pos = seq_start_pos + steps
        max_end_pos = seq_end_pos.amax().item()
        # Crop sequences to max end position
        x = x[:, :max_end_pos]

        # Align sequences to the right based on their end positions
        x = align_right(x, seq_end_pos)

        # get the input

        # Split into input (all but last 'steps' tokens) and targets (last 'steps' tokens)
        inp, targets = x[:, :-steps], x[:, -steps:]

        # maybe rollout

        # Initialize cache and predictions list
        cache = None
        preds = []

        # Perform rollout for the specified number of steps
        for _ in range(steps):

            # Forward pass through model
            out, cache = self.net(
                inp,
                seq_start_pos = seq_start_pos,
                return_intermediates = True,
                **kwargs
            )

            # Get prediction for last position
            last_pred = out[..., -1:, :]

            # For next iteration, use either sampled prediction (probabilistic)
            # or direct prediction (deterministic)
            if self.probabilistic:
                mean, var = last_pred
                # Sample from predicted distribution
                inp = sample_from_mean_variance(mean, var)
            else:
                # Use prediction directly
                inp = last_pred

            # Store prediction for loss calculation
            preds.append(last_pred)

        # stack for predictions

        # Concatenate all predictions along sequence dimension
        preds = cat(preds, dim = 1)

        # loss

        # Compute loss between predictions and targets
        loss = self.loss_fn(preds, targets)

        # Return mean loss
        return loss.mean()

    def forward(
        self,
        x,
        rollout_steps = 1, # they used 2 rollout steps in a successful world model paper https://ai.meta.com/vjepa/
        **kwargs
    ):
        """
        Forward pass for training with next-step prediction or multi-step rollout.

        Performs autoregressive training by predicting the next token at each position.
        If rollout_steps > 1, uses the forward_rollout method instead for multi-step
        prediction training.

        Args:
            x: Input sequence of shape (batch, seq_len, dim)
            rollout_steps (int): Number of rollout steps. If > 1, uses multi-step rollout.
                               If = 1, uses standard next-step prediction. Default: 1
            **kwargs: Additional arguments passed to the model

        Returns:
            Tensor: Mean loss value
        """
        # Use rollout training if rollout_steps > 1
        if rollout_steps > 1:
            return self.forward_rollout(x, rollout_steps = rollout_steps, **kwargs)

        # Standard autoregressive training: predict next token
        # Input is all tokens except the last, target is all tokens except the first
        inp, target = x[:, :-1], x[:, 1:]

        # Prepend embeds not supported in standard forward
        assert 'prepend_embeds' not in kwargs

        # lens

        # Handle sequence lengths if provided
        lens = kwargs.pop('lens', None)

        # Convert lens to mask
        if exists(lens):
            assert 'mask' not in kwargs, 'either `mask` or `lens` passed in, but not both'
            seq_len, device = inp.shape[1], inp.device
            seq_arange = torch.arange(seq_len, device = device)
            # Create mask based on sequence lengths
            mask = einx.less('j, i -> i j', seq_arange, lens)

            kwargs['mask'] = mask

        # mask

        # Adjust mask to match input shape (if it matches original x shape)
        mask = kwargs.get('mask', None)

        if exists(mask) and mask.shape[1] == x.shape[1]:
            # Remove last position from mask to match input length
            mask = mask[:, :-1]
            kwargs['mask'] = mask

        # Forward pass through network
        out = self.net(inp, **kwargs)

        # Compute loss between predictions and targets
        loss = self.loss_fn(out, target)

        # Handle masked loss computation
        if exists(mask):
            # Loss should not be pre-reduced if mask is used
            assert loss.ndim > 1, 'loss should not be reduced if mask is passed in'

            if self.equal_loss_weight_batch:
                # Weight each sequence equally (compute mean per sequence, then average)
                loss = masked_mean(loss, mask)
            else:
                # Weight each token equally (only keep losses for valid positions)
                loss = loss[mask]

        # Return mean loss
        return loss.mean()
