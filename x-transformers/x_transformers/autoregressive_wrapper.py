"""
Autoregressive Wrapper Module

This module provides an autoregressive wrapper for transformer models, enabling
auto-regressive text generation with various sampling strategies including beam search,
nucleus sampling, top-k sampling, and contrastive decoding.

The wrapper handles:
- Training with cross-entropy loss and optional masking
- Inference with multiple sampling strategies
- Key-value caching for efficient generation
- Variable-length prompts and sequences
- Beam search with optional stochasticity
- Contrastive decoding between expert and amateur models
"""

from __future__ import annotations

from math import ceil, log
from typing import Tuple, Callable

import torch
from torch import nn, tensor, Tensor
from torch.nn import Module
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from einops import rearrange, repeat, pack, unpack

# Utility helper functions

def exists(val):
    """
    Check if a value exists (is not None).

    Args:
        val: The value to check

    Returns:
        bool: True if val is not None, False otherwise
    """
    return val is not None

def default(val, d):
    """
    Return the value if it exists, otherwise return a default.

    Args:
        val: The value to check
        d: The default value to return if val is None

    Returns:
        The value if it exists, otherwise the default
    """
    return val if exists(val) else d

def identity(t, *args, **kwargs):
    """
    Identity function that returns its input unchanged.
    Used as a no-op filter function.

    Args:
        t: The input to return
        *args: Ignored positional arguments
        **kwargs: Ignored keyword arguments

    Returns:
        The input t unchanged
    """
    return t

def join(arr, delimiter = ', '):
    """
    Join an array of strings with a delimiter.

    Args:
        arr: Array of strings to join
        delimiter: String to use as delimiter (default: ', ')

    Returns:
        str: Joined string
    """
    return delimiter.join(arr)

def cast_tuple(t, length = 1):
    """
    Cast a value to a tuple of specified length.
    If already a tuple, return as-is. Otherwise, repeat the value.

    Args:
        t: Value to cast to tuple
        length: Desired tuple length (default: 1)

    Returns:
        tuple: The input as a tuple of specified length
    """
    return t if isinstance(t, tuple) else (t,) * length

def eval_decorator(fn):
    """
    Decorator that temporarily sets a model to evaluation mode for the duration
    of a function call, then restores the original training state.

    Args:
        fn: Function to wrap

    Returns:
        Wrapped function that executes in eval mode
    """
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out
    return inner

# Gumbel sampling functions
# Used for stochastic sampling during generation

def log(t, eps = 1e-20):
    """
    Safe logarithm function that clamps values to avoid log(0).

    Args:
        t: Input tensor
        eps: Minimum value to clamp to (default: 1e-20)

    Returns:
        Tensor: Logarithm of the clamped input
    """
    return t.clamp(min = eps).log()

def gumbel_noise(t):
    """
    Generate Gumbel noise for stochastic sampling.
    Uses the Gumbel distribution: -log(-log(U)) where U ~ Uniform(0,1)

    Args:
        t: Input tensor (used only for shape and device)

    Returns:
        Tensor: Gumbel noise with the same shape as input
    """
    return -log(-log(torch.rand_like(t)))

def gumbel_sample(logits, temperature = 1., eps = 1e-6):
    """
    Sample from logits using the Gumbel-Max trick.
    This adds Gumbel noise to logits and takes the argmax.

    Args:
        logits: Input logits tensor
        temperature: Sampling temperature (default: 1.0)
        eps: Small value to prevent division by zero (default: 1e-6)

    Returns:
        Tensor: Sampled indices
    """
    noise = gumbel_noise(logits)
    return ((logits / max(temperature, eps)) + noise).argmax(dim = -1)

# Cache manipulation functions
# Used for efficient key-value caching during generation

def modify_cached_kv(cache, fn):
    """
    Apply a function to all cached key-value pairs in attention layers.
    This is useful for operations like slicing or repeating cached states.

    Args:
        cache: Cache object containing attention intermediates
        fn: Function to apply to each cached key-value tensor
    """
    for inter in cache.attn_intermediates:
        # Only modify attention layers (layer_type == 'a')
        if inter.layer_type == 'a':
            inter.cached_kv = [fn(t) for t in inter.cached_kv]

# Functions for handling variable-length prefixes
# These allow processing prompts of different lengths in a batch

def pad_at_dim(t, pad: tuple[int, int], dim = -1, value = 0.):
    """
    Pad a tensor at a specific dimension.

    Args:
        t: Input tensor to pad
        pad: Tuple of (left_pad, right_pad) amounts
        dim: Dimension to pad along (default: -1 for last dimension)
        value: Padding value (default: 0.0)

    Returns:
        Tensor: Padded tensor
    """
    if pad == (0, 0):
        return t

    # Calculate how many dimensions are to the right of the target dimension
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    # Create zero padding for all dimensions to the right
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def align_right(t, lens, pad_id = 0):
    """
    Right-align sequences of variable lengths within a batch.
    This is useful for handling prompts of different lengths.

    For example, if we have sequences:
        [1, 2, 3]       (len=3)
        [4, 5, 6, 7, 8] (len=5)

    They will be aligned to:
        [0, 0, 1, 2, 3]
        [4, 5, 6, 7, 8]

    Args:
        t: Input tensor of shape (batch, seq_len, ...)
        lens: Tensor of actual lengths for each sequence in batch
        pad_id: Value to use for padding (default: 0)

    Returns:
        Tensor: Right-aligned sequences
    """
    batch, seq_len, device, dtype = *t.shape[:2], t.device, t.dtype

    assert lens.ndim == 1 and lens.shape[0] == batch
    assert lens.amax() <= seq_len

    # Calculate how much padding each sequence needs
    pad_lens = seq_len - lens
    max_pad_len = pad_lens.amax()

    # Create index tensors for gathering
    batch_arange = torch.arange(batch, device = device, dtype = torch.long)[..., None]
    prompt_len_arange = torch.arange(seq_len, device = device, dtype = torch.long)

    # Add padding to the left to accommodate the longest padding needed
    t = pad_at_dim(t, (max_pad_len, 0), value = pad_id, dim = 1)

    # Calculate offset for each sequence to right-align them
    offset = max_pad_len - pad_lens

    # Gather the right-aligned sequences
    aligned = t[batch_arange, prompt_len_arange + offset[..., None], ...]
    return aligned

# Logit filtering functions for sampling
# These functions filter the probability distribution over tokens

def top_p(logits, thres = 0.9):
    """
    Nucleus (top-p) sampling: sample from the smallest set of tokens whose
    cumulative probability exceeds the threshold p.

    This method filters out low-probability tokens by keeping only the top tokens
    whose cumulative probability mass is above a threshold.

    Args:
        logits: Logits tensor of shape (batch, vocab_size)
        thres: Cumulative probability threshold (default: 0.9)

    Returns:
        Tensor: Filtered logits with low-probability tokens set to -inf
    """
    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending = True)
    # Calculate cumulative probabilities
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim = -1), dim = -1)

    # Mark tokens to remove (those beyond the cumulative threshold)
    sorted_indices_to_remove = cum_probs > thres
    # Shift right to keep at least one token (the highest probability one)
    sorted_indices_to_remove = F.pad(sorted_indices_to_remove, (1, -1), value = False)

    # Set removed tokens to -inf
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    # Scatter back to original order
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

# Top-k sampling

def top_k(logits, frac_num_tokens = 0.1, k = None):
    """
    Top-k sampling: sample only from the k tokens with highest probability.

    Args:
        logits: Logits tensor of shape (batch, vocab_size)
        frac_num_tokens: Fraction of vocabulary to keep if k is not specified (default: 0.1)
        k: Number of top tokens to keep (default: None, computed from frac_num_tokens)

    Returns:
        Tensor: Filtered logits with all but top-k tokens set to -inf
    """
    num_tokens = logits.shape[-1]

    # Determine k: use provided value or calculate from fraction
    k = default(k, ceil(frac_num_tokens * num_tokens))
    k = min(k, num_tokens)

    # Get top k values and indices
    val, ind = torch.topk(logits, k)
    # Create output tensor with all -inf
    probs = torch.full_like(logits, float('-inf'))
    # Scatter top-k values back
    probs.scatter_(1, ind, val)
    return probs

# Top-a sampling

def top_a(logits, min_p_pow = 2.0, min_p_ratio = 0.02):
    """
    Top-a sampling: dynamically filter tokens based on the maximum probability.
    Filters out tokens with probability less than (max_prob^min_p_pow) * min_p_ratio.

    This adaptive method adjusts the filtering threshold based on the confidence
    of the model (maximum probability).

    Args:
        logits: Logits tensor of shape (batch, vocab_size)
        min_p_pow: Power to raise max probability to (default: 2.0)
        min_p_ratio: Ratio to multiply with powered max prob (default: 0.02)

    Returns:
        Tensor: Filtered logits with low-probability tokens set to -inf
    """
    probs = logits.softmax(dim = -1)
    max_probs = probs.amax(dim = -1, keepdim = True)
    # Calculate adaptive threshold
    limit = torch.pow(max_probs, min_p_pow) * min_p_ratio
    return torch.where(probs < limit, float('-inf'), logits)

# Min-p sampling
# Reference: https://arxiv.org/abs/2407.01082

def min_p(logits, min_p = 0.1):
    """
    Min-p sampling: filter out tokens with probability less than min_p * max_probability.

    This method sets a minimum probability threshold relative to the maximum probability,
    filtering out unlikely tokens while adapting to the model's confidence.

    Args:
        logits: Logits tensor of shape (batch, vocab_size)
        min_p: Minimum probability ratio relative to max (default: 0.1)

    Returns:
        Tensor: Filtered logits with low-probability tokens set to -inf

    Reference:
        https://arxiv.org/abs/2407.01082
    """
    probs = logits.softmax(dim = -1)
    max_probs = probs.amax(dim = -1, keepdim = True)
    # Calculate threshold as fraction of max probability
    limit = min_p * max_probs
    return torch.where(probs < limit, float('-inf'), logits)

# Dictionary mapping filter names to their functions
# This allows string-based selection of filtering methods

FILTER_LOGITS_FN = dict(
    top_p = top_p,
    top_k = top_k,
    top_a = top_a,
    min_p = min_p
)

# Contrastive decoding function
# Used to improve generation quality by contrasting expert and amateur models

def contrastive_decode_fn(
    expert_logits,
    amateur_logits,
    alpha = 0.1,
    beta = 0.5
):
    """
    Contrastive decoding: amplify the difference between expert and amateur model logits.

    This method improves generation by emphasizing tokens where the expert model has
    higher confidence than the amateur model, effectively filtering out tokens that
    both models find likely (which may be generic or low-quality).

    The formula is:
        CD(x) = (1 + beta) * expert(x) - beta * amateur(x)
    with a cutoff applied to avoid very low probability tokens.

    Args:
        expert_logits: Logits from the expert (better) model
        amateur_logits: Logits from the amateur (weaker) model
        alpha: Cutoff threshold parameter (default: 0.1)
        beta: Contrastive weight parameter (default: 0.5)

    Returns:
        Tensor: Contrastively decoded logits

    Reference:
        Appendix A Algorithm 2 from https://arxiv.org/abs/2309.09117
    """
    # Calculate cutoff threshold for expert logits
    cutoff = log(alpha) + expert_logits.amax(dim = -1, keepdim = True)

    # Apply contrastive decoding formula
    diffs = (1 + beta) * expert_logits - beta * amateur_logits

    # Mask out logits below the cutoff threshold
    contrastive_decode_logits = diffs.masked_fill(expert_logits < cutoff, -torch.finfo(expert_logits.dtype).max)
    return contrastive_decode_logits

# Main autoregressive wrapper class

class AutoregressiveWrapper(Module):
    """
    Autoregressive wrapper for transformer models.

    This wrapper enables autoregressive language modeling with:
    - Training: Cross-entropy loss with optional masking and auxiliary losses
    - Generation: Multiple sampling strategies (greedy, top-k, top-p, beam search)
    - Key-value caching for efficient generation
    - Variable-length prompt handling
    - Contrastive decoding support

    The wrapper handles all the complexities of autoregressive generation including
    maintaining KV caches, handling EOS tokens, and applying various sampling strategies.
    """

    def __init__(
        self,
        net,
        ignore_index = -100,
        pad_value = 0,
        mask_prob = 0.,
        add_attn_z_loss = False,
        next_embed_loss_weight = 0.1
    ):
        """
        Initialize the AutoregressiveWrapper.

        Args:
            net: The underlying transformer network to wrap
            ignore_index: Index to ignore in loss calculation (default: -100)
            pad_value: Value used for padding sequences (default: 0)
            mask_prob: Probability of masking tokens during training for MLM-style
                      training alongside autoregressive training (default: 0.0)
                      Paper: https://arxiv.org/abs/2210.13432
            add_attn_z_loss: Whether to add attention z-loss for stability (default: False)
            next_embed_loss_weight: Weight for continuous embedding prediction loss (default: 0.1)
        """
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = net
        self.max_seq_len = net.max_seq_len

        # Masking probability for MLM-style training
        # Paper shows masking (MLM) in conjunction with autoregressive decoder-only
        # training leads to big improvements: https://arxiv.org/abs/2210.13432
        assert mask_prob < 1.
        self.mask_prob = mask_prob

        # Whether to add attention router z-loss for training stability
        self.add_attn_z_loss = add_attn_z_loss

        # Whether to add a continuous embedding prediction loss
        self.add_continuous_pred_head = net.add_continuous_pred_head
        self.next_embed_loss_weight = next_embed_loss_weight

    @torch.no_grad()
    @eval_decorator
    def beam_search(
        self,
        prompts,
        seq_len,
        beams = 4,
        return_beams_and_scores = False,
        eos_token = None,
        temperature = 1.,
        stochastic = False,
        prompt_lens: Tensor | None = None,
        filter_logits_fn: str | Callable = identity,
        restrict_to_max_seq_len = True,
        filter_kwargs: dict = dict(),
        cache_kv = True,
        **kwargs
    ):
        """
        Generate text using beam search.

        Beam search maintains multiple hypothesis sequences (beams) in parallel,
        keeping the top-scoring sequences at each step. This typically produces
        higher quality outputs than greedy decoding.

        Args:
            prompts: Input prompt tokens of shape (batch, seq_len) or (..., seq_len)
            seq_len: Number of tokens to generate
            beams: Number of beams to maintain (default: 4)
            return_beams_and_scores: If True, return all beams with their scores
                                    instead of just the top beam (default: False)
            eos_token: End-of-sequence token ID. Currently not fully supported (default: None)
            temperature: Sampling temperature. Use 0.0 for greedy (default: 1.0)
            stochastic: Whether to use stochastic beam search with Gumbel noise (default: False)
            prompt_lens: Actual lengths of prompts if they have different lengths (default: None)
            filter_logits_fn: Function or string name to filter logits ('top_k', 'top_p', etc.)
                            Only used if stochastic=True (default: identity)
            restrict_to_max_seq_len: Whether to restrict sequence length to model's max (default: True)
            filter_kwargs: Additional kwargs for the filter function (default: {})
            cache_kv: Whether to use key-value caching for efficiency (default: True)
            **kwargs: Additional arguments passed to the model

        Returns:
            Tensor: Generated sequences. Shape depends on return_beams_and_scores:
                   - If False: (batch, seq_len) - only the top beam
                   - If True: tuple of (beams, scores) where:
                     * beams: (beams, batch, seq_len)
                     * scores: (beams, batch)
        """
        assert not exists(eos_token), 'eos token not supported yet'

        max_seq_len, greedy, device = self.max_seq_len, temperature == 0., prompts.device

        # Pack prompts to ensure batch dimension (handles inputs with or without batch dim)
        prompts, packed_shape = pack([prompts], '* n')

        batch, orig_seq_len = prompts.shape

        # Handle filter logits function given as string name
        # Convert string to actual function from the FILTER_LOGITS_FN dictionary
        if isinstance(filter_logits_fn, str):
            assert filter_logits_fn in FILTER_LOGITS_FN, f"only {join(FILTER_LOGITS_FN.keys())} are available"

            filter_logits_fn = FILTER_LOGITS_FN[filter_logits_fn]

        # Handle variable length prompts (prefixes)
        # Right-align prompts so padding is on the left
        seq_start_pos = None
        if exists(prompt_lens):
            prompts = align_right(prompts, prompt_lens, pad_id = self.pad_value)
            seq_start_pos = orig_seq_len - prompt_lens

        # Initialize output with the prompts
        # Sampled tokens will be appended to this
        out = prompts

        # Initialize key-value cache for efficient generation
        cache = None

        should_cache = cache_kv and self.net.can_cache_kv

        # Initialize scores for ranking beams
        # Scores accumulate log probabilities for each beam
        scores = torch.zeros((batch,), device = device)

        batch_arange = torch.arange(batch, device = device)

        # Main generation loop - generate seq_len tokens
        for i in range(seq_len):
            is_first = i == 0

            # Restrict sequence length to model's maximum if needed
            if restrict_to_max_seq_len:
                max_len_exceeded = out.shape[-1] > max_seq_len

                assert not (cache_kv and max_len_exceeded and not self.net.can_cache_kv_outside_max_seq_len), 'the network cannot use cached key values when decoding outside the max sequence length. most likely because you are using absolute positional embedding. you can switch to rotary embeddings to resolve this issue'

                # Only use the last max_seq_len tokens
                x = out[:, -max_seq_len:]

                # Truncate cache to match
                if exists(cache):
                    modify_cached_kv(cache, lambda t: t[..., -(max_seq_len - 1):, :])

            # Forward pass through the model
            logits, new_cache = self.net(
                x,
                return_intermediates = True,
                cache = cache,
                seq_start_pos = seq_start_pos,
                **kwargs
            )

            # Update cache if caching is enabled
            if should_cache:
                cache = new_cache

            # Get logits for the last position only
            logits = logits[:, -1]

            # Calculate log probabilities for scoring beams
            log_probs = logits.log_softmax(dim = -1)

            # Apply filtering for stochastic beam search
            # (e.g., top-k or nucleus sampling)
            if stochastic and not greedy:
                logits = filter_logits_fn(logits, **filter_kwargs)
                logits = (logits / temperature) + gumbel_noise(logits)

            # Sample top-k tokens to create candidate beams
            # In stochastic mode, Gumbel noise makes this probabilistic
            samples = logits.topk(beams, dim = -1).indices

            # Get log probabilities of the sampled tokens
            next_scores = log_probs.gather(-1, samples)

            # Update beam scores by adding new token log probabilities
            scores = repeat(scores, 'b -> b beams', beams = beams)
            scores = scores + next_scores

            # Expand sequences to accommodate new beams
            out = repeat(out, 'b ... -> (b beams) ...', beams = beams)
            samples = rearrange(samples, 'b beams -> (b beams) 1')

            # Expand cache on first iteration to match beam dimension
            if should_cache and is_first:
                modify_cached_kv(cache, lambda t: repeat(t, 'b ... -> (b beams) ...', beams = beams))

            # Append sampled tokens to sequences
            out = torch.cat((out, samples), dim=-1)

            # Prune beams: keep only the top-scoring beams
            # Flatten all beam candidates and select top beams
            scores = rearrange(scores, '(b prev_beams) next_beams -> b (prev_beams next_beams)', b = batch)
            curr_num_beams = scores.shape[-1]

            if curr_num_beams > beams:
                # Sort by score and keep top beams
                scores, sort_indices = scores.sort(dim = -1, descending = True)

                scores = scores[:, :beams]
                top_beams_indices = sort_indices[:, :beams]

                # Calculate flattened indices for gathering sequences
                top_beams_indices = curr_num_beams * batch_arange[:, None] + top_beams_indices

                flattened_beam_indices = rearrange(top_beams_indices, 'b beams -> (b beams)')

                # Keep only top beam sequences
                out = out[flattened_beam_indices]

            scores = rearrange(scores, 'b beams -> (b beams)')

            # Check for end-of-sequence token if provided
            if not exists(eos_token):
                continue

            is_eos_tokens = (out == eos_token)

            # Stop if all sequences have generated EOS token
            if is_eos_tokens.any(dim = -1).all():
                break

        # Post-process: mask out everything after the first EOS token in each sequence
        if exists(eos_token):
            # Shift EOS mask right so we keep the EOS token itself
            shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
            # Create cumulative mask: everything after first EOS is masked
            mask = shifted_is_eos_tokens.float().cumsum(dim = -1) >= 1
            out = out.masked_fill(mask, self.pad_value)

        # Reshape to separate batch and beam dimensions
        out = rearrange(out, '(b beams) seq -> b beams seq', b = batch)

        # Remove the original prompt, keep only generated tokens
        out = out[..., orig_seq_len:]

        # Unpack to restore original shape (may not have had batch dimension)
        out, = unpack(out, packed_shape, '* beams n')

        # Return only top beam if not requesting all beams
        if not return_beams_and_scores:
            return out[..., 0, :]

        # Return all beams with their scores
        scores = rearrange(scores, '(b beams) -> beams b', b = batch)
        out = rearrange(out, 'b beams n -> beams b n')

        return out, scores

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        prompts: list[Tensor] | Tensor,
        seq_len,
        eos_token = None,
        temperature = 1.,
        prompt_lens: Tensor | None = None,
        filter_logits_fn: str | Callable = top_k,
        restrict_to_max_seq_len = True,
        amateur_model: Module | Tuple[Module] | None = None,
        filter_kwargs: dict = dict(),
        contrastive_decode_kwargs: dict | Tuple[dict] = dict(
            beta = 0.5,
            alpha = 0.1
        ),
        cache_kv = True,
        **kwargs
    ):
        """
        Generate text autoregressively with various sampling strategies.

        This is the main generation method supporting:
        - Greedy decoding (temperature=0)
        - Sampling with temperature
        - Top-k, top-p (nucleus), top-a, and min-p filtering
        - Contrastive decoding with amateur models
        - Variable-length prompts
        - Key-value caching for efficiency

        Args:
            prompts: Input prompts. Can be:
                    - Tensor of shape (batch, seq_len)
                    - List of Tensors with variable lengths
            seq_len: Number of tokens to generate
            eos_token: End-of-sequence token ID. Generation stops when all
                      sequences generate this token (default: None)
            temperature: Sampling temperature. Use 0.0 for greedy decoding,
                        >1.0 for more random, <1.0 for more focused (default: 1.0)
            prompt_lens: Actual lengths of prompts if variable. Auto-computed
                        if prompts is a list (default: None)
            filter_logits_fn: Function or string to filter logits before sampling.
                             Options: 'top_k', 'top_p', 'top_a', 'min_p' or callable
                             (default: top_k)
            restrict_to_max_seq_len: Whether to restrict to model's max sequence length
                                    (default: True)
            amateur_model: Model(s) for contrastive decoding. Can be single model
                          or tuple of models (default: None)
            filter_kwargs: Keyword arguments for the filter function (default: {})
            contrastive_decode_kwargs: Parameters for contrastive decoding (alpha, beta)
                                      Single dict or tuple matching amateur_model (default: {beta: 0.5, alpha: 0.1})
            cache_kv: Whether to use key-value caching for efficiency (default: True)
            **kwargs: Additional arguments passed to the model

        Returns:
            Tensor: Generated token sequences of shape (batch, seq_len)
                   excluding the original prompts
        """
        max_seq_len, greedy = self.max_seq_len, temperature == 0.

        # Handle prompts given as list of variable-length token sequences
        # Convert to padded tensor and compute lengths
        if isinstance(prompts, list):
            assert len(prompts) > 0, 'prompts cannot be empty list'
            assert not exists(prompt_lens), '`prompt_len` will be auto derived if prompts are passed in as list of Tensors'

            # Compute actual length of each prompt
            prompt_lens = tensor([t.shape[0] for t in prompts], device = prompts[0].device)

            # Pad sequences to same length (left-padded by default)
            prompts = pad_sequence(prompts, batch_first = True)

        # Pack prompts to ensure batch dimension (handles inputs with or without batch dim)
        prompts, ps = pack([prompts], '* n')

        b, t, device = *prompts.shape, prompts.device

        # Handle filter logits function given as string name
        # Convert string to actual function from the FILTER_LOGITS_FN dictionary
        if isinstance(filter_logits_fn, str):
            assert filter_logits_fn in FILTER_LOGITS_FN, f"only {join(FILTER_LOGITS_FN.keys())} are available"

            filter_logits_fn = FILTER_LOGITS_FN[filter_logits_fn]

        # Handle variable length prompts (prefixes)
        # Right-align prompts so padding is on the left
        seq_start_pos = None
        if exists(prompt_lens):
            prompts = align_right(prompts, prompt_lens, pad_id = self.pad_value)
            seq_start_pos = t - prompt_lens

        # Initialize output with the prompts
        # Sampled tokens will be appended to this
        out = prompts

        # Initialize key-value cache for efficient generation
        cache = None

        # Setup contrastive decoding if amateur model(s) provided
        # Contrastive decoding compares expert (this model) with amateur model(s)
        if exists(amateur_model):
            amateur_model = cast_tuple(amateur_model)
            contrastive_decode_kwargs = cast_tuple(contrastive_decode_kwargs)

            assert len(amateur_model) == len(contrastive_decode_kwargs)

            # Initialize caches for each amateur model
            amateur_caches = [None] * len(amateur_model)
            # Disable filtering when using contrastive decoding
            filter_logits_fn = identity

            # Unwrap amateur models if they're also AutoregressiveWrapper instances
            for i, module in enumerate(amateur_model):
                if isinstance(module, AutoregressiveWrapper):
                    amateur_model[i] = module.net

                module.eval()

        # Main generation loop - generate seq_len tokens
        for _ in range(seq_len):

            # Restrict sequence length to model's maximum if needed
            if restrict_to_max_seq_len:
                max_len_exceeded = out.shape[-1] > max_seq_len

                assert not (cache_kv and max_len_exceeded and not self.net.can_cache_kv_outside_max_seq_len), 'the network cannot use cached key values when decoding outside the max sequence length. most likely because you are using absolute positional embedding. you can switch to rotary embeddings to resolve this issue'

                # Only use the last max_seq_len tokens
                x = out[:, -max_seq_len:]

                # Truncate cache to match
                if exists(cache):
                    for inter in cache.attn_intermediates:
                        if inter.layer_type == 'a':
                            inter.cached_kv = [t[..., -(max_seq_len - 1):, :] for t in inter.cached_kv]

            # Forward pass through the expert model
            logits, new_cache = self.net(
                x,
                return_intermediates = True,
                cache = cache,
                seq_start_pos = seq_start_pos,
                **kwargs
            )

            # Update cache if caching is enabled
            if cache_kv and self.net.can_cache_kv:
                cache = new_cache

            # Get logits for the last position only
            logits = logits[:, -1]

            # Handle contrastive decoding using amateur model(s)
            # Reference: https://arxiv.org/abs/2210.15097
            if exists(amateur_model):
                # Process each amateur model and apply contrastive decoding
                for i, (amateur, amateur_cache, amateur_contrastive_decode_kwargs) in enumerate(zip(amateur_model, amateur_caches, contrastive_decode_kwargs)):
                    # Get logits from amateur model
                    amateur_logits, next_amateur_cache = amateur(
                        x,
                        return_intermediates = True,
                        cache = amateur_cache,
                        seq_start_pos = seq_start_pos,
                        **kwargs
                    )

                    amateur_logits = amateur_logits[:, -1]

                    # Ensure amateur and expert have same vocabulary
                    assert amateur_logits.shape == logits.shape, 'logits dimension are not the same between amateur and expert model'

                    # Apply contrastive decoding formula
                    logits = contrastive_decode_fn(logits, amateur_logits, **amateur_contrastive_decode_kwargs)

                    # Update amateur cache if caching is enabled
                    if cache_kv and amateur.can_cache_kv:
                        amateur_caches[i] = next_amateur_cache

            # Sample next token using specified strategy
            # Either greedy (argmax) or filtered sampling
            if greedy:
                sample = logits.argmax(dim = -1, keepdim = True)
            else:
                # Apply filtering (top_k, top_p, etc.) then sample
                filtered_logits = filter_logits_fn(logits, **filter_kwargs)
                probs = F.softmax(filtered_logits / temperature, dim=-1)
                sample = torch.multinomial(probs, 1)

            # Append sampled token to sequence
            out = torch.cat((out, sample), dim=-1)

            # Check for end-of-sequence token if provided
            if not exists(eos_token):
                continue

            is_eos_tokens = (out == eos_token)

            # Stop if all sequences have generated EOS token
            if is_eos_tokens.any(dim = -1).all():
                break

        # Post-process: mask out everything after the first EOS token in each sequence
        if exists(eos_token):
            # Shift EOS mask right so we keep the EOS token itself
            shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
            # Create cumulative mask: everything after first EOS is masked
            mask = shifted_is_eos_tokens.float().cumsum(dim = -1) >= 1
            out = out.masked_fill(mask, self.pad_value)

        # Remove the original prompt, keep only generated tokens
        out = out[:, t:]

        # Unpack to restore original shape (may not have had batch dimension)
        out, = unpack(out, ps, '* n')

        return out

    def forward(
        self,
        x,
        return_outputs = False,
        prepend_embeds = None,
        **kwargs
    ):
        """
        Forward pass for training with autoregressive language modeling.

        Computes the cross-entropy loss for predicting next tokens, with optional
        auxiliary losses:
        - Attention z-loss for training stability
        - Continuous embedding prediction loss
        - Optional masking (MLM-style) alongside autoregressive training

        Args:
            x: Input token IDs of shape (batch, seq_len)
            return_outputs: If True, return both loss and outputs (logits, cache)
                          If False, return only loss (default: False)
            prepend_embeds: Optional embeddings to prepend to the sequence
                          (e.g., for prefix tuning) (default: None)
            **kwargs: Additional arguments passed to the model (e.g., self_attn_kv_mask)

        Returns:
            If return_outputs=False:
                Tensor: Scalar loss value
            If return_outputs=True:
                Tuple[Tensor, Tuple]: (loss, (logits, cache))
                - loss: Scalar loss value
                - logits: Predicted logits of shape (batch, seq_len-1, vocab_size)
                - cache: Intermediate cache from the model
        """
        seq, ignore_index, add_attn_z_loss, add_next_embed_loss = x.shape[1], self.ignore_index, self.add_attn_z_loss, self.add_continuous_pred_head

        # Setup input and target
        # Target is the input shifted by one position (next token prediction)
        inp, target = x, x[:, 1:]
        # Replace ignore_index tokens with pad_value for input
        inp = torch.where(inp == ignore_index, self.pad_value, inp)

        # Apply random masking for MLM-style training alongside autoregressive
        # This has been shown to improve training (https://arxiv.org/abs/2210.13432)
        if self.mask_prob > 0.:
            # Generate random values for selecting which tokens to mask
            rand = torch.randn(inp.shape, device = x.device)
            # Never mask the first token (important for sequence coherence)
            rand[:, 0] = -torch.finfo(rand.dtype).max
            # Determine how many tokens to mask
            num_mask = min(int(seq * self.mask_prob), seq - 1)
            # Select top-k random positions to mask
            indices = rand.topk(num_mask, dim = -1).indices
            # Create mask: True where NOT masked (inverted for attention mask)
            mask = ~torch.zeros_like(inp).scatter(1, indices, 1.).bool()
            # Pass mask to model so it can mask key/values in self-attention
            kwargs.update(self_attn_kv_mask = mask)

        # Forward pass through the network
        out, cache = self.net(
            inp,
            return_intermediates = True,
            return_attn_z_loss = add_attn_z_loss,
            return_next_embed_pred = add_next_embed_loss,
            prepend_embeds = prepend_embeds,
            **kwargs
        )

        # Unpack output based on whether continuous prediction is enabled
        if add_next_embed_loss:
            logits, (next_embed_pred, init_embeds) = out
        else:
            logits = out

        # Remove prepended embeddings from logits if they were used
        # This ensures logits align with the actual input sequence
        if exists(prepend_embeds):
            prepend_len = prepend_embeds.shape[1]
            logits = logits[:, prepend_len:]

        # Remove the last logit (we don't have a target for it)
        # Logits shape: (batch, seq_len-1, vocab_size)
        # Target shape: (batch, seq_len-1)
        logits = logits[:, :-1]

        # Select appropriate loss function based on model output type
        # Use NLL loss if model outputs log probabilities, otherwise cross-entropy
        loss_fn = F.cross_entropy if not self.net.output_is_log_prob else F.nll_loss

        # Compute cross-entropy loss for next token prediction
        # Rearrange for PyTorch loss function: (batch, classes, sequence)
        loss = loss_fn(
            rearrange(logits, 'b n c -> b c n'),
            target,
            ignore_index = ignore_index
        )

        # Add attention z-loss if enabled (for training stability)
        if add_attn_z_loss:
            loss = loss + cache.attn_z_loss

        # Add continuous embedding prediction loss if enabled
        # This predicts the next token's embedding (continuous space) alongside discrete prediction
        if add_next_embed_loss:
            # Only compute loss where we have valid targets (not ignore_index)
            mask = target != ignore_index
            # Align predictions and targets
            embed_pred = next_embed_pred[:, :-1]
            cont_targets = init_embeds[:, 1:].detach()

            # Compute L1 loss between predicted and target embeddings
            cont_loss = F.l1_loss(embed_pred, cont_targets, reduction = 'none')
            # Average only over non-ignored positions
            cont_loss = cont_loss[mask].mean()

            # Add weighted continuous loss to total loss
            loss = loss + cont_loss * self.next_embed_loss_weight

        # Return loss only or loss with outputs based on flag
        if not return_outputs:
            return loss

        return loss, (logits, cache)
