
"""
Belief State Transformer Implementation

This module implements the Belief State Transformer as described in:
Hu et al. https://arxiv.org/abs/2410.23506
Video explanation: https://www.youtube.com/watch?v=aqhbRtB2Fyg

The Belief State Transformer uses bidirectional context by training both forward and backward
autoregressive decoders. The model learns to predict tokens by combining embeddings from both
directions (prefix and suffix), creating a "belief state" about the sequence.

Key components:
- Forward decoder: processes sequences left-to-right (standard autoregressive)
- Backward decoder: processes sequences right-to-left (reverse autoregressive)
- Text head: predicts next/previous tokens given combined forward/backward embeddings
- Distance prediction: optionally predicts the distance between prefix and suffix positions
"""

from __future__ import annotations
from random import random

import torch
from torch.autograd import Function
from torch.nn import Module, ModuleList
from torch import nn, cat, stack, tensor, Tensor, arange, cartesian_prod
import torch.nn.functional as F

from x_transformers.autoregressive_wrapper import (
    eval_decorator,
    min_p,
)

from x_transformers.x_transformers import (
    Decoder,
    TransformerWrapper
)

import einx
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

# helper functions

def exists(v):
    """
    Check if a value is not None.

    Args:
        v: Any value to check

    Returns:
        bool: True if v is not None, False otherwise
    """
    return v is not None

def default(v, d):
    """
    Return a default value if the provided value is None.

    Args:
        v: The value to check
        d: The default value to return if v is None

    Returns:
        The value v if it exists (is not None), otherwise the default value d
    """
    return v if exists(v) else d

# a custom flip that can handle variable lengths across batch

def flip(x, dim = 1, lens = None):
    """
    Flip a tensor along a dimension, with support for variable-length sequences in a batch.

    This function can handle batches where different sequences have different lengths.
    When lengths are provided, it only flips the valid portion of each sequence (up to its length),
    leaving padding tokens in their original positions.

    Args:
        x (Tensor): Input tensor to flip, shape (batch, seq_len, ...) or (batch, seq_len)
        dim (int): Dimension along which to flip. Default is 1 (sequence dimension)
        lens (Tensor | None): Optional tensor of shape (batch,) containing the length of each
                              sequence in the batch. If None, all sequences are flipped completely.

    Returns:
        Tensor: The flipped tensor with the same shape as input

    Example:
        >>> x = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])  # padded sequences
        >>> lens = torch.tensor([3, 2])  # actual lengths
        >>> flip(x, lens=lens)
        tensor([[3, 2, 1, 0], [5, 4, 0, 0]])  # only valid tokens are flipped
    """
    # If no lengths specified, use standard PyTorch flip (all sequences have same length)
    if not exists(lens):
        return x.flip(dim)

    # Extract batch size, sequence length, and device from input tensor
    batch, seq_len, device = *x.shape[:2], x.device

    # Create a sequence of indices [0, 1, 2, ..., seq_len-1]
    seq = arange(seq_len, device = device)

    # Create a mask where True indicates valid positions (within the sequence length)
    # Shape: (batch, seq_len)
    mask = einx.less('j, i -> i j', seq, lens)

    # Create a masked sequence where invalid positions are set to -1
    # Valid positions keep their original indices
    masked_seq = einx.where('i j, j,', mask, seq, -1)

    # Sort indices in descending order to get flip indices
    # -1 values (padding) will end up at the end after sorting
    flip_indices = masked_seq.argsort(dim = -1, descending = True)

    # If input is 3D (has feature dimension), expand flip_indices to match
    if x.ndim == 3:
        flip_indices = repeat(flip_indices, '... -> ... d', d = x.shape[-1])

    # Gather elements according to flip_indices to perform the flip
    return x.gather(dim, flip_indices)

# detach multiple tensors and backward the gradients once

class DetachMultiple(Function):
    """
    Custom autograd function to detach multiple tensors from the computation graph.

    This is a memory optimization technique. By detaching tensors, we break the computation
    graph at that point, preventing PyTorch from storing all intermediate activations for
    backpropagation. However, gradients can still flow through these detached tensors during
    the backward pass.

    This is particularly useful in the Belief State Transformer where we compute both forward
    and backward embeddings. We can detach them to save memory while still allowing gradient
    flow during backpropagation.

    The trick: Detach tensors in forward pass to reduce memory, but still allow gradients
    to flow through them in backward pass.
    """

    @classmethod
    def forward(self, ctx, *tensors):
        """
        Forward pass: detach all input tensors from the computation graph.

        Args:
            ctx: Context object to save information for backward pass (not used here)
            *tensors: Variable number of tensors to detach

        Returns:
            tuple: Tuple of detached tensors, each requiring gradients
        """
        # Detach each tensor from the computation graph
        detached_tensors = tuple(t.detach() for t in tensors)

        # Re-enable gradient computation for each detached tensor
        # This allows gradients to flow through them in the backward pass
        for detached_tensor in detached_tensors:
            detached_tensor.requires_grad_()

        return detached_tensors

    @classmethod
    def backward(self, ctx, *grads):
        """
        Backward pass: simply pass through the gradients without modification.

        Args:
            ctx: Context object (not used here)
            *grads: Gradients flowing back from the subsequent layers

        Returns:
            tuple: The same gradients, passed through unchanged
        """
        return grads

# Create a convenient function alias for applying the DetachMultiple operation
detach_multiple = DetachMultiple.apply

# wrappers

class BeliefStateWrapper(Module):
    """
    Belief State Transformer Wrapper

    This module implements the Belief State Transformer architecture described in Figure 13
    of https://arxiv.org/abs/2410.23506

    The key idea is to train two autoregressive models simultaneously:
    1. Forward decoder: processes sequences left-to-right (prefix)
    2. Backward decoder: processes sequences right-to-left (suffix)

    The model learns to predict tokens by combining embeddings from both directions,
    creating a "belief state" about the sequence. This bidirectional context allows
    for more informed predictions and enables fill-in-the-middle generation.

    Training objective:
    - For each valid (forward_index, backward_index) pair where backward_index - forward_index >= 2:
      * Use forward embedding at forward_index and backward embedding at backward_index
      * Predict the next token after forward_index (forward prediction)
      * Predict the previous token before backward_index (backward prediction)

    Optional features:
    - Distance prediction: predict the distance between prefix and suffix positions
    - Distance conditioning: condition predictions on the known distance between positions
    - Variable loss weighting: weight forward and backward losses differently
    - Pair subsampling: train on a fraction of all valid pairs for memory efficiency
    """

    def __init__(
        self,
        forward_decoder: TransformerWrapper,
        backward_decoder: TransformerWrapper | None = None,
        train_frac_forward_backward_pairs: float = 1.,
        text_head: Module | None = None,
        backward_ar_loss_weight: float = 1., # can weigh the training of the backwards decoder differently, perhaps fwd/bwd have a shared backbone etc etc
        pred_distance = False,
        pred_distance_loss_weight: float = 1.,
        cond_on_distance = False,
        cond_on_distance_prob = 0.5,
        max_pred_distance = None
    ):
        """
        Initialize the Belief State Transformer wrapper.

        Args:
            forward_decoder (TransformerWrapper): The forward (left-to-right) autoregressive decoder
            backward_decoder (TransformerWrapper | None): The backward (right-to-left) autoregressive decoder.
                                                          If None, uses the same model as forward_decoder.
                                                          Default: None
            train_frac_forward_backward_pairs (float): Fraction of valid forward-backward pairs to train on.
                                                       Useful for memory efficiency. Range: (0, 1].
                                                       Default: 1.0 (train on all pairs)
            text_head (Module | None): Neural network head that takes concatenated forward/backward embeddings
                                       and predicts both next and previous tokens. If None, creates a default
                                       2-layer MLP. Default: None
            backward_ar_loss_weight (float): Weight for the backward autoregressive loss. Can be used to
                                            balance forward and backward training. Default: 1.0
            pred_distance (bool): Whether to predict the distance between prefix and suffix positions.
                                 This adds an auxiliary task. Default: False
            pred_distance_loss_weight (float): Weight for the distance prediction loss when pred_distance=True.
                                              Default: 1.0
            cond_on_distance (bool): Whether to condition the predictions on the distance between positions.
                                    When True, distance information is injected into the embeddings.
                                    Default: False
            cond_on_distance_prob (float): Probability of conditioning on distance during training when
                                          cond_on_distance=True. Range: (0, 1). Default: 0.5
            max_pred_distance (int | None): Maximum distance to predict when pred_distance=True.
                                           If None, uses the model's max_seq_len. Default: None

        Raises:
            AssertionError: If forward and backward decoders have different embedding dimensions or vocab sizes
        """
        super().__init__()
        # If backward decoder not specified, use the same transformer as forward decoder
        # Assumes the model can switch between forward/backward modes based on the suffix token
        backward_decoder = default(backward_decoder, forward_decoder)

        # Ensure forward and backward decoders are compatible
        assert forward_decoder.emb_dim == backward_decoder.emb_dim, 'forward and backwards model must have the same embedding dimension'
        assert forward_decoder.num_tokens == backward_decoder.num_tokens, 'forward and backwards model must have the same number of tokens'

        # Extract key dimensions from the forward decoder
        dim = forward_decoder.emb_dim
        num_tokens = forward_decoder.num_tokens
        max_seq_len = forward_decoder.max_seq_len

        self.num_tokens = num_tokens

        # ===== Suffix Token =====
        # Special learnable token that marks the start of the backward (suffix) sequence
        # This token is prepended to the reversed sequence when processing with backward decoder
        self.suffix_token = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.suffix_token, std = 0.02)

        # ===== Text Prediction Head =====
        # Takes concatenated forward and backward embeddings (dim * 2) and predicts:
        # - Next token for the forward sequence (num_tokens logits)
        # - Previous token for the backward sequence (num_tokens logits)
        # Total output: num_tokens * 2 logits
        if not exists(text_head):
            text_head = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.LeakyReLU(),
                nn.Linear(dim, num_tokens * 2),
            )

        self.text_head = text_head

        # ===== Distance Prediction (Optional) =====
        # Auxiliary task: predict the distance between prefix and suffix positions
        # This helps the model learn when it's close to the terminal state
        # (when suffix and prefix meet in the middle)
        self.max_pred_distance = default(max_pred_distance, max_seq_len)

        self.to_distance_logits = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, self.max_pred_distance),
        ) if pred_distance else None

        self.pred_distance_loss_weight = pred_distance_loss_weight

        # ===== Distance Conditioning (Optional) =====
        # When enabled, inject distance information into the embeddings
        # Applied probabilistically during training to avoid over-reliance
        assert 0. < cond_on_distance_prob < 1.

        self.cond_on_distance = cond_on_distance
        self.cond_on_distance_prob = cond_on_distance_prob

        if cond_on_distance:
            # Network to convert distance scalar to conditioning vector
            # Output has shape (dim * 2) to match concatenated embeddings
            self.to_distance_cond = nn.Sequential(
                Rearrange('... -> ... 1'),
                nn.Linear(1, dim),
                nn.LeakyReLU(),
                nn.Linear(dim, dim * 2),
            )

        # ===== Forward and Backward Decoders =====
        # Store the two autoregressive models
        self.forward_decoder = forward_decoder
        self.backward_decoder = backward_decoder

        # ===== Forward-Backward Pair Subsampling =====
        # For memory efficiency, can train on only a fraction of all valid pairs
        # Useful when sequences are long and number of pairs grows quadratically
        assert 0 < train_frac_forward_backward_pairs <= 1.
        self.train_frac_fb_pairs = train_frac_forward_backward_pairs
        self.needs_subsample_fb_pairs = train_frac_forward_backward_pairs < 1.

        # ===== Loss Weighting =====
        # Allow different weights for forward and backward autoregressive losses
        # Useful when forward and backward decoders share parameters
        self.backward_ar_loss_weight = backward_ar_loss_weight
        self.needs_loss_weight = backward_ar_loss_weight != 1.

        # Register as buffer so it moves to the correct device automatically
        self.register_buffer('loss_weights', tensor([1., self.backward_ar_loss_weight]))

        # ===== Sampling Configuration =====
        self.max_seq_len = self.forward_decoder.max_seq_len

    @torch.no_grad()
    @eval_decorator
    def generate_with_suffix_cond(
        self,
        prompts,
        seq_len,
        temperature = 1.25,
        cache_kv = False,
        suffix: Tensor | None = None, # the goal conditioning
        filter_logits_fn = min_p,
        filter_kwargs = dict(
            min_p = 0.1
        ),
        decode_backwards = False,
        **kwargs
    ):
        """
        Generate tokens autoregressively with optional suffix conditioning (fill-in-the-middle).

        This method enables goal-conditioned generation where the model fills in tokens between
        a given prefix (prompt) and an optional suffix (goal). It can also decode backwards
        for testing purposes.

        The generation process:
        1. Encode the suffix (if provided) using the backward decoder
        2. At each step, combine the current prefix embedding with the suffix embedding
        3. Predict the next token using the text head
        4. Append the sampled token and repeat

        Args:
            prompts (Tensor): Input prompt tokens, shape (batch, prompt_len) or (prompt_len,)
            seq_len (int): Number of tokens to generate
            temperature (float): Sampling temperature. Higher values = more random.
                                Temperature of 0 = greedy decoding. Default: 1.25
            cache_kv (bool): Whether to use key-value caching for faster generation.
                            Only works if the decoder supports it. Default: False
            suffix (Tensor | None): Optional goal conditioning suffix tokens.
                                   Shape: (batch, suffix_len) or (suffix_len,)
                                   If provided, performs fill-in-the-middle generation.
                                   Default: None
            filter_logits_fn (callable): Function to filter logits before sampling.
                                        Applied to reduce low-probability tokens.
                                        Default: min_p filtering
            filter_kwargs (dict): Keyword arguments for filter_logits_fn.
                                 Default: {'min_p': 0.1}
            decode_backwards (bool): If True, decode in reverse (for testing backward decoder).
                                    Default: False
            **kwargs: Additional arguments passed to the decoder

        Returns:
            Tensor: Generated tokens, shape matches input prompts shape with seq_len tokens added

        Note:
            - This method is decorated with @torch.no_grad() for inference-only usage
            - Suffix is automatically reversed when processing (backward autoregressive)
            - When decode_backwards=True, prompts are reversed and backward decoder is used
        """
        # Extract configuration
        max_seq_len, greedy, device = self.max_seq_len, temperature == 0., prompts.device

        # Pack prompts to ensure batch dimension exists
        prompts, batch_ps = pack([prompts], '* d')

        batch, orig_seq_len = prompts.shape

        # ===== Setup Decoder Direction =====
        # Normally decode forward, but can decode backwards for testing
        main_decoder = self.forward_decoder

        if decode_backwards:
            prompts = prompts.flip(1)  # Reverse the prompt
            main_decoder = self.backward_decoder

        out = prompts

        # ===== Initialize KV Cache =====
        # Cache stores key-value pairs from attention layers for faster generation
        cache = None

        # ===== Prepare Suffix Token =====
        # The suffix token marks the boundary between forward and backward sequences
        suffix_sos_tokens = rearrange(self.suffix_token, 'd -> 1 1 d')
        suffix_sos_tokens = repeat(suffix_sos_tokens, '1 1 d -> b 1 d', b = batch)

        # ===== Encode Suffix or Prefix (depending on direction) =====
        if not decode_backwards:
            # Standard forward decoding with optional suffix conditioning
            if exists(suffix):
                # Ensure suffix has batch dimension
                if suffix.ndim == 1:
                    suffix = repeat(suffix, 'n -> b n', b = batch)

                # Reverse suffix for backward autoregressive processing
                suffix = suffix.flip(1)

            # Encode the suffix using backward decoder
            suffix_embed = self.backward_decoder(
                suffix,
                prepend_embeds = suffix_sos_tokens,
                return_embeddings = True
            )

            # Extract the last embedding (represents the full suffix context)
            # This embedding will be combined with prefix embeddings during generation
            suffix_embed = suffix_embed[:, -1:]

        else:
            # When decoding backwards, we need a prefix embedding instead
            # For now, use a random token as placeholder
            prefix_embed = torch.randint(0, self.num_tokens, (batch, 1), device = device)
            prefix_embed = self.forward_decoder(prefix_embed, return_embeddings = True)

        # ===== Autoregressive Generation Loop =====
        for _ in range(seq_len):
            # Get embeddings from the main decoder
            embeds, new_cache = main_decoder(
                out,
                prepend_embeds = suffix_sos_tokens if decode_backwards else None,
                return_intermediates = True,
                return_embeddings = True,
                cache = cache,
                **kwargs
            )

            # Extract the embedding of the last token
            last_embeds = embeds[:, -1:]

            # Concatenate prefix and suffix embeddings for belief state prediction
            if not decode_backwards:
                # Forward: combine current prefix with suffix
                embeds = cat((last_embeds, suffix_embed), dim = -1)
            else:
                # Backward: combine prefix with current suffix
                embeds = cat((prefix_embed, last_embeds), dim = -1)

            # Update cache for next iteration if using KV caching
            if cache_kv and self.forward_decoder.can_cache_kv:
                cache = new_cache

            # Get predictions from text head
            # Output shape: (batch, 1, num_tokens * 2)
            # First half: forward predictions, second half: backward predictions
            forward_logits, backward_logits = self.text_head(embeds).chunk(2, dim = -1)

            # Select appropriate logits based on decoding direction
            logits = forward_logits if not decode_backwards else backward_logits

            logits = logits[:, -1]  # Remove sequence dimension

            # Sample next token
            if greedy:
                # Greedy decoding: always pick the most likely token
                sample = logits.argmax(dim = -1, keepdim = True)
            else:
                # Stochastic sampling with temperature
                filtered_logits = filter_logits_fn(logits, **filter_kwargs)
                probs = F.softmax(filtered_logits / temperature, dim = -1)
                sample = torch.multinomial(probs, 1)

            # Append sampled token to the output sequence
            out = torch.cat((out, sample), dim = -1)

        # Remove the original prompt, keeping only generated tokens
        out = out[:, orig_seq_len:]

        # Unpack to restore original shape (remove batch dim if input was 1D)
        out, = unpack(out, batch_ps, '* n')

        return out

    def forward(
        self,
        seq,
        lens: Tensor | None = None, # Int['b']
        loss_weight_by_fb_indices: callable | None = None
    ):
        """
        Forward pass for training the Belief State Transformer.

        This method implements the core training objective described in Figure 11 of the paper.
        It processes sequences in both forward and backward directions, then trains the model
        to predict tokens by combining embeddings from both directions.

        Training procedure:
        1. Process sequence with forward decoder (left-to-right)
        2. Process reversed sequence with backward decoder (right-to-left, prepended with suffix token)
        3. Generate all valid (forward_index, backward_index) pairs where backward_index - forward_index >= 2
        4. For each pair:
           - Concatenate forward_embedding[forward_index] and backward_embedding[backward_index]
           - Predict next token after forward_index (forward prediction)
           - Predict previous token before backward_index (backward prediction)
        5. Compute cross-entropy loss for all predictions

        The constraint (backward_index - forward_index >= 2) ensures there's at least one token
        between the positions, making the prediction task meaningful.

        Args:
            seq (Tensor): Input token sequence, shape (batch, seq_len)
                         Contains the training sequences
            lens (Tensor | None): Optional lengths of each sequence in the batch, shape (batch,)
                                 Used for variable-length sequences with padding.
                                 Padded positions are ignored in loss computation.
                                 Default: None (all sequences are full length)
            loss_weight_by_fb_indices (callable | None): Optional function to compute loss weights
                                                        based on forward-backward index pairs.
                                                        Takes fb_pairs tensor of shape (num_pairs, 2) and returns:
                                                        - 1D tensor (num_pairs,): weight per pair
                                                        - 2D tensor (num_pairs, 2): weight per (pair, direction)
                                                        Useful for normalizing loss across positions.
                                                        Default: None

        Returns:
            Tensor: Scalar loss value combining forward and backward prediction losses,
                   with optional distance prediction loss

        Implementation details:
            - Embeddings are detached after encoding to save memory during backpropagation
            - Can optionally subsample forward-backward pairs for memory efficiency
            - Can optionally predict distance between positions as auxiliary task
            - Can optionally condition predictions on distance information
            - Supports variable loss weighting for forward vs backward predictions
        """
        # Extract batch size, sequence length, and device
        batch, seq_len, device = *seq.shape, seq.device

        # ===== Handle Variable Length Sequences =====
        # Create a version of seq where padding tokens are masked with -1
        # These will be ignored in cross-entropy loss computation
        seq_for_labels = seq

        if exists(lens):
            # Create mask: True for valid positions, False for padding
            mask = einx.less('j, i -> i j', arange(seq_len, device = device), lens)
            # Set padding positions to -1 (will be ignored by cross_entropy)
            seq_for_labels = torch.where(mask, seq, -1)

        # ===== Forward Autoregressive Encoding =====
        # Process sequence left-to-right to get forward embeddings
        # Shape: (batch, seq_len, dim)
        forward_embeds = self.forward_decoder(seq, return_embeddings = True)

        # ===== Backward Autoregressive Encoding =====
        # Reverse the sequence for backward processing
        backward_seq = flip(seq, lens = lens)

        # Prepare suffix tokens to prepend to backward sequence
        # This marks the start of the backward (suffix) sequence
        suffix_tokens = repeat(self.suffix_token, 'd -> b 1 d', b = batch)

        # Process reversed sequence with backward decoder
        # The suffix token is prepended, so output has shape (batch, seq_len + 1, dim)
        backward_embeds = self.backward_decoder(
            backward_seq,
            prepend_embeds = suffix_tokens,
            return_embeddings = True
        )

        # Flip backward embeddings back to align with forward sequence positions
        # Now backward_embeds[i] corresponds to the backward context from position i
        backward_embeds = flip(backward_embeds, lens = lens)

        # ===== Memory Optimization =====
        # Detach embeddings to reduce memory usage during backpropagation
        # Gradients can still flow through, but we don't store the full computation graph
        forward_embeds, backward_embeds = detach_multiple(forward_embeds, backward_embeds)

        # ===== Generate Forward-Backward Index Pairs =====
        # Create all possible combinations of (forward_index, backward_index)
        seq_arange = arange(seq_len, device = device)

        # backward indices are offset by 1 to account for the prepended suffix token
        fb_pairs = cartesian_prod(seq_arange, seq_arange + 1)

        # ===== Filter to Valid Pairs (Figure 11) =====
        # Only keep pairs where there's at least 2 positions between them
        # This ensures there's meaningful prediction to be made
        # f - forward index, b - backward index, i - indices
        fi, bi = fb_pairs.unbind(dim = -1)

        # Valid if: backward_index - forward_index >= 2
        # Example: if fi=0, bi must be >= 2 (so there's at least token at position 1 to predict)
        valid_mask = (bi - fi) >= 2

        fb_pairs = fb_pairs[valid_mask]

        # ===== Optional Pair Subsampling =====
        # For memory efficiency, can train on only a fraction of pairs
        if self.needs_subsample_fb_pairs:
            num_pairs = fb_pairs.shape[0]

            # Calculate how many pairs to keep (at least 1)
            num_subsampled = max(int(num_pairs * self.train_frac_fb_pairs), 1)

            # Randomly select pairs to train on
            rand_subsampled_indices = torch.randperm(num_pairs, device = device)[:num_subsampled]

            fb_pairs = fb_pairs[rand_subsampled_indices]

        # ===== Get Prediction Labels =====
        fi, bi = fb_pairs.unbind(dim = -1)

        # Forward label: token at position (fi + 1) - next token after forward position
        # Backward label: token at position (bi - 1) - previous token before backward position
        labels_fi, labels_bi = (fi + 1), (bi - 1)

        # Extract actual token values for labels
        forward_labels, backward_labels = seq_for_labels[:, labels_fi], seq_for_labels[:, labels_bi]

        # Concatenate labels: first forward, then backward
        # Shape: (batch, num_pairs * 2)
        labels = cat((forward_labels, backward_labels), dim = -1)

        # ===== Concatenate Forward and Backward Embeddings =====
        # For each pair, combine the forward embedding at fi with backward embedding at bi
        # This creates the "belief state" representation
        # Shape: (batch, num_pairs, dim * 2)
        fb_embeds = cat((
            forward_embeds[:, fi],
            backward_embeds[:, bi]
        ), dim = -1)

        # ===== Predict Tokens =====
        # Text head outputs num_tokens * 2 logits:
        # - First num_tokens: forward prediction (next token)
        # - Second num_tokens: backward prediction (previous token)
        logits = self.text_head(fb_embeds)

        # ===== Compute Cross-Entropy Loss =====
        # Rearrange to shape (batch, num_tokens * 2, num_pairs) for cross_entropy
        loss = F.cross_entropy(
            rearrange(logits, 'b n (fb l) -> b l (fb n)', fb = 2),
            labels,
            reduction = 'none' if self.needs_loss_weight else 'mean',
            ignore_index = -1  # Ignore padding tokens (marked as -1)
        )

        # ===== Optional Distance Conditioning =====
        # Probabilistically inject distance information into embeddings during training
        cond_on_distance = self.cond_on_distance and (random() < self.cond_on_distance_prob)

        if cond_on_distance:
            # Calculate distance between positions
            distance = (bi - fi).float()
            # Convert distance to conditioning vector
            distance_cond = self.to_distance_cond(distance)

            # Modulate embeddings with distance conditioning (element-wise multiplication)
            fb_embeds = fb_embeds * distance_cond

        # ===== Optional Distance Prediction =====
        # Auxiliary task: predict the distance between positions
        # Only predict when not conditioning (to avoid trivial solution)
        if exists(self.to_distance_logits) and not cond_on_distance:
            # Predict distance from concatenated embeddings
            distance_logits = self.to_distance_logits(fb_embeds)

            # Create distance labels, clamping to max prediction distance
            distance_labels = (bi - fi).clamp(max = self.max_pred_distance - 1)
            distance_labels = repeat(distance_labels, 'n -> b n', b = batch)

            # Compute distance prediction loss
            pred_dist_loss = F.cross_entropy(
                rearrange(distance_logits, 'b n l -> b l n'),
                distance_labels
            )

            # Add weighted distance loss to main loss
            loss = (
                loss +
                pred_dist_loss * self.pred_distance_loss_weight
            )

        # ===== Optional Loss Weighting =====
        # Apply different weights to forward vs backward predictions, and/or per-pair weights
        needs_loss_weight = default(self.needs_loss_weight, exists(loss_weight_by_fb_indices))

        if needs_loss_weight:
            # Reshape to separate forward and backward losses
            # Shape: (batch, 2, num_pairs) where 2 = [forward, backward]
            loss = rearrange(loss, 'b (fb n) -> b fb n', fb = 2)

            # Apply forward/backward loss weighting
            if self.needs_loss_weight:
                # Multiply by [1.0, backward_ar_loss_weight]
                loss = einx.multiply('b fb n, fb', loss, self.loss_weights)

            # Apply custom per-pair or per-(pair,direction) weighting
            # This can normalize for the fact that earlier tokens have more eligible pairs
            if exists(loss_weight_by_fb_indices):
                loss_weight = loss_weight_by_fb_indices(fb_pairs)

                if loss_weight.ndim == 1:
                    # Per-pair weighting: shape (num_pairs,)
                    loss = einx.multiply('b fb n, n', loss, loss_weight)
                elif loss_weight.ndim == 2:
                    # Per-(pair, direction) weighting: shape (num_pairs, 2)
                    loss = einx.multiply('b fb n, n fb', loss, loss_weight)
                else:
                    raise ValueError('invalid loss weight dims')

            # Average all losses into a scalar
            loss = loss.mean()

        return loss
