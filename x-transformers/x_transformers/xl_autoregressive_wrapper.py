"""
XL Autoregressive Wrapper Module

This module provides an autoregressive wrapper for Transformer-XL models, enabling
text generation and training with memory mechanisms. The wrapper handles chunking
of long sequences and maintains memory across segments for improved context retention.
"""

from math import ceil

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, pack, unpack
from x_transformers.autoregressive_wrapper import top_p, top_k, eval_decorator

# helper functions

def exists(val):
    """
    Check if a value exists (is not None).

    Args:
        val: The value to check

    Returns:
        bool: True if val is not None, False otherwise
    """
    return val is not None

def divisible_by(numer, denom):
    """
    Check if a number is evenly divisible by another number.

    Args:
        numer: The numerator (dividend)
        denom: The denominator (divisor)

    Returns:
        bool: True if numer is evenly divisible by denom, False otherwise
    """
    return (numer % denom) == 0 

# xl autoregressive wrapper class

class XLAutoregressiveWrapper(nn.Module):
    """
    Transformer-XL Autoregressive Wrapper

    This wrapper enables autoregressive text generation and training for Transformer-XL models.
    It handles the chunking of long sequences into segments that fit within the model's maximum
    sequence length, and maintains memory across segments to provide extended context.

    The wrapper provides two main functionalities:
    1. Training: Processes sequences in chunks, maintaining memory across segments, and computes
       weighted cross-entropy loss across all chunks.
    2. Generation: Generates new tokens autoregressively using cached memories and optional
       filtering strategies (top-k, top-p, temperature).

    Attributes:
        net: The underlying Transformer-XL network
        max_seq_len (int): Maximum sequence length the network can handle per segment
        ignore_index (int): Index to ignore in loss calculation (typically for padding tokens)
        pad_value (int): Value to use for padding generated sequences
    """

    def __init__(
        self,
        net,
        ignore_index = -100,
        pad_value = 0
    ):
        """
        Initialize the XL Autoregressive Wrapper.

        Args:
            net: The Transformer-XL network to wrap. Must have a max_seq_len attribute.
            ignore_index (int, optional): Token index to ignore when calculating loss.
                Defaults to -100 (PyTorch's default for cross_entropy).
            pad_value (int, optional): Value to use for padding tokens in generated sequences.
                Defaults to 0.
        """
        super().__init__()
        # Store padding configuration
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        # Store the network and extract its maximum sequence length
        self.net = net
        self.max_seq_len = net.max_seq_len

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        start_tokens,
        seq_len,
        eos_token = None,
        temperature = 1.,
        filter_logits_fn = top_k,
        filter_kwargs: dict = dict(),
        mems = None,
        **kwargs
    ):
        """
        Generate new tokens autoregressively using the Transformer-XL model.

        This method generates new tokens one at a time, maintaining memory across segments
        for extended context. It supports various sampling strategies through logit filtering
        (top-k, top-p), temperature scaling, and early stopping with end-of-sequence tokens.

        The generation process works in two phases:
        1. Pre-processing: Process all leading token segments to build up memory
        2. Generation: Autoregressively sample new tokens using the accumulated memory

        Args:
            start_tokens (torch.Tensor): Initial tokens to condition generation on.
                Shape can be (batch,) or (batch, seq_len).
            seq_len (int): Number of new tokens to generate.
            eos_token (int, optional): End-of-sequence token ID. If provided and generated,
                stops generation and pads remaining positions. Defaults to None.
            temperature (float, optional): Sampling temperature. Higher values (>1) increase
                randomness, lower values (<1) make sampling more deterministic. Defaults to 1.0.
            filter_logits_fn (callable, optional): Function to filter logits before sampling
                (e.g., top_k, top_p). Defaults to top_k.
            filter_kwargs (dict, optional): Keyword arguments to pass to filter_logits_fn.
                Defaults to empty dict.
            mems (optional): Initial memory state. If None, starts with no memory.
                Defaults to None.
            **kwargs: Additional keyword arguments to pass to the network's forward pass.

        Returns:
            torch.Tensor: Generated tokens (excluding the start_tokens), with the same
                batch dimensions as the input. Shape: (batch, seq_len) or original shape.
        """
        # Extract device and maximum sequence length for this generation
        device, max_seq_len = start_tokens.device, self.max_seq_len

        # Pack start_tokens to ensure consistent shape handling (batch, seq_len)
        # ps stores the packing specification for unpacking later
        start_tokens, ps = pack([start_tokens], '* n')

        # Get batch size and initial sequence length
        b, t = start_tokens.shape

        # Split start_tokens into segments of max_seq_len
        # all_leading_tokens contains all complete segments except the last (possibly incomplete) one
        *all_leading_tokens, _ = start_tokens.split(max_seq_len, dim = -1)

        # Phase 1: Process all leading segments to build up memory
        # This "catches up" the memory to the current position without generating

        for leading_tokens in all_leading_tokens:
            # Process each leading segment through the network
            # We only care about updating mems, not the logits
            _, mems = self.net(
                leading_tokens,
                mems = mems,
                return_mems = True,
                **kwargs
            )

        # Phase 2: Begin autoregressive generation from the current segment

        # Calculate starting position for generation (after all leading tokens)
        curr_pos = len(all_leading_tokens) * max_seq_len
        # Initialize current memories with the accumulated memory from leading segments
        curr_mems = mems

        # Cache will store intermediate activations for efficient generation
        cache = None
        # Output starts with all the start_tokens
        out = start_tokens

        # Generate seq_len new tokens one at a time
        for _ in range(seq_len):
            # Get the current total sequence length
            curr_segment_len = out.shape[-1]
            # Check if we've reached the end of a segment (divisible by max_seq_len)
            is_last_segment_tokens = divisible_by(curr_segment_len, max_seq_len)

            # Extract tokens from current position onward (current segment being processed)
            x = out[:, curr_pos:]

            # Forward pass through the network
            # - Use accumulated memories from previous segments
            # - Use cache for efficient attention within current segment
            # - return_intermediates gives us cache with mems
            logits, cache = self.net(
                x,
                mems = curr_mems,
                cache = cache,
                return_mems = True,
                return_intermediates = True,
                **kwargs
            )

            # Extract the updated memory from cache
            mems = cache.mems

            # Get logits for the last position only (next token prediction)
            logits = logits[:, -1]
            # Apply filtering function (e.g., top-k, top-p) to logits
            filtered_logits = filter_logits_fn(logits, **filter_kwargs)
            # Convert filtered logits to probabilities with temperature scaling
            # Higher temperature = more random, lower temperature = more deterministic
            probs = F.softmax(filtered_logits / temperature, dim=-1)

            # Sample one token from the probability distribution
            sample = torch.multinomial(probs, 1)

            # If we've completed a full segment, update position and memories
            if is_last_segment_tokens:
                # Move current position to start of next segment
                curr_pos = curr_segment_len
                # Update current memories to use for next segment
                curr_mems = mems

            # Append the sampled token to output sequence
            out = torch.cat((out, sample), dim=-1)

            # Handle early stopping if end-of-sequence token is specified
            if exists(eos_token):
                # Check which positions in output contain the EOS token
                is_eos_tokens = (out == eos_token)

                # Check if all sequences in batch have generated at least one EOS token
                if is_eos_tokens.any(dim = -1).all():
                    # Mask out everything after the first EOS token in each sequence
                    # Shift right by 1 to include the EOS token itself, but mask everything after
                    shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
                    # Cumsum creates mask: 0s before EOS, 1s from EOS onward
                    mask = shifted_is_eos_tokens.float().cumsum(dim = -1) >= 1
                    # Replace positions after EOS with pad_value
                    out = out.masked_fill(mask, self.pad_value)
                    # Stop generation early since all sequences have finished
                    break

        # Remove the initial start_tokens, keeping only newly generated tokens
        out = out[:, t:]

        # Unpack to restore original input shape
        out, = unpack(out, ps, '* n')

        return out

    def forward(
        self,
        x,
        mems = None,
        **kwargs
    ):
        """
        Forward pass for training the Transformer-XL model.

        This method implements the training procedure for autoregressive language modeling.
        It processes the input sequence in chunks (segments) that fit within max_seq_len,
        maintaining memory across segments. The loss is computed as a weighted sum across
        all chunks, where each chunk's weight is proportional to its length.

        The autoregressive training setup:
        - Input: tokens at positions [0, 1, 2, ..., n-1]
        - Labels: tokens at positions [1, 2, 3, ..., n]
        - Each position predicts the next token

        Args:
            x (torch.Tensor): Input token sequence. Shape: (batch, seq_len)
                This should include both context and target tokens.
            mems (optional): Initial memory state from previous segments.
                If None, starts with no memory. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the network's forward pass.

        Returns:
            torch.Tensor: Scalar tensor containing the weighted average cross-entropy loss
                across all chunks. The loss accounts for varying chunk sizes.
        """
        # Extract configuration values
        ignore_index, max_seq_len = self.ignore_index, self.max_seq_len

        # Prepare input-label pairs for autoregressive training
        # Input: all tokens except the last one
        # Labels: all tokens except the first one (shifted by 1 position)
        x, labels = x[:, :-1], x[:, 1:]

        # Get total sequence length (after removing last token)
        seq_len = x.shape[1]

        # Split sequences into chunks of max_seq_len
        # This is necessary because Transformer-XL has a maximum segment length

        # Split input tokens into chunks
        split_x = x.split(max_seq_len, dim = -1)
        # Split corresponding labels into chunks
        split_labels = labels.split(max_seq_len, dim = -1)
        # Calculate weight for each chunk based on its proportion of total length
        # This ensures chunks contribute to loss proportionally to their size
        loss_weights = tuple((t.shape[-1] / seq_len) for t in split_x)

        # Choose appropriate loss function based on model output type
        # If model outputs log probabilities, use NLL loss; otherwise use cross-entropy
        loss_fn = F.cross_entropy if not self.net.output_is_log_prob else F.nll_loss

        # Process each chunk and accumulate weighted losses

        # Initialize total loss accumulator
        total_loss = 0.

        # Iterate through all chunks with their labels and weights
        for chunk, chunk_labels, loss_weight in zip(split_x, split_labels, loss_weights):

            # Forward pass through the network for this chunk
            # mems are automatically updated and passed to next chunk
            logits, mems = self.net(
                chunk,
                mems = mems,
                return_mems = True,
                **kwargs
            )

            # Compute loss for this chunk
            # Rearrange logits from (batch, seq, classes) to (batch, classes, seq)
            # as required by PyTorch's cross_entropy function
            loss = loss_fn(
                rearrange(logits, 'b n c -> b c n'),
                chunk_labels,
                ignore_index = ignore_index
            )

            # Add weighted loss to total
            # Weighting ensures fair contribution regardless of chunk size
            total_loss = total_loss + loss * loss_weight

        return total_loss
