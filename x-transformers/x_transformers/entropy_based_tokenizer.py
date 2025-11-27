from __future__ import annotations
from itertools import zip_longest

import torch
from torch import tensor
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence

import einx
from einops import repeat, rearrange, pack, unpack

# helper functions

def exists(v):
    """
    Check if a value exists (is not None).

    Args:
        v: Any value to check for existence

    Returns:
        bool: True if v is not None, False otherwise
    """
    return v is not None

def default(v, d):
    """
    Return the value if it exists, otherwise return a default value.

    Args:
        v: The value to check
        d: The default value to return if v is None

    Returns:
        The value v if it exists, otherwise the default value d
    """
    return v if exists(v) else d

def log(t, eps = 1e-20):
    """
    Compute the logarithm of a tensor with numerical stability.

    Clamps the tensor values to a minimum epsilon value before taking the log
    to prevent numerical issues with very small or zero values.

    Args:
        t (torch.Tensor): Input tensor
        eps (float): Minimum value to clamp to (default: 1e-20)

    Returns:
        torch.Tensor: Natural logarithm of the clamped tensor
    """
    return t.clamp(min = eps).log()

def calc_entropy_from_logits(logits):
    """
    Calculate the entropy from logits using the Shannon entropy formula.

    Entropy measures the uncertainty or "surprise" in a probability distribution.
    Higher entropy indicates more uncertainty in predictions. The formula is:
    H = -sum(p * log(p)) where p is the probability distribution.

    Args:
        logits (torch.Tensor): Logits tensor of shape [..., num_classes]

    Returns:
        torch.Tensor: Entropy values of shape [...], with the last dimension
                     reduced by the sum operation
    """
    prob = logits.softmax(dim = -1)
    return -(prob * log(prob)).sum(dim = -1)

# entropy based tokenizer applied in byte-latent transformer paper
# they use a simple entropy threshold for segmenting a string into variable sized tokens

# https://arxiv.org/abs/2412.09871

class EntropyBasedTokenizer(Module):
    """
    Entropy-Based Tokenizer for variable-length token segmentation.

    This tokenizer implements the entropy-based segmentation approach from the
    Byte-Latent Transformer paper (https://arxiv.org/abs/2412.09871). It uses
    a decoder model to predict the next token at each position, calculates the
    entropy of the predictions, and creates token boundaries where the entropy
    exceeds a threshold.

    The key idea is that high entropy (high uncertainty) in predictions indicates
    a good place to segment the sequence into separate tokens, as it represents
    positions where the model is "surprised" by what comes next.

    Attributes:
        decoder (Module): A trained decoder model that produces logits for next-token
                         prediction. Used to calculate entropy at each position.
        entropy_threshold (float): The threshold value for entropy. Positions with
                                  entropy above this threshold become token boundaries.
        max_token_size (int | None): Optional maximum size for any single token.
                                     Prevents overly long tokens in cases of
                                     repeating subsequences.
    """
    def __init__(
        self,
        decoder: Module,
        entropy_threshold: float,
        max_token_size: int | None = None
    ):
        """
        Initialize the EntropyBasedTokenizer.

        Args:
            decoder (Module): A trained decoder module that takes a sequence and
                            returns logits for next-token prediction
            entropy_threshold (float): The entropy threshold for creating token
                                      boundaries. Higher values create fewer,
                                      longer tokens; lower values create more,
                                      shorter tokens.
            max_token_size (int | None): Optional maximum number of elements
                                        allowed in a single token. If specified,
                                        tokens exceeding this size will be split
                                        into smaller chunks.
        """
        super().__init__()
        self.decoder = decoder
        self.entropy_threshold = entropy_threshold

        self.max_token_size = max_token_size

    @torch.no_grad()
    def forward(
        self,
        seq,            # Float['b n'] | Float['n']
        lens = None,    # Int['b']
        return_segmented_seq = False,
        decoder_forward_kwargs: dict = dict()
    ):
        """
        Tokenize a sequence based on entropy-driven segmentation.

        This method processes input sequences through a decoder to compute entropy
        at each position, then creates token boundaries where entropy exceeds the
        threshold. The result is a variable-length tokenization that adapts to
        the complexity and predictability of different parts of the sequence.

        Args:
            seq (torch.Tensor): Input sequence tensor. Can be:
                              - Float['b n']: Batched sequences of shape (batch, seq_len)
                              - Float['n']: Single sequence of shape (seq_len,)
            lens (torch.Tensor | None): Optional tensor of actual sequence lengths
                                       for variable-length sequences in the batch.
                                       Shape: Int['b'] where each value indicates
                                       the valid length of the corresponding sequence.
            return_segmented_seq (bool): If True, returns the actual segmented
                                        sequences as lists of tensors. If False,
                                        returns only the token lengths. Default: False.
            decoder_forward_kwargs (dict): Additional keyword arguments to pass
                                          to the decoder's forward method.

        Returns:
            If return_segmented_seq is False:
                torch.Tensor: Token lengths tensor of shape (batch, num_tokens)
                             where each value indicates the length of a token.
                             Zero-padded to the maximum number of tokens in the batch.

            If return_segmented_seq is True:
                list: For batched input, a list of lists where each inner list
                     contains the segmented tokens (as tensors) for one sequence.
                     For single sequence input, a list of tensors representing
                     the segmented tokens.

        Implementation Details:
            1. Runs the decoder on the input to get logits
            2. Calculates entropy from the logits
            3. Creates boundaries where entropy >= threshold
            4. Optionally enforces max_token_size constraint
            5. Computes token lengths from boundaries
            6. Optionally segments the actual sequence based on boundaries
        """
        # Handle single sequence input by adding batch dimension
        no_batch_dim = seq.ndim == 1
        seq, maybe_batch_ps = pack((seq,), '* n')

        # Set decoder to evaluation mode (disables dropout, etc.)
        self.decoder.eval()

        # Determine if we're working with variable-length sequences
        is_var_length = exists(lens)
        batch, seq_len, device, max_token_size = *seq.shape, seq.device, self.max_token_size

        # Create position indices for the sequence (0, 1, 2, ..., seq_len-1)
        arange = torch.arange(seq_len, device = device)

        # Forward through a small trained decoder and get the entropies of the logits
        # The decoder predicts what comes next at each position, producing logits
        logits = self.decoder(seq, **decoder_forward_kwargs)

        # Calculate entropy at each position - higher entropy means more uncertainty/surprise
        entropies = calc_entropy_from_logits(logits)

        # Get length mask for boundaries - this masks out positions beyond the actual sequence length
        # Start with all True mask (for fixed-length sequences)
        mask = tensor(True, device = device)

        # If sequences have different lengths, create a mask that's True only for valid positions
        if is_var_length:
            # mask[b, n] is True if position n < lens[b]
            mask = einx.less('n, b -> b n', arange, lens)

        # Create mask for positions where entropy exceeds threshold (high surprise/uncertainty)
        # These positions are candidates for token boundaries, but only within valid sequence lengths
        over_thres_mask = (entropies >= self.entropy_threshold) & mask

        # Prepare position indices (1-indexed instead of 0-indexed) for later boundary extraction
        # This is needed because we want to extract the positions just after boundaries
        arange_plus_one = arange + 1
        # Expand to batch dimension: shape (seq_len,) -> (batch, seq_len)
        arange_plus_one = repeat(arange_plus_one, 'n -> b n', b = batch)

        # Initialize boundaries tensor from the over-threshold mask
        # boundaries[b, n] will be True at positions that mark the END of a token
        boundaries = over_thres_mask.clone()

        # Ensure the last valid position of each sequence is marked as a boundary
        # This guarantees every sequence ends with a complete token

        if not is_var_length:
            # For fixed-length sequences, mark the last position of all sequences
            boundaries[..., -1] = True
        else:
            # For variable-length sequences, mark the actual last position for each sequence
            # lens - 1 gives the index of the last valid position
            scatter_indices = rearrange(lens - 1, 'b -> b 1')
            # Use scatter to set those specific positions to True
            boundaries.scatter_(-1, scatter_indices, True)

        # Handle max token size constraint - prevents tokens from becoming too large
        # This is important because entropy-based segmentation can create very long tokens
        # for repeating subsequences (a known flaw of the technique)

        if exists(max_token_size):
            # Assign token IDs by doing cumulative sum over boundaries
            # This gives each position a token ID (0, 0, 0, 1, 1, 2, 2, 2, 2, ...)
            token_ids = boundaries.cumsum(dim = -1)
            # Shift right and pad with 0 to get proper token IDs starting from 0
            token_ids = F.pad(token_ids, (1, -1), value = 0)

            # Find the maximum number of tokens across all sequences in the batch
            max_num_tokens = boundaries.sum(dim = -1).amax().item()
            # Create a sequence of token IDs for indexing
            token_ids_seq = torch.arange(max_num_tokens, device = device)

            # Create a 3D mask: token_mask[b, j, i] is True where sequence b at position i belongs to token j
            token_mask = einx.equal('j, b i -> b j i', token_ids_seq, token_ids)

            # For each token, count positions from 0, 1, 2, ... within that token
            # This gives us the sub-sequence position within each token
            token_sub_seq_arange = token_mask.cumsum(dim = -1)

            # Identify positions where we've reached multiples of max_token_size within a token
            # These positions need to become new boundaries to split oversized tokens
            sub_seq_boundaries = (token_sub_seq_arange % max_token_size == 0)
            # Only keep boundaries that are actually part of a token (use token_mask)
            # Reduce from 3D to 2D by checking if any token needs splitting at position i
            sub_seq_boundaries = (sub_seq_boundaries & token_mask).any(dim = 1)

            # Combine original entropy-based boundaries with max-size boundaries
            boundaries = boundaries | sub_seq_boundaries

            # Ensure we don't add boundaries outside the valid sequence length
            if exists(mask):
                boundaries = boundaries & mask

        # Calculate the total number of tokens for each sequence in the batch
        # by counting True values in boundaries tensor along the sequence dimension
        num_tokens = boundaries.sum(dim = -1)

        # Extract the actual position indices where boundaries occur
        # arange_plus_one[boundaries] selects positions, then split by num_tokens
        # This gives us a list of tensors, each containing the end positions of tokens for one sequence
        indices = arange_plus_one[boundaries].split(num_tokens.tolist())

        # Calculate token lengths from the boundary indices
        # Token length = difference between consecutive boundary positions
        token_lengths = []

        for one_indices in indices:
            # Prepend 0 to the indices to represent the start of the sequence
            # So if indices are [3, 7, 10], padded becomes [0, 3, 7, 10]
            padded_indices = F.pad(one_indices, (1, 0), value = 0.)
            # Calculate differences: [3-0, 7-3, 10-7] = [3, 4, 3]
            # These are the lengths of each token
            one_token_lengths = padded_indices[1:] - padded_indices[:-1]

            token_lengths.append(one_token_lengths)

        # Pad all token_lengths to the same size (max number of tokens in batch)
        # Shorter sequences get zero-padded at the end
        token_lengths = pad_sequence(token_lengths, batch_first = True)

        # Early return if we only need token lengths, not the actual segmented sequences
        if not return_segmented_seq:
            # Remove batch dimension if input was a single sequence
            token_lengths, = unpack(token_lengths, maybe_batch_ps, '* num_tokens')

            return token_lengths

        # Segment the actual sequence based on the computed token lengths
        # This creates lists of tensors where each tensor is one variable-length token

        # Convert lens to a tuple if it doesn't exist (for zip_longest)
        lens = default(lens, (None,))
        segmented_seq = []

        # Process each sequence in the batch along with its length and token lengths
        for one_seq, one_len, one_token_length in zip_longest(seq, lens, token_lengths):

            # If this sequence has a specific length, truncate to that length
            # (removes any padding that may have been added)
            if exists(one_len):
                one_seq = one_seq[:one_len]

            # Remove zero-padded token lengths (these were added to make all sequences same size)
            # Only keep actual token lengths (positive values)
            one_token_length = one_token_length[one_token_length > 0]

            # Split the sequence into tokens using the token lengths
            # e.g., if one_token_length = [3, 4, 3], this splits the sequence into
            # 3 tensors of lengths 3, 4, and 3
            splitted_seq = one_seq.split(one_token_length.tolist())
            segmented_seq.append(splitted_seq)

        # If input was a single sequence (no batch dimension), return just that sequence's segments
        # rather than a list of lists
        if no_batch_dim:
            segmented_seq = segmented_seq[0]

        return segmented_seq
