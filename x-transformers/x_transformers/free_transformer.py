"""
Free Transformer Implementation

This module implements the Free Transformer architecture as described in:
https://arxiv.org/abs/2510.17558
by François Fleuret
Video explanation: https://www.youtube.com/watch?v=Nao16-6l6dQ

The Free Transformer uses a unique architecture with:
- An encoder that compresses sequences into discrete latent representations
- Binary mapping for efficient latent representation
- A two-stage decoder (head and tail) for generation
- KL divergence regularization to control latent information
"""
from __future__ import annotations

# https://arxiv.org/abs/2510.17558
# François Fleuret
# https://www.youtube.com/watch?v=Nao16-6l6dQ

import math

import torch
from torch import nn, Tensor, is_tensor, tensor, arange
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from x_transformers.x_transformers import (
    Encoder,
    Decoder,
    TransformerWrapper
)

from x_transformers.autoregressive_wrapper import (
    gumbel_sample,
    top_p,
    top_k
)

from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat, einsum, pack, unpack

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
    Return value v if it exists, otherwise return default value d.

    Args:
        v: The value to check
        d: The default value to return if v is None

    Returns:
        v if v is not None, otherwise d
    """
    return v if exists(v) else d

def log(t, eps = 1e-20):
    """
    Compute the natural logarithm with numerical stability.

    Clamps the input tensor to a minimum value before taking the log
    to avoid numerical instability from log(0).

    Args:
        t (Tensor): Input tensor
        eps (float): Minimum value to clamp to (default: 1e-20)

    Returns:
        Tensor: Natural logarithm of the clamped input
    """
    return t.clamp_min(eps).log()

def pack_with_inverse(t, pattern):
    """
    Pack a tensor according to a pattern and return an inverse function.

    This function packs a tensor using einops and creates a closure that
    can unpack the result back to the original shape.

    Args:
        t (Tensor): Tensor to pack
        pattern (str): Einops pattern string for packing

    Returns:
        tuple: (packed_tensor, inverse_function)
            - packed_tensor: The packed result
            - inverse_function: A function that can unpack tensors back to original shape
    """
    packed, ps = pack([t], pattern)

    def inverse(out, inv_pattern = None):
        """
        Inverse function to unpack tensors.

        Args:
            out (Tensor): Tensor to unpack
            inv_pattern (str, optional): Pattern for unpacking. Defaults to original pattern.

        Returns:
            Tensor: Unpacked tensor
        """
        inv_pattern = default(inv_pattern, pattern)
        unpacked, = unpack(out, ps, inv_pattern)
        return unpacked

    return packed, inverse

# binary mapper

# NAT (nats) is the natural unit of information, equal to log(2) bits
NAT = math.log(2)

def binary_entropy(logits):
    """
    Calculate the binary entropy for a set of logits.

    Binary entropy measures the uncertainty in a binary distribution.
    This function computes: H = -sum(p * log(p) + (1-p) * log(1-p))
    where p = sigmoid(logits).

    Args:
        logits (Tensor): Binary logits of shape (..., bits)

    Returns:
        Tensor: Entropy values summed over the last dimension, shape (...)
    """
    # Convert logits to probabilities using sigmoid
    prob = logits.sigmoid()
    not_prob = 1. - prob

    # Calculate binary entropy: -sum(p * log(p) + (1-p) * log(1-p))
    return -(prob * F.logsigmoid(logits) + not_prob * F.logsigmoid(-logits)).sum(dim = -1)

class BinaryMapper(Module):
    """
    Maps continuous logits to discrete binary codes using straight-through estimation.

    This module converts continuous logit values into discrete binary representations,
    enabling differentiable discrete latent variable modeling. It uses:
    - Binary sampling during the forward pass
    - Straight-through gradient estimation for backpropagation
    - KL divergence loss to regularize the entropy of the binary distribution

    The mapper creates 2^bits possible discrete codes and learns to map inputs
    to these codes while controlling the information bottleneck via KL loss.
    """
    def __init__(
        self,
        bits = 1,
        kl_loss_threshold = NAT # 1 bit
    ):
        """
        Initialize the BinaryMapper.

        Args:
            bits (int): Number of bits for binary encoding. Creates 2^bits possible codes.
                       Default: 1 (2 codes)
            kl_loss_threshold (float): Threshold for KL divergence loss. The loss only
                                      penalizes entropy below this threshold, allowing
                                      the model to use up to this many nats of information.
                                      Default: NAT (1 bit of information)
        """
        super().__init__()

        self.bits = bits
        # Total number of possible discrete codes
        self.num_codes = 2 ** bits

        # Create power of 2 array for converting bits to indices: [1, 2, 4, 8, ...]
        power_two = 2 ** arange(bits)

        # Pre-compute all possible binary codes as a lookup table
        # Shape: (num_codes, bits) where each row is a binary representation
        codes = (arange(self.num_codes)[:, None].bitwise_and(power_two) != 0).byte().bool()

        # Register as buffers (not parameters, but part of model state)
        self.register_buffer('power_two', power_two, persistent = False)
        self.register_buffer('codes', codes, persistent = False)

        # Auxiliary loss configuration

        self.kl_loss_threshold = kl_loss_threshold
        # Zero tensor for returning when aux loss is not calculated
        self.register_buffer('zero', tensor(0.), persistent = False)

    def forward(
        self,
        logits,
        temperature = 1.,
        straight_through = None,
        calc_aux_loss = None
    ):
        """
        Map continuous logits to discrete one-hot encoded codes.

        This method performs differentiable discrete sampling using:
        1. Stochastic binary sampling based on sigmoid probabilities
        2. Straight-through gradient estimation for backpropagation
        3. Optional KL divergence loss for information regularization

        Args:
            logits (Tensor): Binary logits of shape (..., bits)
            temperature (float): Temperature for sampling. Lower values make sampling
                               more deterministic. Default: 1.0
            straight_through (bool, optional): Whether to use straight-through gradients.
                                              Defaults to self.training (True during training)
            calc_aux_loss (bool, optional): Whether to calculate KL divergence loss.
                                           Defaults to self.training (True during training)

        Returns:
            tuple: (one_hot, aux_kl_loss)
                - one_hot (Tensor): One-hot encoded discrete codes of shape (..., num_codes)
                - aux_kl_loss (Tensor): KL divergence loss (scalar), or zero if not calculated
        """
        # Default to training mode behavior if not specified
        straight_through = default(straight_through, self.training)
        calc_aux_loss = default(calc_aux_loss, self.training)

        assert logits.shape[-1] == self.bits, f'logits must have a last dimension of {self.bits}'

        # Temperature scaling and probability computation for sampling

        prob_for_sample = (logits / temperature).sigmoid()

        # Stochastic binary sampling

        # Sample each bit independently based on its probability
        sampled_bits = (torch.rand_like(logits) <= prob_for_sample).long()
        # Convert binary representation to index: e.g., [1,0,1] with bits=3 -> 5
        indices = (self.power_two * sampled_bits).sum(dim = -1)

        # Create one-hot encoding of the sampled code
        one_hot = F.one_hot(indices, self.num_codes).float()

        # Calculate auxiliary KL loss if requested

        aux_kl_loss = self.zero

        if calc_aux_loss:
            # Calculate KL divergence from maximum entropy (uniform) distribution
            # KL = max_entropy - current_entropy
            # Max entropy for 'bits' binary variables is bits * log(2) nats

            kl_div = self.bits * NAT - binary_entropy(logits)
            # Only penalize if KL divergence exceeds threshold (ReLU ensures no negative loss)
            aux_kl_loss = F.relu(kl_div - self.kl_loss_threshold).mean()

        # Apply straight-through gradient estimation if requested

        if straight_through:
            # Compute soft (differentiable) version of one-hot encoding for gradients
            # This computes P(code | logits) for each possible code
            # For each code, multiply probabilities of matching bits and (1-prob) of non-matching bits

            soft_G = (
                einsum(F.logsigmoid(logits), self.codes.float(), '... bits, codes bits -> ... codes') +
                einsum(F.logsigmoid(-logits), (~self.codes).float(), '... bits, codes bits -> ... codes')
            ).exp()

            # Straight-through trick: use hard one-hot for forward pass, soft_G for gradients
            # Gradients flow through soft_G, but forward pass uses discrete one_hot

            one_hot = one_hot + soft_G - soft_G.detach()

        return one_hot, aux_kl_loss

# classes

class FreeTransformer(Module):
    """
    Free Transformer - A transformer with discrete latent bottleneck.

    This architecture implements a unique design with:
    1. An encoder that compresses input sequences into discrete latent codes
    2. Binary quantization for creating an information bottleneck
    3. A two-stage decoder (head and tail) conditioned on the latent codes
    4. Per-token latent representations for fine-grained control

    The model learns to compress sequences through a discrete bottleneck while
    maintaining the ability to reconstruct or generate from these compressed
    representations. The information bottleneck is controlled via KL divergence
    regularization.

    Architecture flow:
    Input -> Token Embedding -> Decoder Head -> Encoder -> Binary Latents
          -> Decoder Tail (conditioned on latents) -> Output Logits
    """
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        dec_head_depth,
        dec_tail_depth,
        max_seq_len,
        enc_depth = 1,
        dim_latent = None,
        attn_dim_head = 64,
        heads = 8,
        latent_bits = 16,
        per_token_latents = True,  # they use a latent per token in the sequence, instead of one for entire sequence, iiuc
        kl_loss_threshold = NAT,
        binary_mapper_kwargs: dict = dict(),
        enc_kwargs: dict = dict(),
        dec_kwargs: dict = dict(),
        kl_loss_weight = 1.,
        latent_dropout_prob = 0.,
        pad_id = -1,
        **kwargs
    ):
        """
        Initialize the FreeTransformer.

        Args:
            num_tokens (int): Size of the vocabulary
            dim (int): Model dimension for embeddings and hidden states
            dec_head_depth (int): Number of transformer layers in the decoder head.
                                 Can be 0 to skip the decoder head entirely.
            dec_tail_depth (int): Number of transformer layers in the decoder tail.
                                 Must be > 0.
            max_seq_len (int): Maximum sequence length for positional embeddings
            enc_depth (int): Number of transformer layers in the encoder. Default: 1
            dim_latent (int, optional): Dimension for latent space. Defaults to dim.
            attn_dim_head (int): Dimension per attention head. Default: 64
            heads (int): Number of attention heads. Default: 8
            latent_bits (int): Number of bits for binary latent encoding.
                              Creates 2^latent_bits possible discrete codes. Default: 16
            per_token_latents (bool): If True, use one latent code per token in sequence.
                                     If False, use one latent code for entire sequence.
                                     Default: True
            kl_loss_threshold (float): Threshold for KL divergence loss in nats.
                                      Controls information bottleneck. Default: NAT (1 bit)
            binary_mapper_kwargs (dict): Additional kwargs for BinaryMapper. Default: {}
            enc_kwargs (dict): Additional kwargs for Encoder. Default: {}
            dec_kwargs (dict): Additional kwargs for Decoder. Default: {}
            kl_loss_weight (float): Weight for KL loss in total loss. Default: 1.0
            latent_dropout_prob (float): Dropout probability for latent codes. Default: 0.0
            pad_id (int): Token ID used for padding. Default: -1
            **kwargs: Additional arguments passed to both encoder and decoders
        """
        super().__init__()
        dim_latent = default(dim_latent, dim)

        # Token embedding and unembedding layers
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.token_unembed = nn.Linear(dim, num_tokens, bias = False)

        # Learnable query token for cross-attention to compute latents
        self.query_token_for_latents = nn.Parameter(torch.randn(dim) * 1e-2)

        self.per_token_latents = per_token_latents

        # Encoder: uses cross-attention to compress sequence into latent representations
        self.encoder = Encoder(
            dim = dim,
            depth = enc_depth,
            attn_dim_head = attn_dim_head,
            heads = heads,
            only_cross = True,  # Only cross-attention, no self-attention
            cross_attend = True,
            use_rmsnorm = True,
            rotary_pos_emb = True,
            pre_norm_has_final_norm = True,
            **kwargs,
            **enc_kwargs
        )

        # Project encoder output to binary logits
        self.to_latent_bit_logits = nn.Linear(dim, latent_bits, bias = False)

        # Binary mapper: converts logits to discrete codes
        self.binary_mapper = BinaryMapper(
            latent_bits,
            kl_loss_threshold,
            **binary_mapper_kwargs
        )

        # Project discrete latent codes back to model dimension for conditioning
        self.from_latent_to_condition = nn.Linear(self.binary_mapper.num_codes, dim, bias = False)

        # Dropout on latent representations for regularization
        self.latent_dropout = nn.Dropout(latent_dropout_prob)

        # Decoder head: optional first stage of decoding (can be depth 0)
        self.decoder_head = Decoder(
            dim = dim,
            depth = dec_head_depth,
            attn_dim_head = attn_dim_head,
            heads = heads,
            rotary_pos_emb = True,
            use_rmsnorm = True,
            pre_norm_has_final_norm = False,
            **kwargs,
            **dec_kwargs
        ) if dec_head_depth > 0 else None

        # Decoder tail must exist (depth > 0)
        assert dec_tail_depth > 0

        # Decoder tail: final stage of decoding, conditioned on latents
        self.decoder_tail = Decoder(
            dim = dim,
            depth = dec_tail_depth,
            attn_dim_head = attn_dim_head,
            heads = heads,
            rotary_pos_emb = True,
            use_rmsnorm = True,
            pre_norm_has_final_norm = True,
            **kwargs,
            **dec_kwargs
        )

        self.pad_id = pad_id
        self.kl_loss_weight = kl_loss_weight

    @property
    def device(self):
        """
        Get the device (CPU/GPU) where the model parameters are located.

        Returns:
            torch.device: Device of the model parameters
        """
        return next(self.parameters()).device

    def encode_to_latents(
        self,
        decoder_head_embeds,
        mask = None,
        return_kl_loss = False,
        per_token_latents = None
    ):
        """
        Encode sequence embeddings into discrete latent codes.

        This method compresses input embeddings through cross-attention pooling
        and binary quantization to create discrete latent representations.

        Process:
        1. Create query tokens (one per sequence or one per token)
        2. Use encoder's cross-attention to pool information from input
        3. Project to binary logits
        4. Quantize to discrete one-hot codes via BinaryMapper

        Args:
            decoder_head_embeds (Tensor): Embeddings from decoder head of shape (batch, seq_len, dim)
            mask (Tensor, optional): Attention mask of shape (batch, seq_len).
                                    True for valid positions, False for padding.
            return_kl_loss (bool): If True, return KL divergence loss. Default: False
            per_token_latents (bool, optional): If True, create one latent per token.
                                               If False, create one latent for entire sequence.
                                               Defaults to self.per_token_latents.

        Returns:
            Tensor or tuple:
                - If return_kl_loss=False: one_hot_latents of shape (batch, 1 or seq_len, num_codes)
                - If return_kl_loss=True: (one_hot_latents, kl_loss)
        """
        per_token_latents = default(per_token_latents, self.per_token_latents)

        batch, seq_len, device = *decoder_head_embeds.shape[:2], decoder_head_embeds.device

        # Create query tokens for cross-attention pooling
        query_tokens = repeat(self.query_token_for_latents, 'd -> b 1 d', b = batch)

        encoder_kwargs = dict()

        # Handle per-token latent mode as described in the paper

        if per_token_latents:
            # Create one query token per position in the sequence
            query_tokens = repeat(query_tokens, 'b 1 d -> b n d', n = seq_len)

            # Use rotary positional embeddings for both queries and context
            rotary_pos = torch.arange(seq_len, device = device)

            encoder_kwargs.update(
                pos = rotary_pos,
                context_pos = rotary_pos
            )

        # Encoder uses cross-attention to pool sequence information
        pooled = self.encoder(
            query_tokens,
            context = decoder_head_embeds,
            context_mask = mask,
            **encoder_kwargs
        )

        # Project pooled representation to binary logits
        bit_logits = self.to_latent_bit_logits(pooled)

        # Quantize to discrete one-hot codes
        one_hot_latents, kl_loss = self.binary_mapper(bit_logits, calc_aux_loss = return_kl_loss)

        if not return_kl_loss:
            return one_hot_latents

        return one_hot_latents, kl_loss

    @torch.no_grad()
    def generate(
        self,
        prompts,
        seq_len,
        latents = None,
        filter_logits_fn = top_p,
        logit_filter_kwargs: dict = dict(thres = 0.9),
        use_kv_cache = True
    ):
        """
        Autoregressively generate sequences conditioned on optional latent codes.

        This method generates tokens one at a time, optionally conditioning the
        generation on discrete latent codes. Uses KV caching for efficiency.

        Args:
            prompts (Tensor): Initial prompt tokens of shape (batch, prompt_len) or any shape
                            that can be packed to (batch, n)
            seq_len (int): Total target sequence length (including prompt)
            latents (Tensor or int, optional): Latent codes for conditioning.
                Can be:
                - None: No conditioning
                - Tensor of indices (int/long): Will be converted to one-hot
                - Tensor of one-hot codes: Used directly
                Shape can be (num_codes,), (batch, num_codes), or (batch, 1, num_codes)
            filter_logits_fn (callable): Function to filter logits before sampling.
                                        Default: top_p (nucleus sampling)
            logit_filter_kwargs (dict): Kwargs for the logit filter function.
                                       Default: {'thres': 0.9} for top_p
            use_kv_cache (bool): Whether to use KV caching for efficiency. Default: True

        Returns:
            Tensor: Generated sequences of shape matching input prompts shape,
                   with length extended to seq_len
        """
        # Pack prompts to standard shape and save inverse function
        prompts, inverse_pack = pack_with_inverse(prompts, '* n')

        batch = prompts.shape[0]

        # Prepare latent conditioning if provided

        condition = None
        if exists(latents):
            # Convert to tensor if needed
            if not is_tensor(latents):
                latents = tensor(latents, device = self.device)

            # Convert indices to one-hot if needed
            if latents.dtype in (torch.int, torch.long):
                latents = F.one_hot(latents, self.binary_mapper.num_codes).float()

            # Reshape to (batch, 1, num_codes) format
            if latents.ndim == 1:  # Single latent code
                latents = repeat(latents, 'd -> b 1 d', b = batch)
            elif latents.ndim == 2:  # Batch of latent codes
                latents = rearrange(latents, 'b d -> b 1 d')

            # Project latent codes to conditioning vectors
            condition = self.from_latent_to_condition(latents)

        # Initialize KV cache

        head_cache = tail_cache = None

        # Start with prompt tokens

        prompt_len = prompts.shape[-1]
        generated = prompts
        tokens = self.token_emb(generated)

        # Autoregressively generate remaining tokens
        for _ in range(max(0, seq_len - prompt_len)):

            # Pass through decoder head (if it exists)

            if exists(self.decoder_head):
                head_embed, next_head_cache = self.decoder_head(tokens, cache = head_cache, return_hiddens = True)
            else:
                head_embed, next_head_cache = tokens, None

            # Calculate sequence position offset for rotary embeddings when using KV cache
            # When caching, we only pass one new token but need correct positional info

            seq_pos_offset = head_cache.cache_length if exists(head_cache) else 0

            # Pass through decoder tail with latent conditioning

            tail_embed, next_tail_cache = self.decoder_tail(
                head_embed,
                cache = tail_cache,
                seq_pos_offset = seq_pos_offset,
                self_attn_kv_residuals = condition,  # Condition on latent codes
                return_hiddens = True
            )

            # Get logits for the last token only
            tail_embed = tail_embed[:, -1]
            logits = self.token_unembed(tail_embed)

            # Apply logit filtering (e.g., top-p, top-k)
            logits = filter_logits_fn(logits, **logit_filter_kwargs)

            # Sample next token using Gumbel sampling
            sampled = gumbel_sample(logits)

            # Append sampled token to generated sequence and embeddings
            generated, _ = pack((generated, sampled), 'b *')
            tokens, _ = pack((tokens, self.token_emb(sampled)), 'b * d')

            # Update KV cache for next iteration
            if use_kv_cache:
                head_cache = next_head_cache
                tail_cache = next_tail_cache

        # Restore original shape
        return inverse_pack(generated)

    def forward(
        self,
        seq,
        seq_for_latents = None,
        return_all_losses = False
    ):
        """
        Forward pass for training the FreeTransformer.

        This method implements the complete training flow:
        1. Embed input tokens
        2. Process through decoder head (if exists)
        3. Encode to discrete latent codes
        4. Decode conditioned on latents
        5. Compute reconstruction and KL losses

        The model can optionally use a separate sequence for encoding latents,
        which is useful for certain training scenarios.

        Args:
            seq (Tensor): Input token sequence of shape (batch, seq_len).
                         Will be split into input (seq[:, :-1]) and labels (seq[:, 1:])
                         for autoregressive training.
            seq_for_latents (Tensor, optional): Separate sequence to encode into latents
                                               of shape (batch, seq_len). If provided:
                                               - Latents are computed from this sequence
                                               - Per-token latents are disabled (one latent for whole seq)
                                               - Useful for conditioning on different context
                                               If None, latents are computed from the main sequence.
            return_all_losses (bool): If True, return individual loss components.
                                     If False, return only total loss. Default: False

        Returns:
            Tensor or tuple:
                - If return_all_losses=False: total_loss (scalar Tensor)
                - If return_all_losses=True: (total_loss, (ar_loss, kl_loss))
                    - total_loss: Combined loss = ar_loss + kl_loss * kl_loss_weight
                    - ar_loss: Cross-entropy autoregressive loss
                    - kl_loss: KL divergence loss from BinaryMapper
        """
        batch, device = seq.shape[0], seq.device

        # Split sequence into input and labels for autoregressive training
        seq, labels = seq[:, :-1], seq[:, 1:]

        # Embed tokens
        tokens = self.token_emb(seq)

        # Pass through decoder head (optional first decoding stage)

        if exists(self.decoder_head):
            tokens = self.decoder_head(tokens)

        # Determine sequence to use for encoding latents

        if exists(seq_for_latents):
            # Use separate sequence for latent encoding
            tokens_for_latents = self.token_emb(seq_for_latents)

            if exists(self.decoder_head):
                tokens_for_latents = self.decoder_head(tokens_for_latents)

            # Create mask: True for non-padding tokens
            encoder_mask = seq_for_latents != self.pad_id
            # When using separate sequence, use single latent for entire sequence
            per_token_latents = False
        else:
            # Use main sequence for latent encoding
            tokens_for_latents = tokens
            encoder_mask = seq != self.pad_id
            # Use default per_token_latents setting
            per_token_latents = None

        # Encode to discrete latent codes with KL loss

        latents, kl_loss = self.encode_to_latents(
            tokens_for_latents,
            mask = encoder_mask,
            per_token_latents = per_token_latents,
            return_kl_loss = True
        )

        # Apply dropout to latents for regularization
        latents = self.latent_dropout(latents)

        # Project latents to conditioning vectors
        condition = self.from_latent_to_condition(latents)

        # Pass through decoder tail conditioned on latents

        tokens = self.decoder_tail(tokens, self_attn_kv_residuals = condition)

        # Compute autoregressive cross-entropy loss

        logits = self.token_unembed(tokens)

        ar_loss = F.cross_entropy(
            rearrange(logits, 'b n l -> b l n'),
            labels,
            ignore_index = self.pad_id
        )

        # Combine losses

        total_loss = (
            ar_loss +
            kl_loss * self.kl_loss_weight
        )

        if not return_all_losses:
            return total_loss

        losses = (ar_loss, kl_loss)

        return total_loss, losses
