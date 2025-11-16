"""
GPT-VAE: Conditional Variational Autoencoder for GPT-style Transformers

This module implements a Conditional Variational Autoencoder (CVAE) combined with a GPT-style
decoder, inspired by the CVAE + DETR design from ACT (Zhou et al.). This architecture enables:
- Controlled generation through latent variable steering
- Diverse output generation for the same input
- Applications in diversity reinforcement learning (RLVR), MAP-Elites in EPO, and more

The model consists of:
1. An encoder that compresses input sequences into latent distributions
2. A VAE framework that samples from the latent space using the reparameterization trick
3. A GPT-style decoder that generates sequences conditioned on the latent variables
"""

from __future__ import annotations

# applying the cvae + detr design from ACT (Zhou et al.) to GPT
# for steering, diversity rlvr, map-elites in epo, and other possibilities

import torch
from torch import nn, Tensor, is_tensor, tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from x_transformers.x_transformers import (
    Encoder,
    Decoder,
    TransformerWrapper
)

from x_transformers.autoregressive_wrapper import AutoregressiveWrapper

from einops.layers.torch import Rearrange
from einops import rearrange, reduce, repeat

# helper functions

def exists(v):
    """
    Check if a value exists (is not None).

    Args:
        v: The value to check

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

# classes

class GPTVAE(Module):
    """
    GPT-based Variational Autoencoder (GPT-VAE).

    This class implements a conditional variational autoencoder that combines a transformer
    encoder with a GPT-style decoder. The encoder compresses input sequences into a latent
    distribution, and the decoder generates sequences conditioned on samples from this latent
    space. This enables controllable and diverse text generation.

    The architecture follows the CVAE + DETR design from ACT (Zhou et al.), adapted for
    autoregressive language modeling tasks.

    Attributes:
        encoder (Module): Transformer encoder that maps sequences to latent representations
        decoder (TransformerWrapper): GPT-style decoder for autoregressive generation
        ar_wrapped_decoder (AutoregressiveWrapper): Autoregressive wrapper around the decoder
        to_latent_mean_log_variance (nn.Sequential): Projects encoder output to mean and log-variance
        from_latent_to_prepend_token (nn.Sequential): Projects latent samples to decoder input
        pad_id (int): Token ID used for padding
        vae_kl_div_floor (float): Floor value for KL divergence (prevents posterior collapse)
        vae_kl_loss_weight (float): Weight for the KL divergence loss term
        latents_dropout (nn.Dropout): Dropout layer for latent variables during training
    """

    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        enc_depth,
        max_seq_len,
        dim_latent = None,
        attn_dim_head = 64,
        heads = 8,
        enc_kwargs: dict = dict(),
        dec_kwargs: dict = dict(),
        vae_kl_loss_weight = 1.,
        vae_kl_div_floor = 0.,      # what was done in free transformer, which in turn came from Kingma 2016
        latents_dropout_prob = 0.5, # what percentage of the time to dropout the latents completely
        pad_id = -1,
        encoder: Module | None = None,
        **kwargs
    ):
        """
        Initialize the GPT-VAE model.

        Args:
            num_tokens (int): Size of the vocabulary (number of unique tokens)
            dim (int): Dimension of the model embeddings and hidden states
            depth (int): Number of decoder transformer layers
            enc_depth (int): Number of encoder transformer layers
            max_seq_len (int): Maximum sequence length for both encoder and decoder
            dim_latent (int, optional): Dimension of the latent space. Defaults to dim if not specified
            attn_dim_head (int, optional): Dimension of each attention head. Defaults to 64
            heads (int, optional): Number of attention heads. Defaults to 8
            enc_kwargs (dict, optional): Additional keyword arguments for the encoder. Defaults to empty dict
            dec_kwargs (dict, optional): Additional keyword arguments for the decoder. Defaults to empty dict
            vae_kl_loss_weight (float, optional): Weight for the KL divergence loss term. Defaults to 1.0
            vae_kl_div_floor (float, optional): Floor value for KL divergence to prevent posterior collapse,
                following the approach from Kingma 2016 and Free Transformer. Defaults to 0.0
            latents_dropout_prob (float, optional): Probability of completely dropping out the latent
                conditioning during training (forces model to learn unconditional generation). Defaults to 0.5
            pad_id (int, optional): Token ID used for padding sequences. Defaults to -1
            encoder (Module, optional): Custom encoder module. If None, a default TransformerWrapper
                encoder will be created. Defaults to None
            **kwargs: Additional keyword arguments passed to both encoder and decoder attention layers
        """
        super().__init__()
        # Set latent dimension to model dimension if not specified
        dim_latent = default(dim_latent, dim)

        # Create default encoder if not provided
        if not exists(encoder):
            # Encoder uses TransformerWrapper with average pooling to produce a single vector per sequence
            encoder = TransformerWrapper(
                num_tokens = num_tokens,
                max_seq_len = max_seq_len + 1,  # +1 to accommodate the prepended latent token
                return_only_embed = True,        # Return embeddings instead of logits
                average_pool_embed = True,       # Average pool over sequence length to get a single vector
                attn_layers = Encoder(
                    dim = dim,
                    depth = enc_depth,
                    attn_dim_head = attn_dim_head,
                    heads = heads,
                    **kwargs,
                    **enc_kwargs
                ),
            )

        self.encoder = encoder

        # Project encoder output to latent mean and log-variance for VAE reparameterization
        # Output shape: [2, batch, dim_latent] where first element is mean, second is log-variance
        self.to_latent_mean_log_variance = nn.Sequential(
            nn.Linear(dim, dim_latent * 2),  # Project to twice the latent dimension
            Rearrange('b (two d) -> two b d', two = 2)  # Split into mean and log-variance
        )

        # Project sampled latent vector back to model dimension and add sequence dimension
        # This latent token will be prepended to the decoder input
        self.from_latent_to_prepend_token = nn.Sequential(
            nn.Linear(dim_latent, dim),  # Project latent to model dimension
            Rearrange('b d -> b 1 d')    # Add sequence dimension (batch, 1, dim)
        )

        # Create GPT-style decoder for autoregressive generation
        self.decoder = TransformerWrapper(
            num_tokens = num_tokens,
            max_seq_len = max_seq_len,
            attn_layers = Decoder(
                dim = dim,
                depth = depth,
                attn_dim_head = attn_dim_head,
                heads = heads,
                **kwargs,
                **dec_kwargs
            ),
        )

        # Wrap decoder with autoregressive capabilities (teacher forcing during training)
        self.ar_wrapped_decoder = AutoregressiveWrapper(self.decoder, ignore_index = pad_id)

        self.pad_id = pad_id

        # loss weights - vae kl loss

        # KL divergence floor prevents posterior collapse (technique from Kingma 2016)
        self.vae_kl_div_floor = vae_kl_div_floor
        # Weight for balancing reconstruction loss and KL divergence
        self.vae_kl_loss_weight = vae_kl_loss_weight

        # Dropout layer for randomly removing latent conditioning during training
        # This ensures the model can generate unconditionally as well
        self.latents_dropout = nn.Dropout(latents_dropout_prob)

    @property
    def device(self):
        """
        Get the device (CPU/GPU) where the model parameters are stored.

        Returns:
            torch.device: The device of the model's parameters
        """
        return next(self.parameters()).device

    def encode_to_latents(
        self,
        seq,
        return_mean_log_var = False
    ):
        """
        Encode input sequences to latent representations using the VAE encoder.

        This method performs the encoding step of the VAE:
        1. Encodes the input sequence to a pooled representation
        2. Projects to mean and log-variance of the latent distribution
        3. Samples from the distribution using the reparameterization trick

        Args:
            seq (Tensor): Input sequence tensor of shape (batch, seq_len) containing token IDs
            return_mean_log_var (bool, optional): If True, also returns the mean and log-variance
                of the latent distribution. Defaults to False

        Returns:
            If return_mean_log_var is False:
                Tensor: Sampled latent vectors of shape (batch, dim_latent)
            If return_mean_log_var is True:
                tuple: (latents, (latents_mean, latents_log_var)) where:
                    - latents: Sampled latent vectors of shape (batch, dim_latent)
                    - latents_mean: Mean of latent distribution of shape (batch, dim_latent)
                    - latents_log_var: Log-variance of latent distribution of shape (batch, dim_latent)
        """
        # Create mask to ignore padding tokens during encoding
        mask = seq != self.pad_id
        # Encode sequence to a pooled representation (single vector per sequence)
        pooled = self.encoder(seq, mask = mask)

        # Project pooled representation to latent mean and log-variance
        latents_mean, latents_log_var = self.to_latent_mean_log_variance(pooled)
        # Convert log-variance to standard deviation: std = exp(0.5 * log_var) = sqrt(var)
        latents_std = (0.5 * latents_log_var).exp()

        # Reparameterization trick: sample from N(mean, std) as mean + std * N(0, 1)
        # This allows gradients to flow through the sampling operation
        latents = latents_mean + latents_std * torch.randn_like(latents_mean)

        # Return only sampled latents if mean and log-variance not requested
        if not return_mean_log_var:
            return latents

        # Return sampled latents along with distribution parameters (needed for KL loss)
        return latents, (latents_mean, latents_log_var)

    @torch.no_grad()
    def generate(
        self,
        prompts,
        seq_len,
        latents = None,
        seq_for_latents = None,
        **generate_kwargs
    ):
        """
        Generate sequences autoregressively conditioned on optional latent variables.

        This method performs inference (no gradient computation) to generate new sequences.
        The generation can be conditioned on latent variables either by:
        1. Providing pre-computed latents directly
        2. Providing a sequence from which to derive latents
        3. Generating unconditionally (no latents)

        Args:
            prompts (Tensor): Starting tokens for generation. Can be 1D (single sequence) or
                2D (batch of sequences) with shape (batch, prompt_len) or (prompt_len,)
            seq_len (int): Number of new tokens to generate (beyond the prompt)
            latents (Tensor or list, optional): Pre-computed latent vectors. Can be:
                - 1D tensor of shape (dim_latent,): same latent for all sequences in batch
                - 2D tensor of shape (batch, dim_latent): different latent per sequence
                - list/array: will be converted to tensor
                Defaults to None (unconditional generation)
            seq_for_latents (Tensor, optional): Sequence to encode into latents. If provided,
                latents will be computed from this sequence. Cannot be used together with
                the latents parameter. Defaults to None
            **generate_kwargs: Additional keyword arguments passed to the autoregressive
                generation function (e.g., temperature, top_k, top_p)

        Returns:
            Tensor: Generated sequences of shape (batch, prompt_len + seq_len) containing
                both the original prompts and newly generated tokens

        Raises:
            AssertionError: If both latents and seq_for_latents are provided, or if
                prompts has invalid dimensions
        """
        # Validate input dimensions and determine batch size
        assert prompts.ndim in {1, 2}
        batch = prompts.shape[0] if prompts.ndim == 2 else 1

        # if seq_for_latents passed in, derive latents from it

        if exists(seq_for_latents):
            # Ensure latents are not provided both directly and via seq_for_latents
            assert not exists(latents), 'latents should not be passed in if given the seq from which to derive them'

            # Encode the provided sequence to obtain latent vectors
            latents = self.encode_to_latents(seq_for_latents)

        # prepend embeds

        prepend_embeds = None
        if exists(latents):
            # Convert latents to tensor if provided as list or numpy array
            if not is_tensor(latents):
                latents = tensor(latents, device = self.device)

            # If latents is 1D, repeat it for all sequences in the batch
            if latents.ndim == 1:
                latents = repeat(latents, 'd -> b d', b = batch)

            # Project latents to decoder embedding space and add sequence dimension
            prepend_embeds = self.from_latent_to_prepend_token(latents)

        # Generate sequences autoregressively

        # The prepend_embeds will be concatenated to the beginning of the decoder input,
        # conditioning the generation on the latent variables
        generated = self.ar_wrapped_decoder.generate(
            prompts,
            seq_len,
            prepend_embeds = prepend_embeds,
            **generate_kwargs
        )

        return generated

    def forward(
        self,
        seq,
        seq_for_latents = None,
        return_all_losses = False
    ):
        """
        Forward pass for training the GPT-VAE model.

        This method computes the total loss for training, which consists of:
        1. Reconstruction loss (autoregressive loss): measures how well the decoder
           reconstructs the input sequence given the latent variables
        2. KL divergence loss: regularizes the latent distribution to be close to
           a standard normal distribution N(0, 1)

        During training, latents are randomly dropped out (according to latents_dropout_prob)
        to ensure the model can generate both conditionally and unconditionally.

        Args:
            seq (Tensor): Input/target sequence tensor of shape (batch, seq_len) containing
                token IDs. This is the sequence to be reconstructed by the decoder
            seq_for_latents (Tensor, optional): Sequence to encode into latents. If None,
                uses the same sequence as seq. This allows for training on different
                sequences for encoding and decoding (e.g., for style transfer). Defaults to None
            return_all_losses (bool, optional): If True, returns individual loss components
                in addition to the total loss. Defaults to False

        Returns:
            If return_all_losses is False:
                Tensor: Total loss (scalar) = reconstruction_loss + kl_weight * kl_loss
            If return_all_losses is True:
                tuple: (total_loss, (reconstruction_loss, kl_loss)) where:
                    - total_loss: Weighted sum of all losses
                    - reconstruction_loss: Autoregressive cross-entropy loss
                    - kl_loss: KL divergence between latent distribution and N(0, 1)
        """
        # Extract batch size and device from input sequence
        batch, device = seq.shape[0], seq.device

        # Use provided sequence for latents, or default to the same sequence being decoded
        seq_for_latents = default(seq_for_latents, seq)

        # Encode sequence to latent distribution and sample from it
        # Returns both sampled latents and distribution parameters (mean, log-variance)
        latents, (latents_mean, latents_log_var) = self.encode_to_latents(seq_for_latents, return_mean_log_var = True)

        # Randomly dropout latents for some samples in the batch
        # Creates a boolean mask: True means latents are dropped (unconditional generation)
        # The dropout layer outputs 0s with probability latents_dropout_prob, so we invert with ~
        dropped_latents = ~self.latents_dropout(torch.ones((batch,), device = device)).bool()

        # Project sampled latents to decoder embedding space
        prepend_embeds = self.from_latent_to_prepend_token(latents)

        # Compute reconstruction loss (cross-entropy) using teacher forcing
        # seq_start_pos indicates where the actual sequence starts:
        # - If latents dropped (True=1): sequence starts at position 1, skipping prepended latent
        # - If latents kept (False=0): sequence starts at position 0, attending to prepended latent
        ar_loss = self.ar_wrapped_decoder(
            seq,
            prepend_embeds = prepend_embeds,
            seq_start_pos = dropped_latents.long()  # sequence starts at 1 and does not attend to the first style latent
        )

        # vae kl loss

        # Compute KL divergence between learned distribution q(z|x) and prior p(z) = N(0, 1)
        # KL(q||p) = 0.5 * sum(exp(log_var) + mean^2 - log_var - 1)
        # Derived from KL divergence formula for Gaussian distributions
        vae_kl_loss = 0.5 * (
            latents_log_var.exp()     # variance term
            + latents_mean.square()   # mean term
            - latents_log_var         # entropy term
            - 1.                      # normalizing constant
        )

        # Apply floor to KL divergence to prevent posterior collapse
        # ReLU ensures KL loss is non-negative after applying floor
        # This technique is from Kingma 2016 and used in Free Transformer
        vae_kl_loss = F.relu(vae_kl_loss - self.vae_kl_div_floor)

        # Sum over latent dimensions and average over batch
        vae_kl_loss = vae_kl_loss.sum(dim = -1).mean()

        # return losses

        # Compute total loss as weighted sum of reconstruction and KL divergence
        total_loss = (
            ar_loss +
            vae_kl_loss * self.vae_kl_loss_weight
        )

        # Return only total loss if individual losses not requested
        if not return_all_losses:
            return total_loss

        # Return both total loss and individual loss components
        losses = (ar_loss, vae_kl_loss)

        return total_loss, losses
