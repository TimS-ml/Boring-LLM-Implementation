"""
Non-Autoregressive Transformer Wrapper

This module implements a non-autoregressive transformer wrapper for masked language modeling
and generation, based on MaskGIT and related approaches. Unlike autoregressive models that
generate tokens sequentially, this approach generates all tokens in parallel through an
iterative demasking process.

Key papers:
- MaskGIT: https://arxiv.org/abs/2202.04200
- BERT MLM: https://arxiv.org/abs/1904.09324
- Simple Diffusion LM: https://arxiv.org/abs/2406.07524
"""

from __future__ import annotations

import math
from random import random
from contextlib import nullcontext
from collections import namedtuple

import torch
from torch import nn, pi
from torch.nn import Module
from torch.func import grad_and_value, vmap
import torch.nn.functional as F

import einx
from einops import rearrange, repeat, pack, unpack

from x_transformers.x_transformers import TransformerWrapper

# constants

# Named tuple to store different types of losses during training
# - loss: total combined loss
# - generator_loss: loss from the generator (MLM predictions)
# - critic_loss: loss from the token critic (if used)
Losses = namedtuple('Losses', ['loss', 'generator_loss', 'critic_loss'])

# helper functions

def exists(val):
    """
    Check if a value exists (is not None).

    Args:
        val: Any value to check

    Returns:
        bool: True if val is not None, False otherwise
    """
    return val is not None

def default(val, d):
    """
    Return a default value if the given value doesn't exist.

    Args:
        val: The value to check
        d: The default value to return if val is None

    Returns:
        val if it exists, otherwise d
    """
    return val if exists(val) else d

# sampling helpers

def top_k(logits, thres = 0.9):
    """
    Filter logits to only keep the top-k values, setting others to -inf.
    This implements nucleus/top-k sampling by keeping only the most likely tokens.

    Args:
        logits (torch.Tensor): Input logits of shape (..., vocab_size)
        thres (float): Threshold value between 0 and 1. A value of 0.9 keeps the top 10% of tokens.

    Returns:
        torch.Tensor: Filtered logits with the same shape, where low-probability
                     positions are set to -inf
    """
    # Calculate k as the number of tokens to keep (complement of threshold)
    k = math.ceil((1 - thres) * logits.shape[-1])
    # Get the top-k values and their indices
    val, ind = logits.topk(k, dim = -1)
    # Create a tensor filled with -inf
    probs = torch.full_like(logits, float('-inf'))
    # Scatter the top-k values back into their positions
    probs.scatter_(2, ind, val)
    return probs

def log(t, eps = 1e-10):
    """
    Numerically stable logarithm that adds a small epsilon to prevent log(0).

    Args:
        t (torch.Tensor): Input tensor
        eps (float): Small epsilon value to add for numerical stability

    Returns:
        torch.Tensor: Natural logarithm of (t + eps)
    """
    return torch.log(t + eps)

def gumbel_noise(t):
    """
    Generate Gumbel noise for Gumbel-Softmax sampling.
    Gumbel noise follows the distribution: -log(-log(U)) where U ~ Uniform(0,1)

    Args:
        t (torch.Tensor): Template tensor to match shape and device

    Returns:
        torch.Tensor: Gumbel noise with the same shape as input
    """
    # Sample uniform noise between 0 and 1
    noise = torch.zeros_like(t).uniform_(0, 1)
    # Apply double-log transformation to get Gumbel distribution
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    """
    Sample from a categorical distribution using the Gumbel-Max trick.
    This provides a differentiable approximation to categorical sampling.

    Args:
        t (torch.Tensor): Logits or log-probabilities to sample from
        temperature (float): Temperature for sampling. Higher = more random, lower = more deterministic
        dim (int): Dimension along which to sample

    Returns:
        torch.Tensor: Sampled indices (argmax of logits + Gumbel noise)
    """
    # Add Gumbel noise to temperature-scaled logits and take argmax
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

# prob helpers

def sample_prob(prob):
    """
    Sample a boolean value with a given probability.

    Args:
        prob (float): Probability of returning True (between 0 and 1)

    Returns:
        bool: True with probability `prob`, False otherwise
    """
    return random() < prob

def coin_flip():
    """
    Simulate a fair coin flip.

    Returns:
        bool: True or False with 50% probability each
    """
    return sample_prob(0.5)

# tensor helpers

def get_mask_subset_prob(mask, prob, min_mask = 0):
    """
    Randomly select a subset of True positions from a mask based on a probability.
    This is useful for selecting which masked tokens to keep, replace, or randomize.

    Args:
        mask (torch.Tensor): Boolean mask of shape (batch, seq_len) indicating valid positions
        prob (float): Probability/fraction of masked positions to select (0 to 1)
        min_mask (int): Minimum number of positions to mask per sequence

    Returns:
        torch.Tensor: Boolean mask of shape (batch, seq_len) with a random subset of
                     True positions from the input mask

    Implementation details:
        - Uses random permutation to ensure uniform random selection
        - Accounts for padding by excluding padded positions from selection
        - Ensures at least min_mask positions are selected if available
    """
    batch, seq, device = *mask.shape, mask.device

    # Calculate how many positions to select from the mask (at least min_mask)
    num_to_mask = (mask.sum(dim = -1, keepdim = True) * prob).clamp(min = min_mask)

    # Generate random values for each position
    logits = torch.rand((batch, seq), device = device)
    # Set invalid positions (padding) to -1 so they sort to the beginning
    logits = logits.masked_fill(~mask, -1)

    # Create a random permutation by sorting twice
    # This gives us a random ranking of positions
    randperm = logits.argsort(dim = -1).argsort(dim = -1).float()

    # Adjust the permutation to account for padding positions
    num_padding = (~mask).sum(dim = -1, keepdim = True)
    randperm -= num_padding

    # Select the first num_to_mask positions from the permutation
    subset_mask = randperm < num_to_mask
    # Ensure padding positions remain False
    subset_mask.masked_fill_(~mask, False)
    return subset_mask

# schedules
# These functions define how the masking ratio changes during generation/training

def linear_schedule(t):
    """
    Linear masking schedule that decreases linearly from 1 to 0.

    Args:
        t (torch.Tensor or float): Time step(s) in range [0, 1]

    Returns:
        torch.Tensor or float: Masking ratio, where 1 means fully masked and 0 means unmasked
                              Returns (1 - t)
    """
    return 1 - t

def cosine_schedule(t):
    """
    Cosine masking schedule that decreases smoothly from 1 to 0.
    This schedule provides more gradual changes at the beginning and end.

    Reference: https://arxiv.org/abs/2202.04200 (MaskGIT paper)

    Args:
        t (torch.Tensor or float): Time step(s) in range [0, 1]

    Returns:
        torch.Tensor or float: Masking ratio using cosine curve: cos(t * π/2)
                              At t=0: returns 1 (fully masked)
                              At t=1: returns 0 (fully unmasked)
    """
    return torch.cos(t * pi / 2)

# self token critic
# inspired by Nijkamp et al. - https://aclanthology.org/2021.naacl-main.409/

class SelfCritic(Module):
    """
    Self-Critic module for evaluating token quality.

    This critic uses the same transformer network as the generator to produce
    embeddings, then projects them to a single score per token. The critic learns
    to identify which tokens are likely to be incorrect or low quality.

    Inspired by: Nijkamp et al. "CERT: Contrastive Self-supervised Learning for
    Language Understanding" - https://aclanthology.org/2021.naacl-main.409/

    Args:
        net (TransformerWrapper): The transformer network to use for generating embeddings
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

        # Get the hidden dimension from the transformer's attention layers
        dim = net.attn_layers.dim
        # Linear projection from embeddings to a single critic score per token
        self.to_logits = nn.Linear(dim, 1)

    def forward(self, x):
        """
        Forward pass to compute critic scores for each token.

        Args:
            x (torch.Tensor): Input token IDs of shape (batch, seq_len)

        Returns:
            torch.Tensor: Critic logits of shape (batch, seq_len, 1)
                         Higher scores indicate the token is more likely to be incorrect
        """
        # Get embeddings from the transformer (without generating logits)
        embed = self.net(x, return_embeddings = True)
        # Project embeddings to critic scores
        return self.to_logits(embed)

class NonAutoregressiveWrapper(Module):
    """
    Non-Autoregressive Transformer Wrapper for Masked Language Modeling and Generation.

    This wrapper enables non-autoregressive generation through iterative demasking, similar to
    MaskGIT and BERT-style masked language models. Instead of generating tokens one at a time,
    it starts with a fully masked sequence and iteratively predicts and unmasks tokens in
    parallel over multiple steps.

    Key features:
    - Iterative parallel generation through scheduled demasking
    - Self-conditioning support for improved generation quality
    - Optional token critic for better token selection during generation
    - BERT-style masking strategies (no-replace, random token replacement)
    - Configurable masking schedules (linear, cosine, or custom)

    References:
        - BERT: https://arxiv.org/abs/1904.09324
        - MaskGIT: https://arxiv.org/abs/2202.04200
        - Simple Diffusion LM: https://arxiv.org/abs/2406.07524

    Args:
        net (TransformerWrapper): The underlying transformer network
        mask_id (int): Token ID used for masking
        steps (int): Number of iterative demasking steps during generation (default: 18)
        self_cond (bool): Whether to use self-conditioning (feeding previous embeddings back)
        self_cond_train_prob (float): Probability of using self-conditioning during training (default: 0.75)
        no_replace_prob (float): Fraction of masked tokens to keep unchanged (BERT strategy, default: 0.15)
        random_token_prob (float): Fraction of masked tokens to replace with random tokens (BERT strategy, default: 0.1)
        schedule (str or callable): Masking schedule - 'linear', 'cosine', or custom function (default: 'linear')
        can_mask_prev_unmasked (bool): Whether previously unmasked tokens can be remasked (default: False)
        token_critic (TransformerWrapper, optional): Separate network for token criticism
        self_token_critic (bool): Whether to use the same network for token criticism (default: False)
        critic_loss_weight (float): Weight for the critic loss in total loss (default: 1.0)
        use_simple_mdlm_loss_weight (bool): Whether to use loss weighting from Simple Diffusion LM paper (default: True)

    Raises:
        AssertionError: If both self_token_critic and token_critic are provided
        ValueError: If an invalid schedule name is provided
    """

    def __init__(
        self,
        net,
        *,
        mask_id,
        steps = 18,
        self_cond = False,
        self_cond_train_prob = 0.75,
        no_replace_prob = 0.15,          # which percentage of the tokens masked will stay the same, done in original MLM paper
        random_token_prob = 0.1,         # which percentage of tokens to be replaced with random token, done in original MLM paper
        schedule = 'linear',
        can_mask_prev_unmasked = False,  # when unmasking, whether it can remask previously unmasked
        token_critic: TransformerWrapper | None = None,
        self_token_critic = False,
        critic_loss_weight = 1.,
        use_simple_mdlm_loss_weight = True # Sahoo et al. https://arxiv.org/abs/2406.07524
    ):
        super().__init__()
        # Cannot use both self-critic and external critic at the same time
        assert not (self_token_critic and exists(token_critic))

        # Store the underlying transformer network
        self.net = net

        # Store network dimensions and vocabulary size
        dim = net.emb_dim
        self.dim = dim
        self.num_tokens = net.num_tokens

        # Store the special mask token ID
        self.mask_id = mask_id

        # BERT-style masking strategies
        # These augmentations were not used in the MaskGIT paper but are from the original BERT
        # They may help with self-conditioning and prevent the model from relying too heavily on mask tokens
        # - no_replace: keep some tokens as their original value even when "masked"
        # - random_token: replace some masked tokens with random vocabulary tokens
        self.no_replace_prob = no_replace_prob
        self.random_token_prob = random_token_prob

        # Store sequence length and number of generation steps
        self.max_seq_len = net.max_seq_len
        self.steps = steps

        # Set up the masking schedule function
        # This determines how many tokens remain masked at each timestep
        if callable(schedule):
            self.schedule_fn = schedule
        if schedule == 'linear':
            self.schedule_fn = linear_schedule
        elif schedule == 'cosine':
            self.schedule_fn = cosine_schedule
        else:
            raise ValueError(f'invalid schedule {schedule}')

        # Set up loss weighting using the Simple Diffusion LM approach
        # This reweights losses based on the difficulty of the masking timestep
        self.loss_weight_fn = None

        if use_simple_mdlm_loss_weight:
            # Create a vectorized version that can compute gradient and value simultaneously
            grad_and_value_schedule_fn = vmap(grad_and_value(self.schedule_fn))

            # Define loss weight function based on eq (10) from Simple Diffusion LM paper
            # Weight = |d(schedule)/dt| / (1 - schedule(t))
            def loss_weight_fn(times):
                grad, value = grad_and_value_schedule_fn(times)
                return grad / (1. - value)

            self.loss_weight_fn = loss_weight_fn

        # Control whether previously unmasked tokens can be remasked during generation
        # The Simple MDLM paper chose not to allow remasking
        self.can_mask_prev_unmasked = can_mask_prev_unmasked

        # Set up self-conditioning
        # Self-conditioning feeds the model's own embeddings back as additional input
        # This can improve generation quality by allowing the model to refine its predictions
        self.self_cond = self_cond

        if self_cond:
            # Learnable null embedding used when no previous embeddings are available
            self.null_embed = nn.Parameter(torch.randn(dim))
            # Linear projection for conditioning embeddings
            self.to_self_cond = nn.Linear(dim, dim, bias = False) if self_cond else None
            # Probability of using self-conditioning during training
            self.self_cond_train_prob = self_cond_train_prob

        # Set up token critic (if used)
        # The critic evaluates which tokens are likely to be incorrect
        # and helps guide the selection of which tokens to mask in subsequent steps
        self.token_critic = token_critic

        if self_token_critic:
            # Use the same network as the generator for token criticism
            self.token_critic = SelfCritic(net)

        # Weight for the critic loss in the total training objective
        self.critic_loss_weight = critic_loss_weight

    @torch.no_grad()
    def generate(
        self,
        batch_size = None,
        start_temperature = 1.,
        filter_thres = 0.7,
        noise_level_scale = 1.,
        **kwargs
    ):
        """
        Generate sequences using iterative parallel demasking.

        This method starts with a fully masked sequence and iteratively predicts and unmasks
        tokens over multiple steps. At each step, the model predicts all tokens, samples from
        the predictions, and then masks the tokens with the lowest confidence (or highest
        critic scores) for the next iteration.

        The generation process:
        1. Start with all tokens masked
        2. For each step:
           a. Predict logits for all positions
           b. Sample tokens from the predictions
           c. Score each token (using critic or prediction confidence)
           d. Keep the best tokens, remask the rest
        3. After all steps, return the final unmasked sequence

        Args:
            batch_size (int, optional): Number of sequences to generate. If None, generates 1 sequence.
            start_temperature (float): Initial sampling temperature. Anneals to 0 over generation steps.
                                      Higher values = more random, lower = more deterministic. (default: 1.0)
            filter_thres (float): Top-k filtering threshold for logits (default: 0.7)
            noise_level_scale (float): Scale factor for Gumbel noise added to critic scores (default: 1.0)
            **kwargs: Additional arguments passed to the transformer network

        Returns:
            torch.Tensor: Generated token sequences of shape (batch, seq_len) or (seq_len,) if batch_size=None
        """
        # Determine if we're generating a single sequence or a batch
        sample_one = not exists(batch_size)
        batch_size = default(batch_size, 1)

        # Get device from model parameters
        device = next(self.net.parameters()).device

        # Store training state and switch to eval mode
        was_training = self.training
        self.eval()

        # Create timesteps from 0 to 1 for the demasking schedule
        times = torch.linspace(0., 1., self.steps + 1)

        # Initialize sequence - start with all tokens masked
        shape = (batch_size, self.max_seq_len)

        seq = torch.full(shape, self.mask_id, device = device)
        mask = torch.full(shape, True, device = device)

        # Calculate how many tokens should remain masked at each step
        # The schedule determines the masking ratio at each timestep
        all_mask_num_tokens = (self.schedule_fn(times[1:]) * self.max_seq_len).long()

        # Initialize self-conditioning with null embedding if enabled
        has_self_cond = self.self_cond
        last_embed = self.null_embed if has_self_cond else None

        # Iteratively demask tokens over multiple steps
        for mask_num_tokens, steps_until_x0 in zip(all_mask_num_tokens.tolist(), reversed(range(self.steps))):

            # Prepare self-conditioning input if enabled
            self_cond = self.to_self_cond(last_embed) if has_self_cond else None

            # Get predictions from the model
            logits, embeds = self.net(
                seq,
                sum_embeds = self_cond,
                return_logits_and_embeddings = True,
                **kwargs
            )

            # Store embeddings for next iteration's self-conditioning
            if has_self_cond:
                last_embed = embeds

            # Apply top-k filtering to logits if specified
            if exists(filter_thres):
                logits = top_k(logits, filter_thres)

            # Calculate annealed temperature (decreases as we approach final step)
            annealing_scale = steps_until_x0 / self.steps
            temperature = start_temperature * annealing_scale

            # Convert logits to probabilities (not used but computed for potential extensions)
            probs = (logits / max(temperature, 1e-3)).softmax(dim = -1)

            # Sample token IDs using Gumbel sampling with temperature
            sampled_ids = gumbel_sample(logits, temperature = max(temperature, 1e-3))

            # Update sequence: keep unmasked tokens, replace masked tokens with samples
            seq = torch.where(mask, sampled_ids, seq)

            # Compute scores for each token to decide which to mask in the next step
            if exists(self.token_critic):
                # Use token critic to score tokens (higher = more likely to be incorrect)
                scores = self.token_critic(seq)
                scores = rearrange(scores, 'b n 1 -> b n')
                # Add annealed Gumbel noise for stochastic selection
                scores = scores + noise_level_scale * gumbel_noise(scores) * annealing_scale
            else:
                # Use prediction uncertainty as scores (1 - predicted probability)
                scores = 1 - logits.softmax(dim = -1)
                # Get the score for the sampled token at each position
                scores = scores.gather(2, rearrange(sampled_ids, 'b n -> b n 1'))
                scores = rearrange(scores, 'b n 1 -> b n')

            # If we've reached the last step (no more tokens to mask), skip remasking
            if mask_num_tokens == 0:
                pass

            # Prevent remasking of previously unmasked tokens if configured
            if not self.can_mask_prev_unmasked:
                scores = scores.masked_fill(~mask, -torch.finfo(scores.dtype).max)

            # Select the top-scoring (worst) tokens to mask for the next iteration
            mask_indices = scores.topk(mask_num_tokens, dim = -1).indices
            mask = torch.zeros_like(scores, dtype = torch.bool).scatter(1, mask_indices, True)
            # Apply the mask to the sequence
            seq = seq.masked_fill(mask, self.mask_id)

        # Restore original training state
        self.train(was_training)

        # If generating a single sequence, remove batch dimension
        if sample_one:
            seq = rearrange(seq, '1 n -> n')

        return seq

    def forward(
        self,
        x,
        only_train_generator = False,
        only_train_critic = False,
        generator_sample_temperature = None,
        **kwargs
    ):
        """
        Forward pass for training the masked language model (and optional token critic).

        This method implements the training procedure:
        1. Randomly mask a subset of tokens based on a random timestep
        2. Apply BERT-style masking strategies (no-replace, random token replacement)
        3. Optionally apply self-conditioning
        4. Predict the masked tokens
        5. Compute generator loss (cross-entropy or NLL)
        6. Optionally train a token critic to identify incorrect predictions

        Args:
            x (torch.Tensor): Input token sequences of shape (batch, seq_len)
            only_train_generator (bool): If True, only compute generator loss (default: False)
            only_train_critic (bool): If True, only compute critic loss (default: False)
            generator_sample_temperature (float, optional): Temperature for sampling when training critic.
                                                           If None, uses random temperature.
            **kwargs: Additional arguments passed to the transformer network

        Returns:
            Losses: Named tuple containing:
                - loss: Total combined loss (or the relevant loss based on training mode)
                - generator_loss: MLM prediction loss (None if only_train_critic=True)
                - critic_loss: Token critic loss (None if no critic or only_train_generator=True)

        Raises:
            AssertionError: If input sequence length doesn't match max_seq_len
        """
        # Extract batch size, sequence length, and device
        b, n, device = *x.shape, x.device
        assert n == self.max_seq_len

        # Store original sequence for computing loss later
        orig_seq = x.clone()

        # Sample random timesteps for each sequence in the batch
        rand_times = torch.empty(b, device = device).uniform_(0, 1)
        # Create random permutations for selecting which tokens to mask
        batched_randperm = torch.rand((b, n), device = device).argsort(dim = -1).float()

        # Determine how many tokens to mask based on the schedule and random timestep
        rand_probs = self.schedule_fn(rand_times)
        num_tokens_mask = (rand_probs * n).clamp(min = 1.)
        # Create mask indicating which tokens to mask (True = will be masked)
        mask = batched_randperm < rearrange(num_tokens_mask, 'b -> b 1')

        # Apply BERT-style masking strategies
        # Not all masked positions will be replaced with [MASK] token
        # Some will keep their original value, some will be replaced with random tokens
        # This ensures all tokens produce embeddings and helps the model learn better representations
        replace_mask_id_mask = mask.clone()
        frac_seq_left = 1.

        # Strategy 1: Keep some masked tokens as their original value (no replacement)
        if self.no_replace_prob > 0. and coin_flip():
            frac_seq_left -= self.no_replace_prob

            # Select a random subset of masked positions to keep unchanged
            no_replace_prob_mask = get_mask_subset_prob(mask, self.no_replace_prob)
            # Remove these positions from the mask replacement set
            replace_mask_id_mask &= ~no_replace_prob_mask

        # Strategy 2: Replace some masked tokens with random vocabulary tokens
        if self.random_token_prob > 0. and coin_flip():
            # Select positions to replace with random tokens (from remaining masked positions)
            random_token_prob_mask = get_mask_subset_prob(replace_mask_id_mask, self.random_token_prob * frac_seq_left)
            # Generate random token IDs
            random_tokens = torch.randint(0, self.num_tokens, (b, n), device = device)

            # Replace selected positions with random tokens
            x = torch.where(random_token_prob_mask, random_tokens, x)
            # Remove these positions from the mask replacement set
            replace_mask_id_mask &= ~random_token_prob_mask

        # Finally, replace remaining masked positions with the [MASK] token
        masked = torch.where(replace_mask_id_mask, self.mask_id, x)

        # Apply self-conditioning if enabled
        # Self-conditioning provides the model with its own previous predictions as additional context
        if self.self_cond:
            # Start with null embedding
            self_cond = self.null_embed

            # With some probability, compute actual self-conditioning embeddings
            if sample_prob(self.self_cond_train_prob):
                with torch.no_grad():
                    # Get embeddings from a forward pass (without gradients)
                    self_cond = self.net(masked, return_embeddings = True, **kwargs).detach()

            # Add self-conditioning embeddings to the model input
            kwargs.update(sum_embeds = self.to_self_cond(self_cond))

        # Forward pass through the network to get logits
        # If only training critic, don't compute gradients for the generator
        context = torch.no_grad if only_train_critic else nullcontext

        with context():
            logits = self.net(masked, **kwargs)

        # Select appropriate loss function based on model output type
        loss_fn = F.cross_entropy if not self.net.output_is_log_prob else F.nll_loss

        # Compute generator loss
        if exists(self.loss_weight_fn):
            # Use the Simple Diffusion LM loss weighting scheme
            # This reweights losses based on the difficulty of the timestep

            # Compute per-token losses without reduction
            loss = loss_fn(
                rearrange(logits, 'b n l -> b l n'),
                orig_seq,
                reduction = 'none'
            )

            # Calculate loss weights based on the timestep
            loss_weights = self.loss_weight_fn(rand_times)
            # Apply loss weights to each sequence in the batch
            loss = einx.multiply('b n, b', loss, loss_weights)

            # Average loss only over masked positions
            loss = loss[mask].mean()

        else:
            # Standard cross-entropy loss over masked positions only
            loss = loss_fn(
                logits[mask],
                orig_seq[mask],
            )

        # If no critic is being used or only training generator, return early
        if not exists(self.token_critic) or only_train_generator:
            return Losses(loss, loss, None)

        # Train token critic
        # Sample from the generator's predictions using Gumbel sampling
        sampled_ids = gumbel_sample(logits, temperature = default(generator_sample_temperature, random()))
        # Create generated sequence by combining samples (at masked positions) with original tokens
        generated = torch.where(mask, sampled_ids, orig_seq)

        # Get critic scores for the generated sequence
        critic_logits = self.token_critic(generated)
        # Labels: 1 if token is incorrect, 0 if correct
        critic_labels = (sampled_ids != orig_seq).float()

        # Compute binary cross-entropy loss for the critic
        critic_loss = F.binary_cross_entropy_with_logits(
            rearrange(critic_logits, '... 1 -> ...'),
            critic_labels
        )

        # Combine losses based on training mode
        if only_train_critic:
            # Only train critic: use critic loss only, set generator loss to None
            total_loss = critic_loss
            loss = None
        else:
            # Train both: combine generator and critic losses
            total_loss = loss + critic_loss * self.critic_loss_weight

        return Losses(total_loss, loss,  critic_loss)
