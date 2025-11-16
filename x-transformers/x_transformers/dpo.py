"""
Direct Preference Optimization (DPO) Implementation

This module implements Direct Preference Optimization, a method for training language models
from preference data without the need for reinforcement learning. DPO directly optimizes
the policy model to increase the likelihood of preferred responses over unpreferred ones,
using a reference model to prevent the policy from deviating too far from the original behavior.

Reference: https://arxiv.org/abs/2305.18290
"""

from copy import deepcopy

import torch
from torch.nn import Module
import torch.nn.functional as F
from x_transformers.x_transformers import TransformerWrapper

import einx
from einops import rearrange

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

def freeze_all_layers_(module):
    """
    Freeze all parameters in a module by disabling gradient computation.

    This function sets requires_grad to False for all parameters in the given module,
    effectively freezing the module's weights so they won't be updated during training.
    This is typically used for reference models in DPO to keep them fixed.

    Args:
        module (torch.nn.Module): The module whose parameters should be frozen

    Note:
        The underscore suffix indicates this is an in-place operation that modifies
        the module directly.
    """
    for param in module.parameters():
        param.requires_grad = False

def log_prob_from_model_and_seq(model, seq):
    """
    Calculate log probabilities for a sequence using a language model.

    This function computes the log probability of each token in the sequence given
    the preceding context. It uses teacher forcing: the source sequence is everything
    except the last token, and the target sequence is everything except the first token.

    Args:
        model (TransformerWrapper): The language model to compute probabilities with
        seq (torch.Tensor): Input sequence of shape (batch, sequence_length)

    Returns:
        torch.Tensor: Log probabilities for each target token in the sequence,
                     shape (batch, sequence_length - 1)

    Implementation details:
        - Splits sequence into source (all but last token) and target (all but first token)
        - Gets logits from the model for the source sequence
        - Applies log softmax to convert logits to log probabilities
        - Extracts log probabilities for the actual target tokens using einx
    """
    # Split sequence: source is input tokens, target is tokens to predict
    src_seq, tgt_seq = seq[:, :-1], seq[:, 1:]

    # Get model predictions (logits) for each position
    logits = model(src_seq)

    # Convert logits to log probabilities over vocabulary
    log_prob = logits.log_softmax(dim = -1)

    # Extract log probability for the actual target token at each position
    # einx.get_at: 'b n [l], b n -> b n' means:
    #   - b: batch dimension
    #   - n: sequence dimension
    #   - [l]: vocabulary dimension (indexing into this)
    #   - Second 'b n' is the indices (tgt_seq)
    #   - Output 'b n' is the selected log probabilities
    return einx.get_at('b n [l], b n -> b n', log_prob, tgt_seq)

def masked_mean(log_probs, mask = None):
    """
    Calculate the mean of log probabilities while respecting a mask.

    This function computes the average of log probabilities, but only for positions
    where the mask is True. This is useful for ignoring padding tokens or prompt tokens
    when calculating average log probability of generated responses.

    Args:
        log_probs (torch.Tensor): Log probabilities of shape (batch, sequence_length)
        mask (torch.Tensor, optional): Boolean mask of shape (batch, sequence_length) or
                                       (batch, sequence_length + 1). True indicates positions
                                       to include in the mean. If None, computes regular mean.

    Returns:
        torch.Tensor: Masked mean of log probabilities of shape (batch,)

    Implementation details:
        - If no mask is provided, returns standard mean along sequence dimension
        - Handles mask length mismatch (adjusts if mask is 1 token longer than log_probs)
        - Sets masked-out positions to 0 before summing
        - Uses clamped denominator to avoid division by zero
    """
    # If no mask provided, compute regular mean
    if not exists(mask):
        return log_probs.mean(dim = -1)

    # Handle case where mask includes an extra token (e.g., from original sequence length)
    # Trim the mask to match log_probs length
    if mask.shape[-1] == (log_probs.shape[-1] + 1):
        mask = mask[:, :-1]

    # Zero out log probabilities at masked positions
    log_probs = log_probs.masked_fill(~mask, 0.)

    # Calculate mean: sum of log probs divided by number of valid positions
    num = log_probs.sum(dim = -1)  # Numerator: sum of valid log probs
    den = mask.sum(dim = -1)  # Denominator: count of valid positions

    # Clamp denominator to avoid division by zero
    return num / den.clamp(min = 1e-5)

def maybe_and_mask(*masks):
    """
    Combine multiple optional masks using logical AND operation.

    This function takes multiple masks (some of which may be None) and combines
    them with logical AND. If all masks are None, returns None. If only one mask
    exists, returns that mask. If multiple masks exist, returns their intersection.

    Args:
        *masks: Variable number of mask tensors (torch.Tensor or None)

    Returns:
        torch.Tensor or None: The combined mask (logical AND of all non-None masks),
                             or None if all input masks are None

    Example:
        >>> mask1 = torch.tensor([True, True, False])
        >>> mask2 = torch.tensor([True, False, False])
        >>> result = maybe_and_mask(mask1, mask2)
        >>> # result: [True, False, False]
        >>> maybe_and_mask(None, None)  # Returns None
    """
    # Filter out None masks, keeping only existing ones
    masks = [*filter(exists, masks)]

    # If no masks exist, return None
    if len(masks) == 0:
        return None

    # Start with the first mask and iteratively AND with remaining masks
    mask, *rest_masks = masks
    for rest_mask in rest_masks:
        mask = mask & rest_mask  # Element-wise logical AND

    return mask

# main class

class DPO(Module):
    """
    Direct Preference Optimization (DPO) training wrapper.

    This class implements the DPO algorithm for training language models from preference data.
    DPO trains a policy model to prefer certain responses over others while staying close to
    a frozen reference model. Unlike traditional RLHF (Reinforcement Learning from Human Feedback),
    DPO doesn't require a separate reward model or complex RL training loops.

    The core idea is to optimize the policy to maximize the log-ratio of probabilities between
    preferred and unpreferred responses, regularized by the KL divergence from a reference model.

    Key components:
        - Policy model: The model being trained (has gradients enabled)
        - Reference model: A frozen copy of the initial model (no gradients)
        - Beta: Temperature parameter controlling how much the policy can deviate from reference

    Reference:
        "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
        https://arxiv.org/abs/2305.18290

    Attributes:
        policy_model (TransformerWrapper): The trainable language model
        ref_model (TransformerWrapper): Frozen reference model (deep copy of initial policy)
        beta (float): Temperature parameter for the DPO loss
        pad_id (int or None): Token ID used for padding (used to create masks)
    """

    def __init__(
        self,
        model: TransformerWrapper,
        *,
        beta = 0.1,
        pad_id = None
    ):
        """
        Initialize the DPO training wrapper.

        Args:
            model (TransformerWrapper): The language model to train with DPO.
                                       This becomes the policy model.
            beta (float, optional): Temperature parameter that controls the strength of
                                   the KL constraint. Higher values make the policy stay
                                   closer to the reference model. Defaults to 0.1.
            pad_id (int, optional): Token ID used for padding. If provided, automatically
                                   creates masks to ignore padding tokens. Defaults to None.

        Note:
            The reference model is created as a deep copy of the input model and is
            immediately frozen (all parameters set to requires_grad=False).
        """
        super().__init__()

        # Store the trainable policy model
        self.policy_model = model

        # Create a frozen copy of the model to serve as the reference
        # This reference model will not be updated during training
        self.ref_model = deepcopy(model)
        freeze_all_layers_(self.ref_model)

        # Store hyperparameters
        self.beta = beta  # KL regularization strength
        self.pad_id = pad_id  # Padding token ID for automatic masking

    def parameters(self):
        """
        Get the trainable parameters for optimization.

        This method returns only the policy model's parameters, not the reference model's
        parameters. This is important because the reference model should remain frozen
        during training.

        Returns:
            Iterator[torch.nn.Parameter]: Iterator over the policy model's parameters

        Note:
            When using this with an optimizer, only the policy model will be updated.
            The reference model remains frozen throughout training.
        """
        return self.policy_model.parameters()

    def forward(
        self,
        preferred_seq,
        unpreferred_seq,
        *,
        prompt_mask,
        preferred_seq_mask = None,
        unpreferred_seq_mask = None,
    ):
        """
        Compute the DPO loss for a batch of preference pairs.

        This method implements the core DPO algorithm. Given pairs of preferred and unpreferred
        sequences (typically completions to the same prompt), it computes a loss that encourages
        the policy model to increase the probability of preferred responses while decreasing
        the probability of unpreferred ones, relative to the reference model.

        The DPO loss formula (per Appendix B of the paper):
            L = -log(sigmoid(beta * (log(pi_policy(y_w|x) / pi_ref(y_w|x)) -
                                      log(pi_policy(y_l|x) / pi_ref(y_l|x)))))

        Where:
            - pi_policy: policy model probabilities
            - pi_ref: reference model probabilities
            - y_w: preferred (winning) response
            - y_l: unpreferred (losing) response
            - x: prompt
            - beta: temperature parameter

        Args:
            preferred_seq (torch.Tensor): Preferred sequences of shape (batch, seq_len).
                                         Each sequence includes both prompt and completion.
            unpreferred_seq (torch.Tensor): Unpreferred sequences of shape (batch, seq_len).
                                           Must have same shape as preferred_seq.
            prompt_mask (torch.Tensor): Boolean mask indicating prompt positions (True for
                                       prompt tokens). Shape (batch, seq_len). This ensures
                                       we only compute loss on the completion, not the prompt.
            preferred_seq_mask (torch.Tensor, optional): Boolean mask for valid tokens in
                                                        preferred_seq (True for valid tokens).
                                                        If None and pad_id is set, automatically
                                                        created from pad_id.
            unpreferred_seq_mask (torch.Tensor, optional): Boolean mask for valid tokens in
                                                          unpreferred_seq (True for valid tokens).
                                                          If None and pad_id is set, automatically
                                                          created from pad_id.

        Returns:
            torch.Tensor: Scalar DPO loss averaged over the batch

        Implementation steps:
            1. Generate masks if pad_id is provided and masks are not given
            2. Compute log probabilities from both policy and reference models
            3. Apply masks to exclude prompt and padding tokens
            4. Calculate log-ratios for policy and reference models
            5. Compute DPO loss using log-sigmoid
        """
        # Validate input shapes
        assert preferred_seq.ndim == 2
        assert preferred_seq.shape == unpreferred_seq.shape

        # Automatically create masks from pad_id if provided and masks not explicitly given
        if exists(self.pad_id):
            if not exists(preferred_seq_mask):
                # Mark non-padding tokens as True
                preferred_seq_mask = preferred_seq != self.pad_id

            if not exists(unpreferred_seq_mask):
                # Mark non-padding tokens as True
                unpreferred_seq_mask = unpreferred_seq != self.pad_id

        # Following Appendix B in https://arxiv.org/abs/2305.18290
        # Compute reference model log probabilities without gradients (frozen model)

        with torch.no_grad():
            # Set reference model to eval mode to ensure consistent behavior
            self.ref_model.eval()

            # Get log probabilities for preferred sequence from reference model
            ref_preferred_logprob = log_prob_from_model_and_seq(self.ref_model, preferred_seq)

            # Get log probabilities for unpreferred sequence from reference model
            ref_unpreferred_logprob = log_prob_from_model_and_seq(self.ref_model, unpreferred_seq)

        # Compute policy model log probabilities (with gradients for training)
        policy_preferred_logprob = log_prob_from_model_and_seq(self.policy_model, preferred_seq)
        policy_unpreferred_logprob = log_prob_from_model_and_seq(self.policy_model, unpreferred_seq)

        # Compute masked mean of log probabilities
        # We need to exclude: (1) prompt tokens and (2) padding tokens
        # Only compute loss on the actual completion tokens

        # Combine prompt mask (inverted) with sequence mask to get completion-only mask
        # ~prompt_mask gives us non-prompt tokens (i.e., completion tokens)
        preferred_seq_mask = maybe_and_mask(~prompt_mask, preferred_seq_mask)
        unpreferred_seq_mask = maybe_and_mask(~prompt_mask, unpreferred_seq_mask)

        # Average log probabilities over completion tokens only (excluding prompt and padding)
        # Apply the same mask to both reference and policy log probs for fair comparison
        ref_preferred_logprob, policy_preferred_logprob = map(
            lambda t: masked_mean(t, preferred_seq_mask),
            (ref_preferred_logprob, policy_preferred_logprob)
        )
        ref_unpreferred_logprob, policy_unpreferred_logprob = map(
            lambda t: masked_mean(t, unpreferred_seq_mask),
            (ref_unpreferred_logprob, policy_unpreferred_logprob)
        )

        # Compute the main DPO loss formula
        # The core idea: maximize the difference between log-ratios of preferred vs unpreferred

        # Log-ratio of policy model: log(P_policy(preferred) / P_policy(unpreferred))
        # Equivalently: log P_policy(preferred) - log P_policy(unpreferred)
        policy_logratios = policy_preferred_logprob - policy_unpreferred_logprob

        # Log-ratio of reference model: log(P_ref(preferred) / P_ref(unpreferred))
        # Equivalently: log P_ref(preferred) - log P_ref(unpreferred)
        ref_logratios = ref_preferred_logprob - ref_unpreferred_logprob

        # DPO loss: -log(sigmoid(beta * (policy_logratios - ref_logratios)))
        # This encourages policy_logratios > ref_logratios, meaning the policy
        # should prefer the preferred response MORE than the reference model does
        # Beta controls how much we penalize deviation from the reference
        losses = -F.logsigmoid(self.beta * (policy_logratios - ref_logratios))

        # Return mean loss over the batch
        return losses.mean()
