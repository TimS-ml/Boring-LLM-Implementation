"""
X-Transformers: A collection of Transformer variants and utilities

This package provides a comprehensive suite of transformer architectures and wrapper
utilities for various use cases. It includes:

Core Transformers:
    - XTransformer: Full encoder-decoder transformer
    - Encoder: Encoder-only transformer (e.g., BERT-style)
    - Decoder: Decoder-only transformer (e.g., GPT-style)
    - PrefixDecoder: Decoder with prefix (non-causal) attention
    - TransformerWrapper: Generic wrapper for transformer models
    - ViTransformerWrapper: Vision transformer wrapper

Attention Components:
    - Attention: Multi-head attention layer
    - CrossAttender: Cross-attention between two sequences
    - AttentionPool: Attention-based pooling

Normalization:
    - RMSNorm: Root Mean Square Layer Normalization
    - AdaptiveRMSNorm: Adaptive RMSNorm with learned scaling

Feed Forward:
    - FeedForward: Position-wise feed-forward network

Training Wrappers:
    - AutoregressiveWrapper: For autoregressive (next-token prediction) training
    - NonAutoregressiveWrapper: For non-autoregressive generation
    - XLAutoregressiveWrapper: Transformer-XL style autoregressive wrapper
    - BeliefStateWrapper: Wrapper for belief state modeling

Specialized Architectures:
    - ContinuousTransformerWrapper: For continuous (non-tokenized) inputs
    - ContinuousAutoregressiveWrapper: Autoregressive wrapper for continuous inputs
    - MultiInputTransformerWrapper: Handle multiple input modalities
    - XValTransformerWrapper: Transformer with continuous numerical values
    - XValAutoregressiveWrapper: Autoregressive wrapper for XVal
    - NeoMLP: MLP replacement using message passing via self-attention

Training Methods:
    - DPO: Direct Preference Optimization

Utilities:
    - EntropyBasedTokenizer: Tokenizer based on entropy

For detailed documentation on each component, see their respective modules.
"""

# Core transformer components
from x_transformers.x_transformers import (
    XTransformer,
    Encoder,
    Decoder,
    PrefixDecoder,
    CrossAttender,
    AttentionPool,
    Attention,
    FeedForward,
    RMSNorm,
    AdaptiveRMSNorm,
    TransformerWrapper,
    ViTransformerWrapper,
)

# Training wrappers
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper
from x_transformers.nonautoregressive_wrapper import NonAutoregressiveWrapper
from x_transformers.belief_state_wrapper import BeliefStateWrapper

# Continuous transformers
from x_transformers.continuous import (
    ContinuousTransformerWrapper,
    ContinuousAutoregressiveWrapper
)

# Multi-input transformer
from x_transformers.multi_input import MultiInputTransformerWrapper

# XVal: Transformers with continuous numerical values
from x_transformers.xval import (
    XValTransformerWrapper,
    XValAutoregressiveWrapper
)

# Transformer-XL style wrapper
from x_transformers.xl_autoregressive_wrapper import XLAutoregressiveWrapper

# Direct Preference Optimization
from x_transformers.dpo import (
    DPO
)

# NeoMLP: MLP using message passing
from x_transformers.neo_mlp import (
    NeoMLP
)

# Entropy-based tokenization
from x_transformers.entropy_based_tokenizer import EntropyBasedTokenizer
