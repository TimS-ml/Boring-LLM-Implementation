"""
Hyperparameter Configuration Sets for Transformer Training Experiments

This module defines predefined sets of hyperparameters for training transformer models
of different sizes. The parameter sets are organized by model size and provide various
combinations of batch sizes, learning rates, and gradient accumulation settings for
systematic hyperparameter tuning.

The configurations are designed to:
1. Explore different effective batch sizes (batch_size × gradient_accumulate_every)
2. Test learning rates appropriate for each batch size (larger batches typically need higher LR)
3. Enable systematic comparison of training configurations

Usage:
    Import this module in training scripts and select a configuration by index:
    >>> from param_set import PARAM_SETS_BATCH_AND_LR_19M
    >>> config = PARAM_SETS_BATCH_AND_LR_19M[0]
    >>> batch_size = config['batch_size']
"""

# ========================================
# Parameter Sets for 19M Parameter Model
# ========================================
# These configurations are optimized for the 19M parameter model
# (512 dim, 6 layers, 8 heads) trained on 380M tokens

PARAM_SETS_BATCH_AND_LR_19M = [
    # Index 0: Small batch, low learning rate baseline
    # Effective batch size: 4 × 4 = 16 sequences
    {
        'batch_size': 4,                    # Number of sequences per micro-batch
        'gradient_accumulate_every': 4,     # Number of micro-batches to accumulate before update
        'learning_rate': 5e-5               # Conservative learning rate for stability
    },

    # Index 1: Small batch, medium learning rate
    # Effective batch size: 4 × 4 = 16 sequences
    {
        'batch_size': 4,
        'gradient_accumulate_every': 4,
        'learning_rate': 1e-4               # Standard learning rate for small models
    },

    # Index 2: Small batch, high learning rate
    # Effective batch size: 4 × 4 = 16 sequences
    {
        'batch_size': 4,
        'gradient_accumulate_every': 4,
        'learning_rate': 3e-4               # Aggressive learning rate
    },

    # Index 3: Medium batch, low learning rate
    # Effective batch size: 8 × 4 = 32 sequences
    {
        'batch_size': 8,
        'gradient_accumulate_every': 4,
        'learning_rate': 5e-5
    },

    # Index 4: Medium batch, medium learning rate
    # Effective batch size: 8 × 4 = 32 sequences
    {
        'batch_size': 8,
        'gradient_accumulate_every': 4,
        'learning_rate': 1e-4
    },

    # Index 5: Medium batch, high learning rate
    # Effective batch size: 8 × 4 = 32 sequences
    {
        'batch_size': 8,
        'gradient_accumulate_every': 4,
        'learning_rate': 3e-4
    },

    # Index 6: Large batch, medium learning rate
    # Effective batch size: 16 × 4 = 64 sequences
    {
        'batch_size': 16,
        'gradient_accumulate_every': 4,
        'learning_rate': 1e-4
    },

    # Index 7: Large batch, medium-high learning rate
    # Effective batch size: 16 × 4 = 64 sequences
    {
        'batch_size': 16,
        'gradient_accumulate_every': 4,
        'learning_rate': 2e-4               # Scaled up for larger batch
    },

    # Index 8: Large batch, high learning rate
    # Effective batch size: 16 × 4 = 64 sequences
    {
        'batch_size': 16,
        'gradient_accumulate_every': 4,
        'learning_rate': 5e-4
    },

    # Index 9: Very large batch, medium-high learning rate
    # Effective batch size: 32 × 4 = 128 sequences
    {
        'batch_size': 32,
        'gradient_accumulate_every': 4,
        'learning_rate': 2e-4
    },

    # Index 10: Very large batch, high learning rate
    # Effective batch size: 32 × 4 = 128 sequences
    {
        'batch_size': 32,
        'gradient_accumulate_every': 4,
        'learning_rate': 4e-4               # Higher LR for large batch training
    },

    # Index 11: Maximum batch, high learning rate
    # Effective batch size: 64 × 4 = 256 sequences
    {
        'batch_size': 64,
        'gradient_accumulate_every': 4,
        'learning_rate': 3e-4
    }
]

# ========================================
# Parameter Sets for 64M Parameter Model
# ========================================
# These configurations are optimized for the 64M parameter model
# (640 dim, 12 layers, 10 heads) trained on 1.28B tokens
#
# Larger models typically require:
# - Smaller learning rates (to prevent instability)
# - More gradient accumulation (due to memory constraints)
# - Longer training (more tokens to see)

PARAM_SETS_BATCH_AND_LR_64M = [
    # Index 0: Conservative baseline - very small batch, low LR
    # Effective batch size: 2 × 8 = 16 sequences
    # Good for initial stability testing with larger models
    {
        'batch_size': 2,                    # Smallest micro-batch (memory constrained)
        'gradient_accumulate_every': 8,     # High accumulation to achieve reasonable batch size
        'learning_rate': 5e-5               # Conservative LR for large model
    },

    # Index 1: Small batch, medium learning rate
    # Effective batch size: 2 × 8 = 16 sequences
    {
        'batch_size': 2,
        'gradient_accumulate_every': 8,
        'learning_rate': 1e-4               # Standard LR, may be aggressive for 64M model
    },

    # Index 2: Small batch, very low learning rate
    # Effective batch size: 2 × 8 = 16 sequences
    {
        'batch_size': 2,
        'gradient_accumulate_every': 8,
        'learning_rate': 3e-5               # Very conservative for maximum stability
    },

    # Index 3: Medium batch, low learning rate
    # Effective batch size: 4 × 4 = 16 sequences
    {
        'batch_size': 4,
        'gradient_accumulate_every': 4,
        'learning_rate': 5e-5
    },

    # Index 4: Medium batch, medium learning rate
    # Effective batch size: 4 × 4 = 16 sequences
    {
        'batch_size': 4,
        'gradient_accumulate_every': 4,
        'learning_rate': 1e-4
    },

    # Index 5: Medium batch with high accumulation
    # Effective batch size: 4 × 8 = 32 sequences
    {
        'batch_size': 4,
        'gradient_accumulate_every': 8,     # Higher accumulation for larger effective batch
        'learning_rate': 7e-5               # Intermediate LR
    },

    # Index 6: Larger batch with minimal accumulation
    # Effective batch size: 8 × 2 = 16 sequences
    {
        'batch_size': 8,
        'gradient_accumulate_every': 2,     # Minimal accumulation if memory allows
        'learning_rate': 5e-5
    },

    # Index 7: Larger batch, balanced accumulation
    # Effective batch size: 8 × 4 = 32 sequences
    {
        'batch_size': 8,
        'gradient_accumulate_every': 4,
        'learning_rate': 7e-5               # Slightly higher LR for larger batch
    },

    # Index 8: Larger batch, high accumulation
    # Effective batch size: 8 × 8 = 64 sequences
    {
        'batch_size': 8,
        'gradient_accumulate_every': 8,
        'learning_rate': 1e-4               # Higher LR scaled for large batch
    },

    # Index 9: Large batch, minimal accumulation
    # Effective batch size: 16 × 2 = 32 sequences
    {
        'batch_size': 16,
        'gradient_accumulate_every': 2,
        'learning_rate': 7e-5
    },

    # Index 10: Large batch, balanced accumulation
    # Effective batch size: 16 × 4 = 64 sequences
    # Recommended for stable 64M model training
    {
        'batch_size': 16,
        'gradient_accumulate_every': 4,
        'learning_rate': 1e-4
    }
]