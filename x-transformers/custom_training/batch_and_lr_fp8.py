"""
Transformer Training Script with FP8 Precision and Batch Size/Learning Rate Experimentation

This module provides a comprehensive training script for transformer language models
with FP8 (8-bit floating point) precision support using NVIDIA Transformer Engine.

Features:
- Multiple batch size and learning rate configurations
- FP8 mixed precision training for maximum performance on modern GPUs (H100, etc.)
- Checkpoint resumption and configuration recovery
- WandB experiment tracking and logging
- Training on enwik9 dataset (character-level language modeling)
- Gradient accumulation for effective large batch training
- Throughput and optimizer statistics monitoring

FP8 Training:
- Uses Transformer Engine's fp8_autocast for automatic FP8 precision
- Hybrid format: E4M3 for forward pass, E5M2 for backward pass
- Delayed scaling with dynamic amax tracking for numerical stability

The script supports flexible training scenarios:
1. Starting new training with a specific configuration (RUN parameter)
2. Resuming from checkpoint with original configuration (RUN_NAME parameter)
3. Resuming from checkpoint but overriding with new configuration (both parameters)

Usage:
    Set environment variables or context vars for RUN and/or RUN_NAME to control training.
"""

from x_transformers import TransformerWrapper, Decoder
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper

import os
import time
import random
import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# Transformer Engine for FP8 training
# Install with: pip install --no-build-isolation transformer_engine[pytorch]
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

import wandb
import numpy as np
from datetime import datetime

from boring_utils.utils import cprint, tprint
from boring_utils.nn_utils import (
    cycle, resume_checkpoint,
    calculate_optimizer_stats, calculate_throughput
)
from boring_utils.helpers import DEBUG, ContextVar

from param_set import *

# ========================================
# Context Variables for Training Configuration
# ========================================

# RUN: Index into PARAM_SETS_BATCH_AND_LR to select a specific hyperparameter configuration
# Set this to start new training or override checkpoint config
RUN = ContextVar("RUN", None)

# RUN_NAME: Name of checkpoint directory to resume from
# Set this to resume training from a previous checkpoint
RUN_NAME = ContextVar("RUN_NAME", None)

# SIZE: Model size selector (0 = 19M parameters, 1 = 64M parameters)
# Controls model architecture and training token budget
SIZE = ContextVar("SIZE", 0)

# ========================================
# Model Size Configuration
# ========================================

# Suffix for run naming to distinguish model sizes and precision
post_fix = ""

if SIZE.value == 0:
    # Small model configuration: 19M parameters
    PARAM_SETS_BATCH_AND_LR = PARAM_SETS_BATCH_AND_LR_19M  # Parameter sweep configurations
    PROJECT_NAME = "x-transformers-tuning-practice_19M"    # WandB project name
    post_fix += "_19M"
    MODEL_DIM = 512           # Model hidden dimension
    MODEL_DEPTH = 6           # Number of transformer layers
    MODEL_HEADS = 8           # Number of attention heads
    TOTAL_TRAINING_TOKENS = 380e6  # Total tokens to train on (380 million)

elif SIZE.value == 1:
    # Medium model configuration: 64M parameters
    PARAM_SETS_BATCH_AND_LR = PARAM_SETS_BATCH_AND_LR_64M  # Parameter sweep configurations
    PROJECT_NAME = "x-transformers-tuning-practice"        # WandB project name
    post_fix += "_64M"
    MODEL_DIM = 640           # Model hidden dimension
    MODEL_DEPTH = 12          # Number of transformer layers
    MODEL_HEADS = 10          # Number of attention heads
    TOTAL_TRAINING_TOKENS = 1280e6  # Total tokens to train on (1.28 billion)

# ========================================
# Configuration Loading Logic
# ========================================

# Setup directory paths for checkpoints and data
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory containing this script
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')            # Project root directory
CHECKPOINTS_BASE_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')  # Base directory for all checkpoints

def load_run_config_from_checkpoint_or_wandb(checkpoint_file):
    """
    Load training configuration from checkpoint file or WandB API.

    This function attempts to recover the training hyperparameters (batch size,
    learning rate, gradient accumulation) from a checkpoint. It tries three sources
    in order:
    1. The checkpoint file itself (if 'training_config' is saved)
    2. WandB API (by matching the run name from the checkpoint directory)
    3. Default values (fallback if all else fails)

    Args:
        checkpoint_file (str): Path to the checkpoint file (.pt)

    Returns:
        dict: Dictionary containing 'batch_size', 'learning_rate', and
              'gradient_accumulate_every'

    Note:
        This function is crucial for resuming training with the exact same
        hyperparameters that were used originally.
    """
    # Load the checkpoint file
    checkpoint = torch.load(checkpoint_file)

    # Try checkpoint first - this is the most reliable source
    if 'training_config' in checkpoint:
        config = checkpoint['training_config']
        print(f"Loaded config from checkpoint: batch_size={config.get('batch_size')}, lr={config.get('learning_rate')}, grad_accum={config.get('gradient_accumulate_every')}")
        return config

    # Fallback to wandb API if checkpoint doesn't contain config
    try:
        api = wandb.Api()
        # Extract run name from checkpoint directory
        checkpoint_dir = os.path.dirname(checkpoint_file)
        wandb_run_name = os.path.basename(checkpoint_dir)

        # Try to find the wandb run by display name or ID
        wandb_run = None
        # First attempt: search by display name
        runs = api.runs("x-transformers-tuning-practice",
                      filters={"display_name": wandb_run_name})
        runs_list = list(runs)

        if len(runs_list) > 0:
            wandb_run = runs_list[0]
            print(f"Found wandb run by display name: {wandb_run_name}")
        else:
            # Second attempt: try using run name as ID directly
            try:
                wandb_run = api.run(f"x-transformers-tuning-practice/{wandb_run_name}")
                print(f"Found wandb run by ID: {wandb_run_name}")
            except:
                pass

        # Extract config from wandb run if found
        if wandb_run:
            wandb_config = wandb_run.config
            config = {
                'batch_size': wandb_config.get('batch_size', 4),
                'learning_rate': wandb_config.get('learning_rate', 1e-4),
                'gradient_accumulate_every': wandb_config.get('gradient_accumulate_every', 1)
            }
            print(f"Loaded config from wandb: batch_size={config['batch_size']}, lr={config['learning_rate']}, grad_accum={config['gradient_accumulate_every']}")
            return config
    except Exception as e:
        print(f"Failed to load config from wandb: {e}")

    # Return default config if all else fails
    print("Using default config")
    return {
        'batch_size': 4,
        'learning_rate': 1e-4,
        'gradient_accumulate_every': 1
    }

# ========================================
# Configuration Loading Scenarios
# ========================================
# The script supports three scenarios based on which parameters are provided:
# 1. RUN only: Start new training with config from PARAM_SETS_BATCH_AND_LR[RUN]
# 2. RUN_NAME + RUN: Resume from checkpoint but override config with PARAM_SETS_BATCH_AND_LR[RUN]
# 3. RUN_NAME only: Resume from checkpoint with its original config

# Scenario 1: Only RUN - start new training with specified config
if RUN.value is not None and RUN_NAME.value is None:
    BATCH_SIZE = PARAM_SETS_BATCH_AND_LR[int(RUN.value)]['batch_size']
    LEARNING_RATE = PARAM_SETS_BATCH_AND_LR[int(RUN.value)]['learning_rate']
    GRADIENT_ACCUMULATE_EVERY = PARAM_SETS_BATCH_AND_LR[int(RUN.value)]['gradient_accumulate_every']
    resolved_checkpoint_file = None
    print(f"Scenario 1: Starting new training with RUN {RUN.value} config")

# Scenario 2: RUN_NAME + RUN - load checkpoint + override with RUN config
# This is useful for continuing training with different hyperparameters
elif RUN_NAME.value is not None and RUN.value is not None:
    # Find checkpoint directory - try exact match first, then prefix match
    potential_dir_path = os.path.join(CHECKPOINTS_BASE_DIR, RUN_NAME.value)
    if os.path.isdir(potential_dir_path):
        RESUME_TARGET_DIR = potential_dir_path
    else:
        # Try prefix matching (useful for partial names)
        all_items = os.listdir(CHECKPOINTS_BASE_DIR)
        matching_dirs = [d for d in all_items if d.startswith(RUN_NAME.value) and os.path.isdir(os.path.join(CHECKPOINTS_BASE_DIR, d))]
        RESUME_TARGET_DIR = os.path.join(CHECKPOINTS_BASE_DIR, matching_dirs[0]) if matching_dirs else None

    # Find the most recent checkpoint file in the directory
    resolved_checkpoint_file = resume_checkpoint(RESUME_TARGET_DIR) if RESUME_TARGET_DIR else None

    # Use RUN config (override checkpoint config)
    BATCH_SIZE = PARAM_SETS_BATCH_AND_LR[int(RUN.value)]['batch_size']
    LEARNING_RATE = PARAM_SETS_BATCH_AND_LR[int(RUN.value)]['learning_rate']
    GRADIENT_ACCUMULATE_EVERY = PARAM_SETS_BATCH_AND_LR[int(RUN.value)]['gradient_accumulate_every']
    print(f"Scenario 2: Loading checkpoint from {RUN_NAME.value} but using RUN {RUN.value} config")

# Scenario 3: RUN_NAME only - load checkpoint + use saved config
# This preserves the exact training setup from the original run
elif RUN_NAME.value is not None and RUN.value is None:
    # Find checkpoint directory - try exact match first, then prefix match
    potential_dir_path = os.path.join(CHECKPOINTS_BASE_DIR, RUN_NAME.value)
    if os.path.isdir(potential_dir_path):
        RESUME_TARGET_DIR = potential_dir_path
    else:
        # Try prefix matching (useful for partial names)
        all_items = os.listdir(CHECKPOINTS_BASE_DIR)
        matching_dirs = [d for d in all_items if d.startswith(RUN_NAME.value) and os.path.isdir(os.path.join(CHECKPOINTS_BASE_DIR, d))]
        RESUME_TARGET_DIR = os.path.join(CHECKPOINTS_BASE_DIR, matching_dirs[0]) if matching_dirs else None

    # Find the most recent checkpoint file in the directory
    resolved_checkpoint_file = resume_checkpoint(RESUME_TARGET_DIR) if RESUME_TARGET_DIR else None

    if resolved_checkpoint_file:
        # Load config from checkpoint or wandb
        saved_config = load_run_config_from_checkpoint_or_wandb(resolved_checkpoint_file)
        BATCH_SIZE = saved_config.get('batch_size', 4)
        LEARNING_RATE = saved_config.get('learning_rate', 1e-4)
        GRADIENT_ACCUMULATE_EVERY = saved_config.get('gradient_accumulate_every', 1)
        print(f"Scenario 3: Loading checkpoint and config from {RUN_NAME.value}")
    else:
        raise ValueError(f"Could not find checkpoint for RUN_NAME: {RUN_NAME.value}")

# Scenario 4: Neither RUN nor RUN_NAME - error
else:
    raise ValueError("Must provide either RUN or RUN_NAME parameter")

# ========================================
# Training Hyperparameters and Constants
# ========================================

# Format learning rate in scientific notation for run naming (e.g., "1e-4")
FORMATTED_LR = f"{LEARNING_RATE:.0e}"

# Validation and generation frequencies
VALIDATE_EVERY = 100    # Validate on validation set every N steps
GENERATE_EVERY = 500    # Generate sample text every N steps
GENERATE_LENGTH = 1024  # Length of generated samples (in tokens)

# Sequence length for training
SEQ_LEN = 1024  # Maximum sequence length for input (in tokens/characters)

# Checkpoint saving frequency
SAVE_EVERY = 1000  # Save model checkpoint every N steps

# Precision configuration - FP8 for maximum performance
PRECISION = "fp8"  # Use FP8 precision (8-bit floating point) via Transformer Engine
post_fix += f"_{PRECISION}"

# ========================================
# Training Budget Calculation
# ========================================

# Calculate total number of training batches needed
TOTAL_TRAINING_TOKENS = 1e9  # 1 billion tokens for FP8 training
TOKENS_PER_BATCH = BATCH_SIZE * SEQ_LEN  # Tokens processed per batch
NUM_BATCHES = int(TOTAL_TRAINING_TOKENS / TOKENS_PER_BATCH)  # Total training steps

print(f"Final config: BATCH_SIZE={BATCH_SIZE}, LEARNING_RATE={LEARNING_RATE}, GRADIENT_ACCUMULATE_EVERY={GRADIENT_ACCUMULATE_EVERY}, NUM_BATCHES={NUM_BATCHES}")

# ========================================
# Run Naming and Directory Setup
# ========================================

# Setup run names and checkpoint directories
if resolved_checkpoint_file:
    # Resuming from checkpoint: reuse the existing run name and directory
    run_name = os.path.basename(os.path.dirname(resolved_checkpoint_file))
    wandb_run_id = run_name  # Use same ID to continue the wandb run
    CHECKPOINT_DIR = os.path.dirname(resolved_checkpoint_file)
    tprint(f"Resuming run '{run_name}' from checkpoint: {resolved_checkpoint_file}")
else:
    # Starting new training: create new run name with timestamp
    current_time = datetime.now().strftime("%y%m%d_%H%M")  # Format: YYMMDD_HHMM
    base_run_name = f"lr_{FORMATTED_LR}_bs_{BATCH_SIZE}{post_fix}"
    run_name = f"{current_time}_{base_run_name}"
    wandb_run_id = run_name
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", run_name)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    tprint(f"Starting new training run: {run_name}")

# Monitoring and logging frequencies
LOG_OPTIMIZER_STATS_EVERY = 100  # Log optimizer statistics (grad norm, param norm) every N steps
LOG_THROUGHPUT_EVERY = 10        # Log training throughput (tokens/sec) every N steps

# WandB project name
PROJECT_NAME = "x-transformers-tuning-practice"

# ========================================
# WandB Initialization
# ========================================

# Initialize Weights & Biases for experiment tracking
wandb.init(
    project=PROJECT_NAME,      # Project name in wandb dashboard
    name=run_name,             # Run name (includes timestamp, lr, batch size, precision)
    id=wandb_run_id,           # Unique run ID (allows resuming the same wandb run)
    config={
        # Log all hyperparameters for tracking and comparison
        "learning_rate": FORMATTED_LR,
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "gradient_accumulate_every": GRADIENT_ACCUMULATE_EVERY,
        "num_batches": NUM_BATCHES,
        "model_dim": MODEL_DIM,
        "model_depth": MODEL_DEPTH,
        "model_heads": MODEL_HEADS,
        "rotary_pos_emb": True,
        "precision": PRECISION  # FP8 precision
    },
    resume='allow' if resolved_checkpoint_file else 'never'  # Allow resuming if checkpoint exists
)

# ========================================
# Helper Functions for Text Decoding
# ========================================

def decode_token(token):
    """
    Convert a single token (byte value) to its character representation.

    Args:
        token (int): Token value (0-255 for byte-level encoding)

    Returns:
        str: Character representation of the token

    Note:
        Uses max(32, token) to avoid non-printable ASCII characters below 32
    """
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    """
    Convert a sequence of tokens to a readable string.

    Args:
        tokens (list or tensor): Sequence of token values

    Returns:
        str: Decoded string representation of the tokens
    """
    return ''.join(list(map(decode_token, tokens)))

# ========================================
# Model Instantiation
# ========================================

# Instantiate GPT-like decoder-only transformer model
model = TransformerWrapper(
    num_tokens = 256,           # Vocabulary size (256 for byte-level/character encoding)
    max_seq_len = SEQ_LEN,      # Maximum sequence length the model can handle
    attn_layers = Decoder(
        dim = MODEL_DIM,        # Hidden dimension of the model
        depth = MODEL_DEPTH,    # Number of transformer layers
        heads = MODEL_HEADS,    # Number of attention heads per layer
        rotary_pos_emb = True   # Use rotary positional embeddings (RoPE) instead of absolute
    )
)

# Wrap model with autoregressive generation capabilities
model = AutoregressiveWrapper(model)
model.cuda()  # Move model to GPU

# Count number of available CUDA devices for throughput calculations
num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

# ========================================
# Load Model State and Optimizer
# ========================================

# Track which step to start/resume training from
start_step = 0
if resolved_checkpoint_file:
    checkpoint = torch.load(resolved_checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_step = checkpoint.get('step', 0) + 1  # Resume from next step
    print(f"Successfully loaded checkpoint. Resuming from step {start_step}.")

# Create optimizer with the final determined learning rate
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Load optimizer state if resuming from checkpoint
# This preserves momentum and other optimizer statistics
if resolved_checkpoint_file:
    if 'optimizer_state_dict' in checkpoint:
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded optimizer state from checkpoint.")

    # Update wandb config with checkpoint information
    wandb.config.update({
        "resumed_from_checkpoint": resolved_checkpoint_file,
        "resumed_step": start_step
    }, allow_val_change=True)

# ========================================
# Data Loading and Preparation
# ========================================

# Option 1: enwik8 dataset (commented out)
# data_file_path = os.path.join(PROJECT_ROOT, 'data', 'enwik8.gz')
# # train data: 90M tokens (characters)
# # validation data: 5M tokens
# with gzip.open(data_file_path) as file:
#     data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
#     train_x, valid_x = np.split(data, [int(90e6)])
#     data_train, data_val = torch.from_numpy(train_x), torch.from_numpy(valid_x)

# Option 2: enwik9 dataset (currently active)
# Load enwik9: larger Wikipedia dataset for character-level language modeling
data_file_path = os.path.join(PROJECT_ROOT, 'data', 'enwik9.gz')
with gzip.open(data_file_path) as file:
    # Read first 340M bytes from the compressed file
    data = np.frombuffer(file.read(int(340e6)), dtype=np.uint8).copy()
    # Split into train (320M chars) and validation (20M chars)
    train_x, valid_x = np.split(data, [int(320e6)])
    # Convert to PyTorch tensors
    data_train, data_val = torch.from_numpy(train_x), torch.from_numpy(valid_x)


class TextSamplerDataset(Dataset):
    """
    Dataset for sampling random sequences from a large text corpus.

    This dataset randomly samples sequences of a fixed length from the data,
    which is useful for language modeling tasks. Each sample includes an extra
    token for the target (next token prediction).

    Attributes:
        data (torch.Tensor): The full text data as a tensor of token IDs
        seq_len (int): Length of sequences to sample
    """

    def __init__(self, data, seq_len):
        """
        Initialize the dataset.

        Args:
            data (torch.Tensor): Full text data as token IDs
            seq_len (int): Length of sequences to sample
        """
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        """
        Sample a random sequence from the data.

        Args:
            index (int): Not used (samples are random), but required by Dataset API

        Returns:
            torch.Tensor: Sequence of length (seq_len + 1) on CUDA device
                         The +1 is for the target token in next-token prediction
        """
        # Randomly select a starting position
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
        # Extract sequence of length seq_len + 1 (input + target)
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq.cuda()  # Move to GPU

    def __len__(self):
        """
        Return the number of possible sequences.

        Returns:
            int: Number of non-overlapping sequences in the data
        """
        return self.data.size(0) // self.seq_len

# Create datasets and data loaders
train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)

# Wrap loaders with cycle() to create infinite iterators
# drop_last=True ensures all batches have the same size
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE, drop_last = True))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE, drop_last = True))

# ========================================
# FP8 Training Configuration
# ========================================

# Create FP8 recipe for Transformer Engine
# This configures how FP8 precision is applied during training
fp8_recipe = recipe.DelayedScaling(
    fp8_format=recipe.Format.HYBRID,  # Hybrid format: E4M3 (4-bit exponent, 3-bit mantissa) for forward,
                                       # E5M2 (5-bit exponent, 2-bit mantissa) for backward
    amax_history_len=16,              # Track max absolute value (amax) over 16 steps for scaling
    amax_compute_algo="max"           # Use max value from history for scaling factor computation
)
# Note: E4M3 has better precision for forward pass activations,
# while E5M2 has larger dynamic range for gradients in backward pass

# ========================================
# Training Loop with FP8 Precision
# ========================================

for i in tqdm.tqdm(range(start_step, NUM_BATCHES), mininterval=10., desc='training', initial=start_step, total=NUM_BATCHES):
    model.train()  # Set model to training mode

    # Start timing for throughput calculation
    batch_start_time = time.time()
    accumulated_loss = 0

    # ========================================
    # Gradient Accumulation Loop with FP8
    # ========================================
    # Accumulate gradients over multiple micro-batches to simulate larger batch sizes
    # This allows training with effective batch sizes larger than GPU memory permits
    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        # Use FP8 autocast context for forward pass
        # Transformer Engine automatically converts eligible operations to FP8
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            # Forward pass in FP8 precision
            loss = model(next(train_loader))

        # Backward pass (outside fp8_autocast context)
        # Scale loss by accumulation steps to average gradients
        (loss / GRADIENT_ACCUMULATE_EVERY).backward()
        accumulated_loss += loss.item()

    # ========================================
    # Calculate Training Metrics
    # ========================================
    train_loss_val = accumulated_loss / GRADIENT_ACCUMULATE_EVERY  # Average loss
    train_perplexity_val = np.exp(train_loss_val)                  # Perplexity = e^loss
    train_bpc_val = train_loss_val / np.log(2)                     # Bits Per Character

    # Prepare metrics dictionary for logging
    metrics = {
        "train/loss": train_loss_val,
        "train/perplexity": train_perplexity_val,
        "train/bpc": train_bpc_val
    }

    # ========================================
    # Gradient Clipping and Optimizer Step
    # ========================================
    # Clip gradient norm to prevent exploding gradients
    # Note: No need to unscale gradients like with GradScaler since FP8 handles scaling differently
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

    # Log optimizer statistics periodically (gradient norms, parameter norms, etc.)
    if i % LOG_OPTIMIZER_STATS_EVERY == 0:
        optim_stats = calculate_optimizer_stats(optim)
        metrics.update(optim_stats)

    # Update model parameters
    optim.step()
    # Zero gradients for next iteration
    optim.zero_grad()

    # ========================================
    # Throughput Logging
    # ========================================
    # Calculate and log training throughput (tokens/second, etc.)
    batch_end_time = time.time()
    if i % LOG_THROUGHPUT_EVERY == 0:
        throughput_metrics = calculate_throughput(
            BATCH_SIZE,                 # Batch size per step
            SEQ_LEN,                    # Sequence length
            batch_start_time,           # Start time
            batch_end_time,             # End time
            GRADIENT_ACCUMULATE_EVERY,  # Number of gradient accumulation steps
            num_devices                 # Number of GPUs
        )
        metrics.update(throughput_metrics)

    # ========================================
    # Validation with FP8
    # ========================================
    if i % VALIDATE_EVERY == 0:
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient computation for validation
            # Use FP8 for validation forward pass as well
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                loss = model(next(val_loader))
            val_loss_val = loss.item()
            val_perplexity_val = np.exp(val_loss_val)
            val_bpc_val = val_loss_val / np.log(2)

            # Add validation metrics to logging dictionary
            metrics.update({
                "val/loss": val_loss_val,
                "val/perplexity": val_perplexity_val,
                "val/bpc": val_bpc_val
            })

    # Log all collected metrics to WandB
    wandb.log(metrics, step=i)

    # Print training loss to console
    print(f'training loss: {train_loss_val}')

    # ========================================
    # Text Generation with FP8 (for qualitative evaluation)
    # ========================================
    if i % GENERATE_EVERY == 0:
        model.eval()
        # Get a random sample from validation set as prompt
        inp = random.choice(val_dataset)[:-1]  # Remove last token (target)
        prime = decode_tokens(inp)
        print(f'{prime} \n\n {"*" * 100}')

        # Generate continuation of the prompt using FP8
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            sample = model.generate(
                prompts = inp.unsqueeze(0),     # Add batch dimension
                seq_len = GENERATE_LENGTH,      # Length to generate
                cache_kv = True                 # Enable KV caching for faster generation
            )

        # Decode and print generated text
        output_str = decode_tokens(sample[0].tolist())
        print(output_str)

        # Log generated text to wandb as a table for easy viewing
        generated_table = wandb.Table(columns=["step", "prime_text", "generated_text"])
        generated_table.add_data(i, prime, output_str)
        wandb.log({"generated_samples": generated_table, "step": i})

    # ========================================
    # Checkpoint Saving
    # ========================================
    if i % SAVE_EVERY == 0 and i > 0:  # Don't save at step 0
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_step_{i}.pt")
        torch.save({
            'step': i,                              # Current training step
            'model_state_dict': model.state_dict(), # Model weights
            'optimizer_state_dict': optim.state_dict(), # Optimizer state (momentum, etc.)
            'loss': train_loss_val,                 # Most recent training loss
            'training_config': {
                # Save training configuration for easy resumption
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'gradient_accumulate_every': GRADIENT_ACCUMULATE_EVERY,
                'seq_len': SEQ_LEN,
                'model_dim': MODEL_DIM,
                'model_depth': MODEL_DEPTH,
                'model_heads': MODEL_HEADS
            }
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        # Optional: save checkpoint as a wandb artifact for versioning
        # artifact = wandb.Artifact(name=f"{run_name}-step_{i}", type="model")
        # artifact.add_file(checkpoint_path)
        # wandb.log_artifact(artifact)

# ========================================
# Cleanup
# ========================================

# Finish wandb run to ensure all logs are uploaded
wandb.finish()
