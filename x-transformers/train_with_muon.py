"""
Training script for GPT-like transformer model using Muon optimizer on enwik8 dataset.

This script demonstrates training a character-level language model on the enwik8 dataset,
which is the first 100 million bytes of the English Wikipedia XML dump. The model is
trained using the Muon optimizer combined with AdamAtan2, which is designed to provide
better optimization dynamics for transformer models.

The script includes:
- GPT-like decoder-only transformer architecture
- Autoregressive wrapper for language modeling
- Character-level tokenization (256 possible byte values)
- Training with gradient accumulation
- Periodic validation and text generation
- Muon optimizer for improved training dynamics

Dataset: enwik8 (90MB training, 5MB validation)
Model: 6-layer transformer with 512 dimensions, 8 attention heads
"""

# /// script
# dependencies = [
#   "x-transformers",
#   "adam-atan2-pytorch>=0.2.4",
# ]
# ///

from x_transformers import TransformerWrapper, Decoder
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper

import random
import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from adam_atan2_pytorch import MuonAdamAtan2

# Training hyperparameters and configuration constants

# Total number of training iterations (100,000 batches)
NUM_BATCHES = int(1e5)

# Number of sequences to process in each batch
BATCH_SIZE = 4

# Number of gradient accumulation steps before updating model parameters
# Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATE_EVERY = 16
GRADIENT_ACCUMULATE_EVERY = 4

# Learning rate for the Muon optimizer
LEARNING_RATE = 1e-4

# Frequency (in batches) to run validation on the validation set
VALIDATE_EVERY  = 100

# Frequency (in batches) to generate sample text from the model
GENERATE_EVERY  = 500

# Length of text to generate during sampling (in characters)
GENERATE_LENGTH = 1024

# Maximum sequence length for training (in characters)
SEQ_LEN = 1024

# Helper functions for data loading and text decoding

def cycle(loader):
    """
    Creates an infinite iterator from a DataLoader.

    This function wraps a DataLoader to yield batches indefinitely by restarting
    from the beginning once all batches have been exhausted. This is useful for
    training loops that need to run for a fixed number of iterations rather than
    a fixed number of epochs.

    Args:
        loader: A PyTorch DataLoader instance

    Yields:
        Data batches from the loader, cycling infinitely
    """
    while True:
        for data in loader:
            yield data

def decode_token(token):
    """
    Converts a single token (byte value) to its character representation.

    Args:
        token: Integer token value (0-255 for byte-level encoding)

    Returns:
        String containing the character corresponding to the token.
        Non-printable characters (< 32) are converted to space (ASCII 32)
    """
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    """
    Converts a sequence of tokens to a readable string.

    Args:
        tokens: Iterable of integer token values

    Returns:
        String formed by concatenating the decoded characters from all tokens
    """
    return ''.join(list(map(decode_token, tokens)))

# Model instantiation - GPT-like decoder-only transformer

# Create the transformer model with the following architecture:
# - num_tokens: 256 (byte-level encoding, all possible byte values)
# - max_seq_len: Maximum sequence length the model can handle
# - Decoder configuration:
#   - dim: 512 (model dimension, embedding and hidden layer size)
#   - depth: 6 (number of transformer layers)
#   - heads: 8 (number of attention heads per layer)
#   - rotary_pos_emb: True (use rotary position embeddings instead of learned positional encodings)
model = TransformerWrapper(
    num_tokens = 256,
    max_seq_len = SEQ_LEN,
    attn_layers = Decoder(
        dim = 512,
        depth = 6,
        heads = 8,
        rotary_pos_emb = True
    )
)

# Wrap the model with autoregressive functionality for language modeling
# This handles the autoregressive decoding and loss computation
ar_wrapper = AutoregressiveWrapper(model)

# Move model to GPU for faster training
model.cuda()

# Data preparation - loading and splitting enwik8 dataset

# Load the enwik8 dataset from compressed file
# - Read first 95MB of the dataset (95 million bytes)
# - Convert to numpy array of unsigned 8-bit integers (0-255)
# - Split into 90MB for training and 5MB for validation
with gzip.open('./data/enwik8.gz') as file:
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    train_x, valid_x = np.split(data, [int(90e6)])
    data_train, data_val = torch.from_numpy(train_x), torch.from_numpy(valid_x)

class TextSamplerDataset(Dataset):
    """
    PyTorch Dataset for sampling random sequences from text data.

    This dataset randomly samples sequences of a fixed length from the underlying
    text data. Each call to __getitem__ returns a random sequence, which is useful
    for training language models where we want diverse training samples.

    Attributes:
        data: Torch tensor containing the text data as byte values
        seq_len: Length of sequences to sample
    """
    def __init__(self, data, seq_len):
        """
        Initialize the TextSamplerDataset.

        Args:
            data: Torch tensor of text data (as byte values)
            seq_len: Integer specifying the length of sequences to sample
        """
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        """
        Get a random sequence from the dataset.

        The index parameter is ignored; instead, a random starting position is chosen
        each time. This ensures maximum diversity in training samples.

        Args:
            index: Dataset index (ignored, random sampling is used instead)

        Returns:
            Tensor of shape (seq_len + 1,) containing a sequence on GPU.
            The extra token is needed for language modeling (input = seq[:-1], target = seq[1:])
        """
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq.cuda()

    def __len__(self):
        """
        Returns the number of possible sequences in the dataset.

        Returns:
            Integer representing the number of non-overlapping sequences
        """
        return self.data.size(0) // self.seq_len

# Create dataset instances for training and validation
train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)

# Create infinite data loaders using the cycle function
# drop_last=True ensures all batches have the same size by dropping incomplete final batch
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE, drop_last = True))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE, drop_last = True))

# Optimizer setup - Muon optimizer combined with AdamAtan2

# MuonAdamAtan2 is a hybrid optimizer that:
# - Uses Muon optimization for certain parameters (typically attention and normalization layers)
# - Uses AdamAtan2 for the remaining parameters
# - muon_params: Parameters that will be optimized with Muon
# - params: All model parameters
# - remove_muon_params_from_params: Ensures Muon parameters are not optimized twice
# - lr: Learning rate for the optimizer
optim = MuonAdamAtan2(
    muon_params = model.muon_parameters(),
    params = model.parameters(),
    remove_muon_params_from_params = True,
    lr = LEARNING_RATE
)

# Main training loop

# Iterate over the specified number of training batches
# tqdm provides a progress bar with updates every 10 seconds
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    # Set model to training mode (enables dropout, batch norm updates, etc.)
    model.train()

    # Gradient accumulation loop
    # Accumulate gradients over multiple batches before updating parameters
    # This simulates a larger batch size without using more GPU memory
    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        # Get next training batch and compute loss
        # ar_wrapper automatically handles the autoregressive loss computation
        loss = ar_wrapper(next(train_loader))

        # Backward pass with scaled loss
        # Scale by GRADIENT_ACCUMULATE_EVERY to maintain correct gradient magnitudes
        (loss / GRADIENT_ACCUMULATE_EVERY).backward()

    # Display the training loss for the last accumulated batch
    print(f'training loss: {loss.item()}')

    # Gradient clipping to prevent exploding gradients
    # Clips gradient norm to maximum value of 0.5
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

    # Update model parameters using accumulated gradients
    optim.step()

    # Clear gradients for next iteration
    optim.zero_grad()

    # Validation phase - run periodically to monitor generalization
    if i % VALIDATE_EVERY == 0:
        # Set model to evaluation mode (disables dropout, etc.)
        model.eval()

        # Disable gradient computation for validation (saves memory and computation)
        with torch.no_grad():
            # Compute validation loss on a batch from validation set
            loss = ar_wrapper(next(val_loader))
            print(f'validation loss: {loss.item()}')

    # Text generation phase - periodically generate sample text to monitor model quality
    if i % GENERATE_EVERY == 0:
        # Set model to evaluation mode
        model.eval()

        # Select a random sequence from validation set as prompt
        # Remove last token (we'll generate from this prompt)
        inp = random.choice(val_dataset)[:-1]

        # Decode the prompt to human-readable text
        prime = decode_tokens(inp)

        # Print the prompt followed by a separator
        print(f'%s \n\n %s', (prime, '*' * 100))

        # Generate continuation of the prompt
        # - prompts: The initial sequence to continue from
        # - seq_len: Maximum length of generated sequence
        # - cache_kv: Use key-value caching for faster generation
        sample = ar_wrapper.generate(
            prompts = inp,
            seq_len = GENERATE_LENGTH,
            cache_kv = True
        )

        # Decode and print the generated text
        output_str = decode_tokens(sample)
        print(output_str)
