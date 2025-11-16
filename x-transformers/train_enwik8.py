# /// script
# dependencies = [
#   "tqdm",
#   "x-transformers",
#   "wandb"
# ]
# ///

"""
Training script for character-level language modeling on the enwik8 dataset.

This script trains a GPT-like transformer model on the enwik8 benchmark dataset,
which consists of the first 100M bytes of Wikipedia XML. The model learns to predict
the next byte in a sequence, treating text as a sequence of raw bytes (0-255).

Features:
- Character-level (byte-level) language modeling with 256-token vocabulary
- Transformer decoder with rotary positional embeddings (RoPE)
- Orthogonal projected values for attention (experimental feature)
- Gradient accumulation for effective larger batch sizes
- Weights & Biases (wandb) integration for experiment tracking
- Periodic validation and text generation for monitoring training progress

The enwik8 dataset is a standard benchmark for evaluating character-level language models.
"""

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

# constants

# Total number of training batches to process (100,000 iterations)
NUM_BATCHES = int(1e5)

# Number of sequences to process in each batch
BATCH_SIZE = 4

# Number of forward passes to accumulate gradients before updating weights
# Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATE_EVERY = 16
# This allows training with larger effective batch sizes without exceeding GPU memory
GRADIENT_ACCUMULATE_EVERY = 4

# Learning rate for the Adam optimizer
LEARNING_RATE = 1e-4

# Frequency (in batches) to run validation and log validation loss
VALIDATE_EVERY  = 100

# Frequency (in batches) to generate sample text and display output
# Set higher than VALIDATE_EVERY to reduce generation overhead
GENERATE_EVERY  = 500

# Length of generated text samples (in characters/bytes)
GENERATE_LENGTH = 1024

# Maximum sequence length for training (in characters/tokens)
SEQ_LEN = 1024

# Whether to track experiment metrics online with Weights & Biases
# Set to False for offline/local logging only
TRACK_EXPERIMENT_ONLINE = False

# helpers

def cycle(loader):
    """
    Infinitely cycle through a DataLoader.

    This generator function wraps a DataLoader and yields batches indefinitely,
    restarting from the beginning when the dataset is exhausted. This is useful
    for training loops that don't want to manually handle epoch boundaries.

    Args:
        loader: PyTorch DataLoader to cycle through

    Yields:
        Batches of data from the loader, cycling infinitely
    """
    while True:
        for data in loader:
            yield data

def decode_token(token):
    """
    Decode a single byte token to its character representation.

    Converts a byte value (0-255) to its corresponding ASCII/Unicode character.
    Values below 32 (control characters) are clamped to 32 (space) to ensure
    readable output and avoid terminal control issues.

    Args:
        token: Integer token value (0-255) representing a byte

    Returns:
        String containing the single decoded character
    """
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    """
    Decode a sequence of byte tokens to a string.

    Converts a sequence of byte values to their string representation by
    applying decode_token to each element and concatenating the results.

    Args:
        tokens: Iterable of integer token values (0-255)

    Returns:
        Decoded string representation of the token sequence
    """
    return ''.join(list(map(decode_token, tokens)))

# instantiate GPT-like decoder model

# Create the base transformer model for character-level language modeling
# num_tokens = 256: vocabulary size (one token per byte value 0-255)
# max_seq_len = SEQ_LEN: maximum sequence length the model can process
model = TransformerWrapper(
    num_tokens = 256,
    max_seq_len = SEQ_LEN,
    attn_layers = Decoder(
        dim = 512,          # Hidden dimension size for embeddings and attention
        depth = 6,          # Number of transformer layers (blocks)
        heads = 8,          # Number of attention heads per layer
        rotary_pos_emb = True,  # Use rotary positional embeddings (RoPE) instead of learned positional embeddings
        attn_orthog_projected_values = True,  # Apply orthogonal projection to attention values (experimental feature)
        attn_orthog_projected_values_per_head = True  # Apply orthogonal projection per attention head
    )
)

# Wrap the model for autoregressive text generation
# This wrapper handles the forward pass for language modeling (predicting next token)
# and provides a generate() method for sampling text
model = AutoregressiveWrapper(model)

# Move model to GPU for faster training
model.cuda()

# prepare enwik8 data

# Load the enwik8 dataset (first 100M bytes of Wikipedia XML)
# This is a standard benchmark for character-level language modeling
# Read 95M bytes total, split into 90M for training and 5M for validation
with gzip.open('./data/enwik8.gz') as file:
    # Read data as unsigned 8-bit integers (bytes 0-255)
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    # Split into training (first 90M bytes) and validation (remaining 5M bytes)
    train_x, valid_x = np.split(data, [int(90e6)])
    # Convert numpy arrays to PyTorch tensors
    data_train, data_val = torch.from_numpy(train_x), torch.from_numpy(valid_x)

class TextSamplerDataset(Dataset):
    """
    Dataset that samples random subsequences from a text corpus.

    This dataset provides random contiguous sequences of specified length from
    the input data. Each call to __getitem__ returns a randomly positioned
    sequence, enabling the model to see diverse contexts during training without
    being limited by fixed epoch boundaries.
    """

    def __init__(self, data, seq_len):
        """
        Initialize the text sampler dataset.

        Args:
            data: PyTorch tensor containing the full text data as byte values (0-255)
            seq_len: Length of sequences to sample
        """
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        """
        Sample a random sequence from the dataset.

        The index parameter is ignored; sampling is always random. This allows
        the dataset to provide fresh random samples on each iteration.

        Args:
            index: Dataset index (not used; sampling is random regardless)

        Returns:
            Tensor of shape (seq_len + 1,) containing a sequence and its next token,
            moved to GPU. The +1 allows for input/target splitting during training:
            - input: sequence[:seq_len]
            - target: sequence[1:seq_len+1]
        """
        # Randomly select a starting position that leaves room for full sequence
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
        # Extract sequence of length seq_len + 1 (for input and target)
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        # Move to GPU and return
        return full_seq.cuda()

    def __len__(self):
        """
        Return the number of possible sequences in the dataset.

        This is used by DataLoader to determine dataset size, though with random
        sampling the actual number of unique sequences seen depends on training steps.

        Returns:
            Number of non-overlapping sequences that fit in the data
        """
        return self.data.size(0) // self.seq_len

# Create dataset instances for training and validation
train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)

# Create infinite data loaders using the cycle helper function
# drop_last = True ensures all batches have exactly BATCH_SIZE samples,
# preventing issues with the last incomplete batch
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE, drop_last = True))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE, drop_last = True))

# optimizer

# Initialize Adam optimizer with specified learning rate
# Adam is used for its adaptive learning rates per parameter and built-in momentum
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# experiment

# Initialize Weights & Biases for experiment tracking
# This logs training metrics, hyperparameters, and system info
import wandb
# mode = 'online': sync to wandb servers, 'disabled': local logging only
wandb.init(project = 'enwik8', mode = 'online' if TRACK_EXPERIMENT_ONLINE else 'disabled')
# Set a name for this experiment run to distinguish it from other runs
wandb.run.name = 'baseline'

# training

# Main training loop - iterate over NUM_BATCHES batches with progress bar
# mininterval=10. updates progress bar at most every 10 seconds to reduce overhead
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    # Set model to training mode (enables dropout, layer norm in training mode, etc.)
    model.train()

    # Gradient accumulation loop: accumulate gradients over multiple mini-batches
    # This simulates a larger batch size (BATCH_SIZE * GRADIENT_ACCUMULATE_EVERY = 16)
    # without requiring more GPU memory, which is important for large models
    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        # Forward pass: compute loss on next training batch
        # The model computes cross-entropy loss for next-token prediction
        loss = model(next(train_loader))
        # Backward pass: scale loss and compute gradients
        # Scaling by GRADIENT_ACCUMULATE_EVERY ensures the average gradient
        # magnitude matches what we'd get with a single larger batch
        (loss / GRADIENT_ACCUMULATE_EVERY).backward()

    # Print the training loss (from the last accumulated batch)
    print(f'training loss: {loss.item()}')
    # Log training loss to Weights & Biases for tracking over time
    wandb.log(dict(loss = loss.item()))

    # Gradient clipping: prevent exploding gradients by capping the norm
    # Max norm of 0.5 helps stabilize training, especially early on
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

    # Update model parameters using the accumulated gradients
    optim.step()
    # Clear gradients for the next iteration
    optim.zero_grad()

    # Validation: periodically evaluate model on held-out validation set
    if i % VALIDATE_EVERY == 0:
        # Set model to evaluation mode (disables dropout, etc.)
        model.eval()
        with torch.no_grad():  # Disable gradient computation to save memory
            # Compute loss on validation batch
            loss = model(next(val_loader))

            # Print and log validation loss
            print(f'validation loss: {loss.item()}')
            wandb.log(dict(valid_loss = loss.item()))

    # Text generation: periodically generate samples to qualitatively monitor training
    if i % GENERATE_EVERY == 0:
        # Set model to evaluation mode for generation
        model.eval()
        # Select a random validation sequence to use as prompt
        # Remove last token [:-1] since we want to predict from this prefix
        inp = random.choice(val_dataset)[:-1]
        # Decode the prompt to show what we're starting with
        prime = decode_tokens(inp)
        print(f'%s \n\n %s', (prime, '*' * 100))

        # Generate continuation using autoregressive sampling
        # cache_kv = True enables key-value caching for faster generation
        sample = model.generate(
            prompts = inp,
            seq_len = GENERATE_LENGTH,  # Generate this many additional tokens
            cache_kv = True  # Cache attention keys/values for efficiency
        )

        # Decode and print the generated text
        output_str = decode_tokens(sample)
        print(output_str)
