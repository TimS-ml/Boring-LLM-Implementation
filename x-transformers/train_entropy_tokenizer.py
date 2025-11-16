"""
Training script for evaluating entropy-based tokenization on the enwik8 dataset.

This script trains a GPT-like transformer model on the enwik8 dataset (first 100M bytes
of Wikipedia) while using an entropy-based tokenizer. The entropy-based tokenizer
dynamically segments text based on the model's prediction entropy, creating variable-length
tokens where the model is more uncertain.

The script demonstrates:
- Loading and preparing the enwik8 dataset
- Training a transformer decoder with rotary positional embeddings
- Using EntropyBasedTokenizer to segment sequences based on prediction uncertainty
- Generating text samples with entropy-based token boundaries visualized
"""

from x_transformers import TransformerWrapper, Decoder
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper
from x_transformers.entropy_based_tokenizer import EntropyBasedTokenizer

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
GRADIENT_ACCUMULATE_EVERY = 4

# Learning rate for the Adam optimizer
LEARNING_RATE = 1e-4

# Frequency (in batches) to run validation and print validation loss
VALIDATE_EVERY  = 100

# Frequency (in batches) to generate sample text with entropy-based tokenization
GENERATE_EVERY  = 100

# Length of generated text samples (in characters)
GENERATE_LENGTH = 1024

# Maximum sequence length for training (in characters/tokens)
SEQ_LEN = 1024

# helpers

def cycle(loader):
    """
    Infinitely cycle through a DataLoader.

    This generator function wraps a DataLoader and yields batches indefinitely,
    restarting from the beginning when the dataset is exhausted. Useful for
    training loops that don't want to manually handle epoch boundaries.

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
    Values below 32 (control characters) are clamped to 32 (space) for
    readability.

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

# Create the base transformer model
# num_tokens = 256: vocabulary size (one token per byte value 0-255)
# max_seq_len = SEQ_LEN: maximum sequence length the model can process
model = TransformerWrapper(
    num_tokens = 256,
    max_seq_len = SEQ_LEN,
    attn_layers = Decoder(
        dim = 512,          # Hidden dimension size for embeddings and attention
        depth = 6,          # Number of transformer layers
        heads = 8,          # Number of attention heads per layer
        rotary_pos_emb = True  # Use rotary positional embeddings (RoPE) instead of learned positional embeddings
    )
)

# Wrap the model with an entropy-based tokenizer
# entropy_threshold = 2.5: predictions with entropy above this threshold trigger token boundaries
# Higher entropy indicates the model is uncertain, suggesting a natural segmentation point
tokenizer = EntropyBasedTokenizer(
    model,
    entropy_threshold = 2.5
)

# Wrap the model for autoregressive text generation
# This wrapper handles the forward pass for language modeling (predicting next token)
model = AutoregressiveWrapper(model)

# Move model to GPU for faster training
model.cuda()

# prepare enwik8 data

# Load the enwik8 dataset (first 100M bytes of Wikipedia XML)
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
    sequence, enabling the model to see diverse contexts during training.
    """

    def __init__(self, data, seq_len):
        """
        Initialize the text sampler dataset.

        Args:
            data: PyTorch tensor containing the full text data as byte values
            seq_len: Length of sequences to sample
        """
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        """
        Sample a random sequence from the dataset.

        Args:
            index: Dataset index (not used; sampling is random regardless)

        Returns:
            Tensor of shape (seq_len + 1,) containing a sequence and its next token,
            moved to GPU. The +1 allows for input/target splitting in training.
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

        Returns:
            Number of non-overlapping sequences that fit in the data
        """
        return self.data.size(0) // self.seq_len

# Create dataset instances for training and validation
train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)

# Create infinite data loaders using the cycle helper
# drop_last = True ensures all batches have exactly BATCH_SIZE samples
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE, drop_last = True))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE, drop_last = True))

# optimizer

# Initialize Adam optimizer with specified learning rate
# Adam is used for its adaptive learning rates and momentum properties
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training

# Main training loop
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    # Set model to training mode (enables dropout, etc.)
    model.train()

    # Gradient accumulation loop: accumulate gradients over multiple batches
    # This simulates a larger batch size without requiring more GPU memory
    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        # Forward pass: compute loss on next training batch
        loss = model(next(train_loader))
        # Backward pass: scale loss by accumulation steps and compute gradients
        # Scaling ensures the average gradient magnitude is correct
        (loss / GRADIENT_ACCUMULATE_EVERY).backward()

    # Print the training loss (from the last accumulated batch)
    print(f'training loss: {loss.item()}')

    # Clip gradients to prevent exploding gradients
    # Max norm of 0.5 helps stabilize training
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

    # Update model parameters using accumulated gradients
    optim.step()
    # Clear gradients for next iteration
    optim.zero_grad()

    # Validation: periodically evaluate on validation set
    if i % VALIDATE_EVERY == 0:
        # Set model to evaluation mode (disables dropout, etc.)
        model.eval()
        with torch.no_grad():  # Disable gradient computation for efficiency
            # Compute loss on validation batch
            loss = model(next(val_loader))
            print(f'validation loss: {loss.item()}')

    # Text generation: periodically generate samples with entropy-based tokenization
    if i % GENERATE_EVERY == 0:
        model.eval()
        # Select a random validation sequence (remove last token for prompting)
        inp = random.choice(val_dataset)[:-1]

        # Apply entropy-based tokenizer to segment the sequence
        # return_segmented_seq = True returns a list of token segments
        # where boundaries are placed at high-entropy positions
        tokens = tokenizer(inp, return_segmented_seq = True)

        # Use a visual delimiter to show where token boundaries occur
        delimiter = " \u275A "  # Unicode character for vertical bar
        # Decode each token segment and join with delimiter
        output_str = delimiter.join([decode_tokens(token) for token in tokens])

        # Print the segmented output showing entropy-based token boundaries
        print(f"{output_str}\n\n")
