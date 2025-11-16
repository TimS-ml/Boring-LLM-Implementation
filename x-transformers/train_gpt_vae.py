"""
Training script for GPTVAE model on the enwik8 dataset.

This script trains a GPTVAE (GPT Variational Autoencoder) model on character-level
language modeling using the enwik8 dataset. The GPTVAE architecture combines:

1. GPT-style autoregressive transformer decoder for text generation
2. Variational autoencoder (VAE) component for learning latent representations
3. Conditional generation based on continuous latent codes

The model learns to compress entire sequences into low-dimensional continuous
latent vectors, then generate text conditioned on these latents. This enables
controlled text generation by manipulating latent codes.

Dataset: enwik8 (first 100 million bytes of English Wikipedia dump)
Task: Character-level conditional language modeling with continuous latent variables
"""

from x_transformers.gpt_vae import GPTVAE

import random
import tqdm
import gzip
import numpy as np

import torch
import torch.optim as optim
from torch import tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# ============================================================================
# Training Hyperparameters and Configuration Constants
# ============================================================================

# Training loop configuration
NUM_BATCHES = int(1e5)  # Total number of training batches (100,000)
BATCH_SIZE = 4  # Number of sequences processed in parallel
GRADIENT_ACCUMULATE_EVERY = 4  # Accumulate gradients over this many mini-batches before updating (effective batch size = 4 * 4 = 16)
LEARNING_RATE = 1e-4  # Adam optimizer learning rate

# Evaluation and generation intervals
VALIDATE_EVERY  = 100  # Run validation every N batches
GENERATE_EVERY  = 500  # Generate sample text every N batches to monitor quality
GENERATE_LENGTH = 512  # Number of characters to generate during sampling
SEQ_LEN = 512  # Maximum sequence length for training samples

# ============================================================================
# Helper Functions
# ============================================================================

def cycle(loader):
    """
    Infinite data loader iterator.

    Wraps a PyTorch DataLoader to create an infinite loop that continuously
    yields batches. When the dataloader is exhausted, it automatically restarts
    from the beginning.

    Args:
        loader: PyTorch DataLoader object

    Yields:
        Batches of data from the loader, cycling indefinitely
    """
    while True:
        for data in loader:
            yield data

def decode_token(token):
    """
    Convert a single token (byte value) to its character representation.

    Args:
        token: Integer token value (0-255 for byte-level encoding)

    Returns:
        String containing the character corresponding to the token.
        Non-printable characters (< 32) are converted to space (32).
    """
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    """
    Convert a sequence of tokens to a readable string.

    Args:
        tokens: List or tensor of integer token values

    Returns:
        String formed by decoding each token and concatenating them
    """
    return ''.join(list(map(decode_token, tokens)))

# ============================================================================
# Model Instantiation
# ============================================================================
# Create GPTVAE model - combines GPT-style autoregressive generation with
# VAE latent variable modeling for controlled text generation

model = GPTVAE(
    num_tokens = 256,  # Vocabulary size (256 for byte-level encoding)
    max_seq_len = SEQ_LEN,  # Maximum sequence length the model can handle
    dim = 512,  # Model dimension / embedding size
    depth = 6,  # Number of transformer decoder layers
    heads = 8,  # Number of attention heads
    rotary_pos_emb = True,  # Use rotary positional embeddings (RoPE) for better position encoding
    enc_depth = 3,  # Number of transformer encoder layers (for encoding sequence to latent)
    vae_kl_loss_weight = 1.,  # Weight for KL divergence loss term
    dim_latent = 1  # Dimensionality of continuous latent space (1D for simplicity in this example)
).cuda()

# Fixed latent vector for conditioned text generation during evaluation
# This 1D latent can be varied to control generation behavior
latents = tensor([1.]).cuda()

# ============================================================================
# Data Preparation - enwik8 Dataset
# ============================================================================
# enwik8 is the first 100 million bytes of an English Wikipedia XML dump
# It's commonly used for character-level language modeling benchmarks

with gzip.open('./data/enwik8.gz') as file:
    # Read first 95 million bytes and convert to numpy array of uint8 (byte values 0-255)
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    # Split into training (first 90M bytes) and validation (next 5M bytes)
    train_x, valid_x = np.split(data, [int(90e6)])
    # Convert numpy arrays to PyTorch tensors
    data_train, data_val = torch.from_numpy(train_x), torch.from_numpy(valid_x)

class TextSamplerDataset(Dataset):
    """
    PyTorch Dataset for sampling random text sequences.

    This dataset randomly samples subsequences of a specified length from
    the full text data. Each call to __getitem__ returns a different random
    slice, providing varied training examples.
    """

    def __init__(self, data, seq_len):
        """
        Initialize the dataset.

        Args:
            data: PyTorch tensor containing the full text data as byte values
            seq_len: Length of sequences to sample
        """
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        """
        Sample a random sequence from the data.

        Args:
            index: Dataset index (not used; sampling is always random)

        Returns:
            Tensor of shape (seq_len + 1,) containing a random text sequence.
            The +1 is for language modeling where we need both input and target.
        """
        # Randomly select a starting position
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
        # Extract sequence of length seq_len + 1 (input + target)
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq.cuda()

    def __len__(self):
        """
        Return the number of possible sequences.

        Returns:
            Number of non-overlapping sequences that fit in the data
        """
        return self.data.size(0) // self.seq_len

# Create dataset objects
train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)

# Create infinite data loaders using the cycle helper function
# drop_last=True ensures all batches have consistent size
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE, drop_last = True))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE, drop_last = True))

# ============================================================================
# Optimizer Setup
# ============================================================================

# Adam optimizer with default betas=(0.9, 0.999) and eps=1e-8
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ============================================================================
# Main Training Loop
# ============================================================================

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    # Set model to training mode (enables dropout, batch norm updates, etc.)
    model.train()

    # ========================================================================
    # Gradient Accumulation Loop
    # ========================================================================
    # Accumulate gradients over multiple mini-batches to simulate larger batch size
    # This helps when GPU memory is limited
    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        # Forward pass: compute loss
        # model returns total_loss, (autoregressive_loss, kl_divergence_loss)
        loss, (ar_loss, vae_kl_loss) = model(next(train_loader), return_all_losses = True)

        # Backward pass: scale loss by accumulation steps to average gradients
        (loss / GRADIENT_ACCUMULATE_EVERY).backward()

    # Print training metrics
    # ar_loss: autoregressive (reconstruction) loss - measures how well the model predicts next tokens
    # vae_kl_loss: KL divergence loss - regularizes the latent space
    print(f'training loss: {ar_loss.item():.4f}\t| kl loss: {vae_kl_loss.item():.4f}')

    # ========================================================================
    # Optimizer Step
    # ========================================================================
    # Clip gradients to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

    # Update model parameters
    optim.step()

    # Clear accumulated gradients for next iteration
    optim.zero_grad()

    # ========================================================================
    # Validation
    # ========================================================================
    if i % VALIDATE_EVERY == 0:
        # Set model to evaluation mode (disables dropout, etc.)
        model.eval()

        # Disable gradient computation for validation (saves memory and computation)
        with torch.no_grad():
            # Compute validation loss on a batch from validation set
            loss, (ar_loss, _) = model(next(val_loader), return_all_losses = True)
            print(f'validation loss: {ar_loss.item():.4f}')

    # ========================================================================
    # Text Generation (Quality Monitoring)
    # ========================================================================
    if i % GENERATE_EVERY == 0:
        model.eval()

        # Sample a random sequence from validation set (excluding last token for prompt)
        inp = random.choice(val_dataset)[:-1]
        prime = decode_tokens(inp)
        print(f'%s \n\n %s', (prime, '*' * 100))

        # ====================================================================
        # Generate text with positive latent value
        # ====================================================================
        sample = model.generate(
            prompts = inp,  # Initial prompt/seed text
            seq_len = GENERATE_LENGTH,  # Number of tokens to generate
            cache_kv = True,  # Cache key-value pairs for faster generation
            latents = latents  # Latent code to condition generation (value: 1.0)
        )

        # Decode generated tokens to readable text
        output_str = decode_tokens(sample)

        # Print generated text with the latent code used
        print(f'\n\nlatent {latents.tolist()} - ', output_str)

        # ====================================================================
        # Generate text with negative latent value
        # ====================================================================
        # This demonstrates how the continuous latent space affects generation
        # Negating the latent explores the opposite direction in latent space
        sample_other_direction = model.generate(
            prompts = inp,  # Same prompt for comparison
            seq_len = GENERATE_LENGTH,  # Same length
            cache_kv = True,  # Cache key-value pairs for faster generation
            latents = -latents  # Negated latent code (value: -1.0)
        )

        # Decode and print the alternative generation
        output_str = decode_tokens(sample_other_direction)
        print(f'\n\nlatent {(-latents).tolist()} - ', output_str)
