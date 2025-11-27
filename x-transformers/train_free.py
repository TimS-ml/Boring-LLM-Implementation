"""
Training script for FreeTransformer model on the enwik8 dataset.

This script trains a FreeTransformer model (a VAE-based transformer architecture)
on character-level language modeling using the enwik8 dataset. The FreeTransformer
combines autoregressive language modeling with variational autoencoders (VAE) to
learn latent representations of text sequences.

The model learns to:
1. Generate text autoregressively given a latent code
2. Compress sequences into low-dimensional latent representations
3. Balance reconstruction quality (autoregressive loss) with latent regularization (KL loss)

Dataset: enwik8 (first 100 million bytes of English Wikipedia dump)
Task: Character-level language modeling with latent variables
"""

# /// script
# dependencies = [
#   "tqdm",
#   "x-transformers>=2.11.0",
# ]
# ///

from x_transformers.free_transformer import FreeTransformer

from math import log
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
GENERATE_EVERY  = 250  # Generate sample text every N batches to monitor quality
GENERATE_LENGTH = 512  # Number of characters to generate during sampling
PRIME_LENGTH = 32  # Number of characters to use as prompt/seed for generation
SEQ_LEN = 512  # Maximum sequence length for training samples

# VAE latent space configuration
LATENT_BITS = 8  # Number of bits for latent representation (2^8 = 256 possible discrete latent codes)
NAT = log(2)  # Natural unit of information (1 bit in nats), used as KL divergence threshold

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
# Create FreeTransformer model - a VAE-based transformer that combines
# autoregressive language modeling with latent variable modeling

model = FreeTransformer(
    num_tokens = 256,  # Vocabulary size (256 for byte-level encoding)
    max_seq_len = SEQ_LEN,  # Maximum sequence length the model can handle
    dim = 512,  # Model dimension / embedding size
    heads = 8,  # Number of attention heads
    dec_head_depth = 4,  # Decoder transformer layers before latent injection
    dec_tail_depth = 4,  # Decoder transformer layers after latent injection
    enc_depth = 3,  # Encoder transformer layers (for encoding sequence to latent)
    kl_loss_weight = 1.,  # Weight for KL divergence loss term
    per_token_latents = True,  # Use per-token latent codes instead of sequence-level
    kl_loss_threshold = NAT,  # Threshold for KL loss (information bottleneck in nats)
    latent_bits = LATENT_BITS  # Bits for discrete latent representation
).cuda()

# Random latent code for conditioned text generation during evaluation
one_hot_indices = torch.randint(0, 2 ** LATENT_BITS, ())

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

        # Sample a random sequence from validation set and use first PRIME_LENGTH tokens as prompt
        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        prime = decode_tokens(inp)
        print(f'%s \n\n %s', (prime, '*' * 100))

        # Generate text continuation conditioned on the prompt and a fixed latent code
        sample = model.generate(
            prompts = inp,  # Initial prompt/seed text
            seq_len = GENERATE_LENGTH,  # Number of tokens to generate
            latents = one_hot_indices  # Latent code to condition generation
        )

        # Decode generated tokens to readable text
        output_str = decode_tokens(sample)

        # Print generated text with the latent code used
        print(f'\n\nlatent {one_hot_indices.tolist()} - ', output_str)
