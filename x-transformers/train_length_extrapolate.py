"""
Length Extrapolation Training Script

This script trains a transformer model on the enwik8 dataset to test its ability to
extrapolate to longer sequence lengths than seen during training. The model is trained
on sequences of length 256 but validated on progressively longer sequences (up to 4096)
to evaluate how well it generalizes beyond the training sequence length.

The model uses dynamic positional bias instead of absolute positional embeddings,
which is hypothesized to help with length extrapolation capabilities.
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

# Total number of training batches to run
NUM_BATCHES = int(1e5)

# Number of sequences processed in parallel
BATCH_SIZE = 4

# Number of gradient accumulation steps before updating weights
# Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATE_EVERY = 16
GRADIENT_ACCUMULATE_EVERY = 4

# Learning rate for Adam optimizer
LEARNING_RATE = 1e-4

# Generate sample text every N batches to qualitatively assess model performance
GENERATE_EVERY  = 500

# Length of text to generate during sampling
GENERATE_LENGTH = 256

# Training sequence length - model is trained on this fixed length
SEQ_LEN = 256

# Validate model performance every N batches
VALIDATE_EVERY  = 100

# Sequence lengths to use during validation to test length extrapolation
# Model is trained on 256 but tested on sequences up to 4096
VALIDATE_SEQ_LENS = (256, 512, 1024, 2048, 4096)

# helpers

def cycle(loader):
    """
    Infinite data loader wrapper that cycles through the dataset endlessly.

    Args:
        loader: PyTorch DataLoader to cycle through

    Yields:
        Data batches from the loader, repeating infinitely
    """
    while True:
        for data in loader:
            yield data

def decode_token(token):
    """
    Decode a single integer token to its character representation.

    Args:
        token: Integer token value (0-255)

    Returns:
        String character corresponding to the token. Uses max(32, token) to ensure
        printable ASCII characters (avoiding control characters below 32).
    """
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    """
    Decode a sequence of integer tokens to a readable string.

    Args:
        tokens: List or tensor of integer token values

    Returns:
        String formed by decoding all tokens and concatenating them
    """
    return ''.join(list(map(decode_token, tokens)))

# instantiate GPT-like decoder model

model = TransformerWrapper(
    # 256 tokens representing byte values (0-255) in enwik8
    num_tokens = 256,
    # Maximum sequence length the model will process during training
    max_seq_len = SEQ_LEN,
    # Don't use absolute positional embeddings - this is key for length extrapolation
    use_abs_pos_emb = False,
    attn_layers = Decoder(
        # Model dimension (embedding size)
        dim = 512,
        # Number of transformer layers
        depth = 6,
        # Number of attention heads
        heads = 8,
        # Use dynamic positional bias instead of fixed positions
        # This helps the model extrapolate to longer sequences
        dynamic_pos_bias = True,
    )
)

# Wrap model for autoregressive text generation
model = AutoregressiveWrapper(model)
# Move model to GPU
model.cuda()

# prepare enwik8 data

# Load enwik8 dataset - first 100MB of English Wikipedia in XML format
# This is a standard benchmark for character-level language modeling
with gzip.open('./data/enwik8.gz') as file:
    # Read first 95 million bytes as uint8 (byte values 0-255)
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    # Split into training (first 90M bytes) and validation (last 5M bytes)
    train_x, valid_x = np.split(data, [int(90e6)])
    # Convert numpy arrays to PyTorch tensors
    data_train, data_val = torch.from_numpy(train_x), torch.from_numpy(valid_x)

class TextSamplerDataset(Dataset):
    """
    Dataset that randomly samples text sequences from a larger text corpus.

    Each call to __getitem__ samples a random contiguous sequence of the specified
    length from the underlying data. This provides data augmentation through random
    sampling and ensures the model sees varied contexts during training.
    """

    def __init__(self, data, seq_len):
        """
        Initialize the dataset.

        Args:
            data: PyTorch tensor containing the text data as integer tokens
            seq_len: Length of sequences to sample
        """
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        """
        Sample a random sequence from the data.

        Args:
            index: Dataset index (not used, sampling is random)

        Returns:
            A tensor of length (seq_len + 1) containing a sequence and its target,
            moved to GPU. The extra token is for next-token prediction training.
        """
        # Sample random starting position, ensuring we have enough data for full sequence
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
        # Extract sequence of length (seq_len + 1) for autoregressive training
        # The +1 allows using first seq_len tokens as input and last seq_len as targets
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq.cuda()

    def __len__(self):
        """
        Return the nominal dataset length.

        Returns:
            Number of non-overlapping sequences that fit in the data
        """
        return self.data.size(0) // self.seq_len

# Create training dataset and infinite data loader
train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
# Wrap in cycle() for infinite iteration, drop_last ensures consistent batch sizes
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE, drop_last = True))

# Create validation dataset for text generation sampling
val_dataset_generate = TextSamplerDataset(data_val, SEQ_LEN)

# validation loaders with different sequence lengths

# Dictionary to store validation loaders for different sequence lengths
# This allows testing length extrapolation by validating on sequences longer than training
val_loaders = dict()

for valid_seq_len in VALIDATE_SEQ_LENS:
    # Create dataset and loader for each validation sequence length
    val_dataset   = TextSamplerDataset(data_val, valid_seq_len)
    val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE, drop_last = True))

    val_loaders[valid_seq_len] = val_loader

# optimizer

# Adam optimizer for updating model parameters
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training

# Main training loop
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    # Set model to training mode (enables dropout, etc.)
    model.train()

    # Gradient accumulation loop - accumulate gradients over multiple batches
    # before updating weights. This simulates a larger batch size with limited memory.
    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        # Get next batch and compute loss
        # AutoregressiveWrapper automatically handles input/target splitting
        loss = model(next(train_loader))
        # Scale loss by accumulation steps to maintain gradient scale
        (loss / GRADIENT_ACCUMULATE_EVERY).backward()

    print(f'training loss: {loss.item()}')

    # Clip gradient norms to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    # Update model parameters based on accumulated gradients
    optim.step()
    # Clear gradients for next iteration
    optim.zero_grad()

    # Periodic validation to test length extrapolation
    if i % VALIDATE_EVERY == 0:
        print(f'validation losses:\n')

        # Set model to evaluation mode (disables dropout, etc.)
        model.eval()
        with torch.no_grad():
            # Validate on each of the different sequence lengths
            for valid_seq_len in VALIDATE_SEQ_LENS:
                val_loader = val_loaders[valid_seq_len]

                # Compute validation loss
                loss = model(next(val_loader))
                # Print loss for this sequence length to monitor extrapolation performance
                print(f'[{valid_seq_len}]:\t {loss.item()}')

        print('\n')

    # Periodic text generation to qualitatively assess model performance
    if i % GENERATE_EVERY == 0:
        model.eval()
        # Select random sequence from validation set and remove last token
        inp = random.choice(val_dataset_generate)[:-1]
        # Decode to human-readable text to show as prompt
        prime = decode_tokens(inp)
        print(f'%s \n\n %s', (prime, '*' * 100))

        # Generate continuation of the prompt
        sample = model.generate(
            prompts = inp,
            seq_len = GENERATE_LENGTH,
            # Use KV caching for efficient autoregressive generation
            cache_kv = True
        )

        # Decode and print generated text
        output_str = decode_tokens(sample)
        print(f'{output_str}\n\n')
