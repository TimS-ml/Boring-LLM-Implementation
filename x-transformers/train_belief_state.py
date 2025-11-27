"""
Training script for Belief State models on the enwik8 character-level language modeling task.

This script demonstrates training a bidirectional language model using the BeliefStateWrapper,
which combines forward and backward decoders to model sequences in both directions. The model
is trained on the enwik8 dataset (first 100MB of English Wikipedia) and can generate text
conditioned on both prefix and suffix context.

The belief state approach allows the model to maintain representations that incorporate
information from both past and future context, potentially improving coherence and consistency
in generated text.
"""

from x_transformers import TransformerWrapper, Decoder, BeliefStateWrapper
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

# Total number of training batches (100,000)
NUM_BATCHES = int(1e5)

# Number of sequences to process in parallel
BATCH_SIZE = 2

# Number of gradient accumulation steps before updating weights
# Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATE_EVERY = 16
GRADIENT_ACCUMULATE_EVERY = 8

# Learning rate for the Adam optimizer
LEARNING_RATE = 1e-4

# Validate on validation set every N batches
VALIDATE_EVERY  = 100

# Generate sample text every N batches to monitor quality
GENERATE_EVERY  = 500

# Number of tokens to generate during sampling
GENERATE_LENGTH = 256

# Maximum sequence length for training and inference
SEQ_LEN = 256

# Whether to use the same model for both forward and backward passes
# If True, the backward model will be None and forward model handles both directions
FORWARD_BACKWARD_SAME_MODEL = True

# helpers

def cycle(loader):
    """
    Infinitely cycle through a DataLoader.

    Creates an infinite generator that continuously yields batches from the DataLoader.
    When the DataLoader is exhausted, it automatically restarts from the beginning.
    This is useful for training loops that don't have a fixed number of epochs.

    Args:
        loader: A PyTorch DataLoader to cycle through

    Yields:
        Batches of data from the DataLoader, cycling indefinitely
    """
    while True:
        for data in loader:
            yield data

def decode_token(token):
    """
    Decode a single token (integer) to its corresponding ASCII character.

    Converts a token ID to its ASCII character representation. Ensures the character
    is printable by enforcing a minimum value of 32 (space character).

    Args:
        token: Integer token ID (0-255 for byte-level encoding)

    Returns:
        String containing the single decoded character
    """
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    """
    Decode a sequence of tokens to a human-readable string.

    Converts a list/array of token IDs to a string by mapping each token to its
    corresponding ASCII character and concatenating them.

    Args:
        tokens: List or array of integer token IDs

    Returns:
        String containing the decoded text
    """
    return ''.join(list(map(decode_token, tokens)))

# instantiate GPT-like decoder model for forward and backwards

# Create the forward (left-to-right) decoder model
# This model processes sequences from beginning to end
forward_model = TransformerWrapper(
    num_tokens = 256,  # Vocabulary size (256 for byte-level encoding)
    max_seq_len = SEQ_LEN,  # Maximum sequence length
    attn_layers = Decoder(
        dim = 512,  # Model dimension (hidden size)
        depth = 6,  # Number of transformer layers
        heads = 8,  # Number of attention heads
        rotary_pos_emb = True  # Use rotary positional embeddings (RoPE)
    )
)

# Initialize backward model as None (will be created if needed)
backward_model = None

# Optionally create a separate backward (right-to-left) decoder model
# This model processes sequences from end to beginning
if not FORWARD_BACKWARD_SAME_MODEL:
    backward_model = TransformerWrapper(
        num_tokens = 256,  # Same vocabulary size as forward model
        max_seq_len = SEQ_LEN,  # Same maximum sequence length
        attn_layers = Decoder(
            dim = 512,  # Same model dimension
            depth = 4,  # Smaller depth for efficiency (fewer layers than forward)
            heads = 8,  # Same number of attention heads
            rotary_pos_emb = True  # Use rotary positional embeddings
        )
    )

# Wrap both models in BeliefStateWrapper to enable bidirectional modeling
# If backward_model is None, the forward_model handles both directions
model = BeliefStateWrapper(
    forward_decoder = forward_model,
    backward_decoder = backward_model
)

# Move model to GPU for faster training
model.cuda()

# prepare enwik8 data

# Load and split the enwik8 dataset (first 100MB of English Wikipedia)
# enwik8 is a standard benchmark for character-level language modeling
with gzip.open('./data/enwik8.gz') as file:
    # Read first 95 million bytes as uint8 (byte-level tokens)
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    # Split into training (90M bytes) and validation (5M bytes) sets
    train_x, valid_x = np.split(data, [int(90e6)])
    # Convert numpy arrays to PyTorch tensors
    data_train, data_val = torch.from_numpy(train_x), torch.from_numpy(valid_x)

class TextSamplerDataset(Dataset):
    """
    Dataset for sampling random sequences from a text corpus.

    This dataset randomly samples subsequences of a fixed length from the input data,
    which is useful for training language models on long documents without being
    constrained by a fixed partitioning of the data.
    """

    def __init__(self, data, seq_len):
        """
        Initialize the TextSamplerDataset.

        Args:
            data: PyTorch tensor containing the text data as integer token IDs
            seq_len: Length of sequences to sample
        """
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        """
        Sample a random sequence from the data.

        Args:
            index: Index (not actually used; sampling is random)

        Returns:
            PyTorch tensor of shape (seq_len + 1) containing a random sequence.
            The extra token is used for next-token prediction targets.
        """
        # Sample a random starting position, ensuring we have enough room for seq_len + 1 tokens
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
        # Extract the sequence (seq_len + 1 tokens for input and target)
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        # Move to GPU and return
        return full_seq.cuda()

    def __len__(self):
        """
        Return the number of possible sequences in the dataset.

        Returns:
            Number of non-overlapping sequences of length seq_len in the data
        """
        return self.data.size(0) // self.seq_len

# Create datasets for training and validation
train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)

# Create infinite data loaders using the cycle helper function
# drop_last=True ensures all batches have the same size
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE, drop_last = True))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE, drop_last = True))

# optimizer

# Initialize Adam optimizer with all model parameters
# Adam is used for its adaptive learning rates and good performance on language models
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training

# Main training loop - iterate for the specified number of batches
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval = 10., desc = 'training'):
    # Set model to training mode (enables dropout, etc.)
    model.train()

    # Gradient accumulation loop - accumulate gradients over multiple batches
    # This simulates a larger effective batch size without requiring more GPU memory
    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        # Get next batch from the training data loader
        # The model computes loss internally for the belief state training
        loss = model(next(train_loader))
        # Backward pass: compute gradients, scaled by accumulation steps
        # Scaling ensures gradients have correct magnitude when accumulated
        (loss / GRADIENT_ACCUMULATE_EVERY).backward()

    # Log the training loss (from the last accumulated batch)
    print(f'training loss: {loss.item()}')

    # Gradient clipping to prevent exploding gradients
    # Clips the norm of the gradients to a maximum value of 0.5
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

    # Update model parameters using accumulated gradients
    optim.step()

    # Clear gradients for the next iteration
    optim.zero_grad()

    # Validation: periodically evaluate on held-out validation set
    if i % VALIDATE_EVERY == 0:
        # Set model to evaluation mode (disables dropout, etc.)
        model.eval()
        # Disable gradient computation for validation (saves memory and computation)
        with torch.no_grad():
            # Compute loss on a validation batch
            loss = model(next(val_loader))
            # Log validation loss to monitor for overfitting
            print(f'validation loss: {loss.item()}')

    # Generation: periodically generate sample text to qualitatively assess model quality
    if i % GENERATE_EVERY == 0:
        # Set model to evaluation mode
        model.eval()
        # Randomly select a sequence from validation set as the prompt
        # Remove last token (it will be predicted, not used as input)
        inp = random.choice(val_dataset)[:-1]
        # Decode the prompt to human-readable text
        prime = decode_tokens(inp)

        # Print separator and the prompt text
        print(f'%s \n\n %s', (prime, '*' * 100))

        # Generate text in the forward (left-to-right) direction
        print('forwards:\n')

        # Generate continuation conditioned on the prompt as suffix
        # This demonstrates the model's ability to generate coherent forward continuations
        sample = model.generate_with_suffix_cond(
            prompts = inp,  # Input prompt sequence
            seq_len = GENERATE_LENGTH,  # Number of tokens to generate
            cache_kv = True  # Cache key-value pairs for faster generation
        )

        # Decode and print the generated text
        output_str = decode_tokens(sample)
        print(output_str)

        # Generate text in the backward (right-to-left) direction
        print('\nbackwards:\n')

        # Generate continuation in reverse direction
        # This demonstrates the model's backward generation capability
        sample = model.generate_with_suffix_cond(
            prompts = inp,  # Input prompt sequence
            seq_len = GENERATE_LENGTH,  # Number of tokens to generate
            cache_kv = True,  # Cache key-value pairs for faster generation
            decode_backwards = True  # Generate in reverse direction
        )

        # Flip the output back to forward direction and decode
        output_str = decode_tokens(sample.flip(0))
        print(output_str)
