"""
Training script for sequence-to-sequence copy task using XTransformer.

This script trains an encoder-decoder transformer model on a simple copy task, where
the model learns to reproduce an input sequence. This is a fundamental test to verify
that the model can learn basic sequence-to-sequence mappings and serves as a sanity
check for the transformer architecture.

The task: Given an input sequence of random tokens, the model must generate an output
that exactly copies the input sequence. A special prefix token is prepended to the
target sequence to signal the start of generation.

This task tests the model's ability to:
- Encode input sequences into meaningful representations
- Attend to the encoder output during decoding
- Generate exact reproductions of arbitrary sequences
"""

import tqdm
import torch
import torch.optim as optim
from x_transformers import XTransformer

# constants

# Total number of training batches (100,000 iterations)
NUM_BATCHES = int(1e5)

# Number of sequence pairs to process in parallel
BATCH_SIZE = 32

# Learning rate for the Adam optimizer
LEARNING_RATE = 3e-4

# Generate and evaluate sample outputs every N batches
GENERATE_EVERY  = 100

# Total vocabulary size: 16 regular tokens + 2 special tokens (e.g., padding, start)
NUM_TOKENS = 16 + 2

# Length of encoder input sequences
ENC_SEQ_LEN = 32

# Length of decoder output sequences (including start token)
# The +1 accounts for the start-of-sequence token
DEC_SEQ_LEN = 64 + 1

# Use GPU if available, otherwise fall back to CPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# helpers

def cycle():
    """
    Infinite generator that creates synthetic copy task data batches.

    This function generates training data for the copy task on-the-fly. Each batch
    consists of random source sequences and corresponding target sequences where the
    target is the source repeated twice with a prefix token.

    The data format:
    - Source (encoder input): Random sequence of tokens
    - Target (decoder input/output): [START_TOKEN, source_seq, source_seq]
    - Mask: All-ones mask indicating all source tokens are valid (no padding)

    Yields:
        Tuple of (src, tgt, src_mask) where:
            - src: Source sequences, shape (BATCH_SIZE, ENC_SEQ_LEN)
            - tgt: Target sequences, shape (BATCH_SIZE, 1 + 2*ENC_SEQ_LEN)
            - src_mask: Source mask, shape (BATCH_SIZE, ENC_SEQ_LEN), all True
    """
    while True:
        # Create start-of-sequence token (token ID = 1) for each sequence in batch
        prefix = torch.ones((BATCH_SIZE, 1)).long().to(DEVICE)

        # Generate random source sequences with token IDs in range [2, NUM_TOKENS)
        # Token 0 may be reserved for padding, token 1 for start-of-sequence
        src = torch.randint(2, NUM_TOKENS, (BATCH_SIZE, ENC_SEQ_LEN)).long().to(DEVICE)

        # Create target by concatenating: [START_TOKEN] + [source] + [source]
        # The model learns to copy the source sequence twice
        tgt = torch.cat((prefix, src, src), 1)

        # Create attention mask (all True = attend to all tokens, no padding)
        src_mask = torch.ones(BATCH_SIZE, src.shape[1]).bool().to(DEVICE)

        yield (src, tgt, src_mask)

# instantiate model

# Create an encoder-decoder transformer model (XTransformer)
# This architecture consists of separate encoder and decoder stacks with cross-attention
model = XTransformer(
    # Model dimension (hidden size) shared by encoder and decoder
    dim = 128,

    # Tie token embeddings between encoder and decoder to reduce parameters
    # Both encoder and decoder will share the same embedding matrix
    tie_token_emb = True,

    # Return the loss computed on target sequences during training
    # The model will internally compute cross-entropy loss
    return_tgt_loss = True,

    # Encoder configuration
    enc_num_tokens=NUM_TOKENS,  # Vocabulary size for encoder
    enc_depth = 3,  # Number of encoder transformer layers
    enc_heads = 8,  # Number of attention heads in encoder
    enc_max_seq_len = ENC_SEQ_LEN,  # Maximum encoder sequence length
    enc_attn_cog_signed = True,  # Use signed CoG (Center of Gravity) attention

    # Decoder configuration
    dec_num_tokens = NUM_TOKENS,  # Vocabulary size for decoder (same as encoder)
    dec_depth = 3,  # Number of decoder transformer layers
    dec_heads = 8,  # Number of attention heads in decoder
    dec_max_seq_len = DEC_SEQ_LEN,  # Maximum decoder sequence length
    dec_attn_cog_signed = True  # Use signed CoG attention in decoder
).to(DEVICE)  # Move model to the specified device (GPU or CPU)

# optimizer

# Initialize Adam optimizer with all trainable model parameters
# Adam is commonly used for transformers due to its adaptive learning rates
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training

# Main training loop - iterate through the specified number of batches
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    # Set model to training mode (enables dropout, batch norm updates, etc.)
    model.train()

    # Get next batch of synthetic training data (source, target, mask)
    src, tgt, src_mask = next(cycle())

    # Forward pass: compute loss
    # The model takes source sequence, target sequence, and attention mask
    # With return_tgt_loss=True, the model returns the cross-entropy loss
    loss = model(src, tgt, mask=src_mask)

    # Backward pass: compute gradients with respect to model parameters
    loss.backward()

    # Log the current training loss for monitoring
    print(f'{i}: {loss.item()}')

    # Update model parameters using computed gradients
    optim.step()

    # Clear gradients for the next iteration
    optim.zero_grad()

    # Evaluation: periodically test the model's copy performance
    if i != 0 and i % GENERATE_EVERY == 0:
        # Set model to evaluation mode (disables dropout, etc.)
        model.eval()

        # Get a fresh batch of test data
        src, _, src_mask = next(cycle())

        # Use only the first sequence from the batch for evaluation
        src, src_mask = src[:1], src_mask[:1]

        # Create start token (token ID = 1) to begin generation
        start_tokens = (torch.ones((1, 1)) * 1).long().to(DEVICE)

        # Generate output sequence autoregressively
        # The model should copy the input sequence
        # Generate ENC_SEQ_LEN tokens (same length as input)
        sample = model.generate(src, start_tokens, ENC_SEQ_LEN, mask = src_mask)

        # Count the number of incorrectly predicted tokens
        # Measures how well the model learned the copy task
        incorrects = (src != sample).long().abs().sum()

        # Print evaluation results
        print(f"input:  ", src)
        print(f"predicted output:  ", sample)
        print(f"incorrects: {incorrects}")
