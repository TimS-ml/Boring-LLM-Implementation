"""
Parity Task Training Script

This script trains a transformer model to solve the parity task - predicting whether
the cumulative sum of binary inputs is even (0) or odd (1) at each position. This is
a challenging task that tests a model's ability to track state over long sequences.

The script uses curriculum learning, gradually increasing the sequence length as the
model masters shorter sequences. It also optionally hybridizes the transformer with
an RNN to help with state tracking, testing whether hybrid architectures can better
solve tasks requiring perfect memory.

The key challenge is length generalization: can the model learn on shorter sequences
and then correctly perform the parity task on longer sequences it hasn't seen?
"""

import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F

from x_transformers import TransformerWrapper, Decoder

# constants

# Number of sequences processed in parallel
BATCH_SIZE = 256

# Learning rate for the optimizer
LEARNING_RATE = 3e-4

# Evaluate model performance every N training steps
EVAL_EVERY  = 500

# Different sequence lengths to evaluate on, testing length generalization
EVAL_LENGTHS = (16, 32, 64, 128, 256, 512)

# Maximum length to train on (256) - we'll use curriculum learning to reach this
TRAIN_MAX_LENGTH = EVAL_LENGTHS[-2]

# Loss threshold - if loss drops below this, consider the current length "mastered"
LOSS_THRES_INCREASE_LEN = 1e-3

# Number of consecutive steps meeting the loss threshold before increasing length
# Ensures the model has consistently mastered the current length
MEET_CRITERIA_THRES_INCREASE_LEN = 10

# Whether to use RNN hybridization to help with state tracking
HYBRIDIZE_WITH_RNN = True

# rnn for fully resolving state tracking by hybridization
# but will also look into gated delta net + negative eigenvalues (Songlin Yang et al) as a parallel solution

# Model dimension (embedding size)
dim = 64
# Number of attention heads
heads = 4
# Dimension per attention head
dim_head = 32
# Additional decoder arguments (populated below if using RNN hybrid)
decoder_kwargs = dict()

# Configure RNN hybridization if enabled
if HYBRIDIZE_WITH_RNN:
    from torch.nn import GRU

    decoder_kwargs = dict(
        # Process attention in chunks of 4 tokens, alternating with RNN
        # Even with recurrence every 4 tokens, the model can still generalize for parity
        attn_hybrid_fold_axial_dim = 4,
        # Learn to mix attention and RNN outputs dynamically
        attn_hybrid_learned_mix = True,
        # Use GRU as the recurrent module for state tracking
        # GRU input: dim, GRU output: dim_head * heads (to match attention output)
        attn_hybrid_module = GRU(dim, dim_head * heads, batch_first = True)
    )

# instantiate model

model = TransformerWrapper(
    # Binary tokens: 0 and 1
    num_tokens = 2,
    # Set to 0 to allow variable sequence lengths (no fixed max)
    max_seq_len = 0,
    attn_layers = Decoder(
        # Model dimension
        dim = dim,
        # Number of transformer layers
        depth = 3,
        # Number of attention heads
        heads = heads,
        # Dimension per attention head
        attn_dim_head = dim_head,
        # Shift tokens by 1 position - helps significantly with parity training
        # by giving the model easier access to previous tokens, though it can't
        # generalize to longer sequences on its own without RNN hybridization
        shift_tokens = 1,
        # Apply any RNN hybridization settings if configured
        **decoder_kwargs
    )
).cuda()

# optimizer

from lion_pytorch.cautious_lion import Lion

# Use Lion optimizer with cautious updates
# Cautious Lion applies a damping factor to reduce aggressive updates
optimizer = Lion(model.parameters(), lr = LEARNING_RATE, cautious_factor = 0.1)

# data generator

def cycle(length):
    """
    Infinite generator for parity task data.

    Generates random binary sequences and computes the parity (even/odd) of the
    cumulative sum at each position. The parity task requires the model to track
    whether it has seen an even or odd number of 1s up to each position.

    Args:
        length: Length of sequences to generate

    Yields:
        Tuple of (seq, labels) where:
        - seq: Random binary sequence of shape (BATCH_SIZE, length)
        - labels: Parity labels (0 for even, 1 for odd) at each position
    """
    while True:
        # Generate random binary sequences (0s and 1s)
        seq = torch.randint(0, 2, (BATCH_SIZE, length)).cuda()
        # Compute cumulative sum and take modulo 2 to get parity
        # cumsum counts total 1s seen so far, mod 2 gives even (0) or odd (1)
        labels = (seq.cumsum(dim = -1) % 2)
        yield (seq, labels)

# dataloaders

# Training data loader - will use curriculum learning to gradually increase length
train_dl = cycle(TRAIN_MAX_LENGTH)

# Evaluation data loaders for each test length
eval_dls = {eval_length: cycle(eval_length) for eval_length in EVAL_LENGTHS}

print(f'training at max length: {TRAIN_MAX_LENGTH}')

# training

# Training step counter
i = 0
# Counter for consecutive steps meeting the loss threshold
meet_criteria = 0
# Current training sequence length (starts at 1, grows via curriculum learning)
train_seq_len = 1
# Target length to train up to (256)
stop_length = EVAL_LENGTHS[-2]

# Main training loop with progress bar
with tqdm.tqdm(mininterval = 10., desc = 'training') as pbar:

    # Continue training until we reach the target sequence length
    while train_seq_len < stop_length:
        # Set model to training mode
        model.train()

        # Get next batch of maximum-length sequences
        seq, labels = next(train_dl)

        # length curriculum learning
        # Truncate sequences to current training length
        # This implements curriculum learning: start with short sequences and
        # gradually increase length as the model masters each stage

        seq = seq[:, :train_seq_len]
        labels = labels[:, :train_seq_len]

        # Forward pass: get logits for each position
        logits = model(seq)

        # Compute cross-entropy loss for each position independently
        loss = F.cross_entropy(logits.transpose(-1, -2), labels, reduction = 'none')
        # Focus on the last position's loss - this is what matters for parity
        # (the model must correctly predict parity at the final position)
        last_loss = loss[:, -1].mean()
        # Backpropagate mean loss across all positions
        loss.mean().backward()

        # Check if model has mastered current length
        if last_loss.item() < LOSS_THRES_INCREASE_LEN:
            # Loss is below threshold, increment success counter
            meet_criteria += 1
        else:
            # Loss too high, reset counter (must meet criteria consecutively)
            meet_criteria = 0

        # If model has consistently mastered current length, increase to next length
        if meet_criteria >= MEET_CRITERIA_THRES_INCREASE_LEN:
            meet_criteria = 0
            train_seq_len += 1
            print(f'criteria met, incrementing to {train_seq_len}')

        print(f'({train_seq_len})| {i}: {last_loss.item()}')
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        # Update model parameters
        optimizer.step()
        # Clear gradients for next iteration
        optimizer.zero_grad()

        # Check if we've reached the target training length
        last_step = train_seq_len == stop_length

        if last_step:
            print(f'made it to training length {train_seq_len}. running final eval to check for generalization')

        # Periodic evaluation or final evaluation
        if last_step or (i + 1) % EVAL_EVERY == 0:

            # Set model to evaluation mode
            model.eval()
            print('\n')

            # Evaluate on all test lengths to measure generalization
            for eval_length, eval_dl in eval_dls.items():
                incorrects = 0

                # Get evaluation batch
                seq, labels = next(eval_dl)

                # Forward pass
                logits = model(seq)
                # Predict parity at the last position (0 or 1)
                pred = logits[:, -1].argmax(dim = -1)
                # Count how many predictions are incorrect
                incorrects = (pred != labels[:, -1]).abs().sum().item()

                # Calculate percentage of incorrect predictions
                frac_incorrect = incorrects * 100 / BATCH_SIZE

                # Print results for this sequence length
                print(f"{eval_length}\t - frac incorrect:\t {frac_incorrect:.1f}%")

            print('\n')

        # Increment step counter and progress bar
        i += 1
        pbar.update(1)
