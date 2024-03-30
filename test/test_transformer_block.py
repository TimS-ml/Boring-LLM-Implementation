import torch
from boring_transformer.utils import cprint
from boring_transformer.boring_transformer import BoringEncoderBlock


B, T, C = 4, 8, 32  # batch size, time steps (seq length), channels
x = torch.rand(B, T, C)

# Generate a lower triangular matrix for causal masking
# Then convert the lower triangular matrix to float; positions to attend to are marked as 0, others as -inf
tril = torch.tril(torch.ones(T, T))
mask = tril.float().masked_fill(tril == 0, float('-inf'))
cprint(mask)


# Test BoringEncoderBlock
d_model = 512
num_heads = 8
d_ff = 2048
dropout = 0.1

encoder_block = BoringEncoderBlock(d_model, num_heads, d_ff, dropout)

# Test with padding
seq_lengths = [5, 3, 7]
batch_size = len(seq_lengths)
max_seq_len = max(seq_lengths)
input_seq = [torch.randn(seq_len, d_model) for seq_len in seq_lengths]
output_seq = encoder_block(input_seq, padding=True)
assert output_seq.shape == (batch_size, max_seq_len, d_model)

# Test without padding
input_seq = torch.randn(batch_size, max_seq_len, d_model)
output_seq = encoder_block(input_seq)
assert output_seq.shape == (batch_size, max_seq_len, d_model)

