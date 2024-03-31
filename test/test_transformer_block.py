import torch
from boring_nn.utils import cprint
from boring_transformer.boring_transformer import BoringEncoderBlock


# Test BoringEncoderBlock
d_model = 512
num_heads = 8
d_ff = 2048
dropout = 0.1

encoder_block = BoringEncoderBlock(d_model, num_heads, d_ff, dropout)

seq_len = [5, 3, 7]
batch_size = len(seq_len)
max_seq_len = max(seq_len)

# def encoder_block_with_padding():
#     input_seq = [torch.randn(seq_len, d_model) for seq_len in seq_len]
#     output_seq = encoder_block(input_seq, padding=True)
#     cprint(output_seq.shape == (batch_size, max_seq_len, d_model))
#     cprint(output_seq.shape)


def encoder_block_without_padding():
    input_seq = torch.randn(batch_size, max_seq_len, d_model)
    output_seq = encoder_block(input_seq)
    cprint(output_seq.shape == (batch_size, max_seq_len, d_model))
    cprint(output_seq.shape)


encoder_block_without_padding()
