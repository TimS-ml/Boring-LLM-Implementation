import torch
from boring_utils.utils import cprint
from boring_transformer.legacy.boring_transformer import BoringEncoderBlock
from torch.nn.utils.rnn import pad_sequence


# Test BoringEncoderBlock
d_model = 512
num_heads = 8
d_ff = 2048
dropout = 0.1

encoder_block = BoringEncoderBlock(d_model, num_heads, d_ff, dropout)

seq_len = [5, 3, 7]
batch_size = len(seq_len)
max_seq_len = max(seq_len)


def encoder_block_with_mask():
    # Create input sequences of different lengths
    input_seqs = [torch.randn(seq_len[i], d_model) for i in range(batch_size)]

    # Pad the sequences to the maximum sequence length
    padded_input_seqs = pad_sequence(input_seqs, batch_first=True)

    # Create attention mask
    attn_mask = torch.zeros(batch_size, max_seq_len, max_seq_len)
    for i in range(batch_size):
        attn_mask[i, :seq_len[i], :seq_len[i]] = 1

    # Convert attention mask to boolean tensor
    attn_mask = attn_mask.bool()
    cprint(attn_mask.shape)

    # Pass the padded input sequences and attention mask through the encoder block
    output_seq = encoder_block(padded_input_seqs, attn_mask)

    cprint(output_seq.shape == (batch_size, max_seq_len, d_model))
    cprint(output_seq.shape)


def encoder_block_without_mask():
    input_seq = torch.randn(batch_size, max_seq_len, d_model)
    output_seq = encoder_block(input_seq)
    cprint(output_seq.shape == (batch_size, max_seq_len, d_model))
    cprint(output_seq.shape)


encoder_block_with_mask()

# encoder_block_without_mask()
