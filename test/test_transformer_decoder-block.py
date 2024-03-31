import torch
from boring_nn.utils import cprint
from boring_transformer.boring_transformer import BoringDecoderBlock
from torch.nn.utils.rnn import pad_sequence


# Test BoringDecoderBlock
d_model = 512
num_heads = 8
d_ff = 2048
dropout = 0.1

decoder_block = BoringDecoderBlock(d_model, num_heads, d_ff, dropout)

src_seq_len = [5, 3, 7]
tgt_seq_len = [6, 4, 8]
batch_size = len(src_seq_len)
max_src_seq_len = max(src_seq_len)
max_tgt_seq_len = max(tgt_seq_len)


def decoder_block_with_mask():
    # Create input sequences of different lengths
    src_seqs = [torch.randn(src_seq_len[i], d_model) for i in range(batch_size)]
    tgt_seqs = [torch.randn(tgt_seq_len[i], d_model) for i in range(batch_size)]

    # Pad the sequences to the maximum sequence length
    padded_src_seqs = pad_sequence(src_seqs, batch_first=True)
    padded_tgt_seqs = pad_sequence(tgt_seqs, batch_first=True)

    # Create the padding masks
    src_padding_mask = torch.tensor([[0] * src_seq_len[i] + [1] * (max_src_seq_len - src_seq_len[i]) for i in range(batch_size)])
    tgt_padding_mask = torch.tensor([[0] * tgt_seq_len[i] + [1] * (max_tgt_seq_len - tgt_seq_len[i]) for i in range(batch_size)])

    # Create the attention masks
    src_mask = src_padding_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, max_src_seq_len)
    src_mask = src_mask.to(dtype=torch.bool)  # Convert to boolean mask

    tgt_mask = torch.triu(torch.ones(max_tgt_seq_len, max_tgt_seq_len), diagonal=1) == 1  # (max_tgt_seq_len, max_tgt_seq_len)
    tgt_mask = tgt_mask.to(dtype=torch.bool)  # Convert to boolean mask
    tgt_mask = tgt_mask | tgt_padding_mask.unsqueeze(1)  # Combine the masks

    # Pass the padded input sequences and attention masks to the decoder block
    output_seq = decoder_block(padded_tgt_seqs, padded_src_seqs, src_mask=src_mask, tgt_mask=tgt_mask)

    cprint(output_seq.shape == (batch_size, max_tgt_seq_len, d_model))
    cprint(output_seq.shape)


decoder_block_with_mask()
