import torch
from boring_utils.utils import cprint
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
    # cprint(padded_src_seqs.shape == (batch_size, max_src_seq_len, d_model))
    # cprint(padded_tgt_seqs.shape == (batch_size, max_tgt_seq_len, d_model))

    # Create source attention mask
    src_mask = torch.zeros(batch_size, max_tgt_seq_len, max_src_seq_len)
    for i in range(batch_size):
        src_mask[i, :tgt_seq_len[i], :src_seq_len[i]] = 1

    # Create target attention mask (causal mask)
    tgt_mask = torch.tril(torch.ones(max_tgt_seq_len, max_tgt_seq_len)).unsqueeze(0).expand(batch_size, -1, -1)

    # Convert attention masks to boolean tensors
    src_mask = src_mask.bool()
    tgt_mask = tgt_mask.bool()
    cprint(src_mask) 
    cprint(tgt_mask)
    cprint(src_mask.shape) 
    cprint(tgt_mask.shape)
    cprint(src_mask.shape == tgt_mask.shape)

    # Pass the padded input sequences and attention masks to the decoder block
    output_seq = decoder_block(padded_tgt_seqs, padded_src_seqs, src_mask=src_mask, tgt_mask=tgt_mask)

    cprint(output_seq.shape == (batch_size, max_tgt_seq_len, d_model))
    cprint(output_seq.shape)


def decoder_block_without_mask():
    # Create input sequences of different lengths
    src_seqs = [torch.randn(src_seq_len[i], d_model) for i in range(batch_size)]
    tgt_seqs = [torch.randn(tgt_seq_len[i], d_model) for i in range(batch_size)]

    # Pad the sequences to the maximum sequence length
    padded_src_seqs = pad_sequence(src_seqs, batch_first=True)
    padded_tgt_seqs = pad_sequence(tgt_seqs, batch_first=True)

    # Pass the padded input sequences to the decoder block without masks
    output_seq = decoder_block(padded_tgt_seqs, padded_src_seqs)

    cprint(output_seq.shape == (batch_size, max_tgt_seq_len, d_model))
    cprint(output_seq.shape)


decoder_block_with_mask()

#  decoder_block_without_mask()
