'''
On LayerNorm's location: https://arxiv.org/abs/2002.04745
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange
from torch.nn.utils.rnn import pad_sequence

from torch import Tensor, Size
from typing import Optional, Tuple, Union, List

from boring_transformer.attention import MultiHeadAttention
from boring_transformer.norm import LayerNorm
from boring_transformer.pe import SinusoidalPositionalEncoding, LearnedPositionalEncoding
from boring_transformer.utils import cprint


class BoringEncoderBlock(nn.Module):
    '''
    Typical encoder block in a transformer model.
    Assume the input already padded, otherwise, set padding=True.
      i.e.: expect the input x to have a shape of (batch_size, max_sequence_length, d_model)
    '''
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(BoringEncoderBlock, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm1 = LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None, padding=False):
        '''
        When padding:
          x: list of sequences, with different lengths (seq_len_i, d_model)
          padded_x:     (batch_size, max_sequence_length, d_model)
          padding_mask: (batch_size, max_sequence_length)
          attn_mask:    (batch_size, max_sequence_length, max_sequence_length)
        '''
        if padding:
            # Pad sequences to the same length
            x = pad_sequence(x, batch_first=True)
            
            # Create attn mask
            batch_size, seq_len, _ = x.size()
            padding_mask = (x == 0)  # Assuming padding token is 0
            attn_mask = padding_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)

            # Multi-head self-attn
            attn_output, _ = self.mha(x, x, x, attn_mask=attn_mask)
        else:
            # Multi-head self-attn
            attn_output, _ = self.mha(x, x, x, attn_mask=mask)

        attn_output = self.dropout1(attn_output)
        x = self.layer_norm1(x + attn_output)
        
        # Feed-forward network
        ff_output = self.feed_forward(x)
        ff_output = self.dropout2(ff_output)
        x = self.layer_norm2(x + ff_output)
        
        return x


class BoringDecoderBlock(nn.Module):
    '''
    Typical decoder block in a transformer model.
    Assume the input already padded, otherwise, set padding=True.
      i.e.: expect the input x to have a shape of (batch_size, max_sequence_length, d_model)
    '''
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(BoringDecoderBlock, self).__init__()
        
        self.masked_mha = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm1 = LayerNorm(d_model)
        
        self.mha = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm2 = LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout3 = nn.Dropout(dropout)
        self.layer_norm3 = LayerNorm(d_model)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None, padding=False):
        '''
        src_mask for decoder self-attn
        tgt_mast for encoder-decoder attn
        '''
        if padding:
            # Pad target sequences to the same length
            x = pad_sequence(x, batch_first=True)
            
            # Create target attn mask
            batch_size, seq_len, _ = x.size()
            subsequent_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            subsequent_mask = subsequent_mask.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Masked multi-head self-attn
            attn_output, _ = self.masked_mha(x, x, x, attn_mask=subsequent_mask)
            attn_output = self.dropout1(attn_output)
            x = self.layer_norm1(x + attn_output)
            
            # Pad encoder output if necessary
            if enc_output.size(1) != x.size(1):
                padded_enc_output = pad_sequence(enc_output, batch_first=True)
            else:
                padded_enc_output = enc_output
            
            # Create source attn mask based on encoder padding
            src_padding_mask = (padded_enc_output == 0)  # Assuming padding token is 0
            src_attn_mask = src_padding_mask.unsqueeze(1).expand(batch_size, seq_len, -1)
            
            # Multi-head attn over encoder output
            attn_output, _ = self.mha(x, padded_enc_output, padded_enc_output, attn_mask=src_attn_mask)
            attn_output = self.dropout2(attn_output)

        else:
            # Masked multi-head self-attn
            attn_output, _ = self.masked_mha(x, x, x, attn_mask=tgt_mask)
            attn_output = self.dropout1(attn_output)
            x = self.layer_norm1(x + attn_output)
            
            # Multi-head attn over encoder output: query=x, key=value=enc_output
            attn_output, _ = self.mha(x, enc_output, enc_output, attn_mask=src_mask)
            attn_output = self.dropout2(attn_output)

        x = self.layer_norm2(x + attn_output)
        
        # Feed-forward network
        ff_output = self.feed_forward(x)
        ff_output = self.dropout3(ff_output)
        x = self.layer_norm3(x + ff_output)
        
        return x


class BoringTransformerBlock(nn.Module):
    '''
    Act like EncoderBlock or DecoderBlock in a transformer model.
    Assume the input already padded.
    '''
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(BoringDecoderBlock, self).__init__()
        
        self.masked_mha = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm1 = LayerNorm(d_model)
        
        self.mha = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm2 = LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout3 = nn.Dropout(dropout)
        self.layer_norm3 = LayerNorm(d_model)
    
    def forward(self, x, enc_output=None, src_mask=None, tgt_mask=None):
        '''
        if enc_output is None, then it's an encoder block
        src_mask for decoder self-attn
        tgt_mast for encoder-decoder attn
        '''

        # Masked multi-head self-attn
        attn_output, _ = self.masked_mha(x, x, x, attn_mask=tgt_mask)
        attn_output = self.dropout1(attn_output)
        x = self.layer_norm1(x + attn_output)
        
        if enc_output is not None:
            # Multi-head attn over encoder output: query=x, key=value=enc_output
            attn_output, _ = self.mha(x, enc_output, enc_output, attn_mask=src_mask)
            attn_output = self.dropout2(attn_output)
            x = self.layer_norm2(x + attn_output)
        
        # Feed-forward network
        ff_output = self.feed_forward(x)
        ff_output = self.dropout3(ff_output)
        x = self.layer_norm3(x + ff_output)
        
        return x


class BoringTransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout=0.1):
        super(BoringTransformerModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = LearnedPositionalEncoding(d_model, dropout, max_len)
        
        self.encoder_layers = nn.ModuleList([
            BoringEncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            BoringDecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        
        self.linear = nn.Linear(d_model, vocab_size)
    
    def forward(self, src, tgt):
        # Apply input embedding and positional encoding
        src_embed = self.embedding(src)
        src_embed = self.pos_encoding(src_embed)
        
        tgt_embed = self.embedding(tgt)
        tgt_embed = self.pos_encoding(tgt_embed)
        
        # Pass the input through the encoder layers
        enc_output = src_embed
        for encoder_layer in self.encoder_layers:
            enc_output = encoder_layer(enc_output)
        
        # Pass the encoder output and target through the decoder layers
        dec_output = tgt_embed
        for decoder_layer in self.decoder_layers:
            dec_output = decoder_layer(dec_output, enc_output)
        
        # Apply a linear transformation to get the final output
        output = self.linear(dec_output)
        
        return output

