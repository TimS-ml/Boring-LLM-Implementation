'''
On LayerNorm's location: https://arxiv.org/abs/2002.04745

- https://nn.labml.ai/transformers/models.html
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange

from torch import Tensor, Size
from typing import Optional, Tuple, Union, List

from boring_nn.attention import MultiHeadAttention
from boring_nn.norm import LayerNorm
from boring_nn.pe import SinusoidalPositionalEncoding, LearnedPositionalEncoding
from boring_nn.utils import cprint


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
    
    def forward(self, x, mask=None):
        '''
        padded_x:     (batch_size, max_sequence_length, d_model)
        padding_mask: (batch_size, max_sequence_length)
        attn_mask:    (batch_size * num_heads, max_sequence_length, max_sequence_length)
        '''
        # Multi-head self-attn
        # print('=' * 40)
        # cprint('MHA self-attn', c='normal')
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
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        '''
        src_mask for decoder self-attn
        tgt_mast for encoder-decoder attn
        '''
        # Masked multi-head self-attn
        print('=' * 40)
        cprint('tgt_mask MHA self-attn', c='normal')
        attn_output, _ = self.masked_mha(x, x, x, attn_mask=tgt_mask)
        attn_output = self.dropout1(attn_output)
        x = self.layer_norm1(x + attn_output)
        
        # Multi-head attn over encoder output: query=x, key=value=enc_output
        print('=' * 40)
        cprint('src_mask MHA encoder-decoder attn', c='normal')
        attn_output, _ = self.mha(x, enc_output, enc_output, attn_mask=src_mask)
        attn_output = self.dropout2(attn_output)
        x = self.layer_norm2(x + attn_output)
        
        # Feed-forward network
        ff_output = self.feed_forward(x)
        ff_output = self.dropout3(ff_output)
        x = self.layer_norm3(x + ff_output)
        
        return x


# TODO: update this
class BoringTransformerBlock(nn.Module):
    '''
    Act like EncoderBlock or DecoderBlock in a transformer model.
    Assume the input already padded.
    '''
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(BoringTransformerBlock, self).__init__()
        
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
        # cprint('Masked multi-head self-attn', c='normal')
        attn_output, _ = self.masked_mha(x, x, x, attn_mask=tgt_mask)
        attn_output = self.dropout1(attn_output)
        x = self.layer_norm1(x + attn_output)
        
        if enc_output is not None:
            # Multi-head attn over encoder output: query=x, key=value=enc_output
            # cprint('Multi-head attn over encoder output', c='normal')
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

