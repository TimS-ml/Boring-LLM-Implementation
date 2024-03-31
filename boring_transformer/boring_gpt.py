import torch
import torch.nn as nn
from broing_nn.attention import MultiHeadAttention
from broing_nn.norm import LayerNorm
from broing_nn.pe import LearnedPositionalEncoding


class BoringGPTBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(BoringGPTBlock, self).__init__()
        
        self.masked_mha = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm1 = LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm2 = LayerNorm(d_model)
    
    def forward(self, x, attn_mask=None):
        # Masked multi-head self-attention
        attn_output, _ = self.masked_mha(x, x, x, attn_mask=attn_mask)
        attn_output = self.dropout1(attn_output)
        x = self.layer_norm1(x + attn_output)
        
        # Feed-forward network
        ff_output = self.feed_forward(x)
        ff_output = self.dropout2(ff_output)
        x = self.layer_norm2(x + ff_output)
        
        return x

class BoringGPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout=0.1):
        super(BoringGPT, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = LearnedPositionalEncoding(d_model, dropout, max_len)
        
        self.blocks = nn.ModuleList([
            BoringGPTBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        
        self.layer_norm = LayerNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, attn_mask=None):
        # Apply input embedding and positional encoding
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        # Pass the input through the decoder blocks
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Apply a linear transformation to get the final output
        output = self.linear(x)
        
        return output
    
    def generate(self, context, max_new_tokens, temperature=1.0, top_k=None):
        # TODO: Implement the text generation method
        pass