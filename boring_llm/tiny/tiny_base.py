from __future__ import annotations
from functools import partial

import torch
from torch.nn import Module
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from einops import rearrange

from typing import Optional, Tuple, Union, List, Any, Generator, Type, Callable
from jaxtyping import Float, Bool

from boring_utils.utils import get_device, cprint, tprint
from boring_utils.helpers import DEBUG
# import os; os.environ['DEBUG'] = '3'

# %% [markdown]
# # Config

# %%
# tiny configs
from boring_llm.base.tiny_config import *
device = get_device()

# %% [markdown]
# # FFN

# %%
class TinyFeedForward(nn.Module):
    def __init__(self, dim: int, mul: int = 4, dropout: float = 0.):
        super().__init__()

        hidden_dim = dim * mul
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
    
    def forward(
            self, 
            x: Float[Tensor, "batch seq_len embd"]
        ) -> Float[Tensor, "batch seq_len embd"]:
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def test_Tiny_feed_forward():
    model = TinyFeedForward(EMBEDDING_DIM)
    x = torch.randn(BATCH_SIZE, BLOCK_SIZE, EMBEDDING_DIM)
    out = model(x)
    assert out.shape == x.shape, "Output shape should match input shape"
    
    mul = 2
    model = TinyFeedForward(EMBEDDING_DIM, mul=mul)
    assert model.fc1.weight.shape == (EMBEDDING_DIM * mul, EMBEDDING_DIM), "Hidden layer dimensions should be correct"
    assert model.fc2.weight.shape == (EMBEDDING_DIM, EMBEDDING_DIM * mul), "Output layer dimensions should be correct"
    
    tprint("All tests passed!")

if DEBUG > 1: test_Tiny_feed_forward()

# %% [markdown]
# # SDPA

# %%
def create_causal_mask(seq_q: int, seq_k: int, device: torch.device = device):
    """
    Create a causal mask for attention.
    Args:
        seq_q: sequence length of query
        seq_k: sequence length of key
        device: device to create mask on
    Returns:
        Upper triangular boolean mask of shape (seq_q, seq_k)
    """
    return torch.triu(
            torch.ones(
                (seq_q, seq_k), 
                device=device, 
                dtype=torch.bool
            ), 
            diagonal=1
        )


class TinyScaleDotProductAttention(nn.Module):
    def __init__(self, causal:bool = False, dropout:float = 0.):
        """
        Scale Dot Product Attention module.
        Args:
            causal: whether to use causal masking
            dropout: dropout probability
        """
        super().__init__()
        self.causal = causal
        self.dropout = nn.Dropout(dropout)

    def forward(
            self, 
            q: Float[Tensor, "batch head seq_q embd"], 
            k: Float[Tensor, "batch head seq_k embd"], 
            v: Float[Tensor, "batch head seq_k embd"], 
            # mask: Optional[Float[Tensor, "batch head seq_q seq_k"]] = None
        ) -> Float[Tensor, "batch head seq_q embd"]:
        """
        Forward pass for attention.
        Args:
            q: query tensor of shape (batch, head, seq_q, embd)
            k: key tensor of shape (batch, head, seq_k, embd) 
            v: value tensor of shape (batch, head, seq_k, embd)
        Returns:
            Output tensor of shape (batch, head, seq_q, embd)
        """
        scale = q.shape[-1] ** -0.5
        seq_q, seq_k = q.shape[-2], k.shape[-2]

        # Compute scaled dot product attention scores
        # batch head seq_q embd, batch head seq_k embd -> batch head seq_q seq_k
        qk_sim = einsum(
            "b h i d, b h j d -> b h i j", q, k) * scale
        
        # Apply causal mask if needed
        if self.causal:
            mask = create_causal_mask(seq_q, seq_k, device=q.device)
            mask_val = torch.finfo(qk_sim.dtype).min  # Use minimum value for dtype, better than -float("inf")
            qk_sim = qk_sim.masked_fill(mask, mask_val)

        # Compute attention weights and apply dropout
        attn = F.softmax(qk_sim, dim=-1)  # Softmax over key dimension
        attn = self.dropout(attn)

        # Compute weighted sum of values
        # batch head seq_q seq_k, batch head seq_k embd -> batch head seq_q embd
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        return out


def test_scale_dot_product_attention():
    seq_q, seq_k = BLOCK_SIZE, BLOCK_SIZE
    q = torch.randn(BATCH_SIZE, N_HEAD, seq_q, D_HEAD)
    k = torch.randn(BATCH_SIZE, N_HEAD, seq_k, D_HEAD)
    v = torch.randn(BATCH_SIZE, N_HEAD, seq_k, D_HEAD)
    
    # Test non-causal
    attn = TinyScaleDotProductAttention(causal=False)
    out = attn(q, k, v)
    assert out.shape == (BATCH_SIZE, N_HEAD, seq_q, D_HEAD), "Output shape should match input shape"
    
    # Test causal
    attn = TinyScaleDotProductAttention(causal=True)
    out = attn(q, k, v)
    assert out.shape == (BATCH_SIZE, N_HEAD, seq_q, D_HEAD), "Output shape should match input shape"
    
    # Test mask values
    mask = create_causal_mask(seq_q, seq_k)
    assert mask.shape == (seq_q, seq_k), "Mask shape should be (seq_q, seq_k)"
    assert mask.dtype == torch.bool, "Mask should be boolean"
    assert torch.all(mask == torch.triu(torch.ones_like(mask), diagonal=1)), "Mask should be upper triangular"
    
    tprint("All tests passed!")

if DEBUG > 1: test_scale_dot_product_attention()

# %% [markdown]
# # MHA

# %%
class TinyMultiHeadAttention(nn.Module):
    def __init__(self, dim: int, n_head: int, d_head: int, causal: bool = False, dropout: float = 0.):
        super().__init__()
        self.n_head = n_head
        self.scale = d_head ** -0.5
        inner_dim = d_head * n_head  # NOTE: inner_dim could be uneq to dim
        
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.attend = TinyScaleDotProductAttention(causal=causal, dropout=dropout)
    
    def forward(
            self, 
            x: Float[Tensor, "batch seq embd"],
            context: Optional[Float[Tensor, "batch seq embd"]] = None,  # for cross attn
            # mask: Optional[Float[Tensor, "batch head seq_q seq_k"]] = None
        ) -> Float[Tensor, "batch seq embd"]:

        context = x if context is None else context
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        # batch seq_len (n_head d_head) -> batch head seq_len d_head
        q, k, v = map(lambda t: rearrange(t, 'b i (h d) -> b h i d', h=self.n_head), (q, k, v))

        out = self.attend(q, k, v)  # pass the mask here in future
        out = rearrange(out, 'b h i d -> b i (h d)', h=self.n_head)

        return self.to_out(out)


# Make TinyMultiHeadCrossAttention an alias of TinyMultiHeadAttention
TinyMultiHeadCrossAttention = TinyMultiHeadAttention


def test_multi_head_attention():
    # Test self-attention
    mha = TinyMultiHeadAttention(dim=EMBEDDING_DIM, n_head=N_HEAD, d_head=D_HEAD, causal=False)
    x = torch.randn(BATCH_SIZE, BLOCK_SIZE, EMBEDDING_DIM)
    out = mha(x)
    assert out.shape == (BATCH_SIZE, BLOCK_SIZE, EMBEDDING_DIM), "Output shape should match input shape"
    
    # Test cross-attention
    context = torch.randn(BATCH_SIZE, BLOCK_SIZE, EMBEDDING_DIM)
    out = mha(x, context=context)
    assert out.shape == (BATCH_SIZE, BLOCK_SIZE, EMBEDDING_DIM), "Output shape should match input shape"
    
    # Test causal masking
    mha_causal = TinyMultiHeadAttention(dim=EMBEDDING_DIM, n_head=N_HEAD, d_head=D_HEAD, causal=True)
    out = mha_causal(x)
    assert out.shape == (BATCH_SIZE, BLOCK_SIZE, EMBEDDING_DIM), "Output shape should match input shape"
    
    tprint("All tests passed!")

if DEBUG > 1: test_multi_head_attention()

# %% [markdown]
# # Transform
# 
# MHA and FFN
# 
# 
# In x-transformer
# ```python
# # determine default block layer type order
# 
# if cross_attend and not only_cross:
#     default_block = ('a', 'c', 'f')
# elif cross_attend and only_cross:
#     default_block = ('c', 'f')
# else:
#     default_block = ('a', 'f')
# 
# if macaron:
#     default_block = ('f',) + default_block
# ```

# %%
class TinyTransformBlock(nn.Module):
    def __init__(
            self, 
            dim: int, 
            n_layers: int, 
            n_head: int, 
            d_head: int, 
            ffn_mul: int,
            causal: bool = False, 
            cross_attend: bool = False, 
            dropout: float = 0.
            # **kwargs
        ):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        self.causal = causal
        self.cross_attend = cross_attend

        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleList([
                    nn.LayerNorm(dim),
                    TinyMultiHeadAttention(dim, n_head, d_head, causal=causal, dropout=dropout),
                    nn.LayerNorm(dim) if cross_attend else None,
                    TinyMultiHeadCrossAttention(dim, n_head, d_head) if cross_attend else None,
                    nn.LayerNorm(dim),
                    TinyFeedForward(dim, mul=ffn_mul, dropout=dropout)
                ])
            )
        
    def forward(
            self,
            x: Float[Tensor, "batch seq embd"],
            context: Optional[Float[Tensor, "batch seq embd"]] = None,  # for cross attn
            # mask: Optional[Float[Tensor, "batch head seq_q seq_k"]] = None
            # context_mask: Optional[Float[Tensor, "batch seq embd"]] = None,  # for cross attn
            **kwargs
        ) -> Float[Tensor, "batch seq embd"]:

        for norm1, attn, norm2, cross_attn, norm3, ff in self.layers:
            # Self-attention
            x = x + attn(norm1(x))
            
            # Cross-attention (if available)
            if cross_attn:
                x = x + cross_attn(norm2(x), context=context)
                
            # Feedforward
            x = x + ff(norm3(x))

        return x


class TinyEncoder(TinyTransformBlock):
    """Encoder-specific Transform Layers (non-causal)"""
    def __init__(self, **kwargs):
        super().__init__(causal=False, **kwargs)


class TinyDecoder(TinyTransformBlock):
    """
    Decoder-specific Transform Layers (causal)
    Add cross_attend=True to enable cross-attention (say, enc-dec transformer)
    """
    def __init__(self, **kwargs):
        super().__init__(causal=True, **kwargs)


def test_encoder_decoder():
    # Test Encoder
    encoder = TinyEncoder(
        dim=EMBEDDING_DIM,
        n_layers=N_LAYER,
        n_head=N_HEAD,
        d_head=D_HEAD,
        ffn_mul=FFN_MUL
    ).to(device)
    
    x = torch.randn(BATCH_SIZE, BLOCK_SIZE, EMBEDDING_DIM).to(device)
    enc_out = encoder(x)
    assert enc_out.shape == (BATCH_SIZE, BLOCK_SIZE, EMBEDDING_DIM), "Encoder output shape should match input shape"
    
    # Test Decoder
    decoder = TinyDecoder(
        dim=EMBEDDING_DIM,
        n_layers=N_LAYER,
        n_head=N_HEAD,
        d_head=D_HEAD,
        ffn_mul=FFN_MUL
    ).to(device)
    
    # Test decoder with context (cross attention)
    context = torch.randn(BATCH_SIZE, BLOCK_SIZE, EMBEDDING_DIM).to(device)
    dec_out = decoder(x, context=context)
    assert dec_out.shape == (BATCH_SIZE, BLOCK_SIZE, EMBEDDING_DIM), "Decoder output shape should match input shape with context"
    
    # Test decoder without context
    dec_out = decoder(x)
    assert dec_out.shape == (BATCH_SIZE, BLOCK_SIZE, EMBEDDING_DIM), "Decoder output shape should match input shape without context"
    
    tprint("All tests passed!")


def test_Tiny_transform():
    # Test basic forward pass
    model = TinyTransformBlock(
        dim=EMBEDDING_DIM,
        n_layers=N_LAYER,
        n_head=N_HEAD,
        d_head=D_HEAD,
        causal=False,
        ffn_mul=FFN_MUL
    ).to(device)
    
    x = torch.randn(BATCH_SIZE, BLOCK_SIZE, EMBEDDING_DIM).to(device)
    out = model(x)
    assert out.shape == (BATCH_SIZE, BLOCK_SIZE, EMBEDDING_DIM), "Output shape should match input shape"
    
    # Test with context (cross attention)
    context = torch.randn(BATCH_SIZE, BLOCK_SIZE, EMBEDDING_DIM).to(device)
    out = model(x, context=context)
    assert out.shape == (BATCH_SIZE, BLOCK_SIZE, EMBEDDING_DIM), "Output shape should match input shape with context"
    
    # Test causal masking
    model_causal = TinyTransformBlock(
        dim=EMBEDDING_DIM,
        n_layers=N_LAYER,
        n_head=N_HEAD,
        d_head=D_HEAD,
        causal=True,
        ffn_mul=FFN_MUL
    ).to(device)
    
    out = model_causal(x)
    assert out.shape == (BATCH_SIZE, BLOCK_SIZE, EMBEDDING_DIM), "Output shape should match input shape with causal masking"
    
    tprint("All tests passed!")

if DEBUG > 1: 
    test_Tiny_transform()
    test_encoder_decoder()

# %% [markdown]
# # TinyAutoregressive Wrapper

# %%
class TinyAutoregressiveWrapper(nn.Module):
    """Adds autoregressive capabilities to a transformer"""
    def __init__(self, net, pad_value=0):
        super().__init__()
        self.net = net
        self.pad_value = pad_value
        
    def forward(
            self, 
            x: Float[Tensor, "batch seq"],
            context: Optional[Float[Tensor, "batch seq embd"]] = None,
            **kwargs
        ) -> Float[Tensor, "batch seq-1"]:
        # Training: shifted targets
        x_input, x_target = x[:, :-1], x[:, 1:]
        
        # Get logits
        # "batch seq-1 num_tokens" -> "batch num_tokens seq-1"
        # logits = self.net(x_input, mask=mask[:, :-1] if mask is not None else None)
        if context is not None:
            logits = self.net(x_input, context=context, **kwargs)
        else:
            logits = self.net(x_input, **kwargs) 

        # Calculate loss
        loss = F.cross_entropy(
            # batch 
            rearrange(logits, 'b n c -> b c n'),
            x_target,
            ignore_index=self.pad_value
        )
        
        return loss
    
    @torch.no_grad()
    def generate(
            self, 
            start_tokens: Float[Tensor, "batch seq_t"], 
            seq_len: int, 
            context: Optional[Float[Tensor, "batch seq embd"]] = None,
            temperature: float = 1.0,
            **kwargs
        ) -> Float[Tensor, "batch seq_len"]:
        # device = start_tokens.device
        batch, t = start_tokens.shape

        # "batch seq_t" -> "batch seq_t+seq_len"
        out = start_tokens
        
        for _ in range(seq_len):
            # Get the maximum sequence length the network can handle
            # NOTE: this is from the early version of x-transformers, the latest version is using `restrict_to_max_seq_len` which is more flexible
            # remember to add mask = mask[:, -self.max_seq_len:] in future
            max_seq_len = getattr(self.net, 'max_seq_len', self.net.pos_emb.shape[1])
            x = out[:, -max_seq_len:]

            # Get predictions
            if context is not None:
                logits = self.net(x, context=context, **kwargs)
            else:
                logits = self.net(x, **kwargs)
            
            # Get the last token prediction
            logits = logits[:, -1]

            # Apply temperature
            probs = F.softmax(logits / temperature, dim=-1)
            
            # Sample
            sample = torch.multinomial(probs, 1)
            
            # Append to sequence
            out = torch.cat((out, sample), dim=-1)
            
        # Return only the newly generated tokens
        return out[:, t:]

# %% [markdown]
# # Transformer Wrapper
# 
# We gonna implement enc-dec transformer later

# %%
class TinyTransformerWrapper(nn.Module):
    """TinyTransformBlock with pre and post processing"""
    def __init__(
            self, 
            num_tokens: int,
            max_seq_len: int,
            dim: int, 
            n_layers: int, 
            n_head: int, 
            d_head: int, 
            ffn_mul: int,
            dropout: float = 0.,
            cross_attend: bool = False, 
            transform_layer: TinyTransformBlock = TinyDecoder,
            return_only_embed: bool = False,  # NOTE: for enc-dec transformer's decoder, return_only_embed=True
            **kwargs
        ):
        super().__init__()
        self.dim = dim  # for dim check
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, dim))
        self.max_seq_len = max_seq_len  # explicitly store max sequence length

        self.transformer = transform_layer(
            dim=dim,
            n_layers=n_layers,
            n_head=n_head,
            d_head=d_head,
            ffn_mul=ffn_mul,
            dropout=dropout,
            cross_attend=cross_attend,
            **kwargs
        )

        if return_only_embed:
            self.to_logits = nn.Identity()
        else:
            self.to_logits = nn.Linear(dim, num_tokens, bias=False)  # IMPORTANT: no bias!!!

    def forward(
            self, 
            x: Float[Tensor, "batch seq"],
            context: Optional[Float[Tensor, "batch seq embd"]] = None,  # for cross attn
            # mask: Optional[Float[Tensor, "batch head seq_q seq_k"]] = None
            # context_mask: Optional[Float[Tensor, "batch seq embd"]] = None,  # for cross attn
            **kwargs
        ) -> Float[Tensor, "embd num_tokens"]:
        batch, seq_len = x.shape
        
        # Truncate sequence if it exceeds max length
        if seq_len > self.max_seq_len:
            x = x[:, -self.max_seq_len:]
            seq_len = self.max_seq_len

        x = self.token_emb(x)
        x = x + self.pos_emb[:, :seq_len]

        if self.transformer.cross_attend:
            x = self.transformer(x, context=context)
        else:
            x = self.transformer(x)

        return self.to_logits(x)
        

def test_Tiny_transformer_wrapper():
    # Create model
    model = TinyTransformerWrapper(
        num_tokens=NUM_TOKENS,
        max_seq_len=BLOCK_SIZE,
        dim=EMBEDDING_DIM,
        n_layers=N_LAYER,
        n_head=N_HEAD,
        d_head=D_HEAD,
        ffn_mul=FFN_MUL,
        dropout=DROPOUT
    ).to(device)
    
    # Create test input
    x = torch.randint(0, NUM_TOKENS, (BATCH_SIZE, BLOCK_SIZE)).to(device)
    
    # Forward pass
    logits = model(x)
    
    # Check output shape
    assert logits.shape == (BATCH_SIZE, BLOCK_SIZE, NUM_TOKENS), \
        f"Expected output shape {(BATCH_SIZE, BLOCK_SIZE, NUM_TOKENS)}, got {logits.shape}"
    
    tprint("All tests passed!")

if DEBUG > 1: test_Tiny_transformer_wrapper()

# %% [markdown]
# # Tiny Transformer

# %%
class TinyEncDecTransformer(nn.Module):
    """Encoder-Decoder transformer"""
    def __init__(
        self,
        *,
        dim: int,
        # encoder
        enc_num_tokens: int,
        enc_n_layers: int,
        enc_n_head: int = 8,
        enc_max_seq_len: int = 512,
        # decoder
        dec_num_tokens: int,
        dec_n_layers: int,
        dec_n_head: int = 8,
        dec_max_seq_len: int = 512,
        # misc
        ffn_mul: int = 4,
        tie_token_emb: bool = False,
        dropout: float = 0.
    ):
        super().__init__()
        
        # Encoder
        self.encoder = TinyTransformerWrapper(
            num_tokens=enc_num_tokens,
            max_seq_len=enc_max_seq_len,
            dim=dim,
            n_layers=enc_n_layers,
            n_head=enc_n_head,
            d_head=dim // enc_n_head,
            ffn_mul=ffn_mul,
            dropout=dropout,
            transform_layer=TinyEncoder,
            return_only_embed=True
        )
        
        # Decoder (with cross-attention)
        decoder = TinyTransformerWrapper(
            num_tokens=dec_num_tokens,
            max_seq_len=dec_max_seq_len,
            dim=dim,
            n_layers=dec_n_layers,
            n_head=dec_n_head,
            d_head=dim // dec_n_head,
            ffn_mul=ffn_mul,
            dropout=dropout,
            cross_attend=True,
            transform_layer=TinyDecoder
        )
        
        # Token embedding sharing if specified
        if tie_token_emb:
            decoder.token_emb = self.encoder.token_emb
            
        # Wrap decoder with autoregressive capabilities
        self.decoder = TinyAutoregressiveWrapper(decoder)

    def forward(
            self, 
            src: Float[Tensor, "batch src_seq"], 
            tgt: Float[Tensor, "batch tgt_seq"], 
            # src_mask: Optional[Float[Tensor, "batch src_seq"]] = None, 
            # tgt_mask: Optional[Float[Tensor, "batch tgt_seq"]] = None,
            **kwargs
        ) -> Float[Tensor, "batch tgt_seq vocab"]:
        # Encode source sequence, then decode target sequence with encoder output as context
        # enc_out = self.encoder(src, mask=src_mask)
        # out = self.decoder(tgt, context=enc_out, mask=tgt_mask, context_mask=src_mask) 

        enc_out = self.encoder(src, **kwargs)
        out = self.decoder(tgt, context=enc_out, **kwargs) 
        return out
    
    @torch.no_grad()
    def generate(
            self, 
            src: Float[Tensor, "batch src_seq"], 
            tgt_start: Float[Tensor, "batch start_seq"], 
            seq_len: int, 
            src_mask: Optional[Float[Tensor, "batch src_seq"]] = None,
            **kwargs
        ) -> Float[Tensor, "batch seq_len"]:
        # Encode source sequence, then decode target sequence with encoder output as context
        enc_out = self.encoder(src, mask=src_mask, **kwargs)
        out = self.decoder.generate(tgt_start, seq_len, context=enc_out, context_mask=src_mask, **kwargs)
        return out

def test_Tiny_enc_dec_transformer():
    # Test config
    ENC_NUM_TOKENS = int(NUM_TOKENS * 0.5)
    DEC_NUM_TOKENS = NUM_TOKENS
    ENC_N_LAYER = N_LAYER
    DEC_N_LAYER = N_LAYER
    ENC_N_HEAD = N_HEAD
    DEC_N_HEAD = N_HEAD
    
    # Create model
    model = TinyEncDecTransformer(
        dim=EMBEDDING_DIM,
        # encoder
        enc_num_tokens=ENC_NUM_TOKENS,
        enc_n_layers=ENC_N_LAYER,
        enc_n_head=ENC_N_HEAD,
        enc_max_seq_len=BLOCK_SIZE,
        # decoder
        dec_num_tokens=DEC_NUM_TOKENS,
        dec_n_layers=DEC_N_LAYER,
        dec_n_head=DEC_N_HEAD,
        dec_max_seq_len=BLOCK_SIZE,
        # misc
        tie_token_emb=False,
        ffn_mul=FFN_MUL,
        dropout=DROPOUT
    ).to(device)
    
    # Create test inputs
    src = torch.randint(0, ENC_NUM_TOKENS, (BATCH_SIZE, BLOCK_SIZE)).to(device)
    tgt = torch.randint(0, DEC_NUM_TOKENS, (BATCH_SIZE, BLOCK_SIZE)).to(device)
    
    # Forward pass
    logits = model(src, tgt)
    
    # Check output shape
    # cprint(logits)
    assert isinstance(logits, torch.Tensor) and logits.dim() == 0, \
        f"Expected output to be a scalar loss value, got shape {logits.shape}"
    assert torch.isfinite(logits), f"Expected a finite loss value, got {logits}"
    
    # Test generation
    tgt_start = torch.randint(0, DEC_NUM_TOKENS, (BATCH_SIZE, 1)).to(device)
    generated = model.generate(src, tgt_start, seq_len=BLOCK_SIZE)
    
    # Check generation shape
    assert generated.shape == (BATCH_SIZE, BLOCK_SIZE), \
        f"Expected generation shape {(BATCH_SIZE, BLOCK_SIZE)}, got {generated.shape}"
    
    tprint("All tests passed!")

if DEBUG > 1: test_Tiny_enc_dec_transformer()

