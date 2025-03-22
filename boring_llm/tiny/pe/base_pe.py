from __future__ import annotations
import torch
from torch import nn, Tensor

from typing import Optional, Type
from jaxtyping import Float

from boring_utils.utils import get_device, cprint, tprint
from boring_utils.helpers import DEBUG

# Import the base implementation
from boring_llm.tiny.tiny_base import (
    TinyTransformBlock, 
    TinyDecoder,
)

# Import positional encoding components
from boring_llm.nn.pe.config import PositionalEncodingType
from boring_llm.nn.pe.factory import PositionalEncodingFactory
from boring_llm.nn.pe.base import PositionalEncoding

from boring_llm.base.tiny_config import *
device = get_device()


class PositionalEmbeddingTransformerWrapper(nn.Module):
    """TinyTransformBlock with customizable positional embeddings"""
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
            transform_layer: Type[TinyTransformBlock] = TinyDecoder,
            return_only_embed: bool = False,
            pe_type: PositionalEncodingType = PositionalEncodingType.FIXED,
            l2norm_embed: bool = False
        ):
        super().__init__()
        self.dim = dim  # for dim check
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.max_seq_len = max_seq_len  # explicitly store max sequence length

        # Create positional embedding based on specified type
        self.pe_type = pe_type
        if pe_type == PositionalEncodingType.NONE:
            self.pos_emb = None
        else:
            # Use the factory to create the appropriate positional encoding
            self.pos_emb = PositionalEncodingFactory.create(
                encoding_type=pe_type.value,
                dim=dim,
                max_seq_len=max_seq_len,
                l2norm_embed=l2norm_embed
            )

        self.transformer = transform_layer(
            dim=dim,
            n_layers=n_layers,
            n_head=n_head,
            d_head=d_head,
            ffn_mul=ffn_mul,
            dropout=dropout,
            cross_attend=cross_attend
        )

        if return_only_embed:
            self.to_logits = nn.Identity()
        else:
            self.to_logits = nn.Linear(dim, num_tokens, bias=False)  # IMPORTANT: no bias!!!

    def forward(
            self, 
            x: Float[Tensor, "batch seq"],
            context: Optional[Float[Tensor, "batch seq embd"]] = None,
            **kwargs
        ) -> Float[Tensor, "embd num_tokens"]:
        batch, seq_len = x.shape
        
        # Truncate sequence if it exceeds max length
        if seq_len > self.max_seq_len:
            x = x[:, -self.max_seq_len:]
            seq_len = self.max_seq_len

        # Get token embeddings
        x = self.token_emb(x)
        
        # Add positional embeddings if specified
        if self.pos_emb is not None:
            # Uses the unified PositionalEncoding interface
            pos_emb = self.pos_emb(x)
            x = x + pos_emb

        # Forward through transformer
        if self.transformer.cross_attend:
            x = self.transformer(x, context=context)
        else:
            x = self.transformer(x)

        return self.to_logits(x)


def test_positional_embedding_transformer():
    # Test Transformer with absolute positional embedding
    model_abs = PositionalEmbeddingTransformerWrapper(
        num_tokens=NUM_TOKENS,
        max_seq_len=BLOCK_SIZE,
        dim=EMBEDDING_DIM,
        n_layers=N_LAYER,
        n_head=N_HEAD,
        d_head=D_HEAD,
        ffn_mul=FFN_MUL,
        dropout=DROPOUT,
        pe_type=PositionalEncodingType.ABSOLUTE
    ).to(device)
    
    # Test Transformer with fixed positional embedding
    model_fixed = PositionalEmbeddingTransformerWrapper(
        num_tokens=NUM_TOKENS,
        max_seq_len=BLOCK_SIZE,
        dim=EMBEDDING_DIM,
        n_layers=N_LAYER,
        n_head=N_HEAD,
        d_head=D_HEAD,
        ffn_mul=FFN_MUL,
        dropout=DROPOUT,
        pe_type=PositionalEncodingType.FIXED
    ).to(device)
    
    # Test Transformer without positional embedding
    model_none = PositionalEmbeddingTransformerWrapper(
        num_tokens=NUM_TOKENS,
        max_seq_len=BLOCK_SIZE,
        dim=EMBEDDING_DIM,
        n_layers=N_LAYER,
        n_head=N_HEAD,
        d_head=D_HEAD,
        ffn_mul=FFN_MUL,
        dropout=DROPOUT,
        pe_type=PositionalEncodingType.NONE
    ).to(device)
    
    # Create test input
    x = torch.randint(0, NUM_TOKENS, (BATCH_SIZE, BLOCK_SIZE)).to(device)
    
    # Forward pass for all models
    logits_abs = model_abs(x)
    logits_fixed = model_fixed(x)
    logits_none = model_none(x)
    
    # Check output shapes
    assert logits_abs.shape == (BATCH_SIZE, BLOCK_SIZE, NUM_TOKENS), \
        f"Expected output shape {(BATCH_SIZE, BLOCK_SIZE, NUM_TOKENS)}, got {logits_abs.shape}"
    assert logits_fixed.shape == (BATCH_SIZE, BLOCK_SIZE, NUM_TOKENS), \
        f"Expected output shape {(BATCH_SIZE, BLOCK_SIZE, NUM_TOKENS)}, got {logits_fixed.shape}"
    assert logits_none.shape == (BATCH_SIZE, BLOCK_SIZE, NUM_TOKENS), \
        f"Expected output shape {(BATCH_SIZE, BLOCK_SIZE, NUM_TOKENS)}, got {logits_none.shape}"
    
    tprint("All tests passed!")


if __name__ == "__main__":
    if DEBUG > 1: test_positional_embedding_transformer()