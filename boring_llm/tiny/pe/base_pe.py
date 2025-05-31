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

# Import simplified positional encoding components
from boring_llm.nn.pe import create_pe, PEConfig
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
            pe_type: str = "fixed",
            l2norm_embed: bool = False,
            **pe_kwargs  # Additional PE-specific arguments
        ):
        super().__init__()
        self.dim = dim  # for dim check
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.max_seq_len = max_seq_len  # explicitly store max sequence length

        # Create positional embedding based on specified type
        self.pe_type = pe_type

        # Create simplified positional encoding using new interface
        if pe_type != "none":
            pe_config_kwargs = {
                "dim_model": dim,
                "max_seq_len": max_seq_len,
            }
            
            # Add type-specific arguments
            if pe_type == "absolute":
                pe_config_kwargs["l2norm_embed"] = l2norm_embed
            elif pe_type == "rotary":
                pe_config_kwargs.update({
                    "rotary_percentage": pe_kwargs.get("rotary_percentage", 1.0),
                    "rope_base": pe_kwargs.get("rope_base", 10000)
                })
            elif pe_type == "alibi":
                if "alibi_num_heads" not in pe_kwargs:
                    raise ValueError("alibi_num_heads must be specified for ALiBi encoding")
                pe_config_kwargs["alibi_num_heads"] = pe_kwargs["alibi_num_heads"]
            
            # Additional kwargs from pe_kwargs
            pe_config_kwargs.update({k: v for k, v in pe_kwargs.items() 
                                   if k not in ["rotary_percentage", "rope_base", "alibi_num_heads"]})
            
            self.pos_emb = create_pe(pe_type=pe_type, **pe_config_kwargs)
        else:
            self.pos_emb = None

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
            # Create input tensor for PE (dummy tensor with correct shape for PE interface)
            dummy_input = torch.zeros(batch, seq_len, self.dim, device=x.device)
            pos_emb = self.pos_emb(dummy_input)
            
            # Handle different PE return types
            if pos_emb is not None:
                if pos_emb.dim() == 2:  # [seq_len, dim] - expand to batch
                    pos_emb = pos_emb.unsqueeze(0).expand(batch, -1, -1)
                elif pos_emb.dim() == 4:  # ALiBi case [num_heads, seq_len, seq_len]
                    # ALiBi returns bias for attention, not positional embeddings to add
                    # Store it for potential use in attention (would need to pass through transformer)
                    pass  
                else:  # [batch, seq_len, dim] or other cases
                    pass
                
                # Only add if it's a standard positional embedding (not ALiBi bias)
                if pos_emb is not None and pos_emb.dim() in [2, 3] and pos_emb.shape[-1] == self.dim:
                    x = x + pos_emb

        # Forward through transformer
        if self.transformer.cross_attend:
            x = self.transformer(x, context=context)
        else:
            x = self.transformer(x)

        return self.to_logits(x)


# ============= Convenience Functions =============
def create_pe_transformer(
    num_tokens: int,
    max_seq_len: int, 
    dim: int,
    n_layers: int,
    n_head: int,
    d_head: int,
    ffn_mul: int,
    pe_type: str = "fixed",
    dropout: float = 0.0,
    **kwargs
) -> PositionalEmbeddingTransformerWrapper:
    """Convenience function to create transformer with PE"""
    return PositionalEmbeddingTransformerWrapper(
        num_tokens=num_tokens,
        max_seq_len=max_seq_len,
        dim=dim,
        n_layers=n_layers,
        n_head=n_head,
        d_head=d_head,
        ffn_mul=ffn_mul,
        dropout=dropout,
        pe_type=pe_type,
        **kwargs
    )


def test_positional_embedding_transformer():
    """Test different PE types with the transformer"""
    test_configs = [
        {
            "name": "Absolute PE", 
            "pe_type": "absolute", 
            "l2norm_embed": True
        },
        {
            "name": "Fixed PE", 
            "pe_type": "fixed"
        },
        {
            "name": "No PE", 
            "pe_type": "none"
        },
        {
            "name": "Rotary PE",
            "pe_type": "rotary",
            "rotary_percentage": 0.5,
            "rope_base": 10000
        }
    ]
    
    models = {}
    
    for config in test_configs:
        name = config.pop("name")
        tprint(f"Creating {name}...")
        
        model = create_pe_transformer(
            num_tokens=NUM_TOKENS,
            max_seq_len=BLOCK_SIZE,
            dim=EMBEDDING_DIM,
            n_layers=N_LAYER,
            n_head=N_HEAD,
            d_head=D_HEAD,
            ffn_mul=FFN_MUL,
            dropout=DROPOUT,
            **config
        ).to(device)
        
        models[name] = model
    
    # Create test input
    x = torch.randint(0, NUM_TOKENS, (BATCH_SIZE, BLOCK_SIZE)).to(device)
    
    # Test all models
    for name, model in models.items():
        tprint(f"Testing {name}...")
        logits = model(x)
        
        # Check output shape
        expected_shape = (BATCH_SIZE, BLOCK_SIZE, NUM_TOKENS)
        assert logits.shape == expected_shape, \
            f"{name}: Expected output shape {expected_shape}, got {logits.shape}"
        
        tprint(f"{name} output shape: {logits.shape} ✓")
    
    tprint("All tests passed! 🎉")


# ============= Alternative Implementation with Direct PE Usage =============
class SimplifiedPETransformer(nn.Module):
    """Even more simplified transformer using PE directly"""
    
    def __init__(
        self,
        num_tokens: int,
        dim: int,
        n_layers: int,
        n_head: int,
        pe_config: PEConfig,
        **transformer_kwargs
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = create_pe(pe_config.type, **pe_config.model_dump())
        
        self.transformer = TinyDecoder(
            dim=dim,
            n_layers=n_layers,
            n_head=n_head,
            **transformer_kwargs
        )
        
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.token_emb(x)
        
        # Add positional encoding
        dummy_input = torch.zeros_like(x)
        pos_emb = self.pos_emb(dummy_input)
        if pos_emb is not None and pos_emb.shape[-1] == x.shape[-1]:
            x = x + pos_emb
            
        x = self.transformer(x)
        return self.to_logits(x)


if __name__ == "__main__":
    if DEBUG > 1: 
        test_positional_embedding_transformer()
        
        # Test simplified version
        tprint("\nTesting SimplifiedPETransformer...")
        pe_config = PEConfig(type="fixed", dim_model=EMBEDDING_DIM, max_seq_len=BLOCK_SIZE)
        simple_model = SimplifiedPETransformer(
            num_tokens=NUM_TOKENS,
            dim=EMBEDDING_DIM,
            n_layers=N_LAYER,
            n_head=N_HEAD,
            pe_config=pe_config,
            d_head=D_HEAD,
            ffn_mul=FFN_MUL,
            dropout=DROPOUT
        ).to(device)
        
        x = torch.randint(0, NUM_TOKENS, (BATCH_SIZE, BLOCK_SIZE)).to(device)
        logits = simple_model(x)
        tprint(f"SimplifiedPETransformer output shape: {logits.shape} ✓")