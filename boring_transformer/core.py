from pydantic import BaseModel, Field
from typing import Optional, List
import torch
import torch.nn as nn
from torch import Tensor

from boring_nn.attention.core import AttentionConfig
from boring_nn.attention.attn import BoringAttention


class TransformerLayerWrapConfig(BaseModel):
    attention: AttentionConfig     = Field(default_factory=AttentionConfig, description="Attention configuration")
    d_model: Optional[int]         = Field(512,   description="Model dimension")
    ffn_dim: Optional[int]         = Field(2048,  description="Feed-forward network dimension")
    dropout: Optional[float]       = Field(0.1,   description="Dropout rate")
    use_ffn: Optional[bool]        = Field(True,  description="Whether to use feed-forward network")


class TransformerLayersConfig(BaseModel):
    dim: Optional[int]                 = Field(512,   description="Model dimension")
    depth: Optional[int]               = Field(6,     description="Number of transformer layers")
    causal: Optional[bool]             = Field(False, description="Whether the model is causal")
    cross_attend: Optional[bool]       = Field(False, description="Whether to use cross-attention")
    only_cross: Optional[bool]         = Field(False, description="Whether to use only cross-attention")
    use_scalenorm: Optional[bool]      = Field(False, description="Whether to use ScaleNorm instead of LayerNorm")
    use_rezero: Optional[bool]         = Field(False, description="Whether to use ReZero initialization")
    rel_pos_bias: Optional[bool]       = Field(False, description="Whether to use relative positional bias")
    custom_layers: Optional[List[str]] = Field(None, description="Custom layer configuration")
    sandwich_coef: Optional[float]     = Field(None, description="Sandwich coefficient for layer configuration")
    macaron: Optional[bool]            = Field(False, description="Whether to use Macaron architecture")
    layer_config: TransformerLayerWrapConfig = Field(default_factory=TransformerLayerWrapConfig, description="Layer configuration")


class BoringTransformerLayerWrap(nn.Module):
    def __init__(self, config: TransformerLayerWrapConfig):
        super().__init__()
        self.config = config
        self.attention = BoringAttention(config.attention)
        self.layer_norm = nn.LayerNorm(config.d_model)
        
        if config.use_ffn:
            self.ffn = self._build_ffn()
            self.ffn_layer_norm = nn.LayerNorm(config.d_model)
        
    def _build_ffn(self):
        return nn.Sequential(
            nn.Linear(self.config.d_model, self.config.ffn_dim),
            nn.ReLU(),
            nn.Linear(self.config.ffn_dim, self.config.d_model),
            nn.Dropout(self.config.dropout)
        )

    def forward(
        self, 
        x: Tensor, 
        context: Optional[Tensor] = None, 
        mask: Optional[Tensor] = None, 
        context_mask: Optional[Tensor] = None
    ) -> Tensor:
        residual = x
        x = self.layer_norm(x)
        attention_output, _ = self.attention(x, context, mask, context_mask)
        x = residual + attention_output

        if self.config.use_ffn:
            residual = x
            x = self.ffn_layer_norm(x)
            x = self.ffn(x)
            x = residual + x

        return x


if __name__ == '__main__':
    from boring_utils.utils import cprint
    from boring_transformer.core import TransformerLayersConfig, TransformerLayerWrapConfig
    from boring_nn.attention.core import AttentionConfig
    
    config = TransformerLayersConfig(
        dim=512,
        depth=6,
        causal=True,
        layer_config=TransformerLayerWrapConfig(
            attention=AttentionConfig(
                num_heads=8,
                d_model=512,
                dim_head=64,
                dropout=0.1
            ),
            ffn_dim=2048
        )
    )

