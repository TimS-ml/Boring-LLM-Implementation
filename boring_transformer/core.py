from pydantic import BaseModel
from typing import Optional, List
import torch
import torch.nn as nn
from torch import Tensor

from boring_nn.attention.core import AttentionConfig
from boring_nn.attention.attn import BoringAttention


class TransformerLayerWrapConfig(BaseModel):
    attention: AttentionConfig
    d_model: int
    ffn_dim: int
    dropout: float
    use_ffn: bool = True


class TransformerLayersConfig(BaseModel):
    dim: int
    depth: int
    heads: int = 8
    causal: bool = False
    cross_attend: bool = False
    only_cross: bool = False
    use_scalenorm: bool = False
    use_rezero: bool = False
    rel_pos_bias: bool = False
    custom_layers: Optional[List[str]] = None
    sandwich_coef: Optional[float] = None
    macaron: bool = False
    layer_config: TransformerLayerWrapConfig


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
