from pydantic import BaseModel, Field
from typing import Optional, List
import torch
import torch.nn as nn
from torch import Tensor

from boring_llm_base.base_config import BaseConfig
from boring_nn.attention.core import AttentionConfig, CrossAttentionConfig
from boring_nn.ffn.core import FeedForwardConfig
from boring_nn.attention.main import BoringAttention


class TransformerLayerWrapConfig(BaseConfig):
    attn_kwargs: AttentionConfig = Field(default_factory=AttentionConfig)
    cross_attn_kwargs: CrossAttentionConfig = Field(default_factory=CrossAttentionConfig)
    ff_kwargs: FeedForwardConfig = Field(default_factory=FeedForwardConfig)


# TODO: double check the causal param
class TransformerLayersConfig(BaseConfig):
    # basic
    depth: Optional[int]               = Field(6,     description="Number of transformer layers")
    causal: Optional[bool]             = Field(False, description="Whether the model is causal")
    cross_attend: Optional[bool]       = Field(False, description="Whether to use cross-attention")
    only_cross: Optional[bool]         = Field(False, description="Whether to use only cross-attention")

    # advanced
    use_scalenorm: Optional[bool]      = Field(False, description="Whether to use ScaleNorm instead of LayerNorm")
    use_rezero: Optional[bool]         = Field(False, description="Whether to use ReZero initialization")
    rel_pos_bias: Optional[bool]       = Field(False, description="Whether to use relative positional bias")
    custom_layers: Optional[List[str]] = Field(None,  description="Custom layer configuration")
    sandwich_coef: Optional[float]     = Field(None,  description="Sandwich coefficient for layer configuration")
    macaron: Optional[bool]            = Field(False, description="Whether to use Macaron architecture")
    
    # layer config
    layer_config: TransformerLayerWrapConfig = Field(default_factory=TransformerLayerWrapConfig)


    def __init__(self, **data):
        super().__init__(**data)
        if self.only_cross:
            assert self.cross_attend, "only_cross requires cross_attend to be True"


class BoringTransformerLayerWrap(nn.Module):
    """
    BoringTransformerLayerWrap is a wrapper around a transformer layer.
    """
    def __init__(self, config: TransformerLayerWrapConfig):
        super().__init__()
        self.config = config
        self.attention = BoringAttention(config.attention)
        self.layer_norm = nn.LayerNorm(config.attention.d_model)
        
        if config.use_ffn:
            self.ffn = BoringFeedForward(config.ff_kwargs)
            self.ffn_layer_norm = nn.LayerNorm(config.attention.d_model)
        
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
    
    config = TransformerLayersConfig(
        d_model=512,
        depth=6,
        causal=True,
        layer_config=TransformerLayerWrapConfig(
            attention=AttentionConfig(
                num_heads=8,
                dim_head=64,
                dropout=0.1
            ),
            ffn_dim=2048
        )
    )

