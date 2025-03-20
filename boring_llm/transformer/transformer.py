from pydantic import BaseModel
from typing import Optional, List
import torch
import torch.nn as nn
from torch import Tensor

from transformer.core import TransformerLayersConfig, BoringTransformerLayerWrap

from transformer.core import TransformerLayersConfig, TransformerLayerWrapConfig
from boring_llm.nn.attention.core import AttentionConfig
from boring_llm.nn.ffn.core import FeedForwardConfig
from boring_llm.nn.ffn.main import BoringFeedForward
from boring_llm.nn.attention.main import BoringMultiHeadAttention


class BoringTransformerLayers(nn.Module):
    """
    BoringTransformerLayers is a stack of transformer layers.
    with layer specific modifications like sandwich, macaron, cross-attention, etc.
    """
    def __init__(self, config: TransformerLayersConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([])
        self.layer_types = self._determine_layer_types()
        
        # Basically it pass different set of params (init) 
        # and mod the forward as well
        for layer_type in self.layer_types:

            # NOTE: Normal Attention, causal, attn_kwargs
            if layer_type == 'a':
                layer = BoringTransformerLayerWrap(TransformerLayerWrapConfig(
                    attention=config.attn_kwargs,
                    ff_kwargs=config.ff_kwargs,
                    use_ffn=True
                ))

            # NOTE: CrossAttention, attn_kwargs + cross_attn_kwargs
            elif layer_type == 'c':
                layer = BoringTransformerLayerWrap(TransformerLayerWrapConfig(
                    attention={**config.attn_kwargs.dict(), **config.cross_attn_kwargs.dict()},
                    ff_kwargs=config.ff_kwargs,
                    use_ffn=True
                ))

            # NOTE: FeedForward only + ff_kwargs
            elif layer_type == 'f':
                layer = BoringTransformerLayerWrap(TransformerLayerWrapConfig(
                    attention=AttentionConfig(d_model=config.attn_kwargs.d_model),
                    ff_kwargs=config.ff_kwargs,
                    use_ffn=True
                ))
                if self.config.macaron:
                    layer = Scale(0.5, layer)
            else:
                raise ValueError(f"Invalid layer type: {layer_type}")
            
            self.layers.append(layer)

    def _determine_layer_types(self):
        if self.config.custom_layers:
            return self.config.custom_layers
        
        if self.config.cross_attend and not self.config.only_cross:
            default_block = ('a', 'c', 'f')
        elif self.config.cross_attend and self.config.only_cross:
            default_block = ('c', 'f')
        else:
            default_block = ('a', 'f')

        if self.config.macaron:
            default_block = ('f',) + default_block

        if self.config.sandwich_coef:
            assert 0 < self.config.sandwich_coef <= self.config.depth, 'sandwich coefficient should be between 0 and depth'
            layer_types = ('a',) * self.config.sandwich_coef + default_block * (self.config.depth - self.config.sandwich_coef) + ('f',) * self.config.sandwich_coef
        else:
            layer_types = default_block * self.config.depth

        return layer_types

    def forward(
        self, 
        x: Tensor, 
        context: Optional[Tensor] = None, 
        mask: Optional[Tensor] = None, 
        context_mask: Optional[Tensor] = None
    ) -> Tensor:
        for layer_type, layer in zip(self.layer_types, self.layers):
            if layer_type == 'a':
                x = layer(x, mask=mask)
            elif layer_type == 'c':
                x = layer(x, context=context, mask=mask, context_mask=context_mask)
            elif layer_type == 'f':
                x = layer(x)
        return x

class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.scale = scale
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale


class BoringEncoder(BoringTransformerLayers):
    def __init__(self, config: TransformerLayersConfig):
        assert not config.causal, "Encoder should not be causal"
        super().__init__(config)


class BoringDecoder(BoringTransformerLayers):
    def __init__(self, config: TransformerLayersConfig):
        assert config.causal, "Decoder should be causal"
        super().__init__(config)


class BoringCrossAttender(BoringTransformerLayers):
    def __init__(self, config: TransformerLayersConfig):
        assert config.cross_attend and config.only_cross, "CrossAttender should only have cross attention"
        super().__init__(config)


class BoringTransformerLayersFactory:
    @staticmethod
    def create(config: TransformerLayersConfig, layer_type: str) -> BoringTransformerLayers:
        if layer_type == 'encoder':
            return BoringEncoder(config)
        elif layer_type == 'decoder':
            return BoringDecoder(config)
        elif layer_type == 'cross_attender':
            return BoringCrossAttender(config)
        else:
            raise ValueError(f"Invalid layer type: {layer_type}")

