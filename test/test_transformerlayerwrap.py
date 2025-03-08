import pytest
import torch
from boring_llm.transformer.core import BoringTransformerLayerWrap, TransformerLayerWrapConfig
from boring_llm.nn.attention.config import AttentionConfig
from boring_llm.nn.ffn.config import FeedForwardConfig

@pytest.fixture
def default_config():
    return TransformerLayerWrapConfig(
        attention=AttentionConfig(d_model=512, num_heads=8, dim_head=64),
        ff_kwargs=FeedForwardConfig(d_model=512, ffn_dim=2048),
        use_ffn=True
    )

def test_output_shape(default_config):
    layer = BoringTransformerLayerWrap(default_config)
    x = torch.randn(2, 10, 512)
    output = layer(x)
    assert output.shape == (2, 10, 512)

def test_without_ffn():
    config = TransformerLayerWrapConfig(
        attention=AttentionConfig(d_model=512, num_heads=8, dim_head=64),
        ff_kwargs=FeedForwardConfig(d_model=512, ffn_dim=2048),
        use_ffn=False
    )
    layer = BoringTransformerLayerWrap(config)
    x = torch.randn(2, 10, 512)
    output = layer(x)
    assert output.shape == (2, 10, 512)

def test_cross_attention(default_config):
    layer = BoringTransformerLayerWrap(default_config)
    x = torch.randn(2, 10, 512)
    context = torch.randn(2, 15, 512)
    output = layer(x, context=context)
    assert output.shape == (2, 10, 512)

def test_with_mask(default_config):
    layer = BoringTransformerLayerWrap(default_config)
    x = torch.randn(2, 10, 512)
    mask = torch.ones(2, 10).bool()
    mask[:, 5:] = False
    output = layer(x, mask=mask)
    assert output.shape == (2, 10, 512)
