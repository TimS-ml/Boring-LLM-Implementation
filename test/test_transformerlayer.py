import pytest
import torch
from boring_llm.transformer.transformer import BoringTransformerLayers, TransformerLayersConfig
from boring_llm.nn.attention.config import AttentionConfig
from boring_llm.nn.ffn.config import FeedForwardConfig

@pytest.fixture
def default_config():
    return TransformerLayersConfig(
        d_model=512,
        depth=6,
        causal=False,
        attn_kwargs=AttentionConfig(d_model=512, num_heads=8, dim_head=64),
        ff_kwargs=FeedForwardConfig(d_model=512, ffn_dim=2048)
    )

def test_output_shape(default_config):
    transformer = BoringTransformerLayers(default_config)
    x = torch.randn(2, 10, 512)
    output = transformer(x)
    assert output.shape == (2, 10, 512)

def test_causal_transformer():
    config = TransformerLayersConfig(
        d_model=512,
        depth=6,
        causal=True,
        attn_kwargs=AttentionConfig(d_model=512, num_heads=8, dim_head=64),
        ff_kwargs=FeedForwardConfig(d_model=512, ffn_dim=2048)
    )
    causal_transformer = BoringTransformerLayers(config)
    x = torch.randn(2, 10, 512)
    output = causal_transformer(x)
    assert output.shape == (2, 10, 512)

def test_cross_attention():
    config = TransformerLayersConfig(
        d_model=512,
        depth=6,
        causal=False,
        cross_attend=True,
        attn_kwargs=AttentionConfig(d_model=512, num_heads=8, dim_head=64),
        ff_kwargs=FeedForwardConfig(d_model=512, ffn_dim=2048)
    )
    cross_transformer = BoringTransformerLayers(config)
    x = torch.randn(2, 10, 512)
    context = torch.randn(2, 15, 512)
    output = cross_transformer(x, context=context)
    assert output.shape == (2, 10, 512)

def test_sandwich_coef():
    config = TransformerLayersConfig(
        d_model=512,
        depth=6,
        causal=False,
        sandwich_coef=2,
        attn_kwargs=AttentionConfig(d_model=512, num_heads=8, dim_head=64),
        ff_kwargs=FeedForwardConfig(d_model=512, ffn_dim=2048)
    )
    sandwich_transformer = BoringTransformerLayers(config)
    x = torch.randn(2, 10, 512)
    output = sandwich_transformer(x)
    assert output.shape == (2, 10, 512)

def test_custom_layers():
    config = TransformerLayersConfig(
        d_model=512,
        depth=6,
        causal=False,
        custom_layers=['a', 'f', 'c'],
        attn_kwargs=AttentionConfig(d_model=512, num_heads=8, dim_head=64),
        ff_kwargs=FeedForwardConfig(d_model=512, ffn_dim=2048)
    )
    custom_transformer = BoringTransformerLayers(config)
    x = torch.randn(2, 10, 512)
    output = custom_transformer(x)
    assert output.shape == (2, 10, 512)

def test_macaron():
    config = TransformerLayersConfig(
        d_model=512,
        depth=6,
        causal=False,
        macaron=True,
        attn_kwargs=AttentionConfig(d_model=512, num_heads=8, dim_head=64),
        ff_kwargs=FeedForwardConfig(d_model=512, ffn_dim=2048)
    )
    macaron_transformer = BoringTransformerLayers(config)
    x = torch.randn(2, 10, 512)
    output = macaron_transformer(x)
    assert output.shape == (2, 10, 512)
