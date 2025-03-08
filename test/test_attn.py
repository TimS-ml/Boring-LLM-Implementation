import pytest
import torch
from boring_llm.nn.attention.main import BoringAttention
from boring_llm.nn.attention.config import (
    AttentionConfig, AttentionType, AttentionTypeConfig
)


# ------------------------------
# Base Config
# ------------------------------
@pytest.fixture
def default_config():
    return AttentionConfig(
        d_model=512,
        num_heads=8,
        dim_head=64,
        dropout=0.1,
        causal=False
    )


# ------------------------------
# Shape Tests
# ------------------------------
def test_output_shape(default_config):
    attention = BoringAttention(default_config)
    x = torch.randn(2, 10, 512)
    output, _ = attention(x)
    assert output.shape == (2, 10, 512)

def test_causal_attention():
    config = AttentionConfig(
        d_model=512,
        num_heads=8,
        dim_head=64,
        dropout=0.1,
        causal=True
    )
    causal_attention = BoringAttention(config)
    x = torch.randn(2, 10, 512)
    _, attn = causal_attention(x)
    assert torch.allclose(attn[:,:,-1,:-1], torch.zeros_like(attn[:,:,-1,:-1]))


# ------------------------------
# Shape Tests 2
# ------------------------------
def test_attention_mask(default_config):
    attention = BoringAttention(default_config)
    x = torch.randn(2, 10, 512)
    mask = torch.ones(2, 10).bool()
    mask[:, 5:] = False
    output, attn = attention(x, mask=mask)
    assert torch.allclose(attn[:,:,5:], torch.zeros_like(attn[:,:,5:]))

def test_cross_attention(default_config):
    attention = BoringAttention(default_config)
    x = torch.randn(2, 10, 512)
    context = torch.randn(2, 15, 512)
    output, _ = attention(x, context=context)
    assert output.shape == (2, 10, 512)


# ------------------------------
# Feature Tests
# ------------------------------
def test_num_mem_kv():
    config = AttentionConfig(
        d_model=512,
        num_heads=8,
        dim_head=64,
        dropout=0.1,
        num_mem_kv=4
    )
    attention = BoringAttention(config)
    x = torch.randn(2, 10, 512)
    output, _ = attention(x)
    assert output.shape == (2, 10, 512)

def test_talking_heads():
    config = AttentionConfig(
        d_model=512,
        num_heads=8,
        dim_head=64,
        dropout=0.1,
        talking_heads=True
    )
    attention = BoringAttention(config)
    x = torch.randn(2, 10, 512)
    output, _ = attention(x)
    assert output.shape == (2, 10, 512)

def test_different_attention_types():
    for attention_type in AttentionType:
        config = AttentionConfig(
            d_model=512,
            num_heads=8,
            dim_head=64,
            dropout=0.1,
            attn_type_config=AttentionTypeConfig(type=attention_type)
        )
        attention = BoringAttention(config)
        x = torch.randn(2, 10, 512)
        output, _ = attention(x)
        assert output.shape == (2, 10, 512)


# import IPython; IPython.embed()
