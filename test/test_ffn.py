import pytest
import torch
from boring_nn.ffn.core import BoringFeedForward, FeedForwardConfig, ActivationType, ActivationConfig

@pytest.fixture
def default_config():
    return FeedForwardConfig(
        d_model=512,
        ffn_dim=2048,
        dropout=0.1,
        activation=ActivationConfig(type=ActivationType.GELU)
    )

def test_output_shape(default_config):
    ffn = BoringFeedForward(default_config)
    x = torch.randn(2, 10, 512)
    output = ffn(x)
    assert output.shape == (2, 10, 512)

def test_zero_init_output():
    config = FeedForwardConfig(
        d_model=512,
        ffn_dim=2048,
        dropout=0.1,
        activation=ActivationConfig(type=ActivationType.GELU),
        zero_init_output=True
    )
    ffn = BoringFeedForward(config)
    assert torch.allclose(ffn.net[-1].weight, torch.zeros_like(ffn.net[-1].weight))

def test_glu_activation():
    config = FeedForwardConfig(
        d_model=512,
        ffn_dim=2048,
        dropout=0.1,
        activation=ActivationConfig(type=ActivationType.GELU, use_glu=True)
    )
    ffn = BoringFeedForward(config)
    assert isinstance(ffn.net[0], ffn.glu.__class__)

def test_different_activation_types():
    for activation_type in ActivationType:
        config = FeedForwardConfig(
            d_model=512,
            ffn_dim=2048,
            dropout=0.1,
            activation=ActivationConfig(type=activation_type)
        )
        ffn = BoringFeedForward(config)
        x = torch.randn(2, 10, 512)
        output = ffn(x)
        assert output.shape == (2, 10, 512)

def test_no_bias():
    config = FeedForwardConfig(
        d_model=512,
        ffn_dim=2048,
        dropout=0.1,
        no_bias=True
    )
    ffn = BoringFeedForward(config)
    assert ffn.net[0].bias is None
    assert ffn.net[-1].bias is None
