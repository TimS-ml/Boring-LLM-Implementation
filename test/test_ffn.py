import pytest
import torch
from boring_nn.ffn.core import (
    FeedForwardConfig, ActivationType, ActivationConfig,
)
from boring_nn.ffn.main import BoringFeedForward

# ------------------------------
# Base Config
# ------------------------------
@pytest.fixture
def default_config():
    return FeedForwardConfig(
        d_model=512,
        ffn_dim=2048,
        dropout=0.1,
        activation=ActivationConfig(type=ActivationType.GELU)
    )

@pytest.fixture
def glu_config():
    return FeedForwardConfig(
        d_model=512,
        ffn_dim=2048,
        dropout=0.1,
        activation=ActivationConfig(use_glu=True)
        # activation=ActivationConfig(type=ActivationType.GLU)
    )


# ------------------------------
# Shape Tests 
# ------------------------------
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


# ------------------------------
# Config Tests 
# ------------------------------
def test_different_activation_types():
    for activation_type in ActivationType:
        if activation_type != ActivationType.GLU:
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

    activation_type = config.activation.type
    if activation_type != ActivationType.GLU:
        assert ffn.net[0][0].bias is None
    assert ffn.net[-1].bias is None


# ------------------------------
# Config Tests (GLU)
# ------------------------------
def test_glu_activation(glu_config):
    ffn = BoringFeedForward(glu_config)
    assert isinstance(ffn.net[0], ffn.glu.__class__)

def test_glu_activation_types(glu_config):
    ffn = BoringFeedForward(glu_config)
    x = torch.randn(2, 10, 512)
    output = ffn(x)
    assert output.shape == (2, 10, 512)

def test_glu_no_bias():
    config = FeedForwardConfig(
        d_model=512,
        ffn_dim=2048,
        dropout=0.1,
        activation=ActivationConfig(use_glu=True),
        # activation=ActivationConfig(type=ActivationType.GLU),
        no_bias=True
    )
    ffn = BoringFeedForward(config)

    activation_type = config.activation.type
    if activation_type != ActivationType.GLU:
        assert ffn.net[0][0].bias is None
    assert ffn.net[-1].bias is None


# import IPython; IPython.embed()
