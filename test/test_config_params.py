from boring_llm.nn.ffn.core import FeedForwardConfig, ActivationType, ActivationConfig
from boring_llm.base.base_config import BaseConfig
from boring_llm.nn.attention.config import (
    AttentionConfig,
    CrossAttentionConfig,
    QKNormConfig,
)
from boring_llm.transformer.core import TransformerLayerWrapConfig, TransformerLayersConfig


# TODO: Update this after the implementation of the new configs

# ------------------------------
# Base Config
# ------------------------------
def test_base_config():
    config = BaseConfig()
    assert config.d_model == 512
    assert config.num_tokens == 20000
    assert config.dropout == 0.1


# ------------------------------
# FFN Config
# ------------------------------
def test_activation_config():
    config = ActivationConfig()
    assert config.type == ActivationType.GELU
    assert config.use_glu == False

def test_feed_forward_config():
    config = FeedForwardConfig()
    assert config.ffn_dim == 2048
    assert config.mult == 4
    assert config.dropout == 0.1
    assert isinstance(config.activation, ActivationConfig)

def test_feed_forward_config_activation_types():
    for activation_type in ActivationType:
        config = FeedForwardConfig(activation=ActivationConfig(type=activation_type))
        assert config.activation.type == activation_type


# ------------------------------
# Attention Config
# ------------------------------
def test_attention_config():
    config = AttentionConfig()
    assert config.dim_head == 64
    assert config.num_heads == 8
    assert config.causal == False
    assert config.bias == False
    # assert isinstance(config.attn_type_config, AttentionTypeConfig)
    # assert isinstance(config.qk_norm, QKNormConfig)

def test_attention_config_flash_attention():
    config = AttentionConfig(flash_attention=True)
    assert config.flash_attention == True


# ------------------------------
# TransformerLayerWrap Config
# ------------------------------
def test_transformer_layer_wrap_config():
    config = TransformerLayerWrapConfig()
    assert isinstance(config.attn_kwargs, AttentionConfig)
    assert isinstance(config.cross_attn_kwargs, CrossAttentionConfig)
    assert isinstance(config.ff_kwargs, FeedForwardConfig)


# ------------------------------
# TransformerLayers Config
# ------------------------------
def test_transformer_layers_config():
    config = TransformerLayersConfig()
    assert config.depth == 6
    assert config.causal == False
    assert config.cross_attend == False
    assert config.only_cross == False
    assert isinstance(config.layer_config, TransformerLayerWrapConfig)

def test_transformer_layers_config_sandwich():
    config = TransformerLayersConfig(sandwich_coef=2)
    assert config.sandwich_coef == 2


# import IPython; IPython.embed()
