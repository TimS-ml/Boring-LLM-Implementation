import pytest
from pydantic import ValidationError
from boring_nn.ffn.core import FeedForwardConfig, ActivationType, ActivationConfig
from boring_nn.attention.core import AttentionConfig, AttentionType, AttentionTypeConfig, QKNormConfig
from boring_transformer.core import TransformerLayerWrapConfig, TransformerLayersConfig
from boring_llm_base.base_config import BaseConfig

def test_base_config():
    config = BaseConfig()
    assert config.d_model == 512
    assert config.num_tokens == 20000
    assert config.dropout == 0.1

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

def test_qk_norm_config():
    config = QKNormConfig()
    assert config.enabled == False
    assert config.groups == 1
    assert config.scale == 10.0

def test_attention_type_config():
    config = AttentionTypeConfig()
    assert config.type == AttentionType.SOFTMAX
    assert config.sparse_topk == 10

def test_attention_config():
    config = AttentionConfig()
    assert config.dim_head == 64
    assert config.num_heads == 8
    assert config.causal == False
    assert config.bias == False
    assert isinstance(config.attn_type_config, AttentionTypeConfig)
    assert isinstance(config.qk_norm, QKNormConfig)

def test_transformer_layer_wrap_config():
    config = TransformerLayerWrapConfig()
    assert isinstance(config.attention, AttentionConfig)
    assert isinstance(config.ff_kwargs, FeedForwardConfig)
    assert config.use_ffn == True

def test_transformer_layers_config():
    config = TransformerLayersConfig()
    assert config.depth == 6
    assert config.causal == False
    assert config.cross_attend == False
    assert config.only_cross == False
    assert isinstance(config.layer_config, TransformerLayerWrapConfig)

def test_config_validation():
    with pytest.raises(ValidationError):
        FeedForwardConfig(ffn_dim=-1)
    
    with pytest.raises(ValidationError):
        AttentionConfig(num_heads=-1)
    
    with pytest.raises(ValidationError):
        TransformerLayersConfig(depth=-1)

def test_config_inheritance():
    class ChildConfig(BaseConfig):
        child_param: int = 10

    config = ChildConfig()
    assert config.d_model == 512  # Inherited from BaseConfig
    assert config.child_param == 10  # New parameter

def test_config_override():
    config = BaseConfig(d_model=1024, dropout=0.2)
    assert config.d_model == 1024
    assert config.dropout == 0.2

def test_attention_config_flash_attention():
    config = AttentionConfig(flash_attention=True)
    assert config.flash_attention == True

def test_transformer_layers_config_sandwich():
    config = TransformerLayersConfig(sandwich_coef=2)
    assert config.sandwich_coef == 2

def test_feed_forward_config_activation_types():
    for activation_type in ActivationType:
        config = FeedForwardConfig(activation=ActivationConfig(type=activation_type))
        assert config.activation.type == activation_type

def test_attention_config_attention_types():
    for attention_type in AttentionType:
        config = AttentionConfig(attn_type_config=AttentionTypeConfig(type=attention_type))
        assert config.attn_type_config.type == attention_type
