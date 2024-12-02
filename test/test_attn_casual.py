import pytest
import torch
from boring_utils.utils import cprint
from boring_nn.attention.casual import (
    SimpleScaledDotProductAttention,
    ScaledDotProductAttention,
    MultiHeadAttention
)

@pytest.fixture
def test_params():
    B, T, C = 4, 8, 36  # batch size, time steps (seq length), channels
    T_ENC, T_DEC = 10, T  # encoder/decoder sequence lengths 
    HEAD_SIZE = 6  # number of attention heads
    return B, T, C, T_ENC, T_DEC, HEAD_SIZE

@pytest.fixture
def sample_inputs(test_params):
    B, T, C, _, _, _ = test_params
    x = torch.rand(B, T, C)
    return x

@pytest.fixture
def casual_mask(test_params):
    _, T, _, _, _, _ = test_params
    tril = torch.tril(torch.ones(T, T))
    mask = tril.float().masked_fill(tril == 0, float('-inf'))
    return mask

class TestSimpleAttention:
    def test_same_length_inputs(self, test_params, sample_inputs, casual_mask):
        B, T, C = test_params[:3]
        q = k = v = sample_inputs
        
        # 测试带mask的attention
        output, weights = SimpleScaledDotProductAttention(q, k, v, attn_mask=casual_mask)
        assert output.shape == (B, T, C)
        assert weights.shape == (B, T, T)
        
        # 测试is_causal=True
        output, weights = SimpleScaledDotProductAttention(q, k, v, is_causal=True)
        assert output.shape == (B, T, C)
        assert weights.shape == (B, T, T)

    def test_different_length_inputs(self, test_params):
        B, _, C, T_ENC, T_DEC, _ = test_params
        
        q = torch.randn(B, T_DEC, C)
        k = v = torch.randn(B, T_ENC, C)
        
        tril = torch.tril(torch.ones(T_DEC, T_ENC))
        mask = tril.float().masked_fill(tril == 0, float('-inf'))
        
        output, weights = SimpleScaledDotProductAttention(q, k, v, attn_mask=mask)
        assert output.shape == (B, T_DEC, C)
        assert weights.shape == (B, T_DEC, T_ENC)

class TestScaledDotProductAttention:
    def test_module_same_length(self, test_params, sample_inputs, casual_mask):
        B, T, C = test_params[:3]
        q = k = v = sample_inputs
        
        attention = ScaledDotProductAttention()
        output, weights = attention(q, k, v, casual_mask)
        
        assert output.shape == (B, T, C)
        assert weights.shape == (B, T, T)

    def test_module_with_dropout(self, test_params, sample_inputs):
        B, T, C = test_params[:3]
        q = k = v = sample_inputs
        
        attention = ScaledDotProductAttention(dropout=0.1)
        output, weights = attention(q, k, v)
        
        assert output.shape == (B, T, C)
        assert weights.shape == (B, T, T)

class TestMultiHeadAttention:
    def test_same_length_inputs(self, test_params, sample_inputs):
        B, T, C, _, _, HEAD_SIZE = test_params
        q = k = v = sample_inputs
        
        mha = MultiHeadAttention(d_model=C, num_heads=HEAD_SIZE)
        output, weights = mha(q, k, v)
        
        assert output.shape == (B, T, C)
        assert weights.shape == (B, HEAD_SIZE, T, T)

    def test_different_length_inputs(self, test_params):
        B, _, C, T_ENC, T_DEC, HEAD_SIZE = test_params
        
        q = torch.randn(B, T_DEC, C)
        k = v = torch.randn(B, T_ENC, C)
        
        # 测试2D mask
        tril = torch.tril(torch.ones(T_DEC, T_ENC))
        mask_2d = tril.float().masked_fill(tril == 0, float('-inf'))
        
        mha = MultiHeadAttention(d_model=C, num_heads=HEAD_SIZE)
        output, weights = mha(q, k, v, attn_mask=mask_2d)
        
        assert output.shape == (B, T_DEC, C)
        assert weights.shape == (B, HEAD_SIZE, T_DEC, T_ENC)
        
        # 测试3D mask
        mask_3d = mask_2d.expand(B, T_DEC, T_ENC)
        output, weights = mha(q, k, v, attn_mask=mask_3d)
        
        assert output.shape == (B, T_DEC, C)
        assert weights.shape == (B, HEAD_SIZE, T_DEC, T_ENC)
