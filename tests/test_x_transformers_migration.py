"""
Comprehensive tests for x-transformers migration

Tests all newly migrated modules:
- Normalization
- Activations
- Connections
- Attention
- Positional Encoding
- Memory
- Architecture Variants
"""

import torch
import torch.nn as nn
import pytest


def test_normalization():
    """Test all normalization variants"""
    print("\n=== Testing Normalization ===")
    from boring_llm.nn.norm import create_norm

    batch, seq, dim = 2, 10, 512
    x = torch.randn(batch, seq, dim)

    # Test RMSNorm
    norm1 = create_norm("rmsnorm", dim_model=dim, unit_offset=False)
    y1 = norm1(x)
    assert y1.shape == x.shape, f"RMSNorm shape mismatch: {y1.shape} vs {x.shape}"
    print("✓ RMSNorm")

    # Test LayerNorm
    norm2 = create_norm("layernorm", dim_model=dim, unit_offset=True)
    y2 = norm2(x)
    assert y2.shape == x.shape
    print("✓ LayerNorm")

    # Test ScaleNorm
    norm3 = create_norm("scalenorm", dim_model=dim)
    y3 = norm3(x)
    assert y3.shape == x.shape
    print("✓ ScaleNorm")

    # Test SimpleRMSNorm
    norm4 = create_norm("simple_rmsnorm", dim_model=dim)
    y4 = norm4(x)
    assert y4.shape == x.shape
    print("✓ SimpleRMSNorm")

    # Test MultiheadRMSNorm
    norm5 = create_norm("multihead_rmsnorm", dim_model=dim, num_heads=8)
    y5 = norm5(x)
    assert y5.shape == x.shape
    print("✓ MultiheadRMSNorm")

    # Test AdaptiveLayerNorm
    norm6 = create_norm("adaptive_layernorm", dim_model=dim, dim_condition=256)
    condition = torch.randn(batch, 256)
    y6 = norm6(x, condition=condition)
    assert y6.shape == x.shape
    print("✓ AdaptiveLayerNorm")

    # Test AdaptiveRMSNorm
    norm7 = create_norm("adaptive_rmsnorm", dim_model=dim, dim_condition=256)
    y7 = norm7(x, condition=condition)
    assert y7.shape == x.shape
    print("✓ AdaptiveRMSNorm")

    # Test DynamicTanh
    norm8 = create_norm("dynamic_tanh", dim_model=dim, init_alpha=1.0)
    y8 = norm8(x)
    assert y8.shape == x.shape
    print("✓ DynamicTanh")

    # Test DERF (new - from x-transformers Dec 2025)
    norm9 = create_norm("derf", dim_model=dim, init_alpha=0.5, init_bias=0.)
    y9 = norm9(x)
    assert y9.shape == x.shape
    # DERF uses erf which is bounded [-1, 1], so output should be bounded
    assert y9.abs().max() < 100, "DERF output should be reasonably bounded"
    print("✓ DERF (erf-based normalization)")

    # Test DERF with unit_offset
    norm10 = create_norm("derf", dim_model=dim, init_alpha=0.5, unit_offset=True)
    y10 = norm10(x)
    assert y10.shape == x.shape
    print("✓ DERF with unit_offset")

    print("All normalization tests passed! ✓")


def test_activations():
    """Test custom activations"""
    print("\n=== Testing Activations ===")
    from boring_llm.nn.activation.activation import ReluSquared, SoLU

    batch, seq, dim = 2, 10, 512
    x = torch.randn(batch, seq, dim)

    # Test ReluSquared
    relu_sq = ReluSquared()
    y1 = relu_sq(x)
    assert y1.shape == x.shape
    assert (y1 >= 0).all(), "ReluSquared should be non-negative"
    print("✓ ReluSquared")

    # Test SoLU
    solu = SoLU(dim=dim)
    y2 = solu(x)
    assert y2.shape == x.shape
    print("✓ SoLU")

    print("All activation tests passed! ✓")


def test_connections():
    """Test connection/wrapper modules"""
    print("\n=== Testing Connections ===")
    from boring_llm.nn.connections import create_connection

    batch, seq, dim = 2, 10, 512
    x = torch.randn(batch, seq, dim)
    residual = torch.randn(batch, seq, dim)
    condition = torch.randn(batch, 256)

    # Test Residual
    conn1 = create_connection("residual", dim_model=dim, scale_residual=True)
    y1 = conn1(x, residual=residual)
    assert y1.shape == x.shape
    print("✓ Residual")

    # Test GRU Gating
    conn2 = create_connection("gru_gating", dim_model=dim, scale_residual=False)
    y2 = conn2(x, residual=residual)
    assert y2.shape == x.shape
    print("✓ GRU Gating")

    # Test LayerScale
    conn3 = create_connection("layer_scale", dim_model=dim, init_value=1e-4)
    y3 = conn3(x)
    assert y3.shape == x.shape
    print("✓ LayerScale")

    # Test AdaptiveLayerScale
    conn4 = create_connection("adaptive_layer_scale", dim_model=dim, dim_condition=256)
    y4 = conn4(x, condition=condition)
    assert y4.shape == x.shape
    print("✓ AdaptiveLayerScale")

    # Test DynamicLIMe
    hiddens = [torch.randn(batch, seq, dim) for _ in range(6)]
    conn5 = create_connection("dynamic_lime", dim_model=dim, num_layers=6)
    y5 = conn5(x, hiddens=hiddens)
    assert y5.shape == x.shape
    print("✓ DynamicLIMe")

    # Test ShiftTokens
    conn6 = create_connection("shift_tokens", dim_model=dim, shifts=(0, 1, 2))
    y6 = conn6(x)
    assert y6.shape == x.shape
    print("✓ ShiftTokens")

    # Test HyperConnection with mHC (manifold constrained - from x-transformers Jan 2026)
    from boring_llm.nn.connections.registry import sinkhorn
    num_streams = 4
    # Test the sinkhorn function first
    test_matrix = torch.randn(batch, seq, num_streams, num_streams)
    doubly_stochastic = sinkhorn(test_matrix, iters=10)
    # Check rows sum to ~1
    row_sums = doubly_stochastic.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=0.1), \
        f"Sinkhorn rows should sum to 1: {row_sums.mean()}"
    # Check cols sum to ~1
    col_sums = doubly_stochastic.sum(dim=-2)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=0.1), \
        f"Sinkhorn cols should sum to 1: {col_sums.mean()}"
    print("✓ Sinkhorn-Knopp algorithm")

    # Test HyperConnection
    conn7 = create_connection("hyper_connection", dim_model=dim, layer_index=0,
                              num_residual_streams=num_streams, sinkhorn_iters=5)
    # HyperConnection expects residuals in shape (b*s, n, d)
    hc_residuals = torch.randn(batch * num_streams, seq, dim)
    branch_input, new_residuals, extra = conn7.prepare(hc_residuals)
    assert branch_input.shape == (batch, seq, dim), f"Branch input shape: {branch_input.shape}"
    assert new_residuals.shape == (batch, seq, num_streams, dim), \
        f"Residuals shape: {new_residuals.shape}"
    assert 'beta' in extra, "Should have beta in extra kwargs"
    print("✓ HyperConnection with mHC (manifold constraints)")

    # Test HyperConnection forward (calls apply internally)
    layer_output = torch.randn(batch, seq, dim)
    y7 = conn7(layer_output, residuals=new_residuals, beta=extra['beta'])
    assert y7.shape == (batch * num_streams, seq, dim), f"HyperConnection output shape: {y7.shape}"
    print("✓ HyperConnection forward")

    print("All connection tests passed! ✓")


def test_attention():
    """Test attention mechanisms"""
    print("\n=== Testing Attention ===")
    from boring_llm.nn.attention import create_attention

    batch, seq, dim = 2, 10, 512
    num_heads = 8
    x = torch.randn(batch, seq, dim)

    # Test Standard Attention
    attn1 = create_attention("standard", dim_model=dim, num_heads=num_heads, dim_head=64)
    y1 = attn1(x)
    assert y1.shape == x.shape
    print("✓ Standard Attention")

    # Test Multi-Query Attention (MQA)
    attn2 = create_attention("standard", dim_model=dim, num_heads=num_heads,
                            one_kv_head=True, dim_head=64)
    y2 = attn2(x)
    assert y2.shape == x.shape
    assert attn2.kv_heads == 1, f"MQA should have 1 KV head, got {attn2.kv_heads}"
    print("✓ Multi-Query Attention (MQA)")

    # Test Grouped-Query Attention (GQA)
    attn3 = create_attention("standard", dim_model=dim, num_heads=num_heads,
                            kv_heads=2, dim_head=64)
    y3 = attn3(x)
    assert y3.shape == x.shape
    assert attn3.kv_heads == 2, f"GQA should have 2 KV heads, got {attn3.kv_heads}"
    print("✓ Grouped-Query Attention (GQA)")

    # Test with QK Normalization
    attn4 = create_attention("standard", dim_model=dim, num_heads=num_heads,
                            dim_head=64, qk_norm=True, qk_norm_scale=10.0)
    y4 = attn4(x)
    assert y4.shape == x.shape
    print("✓ Attention with QK Norm")

    # Test Cosine Similarity Attention
    attn5 = create_attention("cosine_sim", dim_model=dim, num_heads=num_heads,
                            dim_head=64, temperature=1.0)
    y5 = attn5(x)
    assert y5.shape == x.shape
    print("✓ Cosine Similarity Attention")

    # Test Sparse TopK Attention
    attn6 = create_attention("sparse_topk", dim_model=dim, num_heads=num_heads,
                            dim_head=64, topk=8)
    y6 = attn6(x)
    assert y6.shape == x.shape
    print("✓ Sparse TopK Attention")

    # Test with Value Gating
    attn7 = create_attention("standard", dim_model=dim, num_heads=num_heads,
                            dim_head=64, gate_values=True)
    y7 = attn7(x)
    assert y7.shape == x.shape
    print("✓ Attention with Value Gating")

    # Test with Talking Heads
    attn8 = create_attention("standard", dim_model=dim, num_heads=num_heads,
                            dim_head=64, pre_talking_heads=True, post_talking_heads=True)
    y8 = attn8(x)
    assert y8.shape == x.shape
    print("✓ Attention with Talking Heads")

    print("All attention tests passed! ✓")


def test_positional_encoding():
    """Test positional encoding variants"""
    print("\n=== Testing Positional Encoding ===")
    from boring_llm.nn.pe import create_pe

    batch, seq, dim = 2, 10, 512
    num_heads = 8
    x = torch.randn(batch, seq, dim)

    # Test Relative Position Bias (use pe_strategy.apply to access strategy method)
    pe1 = create_pe("relative_position_bias", dim_model=dim, num_heads=num_heads,
                   causal=True, num_buckets=32)
    bias1 = pe1.pe_strategy.apply(torch.arange(seq), seq_len_q=seq, seq_len_k=seq)
    assert bias1.shape == (num_heads, seq, seq)
    print("✓ Relative Position Bias")

    # Test Dynamic Position Bias
    pe2 = create_pe("dynamic_position_bias", dim_model=dim, num_heads=num_heads,
                   depth=2, log_distance=False)
    bias2 = pe2.pe_strategy.apply(torch.arange(seq), seq_len_q=seq, seq_len_k=seq)
    assert bias2.shape == (num_heads, seq, seq)
    print("✓ Dynamic Position Bias")

    # Test CoPE (requires query and attn_logits)
    pe3 = create_pe("cope", dim_model=dim, num_heads=num_heads, max_pos=16)
    query = torch.randn(batch, num_heads, seq, dim // num_heads)
    attn_logits = torch.randn(batch, num_heads, seq, seq)
    bias3 = pe3.pe_strategy.apply(torch.arange(seq), query=query, attn_logits=attn_logits)
    assert bias3.shape == (batch, num_heads, seq, seq)
    print("✓ CoPE")

    # Test Data-Dependent ALiBi
    pe4 = create_pe("data_dependent_alibi", dim_model=dim, num_heads=num_heads,
                   causal=True)
    bias4 = pe4.pe_strategy.apply(torch.arange(seq), x=x)
    assert bias4.shape == (batch, num_heads, seq, seq)
    print("✓ Data-Dependent ALiBi")

    # Test PoPE (Polar Positional Encoding - new from x-transformers Dec 2025)
    from boring_llm.nn.pe.registry import apply_polar_pos_emb, PolarPositionalEncoding
    import math
    dim_head = 64
    # Create strategy directly to test PoPE
    pe5_strategy = PolarPositionalEncoding(dim_model=dim_head, num_heads=num_heads, bias_uniform_init=False)
    pos = torch.arange(seq).float().unsqueeze(0)  # [1, seq]
    freqs, bias = pe5_strategy.apply(pos)
    assert freqs.shape == (1, seq, dim_head), f"PoPE freqs shape: {freqs.shape}"
    assert bias.shape == (num_heads, 1, dim_head), f"PoPE bias shape: {bias.shape}"
    print("✓ PoPE (Polar Positional Encoding)")

    # Test PoPE apply_to_qk
    q = torch.randn(batch, num_heads, seq, dim_head)
    k = torch.randn(batch, num_heads, seq, dim_head)
    # Note: apply_polar_pos_emb doubles the dimension
    q_polar = apply_polar_pos_emb(q, freqs)
    k_polar = apply_polar_pos_emb(k, freqs + bias)
    assert q_polar.shape[-1] == dim_head * 2, f"PoPE doubles dim: {q_polar.shape}"
    print("✓ PoPE apply_to_qk")

    # Test PoPE with bias_uniform_init
    pe6_strategy = PolarPositionalEncoding(dim_model=dim_head, num_heads=num_heads, bias_uniform_init=True)
    freqs6, bias6 = pe6_strategy.apply(pos)
    # Check bias is initialized in [-2π, 0]
    assert (bias6 >= -2 * math.pi).all() and (bias6 <= 0).all(), "Bias should be in [-2π, 0]"
    print("✓ PoPE with uniform bias init")

    print("All positional encoding tests passed! ✓")


def test_memory():
    """Test memory mechanisms"""
    print("\n=== Testing Memory ===")
    from boring_llm.nn.memory import MemoryTokens, PersistentMemoryKV

    batch, seq, dim = 2, 100, 512
    num_heads = 8
    dim_head = 64

    # Test Memory Tokens
    mem_tokens = MemoryTokens(dim=dim, num_memory_tokens=20)
    x = torch.randn(batch, seq, dim)
    x_with_mem = mem_tokens(x)
    assert x_with_mem.shape == (batch, seq + 20, dim)
    x_removed = mem_tokens.remove_memory(x_with_mem)
    assert x_removed.shape == x.shape
    print("✓ Memory Tokens")

    # Test Persistent Memory KV
    mem_kv = PersistentMemoryKV(dim_head=dim_head, num_heads=num_heads, num_mem_kv=16)
    k = torch.randn(batch, num_heads, seq, dim_head)
    v = torch.randn(batch, num_heads, seq, dim_head)
    k_mem, v_mem = mem_kv(k, v)
    assert k_mem.shape == (batch, num_heads, seq + 16, dim_head)
    assert v_mem.shape == (batch, num_heads, seq + 16, dim_head)
    print("✓ Persistent Memory KV")

    print("All memory tests passed! ✓")


def test_architecture_variants():
    """Test architecture variants"""
    print("\n=== Testing Architecture Variants ===")
    from boring_llm.nn.arch_variants import (
        SandwichNorm, ResiDual, Normformer, MacaronNet, ResidualAttention
    )

    batch, seq, dim = 2, 10, 512
    x = torch.randn(batch, seq, dim)

    # Dummy modules
    dummy_attn = nn.Identity()
    dummy_ff = nn.Linear(dim, dim)

    # Test Sandwich Norm
    sandwich = SandwichNorm(dim, dummy_attn)
    y1 = sandwich(x)
    assert y1.shape == x.shape
    print("✓ Sandwich Norm")

    # Test ResiDual
    residual = ResiDual(dim, dummy_attn, dummy_ff, learnable_alphas=True)
    y2 = residual(x)
    assert y2.shape == x.shape
    print("✓ ResiDual")

    # Test Normformer
    normformer = Normformer(dim, dummy_attn)
    y3 = normformer(x)
    assert y3.shape == x.shape
    print("✓ Normformer")

    # Test Macaron
    macaron = MacaronNet(dim, dummy_attn, dummy_ff)
    y4 = macaron(x)
    assert y4.shape == x.shape
    print("✓ Macaron")

    # Test Residual Attention
    res_attn = ResidualAttention(dummy_attn, dim)
    y5 = res_attn(x)
    assert y5.shape == x.shape
    print("✓ Residual Attention")

    print("All architecture variant tests passed! ✓")


def test_integration():
    """Test integration of multiple components"""
    print("\n=== Testing Integration ===")
    from boring_llm.nn.attention import create_attention
    from boring_llm.nn.norm import create_norm
    from boring_llm.nn.memory import MemoryTokens

    batch, seq, dim = 2, 50, 512
    num_heads = 8

    # Build a simple transformer block with new components
    x = torch.randn(batch, seq, dim)

    # Add memory tokens
    memory = MemoryTokens(dim=dim, num_memory_tokens=10)
    x_with_mem = memory(x)
    assert x_with_mem.shape == (batch, seq + 10, dim)

    # Apply normalization
    norm = create_norm("rmsnorm", dim_model=dim)
    x_normed = norm(x_with_mem)
    assert x_normed.shape == x_with_mem.shape

    # Apply MQA attention
    attn = create_attention("standard", dim_model=dim, num_heads=num_heads,
                           one_kv_head=True, dim_head=64, qk_norm=True)
    x_attn = attn(x_normed)
    assert x_attn.shape == x_with_mem.shape

    # Remove memory tokens
    x_final = memory.remove_memory(x_attn)
    assert x_final.shape == x.shape

    print("✓ Integration test passed")
    print("All integration tests passed! ✓")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("Running comprehensive x-transformers migration tests")
    print("="*60)

    try:
        test_normalization()
        test_activations()
        test_connections()
        test_attention()
        test_positional_encoding()
        test_memory()
        test_architecture_variants()
        test_integration()

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
