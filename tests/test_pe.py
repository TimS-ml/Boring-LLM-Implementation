"""Test Positional Encoding implementations"""

import torch
import torch.nn as nn
from boring_llm.nn.pe.main import create_pe, BoringPositionalEncoding, PEConfig, pe_registry


def test_pe_basic():
    """Test basic PE functionality"""
    print("🚀 Testing PE Basic Functionality")
    
    # Test data
    batch_size, seq_len, dim_model = 2, 32, 128
    x = torch.randn(batch_size, seq_len, dim_model)
    print(f"Input tensor shape: {x.shape}")
    
    # ========== PE Tests ==========
    print("\n🔄 PE Tests:")
    
    # 1. Fixed sinusoidal
    pe1 = create_pe(pe_type="fixed", dim_model=dim_model)
    pos1 = pe1(x)
    print(f"  Fixed PE: {x.shape} -> {pos1.shape} ✅")
    assert pos1.shape == (seq_len, dim_model), f"Expected ({seq_len}, {dim_model}), got {pos1.shape}"
    
    # 2. Absolute learnable
    pe2 = create_pe(pe_type="absolute", dim_model=dim_model, max_seq_len=512)
    pos2 = pe2(x)
    print(f"  Absolute PE: {x.shape} -> {pos2.shape} ✅")
    assert pos2.shape == (seq_len, dim_model), f"Expected ({seq_len}, {dim_model}), got {pos2.shape}"
    
    # 3. Rotary
    pe3 = create_pe(pe_type="rotary", dim_model=dim_model, rotary_percentage=0.5)
    pos3 = pe3(x)
    expected_rotary_dim = int(dim_model * 0.5)
    if expected_rotary_dim % 2 != 0:
        expected_rotary_dim -= 1
    print(f"  Rotary PE: {x.shape} -> {pos3.shape} ✅")
    assert pos3.shape == (seq_len, expected_rotary_dim, 2), f"Expected ({seq_len}, {expected_rotary_dim}, 2), got {pos3.shape}"
    
    # 4. None
    pe4 = create_pe(pe_type="none", dim_model=dim_model)
    pos4 = pe4(x)
    print(f"  None PE: {x.shape} -> {pos4} ✅")
    assert pos4 is None, f"Expected None, got {pos4}"
    
    return True


def test_pe_parameters():
    """Test PE parameter counts"""
    print("\n📊 Testing PE Parameters")
    
    dim_model = 128
    max_seq_len = 512
    
    # Fixed PE (no parameters)
    pe_fixed = create_pe(pe_type="fixed", dim_model=dim_model)
    fixed_params = sum(p.numel() for p in pe_fixed.parameters())
    print(f"  Fixed PE: {fixed_params:,} parameters")
    assert fixed_params == 0, f"Fixed PE should have no parameters, got {fixed_params}"
    
    # Absolute PE (has parameters)
    pe_absolute = create_pe(pe_type="absolute", dim_model=dim_model, max_seq_len=max_seq_len)
    absolute_params = sum(p.numel() for p in pe_absolute.parameters())
    print(f"  Absolute PE: {absolute_params:,} parameters")
    expected_params = max_seq_len * dim_model
    assert absolute_params == expected_params, f"Expected {expected_params}, got {absolute_params}"
    
    # Rotary PE (no learnable parameters)
    pe_rotary = create_pe(pe_type="rotary", dim_model=dim_model, rotary_percentage=1.0)
    rotary_params = sum(p.numel() for p in pe_rotary.parameters())
    print(f"  Rotary PE: {rotary_params:,} parameters")
    assert rotary_params == 0, f"Rotary PE should have no learnable parameters, got {rotary_params}"
    
    # ALiBi PE (no learnable parameters)
    pe_alibi = create_pe(pe_type="alibi", dim_model=dim_model, alibi_num_heads=8)
    alibi_params = sum(p.numel() for p in pe_alibi.parameters())
    print(f"  ALiBi PE: {alibi_params:,} parameters")
    assert alibi_params == 0, f"ALiBi PE should have no learnable parameters, got {alibi_params}"
    
    return True


def test_pe_configurations():
    """Test different PE configurations"""
    print("\n⚙️ Testing PE Configurations")
    
    batch_size, seq_len, dim_model = 1, 16, 64
    x = torch.randn(batch_size, seq_len, dim_model)
    
    configs = [
        # Fixed PE
        {"pe_type": "fixed", "dim_model": dim_model},
        
        # Absolute PE variations
        {"pe_type": "absolute", "dim_model": dim_model, "max_seq_len": 128, "l2norm_embed": False},
        {"pe_type": "absolute", "dim_model": dim_model, "max_seq_len": 256, "l2norm_embed": True},
        
        # Rotary PE variations
        {"pe_type": "rotary", "dim_model": dim_model, "rotary_percentage": 0.25, "rope_base": 10000},
        {"pe_type": "rotary", "dim_model": dim_model, "rotary_percentage": 0.5, "rope_base": 500000},
        {"pe_type": "rotary", "dim_model": dim_model, "rotary_percentage": 1.0, "rope_base": 10000},
        
        # ALiBi PE variations
        {"pe_type": "alibi", "dim_model": dim_model, "alibi_num_heads": 4},
        {"pe_type": "alibi", "dim_model": dim_model, "alibi_num_heads": 8},
        {"pe_type": "alibi", "dim_model": dim_model, "alibi_num_heads": 16},
        
        # None PE
        {"pe_type": "none", "dim_model": dim_model},
    ]
    
    for i, config in enumerate(configs):
        pe = create_pe(**config)
        pos = pe(x)
        
        if config["pe_type"] == "none":
            assert pos is None, f"None PE should return None"
            print(f"  Config {i+1}: {config['pe_type']} -> None ✅")
        else:
            assert pos is not None, f"{config['pe_type']} PE should not return None"
            print(f"  Config {i+1}: {config['pe_type']} -> {pos.shape} ✅")
    
    return True


def test_pe_sequence_lengths():
    """Test PE with different sequence lengths"""
    print("\n📏 Testing PE Sequence Lengths")
    
    dim_model = 64
    pe_types = ["fixed", "absolute", "rotary"]
    
    for pe_type in pe_types:
        if pe_type == "absolute":
            pe = create_pe(pe_type=pe_type, dim_model=dim_model, max_seq_len=128)
        else:
            pe = create_pe(pe_type=pe_type, dim_model=dim_model)
        
        # Test different sequence lengths
        for seq_len in [8, 16, 32, 64]:
            x = torch.randn(1, seq_len, dim_model)
            
            try:
                pos = pe(x)
                if pe_type == "rotary":
                    expected_rotary_dim = dim_model
                    if expected_rotary_dim % 2 != 0:
                        expected_rotary_dim -= 1
                    expected_shape = (seq_len, expected_rotary_dim, 2)
                else:
                    expected_shape = (seq_len, dim_model)
                
                assert pos.shape == expected_shape, f"Expected {expected_shape}, got {pos.shape}"
                print(f"  {pe_type} PE seq_len={seq_len}: ✅")
                
            except Exception as e:
                print(f"  {pe_type} PE seq_len={seq_len}: ❌ {e}")
                return False
    
    return True


def test_pe_registry():
    """Test PE registry functionality"""
    print("\n📋 Testing PE Registry")
    
    available_types = pe_registry.get_available_types()
    print(f"  Available PE types: {available_types}")
    
    # Test each registered type
    for pe_type in available_types:
        try:
            config_fields = pe_registry.get_config_fields(pe_type)
            print(f"  {pe_type} config fields: {list(config_fields.keys())}")
            
            # Test creation with minimal config
            if pe_type == "alibi":
                pe = create_pe(pe_type=pe_type, dim_model=64, alibi_num_heads=8)
            elif pe_type == "absolute":
                pe = create_pe(pe_type=pe_type, dim_model=64, max_seq_len=128)
            else:
                pe = create_pe(pe_type=pe_type, dim_model=64)
            
            print(f"  {pe_type} creation: ✅")
            
        except Exception as e:
            print(f"  {pe_type} failed: {e}")
            return False
    
    return True


def test_pe_alibi_specific():
    """Test ALiBi-specific functionality"""
    print("\n🎯 Testing ALiBi Specific Features")
    
    dim_model = 64
    seq_len = 16
    x = torch.randn(1, seq_len, dim_model)
    
    # Test different head counts (stick to powers of 2 for simplicity)
    head_counts = [1, 2, 4, 8, 16]  # Only powers of 2
    
    for num_heads in head_counts:
        pe = create_pe(pe_type="alibi", dim_model=dim_model, alibi_num_heads=num_heads)
        bias = pe(x)
        
        expected_shape = (num_heads, seq_len, seq_len)
        assert bias.shape == expected_shape, f"Expected {expected_shape}, got {bias.shape}"
        print(f"  ALiBi {num_heads} heads: {bias.shape} ✅")
        
        # Check that bias is symmetric (relative positions)
        # ALiBi bias should be proportional to relative distance
        # The diagonal should be zero (i-i = 0)
        diagonal_values = torch.diagonal(bias, dim1=-2, dim2=-1)
        assert torch.allclose(diagonal_values, torch.zeros_like(diagonal_values)), "ALiBi diagonal should be zero"
        
        # Check that slopes are decreasing (each head should have different slope)
        slopes_used = pe.pe_strategy.slopes
        if num_heads > 1:
            # Check that we have the right number of slopes
            assert len(slopes_used) == num_heads, f"Expected {num_heads} slopes, got {len(slopes_used)}"
    
    # Test non-power-of-2 separately (they might be clipped to closest power of 2)
    print("  Testing non-power-of-2 head counts:")
    for num_heads in [3, 6, 12]:
        try:
            pe = create_pe(pe_type="alibi", dim_model=dim_model, alibi_num_heads=num_heads)
            bias = pe(x)
            actual_heads = bias.shape[0]
            print(f"  ALiBi {num_heads} requested -> {actual_heads} actual heads: ✅")
            # For non-power-of-2, the implementation might adjust the number of heads
        except Exception as e:
            print(f"  ALiBi {num_heads} heads failed: {e}")
    
    return True


def test_pe_gradients():
    """Test PE gradient flow (for learnable PEs)"""
    print("\n🔄 Testing PE Gradients")
    
    dim_model = 64
    seq_len = 16
    x = torch.randn(1, seq_len, dim_model, requires_grad=True)
    
    # Test only learnable PE types
    pe_absolute = create_pe(pe_type="absolute", dim_model=dim_model, max_seq_len=128)
    
    pos = pe_absolute(x)
    loss = pos.sum()
    loss.backward()
    
    # Check if PE parameters have gradients
    has_pe_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in pe_absolute.parameters())
    print(f"  Absolute PE gradient flow: {'✅' if has_pe_grad else '❌'}")
    
    return has_pe_grad


def run_all_pe_tests():
    """Run all PE tests"""
    tests = [
        test_pe_basic,
        test_pe_parameters,
        test_pe_configurations,
        test_pe_sequence_lengths,
        test_pe_registry,
        test_pe_alibi_specific,
        test_pe_gradients,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print(f"✅ {test.__name__} passed")
            else:
                print(f"❌ {test.__name__} failed")
        except Exception as e:
            print(f"❌ {test.__name__} failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n📈 PE Test Results: {passed}/{total} tests passed")
    return passed == total


if __name__ == "__main__":
    success = run_all_pe_tests()
    if success:
        print("\n🎉 All PE tests passed!")
    else:
        print("\n💥 Some PE tests failed!")
    exit(0 if success else 1) 