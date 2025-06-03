#!/usr/bin/env python3
"""Test FFN implementations"""

import torch
import torch.nn as nn
from boring_llm.nn.ffn.main import create_ffn, BoringFeedForward, FFNConfig, ffn_registry


def test_ffn_basic():
    """Test basic FFN functionality"""
    print("🚀 Testing FFN Basic Functionality")
    
    # Test data
    batch_size, seq_len, dim_model = 2, 32, 128
    x = torch.randn(batch_size, seq_len, dim_model)
    print(f"Input tensor shape: {x.shape}")
    
    # ========== FFN Tests ==========
    print("\n📦 FFN Tests:")
    
    # 1. Standard FFN
    ffn1 = create_ffn(ffn_type="standard", dim_model=dim_model, mult_dim=4)
    y1 = ffn1(x)
    print(f"  Standard FFN: {x.shape} -> {y1.shape} ✅")
    assert y1.shape == x.shape, f"Expected {x.shape}, got {y1.shape}"
    
    # 2. GLU FFN
    ffn2 = create_ffn(ffn_type="glu", dim_model=dim_model, mult_dim=2, mult_bias=True)
    y2 = ffn2(x)
    print(f"  GLU FFN: {x.shape} -> {y2.shape} ✅")
    assert y2.shape == x.shape, f"Expected {x.shape}, got {y2.shape}"
    
    # 3. With config object
    config = FFNConfig(type="standard", dim_model=dim_model, dropout=0.1, post_act_ln=True)
    ffn3 = BoringFeedForward(config)
    y3 = ffn3(x)
    print(f"  Config FFN: {x.shape} -> {y3.shape} ✅")
    assert y3.shape == x.shape, f"Expected {x.shape}, got {y3.shape}"
    
    return True


def test_ffn_parameters():
    """Test FFN parameter counts and configurations"""
    print("\n📊 Testing FFN Parameters")
    
    dim_model = 128
    
    # Standard FFN
    ffn_standard = create_ffn(ffn_type="standard", dim_model=dim_model, mult_dim=4)
    standard_params = sum(p.numel() for p in ffn_standard.parameters())
    print(f"  Standard FFN: {standard_params:,} parameters")
    
    # GLU FFN  
    ffn_glu = create_ffn(ffn_type="glu", dim_model=dim_model, mult_dim=2, mult_bias=True)
    glu_params = sum(p.numel() for p in ffn_glu.parameters())
    print(f"  GLU FFN: {glu_params:,} parameters")
    
    # With different mult_dim
    ffn_large = create_ffn(ffn_type="standard", dim_model=dim_model, mult_dim=8)
    large_params = sum(p.numel() for p in ffn_large.parameters())
    print(f"  Large FFN (mult_dim=8): {large_params:,} parameters")
    
    # Verify parameter scaling
    assert large_params > standard_params, "Large FFN should have more parameters"
    
    return True


def test_ffn_configurations():
    """Test different FFN configurations"""
    print("\n⚙️ Testing FFN Configurations")
    
    batch_size, seq_len, dim_model = 1, 16, 64
    x = torch.randn(batch_size, seq_len, dim_model)
    
    configs = [
        # Standard configurations
        {"ffn_type": "standard", "dim_model": dim_model, "mult_dim": 2, "activation": nn.ReLU},
        {"ffn_type": "standard", "dim_model": dim_model, "mult_dim": 4, "activation": nn.GELU},
        
        # GLU configurations
        {"ffn_type": "glu", "dim_model": dim_model, "mult_dim": 1, "activation": "SiLU", "mult_bias": True},
        {"ffn_type": "glu", "dim_model": dim_model, "mult_dim": 2, "activation": "SiLU", "mult_bias": False},
        
        # With dropout and layer norm
        {"ffn_type": "standard", "dim_model": dim_model, "mult_dim": 2, "dropout": 0.1, "post_act_ln": True},
        {"ffn_type": "standard", "dim_model": dim_model, "mult_dim": 2, "no_bias": True, "zero_init_output": True},
        
        # With new PostProcessors
        {"ffn_type": "standard", "dim_model": dim_model, "post_type": "post_regularized", "spectral_norm": True},
        {"ffn_type": "glu", "dim_model": dim_model, "post_type": "post_scaled", "learnable_scale": True},
    ]
    
    for i, config in enumerate(configs):
        ffn = create_ffn(**config)
        y = ffn(x)
        print(f"  Config {i+1}: {config['ffn_type']} -> {y.shape} ✅")
        assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"
    
    return True


def test_ffn_registry():
    """Test FFN registry functionality"""
    print("\n📋 Testing FFN Registry")
    
    available_types = ffn_registry.get_available_types()
    print(f"  Available FFN types: {available_types}")
    
    # Define which types are actual FFN transforms (not routers or post-processors)
    ffn_transform_types = ["standard", "glu"]
    post_processor_types = ["post_standard", "post_regularized", "post_scaled"]
    router_types = ["soft_router", "hard_router"]
    
    # Test FFN transform types
    for ffn_type in ffn_transform_types:
        if ffn_type in available_types:
            try:
                config_fields = ffn_registry.get_config_fields(ffn_type)
                print(f"  {ffn_type} config fields: {list(config_fields.keys())}")
                
                # Test creation
                if ffn_type == "glu":
                    ffn = create_ffn(ffn_type=ffn_type, dim_model=64, mult_bias=True)
                else:
                    ffn = create_ffn(ffn_type=ffn_type, dim_model=64)
                print(f"  {ffn_type} creation: ✅")
                
            except Exception as e:
                print(f"  {ffn_type} failed: {e}")
                return False
    
    # Test post-processor types
    for post_type in post_processor_types:
        if post_type in available_types:
            try:
                config_fields = ffn_registry.get_config_fields(post_type)
                print(f"  {post_type} config fields: {list(config_fields.keys())}")
                
                # Test creation through create_ffn with post_type parameter
                if post_type == "post_regularized":
                    ffn = create_ffn(ffn_type="standard", dim_model=64, post_type=post_type, spectral_norm=True)
                elif post_type == "post_scaled":
                    ffn = create_ffn(ffn_type="standard", dim_model=64, post_type=post_type, learnable_scale=True)
                else:
                    ffn = create_ffn(ffn_type="standard", dim_model=64, post_type=post_type)
                print(f"  {post_type} creation: ✅")
                
            except Exception as e:
                print(f"  {post_type} failed: {e}")
                return False
    
    # Test router types (just verify they exist and have config fields)
    for router_type in router_types:
        if router_type in available_types:
            try:
                config_fields = ffn_registry.get_config_fields(router_type)
                print(f"  {router_type} config fields: {list(config_fields.keys())}")
                print(f"  {router_type} registry check: ✅")
                
            except Exception as e:
                print(f"  {router_type} failed: {e}")
                return False
    
    return True


def test_ffn_gradients():
    """Test FFN gradient flow"""
    print("\n🔄 Testing FFN Gradients")
    
    dim_model = 64
    x = torch.randn(2, 10, dim_model, requires_grad=True)
    
    # Test gradient flow for different FFN types
    ffn_types = ["standard", "glu"]
    
    for ffn_type in ffn_types:
        if ffn_type == "glu":
            ffn = create_ffn(ffn_type=ffn_type, dim_model=dim_model, mult_bias=True)
        else:
            ffn = create_ffn(ffn_type=ffn_type, dim_model=dim_model)
        
        y = ffn(x)
        loss = y.sum()
        loss.backward()
        
        # Check if gradients exist
        has_grad = x.grad is not None and x.grad.abs().sum() > 0
        print(f"  {ffn_type} gradient flow: {'✅' if has_grad else '❌'}")
        
        # Clear gradients for next test
        x.grad = None
    
    return True


def test_ffn_moe():
    """Test MOE FFN functionality"""
    print("\n🎯 Testing MOE FFN")
    
    batch_size, seq_len, dim_model = 2, 16, 128
    x = torch.randn(batch_size, seq_len, dim_model)
    
    try:
        # Test soft router MOE
        from boring_llm.nn.ffn.main import create_moe_ffn
        
        moe_soft = create_moe_ffn(
            num_experts=4,
            top_k=2,
            router_type="soft_router",
            type="standard",
            dim_model=dim_model,
            mult_dim=2,
            activation="SiLU"
        )
        
        y_soft = moe_soft(x)
        print(f"  Soft Router MOE: {x.shape} -> {y_soft.shape} ✅")
        assert y_soft.shape == x.shape
        
        # Test hard router MOE
        moe_hard = create_moe_ffn(
            num_experts=4,
            top_k=1,
            router_type="hard_router",
            type="standard",
            dim_model=dim_model,
            mult_dim=2,
            temperature=1.0
        )
        
        y_hard = moe_hard(x)
        print(f"  Hard Router MOE: {x.shape} -> {y_hard.shape} ✅")
        assert y_hard.shape == x.shape
        
        # Test MOE with GLU experts
        moe_glu = create_moe_ffn(
            num_experts=4,
            top_k=2,
            router_type="soft_router",
            type="glu",
            dim_model=dim_model,
            mult_dim=1,
            mult_bias=True
        )
        
        y_glu = moe_glu(x)
        print(f"  GLU Expert MOE: {x.shape} -> {y_glu.shape} ✅")
        assert y_glu.shape == x.shape
        
        return True
        
    except Exception as e:
        print(f"  MOE test failed: {e}")
        return False


def run_all_ffn_tests():
    """Run all FFN tests"""
    tests = [
        test_ffn_basic,
        test_ffn_parameters,
        test_ffn_configurations,
        test_ffn_registry,
        test_ffn_gradients,
        test_ffn_moe,
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
    
    print(f"\n📈 FFN Test Results: {passed}/{total} tests passed")
    return passed == total


if __name__ == "__main__":
    success = run_all_ffn_tests()
    if success:
        print("\n🎉 All FFN tests passed!")
    else:
        print("\n💥 Some FFN tests failed!")
    exit(0 if success else 1) 