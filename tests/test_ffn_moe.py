#!/usr/bin/env python3
"""Test MOE FFN implementations"""

import torch
import torch.nn as nn
import pytest
from boring_llm.nn.ffn.main import create_ffn, create_moe_ffn, BoringFeedForward, FFNConfig, ffn_registry


def test_moe_basic():
    """Test basic MOE functionality"""
    print("🚀 Testing MOE Basic Functionality")
    
    # Test data
    batch_size, seq_len, dim_model = 2, 32, 128
    x = torch.randn(batch_size, seq_len, dim_model)
    print(f"Input tensor shape: {x.shape}")
    
    # ========== MOE Tests ==========
    print("\n📦 MOE Tests:")
    
    # 1. Basic MOE
    moe1 = create_moe_ffn(num_experts=4, top_k=2, dim_model=dim_model)
    y1 = moe1(x)
    print(f"  Basic MOE: {x.shape} -> {y1.shape} ✅")
    assert y1.shape == x.shape, f"Expected {x.shape}, got {y1.shape}"
    
    # 2. MOE with different parameters
    moe2 = create_moe_ffn(
        num_experts=8, 
        top_k=1, 
        dim_model=dim_model, 
        mult_dim=2,
        capacity_factor=1.5,
        noise_std=0.1
    )
    y2 = moe2(x)
    print(f"  Configured MOE: {x.shape} -> {y2.shape} ✅")
    assert y2.shape == x.shape, f"Expected {x.shape}, got {y2.shape}"
    
    # 3. MOE with config object
    config = FFNConfig(
        type="sparse_moe", 
        dim_model=dim_model, 
        num_experts=6, 
        top_k=2,
        dropout=0.1,
        activation=nn.SiLU
    )
    moe3 = BoringFeedForward(config)
    y3 = moe3(x)
    print(f"  Config MOE: {x.shape} -> {y3.shape} ✅")
    assert y3.shape == x.shape, f"Expected {x.shape}, got {y3.shape}"
    
    # 4. MOE using create_ffn directly
    moe4 = create_ffn(
        ffn_type="sparse_moe",
        dim_model=dim_model,
        num_experts=4,
        top_k=2
    )
    y4 = moe4(x)
    print(f"  Direct create_ffn MOE: {x.shape} -> {y4.shape} ✅")
    assert y4.shape == x.shape, f"Expected {x.shape}, got {y4.shape}"
    
    return True


def test_moe_parameters():
    """Test MOE parameter counts and configurations"""
    print("\n📊 Testing MOE Parameters")
    
    dim_model = 128
    
    # Standard FFN for comparison
    ffn_standard = create_ffn(ffn_type="standard", dim_model=dim_model, mult_dim=4)
    standard_params = sum(p.numel() for p in ffn_standard.parameters())
    print(f"  Standard FFN: {standard_params:,} parameters")
    
    # MOE FFN with different expert counts
    moe_4 = create_moe_ffn(num_experts=4, top_k=2, dim_model=dim_model, mult_dim=4)
    moe_4_params = sum(p.numel() for p in moe_4.parameters())
    print(f"  MOE (4 experts): {moe_4_params:,} parameters")
    
    moe_8 = create_moe_ffn(num_experts=8, top_k=2, dim_model=dim_model, mult_dim=4)
    moe_8_params = sum(p.numel() for p in moe_8.parameters())
    print(f"  MOE (8 experts): {moe_8_params:,} parameters")
    
    # Verify parameter scaling
    assert moe_8_params > moe_4_params, "8-expert MOE should have more parameters than 4-expert"
    assert moe_4_params > standard_params, "MOE should have more parameters than standard FFN"
    
    # Test expert-specific parameters
    print(f"  Expert scaling ratio (8/4): {moe_8_params / moe_4_params:.2f}")
    print(f"  MOE vs Standard ratio (4-expert): {moe_4_params / standard_params:.2f}")
    
    return True


def test_moe_configurations():
    """Test different MOE configurations"""
    print("\n⚙️ Testing MOE Configurations")
    
    batch_size, seq_len, dim_model = 1, 16, 64
    x = torch.randn(batch_size, seq_len, dim_model)
    
    configs = [
        # Basic configurations
        {"num_experts": 4, "top_k": 1, "dim_model": dim_model},
        {"num_experts": 4, "top_k": 2, "dim_model": dim_model},
        {"num_experts": 8, "top_k": 2, "dim_model": dim_model},
        
        # With different activations
        {"num_experts": 4, "top_k": 2, "dim_model": dim_model, "activation": nn.ReLU},
        {"num_experts": 4, "top_k": 2, "dim_model": dim_model, "activation": "SiLU"},
        
        # With capacity and noise settings
        {"num_experts": 6, "top_k": 2, "dim_model": dim_model, "capacity_factor": 1.5, "noise_std": 0.1},
        {"num_experts": 4, "top_k": 1, "dim_model": dim_model, "capacity_factor": 0.8, "noise_std": 0.0},
        
        # With dropout
        {"num_experts": 4, "top_k": 2, "dim_model": dim_model, "dropout": 0.1},
        {"num_experts": 4, "top_k": 2, "dim_model": dim_model, "mult_dim": 2, "dropout": 0.2},
    ]
    
    for i, config in enumerate(configs):
        try:
            moe = create_moe_ffn(**config)
            y = moe(x)
            print(f"  Config {i+1}: experts={config['num_experts']}, top_k={config['top_k']} -> {y.shape} ✅")
            assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"
        except Exception as e:
            print(f"  Config {i+1}: Failed with error: {e} ❌")
            return False
    
    return True


def test_moe_registry():
    """Test MOE registry functionality"""
    print("\n📋 Testing MOE Registry")
    
    available_types = ffn_registry.get_available_types()
    print(f"  Available FFN types: {available_types}")
    
    # Check if MOE types are registered
    moe_types = ["sparse_moe", "expert", "router"]
    for moe_type in moe_types:
        if moe_type in available_types:
            print(f"  {moe_type} registered: ✅")
            
            # Test config fields
            config_fields = ffn_registry.get_config_fields(moe_type)
            print(f"    Config fields: {list(config_fields.keys())}")
        else:
            print(f"  {moe_type} not registered: ❌")
            return False
    
    # Test direct registry usage
    try:
        # Test expert creation
        expert = ffn_registry.create_strategy(
            "expert", 
            dim_model=64, 
            inner_dim=256, 
            expert_id=0
        )
        print(f"  Expert creation: ✅")
        
        # Test router creation
        router = ffn_registry.create_strategy(
            "router",
            dim_model=64,
            num_experts=4,
            top_k=2
        )
        print(f"  Router creation: ✅")
        
        # Test sparse_moe creation
        sparse_moe = ffn_registry.create_strategy(
            "sparse_moe",
            dim_model=64,
            inner_dim=256,
            num_experts=4,
            top_k=2
        )
        print(f"  Sparse MOE creation: ✅")
        
    except Exception as e:
        print(f"  Registry creation failed: {e} ❌")
        return False
    
    return True


def test_moe_routing():
    """Test MOE routing behavior"""
    print("\n🔀 Testing MOE Routing")
    
    dim_model = 64
    batch_size, seq_len = 2, 8
    x = torch.randn(batch_size, seq_len, dim_model)
    
    # Test different top-k values
    for num_experts in [4, 8]:
        for top_k in [1, 2]:
            if top_k <= num_experts:
                moe = create_moe_ffn(
                    num_experts=num_experts,
                    top_k=top_k,
                    dim_model=dim_model,
                    noise_std=0.0  # Disable noise for deterministic testing
                )
                
                # Test forward pass
                y = moe(x)
                print(f"  Experts={num_experts}, top_k={top_k}: {y.shape} ✅")
                assert y.shape == x.shape
                
                # Test that output is not all zeros (routing is working)
                assert y.abs().sum() > 0, "MOE output should not be all zeros"
    
    return True


def test_moe_gradients():
    """Test MOE gradient flow"""
    print("\n🔄 Testing MOE Gradients")
    
    dim_model = 64
    x = torch.randn(2, 10, dim_model, requires_grad=True)
    
    # Test gradient flow for different MOE configurations
    configs = [
        {"num_experts": 4, "top_k": 1},
        {"num_experts": 4, "top_k": 2},
        {"num_experts": 8, "top_k": 2},
    ]
    
    for config in configs:
        moe = create_moe_ffn(dim_model=dim_model, **config)
        
        y = moe(x)
        loss = y.sum()
        loss.backward()
        
        # Check if gradients exist
        has_grad = x.grad is not None and x.grad.abs().sum() > 0
        print(f"  experts={config['num_experts']}, top_k={config['top_k']} gradient flow: {'✅' if has_grad else '❌'}")
        
        # Check expert gradients
        expert_has_grads = all(
            p.grad is not None and p.grad.abs().sum() > 0 
            for expert in moe.transform.experts 
            for p in expert.parameters()
        )
        print(f"    Expert gradients: {'✅' if expert_has_grads else '❌'}")
        
        # Clear gradients for next test
        x.grad = None
        moe.zero_grad()
    
    return True


def test_moe_training_mode():
    """Test MOE behavior in training vs eval mode"""
    print("\n🎓 Testing MOE Training/Eval Modes")
    
    dim_model = 64
    x = torch.randn(2, 10, dim_model)
    
    # Create MOE with noise
    moe = create_moe_ffn(
        num_experts=4,
        top_k=2,
        dim_model=dim_model,
        noise_std=1.0  # High noise for testing
    )
    
    # Test training mode
    moe.train()
    y_train1 = moe(x)
    y_train2 = moe(x)
    
    # Test eval mode
    moe.eval()
    y_eval1 = moe(x)
    y_eval2 = moe(x)
    
    # In eval mode, outputs should be more consistent (less noise)
    train_diff = (y_train1 - y_train2).abs().mean()
    eval_diff = (y_eval1 - y_eval2).abs().mean()
    
    print(f"  Training mode difference: {train_diff:.6f}")
    print(f"  Eval mode difference: {eval_diff:.6f}")
    print(f"  Training/Eval behavior: {'✅' if train_diff >= eval_diff else '❌'}")
    
    return train_diff >= eval_diff


def test_moe_capacity():
    """Test MOE expert capacity functionality"""
    print("\n📊 Testing MOE Expert Capacity")
    
    dim_model = 64
    x = torch.randn(2, 10, dim_model)
    
    # Test different capacity factors
    capacity_factors = [0.5, 1.0, 1.5, float('inf')]
    
    for capacity_factor in capacity_factors:
        moe = create_moe_ffn(
            num_experts=4,
            top_k=2,
            dim_model=dim_model,
            capacity_factor=capacity_factor
        )
        
        y = moe(x)
        print(f"  Capacity factor {capacity_factor}: {y.shape} ✅")
        assert y.shape == x.shape
        
        # Verify output is not all zeros
        assert y.abs().sum() > 0
    
    return True


def run_all_moe_tests():
    """Run all MOE tests"""
    tests = [
        test_moe_basic,
        test_moe_parameters,
        test_moe_configurations,
        test_moe_registry,
        test_moe_routing,
        test_moe_gradients,
        test_moe_training_mode,
        test_moe_capacity,
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
    
    print(f"\n📈 MOE Test Results: {passed}/{total} tests passed")
    return passed == total


if __name__ == "__main__":
    success = run_all_moe_tests()
    if success:
        print("\n🎉 All MOE tests passed!")
    else:
        print("\n💥 Some MOE tests failed!")
    exit(0 if success else 1) 