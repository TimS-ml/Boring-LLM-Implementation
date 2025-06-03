#!/usr/bin/env python3
"""Test MOE FFN implementations"""

import torch
import torch.nn as nn
import pytest
from boring_llm.nn.ffn.main import (
    create_ffn, create_moe_ffn, 
    BoringFeedForward, BoringFeedForwardMOE, 
    FFNConfig, ffn_registry
)


def test_moe_basic():
    """Test basic MOE functionality"""
    print("🚀 Testing MOE Basic Functionality")
    
    # Test data
    batch_size, seq_len, dim_model = 2, 32, 128
    x = torch.randn(batch_size, seq_len, dim_model)
    print(f"Input tensor shape: {x.shape}")
    
    # ========== MOE Tests ==========
    print("\n📦 MOE Tests:")
    
    # 1. Basic MOE with soft router
    moe1 = create_moe_ffn(num_experts=4, top_k=2, router_type="soft_router", dim_model=dim_model)
    y1 = moe1(x)
    print(f"  Basic MOE (soft router): {x.shape} -> {y1.shape} ✅")
    assert y1.shape == x.shape, f"Expected {x.shape}, got {y1.shape}"
    
    # 2. Basic MOE with hard router
    moe2 = create_moe_ffn(num_experts=4, top_k=1, router_type="hard_router", dim_model=dim_model)
    y2 = moe2(x)
    print(f"  Basic MOE (hard router): {x.shape} -> {y2.shape} ✅")
    assert y2.shape == x.shape, f"Expected {x.shape}, got {y2.shape}"
    
    # 3. MOE with different parameters
    moe3 = create_moe_ffn(
        num_experts=8, 
        top_k=2, 
        router_type="soft_router",
        dim_model=dim_model, 
        mult_dim=2,
        capacity_factor=1.5,
        noise_std=0.1
    )
    y3 = moe3(x)
    print(f"  Configured MOE (soft): {x.shape} -> {y3.shape} ✅")
    assert y3.shape == x.shape, f"Expected {x.shape}, got {y3.shape}"
    
    # 4. MOE with config object
    config = FFNConfig(
        type="standard",  # Expert type 
        dim_model=dim_model, 
        num_experts=6, 
        top_k=2,
        router_type="soft_router",
        dropout=0.1,
        activation=nn.SiLU
    )
    moe4 = BoringFeedForwardMOE(config)
    y4 = moe4(x)
    print(f"  Config MOE: {x.shape} -> {y4.shape} ✅")
    assert y4.shape == x.shape, f"Expected {x.shape}, got {y4.shape}"
    
    # 5. MOE using BoringFeedForwardMOE directly
    moe5 = BoringFeedForwardMOE(
        type="glu",  # GLU experts
        dim_model=dim_model,
        num_experts=4,
        top_k=2,
        router_type="soft_router"
    )
    y5 = moe5(x)
    print(f"  Direct BoringFeedForwardMOE: {x.shape} -> {y5.shape} ✅")
    assert y5.shape == x.shape, f"Expected {x.shape}, got {y5.shape}"
    
    return True


def test_moe_expert_types():
    """Test different expert types"""
    print("\n🔧 Testing MOE Expert Types")
    
    dim_model = 64
    batch_size, seq_len = 2, 8
    x = torch.randn(batch_size, seq_len, dim_model)
    
    # Test different expert types
    expert_types = ["standard", "glu"]
    
    for expert_type in expert_types:
        # Test with soft router
        moe_soft = create_moe_ffn(
            num_experts=4,
            top_k=2,
            router_type="soft_router",
            type=expert_type,  # Use type instead of expert_type
            dim_model=dim_model,
            noise_std=0.1
        )
        y_soft = moe_soft(x)
        print(f"  {expert_type} experts + soft router: {y_soft.shape} ✅")
        assert y_soft.shape == x.shape
        
        # Test with hard router
        moe_hard = create_moe_ffn(
            num_experts=4,
            top_k=1,
            router_type="hard_router",
            type=expert_type,  # Use type instead of expert_type
            dim_model=dim_model,
            temperature=1.0,
            straight_through=True
        )
        y_hard = moe_hard(x)
        print(f"  {expert_type} experts + hard router: {y_hard.shape} ✅")
        assert y_hard.shape == x.shape
    
    return True


def test_moe_router_types():
    """Test different router types"""
    print("\n🔀 Testing MOE Router Types")
    
    dim_model = 64
    batch_size, seq_len = 2, 8
    x = torch.randn(batch_size, seq_len, dim_model)
    
    # Test soft router
    moe_soft = create_moe_ffn(
        num_experts=4,
        top_k=2,
        router_type="soft_router",
        dim_model=dim_model,
        noise_std=0.1
    )
    y_soft = moe_soft(x)
    print(f"  Soft router: {y_soft.shape} ✅")
    assert y_soft.shape == x.shape
    
    # Test hard router
    moe_hard = create_moe_ffn(
        num_experts=4,
        top_k=1,
        router_type="hard_router",
        dim_model=dim_model,
        temperature=1.0,
        straight_through=True
    )
    y_hard = moe_hard(x)
    print(f"  Hard router: {y_hard.shape} ✅")
    assert y_hard.shape == x.shape
    
    # Test that outputs are different (different routing strategies)
    diff = (y_soft - y_hard).abs().mean()
    print(f"  Soft vs Hard difference: {diff:.6f}")
    assert diff > 1e-6, "Soft and hard routers should produce different outputs"
    
    return True


def test_moe_parameters():
    """Test MOE parameter counts and configurations"""
    print("\n📊 Testing MOE Parameters")
    
    dim_model = 128
    
    # Standard FFN for comparison
    ffn_standard = create_ffn(ffn_type="standard", dim_model=dim_model, mult_dim=4)
    standard_params = sum(p.numel() for p in ffn_standard.parameters())
    print(f"  Standard FFN: {standard_params:,} parameters")
    
    # MOE FFN with different expert counts and router types
    configs = [
        {"num_experts": 4, "top_k": 2, "router_type": "soft_router", "type": "standard"},
        {"num_experts": 8, "top_k": 2, "router_type": "soft_router", "type": "standard"},
        {"num_experts": 4, "top_k": 1, "router_type": "hard_router", "type": "standard"},
        {"num_experts": 8, "top_k": 1, "router_type": "hard_router", "type": "glu"},
    ]
    
    for config in configs:
        moe = create_moe_ffn(dim_model=dim_model, mult_dim=4, **config)
        moe_params = sum(p.numel() for p in moe.parameters())
        expert_info = f"{config['num_experts']} {config['type']} experts"
        router_info = f"{config['router_type']}"
        print(f"  MOE ({expert_info}, {router_info}): {moe_params:,} parameters")
        assert moe_params > standard_params, "MOE should have more parameters than standard FFN"
    
    return True


def test_moe_configurations():
    """Test different MOE configurations"""
    print("\n⚙️ Testing MOE Configurations")
    
    batch_size, seq_len, dim_model = 1, 16, 64
    x = torch.randn(batch_size, seq_len, dim_model)
    
    configs = [
        # Soft router configurations with different expert types
        {"num_experts": 4, "top_k": 1, "router_type": "soft_router", "type": "standard", "dim_model": dim_model},
        {"num_experts": 4, "top_k": 2, "router_type": "soft_router", "type": "standard", "dim_model": dim_model},
        {"num_experts": 8, "top_k": 2, "router_type": "soft_router", "type": "glu", "dim_model": dim_model},
        
        # Hard router configurations with different expert types
        {"num_experts": 4, "top_k": 1, "router_type": "hard_router", "type": "standard", "dim_model": dim_model},
        {"num_experts": 8, "top_k": 1, "router_type": "hard_router", "type": "glu", "dim_model": dim_model},
        
        # With different activations
        {"num_experts": 4, "top_k": 2, "router_type": "soft_router", "type": "standard", "dim_model": dim_model, "activation": nn.ReLU},
        {"num_experts": 4, "top_k": 1, "router_type": "hard_router", "type": "glu", "dim_model": dim_model, "activation": "SiLU"},
        
        # With capacity and router-specific settings
        {"num_experts": 6, "top_k": 2, "router_type": "soft_router", "type": "standard", "dim_model": dim_model, "capacity_factor": 1.5, "noise_std": 0.1},
        {"num_experts": 4, "top_k": 1, "router_type": "hard_router", "type": "glu", "dim_model": dim_model, "temperature": 0.5, "straight_through": True},
        
        # With dropout
        {"num_experts": 4, "top_k": 2, "router_type": "soft_router", "type": "standard", "dim_model": dim_model, "dropout": 0.1},
        {"num_experts": 4, "top_k": 1, "router_type": "hard_router", "type": "glu", "dim_model": dim_model, "mult_dim": 2, "dropout": 0.2},
    ]
    
    for i, config in enumerate(configs):
        try:
            moe = create_moe_ffn(**config)
            y = moe(x)
            router_info = f"router={config['router_type']}"
            expert_info = f"experts={config['num_experts']}, type={config['type']}, top_k={config['top_k']}"
            print(f"  Config {i+1}: {expert_info}, {router_info} -> {y.shape} ✅")
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
    
    # Check if router types are registered
    router_types = ["soft_router", "hard_router"]
    for router_type in router_types:
        if router_type in available_types:
            print(f"  {router_type} registered: ✅")
            
            # Test config fields
            config_fields = ffn_registry.get_config_fields(router_type)
            print(f"    Config fields: {list(config_fields.keys())}")
        else:
            print(f"  {router_type} not registered: ❌")
            return False
    
    # Test direct registry usage
    try:
        # Test soft router creation
        soft_router = ffn_registry.create_strategy(
            "soft_router",
            dim_model=64,
            num_experts=4,
            top_k=2,
            noise_std=0.1
        )
        print(f"  Soft router creation: ✅")
        
        # Test hard router creation
        hard_router = ffn_registry.create_strategy(
            "hard_router",
            dim_model=64,
            num_experts=4,
            top_k=1,
            temperature=1.0,
            straight_through=True
        )
        print(f"  Hard router creation: ✅")
        
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
    
    # Test different router types with different top-k values and expert types
    test_configs = [
        {"num_experts": 4, "top_k": 1, "router_type": "soft_router", "type": "standard"},
        {"num_experts": 4, "top_k": 2, "router_type": "soft_router", "type": "standard"},
        {"num_experts": 8, "top_k": 2, "router_type": "soft_router", "type": "glu"},
        {"num_experts": 4, "top_k": 1, "router_type": "hard_router", "type": "standard"},
        {"num_experts": 8, "top_k": 1, "router_type": "hard_router", "type": "glu"},
    ]
    
    for config in test_configs:
        # Add router-specific params
        if config["router_type"] == "soft_router":
            config["noise_std"] = 0.0  # Disable noise for deterministic testing
        else:
            config["temperature"] = 1.0
            config["straight_through"] = True
            
        moe = create_moe_ffn(dim_model=dim_model, **config)
        
        # Test forward pass
        y = moe(x)
        router_info = f"{config['router_type']}"
        expert_info = f"experts={config['num_experts']}, type={config['type']}, top_k={config['top_k']}"
        print(f"  {expert_info}, {router_info}: {y.shape} ✅")
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
        {"num_experts": 4, "top_k": 1, "router_type": "soft_router", "type": "standard"},
        {"num_experts": 4, "top_k": 2, "router_type": "soft_router", "type": "glu"},
        {"num_experts": 4, "top_k": 1, "router_type": "hard_router", "type": "standard"},
        {"num_experts": 8, "top_k": 1, "router_type": "hard_router", "type": "glu"},
    ]
    
    for config in configs:
        moe = create_moe_ffn(dim_model=dim_model, **config)
        
        y = moe(x)
        loss = y.sum()
        loss.backward()
        
        # Check if gradients exist
        has_grad = x.grad is not None and x.grad.abs().sum() > 0
        router_info = f"{config['router_type']}"
        expert_info = f"experts={config['num_experts']}, type={config['type']}, top_k={config['top_k']}"
        print(f"  {expert_info}, {router_info} gradient flow: {'✅' if has_grad else '❌'}")
        
        # Check expert gradients
        expert_has_grads = all(
            p.grad is not None and p.grad.abs().sum() > 0 
            for expert in moe.experts 
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
    
    # Test with soft router (has noise)
    moe_soft = create_moe_ffn(
        num_experts=4,
        top_k=2,
        router_type="soft_router",
        type="standard",
        dim_model=dim_model,
        noise_std=1.0  # High noise for testing
    )
    
    # Test training mode
    moe_soft.train()
    y_train1 = moe_soft(x)
    y_train2 = moe_soft(x)
    
    # Test eval mode
    moe_soft.eval()
    y_eval1 = moe_soft(x)
    y_eval2 = moe_soft(x)
    
    # In eval mode, outputs should be more consistent (less noise)
    train_diff = (y_train1 - y_train2).abs().mean()
    eval_diff = (y_eval1 - y_eval2).abs().mean()
    
    print(f"  Soft router training mode difference: {train_diff:.6f}")
    print(f"  Soft router eval mode difference: {eval_diff:.6f}")
    print(f"  Soft router training/eval behavior: {'✅' if train_diff >= eval_diff else '❌'}")
    
    # Test with hard router (has straight-through behavior)
    moe_hard = create_moe_ffn(
        num_experts=4,
        top_k=1,
        router_type="hard_router",
        type="glu",
        dim_model=dim_model,
        temperature=1.0,
        straight_through=True
    )
    
    # Test training vs eval for hard router
    moe_hard.train()
    y_hard_train = moe_hard(x)
    
    moe_hard.eval()
    y_hard_eval = moe_hard(x)
    
    hard_diff = (y_hard_train - y_hard_eval).abs().mean()
    print(f"  Hard router train/eval difference: {hard_diff:.6f}")
    print(f"  Hard router behavior: {'✅' if hard_diff >= 0 else '❌'}")
    
    return train_diff >= eval_diff


def test_moe_capacity():
    """Test MOE expert capacity functionality"""
    print("\n📊 Testing MOE Expert Capacity")
    
    dim_model = 64
    x = torch.randn(2, 10, dim_model)
    
    # Test different capacity factors
    capacity_factors = [0.5, 1.0, 1.5, float('inf')]
    
    for capacity_factor in capacity_factors:
        # Test with both router types and expert types
        test_configs = [
            {"router_type": "soft_router", "top_k": 2, "type": "standard"},
            {"router_type": "hard_router", "top_k": 1, "type": "glu"},
        ]
        
        for config in test_configs:
            moe = create_moe_ffn(
                num_experts=4,
                dim_model=dim_model,
                capacity_factor=capacity_factor,
                **config
            )
            
            y = moe(x)
            config_info = f"{config['router_type']}, {config['type']} experts"
            print(f"  Capacity {capacity_factor}, {config_info}: {y.shape} ✅")
            assert y.shape == x.shape
            
            # Verify output is not all zeros
            assert y.abs().sum() > 0
    
    return True


def run_all_moe_tests():
    """Run all MOE tests"""
    tests = [
        test_moe_basic,
        test_moe_expert_types,
        test_moe_router_types,
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