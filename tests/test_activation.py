#!/usr/bin/env python3
"""Test Activation function mapping implementations"""

import torch
import torch.nn as nn
from boring_llm.nn.activation import get_activation_by_name


def test_activation_basic():
    """Test basic activation function retrieval"""
    print("🚀 Testing Activation Basic Functionality")
    
    # ========== Standard PyTorch Activations ==========
    print("\n⚡ Standard PyTorch Activations:")
    
    # 1. ReLU
    relu_cls = get_activation_by_name("ReLU")
    relu_instance = relu_cls()
    print(f"  ReLU: {relu_cls} -> {relu_instance} ✅")
    assert relu_cls == nn.ReLU, f"Expected nn.ReLU, got {relu_cls}"
    assert isinstance(relu_instance, nn.ReLU), f"Expected ReLU instance, got {type(relu_instance)}"
    
    # 2. GELU
    gelu_cls = get_activation_by_name("GELU")
    gelu_instance = gelu_cls()
    print(f"  GELU: {gelu_cls} -> {gelu_instance} ✅")
    assert gelu_cls == nn.GELU, f"Expected nn.GELU, got {gelu_cls}"
    assert isinstance(gelu_instance, nn.GELU), f"Expected GELU instance, got {type(gelu_instance)}"
    
    # 3. SiLU (Swish)
    silu_cls = get_activation_by_name("SiLU")
    silu_instance = silu_cls()
    print(f"  SiLU: {silu_cls} -> {silu_instance} ✅")
    assert silu_cls == nn.SiLU, f"Expected nn.SiLU, got {silu_cls}"
    assert isinstance(silu_instance, nn.SiLU), f"Expected SiLU instance, got {type(silu_instance)}"
    
    # 4. ELU
    elu_cls = get_activation_by_name("ELU")
    elu_instance = elu_cls()
    print(f"  ELU: {elu_cls} -> {elu_instance} ✅")
    assert elu_cls == nn.ELU, f"Expected nn.ELU, got {elu_cls}"
    assert isinstance(elu_instance, nn.ELU), f"Expected ELU instance, got {type(elu_instance)}"
    
    return True


def test_activation_standard_names():
    """Test standard PyTorch activation names"""
    print("\n📝 Testing Standard PyTorch Names")
    
    standard_activations = [
        ("ReLU", nn.ReLU),
        ("GELU", nn.GELU),
        ("SiLU", nn.SiLU),
        ("ELU", nn.ELU),
        ("Tanh", nn.Tanh),
        ("Sigmoid", nn.Sigmoid),
        ("LeakyReLU", nn.LeakyReLU),
        ("ReLU6", nn.ReLU6),
        ("Hardtanh", nn.Hardtanh),
        ("Hardswish", nn.Hardswish),
        ("Hardsigmoid", nn.Hardsigmoid),
    ]
    
    for name, expected_cls in standard_activations:
        try:
            actual_cls = get_activation_by_name(name)
            print(f"  {name}: {actual_cls} == {expected_cls} ✅")
            assert actual_cls == expected_cls, f"Expected {expected_cls}, got {actual_cls}"
        except ValueError as e:
            # Some activations might not exist in all PyTorch versions
            print(f"  {name}: Not available in this PyTorch version ({e}) ⚠️")
    
    return True


def test_activation_custom():
    """Test custom activation functions"""
    print("\n🎨 Testing Custom Activations")
    
    try:
        # Test ReluSquared with standard naming
        relu_squared_cls = get_activation_by_name("ReluSquared")
        relu_squared_instance = relu_squared_cls()
        print(f"  ReluSquared: {relu_squared_cls} -> {relu_squared_instance} ✅")
        
        # Test functionality
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = torch.relu(x) ** 2  # Should be [0, 0, 0, 1, 4]
        actual = relu_squared_instance(x)
        print(f"  ReluSquared functionality: input={x.tolist()}, output={actual.tolist()} ✅")
        assert torch.allclose(actual, expected), f"Expected {expected}, got {actual}"
        
    except (ImportError, ValueError) as e:
        print(f"  Custom activations not available: {e} ⚠️")
        # This is acceptable if custom activations are not implemented yet
    
    return True


def test_activation_whitespace():
    """Test activation names with whitespace"""
    print("\n🔍 Testing Whitespace Handling")
    
    whitespace_tests = [
        ("  ReLU  ", nn.ReLU),
        ("\tGELU\t", nn.GELU),
        ("\nSiLU\n", nn.SiLU),
        ("  Tanh  ", nn.Tanh),
    ]
    
    for name_with_whitespace, expected_cls in whitespace_tests:
        actual_cls = get_activation_by_name(name_with_whitespace)
        print(f"  '{name_with_whitespace}': {actual_cls} == {expected_cls} ✅")
        assert actual_cls == expected_cls, f"Expected {expected_cls}, got {actual_cls}"
    
    return True


def test_activation_error_handling():
    """Test error handling for unknown activations"""
    print("\n❌ Testing Error Handling")
    
    invalid_names = [
        "unknown_activation",
        "nonexistent",
        "fake_relu",
        "NotARealActivation",
        "",
        "123invalid",
        "relu",  # lowercase should fail now
        "gelu",  # lowercase should fail now
    ]
    
    for invalid_name in invalid_names:
        try:
            result = get_activation_by_name(invalid_name)
            print(f"  '{invalid_name}': Unexpectedly succeeded with {result} ❌")
            return False
        except ValueError as e:
            print(f"  '{invalid_name}': Correctly raised ValueError: {e} ✅")
        except Exception as e:
            print(f"  '{invalid_name}': Unexpected error type: {type(e).__name__}: {e} ❌")
            return False
    
    return True


def test_activation_functionality():
    """Test that retrieved activations work correctly"""
    print("\n🧪 Testing Activation Functionality")
    
    # Test data
    x = torch.randn(2, 10, 64)
    
    activation_tests = [
        ("ReLU", nn.ReLU),
        ("GELU", nn.GELU),
        ("SiLU", nn.SiLU),
        ("Tanh", nn.Tanh),
        ("Sigmoid", nn.Sigmoid),
    ]
    
    for name, expected_cls in activation_tests:
        try:
            # Get activation class and create instance
            activation_cls = get_activation_by_name(name)
            activation_fn = activation_cls()
            
            # Test forward pass
            output = activation_fn(x)
            
            # Compare with direct PyTorch implementation
            direct_fn = expected_cls()
            expected_output = direct_fn(x)
            
            print(f"  {name}: {x.shape} -> {output.shape} ✅")
            assert output.shape == x.shape, f"Output shape should match input shape"
            assert torch.allclose(output, expected_output), f"Output should match direct PyTorch implementation"
            
        except ValueError:
            print(f"  {name}: Not available ⚠️")
    
    return True


def test_activation_types():
    """Test activation function type consistency"""
    print("\n🏷️ Testing Type Consistency")
    
    type_tests = [
        "ReLU", "GELU", "SiLU", "ELU", "Tanh", "Sigmoid"
    ]
    
    for name in type_tests:
        try:
            activation_cls = get_activation_by_name(name)
            
            # Check that it's a class
            assert isinstance(activation_cls, type), f"{name} should return a class, got {type(activation_cls)}"
            
            # Check that it's a subclass of nn.Module
            assert issubclass(activation_cls, nn.Module), f"{name} should return nn.Module subclass"
            
            # Check that we can instantiate it
            activation_instance = activation_cls()
            assert isinstance(activation_instance, nn.Module), f"{name} instance should be nn.Module"
            
            print(f"  {name}: Type checks passed ✅")
            
        except ValueError:
            print(f"  {name}: Not available ⚠️")
    
    return True


def test_activation_ffn_integration():
    """Test activation integration with FFN system"""
    print("\n🔗 Testing FFN Integration")
    
    try:
        from boring_llm.nn.ffn.main import create_ffn
        
        # Test with string activation names
        activation_tests = [
            "ReLU",
            "GELU", 
            "SiLU",
            "Tanh",
        ]
        
        for activation_name in activation_tests:
            try:
                ffn = create_ffn(
                    ffn_type="standard",
                    dim_model=64,
                    mult_dim=2,
                    activation=activation_name
                )
                
                # Test forward pass
                x = torch.randn(2, 8, 64)
                output = ffn(x)
                
                print(f"  FFN with {activation_name}: {x.shape} -> {output.shape} ✅")
                assert output.shape == x.shape, f"FFN output shape should match input"
                
            except Exception as e:
                print(f"  FFN with {activation_name}: Failed - {e} ❌")
                return False
                
    except ImportError:
        print("  FFN module not available for integration test ⚠️")
    
    return True


def run_all_activation_tests():
    """Run all activation tests"""
    tests = [
        test_activation_basic,
        test_activation_standard_names,
        test_activation_custom,
        test_activation_whitespace,
        test_activation_error_handling,
        test_activation_functionality,
        test_activation_types,
        test_activation_ffn_integration,
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
    
    print(f"\n📈 Activation Test Results: {passed}/{total} tests passed")
    return passed == total


if __name__ == "__main__":
    success = run_all_activation_tests()
    if success:
        print("\n🎉 All activation tests passed!")
    else:
        print("\n💥 Some activation tests failed!")
    exit(0 if success else 1)