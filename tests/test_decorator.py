#!/usr/bin/env python3

import os
os.environ['VERBOSE'] = '1'

import torch
import torch.nn as nn
from boring_utils.colorprint import tprint, cprint

# Import our utilities and classes
from boring_llm.utils.utils import debug_init, PrintInitParamsMixin
from boring_llm.nn.ffn.registry import GLUFFN, StandardFFN, PostProcessor, ffn_registry
from boring_llm.nn.pe.registry import FixedPositionalEncoding, pe_registry

def test_decorator_basic():
    """Test basic decorator functionality"""
    tprint("Test 1: Basic Decorator Functionality", sep='=', c='blue')
    
    print("1.1 Testing manual decoration of GLUFFN...")
    DecoratedGLUFFN = debug_init(GLUFFN)
    
    decorated_glu = DecoratedGLUFFN(
        dim_model=128, 
        inner_dim=512, 
        activation=nn.GELU, 
        mult_bias=True, 
        no_bias=False,
        extra_param="test_value"
    )
    
    print("1.2 Testing manual decoration of StandardFFN...")
    DecoratedStandardFFN = debug_init(StandardFFN)
    
    decorated_std = DecoratedStandardFFN(
        dim_model=64,
        inner_dim=256,
        activation=nn.ReLU,
        no_bias=True
    )
    
    return True

def test_registry_decoration():
    """Test that registry automatically decorates classes"""
    tprint("Test 2: Registry Auto-Decoration", sep='=', c='green')
    
    print("2.1 Creating FFN instances through registry...")
    
    # These should be automatically decorated because of the registry
    glu_instance = ffn_registry.create_strategy(
        "glu", 
        dim_model=128, 
        inner_dim=512, 
        activation=nn.SiLU,
        mult_bias=False
    )
    
    std_instance = ffn_registry.create_strategy(
        "standard",
        dim_model=64,
        inner_dim=256,
        activation=nn.GELU
    )
    
    print("2.2 Creating PE instances through registry...")
    
    pe_instance = pe_registry.create_strategy(
        "fixed",
        dim_model=128
    )
    
    return True

def test_functionality_preservation():
    """Test that decoration doesn't break normal functionality"""
    tprint("Test 3: Functionality Preservation", sep='=', c='yellow')
    
    print("3.1 Testing decorated vs original class functionality...")
    
    # Create instances
    original_glu = GLUFFN(dim_model=128, inner_dim=512, activation=nn.GELU, mult_bias=True, no_bias=False)
    decorated_glu_class = debug_init(GLUFFN)
    decorated_glu = decorated_glu_class(dim_model=128, inner_dim=512, activation=nn.GELU, mult_bias=True, no_bias=False)
    
    # Test forward pass
    x = torch.randn(2, 10, 128)
    
    original_output = original_glu.apply(x)
    decorated_output = decorated_glu.apply(x)
    
    print(f"   Original output shape: {original_output.shape}")
    print(f"   Decorated output shape: {decorated_output.shape}")
    
    # Check if shapes match
    assert original_output.shape == decorated_output.shape, "Shapes should match!"
    print("   ✅ Shapes match - functionality preserved!")
    
    # Check output_dim property
    assert original_glu.output_dim == decorated_glu.output_dim, "Output dims should match!"
    print("   ✅ Properties preserved!")
    
    return True

def test_different_parameter_combinations():
    """Test decorator with various parameter combinations"""
    tprint("Test 4: Different Parameter Combinations", sep='=', c='magenta')
    
    test_cases = [
        {
            "name": "GLU with minimal params",
            "class": GLUFFN,
            "params": {"dim_model": 64, "inner_dim": 128}
        },
        {
            "name": "GLU with all params",
            "class": GLUFFN, 
            "params": {
                "dim_model": 128,
                "inner_dim": 512,
                "activation": nn.SiLU,
                "mult_bias": False,
                "no_bias": True,
                "custom_param": "test",
                "another_param": 42
            }
        },
        {
            "name": "Standard FFN with function activation",
            "class": StandardFFN,
            "params": {
                "dim_model": 96,
                "inner_dim": 384,
                "activation": nn.ReLU,
                "no_bias": False
            }
        },
        {
            "name": "PE with kwargs",
            "class": FixedPositionalEncoding,
            "params": {
                "dim_model": 128,
                "extra_arg": "value",
                "numeric_arg": 3.14
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"4.{i} Testing {test_case['name']}...")
        
        DecoratedClass = debug_init(test_case['class'])
        instance = DecoratedClass(**test_case['params'])
        
        print(f"   ✅ {test_case['name']} created successfully!")
    
    return True

def test_edge_cases():
    """Test edge cases and error handling"""
    tprint("Test 5: Edge Cases", sep='=', c='red')
    
    print("5.1 Testing decoration of already decorated class...")
    
    # Double decoration should work
    DecoratedOnce = debug_init(GLUFFN)
    DecoratedTwice = debug_init(DecoratedOnce)
    
    instance = DecoratedTwice(dim_model=64, inner_dim=128)
    print("   ✅ Double decoration works!")
    
    print("5.2 Testing with empty kwargs...")
    
    DecoratedStd = debug_init(StandardFFN)
    instance = DecoratedStd(dim_model=32, inner_dim=64)
    print("   ✅ Empty kwargs handled!")
    
    print("5.3 Testing custom class decoration...")
    
    @debug_init
    class CustomTestClass(PrintInitParamsMixin):
        def __init__(self, a, b, c=None, **kwargs):
            self.a = a
            self.b = b 
            self.c = c
            self.kwargs = kwargs
    
    custom = CustomTestClass(1, 2, c=3, extra="value")
    print("   ✅ Custom class decoration works!")
    
    return True

def test_verbose_control():
    """Test VERBOSE environment variable control"""
    tprint("Test 6: VERBOSE Control", sep='=', c='cyan')
    
    print("6.1 Testing with VERBOSE=1 (current setting)...")
    
    # Current behavior (should print debug info)
    DecoratedClass = debug_init(GLUFFN)
    print("   Creating instance (should see debug output above)...")
    instance = DecoratedClass(dim_model=64, inner_dim=128)
    
    print("6.2 Testing environment variable integration...")
    
    # Test that VERBOSE is properly imported and used
    from boring_utils.helpers import VERBOSE
    print(f"   VERBOSE value: {VERBOSE}")
    print(f"   VERBOSE bool: {bool(VERBOSE)}")
    
    if VERBOSE:
        print("   ✅ VERBOSE is active - debug output should be visible")
    else:
        print("   ⚠️  VERBOSE is inactive - no debug output expected")
    
    return True

def run_all_tests():
    """Run all decorator tests"""
    tprint("Running All Decorator Tests", sep='*', c='white')
    
    tests = [
        test_decorator_basic,
        test_registry_decoration, 
        test_functionality_preservation,
        test_different_parameter_combinations,
        test_edge_cases,
        test_verbose_control
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            print()  # Add spacing between tests
            if test():
                passed += 1
                cprint(f"✅ {test.__name__} PASSED", c='green')
            else:
                cprint(f"❌ {test.__name__} FAILED", c='red')
        except Exception as e:
            cprint(f"❌ {test.__name__} FAILED with error: {e}", c='red')
            import traceback
            traceback.print_exc()
    
    print()
    tprint(f"Test Results: {passed}/{total} tests passed", sep='=', c='white')
    
    if passed == total:
        cprint("🎉 All decorator tests passed!", c='green')
    else:
        cprint(f"💥 {total - passed} tests failed!", c='red')
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)