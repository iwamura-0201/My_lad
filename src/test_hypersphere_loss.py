#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit test for HyperSphereLoss to verify numerical stability fixes.

This test verifies that HyperSphereLoss properly handles:
- Normal tensors
- Batch size of 1
- NaN inputs
- Inf inputs
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path to import the loss module
sys.path.insert(0, str(Path(__file__).parent))

from loss.hypersphereloss import HyperSphereLoss


def test_normal_case():
    """Test with normal tensor input"""
    print("Test 1: Normal case...")
    loss_fn = HyperSphereLoss()
    normal_tensor = torch.randn(32, 256)
    output = {'cls_output': normal_tensor}
    data = {}
    loss = loss_fn(output, data)
    
    assert not torch.isnan(loss), "Loss should not be NaN for normal input"
    assert not torch.isinf(loss), "Loss should not be Inf for normal input"
    print(f"   ✅ Normal case loss: {loss.item():.6f}")


def test_batch_size_one():
    """Test with batch size = 1 (variance undefined)"""
    print("Test 2: Batch size 1...")
    loss_fn = HyperSphereLoss()
    single_batch = torch.randn(1, 256)
    output = {'cls_output': single_batch}
    data = {}
    loss = loss_fn(output, data)
    
    assert not torch.isnan(loss), "Loss should not be NaN for batch size 1"
    assert loss.item() == 0.0, "Loss should be 0 for batch size 1"
    print(f"   ✅ Batch size 1 loss: {loss.item():.6f} (correctly returns 0)")


def test_nan_input():
    """Test with NaN input"""
    print("Test 3: NaN input...")
    loss_fn = HyperSphereLoss()
    nan_tensor = torch.full((32, 256), float('nan'))
    output = {'cls_output': nan_tensor}
    data = {}
    loss = loss_fn(output, data)
    
    assert not torch.isnan(loss), "Loss should handle NaN input gracefully"
    assert loss.item() == 0.0, "Loss should be 0 for NaN input"
    print(f"   ✅ NaN input loss: {loss.item():.6f} (safely handled)")


def test_inf_input():
    """Test with Inf input"""
    print("Test 4: Inf input...")
    loss_fn = HyperSphereLoss()
    inf_tensor = torch.full((32, 256), float('inf'))
    output = {'cls_output': inf_tensor}
    data = {}
    loss = loss_fn(output, data)
    
    assert not torch.isnan(loss), "Loss should handle Inf input gracefully"
    assert not torch.isinf(loss), "Loss should not be Inf for Inf input"
    assert loss.item() == 0.0, "Loss should be 0 for Inf input"
    print(f"   ✅ Inf input loss: {loss.item():.6f} (safely handled)")


def test_large_batch():
    """Test with large batch size"""
    print("Test 5: Large batch (128)...")
    loss_fn = HyperSphereLoss()
    large_batch = torch.randn(128, 512)
    output = {'cls_output': large_batch}
    data = {}
    loss = loss_fn(output, data)
    
    assert not torch.isnan(loss), "Loss should not be NaN for large batch"
    assert not torch.isinf(loss), "Loss should not be Inf for large batch"
    print(f"   ✅ Large batch loss: {loss.item():.6f}")


def test_gradient_flow():
    """Test that gradients flow properly"""
    print("Test 6: Gradient flow...")
    loss_fn = HyperSphereLoss()
    tensor = torch.randn(32, 256, requires_grad=True)
    output = {'cls_output': tensor}
    data = {}
    loss = loss_fn(output, data)
    
    # Backward pass
    loss.backward()
    
    assert tensor.grad is not None, "Gradients should be computed"
    assert not torch.isnan(tensor.grad).any(), "Gradients should not contain NaN"
    assert not torch.isinf(tensor.grad).any(), "Gradients should not contain Inf"
    print(f"   ✅ Gradients computed successfully")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" HyperSphereLoss Numerical Stability Test")
    print("=" * 60 + "\n")
    
    try:
        test_normal_case()
        test_batch_size_one()
        test_nan_input()
        test_inf_input()
        test_large_batch()
        test_gradient_flow()
        
        print("\n" + "=" * 60)
        print(" ✅ ALL TESTS PASSED!")
        print("=" * 60 + "\n")
        print("HyperSphereLoss is numerically stable and handles edge cases correctly.")
        
    except AssertionError as e:
        print("\n" + "=" * 60)
        print(" ❌ TEST FAILED!")
        print("=" * 60 + "\n")
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print("\n" + "=" * 60)
        print(" ❌ UNEXPECTED ERROR!")
        print("=" * 60 + "\n")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
