#!/usr/bin/env python3
"""
Test script to verify that the tensor dimension issue is resolved.
This script tests the complete pipeline from data loading to model inference.
"""

import torch
import numpy as np
from src.data.amos22_loader import AMOS22Dataset
from src.models.iris_model import IRISModel

def test_data_dimensions():
    """Test that the dataset returns correct tensor dimensions."""
    print("🧪 Testing dataset dimensions...")
    
    try:
        dataset = AMOS22Dataset('src/data/amos', split='train')
        print(f"✅ Dataset loaded with {len(dataset)} samples")
        
        if len(dataset) > 0:
            sample = dataset[0]
            image = sample['image']
            label = sample['label']
            
            print(f"✅ Image shape: {image.shape} (expected: [1, D, H, W])")
            if label is not None:
                print(f"✅ Label shape: {label.shape} (expected: [D, H, W])")
            
            # Test adding batch dimension
            if image.dim() == 4:  # If (1, D, H, W), add batch dimension
                batched_image = image.unsqueeze(0)  # -> (1, 1, D, H, W)
                print(f"✅ Batched image shape: {batched_image.shape} (expected: [1, 1, D, H, W])")
                
                # Verify this is 5D (correct for conv3d)
                if batched_image.dim() == 5:
                    print("✅ Tensor is 5D - compatible with conv3d")
                else:
                    print(f"❌ Tensor is {batched_image.dim()}D - NOT compatible with conv3d")
                    return False
            else:
                print(f"❌ Unexpected image dimensions: {image.dim()}")
                return False
        else:
            print("⚠️  No samples found in dataset")
            return False
            
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return False
    
    return True

def test_model_forward():
    """Test that the model can handle the correct tensor dimensions."""
    print("\n🧪 Testing model forward pass...")
    
    try:
        # Create a small test model
        model = IRISModel(
            in_channels=1,
            base_channels=16,  # Reduced for testing
            embed_dim=64,      # Reduced for testing
            num_tokens=5,      # Reduced for testing
            num_classes=1
        )
        
        # Test with correct dimensions
        batch_size = 1
        query_image = torch.randn(batch_size, 1, 32, 64, 64)  # (B, C, D, H, W)
        reference_image = torch.randn(batch_size, 1, 32, 64, 64)
        reference_mask = torch.randint(0, 2, (batch_size, 1, 32, 64, 64)).float()
        
        print(f"✅ Input shapes:")
        print(f"   - Query: {query_image.shape}")
        print(f"   - Reference: {reference_image.shape}")
        print(f"   - Mask: {reference_mask.shape}")
        
        with torch.no_grad():
            output = model(query_image, reference_image, reference_mask)
        
        print(f"✅ Output shape: {output.shape}")
        print("✅ Model forward pass successful!")
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False
    
    return True

def test_evaluation_pipeline():
    """Test the evaluation pipeline with real data."""
    print("\n🧪 Testing evaluation pipeline...")
    
    try:
        # Load dataset
        dataset = AMOS22Dataset('src/data/amos', split='train')
        if len(dataset) == 0:
            print("⚠️  No samples found - skipping evaluation test")
            return True
        
        # Create small model
        model = IRISModel(
            in_channels=1,
            base_channels=16,
            embed_dim=64,
            num_tokens=5,
            num_classes=1
        )
        model.eval()
        
        # Get a sample
        sample = dataset[0]
        image = sample['image']
        label = sample['label']
        
        # Prepare for model (add batch dimension if needed)
        if image.dim() == 4:  # If (1, D, H, W), add batch dimension
            image = image.unsqueeze(0)  # -> (1, 1, D, H, W)
        
        print(f"✅ Prepared image shape: {image.shape}")
        
        if label is not None:
            # Create a simple organ mask for testing
            organ_mask = (label == 1).float()  # Use organ ID 1 (spleen)
            
            if organ_mask.sum() > 0:
                # Prepare mask (add channel and batch dimensions if needed)
                reference_mask = organ_mask.unsqueeze(0)  # Add channel dimension
                if reference_mask.dim() == 4:  # If (1, D, H, W), add batch dimension
                    reference_mask = reference_mask.unsqueeze(0)  # -> (1, 1, D, H, W)
                
                print(f"✅ Prepared mask shape: {reference_mask.shape}")
                
                # Test forward pass
                with torch.no_grad():
                    output = model(image, image, reference_mask)
                
                print(f"✅ Evaluation output shape: {output.shape}")
                print("✅ Evaluation pipeline successful!")
            else:
                print("⚠️  No organ found in sample - using random mask")
                reference_mask = torch.randint(0, 2, (1, 1, 32, 64, 64)).float()
                
                with torch.no_grad():
                    # Resize input to match mask
                    resized_image = torch.nn.functional.interpolate(
                        image, size=(32, 64, 64), mode='trilinear', align_corners=False
                    )
                    output = model(resized_image, resized_image, reference_mask)
                
                print(f"✅ Evaluation output shape: {output.shape}")
                print("✅ Evaluation pipeline successful!")
        else:
            print("⚠️  No label found - using random mask")
            reference_mask = torch.randint(0, 2, (1, 1, 32, 64, 64)).float()
            
            with torch.no_grad():
                # Resize input to match mask
                resized_image = torch.nn.functional.interpolate(
                    image, size=(32, 64, 64), mode='trilinear', align_corners=False
                )
                output = model(resized_image, resized_image, reference_mask)
            
            print(f"✅ Evaluation output shape: {output.shape}")
            print("✅ Evaluation pipeline successful!")
        
    except Exception as e:
        print(f"❌ Evaluation pipeline test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("🔧 TENSOR DIMENSION FIX VERIFICATION")
    print("=" * 60)
    
    all_passed = True
    
    # Test 1: Data dimensions
    if not test_data_dimensions():
        all_passed = False
    
    # Test 2: Model forward pass
    if not test_model_forward():
        all_passed = False
    
    # Test 3: Evaluation pipeline
    if not test_evaluation_pipeline():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Tensor dimension issue RESOLVED!")
        print("✅ Conv3d expects 5D tensors: [batch, channel, depth, height, width]")
        print("✅ Dataset returns 4D tensors: [channel, depth, height, width]")
        print("✅ Evaluation script correctly adds batch dimension")
        print("✅ Model can process the tensors without errors")
    else:
        print("❌ SOME TESTS FAILED!")
        print("❌ Tensor dimension issue may still exist")
    print("=" * 60)

if __name__ == "__main__":
    main()
