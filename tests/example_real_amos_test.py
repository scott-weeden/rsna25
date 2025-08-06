#!/usr/bin/env python3
"""
Example: How to PROPERLY test IRIS with real AMOS data

This shows the correct way to validate medical image segmentation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
from pathlib import Path


def example_real_validation():
    """Example of how real validation SHOULD work."""
    print("📚 EXAMPLE: Real AMOS Data Validation")
    print("="*60)
    
    print("\nStep 1: Import required libraries")
    print("```python")
    print("import nibabel as nib")
    print("import numpy as np")
    print("import torch")
    print("from models.iris_model import IRISModel")
    print("from evaluation.evaluation_metrics import SegmentationMetrics")
    print("```")
    
    print("\nStep 2: Load REAL medical images")
    print("```python")
    print("# Load reference image and mask")
    print("ref_image_path = 'src/data/amos/imagesTr/amos_0005.nii.gz'")
    print("ref_label_path = 'src/data/amos/labelsTr/amos_0005.nii.gz'")
    print("")
    print("ref_nii = nib.load(ref_image_path)")
    print("ref_image = ref_nii.get_fdata()  # Shape: (512, 512, 85)")
    print("ref_label = nib.load(ref_label_path).get_fdata()")
    print("")
    print("# Extract liver mask (label 6 in AMOS)")
    print("ref_liver_mask = (ref_label == 6).astype(np.float32)")
    print("```")
    
    print("\nStep 3: Preprocess for model")
    print("```python")
    print("# Normalize CT values")
    print("ref_image = np.clip(ref_image, -1000, 1000)")
    print("ref_image = (ref_image + 1000) / 2000  # [0, 1]")
    print("")
    print("# Resize to model input size")
    print("target_size = (64, 128, 128)")
    print("ref_image_resized = resize_medical_volume(ref_image, target_size)")
    print("ref_mask_resized = resize_medical_volume(ref_liver_mask, target_size)")
    print("")
    print("# Convert to tensors")
    print("ref_image_tensor = torch.from_numpy(ref_image_resized)")
    print("ref_mask_tensor = torch.from_numpy(ref_mask_resized)")
    print("```")
    
    print("\nStep 4: Create model and encode task")
    print("```python")
    print("# Initialize IRIS model")
    print("model = IRISModel(")
    print("    in_channels=1,")
    print("    base_channels=32,")
    print("    embed_dim=512,")
    print("    num_tokens=10,")
    print("    num_classes=1")
    print(")")
    print("")
    print("# Encode liver segmentation task from reference")
    print("with torch.no_grad():")
    print("    task_embedding = model.encode_task(ref_image_tensor, ref_mask_tensor)")
    print("    print(f'Task embedding shape: {task_embedding.shape}')")
    print("```")
    
    print("\nStep 5: Segment a different patient")
    print("```python")
    print("# Load query image (different patient)")
    print("query_image_path = 'src/data/amos/imagesTr/amos_0006.nii.gz'")
    print("query_label_path = 'src/data/amos/labelsTr/amos_0006.nii.gz'")
    print("")
    print("query_image = load_and_preprocess(query_image_path)")
    print("query_label = nib.load(query_label_path).get_fdata()")
    print("query_liver_gt = (query_label == 6)  # Ground truth")
    print("")
    print("# Perform segmentation using task embedding")
    print("with torch.no_grad():")
    print("    pred_logits = model.segment_with_task(query_image, task_embedding)")
    print("    pred_mask = torch.sigmoid(pred_logits) > 0.5")
    print("```")
    
    print("\nStep 6: Compute REAL metrics")
    print("```python")
    print("# Calculate actual Dice score")
    print("dice_score = SegmentationMetrics.dice_coefficient(pred_mask, query_liver_gt)")
    print("print(f'Real Dice Score: {dice_score:.4f}')")
    print("")
    print("# Calculate IoU")
    print("iou = SegmentationMetrics.iou(pred_mask, query_liver_gt)")
    print("print(f'IoU: {iou:.4f}')")
    print("")
    print("# This is REAL validation - not fake formulas!")
    print("```")


def compare_fake_vs_real():
    """Show the stark difference between fake and real testing."""
    print("\n\n⚖️  FAKE vs REAL Testing Comparison")
    print("="*60)
    
    print("\n❌ FAKE (Current Tests):")
    print("```python")
    print("# Random noise as 'medical image'")
    print("image = torch.randn(1, 1, 32, 64, 64)")
    print("")
    print("# Random binary as 'organ mask'")
    print("mask = torch.randint(0, 2, (1, 1, 32, 64, 64))")
    print("")
    print("# Hard-coded 'Dice score'")
    print("dice = similarity * 0.5 + 0.2  # Always 0.1-0.7")
    print("```")
    
    print("\n✅ REAL (Required):")
    print("```python")
    print("# Actual CT scan")
    print("image = nib.load('amos_0005.nii.gz').get_fdata()")
    print("")
    print("# Radiologist-annotated organ mask")
    print("mask = (label_volume == 6)  # Liver")
    print("")
    print("# Actual segmentation accuracy")
    print("dice = 2 * (pred & gt).sum() / (pred.sum() + gt.sum())")
    print("```")
    
    print("\n📊 Results:")
    print("FAKE: Always produces 10-70% 'Dice' regardless of model")
    print("REAL: Measures actual segmentation performance")


def show_amos_organ_labels():
    """Show what organs are available in AMOS."""
    print("\n\n🏥 AMOS Dataset Organ Labels")
    print("="*60)
    
    organs = {
        0: 'background',
        1: 'spleen',
        2: 'right kidney',
        3: 'left kidney', 
        4: 'gallbladder',
        5: 'esophagus',
        6: 'liver',
        7: 'stomach',
        8: 'aorta',
        9: 'inferior vena cava',
        10: 'pancreas',
        11: 'right adrenal gland',
        12: 'left adrenal gland',
        13: 'duodenum',
        14: 'bladder',
        15: 'prostate/uterus'
    }
    
    print("\nAvailable organs for testing:")
    for label, name in organs.items():
        if label > 0:
            print(f"  {label:2d}: {name}")
    
    print("\nFor novel class testing:")
    print("  Train on: organs 1-10")
    print("  Test on: organs 11-15 (never seen during training)")


def create_validation_checklist():
    """Create a checklist for proper validation."""
    print("\n\n✅ VALIDATION CHECKLIST")
    print("="*60)
    
    steps = [
        "Install nibabel: pip install nibabel",
        "Check AMOS data exists in src/data/amos/",
        "Load real CT/MRI images (not torch.randn)",
        "Load real organ masks (not torch.randint)",
        "Preprocess: normalize HU values, resize appropriately",
        "Use fixed decoder from decoder_3d_fixed.py",
        "Encode task from reference patient",
        "Segment query patient (different from reference)",
        "Compute real Dice score (not formula)",
        "Compare against paper claims (28-69% for novel)",
        "Document actual vs expected performance",
        "Update all test files to use real data"
    ]
    
    print("\nSteps for proper validation:")
    for i, step in enumerate(steps, 1):
        print(f"  {i:2d}. {step}")


def main():
    """Main example demonstration."""
    print("🎯 IRIS Framework: Real Validation Example")
    print("="*60)
    print("This shows how to PROPERLY test with medical data\n")
    
    # Show example of real validation
    example_real_validation()
    
    # Compare fake vs real
    compare_fake_vs_real()
    
    # Show available organs
    show_amos_organ_labels()
    
    # Create checklist
    create_validation_checklist()
    
    # Final message
    print("\n\n⚠️  IMPORTANT:")
    print("="*60)
    print("The IRIS framework claims to achieve:")
    print("  • 28-69% Dice on novel organs")
    print("  • 82-86% Dice on cross-dataset") 
    print("  • 89.56% Dice on in-distribution")
    print("")
    print("But current tests use:")
    print("  • torch.randn() → random noise")
    print("  • dice = sim * 0.5 + 0.2 → fake formula")
    print("")
    print("This example shows how to test PROPERLY with real data!")
    print("")
    print("Next step: Implement this example and get REAL results!")


if __name__ == "__main__":
    main()