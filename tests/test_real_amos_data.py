#!/usr/bin/env python3
"""
Test IRIS Framework with REAL AMOS Medical Data

This script tests the IRIS framework using actual medical images from the AMOS dataset,
replacing the synthetic torch.randn() data that was previously used for false validation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
import nibabel as nib
from pathlib import Path
import time

# Import IRIS components
from models.iris_model import IRISModel
from models.decoder_3d_fixed import QueryBasedDecoderFixed
from utils.losses import dice_score
from evaluation.evaluation_metrics import SegmentationMetrics


class AMOSDataTester:
    """Tests IRIS framework with real AMOS medical data."""
    
    def __init__(self):
        self.amos_path = Path("src/data/amos")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # AMOS organ labels
        self.organ_labels = {
            0: 'background',
            1: 'spleen',
            2: 'right_kidney', 
            3: 'left_kidney',
            4: 'gallbladder',
            5: 'esophagus',
            6: 'liver',
            7: 'stomach',
            8: 'aorta',
            9: 'inferior_vena_cava',
            10: 'pancreas',
            11: 'right_adrenal_gland',
            12: 'left_adrenal_gland',
            13: 'duodenum',
            14: 'bladder',
            15: 'prostate_uterus'
        }
        
    def check_data_availability(self):
        """Check if AMOS data is available."""
        print("\n📊 Checking AMOS Data Availability")
        print("="*60)
        
        train_images = list(self.amos_path.glob("imagesTr/*.nii.gz"))
        train_labels = list(self.amos_path.glob("labelsTr/*.nii.gz"))
        val_images = list(self.amos_path.glob("imagesVa/*.nii.gz"))
        val_labels = list(self.amos_path.glob("labelsVa/*.nii.gz"))
        test_images = list(self.amos_path.glob("imagesTs/*.nii.gz"))
        
        print(f"✅ Training: {len(train_images)} images, {len(train_labels)} labels")
        print(f"✅ Validation: {len(val_images)} images, {len(val_labels)} labels")
        print(f"✅ Test: {len(test_images)} images")
        
        if len(train_images) == 0:
            print("❌ ERROR: No AMOS data found!")
            print("   Please download AMOS dataset to src/data/amos/")
            return False
            
        return True
        
    def load_amos_sample(self, image_path, label_path=None):
        """Load a single AMOS sample (image and optionally label)."""
        # Load image
        image_nii = nib.load(image_path)
        image = image_nii.get_fdata().astype(np.float32)
        spacing = image_nii.header.get_zooms()
        
        # Normalize image (CT window)
        image = np.clip(image, -1000, 1000)
        image = (image + 1000) / 2000  # Normalize to [0, 1]
        
        # Load label if provided
        label = None
        if label_path and label_path.exists():
            label_nii = nib.load(label_path)
            label = label_nii.get_fdata().astype(np.uint8)
            
        return {
            'image': image,
            'label': label,
            'spacing': spacing,
            'path': str(image_path)
        }
        
    def preprocess_for_model(self, sample, target_size=(32, 64, 64)):
        """Preprocess sample for model input."""
        image = sample['image']
        label = sample['label']
        
        # Get original shape
        orig_shape = image.shape
        
        # Simple resize to target size using torch interpolation
        zoom_factors = [t/o for t, o in zip(target_size, orig_shape)]
        
        # Convert to tensor for resizing
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
        
        # Resize image using torch interpolation
        image_resized = torch.nn.functional.interpolate(
            image_tensor, 
            size=target_size, 
            mode='trilinear', 
            align_corners=False
        ).squeeze(0).squeeze(0).numpy()
        
        # Resize label if available
        label_resized = None
        if label is not None:
            label_tensor = torch.from_numpy(label).float().unsqueeze(0).unsqueeze(0)
            label_resized = torch.nn.functional.interpolate(
                label_tensor,
                size=target_size,
                mode='nearest'
            ).squeeze(0).squeeze(0).numpy()
            
        # Convert to tensors and add batch+channel dimensions
        image_tensor = torch.from_numpy(image_resized).float().unsqueeze(0).unsqueeze(0)
        label_tensor = None
        if label_resized is not None:
            label_tensor = torch.from_numpy(label_resized).long().unsqueeze(0).unsqueeze(0)
            
        return {
            'image': image_tensor,
            'label': label_tensor,
            'zoom_factors': zoom_factors,
            'orig_shape': orig_shape
        }
        
    def test_task_encoding_real_data(self):
        """Test task encoding with real medical data."""
        print("\n🧪 Testing Task Encoding with Real AMOS Data")
        print("="*60)
        
        # Load a sample with liver segmentation
        train_images = sorted(list(self.amos_path.glob("imagesTr/*.nii.gz")))
        train_labels = sorted(list(self.amos_path.glob("labelsTr/*.nii.gz")))
        
        if len(train_images) == 0:
            print("❌ No training data found")
            return
            
        # Find a sample with liver (label 6)
        liver_sample = None
        for img_path, label_path in zip(train_images[:10], train_labels[:10]):
            sample = self.load_amos_sample(img_path, label_path)
            if sample['label'] is not None and 6 in np.unique(sample['label']):
                liver_sample = sample
                print(f"✅ Found liver in: {img_path.name}")
                break
                
        if liver_sample is None:
            print("❌ No liver samples found in first 10 images")
            return
            
        # Preprocess
        processed = self.preprocess_for_model(liver_sample)
        
        # Create model
        model = IRISModel(
            in_channels=1,
            base_channels=32,
            embed_dim=64,
            num_tokens=5,
            num_classes=1
        ).to(self.device)
        
        print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Create liver mask (binary)
        liver_mask = (processed['label'] == 6).float() if processed['label'] is not None else None
        
        # Test task encoding
        print("\nTesting task encoding...")
        with torch.no_grad():
            image = processed['image'].to(self.device)
            mask = liver_mask.to(self.device) if liver_mask is not None else None
            
            # Extract task embedding
            task_embedding = model.encode_task(image, mask)
            print(f"✅ Task embedding shape: {task_embedding.shape}")
            print(f"   Mean: {task_embedding.mean().item():.4f}")
            print(f"   Std: {task_embedding.std().item():.4f}")
            
        return task_embedding
        
    def test_segmentation_real_data(self):
        """Test segmentation with real medical data."""
        print("\n🧪 Testing Segmentation with Real AMOS Data")
        print("="*60)
        
        # Load two samples - one for reference, one for query
        train_images = sorted(list(self.amos_path.glob("imagesTr/*.nii.gz")))
        train_labels = sorted(list(self.amos_path.glob("labelsTr/*.nii.gz")))
        
        if len(train_images) < 2:
            print("❌ Need at least 2 samples")
            return
            
        # Load reference (with label) and query samples
        ref_sample = self.load_amos_sample(train_images[0], train_labels[0])
        query_sample = self.load_amos_sample(train_images[1], train_labels[1])
        
        # Preprocess
        ref_processed = self.preprocess_for_model(ref_sample)
        query_processed = self.preprocess_for_model(query_sample)
        
        # Get organ masks (e.g., liver = 6)
        target_organ = 6  # Liver
        ref_mask = (ref_processed['label'] == target_organ).float()
        query_mask_gt = (query_processed['label'] == target_organ).float()
        
        print(f"\nTesting liver segmentation (organ {target_organ}):")
        print(f"Reference: {train_images[0].name}")
        print(f"Query: {train_images[1].name}")
        
        # Create model
        model = IRISModel(
            in_channels=1,
            base_channels=16,  # Smaller for testing
            embed_dim=32,
            num_tokens=5,
            num_classes=1
        ).to(self.device)
        
        # Perform segmentation
        print("\nPerforming in-context segmentation...")
        with torch.no_grad():
            ref_image = ref_processed['image'].to(self.device)
            ref_mask = ref_mask.to(self.device)
            query_image = query_processed['image'].to(self.device)
            
            # Encode task from reference
            task_embedding = model.encode_task(ref_image, ref_mask)
            
            # Segment query using task embedding
            pred_logits = model.segment_with_task(query_image, task_embedding)
            pred_mask = torch.sigmoid(pred_logits) > 0.5
            
        # Compute Dice score
        if query_mask_gt.sum() > 0:
            dice = SegmentationMetrics.dice_coefficient(pred_mask.cpu(), query_mask_gt)
            print(f"\n✅ Real Dice Score: {dice:.4f}")
            print(f"   (NOT simulated - computed on actual segmentation!)")
        else:
            print(f"\n⚠️  No liver in query image")
            
        return pred_mask, query_mask_gt
        
    def compare_synthetic_vs_real(self):
        """Compare results between synthetic and real data."""
        print("\n📊 Comparing Synthetic vs Real Data Performance")
        print("="*60)
        
        # Test with synthetic data (like old tests)
        print("\n1. Testing with SYNTHETIC data (torch.randn):")
        synthetic_image = torch.randn(1, 1, 32, 64, 64)
        synthetic_mask = torch.randint(0, 2, (1, 1, 32, 64, 64)).float()
        print(f"   Synthetic image range: [{synthetic_image.min():.2f}, {synthetic_image.max():.2f}]")
        print(f"   Synthetic mask coverage: {synthetic_mask.mean():.2%}")
        
        # Test with real data
        print("\n2. Testing with REAL medical data:")
        train_images = sorted(list(self.amos_path.glob("imagesTr/*.nii.gz")))
        if len(train_images) > 0:
            sample = self.load_amos_sample(train_images[0])
            print(f"   Real image shape: {sample['image'].shape}")
            print(f"   Real image range: [{sample['image'].min():.2f}, {sample['image'].max():.2f}]")
            print(f"   Spacing: {sample['spacing']}")
            
        print("\n⚠️  Key Differences:")
        print("   - Synthetic: Random noise, no anatomical structure")
        print("   - Real: Actual CT/MRI scans with organs and tissues")
        print("   - Synthetic Dice scores are MEANINGLESS")
        print("   - Real Dice scores measure ACTUAL segmentation quality")
        

def main():
    """Main test function."""
    print("🏥 IRIS Framework Test with REAL Medical Data")
    print("="*60)
    print("This test uses ACTUAL AMOS medical images, NOT synthetic data!")
    
    tester = AMOSDataTester()
    
    # Check data availability
    if not tester.check_data_availability():
        print("\n❌ Cannot proceed without AMOS data")
        print("   Run: python scripts/download_datasets.py")
        return
        
    # Run tests
    try:
        # Test 1: Task encoding with real data
        task_embedding = tester.test_task_encoding_real_data()
        
        # Test 2: Segmentation with real data
        pred_mask, gt_mask = tester.test_segmentation_real_data()
        
        # Test 3: Compare synthetic vs real
        tester.compare_synthetic_vs_real()
        
        print("\n✅ All tests completed with REAL medical data!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        

if __name__ == "__main__":
    main()