#!/usr/bin/env python3
"""Simple test to check if AMOS data can be loaded."""

import nibabel as nib
import numpy as np
from pathlib import Path

def test_amos_loading():
    """Test if we can load AMOS data."""
    print("Testing AMOS data loading...")
    
    amos_path = Path("src/data/amos")
    
    # Check directories
    print(f"\nAMOS path exists: {amos_path.exists()}")
    
    if amos_path.exists():
        # List subdirectories
        subdirs = [d for d in amos_path.iterdir() if d.is_dir()]
        print(f"Subdirectories: {[d.name for d in subdirs]}")
        
        # Check for images
        train_images = list(amos_path.glob("imagesTr/*.nii.gz"))
        print(f"\nTraining images found: {len(train_images)}")
        
        if len(train_images) > 0:
            # Try to load first image
            first_image = train_images[0]
            print(f"\nLoading: {first_image.name}")
            
            try:
                nii = nib.load(first_image)
                data = nii.get_fdata()
                print(f"Shape: {data.shape}")
                print(f"Data type: {data.dtype}")
                print(f"Value range: [{data.min():.1f}, {data.max():.1f}]")
                print(f"Spacing: {nii.header.get_zooms()}")
                
                # Check if label exists
                label_path = amos_path / "labelsTr" / first_image.name
                if label_path.exists():
                    label_nii = nib.load(label_path)
                    label_data = label_nii.get_fdata()
                    print(f"\nLabel shape: {label_data.shape}")
                    print(f"Unique labels: {np.unique(label_data).astype(int)}")
                else:
                    print(f"\nNo corresponding label found")
                    
            except Exception as e:
                print(f"Error loading: {e}")
    else:
        print("AMOS directory not found!")
        print("Expected path: src/data/amos/")
        

if __name__ == "__main__":
    test_amos_loading()