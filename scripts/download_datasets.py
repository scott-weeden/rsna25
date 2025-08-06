#!/usr/bin/env python3
"""
Dataset download script for IRIS framework validation.

This script downloads the necessary medical imaging datasets for proper validation
of the IRIS framework implementation. The test files currently use synthetic data,
but real validation requires actual medical images.

Datasets:
1. AMOS - Already partially available in src/data/amos/
2. KiTS19 - Kidney tumor segmentation challenge
3. Medical Segmentation Decathlon - Multiple organ segmentation tasks
"""

import os
import sys
import requests
import zipfile
import tarfile
from pathlib import Path
import hashlib
from tqdm import tqdm


class DatasetDownloader:
    """Downloads and organizes medical imaging datasets."""
    
    def __init__(self, base_dir="data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Dataset URLs and metadata
        self.datasets = {
            'amos': {
                'name': 'AMOS - Abdominal Multi-Organ Segmentation',
                'url': 'https://zenodo.org/record/7262581',  # AMOS22 challenge
                'size': '~50GB',
                'format': 'nii.gz',
                'note': 'Partially available in src/data/amos/',
                'organs': ['spleen', 'kidney_right', 'kidney_left', 'gallbladder', 
                          'liver', 'stomach', 'pancreas', 'adrenal_gland']
            },
            'kits19': {
                'name': 'KiTS19 - Kidney Tumor Segmentation',
                'url': 'https://github.com/neheller/kits19',
                'size': '~30GB',
                'format': 'nii.gz',
                'note': 'Requires git-lfs for full data',
                'organs': ['kidney', 'tumor']
            },
            'msd': {
                'name': 'Medical Segmentation Decathlon',
                'url': 'http://medicaldecathlon.com/',
                'size': 'Varies by task',
                'format': 'nii.gz',
                'tasks': ['Task01_BrainTumour', 'Task02_Heart', 'Task03_Liver',
                         'Task04_Hippocampus', 'Task05_Prostate', 'Task06_Lung',
                         'Task07_Pancreas', 'Task08_HepaticVessel', 'Task09_Spleen',
                         'Task10_Colon']
            },
            'acdc': {
                'name': 'ACDC - Automated Cardiac Diagnosis Challenge',
                'url': 'https://www.creatis.insa-lyon.fr/Challenge/acdc/',
                'size': '~1GB',
                'format': 'nii.gz',
                'note': 'For generalization testing',
                'organs': ['left_ventricle', 'right_ventricle', 'myocardium']
            },
            'segthor': {
                'name': 'SegTHOR - Thoracic Organs at Risk',
                'url': 'https://competitions.codalab.org/competitions/21145',
                'size': '~5GB',
                'format': 'nii.gz',
                'note': 'For generalization testing',
                'organs': ['heart', 'aorta', 'trachea', 'esophagus']
            }
        }
    
    def check_existing_data(self):
        """Check what data is already available."""
        print("\n📊 Checking existing data...")
        print("="*60)
        
        # Check AMOS data
        amos_path = Path("src/data/amos")
        if amos_path.exists():
            train_images = list(amos_path.glob("imagesTr/*.nii.gz"))
            train_labels = list(amos_path.glob("labelsTr/*.nii.gz"))
            val_images = list(amos_path.glob("imagesVa/*.nii.gz"))
            val_labels = list(amos_path.glob("labelsVa/*.nii.gz"))
            test_images = list(amos_path.glob("imagesTs/*.nii.gz"))
            
            print(f"✅ AMOS dataset found:")
            print(f"   - Training: {len(train_images)} images, {len(train_labels)} labels")
            print(f"   - Validation: {len(val_images)} images, {len(val_labels)} labels")
            print(f"   - Test: {len(test_images)} images")
            
            if len(train_images) > 0 and len(train_labels) > 0:
                print(f"   ⚠️  AMOS data is available but tests are not using it!")
        else:
            print("❌ AMOS dataset not found at expected location")
        
        # Check KiTS19 data
        kits_path = Path("src/data/kits19")
        if kits_path.exists():
            print(f"✅ KiTS19 directory found (check for actual data)")
        
        print("\n")
    
    def download_instructions(self):
        """Print download instructions for each dataset."""
        print("\n📥 Dataset Download Instructions")
        print("="*60)
        
        for key, info in self.datasets.items():
            print(f"\n{key.upper()}: {info['name']}")
            print(f"  URL: {info['url']}")
            print(f"  Size: {info['size']}")
            print(f"  Format: {info['format']}")
            if 'note' in info:
                print(f"  Note: {info['note']}")
            if 'organs' in info:
                print(f"  Organs: {', '.join(info['organs'])}")
            if 'tasks' in info:
                print(f"  Tasks: {len(info['tasks'])} different segmentation tasks")
        
        print("\n" + "="*60)
    
    def download_msd_sample(self, task='Task09_Spleen'):
        """Download a sample task from Medical Segmentation Decathlon."""
        print(f"\n🔄 Downloading MSD {task} as sample...")
        
        # MSD Google Drive links (these are public)
        msd_links = {
            'Task09_Spleen': 'https://drive.google.com/uc?id=1jzeNU1EKnK81PyTsrx0ujfNl-t0Jo8uE',
            'Task04_Hippocampus': 'https://drive.google.com/uc?id=1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C'
        }
        
        if task not in msd_links:
            print(f"❌ Download link not available for {task}")
            return False
        
        output_dir = self.base_dir / 'msd' / task
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Note: Actual download would require handling Google Drive API
        print(f"📁 Download directory: {output_dir}")
        print(f"🔗 Manual download required from: {msd_links[task]}")
        print("   Extract the tar file to the directory above")
        
        return True
    
    def create_data_loader_config(self):
        """Create configuration file for data loaders."""
        config = """# Data Loader Configuration for IRIS Framework

# Paths to datasets (update based on your downloads)
dataset_paths:
  amos:
    train_images: "src/data/amos/imagesTr"
    train_labels: "src/data/amos/labelsTr"
    val_images: "src/data/amos/imagesVa"
    val_labels: "src/data/amos/labelsVa"
    test_images: "src/data/amos/imagesTs"
    
  kits19:
    base_path: "data/kits19"
    
  msd:
    base_path: "data/msd"
    tasks:
      - "Task09_Spleen"
      - "Task04_Hippocampus"
      
  acdc:
    base_path: "data/acdc"
    
  segthor:
    base_path: "data/segthor"

# Data preprocessing settings
preprocessing:
  target_spacing: [1.5, 1.5, 1.5]  # mm
  target_size: [128, 128, 128]      # voxels
  normalize: true
  clip_range: [-1000, 1000]         # HU for CT
  
# Training data split
data_split:
  train: 0.7
  val: 0.15
  test: 0.15
"""
        
        config_path = self.base_dir / "data_config.yaml"
        with open(config_path, 'w') as f:
            f.write(config)
        
        print(f"\n✅ Created data configuration at: {config_path}")
        return config_path
    
    def create_amos_loader_example(self):
        """Create example code for loading AMOS data."""
        example_code = '''"""
Example: Loading AMOS dataset for IRIS framework validation.

This shows how to properly load and use the AMOS data that's already available.
"""

import nibabel as nib
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader


class AMOSDataset(Dataset):
    """AMOS dataset loader for medical image segmentation."""
    
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Set paths based on split
        if split == 'train':
            self.image_dir = self.root_dir / 'imagesTr'
            self.label_dir = self.root_dir / 'labelsTr'
        elif split == 'val':
            self.image_dir = self.root_dir / 'imagesVa'
            self.label_dir = self.root_dir / 'labelsVa'
        elif split == 'test':
            self.image_dir = self.root_dir / 'imagesTs'
            self.label_dir = None  # Test set has no labels
        
        # Get all image files
        self.image_files = sorted(list(self.image_dir.glob('*.nii.gz')))
        
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
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        image_nii = nib.load(image_path)
        image = image_nii.get_fdata()
        
        # Load label if available
        label = None
        if self.label_dir is not None:
            label_path = self.label_dir / image_path.name
            if label_path.exists():
                label_nii = nib.load(label_path)
                label = label_nii.get_fdata()
        
        # Get metadata
        spacing = image_nii.header.get_zooms()
        patient_id = image_path.stem
        
        sample = {
            'image': image,
            'label': label,
            'spacing': spacing,
            'patient_id': patient_id,
            'image_path': str(image_path)
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


def load_amos_for_testing():
    """Load AMOS dataset for IRIS framework testing."""
    # Create dataset
    amos_dataset = AMOSDataset(
        root_dir='src/data/amos',
        split='train'
    )
    
    print(f"Loaded {len(amos_dataset)} AMOS training samples")
    
    # Load a sample
    sample = amos_dataset[0]
    print(f"Sample shape: {sample['image'].shape}")
    if sample['label'] is not None:
        print(f"Label shape: {sample['label'].shape}")
        unique_labels = np.unique(sample['label'])
        print(f"Unique labels: {unique_labels}")
    
    return amos_dataset


if __name__ == "__main__":
    # Test loading
    dataset = load_amos_for_testing()
    
    # Create a proper data loader
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )
    
    # Test iteration
    for batch_idx, batch in enumerate(dataloader):
        print(f"\\nBatch {batch_idx}:")
        print(f"  Image shape: {batch['image'].shape}")
        if batch['label'] is not None:
            print(f"  Label shape: {batch['label'].shape}")
        break
'''
        
        example_path = self.base_dir / "load_amos_example.py"
        with open(example_path, 'w') as f:
            f.write(example_code)
        
        print(f"\n✅ Created AMOS loader example at: {example_path}")
        return example_path


def main():
    """Main function to check data and provide download instructions."""
    print("🏥 IRIS Framework Dataset Download Manager")
    print("="*60)
    
    downloader = DatasetDownloader()
    
    # Check existing data
    downloader.check_existing_data()
    
    # Print download instructions
    downloader.download_instructions()
    
    # Create configuration
    downloader.create_data_loader_config()
    
    # Create AMOS loader example
    downloader.create_amos_loader_example()
    
    print("\n⚠️  IMPORTANT NOTES:")
    print("1. AMOS data is already available in src/data/amos/")
    print("2. The test files are NOT using this real data")
    print("3. Update the test files to use the AMOSDataset loader")
    print("4. Replace synthetic data with actual medical images")
    print("5. Compute real Dice scores instead of simulated ones")
    
    print("\n📝 Next Steps:")
    print("1. Use the example loader in data/load_amos_example.py")
    print("2. Update test files to load real AMOS data")
    print("3. Implement proper Dice score computation")
    print("4. Validate claims with actual segmentation results")


if __name__ == "__main__":
    main()