#!/usr/bin/env python3
"""
Medical Dataset Download Script for IRIS Framework Validation

This script downloads the real medical imaging datasets needed to properly
validate the IRIS framework claims, replacing the synthetic data that was
previously used for false validation.

Required Datasets:
1. AMOS22 (500 CT + 100 MRI) - Primary training dataset
2. BCV (30 CT scans) - Cross-dataset generalization
3. LiTS (131 CT scans) - Liver segmentation validation  
4. KiTS19 (210 CT scans) - Kidney segmentation validation
5. MSD Pancreas - Novel class testing
6. Pelvic1K - Bone segmentation testing
"""

import os
import sys
import requests
import zipfile
import tarfile
import gzip
import shutil
from pathlib import Path
from urllib.parse import urlparse
import hashlib

class MedicalDatasetDownloader:
    """Downloads and organizes medical imaging datasets for IRIS validation."""
    
    def __init__(self, data_dir="./datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Dataset configurations
        self.datasets = {
            'amos22': {
                'name': 'AMOS22 (A large-scale abdominal multi-organ benchmark)',
                'url': 'https://zenodo.org/record/7155725/files/amos22.zip',
                'description': '500 CT + 100 MRI scans with 15 anatomical structures',
                'size': '~50GB',
                'structures': 15,
                'modalities': ['CT', 'MRI'],
                'required_for': ['Primary training', 'In-distribution validation', 'Novel class testing']
            },
            'bcv': {
                'name': 'Beyond the Cranial Vault (BCV)',
                'url': 'https://www.synapse.org/#!Synapse:syn3193805/wiki/217789',
                'description': '30 CT scans with 13 abdominal organs',
                'size': '~5GB', 
                'structures': 13,
                'modalities': ['CT'],
                'required_for': ['Cross-dataset generalization']
            },
            'lits': {
                'name': 'Liver Tumor Segmentation (LiTS)',
                'url': 'https://competitions.codalab.org/competitions/17094',
                'description': '131 CT scans with liver and tumor segmentation',
                'size': '~15GB',
                'structures': 2,
                'modalities': ['CT'],
                'required_for': ['Cross-dataset generalization', 'Liver-specific validation']
            },
            'kits19': {
                'name': 'Kidney Tumor Segmentation (KiTS19)',
                'url': 'https://github.com/neheller/kits19',
                'description': '210 CT scans with kidney and tumor segmentation',
                'size': '~25GB',
                'structures': 2,
                'modalities': ['CT'],
                'required_for': ['Cross-dataset generalization', 'Kidney-specific validation']
            },
            'msd_pancreas': {
                'name': 'Medical Segmentation Decathlon - Pancreas',
                'url': 'http://medicaldecathlon.com/',
                'description': 'Pancreas tumor segmentation for novel class testing',
                'size': '~2GB',
                'structures': 2,
                'modalities': ['CT'],
                'required_for': ['Novel class performance validation']
            },
            'pelvic1k': {
                'name': 'Pelvic1K Dataset',
                'url': 'https://github.com/MIRACLE-Center/Pelvic1K',
                'description': 'Pelvic bone segmentation for novel class testing',
                'size': '~10GB',
                'structures': 4,
                'modalities': ['CT'],
                'required_for': ['Novel class performance validation']
            }
        }
    
    def print_dataset_info(self):
        """Print information about all required datasets."""
        print("IRIS Framework - Required Medical Datasets")
        print("=" * 80)
        print("\nThe following datasets are required to properly validate the 6 IRIS paper claims:")
        print("(Previous validation used synthetic data and was invalid)\n")
        
        for dataset_id, info in self.datasets.items():
            print(f"📊 {info['name']}")
            print(f"   URL: {info['url']}")
            print(f"   Description: {info['description']}")
            print(f"   Size: {info['size']}")
            print(f"   Anatomical Structures: {info['structures']}")
            print(f"   Modalities: {', '.join(info['modalities'])}")
            print(f"   Required for: {', '.join(info['required_for'])}")
            print()
    
    def check_dataset_availability(self, dataset_id):
        """Check if dataset is already downloaded."""
        dataset_path = self.data_dir / dataset_id
        if dataset_path.exists() and any(dataset_path.iterdir()):
            return True
        return False
    
    def download_amos22(self):
        """Download AMOS22 dataset - Primary dataset for IRIS validation."""
        print("📥 Downloading AMOS22 Dataset...")
        print("⚠️  This is a large dataset (~50GB) and may take several hours")
        print("⚠️  You may need to register at https://amos22.grand-challenge.org/")
        
        dataset_dir = self.data_dir / 'amos22'
        dataset_dir.mkdir(exist_ok=True)
        
        # AMOS22 requires registration - provide instructions
        print("\n🔐 AMOS22 Dataset Access Instructions:")
        print("1. Visit: https://amos22.grand-challenge.org/")
        print("2. Register for an account")
        print("3. Request dataset access")
        print("4. Download the dataset files to:", dataset_dir)
        print("5. Expected structure:")
        print("   amos22/")
        print("   ├── imagesTr/  (500 CT + 100 MRI training images)")
        print("   ├── labelsTr/  (corresponding segmentation masks)")
        print("   ├── imagesTs/  (test images)")
        print("   └── dataset.json (metadata)")
        
        return False  # Manual download required
    
    def download_public_datasets(self):
        """Download publicly available datasets."""
        print("📥 Downloading publicly available datasets...")
        
        # Create instructions for datasets that require manual download
        instructions = {
            'bcv': "Visit Synapse and download BCV dataset",
            'lits': "Register at CodaLab competition and download LiTS",
            'kits19': "Clone GitHub repository: git clone https://github.com/neheller/kits19",
            'msd_pancreas': "Download from Medical Segmentation Decathlon",
            'pelvic1k': "Clone GitHub repository: git clone https://github.com/MIRACLE-Center/Pelvic1K"
        }
        
        print("\n📋 Manual Download Instructions:")
        for dataset_id, instruction in instructions.items():
            info = self.datasets[dataset_id]
            print(f"\n{info['name']}:")
            print(f"   {instruction}")
            print(f"   URL: {info['url']}")
            print(f"   Save to: {self.data_dir / dataset_id}")
    
    def validate_dataset_structure(self, dataset_id):
        """Validate that downloaded dataset has correct structure."""
        dataset_path = self.data_dir / dataset_id
        
        if not dataset_path.exists():
            return False, f"Dataset directory {dataset_path} does not exist"
        
        # Basic validation - check for common medical imaging file formats
        valid_extensions = {'.nii', '.nii.gz', '.dcm', '.mha', '.mhd'}
        
        files = list(dataset_path.rglob('*'))
        medical_files = [f for f in files if f.suffix.lower() in valid_extensions or 
                        f.name.lower().endswith('.nii.gz')]
        
        if len(medical_files) == 0:
            return False, f"No medical imaging files found in {dataset_path}"
        
        return True, f"Found {len(medical_files)} medical imaging files"
    
    def create_dataset_summary(self):
        """Create a summary of available datasets."""
        summary_path = self.data_dir / 'dataset_summary.md'
        
        with open(summary_path, 'w') as f:
            f.write("# Medical Datasets for IRIS Framework Validation\n\n")
            f.write("This directory contains the real medical imaging datasets required to validate the IRIS framework claims.\n\n")
            f.write("**IMPORTANT**: Previous validation used synthetic data and was invalid. All claims must be re-tested with real data.\n\n")
            
            f.write("## Dataset Status\n\n")
            
            for dataset_id, info in self.datasets.items():
                available = self.check_dataset_availability(dataset_id)
                status = "✅ Available" if available else "❌ Not Downloaded"
                
                f.write(f"### {info['name']}\n")
                f.write(f"- **Status**: {status}\n")
                f.write(f"- **Description**: {info['description']}\n")
                f.write(f"- **Required for**: {', '.join(info['required_for'])}\n")
                f.write(f"- **URL**: {info['url']}\n")
                
                if available:
                    valid, msg = self.validate_dataset_structure(dataset_id)
                    f.write(f"- **Validation**: {msg}\n")
                
                f.write("\n")
            
            f.write("## IRIS Claims Requiring Real Data Validation\n\n")
            f.write("1. **Novel Class Performance**: Test on held-out anatomical structures from AMOS22\n")
            f.write("2. **Cross-Dataset Generalization**: Train on AMOS22, test on BCV/LiTS/KiTS19\n")
            f.write("3. **In-Distribution Performance**: Train and test on AMOS22 (89.56% Dice target)\n")
            f.write("4. **In-Context Learning**: Demonstrate frozen parameters during inference\n")
            f.write("5. **Multi-Class Efficiency**: Single forward pass on multi-organ AMOS22 images\n")
            f.write("6. **Task Embedding Reusability**: Cross-patient embedding reuse\n\n")
            
            f.write("## Next Steps\n\n")
            f.write("1. Download all required datasets\n")
            f.write("2. Fix decoder channel mismatch in Phase 2\n")
            f.write("3. Implement real data loading pipelines\n")
            f.write("4. Train IRIS model on real AMOS22 data\n")
            f.write("5. Validate all 6 claims systematically\n")
        
        print(f"📄 Dataset summary created: {summary_path}")

def main():
    """Main function to run dataset download process."""
    print("IRIS Framework - Medical Dataset Downloader")
    print("=" * 80)
    print("This script helps download real medical datasets to replace synthetic data")
    print("used in previous (invalid) validation attempts.\n")
    
    downloader = MedicalDatasetDownloader()
    
    # Print dataset information
    downloader.print_dataset_info()
    
    # Check current status
    print("📊 Checking current dataset availability...")
    available_count = 0
    for dataset_id in downloader.datasets.keys():
        if downloader.check_dataset_availability(dataset_id):
            available_count += 1
            print(f"✅ {dataset_id}: Available")
        else:
            print(f"❌ {dataset_id}: Not available")
    
    print(f"\n📈 Status: {available_count}/{len(downloader.datasets)} datasets available")
    
    if available_count == 0:
        print("\n⚠️  No datasets are currently available.")
        print("📥 Starting download process...")
        
        # Download AMOS22 (requires manual registration)
        downloader.download_amos22()
        
        # Provide instructions for other datasets
        downloader.download_public_datasets()
    
    # Create dataset summary
    downloader.create_dataset_summary()
    
    print("\n✅ Dataset download process completed.")
    print("📋 See dataset_summary.md for detailed status and next steps.")
    print("\n🔧 Remember: Fix decoder channel mismatch before training on real data!")

if __name__ == "__main__":
    main()
