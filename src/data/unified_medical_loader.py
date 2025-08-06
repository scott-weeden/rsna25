"""
Unified Medical Dataset Loader

This module provides a unified interface for loading multiple medical datasets:
- AMOS22: Abdominal Multi-Organ Segmentation
- BCV: Beyond the Cranial Vault (Synapse)
- CHAOS: Combined Healthy Abdominal Organ Segmentation
- KiTS19: Kidney Tumor Segmentation

All datasets are normalized to a common format for training.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import nibabel as nib
import glob
from typing import Dict, List, Tuple, Optional
import json


class UnifiedMedicalDataset(Dataset):
    """
    Unified dataset loader for multiple medical segmentation datasets.
    """
    
    def __init__(
        self,
        data_dir: str,
        dataset_type: str,
        split: str = 'train',
        target_size: Tuple[int, int, int] = (128, 128, 128),
        transform: Optional[callable] = None,
        cache_data: bool = False
    ):
        super().__init__()
        
        self.data_dir = data_dir
        self.dataset_type = dataset_type.lower()
        self.split = split
        self.target_size = target_size
        self.transform = transform
        self.cache_data = cache_data
        
        # Dataset-specific configurations
        self.dataset_configs = {
            'amos': {
                'organ_labels': {
                    1: 'spleen', 2: 'right_kidney', 3: 'left_kidney', 4: 'gallbladder',
                    5: 'esophagus', 6: 'liver', 7: 'stomach', 8: 'aorta',
                    9: 'inferior_vena_cava', 10: 'pancreas', 11: 'right_adrenal_gland',
                    12: 'left_adrenal_gland', 13: 'duodenum', 14: 'bladder', 15: 'prostate_uterus'
                },
                'num_classes': 15,
                'modality': 'CT'
            },
            'bcv': {
                'organ_labels': {
                    1: 'spleen', 2: 'right_kidney', 3: 'left_kidney', 4: 'gallbladder',
                    5: 'esophagus', 6: 'liver', 7: 'stomach', 8: 'aorta',
                    9: 'inferior_vena_cava', 10: 'pancreas', 11: 'right_adrenal_gland',
                    12: 'left_adrenal_gland', 13: 'duodenum'
                },
                'num_classes': 13,
                'modality': 'CT'
            },
            'chaos': {
                'organ_labels': {
                    1: 'liver', 2: 'right_kidney', 3: 'left_kidney', 4: 'spleen'
                },
                'num_classes': 4,
                'modality': 'CT_MR'
            },
            'kits19': {
                'organ_labels': {
                    1: 'kidney', 2: 'tumor'
                },
                'num_classes': 2,
                'modality': 'CT'
            }
        }
        
        if self.dataset_type not in self.dataset_configs:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        
        self.config = self.dataset_configs[self.dataset_type]
        self.organ_labels = self.config['organ_labels']
        
        # Load samples
        self.samples = self._load_samples()
        
        # Cache for loaded data
        self.cache = {} if cache_data else None
        
        print(f"Loaded {self.dataset_type.upper()} {split} dataset with {len(self.samples)} samples")
    
    def _load_samples(self) -> List[Dict[str, str]]:
        """Load list of available samples based on dataset type."""
        if self.dataset_type == 'amos':
            return self._load_amos_samples()
        elif self.dataset_type == 'bcv':
            return self._load_bcv_samples()
        elif self.dataset_type == 'chaos':
            return self._load_chaos_samples()
        elif self.dataset_type == 'kits19':
            return self._load_kits19_samples()
        else:
            return []
    
    def _load_amos_samples(self) -> List[Dict[str, str]]:
        """Load AMOS22 samples."""
        samples = []
        
        if self.split == 'train':
            image_dir = os.path.join(self.data_dir, 'imagesTr')
            label_dir = os.path.join(self.data_dir, 'labelsTr')
        elif self.split == 'val':
            image_dir = os.path.join(self.data_dir, 'imagesVa')
            label_dir = os.path.join(self.data_dir, 'labelsVa')
        else:  # test
            image_dir = os.path.join(self.data_dir, 'imagesTs')
            label_dir = os.path.join(self.data_dir, 'labelsTs')
        
        if not os.path.exists(image_dir):
            return samples
        
        image_files = sorted(glob.glob(os.path.join(image_dir, '*.nii.gz')))
        
        for img_path in image_files:
            img_name = os.path.basename(img_path)
            sample_id = img_name.replace('.nii.gz', '')
            label_path = os.path.join(label_dir, img_name)
            
            samples.append({
                'id': sample_id,
                'image': img_path,
                'label': label_path if os.path.exists(label_path) else None,
                'dataset': 'amos'
            })
        
        return samples
    
    def _load_bcv_samples(self) -> List[Dict[str, str]]:
        """Load BCV (Synapse) samples."""
        samples = []
        
        if self.split == 'train':
            image_dir = os.path.join(self.data_dir, 'averaged-training-images')
            label_dir = os.path.join(self.data_dir, 'averaged-training-labels')
        else:
            image_dir = os.path.join(self.data_dir, 'averaged-testing-images')
            label_dir = None  # No labels for test set
        
        if not os.path.exists(image_dir):
            return samples
        
        image_files = sorted(glob.glob(os.path.join(image_dir, '*.nii.gz')))
        
        for img_path in image_files:
            img_name = os.path.basename(img_path)
            sample_id = img_name.replace('.nii.gz', '')
            
            label_path = None
            if label_dir and os.path.exists(label_dir):
                label_path = os.path.join(label_dir, img_name)
                if not os.path.exists(label_path):
                    label_path = None
            
            samples.append({
                'id': sample_id,
                'image': img_path,
                'label': label_path,
                'dataset': 'bcv'
            })
        
        return samples
    
    def _load_chaos_samples(self) -> List[Dict[str, str]]:
        """Load CHAOS samples."""
        samples = []
        
        # CHAOS has a complex structure, let's check what's available
        if self.split == 'train':
            # Look for training data
            train_dirs = glob.glob(os.path.join(self.data_dir, '*Train*'))
            for train_dir in train_dirs:
                if os.path.isdir(train_dir):
                    # Find CT and MR data
                    ct_dirs = glob.glob(os.path.join(train_dir, '*/CT'))
                    mr_dirs = glob.glob(os.path.join(train_dir, '*/MR'))
                    
                    # Process CT data
                    for ct_dir in ct_dirs:
                        dicom_dir = os.path.join(ct_dir, 'DICOM_anon')
                        ground_dir = os.path.join(ct_dir, 'Ground')
                        
                        if os.path.exists(dicom_dir) and os.path.exists(ground_dir):
                            # For now, skip DICOM processing - would need more complex handling
                            pass
        
        # For now, return empty list - CHAOS needs special DICOM handling
        return samples
    
    def _load_kits19_samples(self) -> List[Dict[str, str]]:
        """Load KiTS19 samples."""
        samples = []
        
        data_dir = os.path.join(self.data_dir, 'data')
        if not os.path.exists(data_dir):
            return samples
        
        # KiTS19 has case_XXXXX directories
        case_dirs = sorted(glob.glob(os.path.join(data_dir, 'case_*')))
        
        for case_dir in case_dirs:
            case_id = os.path.basename(case_dir)
            
            # Look for imaging.nii.gz and segmentation.nii.gz
            image_path = os.path.join(case_dir, 'imaging.nii.gz')
            label_path = os.path.join(case_dir, 'segmentation.nii.gz')
            
            if os.path.exists(image_path):
                samples.append({
                    'id': case_id,
                    'image': image_path,
                    'label': label_path if os.path.exists(label_path) else None,
                    'dataset': 'kits19'
                })
        
        return samples
    
    def load_nifti(self, path: str) -> np.ndarray:
        """Load NIfTI file and return as numpy array."""
        nii = nib.load(path)
        data = nii.get_fdata()
        return data.astype(np.float32)
    
    def preprocess_volume(
        self,
        image: np.ndarray,
        label: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Preprocess medical volume for training.
        """
        # Dataset-specific preprocessing
        if self.dataset_type in ['amos', 'bcv', 'kits19']:
            # CT preprocessing
            image = np.clip(image, -175, 250)  # CT window
            image = (image - (-175)) / (250 - (-175))  # Normalize to [0, 1]
        elif self.dataset_type == 'chaos':
            # Mixed CT/MR preprocessing
            # Simple min-max normalization
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Convert to tensor and resize
        image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)  # Add batch and channel
        image = F.interpolate(image, size=self.target_size, mode='trilinear', align_corners=False)
        image = image.squeeze(0)  # Remove batch dimension, keep channel -> (1, D, H, W)
        
        # Process label if provided
        if label is not None:
            label = torch.from_numpy(label).float().unsqueeze(0).unsqueeze(0)  # Add batch and channel
            label = F.interpolate(label, size=self.target_size, mode='nearest')
            label = label.squeeze(0).squeeze(0).long()  # Remove batch and channel -> (D, H, W)
        
        return image, label
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        """
        # Check cache
        if self.cache_data and idx in self.cache:
            return self.cache[idx]
        
        sample = self.samples[idx]
        
        # Load image
        try:
            image = self.load_nifti(sample['image'])
        except Exception as e:
            print(f"Failed to load image {sample['image']}: {e}")
            # Return dummy data
            image = np.zeros((64, 64, 64), dtype=np.float32)
        
        # Load label if available
        label = None
        if sample['label'] is not None and os.path.exists(sample['label']):
            try:
                label = self.load_nifti(sample['label'])
            except Exception as e:
                print(f"Failed to load label {sample['label']}: {e}")
        
        # Preprocess
        image, label = self.preprocess_volume(image, label)
        
        # Create organ-specific binary masks
        organ_masks = {}
        if label is not None:
            for organ_id, organ_name in self.organ_labels.items():
                organ_mask = (label == organ_id).float()
                organ_masks[organ_name] = organ_mask
        
        result = {
            'image': image,
            'label': label,
            'id': sample['id'],
            'dataset': sample['dataset'],
            'organ_masks': organ_masks
        }
        
        # Apply additional transforms if provided
        if self.transform:
            result = self.transform(result)
        
        # Cache if enabled
        if self.cache_data:
            self.cache[idx] = result
        
        return result


def create_dataset(dataset_type: str, data_dir: str, split: str = 'train', **kwargs):
    """
    Factory function to create dataset instances.
    """
    return UnifiedMedicalDataset(
        data_dir=data_dir,
        dataset_type=dataset_type,
        split=split,
        **kwargs
    )


def test_unified_loader():
    """Test function to verify the unified loader works."""
    print("🧪 Testing Unified Medical Dataset Loader...")
    
    # Test configurations
    test_configs = [
        {'dataset_type': 'amos', 'data_dir': 'src/data/amos'},
        {'dataset_type': 'bcv', 'data_dir': 'src/data/bcv'},
        {'dataset_type': 'kits19', 'data_dir': 'src/data/kits19'},
        # {'dataset_type': 'chaos', 'data_dir': 'src/data/chaos'},  # Skip CHAOS for now
    ]
    
    for config in test_configs:
        print(f"\n📂 Testing {config['dataset_type'].upper()} dataset...")
        
        try:
            dataset = create_dataset(
                dataset_type=config['dataset_type'],
                data_dir=config['data_dir'],
                split='train',
                target_size=(64, 64, 64)  # Smaller for testing
            )
            
            print(f"   ✅ Dataset loaded with {len(dataset)} samples")
            
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"   ✅ Sample keys: {list(sample.keys())}")
                print(f"   ✅ Image shape: {sample['image'].shape}")
                if sample['label'] is not None:
                    print(f"   ✅ Label shape: {sample['label'].shape}")
                    print(f"   ✅ Unique labels: {torch.unique(sample['label']).tolist()}")
                print(f"   ✅ Organ masks: {len(sample['organ_masks'])}")
            
        except Exception as e:
            print(f"   ❌ Failed to load {config['dataset_type']}: {e}")
    
    print("\n✅ Unified loader testing completed!")


if __name__ == "__main__":
    test_unified_loader()
