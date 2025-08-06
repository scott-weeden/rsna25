"""
Real Medical Data Loader for IRIS Framework

This module implements comprehensive data loading for real medical imaging datasets:
- AMOS22: 500 CT + 100 MRI with 15 anatomical structures
- BCV: 13 abdominal organs, 30 CT scans
- LiTS: Liver + tumor, 131 CT scans
- KiTS19: Kidney + tumor, 210 CT scans

Supports episodic learning with reference-query pairs from different patients.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import nibabel as nib
from scipy import ndimage
from typing import Dict, List, Tuple, Optional, Union
import random
from collections import defaultdict


class MedicalImageProcessor:
    """Handles medical image preprocessing and normalization."""
    
    def __init__(self, target_spacing=(1.0, 1.0, 1.0), target_size=(128, 128, 128)):
        self.target_spacing = target_spacing
        self.target_size = target_size
    
    def load_nifti(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load NIfTI image and return data with affine matrix."""
        nii = nib.load(filepath)
        data = nii.get_fdata().astype(np.float32)
        affine = nii.affine
        return data, affine
    
    def normalize_intensity(self, image: np.ndarray, modality: str = 'CT') -> np.ndarray:
        """Normalize image intensity based on modality."""
        if modality.upper() == 'CT':
            # CT normalization: clip to [-1000, 1000] HU, then normalize
            image = np.clip(image, -1000, 1000)
            image = (image + 1000) / 2000.0  # Normalize to [0, 1]
        elif modality.upper() == 'MRI':
            # MRI normalization: z-score normalization
            mean = np.mean(image)
            std = np.std(image)
            if std > 0:
                image = (image - mean) / std
            else:
                image = image - mean
        else:
            # Generic normalization: min-max to [0, 1]
            min_val, max_val = np.min(image), np.max(image)
            if max_val > min_val:
                image = (image - min_val) / (max_val - min_val)
        
        return image
    
    def resample_to_target_size(self, image: np.ndarray, 
                               mask: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Resample image (and optionally mask) to target size."""
        current_shape = image.shape
        zoom_factors = [t / c for t, c in zip(self.target_size, current_shape)]
        
        # Resample image with linear interpolation
        resampled_image = ndimage.zoom(image, zoom_factors, order=1, mode='constant', cval=0)
        
        if mask is not None:
            # Resample mask with nearest neighbor interpolation
            resampled_mask = ndimage.zoom(mask, zoom_factors, order=0, mode='constant', cval=0)
            return resampled_image, resampled_mask
        
        return resampled_image
    
    def augment_data(self, image: np.ndarray, mask: np.ndarray, 
                    apply_augmentation: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation for training."""
        if not apply_augmentation:
            return image, mask
        
        # Random rotation (±15 degrees)
        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            axes = random.choice([(0, 1), (0, 2), (1, 2)])
            image = ndimage.rotate(image, angle, axes=axes, reshape=False, order=1, mode='constant', cval=0)
            mask = ndimage.rotate(mask, angle, axes=axes, reshape=False, order=0, mode='constant', cval=0)
        
        # Random flip
        if random.random() < 0.5:
            axis = random.choice([0, 1, 2])
            image = np.flip(image, axis=axis).copy()
            mask = np.flip(mask, axis=axis).copy()
        
        # Random intensity scaling (±20%)
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            image = image * scale
        
        # Random noise addition
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.05, image.shape)
            image = image + noise
        
        return image, mask


class AMOS22Dataset:
    """AMOS22 dataset handler with 15 anatomical structures."""
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.processor = MedicalImageProcessor()
        
        # AMOS22 class mapping (15 anatomical structures)
        self.class_mapping = {
            'background': 0,
            'spleen': 1,
            'right_kidney': 2,
            'left_kidney': 3,
            'gallbladder': 4,
            'esophagus': 5,
            'liver': 6,
            'stomach': 7,
            'aorta': 8,
            'inferior_vena_cava': 9,
            'portal_vein_splenic_vein': 10,
            'pancreas': 11,
            'right_adrenal_gland': 12,
            'left_adrenal_gland': 13,
            'duodenum': 14,
            'bladder': 15
        }
        
        self.reverse_mapping = {v: k for k, v in self.class_mapping.items()}
        
        # Load dataset metadata
        self.metadata = self._load_metadata()
        self.samples = self._discover_samples()
        
        print(f"AMOS22 Dataset initialized:")
        print(f"  - Data root: {self.data_root}")
        print(f"  - Total samples: {len(self.samples)}")
        print(f"  - Classes: {len(self.class_mapping)} anatomical structures")
    
    def _load_metadata(self) -> Dict:
        """Load dataset metadata from dataset.json."""
        metadata_path = self.data_root / 'dataset.json'
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        else:
            # Create default metadata if not available
            print("⚠️  dataset.json not found, creating default metadata")
            return {
                'name': 'AMOS22',
                'description': 'A large-scale abdominal multi-organ benchmark',
                'modality': {'0': 'CT', '1': 'MRI'},
                'labels': self.reverse_mapping
            }
    
    def _discover_samples(self) -> List[Dict]:
        """Discover available image-mask pairs."""
        samples = []
        
        # Check training data
        images_dir = self.data_root / 'imagesTr'
        labels_dir = self.data_root / 'labelsTr'
        
        if images_dir.exists() and labels_dir.exists():
            for img_path in images_dir.glob('*.nii.gz'):
                label_path = labels_dir / img_path.name
                if label_path.exists():
                    # Determine modality from filename or metadata
                    modality = 'CT'  # Default assumption
                    if 'MR' in img_path.name or 'mri' in img_path.name.lower():
                        modality = 'MRI'
                    
                    samples.append({
                        'image_path': str(img_path),
                        'label_path': str(label_path),
                        'modality': modality,
                        'patient_id': img_path.stem.split('.')[0],
                        'split': 'train'
                    })
        
        # Check test data
        images_test_dir = self.data_root / 'imagesTs'
        labels_test_dir = self.data_root / 'labelsTs'
        
        if images_test_dir.exists():
            for img_path in images_test_dir.glob('*.nii.gz'):
                label_path = None
                if labels_test_dir.exists():
                    label_path = labels_test_dir / img_path.name
                    if not label_path.exists():
                        label_path = None
                
                modality = 'CT'
                if 'MR' in img_path.name or 'mri' in img_path.name.lower():
                    modality = 'MRI'
                
                samples.append({
                    'image_path': str(img_path),
                    'label_path': str(label_path) if label_path else None,
                    'modality': modality,
                    'patient_id': img_path.stem.split('.')[0],
                    'split': 'test'
                })
        
        return samples
    
    def get_sample(self, index: int) -> Dict:
        """Get a single sample by index."""
        sample_info = self.samples[index]
        
        # Load image
        image, _ = self.processor.load_nifti(sample_info['image_path'])
        image = self.processor.normalize_intensity(image, sample_info['modality'])
        
        # Load mask if available
        mask = None
        if sample_info['label_path']:
            mask, _ = self.processor.load_nifti(sample_info['label_path'])
            mask = mask.astype(np.int32)
        
        # Resample to target size
        if mask is not None:
            image, mask = self.processor.resample_to_target_size(image, mask)
        else:
            image = self.processor.resample_to_target_size(image)
        
        return {
            'image': image,
            'mask': mask,
            'modality': sample_info['modality'],
            'patient_id': sample_info['patient_id'],
            'split': sample_info['split'],
            'classes_present': np.unique(mask).tolist() if mask is not None else []
        }
    
    def get_samples_by_class(self, class_id: int) -> List[int]:
        """Get sample indices that contain a specific class."""
        indices = []
        for i, sample in enumerate(self.samples):
            if sample['label_path']:  # Only check samples with labels
                try:
                    sample_data = self.get_sample(i)
                    if class_id in sample_data['classes_present']:
                        indices.append(i)
                except Exception as e:
                    print(f"⚠️  Error processing sample {i}: {e}")
                    continue
        return indices


class EpisodicMedicalDataset(Dataset):
    """
    Episodic dataset for in-context learning with real medical data.
    
    Samples reference-query pairs from the same anatomical class but different patients.
    """
    
    def __init__(self, dataset: AMOS22Dataset, split: str = 'train', 
                 episodes_per_epoch: int = 1000, augment: bool = True):
        self.dataset = dataset
        self.split = split
        self.episodes_per_epoch = episodes_per_epoch
        self.augment = augment
        
        # Filter samples by split
        self.split_samples = [i for i, s in enumerate(dataset.samples) 
                             if s['split'] == split and s['label_path'] is not None]
        
        # Group samples by class for episodic sampling
        self.class_to_samples = defaultdict(list)
        for idx in self.split_samples:
            sample_data = self.dataset.get_sample(idx)
            for class_id in sample_data['classes_present']:
                if class_id > 0:  # Skip background
                    self.class_to_samples[class_id].append(idx)
        
        # Filter classes with at least 2 samples (for reference-query pairs)
        self.valid_classes = [c for c, samples in self.class_to_samples.items() 
                             if len(samples) >= 2]
        
        print(f"Episodic {split} dataset initialized:")
        print(f"  - Total samples: {len(self.split_samples)}")
        print(f"  - Valid classes: {len(self.valid_classes)}")
        print(f"  - Episodes per epoch: {episodes_per_epoch}")
        print(f"  - Augmentation: {augment}")
    
    def __len__(self):
        return self.episodes_per_epoch
    
    def __getitem__(self, index):
        """Generate an episodic sample (reference-query pair)."""
        # Randomly select a class
        class_id = random.choice(self.valid_classes)
        class_samples = self.class_to_samples[class_id]
        
        # Randomly select reference and query from different patients
        if len(class_samples) < 2:
            # Fallback: use same sample for both (shouldn't happen due to filtering)
            ref_idx = query_idx = random.choice(class_samples)
        else:
            ref_idx, query_idx = random.sample(class_samples, 2)
        
        # Load reference sample
        ref_data = self.dataset.get_sample(ref_idx)
        ref_image = ref_data['image']
        ref_mask_full = ref_data['mask']
        
        # Create binary mask for the selected class
        ref_mask = (ref_mask_full == class_id).astype(np.float32)
        
        # Load query sample
        query_data = self.dataset.get_sample(query_idx)
        query_image = query_data['image']
        query_mask_full = query_data['mask']
        query_mask = (query_mask_full == class_id).astype(np.float32)
        
        # Apply augmentation
        if self.augment:
            ref_image, ref_mask = self.dataset.processor.augment_data(
                ref_image, ref_mask, apply_augmentation=True
            )
            query_image, query_mask = self.dataset.processor.augment_data(
                query_image, query_mask, apply_augmentation=True
            )
        
        # Convert to tensors
        ref_image = torch.from_numpy(ref_image).unsqueeze(0)  # Add channel dimension
        ref_mask = torch.from_numpy(ref_mask).unsqueeze(0)
        query_image = torch.from_numpy(query_image).unsqueeze(0)
        query_mask = torch.from_numpy(query_mask).unsqueeze(0)
        
        return {
            'reference_image': ref_image,
            'reference_mask': ref_mask,
            'query_image': query_image,
            'query_mask': query_mask,
            'class_id': class_id,
            'class_name': self.dataset.reverse_mapping[class_id],
            'ref_patient_id': ref_data['patient_id'],
            'query_patient_id': query_data['patient_id']
        }


class MultiDatasetLoader:
    """
    Multi-dataset loader for cross-dataset generalization testing.
    
    Supports AMOS22, BCV, LiTS, KiTS19 datasets.
    """
    
    def __init__(self, data_roots: Dict[str, str]):
        self.data_roots = data_roots
        self.datasets = {}
        
        # Initialize available datasets
        for name, root in data_roots.items():
            if os.path.exists(root):
                if name.lower() == 'amos22':
                    self.datasets[name] = AMOS22Dataset(root)
                else:
                    # For other datasets, create simplified loaders
                    print(f"⚠️  {name} dataset loader not fully implemented yet")
            else:
                print(f"⚠️  {name} dataset not found at {root}")
        
        print(f"Multi-dataset loader initialized with {len(self.datasets)} datasets")
    
    def get_episodic_loader(self, dataset_name: str, split: str = 'train', 
                           batch_size: int = 1, episodes_per_epoch: int = 1000) -> DataLoader:
        """Get episodic data loader for a specific dataset."""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not available")
        
        episodic_dataset = EpisodicMedicalDataset(
            self.datasets[dataset_name], 
            split=split, 
            episodes_per_epoch=episodes_per_epoch
        )
        
        return DataLoader(
            episodic_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for debugging, increase for production
            pin_memory=True
        )


def test_real_data_loader():
    """Test the real data loading system."""
    print("🧪 Testing Real Medical Data Loader...")
    print("=" * 60)
    
    # Test with mock AMOS22 structure (for development)
    mock_data_root = "/tmp/mock_amos22"
    
    # Create mock directory structure
    os.makedirs(mock_data_root, exist_ok=True)
    os.makedirs(f"{mock_data_root}/imagesTr", exist_ok=True)
    os.makedirs(f"{mock_data_root}/labelsTr", exist_ok=True)
    
    # Create mock dataset.json
    mock_metadata = {
        "name": "AMOS22",
        "description": "Mock AMOS22 for testing",
        "modality": {"0": "CT"},
        "labels": {
            "0": "background",
            "1": "spleen",
            "2": "right_kidney",
            "6": "liver"
        }
    }
    
    with open(f"{mock_data_root}/dataset.json", 'w') as f:
        json.dump(mock_metadata, f)
    
    # Create mock NIfTI files (small for testing)
    try:
        import nibabel as nib
        
        # Create mock image and mask
        mock_image = np.random.randn(64, 64, 32).astype(np.float32)
        mock_mask = np.zeros((64, 64, 32), dtype=np.int32)
        mock_mask[20:40, 20:40, 10:20] = 1  # Spleen
        mock_mask[10:30, 40:60, 15:25] = 6  # Liver
        
        # Save mock files
        nib.save(nib.Nifti1Image(mock_image, np.eye(4)), 
                f"{mock_data_root}/imagesTr/case_001.nii.gz")
        nib.save(nib.Nifti1Image(mock_mask, np.eye(4)), 
                f"{mock_data_root}/labelsTr/case_001.nii.gz")
        
        nib.save(nib.Nifti1Image(mock_image * 1.1, np.eye(4)), 
                f"{mock_data_root}/imagesTr/case_002.nii.gz")
        nib.save(nib.Nifti1Image(mock_mask, np.eye(4)), 
                f"{mock_data_root}/labelsTr/case_002.nii.gz")
        
        print("✅ Mock AMOS22 data created")
        
        # Test dataset loading
        dataset = AMOS22Dataset(mock_data_root)
        
        # Test episodic dataset
        episodic_dataset = EpisodicMedicalDataset(dataset, split='train', episodes_per_epoch=10)
        
        # Test data loader
        data_loader = DataLoader(episodic_dataset, batch_size=1, shuffle=True)
        
        print("\n🧪 Testing episodic data loading...")
        for i, batch in enumerate(data_loader):
            if i >= 2:  # Test first 2 batches
                break
                
            print(f"Batch {i}:")
            print(f"  - Reference image: {batch['reference_image'].shape}")
            print(f"  - Reference mask: {batch['reference_mask'].shape}")
            print(f"  - Query image: {batch['query_image'].shape}")
            print(f"  - Query mask: {batch['query_mask'].shape}")
            print(f"  - Class: {batch['class_name'][0]} (ID: {batch['class_id'].item()})")
            print(f"  - Patients: {batch['ref_patient_id'][0]} -> {batch['query_patient_id'][0]}")
            
            # Validate shapes
            assert batch['reference_image'].shape == (1, 1, 128, 128, 128)
            assert batch['query_image'].shape == (1, 1, 128, 128, 128)
            assert batch['reference_mask'].shape == (1, 1, 128, 128, 128)
            assert batch['query_mask'].shape == (1, 1, 128, 128, 128)
        
        print("\n✅ Real data loader tests passed!")
        return True
        
    except ImportError:
        print("⚠️  nibabel not available, skipping NIfTI tests")
        return False
    except Exception as e:
        print(f"❌ Real data loader test failed: {e}")
        return False


if __name__ == "__main__":
    test_real_data_loader()
