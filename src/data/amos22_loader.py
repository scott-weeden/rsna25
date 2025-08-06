"""
AMOS22 Dataset Loader for IRIS Framework

This module implements a real medical data loader for AMOS22 dataset that:
- Loads NIfTI medical images using nibabel
- Handles 3D volumetric data properly
- Supports episodic training for in-context learning
- Provides proper data augmentation and preprocessing
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from typing import Dict, List, Tuple, Optional
import glob
import json


class AMOS22Dataset(Dataset):
    """
    AMOS22 Dataset for in-context medical image segmentation.
    
    Args:
        data_dir (str): Path to AMOS22 dataset directory
        split (str): 'train', 'val', or 'test'
        transform (callable): Optional transform to apply
        cache_data (bool): Whether to cache data in memory
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[callable] = None,
        cache_data: bool = False,
        target_size: Tuple[int, int, int] = (128, 128, 128)
    ):
        super().__init__()
        
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.cache_data = cache_data
        self.target_size = target_size
        
        # Define paths based on split
        if split == 'train':
            self.image_dir = os.path.join(data_dir, 'imagesTr')
            self.label_dir = os.path.join(data_dir, 'labelsTr')
        elif split == 'val':
            self.image_dir = os.path.join(data_dir, 'imagesVa')
            self.label_dir = os.path.join(data_dir, 'labelsVa')
        elif split == 'test':
            self.image_dir = os.path.join(data_dir, 'imagesTs')
            self.label_dir = os.path.join(data_dir, 'labelsTs')
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Get list of available samples
        self.samples = self._load_samples()
        
        # Cache for loaded data
        self.cache = {} if cache_data else None
        
        # Define anatomical structures in AMOS22
        self.organ_labels = {
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
        
        print(f"Loaded AMOS22 {split} dataset with {len(self.samples)} samples")
    
    def _load_samples(self) -> List[Dict[str, str]]:
        """Load list of available samples."""
        samples = []
        
        # Get all image files
        image_files = sorted(glob.glob(os.path.join(self.image_dir, '*.nii.gz')))
        
        for img_path in image_files:
            # Extract ID from filename
            img_name = os.path.basename(img_path)
            sample_id = img_name.replace('.nii.gz', '')
            
            # Check if corresponding label exists
            label_path = os.path.join(self.label_dir, img_name)
            
            if os.path.exists(label_path) or self.split == 'test':
                samples.append({
                    'id': sample_id,
                    'image': img_path,
                    'label': label_path if os.path.exists(label_path) else None
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
        
        Args:
            image: Input image volume
            label: Optional label volume
        
        Returns:
            Preprocessed image and label tensors
        """
        # Simple preprocessing without MONAI transforms
        # Normalize intensity
        image = np.clip(image, -175, 250)  # CT window
        image = (image - (-175)) / (250 - (-175))  # Normalize to [0, 1]
        
        # Resize to target size using torch interpolation
        image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        image = F.interpolate(image, size=self.target_size, mode='trilinear', align_corners=False)
        image = image.squeeze(0)  # Remove batch dimension, keep channel dimension -> (1, D, H, W)
        
        # Process label if provided
        if label is not None:
            label = torch.from_numpy(label).float().unsqueeze(0).unsqueeze(0)  # Add batch and channel
            label = F.interpolate(label, size=self.target_size, mode='nearest')
            label = label.squeeze(0).squeeze(0).long()  # Remove batch and channel, convert to long -> (D, H, W)
        
        return image, label
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            Dictionary containing:
            - 'image': Preprocessed image tensor (1, D, H, W)
            - 'label': Label tensor (1, D, H, W) or None
            - 'id': Sample ID
            - 'organ_mask': Binary masks for each organ (K, D, H, W)
        """
        # Check cache
        if self.cache_data and idx in self.cache:
            return self.cache[idx]
        
        sample = self.samples[idx]
        
        # Load image
        image = self.load_nifti(sample['image'])
        
        # Load label if available
        label = None
        if sample['label'] is not None:
            label = self.load_nifti(sample['label'])
        
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
            'organ_masks': organ_masks
        }
        
        # Apply additional transforms if provided
        if self.transform:
            result = self.transform(result)
        
        # Cache if enabled
        if self.cache_data:
            self.cache[idx] = result
        
        return result


class EpisodicAMOS22Dataset(Dataset):
    """
    Episodic dataset wrapper for in-context learning.
    
    Creates episodes with reference and query pairs for few-shot segmentation.
    """
    
    def __init__(
        self,
        base_dataset: AMOS22Dataset,
        episodes_per_epoch: int = 1000,
        num_classes_per_episode: int = 3,
        num_support: int = 1,
        num_query: int = 1
    ):
        self.base_dataset = base_dataset
        self.episodes_per_epoch = episodes_per_epoch
        self.num_classes_per_episode = num_classes_per_episode
        self.num_support = num_support
        self.num_query = num_query
        
        # Filter samples with labels
        self.labeled_samples = [
            i for i in range(len(base_dataset))
            if base_dataset.samples[i]['label'] is not None
        ]
    
    def __len__(self) -> int:
        return self.episodes_per_epoch
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Create an episode for in-context learning.
        
        Returns:
            Dictionary containing:
            - 'support_images': (num_support, 1, D, H, W)
            - 'support_masks': (num_support, 1, D, H, W)
            - 'query_images': (num_query, 1, D, H, W)
            - 'query_masks': (num_query, 1, D, H, W)
            - 'class_ids': Selected organ classes for this episode
        """
        # Randomly select organ classes for this episode
        available_organs = list(self.base_dataset.organ_labels.keys())
        selected_organs = np.random.choice(
            available_organs,
            size=min(self.num_classes_per_episode, len(available_organs)),
            replace=False
        )
        
        # Randomly select samples for support and query
        selected_indices = np.random.choice(
            self.labeled_samples,
            size=self.num_support + self.num_query,
            replace=False
        )
        
        support_indices = selected_indices[:self.num_support]
        query_indices = selected_indices[self.num_support:]
        
        # Load support samples
        support_images = []
        support_masks = []
        for idx in support_indices:
            sample = self.base_dataset[idx]
            support_images.append(sample['image'])
            
            # Create multi-class mask for selected organs
            mask = torch.zeros_like(sample['label'])
            for organ_id in selected_organs:
                organ_mask = (sample['label'] == organ_id)
                mask[organ_mask] = organ_id
            support_masks.append(mask)
        
        # Load query samples
        query_images = []
        query_masks = []
        for idx in query_indices:
            sample = self.base_dataset[idx]
            query_images.append(sample['image'])
            
            # Create multi-class mask for selected organs
            mask = torch.zeros_like(sample['label'])
            for organ_id in selected_organs:
                organ_mask = (sample['label'] == organ_id)
                mask[organ_mask] = organ_id
            query_masks.append(mask)
        
        return {
            'support_images': torch.stack(support_images),
            'support_masks': torch.stack(support_masks),
            'query_images': torch.stack(query_images),
            'query_masks': torch.stack(query_masks),
            'class_ids': torch.tensor(selected_organs)
        }


def create_amos22_dataloaders(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    episodic: bool = True,
    target_size: Tuple[int, int, int] = (128, 128, 128)
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for AMOS22 dataset.
    
    Args:
        data_dir: Path to AMOS22 dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        episodic: Whether to use episodic training
        target_size: Target volume size
    
    Returns:
        Train, validation, and test DataLoaders
    """
    # Create base datasets
    train_dataset = AMOS22Dataset(
        data_dir, split='train', target_size=target_size
    )
    val_dataset = AMOS22Dataset(
        data_dir, split='val', target_size=target_size
    )
    test_dataset = AMOS22Dataset(
        data_dir, split='test', target_size=target_size
    )
    
    # Wrap in episodic datasets if needed
    if episodic:
        train_dataset = EpisodicAMOS22Dataset(train_dataset)
        val_dataset = EpisodicAMOS22Dataset(val_dataset, episodes_per_epoch=100)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Test one at a time
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the data loader
    print("Testing AMOS22 data loader...")
    
    data_dir = "/Users/owner/Documents/lectures/segmentation_learning/github/src/data/amos"
    
    # Test basic dataset
    dataset = AMOS22Dataset(data_dir, split='train', target_size=(64, 64, 64))
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        if 'image' in sample:
            print(f"Image shape: {sample['image'].shape}")
        if 'label' in sample:
            print(f"Label shape: {sample['label'].shape if sample['label'] is not None else 'None'}")
    
    # Test episodic dataset
    if len(dataset) > 1:
        episodic_dataset = EpisodicAMOS22Dataset(dataset, episodes_per_epoch=10)
        episode = episodic_dataset[0]
        print(f"\nEpisode keys: {episode.keys()}")
        print(f"Support images shape: {episode['support_images'].shape}")
        print(f"Query images shape: {episode['query_images'].shape}")