#!/usr/bin/env python3
"""
Multi-Dataset Training Script for IRIS Framework

This script trains the IRIS model on multiple medical datasets:
- AMOS22: Abdominal organ segmentation
- BCV: Beyond the Cranial Vault
- CHAOS: Combined Healthy Abdominal Organ Segmentation
- KiTS19: Kidney Tumor Segmentation

Uses 75%/5%/20% train/validation/test split across all datasets.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import json
import argparse
from tqdm import tqdm
from collections import defaultdict
import time
from datetime import datetime

# Try to import wandb, but make it optional
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Logging will be disabled.")

# Add src to path
sys.path.append('src')

from models.iris_model import IRISModel
from data.unified_medical_loader import create_dataset
from losses.dice_loss import DiceLoss, GeneralizedDiceLoss, CombinedSegmentationLoss, compute_dice_score


class MultiDatasetLoader:
    """
    Handles loading and splitting multiple medical datasets.
    """
    
    def __init__(self, data_dirs, target_size=(128, 128, 128), cache_data=False):
        self.data_dirs = data_dirs
        self.target_size = target_size
        self.cache_data = cache_data
        
        # Dataset configurations
        self.dataset_configs = {
            'amos': {
                'organ_classes': 15,
                'description': 'Abdominal Multi-Organ Segmentation'
            },
            'bcv': {
                'organ_classes': 13,
                'description': 'Beyond the Cranial Vault'
            },
            'chaos': {
                'organ_classes': 4,
                'description': 'Combined Healthy Abdominal Organ Segmentation'
            },
            'kits19': {
                'organ_classes': 2,
                'description': 'Kidney Tumor Segmentation'
            }
        }
        
        print("🏥 Multi-Dataset Loader Initialized")
        print(f"   Target size: {target_size}")
        print(f"   Cache data: {cache_data}")
    
    def load_datasets(self):
        """Load all available datasets."""
        datasets = {}
        total_samples = 0
        
        for dataset_name, data_dir in self.data_dirs.items():
            if not os.path.exists(data_dir):
                print(f"⚠️  Dataset {dataset_name} not found at {data_dir}, skipping...")
                continue
            
            try:
                config = self.dataset_configs[dataset_name]
                
                print(f"\n📂 Loading {dataset_name.upper()} dataset...")
                print(f"   Path: {data_dir}")
                print(f"   Description: {config['description']}")
                
                # Load training data using unified loader
                dataset = create_dataset(
                    dataset_type=dataset_name,
                    data_dir=data_dir,
                    split='train',  # Load all training data
                    target_size=self.target_size,
                    cache_data=self.cache_data
                )
                
                datasets[dataset_name] = {
                    'dataset': dataset,
                    'config': config,
                    'samples': len(dataset)
                }
                
                total_samples += len(dataset)
                print(f"   ✅ Loaded {len(dataset)} samples")
                
            except Exception as e:
                print(f"   ❌ Failed to load {dataset_name}: {e}")
                continue
        
        print(f"\n📊 Total samples across all datasets: {total_samples}")
        return datasets
    
    def create_splits(self, datasets, train_ratio=0.75, val_ratio=0.05, test_ratio=0.20):
        """Create train/validation/test splits across all datasets."""
        print(f"\n🔄 Creating data splits ({train_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%})...")
        
        train_datasets = []
        val_datasets = []
        test_datasets = []
        
        split_info = {}
        
        for dataset_name, dataset_info in datasets.items():
            dataset = dataset_info['dataset']
            total_samples = len(dataset)
            
            # Calculate split sizes
            train_size = int(total_samples * train_ratio)
            val_size = int(total_samples * val_ratio)
            test_size = total_samples - train_size - val_size
            
            # Create indices for splitting
            indices = np.random.permutation(total_samples)
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]
            
            # Create subset datasets
            train_subset = torch.utils.data.Subset(dataset, train_indices)
            val_subset = torch.utils.data.Subset(dataset, val_indices)
            test_subset = torch.utils.data.Subset(dataset, test_indices)
            
            train_datasets.append(train_subset)
            val_datasets.append(val_subset)
            test_datasets.append(test_subset)
            
            split_info[dataset_name] = {
                'total': total_samples,
                'train': len(train_subset),
                'val': len(val_subset),
                'test': len(test_subset)
            }
            
            print(f"   {dataset_name.upper()}: {len(train_subset)}/{len(val_subset)}/{len(test_subset)} (train/val/test)")
        
        # Combine all datasets
        combined_train = ConcatDataset(train_datasets) if train_datasets else None
        combined_val = ConcatDataset(val_datasets) if val_datasets else None
        combined_test = ConcatDataset(test_datasets) if test_datasets else None
        
        print(f"\n📈 Combined splits:")
        if combined_train:
            print(f"   Train: {len(combined_train)} samples")
        if combined_val:
            print(f"   Validation: {len(combined_val)} samples")
        if combined_test:
            print(f"   Test: {len(combined_test)} samples")
        
        return combined_train, combined_val, combined_test, split_info


class EpisodicSampler:
    """
    Episodic sampler for in-context learning.
    Each episode contains a support set and query set.
    """
    
    def __init__(self, dataset, n_way=1, k_shot=1, query_shots=1):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_shots = query_shots
    
    def sample_episode(self):
        """Sample one episode for training."""
        # For medical segmentation, we typically use 1-way (binary segmentation)
        # with k-shot support examples
        
        # Randomly sample indices
        total_samples = len(self.dataset)
        episode_indices = np.random.choice(total_samples, size=self.k_shot + self.query_shots, replace=False)
        
        support_indices = episode_indices[:self.k_shot]
        query_indices = episode_indices[self.k_shot:]
        
        # Get samples
        support_samples = [self.dataset[i] for i in support_indices]
        query_samples = [self.dataset[i] for i in query_indices]
        
        return support_samples, query_samples


class IRISTrainer:
    """
    Trainer for IRIS model on multiple medical datasets.
    """
    
    def __init__(self, model, train_dataset, val_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup loss function
        self.criterion = CombinedSegmentationLoss(
            dice_weight=config.get('dice_weight', 1.0),
            ce_weight=config.get('ce_weight', 0.5)
        )
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('max_epochs', 100),
            eta_min=config.get('min_lr', 1e-6)
        )
        
        # Setup episodic sampler
        self.episodic_sampler = EpisodicSampler(
            train_dataset,
            k_shot=config.get('k_shot', 1),
            query_shots=config.get('query_shots', 1)
        )
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.val_dice_scores = []
        
        print(f"🚀 IRIS Trainer initialized on {self.device}")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset) if val_dataset else 0}")
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        epoch_dice_scores = []
        
        # Calculate number of episodes per epoch
        episodes_per_epoch = self.config.get('episodes_per_epoch', 1000)
        
        pbar = tqdm(range(episodes_per_epoch), desc=f'Epoch {epoch+1}')
        
        for episode_idx in pbar:
            try:
                # Sample episode
                support_samples, query_samples = self.episodic_sampler.sample_episode()
                
                # Process episode
                episode_loss, episode_dice = self.process_episode(support_samples, query_samples)
                
                if episode_loss is not None:
                    epoch_losses.append(episode_loss)
                    epoch_dice_scores.append(episode_dice)
                
                # Update progress bar
                if len(epoch_losses) > 0:
                    pbar.set_postfix({
                        'Loss': f'{np.mean(epoch_losses):.4f}',
                        'DICE': f'{np.mean(epoch_dice_scores):.4f}',
                        'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                    })
                
            except Exception as e:
                print(f"Episode {episode_idx} failed: {e}")
                continue
        
        # Calculate epoch metrics
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        avg_dice = np.mean(epoch_dice_scores) if epoch_dice_scores else 0.0
        
        self.train_losses.append(avg_loss)
        
        return avg_loss, avg_dice
    
    def process_episode(self, support_samples, query_samples):
        """Process one training episode."""
        if not support_samples or not query_samples:
            return None, None
        
        # Get support and query data
        support_sample = support_samples[0]  # Use first support sample
        query_sample = query_samples[0]      # Use first query sample
        
        # Check if samples have labels
        if support_sample['label'] is None or query_sample['label'] is None:
            return None, None
        
        # Prepare tensors
        support_image = support_sample['image'].to(self.device)
        support_label = support_sample['label'].to(self.device)
        query_image = query_sample['image'].to(self.device)
        query_label = query_sample['label'].to(self.device)
        
        # Add batch dimensions if needed
        if support_image.dim() == 4:
            support_image = support_image.unsqueeze(0)
        if query_image.dim() == 4:
            query_image = query_image.unsqueeze(0)
        
        # Find a random organ in support sample
        unique_organs = torch.unique(support_label)
        unique_organs = unique_organs[unique_organs > 0]  # Remove background
        
        if len(unique_organs) == 0:
            return None, None
        
        # Select random organ
        target_organ = unique_organs[torch.randint(len(unique_organs), (1,))].item()
        
        # Create binary masks
        support_mask = (support_label == target_organ).float()
        query_mask = (query_label == target_organ).float()
        
        # Add channel and batch dimensions to masks
        support_mask = support_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        query_mask = query_mask.unsqueeze(0)  # (1, D, H, W)
        
        # Forward pass
        self.optimizer.zero_grad()
        
        predictions = self.model(query_image, support_image, support_mask)
        
        # Compute loss
        loss = self.criterion(predictions, query_mask.long())
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Compute DICE score
        with torch.no_grad():
            pred_binary = (torch.sigmoid(predictions) > 0.5).float()
            dice_score = compute_dice_score(pred_binary.squeeze(), query_mask.squeeze())
        
        return loss.item(), dice_score.item()
    
    def validate(self, epoch):
        """Validate the model."""
        if self.val_dataset is None:
            return 0.0, 0.0
        
        self.model.eval()
        val_losses = []
        val_dice_scores = []
        
        # Create validation episodic sampler
        val_sampler = EpisodicSampler(
            self.val_dataset,
            k_shot=self.config.get('k_shot', 1),
            query_shots=1
        )
        
        num_val_episodes = min(100, len(self.val_dataset) // 2)
        
        with torch.no_grad():
            for _ in tqdm(range(num_val_episodes), desc='Validation'):
                try:
                    support_samples, query_samples = val_sampler.sample_episode()
                    
                    if not support_samples or not query_samples:
                        continue
                    
                    support_sample = support_samples[0]
                    query_sample = query_samples[0]
                    
                    if support_sample['label'] is None or query_sample['label'] is None:
                        continue
                    
                    # Prepare tensors (same as training)
                    support_image = support_sample['image'].to(self.device)
                    support_label = support_sample['label'].to(self.device)
                    query_image = query_sample['image'].to(self.device)
                    query_label = query_sample['label'].to(self.device)
                    
                    if support_image.dim() == 4:
                        support_image = support_image.unsqueeze(0)
                    if query_image.dim() == 4:
                        query_image = query_image.unsqueeze(0)
                    
                    # Find organ
                    unique_organs = torch.unique(support_label)
                    unique_organs = unique_organs[unique_organs > 0]
                    
                    if len(unique_organs) == 0:
                        continue
                    
                    target_organ = unique_organs[torch.randint(len(unique_organs), (1,))].item()
                    
                    support_mask = (support_label == target_organ).float()
                    query_mask = (query_label == target_organ).float()
                    
                    support_mask = support_mask.unsqueeze(0).unsqueeze(0)
                    query_mask = query_mask.unsqueeze(0)
                    
                    # Forward pass
                    predictions = self.model(query_image, support_image, support_mask)
                    
                    # Compute metrics
                    loss = self.criterion(predictions, query_mask.long())
                    pred_binary = (torch.sigmoid(predictions) > 0.5).float()
                    dice_score = compute_dice_score(pred_binary.squeeze(), query_mask.squeeze())
                    
                    val_losses.append(loss.item())
                    val_dice_scores.append(dice_score.item())
                    
                except Exception as e:
                    continue
        
        avg_val_loss = np.mean(val_losses) if val_losses else 0.0
        avg_val_dice = np.mean(val_dice_scores) if val_dice_scores else 0.0
        
        self.val_losses.append(avg_val_loss)
        self.val_dice_scores.append(avg_val_dice)
        
        return avg_val_loss, avg_val_dice
    
    def save_checkpoint(self, epoch, save_path, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_dice_scores': self.val_dice_scores,
            'model_info': self.model.get_model_info()
        }
        
        torch.save(checkpoint, save_path)
        
        if is_best:
            best_path = save_path.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
        
        print(f"💾 Checkpoint saved: {save_path}")
    
    def train(self, num_epochs, save_dir='checkpoints', save_every=10):
        """Main training loop."""
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_dice = 0.0
        
        print(f"\n🏋️  Starting training for {num_epochs} epochs...")
        print(f"   Save directory: {save_dir}")
        print(f"   Save every: {save_every} epochs")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training
            train_loss, train_dice = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_dice = self.validate(epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Calculate epoch time
            epoch_time = time.time() - start_time
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train - Loss: {train_loss:.4f}, DICE: {train_dice:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, DICE: {val_dice:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Log to wandb if available
            if WANDB_AVAILABLE:
                try:
                    wandb.log({
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'train_dice': train_dice,
                        'val_loss': val_loss,
                        'val_dice': val_dice,
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        'epoch_time': epoch_time
                    })
                except:
                    pass  # wandb logging failed
            
            # Save checkpoint
            is_best = val_dice > best_val_dice
            if is_best:
                best_val_dice = val_dice
            
            if (epoch + 1) % save_every == 0 or is_best:
                checkpoint_path = os.path.join(save_dir, f'iris_multi_dataset_epoch_{epoch+1}.pth')
                self.save_checkpoint(epoch + 1, checkpoint_path, is_best)
        
        print(f"\n🎉 Training completed!")
        print(f"   Best validation DICE: {best_val_dice:.4f}")
        
        # Save final checkpoint
        final_path = os.path.join(save_dir, 'iris_multi_dataset_final.pth')
        self.save_checkpoint(num_epochs, final_path)
        
        return best_val_dice


def main():
    parser = argparse.ArgumentParser(description='Train IRIS on multiple medical datasets')
    
    # Dataset arguments
    parser.add_argument('--amos_dir', type=str, default='src/data/amos',
                        help='Path to AMOS dataset')
    parser.add_argument('--bcv_dir', type=str, default='src/data/bcv',
                        help='Path to BCV dataset')
    parser.add_argument('--chaos_dir', type=str, default='src/data/chaos',
                        help='Path to CHAOS dataset')
    parser.add_argument('--kits19_dir', type=str, default='src/data/kits19',
                        help='Path to KiTS19 dataset')
    
    # Model arguments
    parser.add_argument('--in_channels', type=int, default=1,
                        help='Number of input channels')
    parser.add_argument('--base_channels', type=int, default=32,
                        help='Base number of channels')
    parser.add_argument('--embed_dim', type=int, default=512,
                        help='Embedding dimension')
    parser.add_argument('--num_tokens', type=int, default=10,
                        help='Number of query tokens')
    parser.add_argument('--num_classes', type=int, default=1,
                        help='Number of output classes')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--episodes_per_epoch', type=int, default=1000,
                        help='Number of episodes per epoch')
    parser.add_argument('--k_shot', type=int, default=1,
                        help='Number of support examples')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--dice_weight', type=float, default=1.0,
                        help='Weight for DICE loss')
    parser.add_argument('--ce_weight', type=float, default=0.5,
                        help='Weight for cross-entropy loss')
    
    # Other arguments
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--target_size', type=int, nargs=3, default=[128, 128, 128],
                        help='Target image size (D H W)')
    parser.add_argument('--cache_data', action='store_true',
                        help='Cache data in memory')
    parser.add_argument('--wandb_project', type=str, default='iris-medical-segmentation',
                        help='Weights & Biases project name')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize wandb
    if WANDB_AVAILABLE:
        try:
            wandb.init(
                project=args.wandb_project,
                config=vars(args),
                name=f"iris_multi_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            print("📊 Weights & Biases logging enabled")
        except:
            print("⚠️  Weights & Biases initialization failed, logging disabled")
    else:
        print("⚠️  Weights & Biases not available, logging disabled")
    
    print("🏥 IRIS Multi-Dataset Training")
    print("=" * 60)
    
    # Setup data directories
    data_dirs = {
        'amos': args.amos_dir,
        'bcv': args.bcv_dir,
        'chaos': args.chaos_dir,
        'kits19': args.kits19_dir
    }
    
    # Load datasets
    loader = MultiDatasetLoader(
        data_dirs=data_dirs,
        target_size=tuple(args.target_size),
        cache_data=args.cache_data
    )
    
    datasets = loader.load_datasets()
    
    if not datasets:
        print("❌ No datasets loaded! Please check your data paths.")
        return
    
    # Create splits
    train_dataset, val_dataset, test_dataset, split_info = loader.create_splits(datasets)
    
    if train_dataset is None:
        print("❌ No training data available!")
        return
    
    # Create model
    print(f"\n🧠 Creating IRIS model...")
    model = IRISModel(
        in_channels=args.in_channels,
        base_channels=args.base_channels,
        embed_dim=args.embed_dim,
        num_tokens=args.num_tokens,
        num_classes=args.num_classes
    )
    
    # Training configuration
    config = {
        'in_channels': args.in_channels,
        'base_channels': args.base_channels,
        'embed_dim': args.embed_dim,
        'num_tokens': args.num_tokens,
        'num_classes': args.num_classes,
        'max_epochs': args.epochs,
        'episodes_per_epoch': args.episodes_per_epoch,
        'k_shot': args.k_shot,
        'query_shots': 1,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'dice_weight': args.dice_weight,
        'ce_weight': args.ce_weight,
        'min_lr': 1e-6,
        'target_size': args.target_size,
        'datasets': list(datasets.keys()),
        'split_info': split_info,
        'seed': args.seed
    }
    
    # Create trainer
    trainer = IRISTrainer(model, train_dataset, val_dataset, config)
    
    # Start training
    best_dice = trainer.train(
        num_epochs=args.epochs,
        save_dir=args.save_dir,
        save_every=args.save_every
    )
    
    print(f"\n🎯 Training Summary:")
    print(f"   Best validation DICE: {best_dice:.4f}")
    print(f"   Checkpoints saved in: {args.save_dir}")
    
    # Save training info
    info_path = os.path.join(args.save_dir, 'training_info.json')
    with open(info_path, 'w') as f:
        json.dump({
            'config': config,
            'best_val_dice': best_dice,
            'datasets_used': list(datasets.keys()),
            'total_train_samples': len(train_dataset),
            'total_val_samples': len(val_dataset) if val_dataset else 0,
            'total_test_samples': len(test_dataset) if test_dataset else 0
        }, f, indent=2)
    
    print(f"   Training info saved: {info_path}")


if __name__ == "__main__":
    main()
