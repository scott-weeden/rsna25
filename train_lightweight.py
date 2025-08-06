#!/usr/bin/env python3
"""
Lightweight Training Script for IRIS Framework

This script trains a smaller IRIS model on medical datasets with memory-efficient settings.
Designed to run on systems with limited memory.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from tqdm import tqdm
import time
from datetime import datetime

# Add src to path
sys.path.append('src')

from models.iris_model import IRISModel
from data.unified_medical_loader import create_dataset
from losses.dice_loss import DiceLoss, compute_dice_score


class LightweightTrainer:
    """
    Memory-efficient trainer for IRIS model.
    """
    
    def __init__(self, config):
        self.config = config
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")
        
        # Load datasets
        self.datasets = self.load_datasets()
        self.train_samples, self.val_samples = self.create_splits()
        
        # Create model
        self.model = IRISModel(
            in_channels=config['in_channels'],
            base_channels=config['base_channels'],
            embed_dim=config['embed_dim'],
            num_tokens=config['num_tokens'],
            num_classes=config['num_classes']
        ).to(self.device)
        
        # Setup training
        self.criterion = DiceLoss(smooth=1e-5)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.val_dice_scores = []
        
        print(f"🚀 Lightweight Trainer initialized")
        print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Training samples: {len(self.train_samples)}")
        print(f"   Validation samples: {len(self.val_samples)}")
    
    def load_datasets(self):
        """Load available datasets."""
        datasets = {}
        
        dataset_configs = [
            ('amos', 'src/data/amos'),
            ('kits19', 'src/data/kits19'),
            ('bcv', 'src/data/bcv')
        ]
        
        for dataset_name, data_dir in dataset_configs:
            if os.path.exists(data_dir):
                try:
                    dataset = create_dataset(
                        dataset_type=dataset_name,
                        data_dir=data_dir,
                        split='train',
                        target_size=self.config['target_size']
                    )
                    
                    if len(dataset) > 0:
                        datasets[dataset_name] = dataset
                        print(f"✅ Loaded {dataset_name.upper()}: {len(dataset)} samples")
                    
                except Exception as e:
                    print(f"⚠️  Failed to load {dataset_name}: {e}")
        
        return datasets
    
    def create_splits(self):
        """Create train/validation splits."""
        all_samples = []
        
        for dataset_name, dataset in self.datasets.items():
            for i in range(len(dataset)):
                all_samples.append((dataset_name, i))
        
        # Shuffle and split
        np.random.shuffle(all_samples)
        split_idx = int(len(all_samples) * 0.9)  # 90% train, 10% val
        
        train_samples = all_samples[:split_idx]
        val_samples = all_samples[split_idx:]
        
        print(f"📊 Data split: {len(train_samples)} train, {len(val_samples)} val")
        
        return train_samples, val_samples
    
    def get_sample(self, dataset_name, idx):
        """Get a sample from a specific dataset."""
        return self.datasets[dataset_name][idx]
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        epoch_dice_scores = []
        
        # Shuffle training samples
        np.random.shuffle(self.train_samples)
        
        # Process samples in small batches
        episodes_per_epoch = min(self.config['episodes_per_epoch'], len(self.train_samples) // 2)
        
        pbar = tqdm(range(episodes_per_epoch), desc=f'Epoch {epoch+1}')
        
        for episode_idx in pbar:
            try:
                # Sample support and query
                support_idx = np.random.randint(len(self.train_samples))
                query_idx = np.random.randint(len(self.train_samples))
                
                support_dataset, support_sample_idx = self.train_samples[support_idx]
                query_dataset, query_sample_idx = self.train_samples[query_idx]
                
                support_sample = self.get_sample(support_dataset, support_sample_idx)
                query_sample = self.get_sample(query_dataset, query_sample_idx)
                
                # Process episode
                loss, dice = self.process_episode(support_sample, query_sample)
                
                if loss is not None:
                    epoch_losses.append(loss)
                    epoch_dice_scores.append(dice)
                
                # Update progress bar
                if len(epoch_losses) > 0:
                    pbar.set_postfix({
                        'Loss': f'{np.mean(epoch_losses):.4f}',
                        'DICE': f'{np.mean(epoch_dice_scores):.4f}'
                    })
                
            except Exception as e:
                continue
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        avg_dice = np.mean(epoch_dice_scores) if epoch_dice_scores else 0.0
        
        self.train_losses.append(avg_loss)
        
        return avg_loss, avg_dice
    
    def process_episode(self, support_sample, query_sample):
        """Process one training episode."""
        # Check if samples have labels
        if support_sample['label'] is None or query_sample['label'] is None:
            return None, None
        
        # Prepare tensors
        support_image = support_sample['image'].to(self.device)
        support_label = support_sample['label'].to(self.device)
        query_image = query_sample['image'].to(self.device)
        query_label = query_sample['label'].to(self.device)
        
        # Add batch dimensions
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
        
        # Skip if masks are too small
        if support_mask.sum() < 10 or query_mask.sum() < 10:
            return None, None
        
        # Add dimensions to masks
        support_mask = support_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        query_mask = query_mask.unsqueeze(0)  # (1, D, H, W)
        
        # Forward pass
        self.optimizer.zero_grad()
        
        try:
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
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("⚠️  GPU out of memory, skipping episode")
                torch.cuda.empty_cache()
            return None, None
    
    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        val_losses = []
        val_dice_scores = []
        
        num_val_episodes = min(50, len(self.val_samples) // 2)
        
        with torch.no_grad():
            for _ in tqdm(range(num_val_episodes), desc='Validation'):
                try:
                    # Sample validation episode
                    support_idx = np.random.randint(len(self.val_samples))
                    query_idx = np.random.randint(len(self.val_samples))
                    
                    support_dataset, support_sample_idx = self.val_samples[support_idx]
                    query_dataset, query_sample_idx = self.val_samples[query_idx]
                    
                    support_sample = self.get_sample(support_dataset, support_sample_idx)
                    query_sample = self.get_sample(query_dataset, query_sample_idx)
                    
                    if support_sample['label'] is None or query_sample['label'] is None:
                        continue
                    
                    # Prepare tensors
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
                    
                    if support_mask.sum() < 10 or query_mask.sum() < 10:
                        continue
                    
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
    
    def train(self, num_epochs, save_dir='checkpoints', save_every=5):
        """Main training loop."""
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_dice = 0.0
        
        print(f"\n🏋️  Starting lightweight training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training
            train_loss, train_dice = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_dice = self.validate(epoch)
            
            # Calculate epoch time
            epoch_time = time.time() - start_time
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train - Loss: {train_loss:.4f}, DICE: {train_dice:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, DICE: {val_dice:.4f}")
            
            # Save checkpoint
            is_best = val_dice > best_val_dice
            if is_best:
                best_val_dice = val_dice
            
            if (epoch + 1) % save_every == 0 or is_best:
                checkpoint_path = os.path.join(save_dir, f'iris_lightweight_epoch_{epoch+1}.pth')
                self.save_checkpoint(epoch + 1, checkpoint_path, is_best)
        
        print(f"\n🎉 Training completed!")
        print(f"   Best validation DICE: {best_val_dice:.4f}")
        
        # Save final checkpoint
        final_path = os.path.join(save_dir, 'iris_lightweight_final.pth')
        self.save_checkpoint(num_epochs, final_path)
        
        return best_val_dice


def main():
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("🏥 IRIS Lightweight Training")
    print("=" * 60)
    
    # Lightweight configuration
    config = {
        'in_channels': 1,
        'base_channels': 16,      # Reduced from 32
        'embed_dim': 128,         # Reduced from 256
        'num_tokens': 4,          # Reduced from 8
        'num_classes': 1,
        'target_size': (64, 64, 64),  # Smaller size
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'episodes_per_epoch': 200,    # Reduced from 500
        'datasets': ['amos', 'kits19', 'bcv']
    }
    
    print("📋 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Create trainer
    trainer = LightweightTrainer(config)
    
    # Start training
    best_dice = trainer.train(
        num_epochs=20,  # Reduced from 50
        save_dir='checkpoints',
        save_every=5
    )
    
    print(f"\n🎯 Training Summary:")
    print(f"   Best validation DICE: {best_dice:.4f}")
    print(f"   Checkpoints saved in: checkpoints/")
    
    # Save training info
    info_path = os.path.join('checkpoints', 'lightweight_training_info.json')
    with open(info_path, 'w') as f:
        json.dump({
            'config': config,
            'best_val_dice': best_dice,
            'datasets_used': list(trainer.datasets.keys()),
            'total_train_samples': len(trainer.train_samples),
            'total_val_samples': len(trainer.val_samples)
        }, f, indent=2)
    
    print(f"   Training info saved: {info_path}")


if __name__ == "__main__":
    main()
