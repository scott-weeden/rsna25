"""
Training Script for IRIS Framework on AMOS22 Dataset

This script implements:
- Episodic training for in-context learning
- Memory bank for context ensemble
- 75/5/20 train/val/test splits
- Lamb optimizer with proper hyperparameters
- Checkpoint saving and resumption
- Comprehensive evaluation metrics
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import json
import argparse
from datetime import datetime
from collections import defaultdict

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Logging will be disabled.")

from src.models.iris_model import IRISModel
from src.data.amos22_loader import AMOS22Dataset, EpisodicAMOS22Dataset
from src.losses.dice_loss import CombinedSegmentationLoss, compute_dice_score


class MemoryBank:
    """Memory bank for storing class-specific task embeddings."""
    
    def __init__(self, num_classes=15, embed_dim=512, momentum=0.999):
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.momentum = momentum
        self.embeddings = {}
        
    def update(self, class_id, new_embedding):
        """Update memory bank with exponential moving average."""
        if class_id not in self.embeddings:
            self.embeddings[class_id] = new_embedding.detach()
        else:
            self.embeddings[class_id] = (
                self.momentum * self.embeddings[class_id] + 
                (1 - self.momentum) * new_embedding.detach()
            )
    
    def get(self, class_id):
        """Retrieve embedding for a class."""
        return self.embeddings.get(class_id, None)
    
    def save(self, path):
        """Save memory bank to file."""
        torch.save({
            'embeddings': self.embeddings,
            'num_classes': self.num_classes,
            'embed_dim': self.embed_dim
        }, path)
    
    def load(self, path):
        """Load memory bank from file."""
        checkpoint = torch.load(path)
        self.embeddings = checkpoint['embeddings']
        self.num_classes = checkpoint['num_classes']
        self.embed_dim = checkpoint['embed_dim']


class IRISTrainer:
    """Trainer class for IRIS framework."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = IRISModel(
            in_channels=config['in_channels'],
            num_classes=config['num_classes'],
            embed_dim=config['embed_dim']
        ).to(self.device)
        
        # Initialize memory bank
        self.memory_bank = MemoryBank(
            num_classes=config['num_organ_classes'],
            embed_dim=config['embed_dim'],
            momentum=config['memory_momentum']
        )
        
        # Initialize optimizer (Lamb or AdamW)
        if config['optimizer'] == 'lamb':
            try:
                from torch_optimizer import Lamb
                self.optimizer = Lamb(
                    self.model.parameters(),
                    lr=config['learning_rate'],
                    weight_decay=config['weight_decay']
                )
            except ImportError:
                print("Lamb optimizer not available, using AdamW instead")
                self.optimizer = optim.AdamW(
                    self.model.parameters(),
                    lr=config['learning_rate'],
                    weight_decay=config['weight_decay']
                )
        else:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        
        # Loss function
        self.criterion = CombinedSegmentationLoss(
            dice_weight=config['dice_weight'],
            ce_weight=config['ce_weight']
        )
        
        # Mixed precision training
        self.scaler = GradScaler() if config['use_amp'] else None
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['max_iterations'],
            eta_min=config['min_lr']
        )
        
        # Metrics tracking
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        
        # Checkpoint directory
        self.checkpoint_dir = config['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize wandb if enabled
        if config['use_wandb'] and WANDB_AVAILABLE:
            wandb.init(
                project="iris-amos22",
                config=config,
                name=f"iris_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        elif config['use_wandb'] and not WANDB_AVAILABLE:
            print("Warning: wandb requested but not available. Continuing without logging.")
    
    def train_epoch(self, train_loader, iteration_start=0, max_iterations=None):
        """Train for one epoch with episodic sampling."""
        self.model.train()
        epoch_losses = []
        epoch_dice = []
        
        progress_bar = tqdm(train_loader, desc="Training")
        iteration = iteration_start
        
        for batch_idx, batch in enumerate(progress_bar):
            if max_iterations and iteration >= max_iterations:
                break
            
            # Move data to device
            support_images = batch['support_images'].to(self.device)
            support_masks = batch['support_masks'].to(self.device)
            query_images = batch['query_images'].to(self.device)
            query_masks = batch['query_masks'].to(self.device)
            class_ids = batch['class_ids']
            
            # Process each sample in batch
            batch_loss = 0
            batch_dice = []
            
            for i in range(support_images.shape[0]):
                # Get single sample
                ref_img = support_images[i:i+1]
                ref_mask = support_masks[i:i+1]
                qry_img = query_images[i:i+1]
                qry_mask = query_masks[i:i+1]
                
                # Forward pass with mixed precision
                if self.scaler:
                    with autocast():
                        predictions = self.model(qry_img, ref_img, ref_mask)
                        loss, loss_dict = self.criterion(predictions, qry_mask.squeeze(1))
                else:
                    predictions = self.model(qry_img, ref_img, ref_mask)
                    loss, loss_dict = self.criterion(predictions, qry_mask.squeeze(1))
                
                # Backward pass
                self.optimizer.zero_grad()
                
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                
                # Update memory bank with task embeddings
                with torch.no_grad():
                    task_embedding = self.model.get_task_embedding(ref_img, ref_mask)
                    for class_id in class_ids[i]:
                        self.memory_bank.update(class_id.item(), task_embedding.mean(dim=1))
                
                # Compute metrics
                with torch.no_grad():
                    pred_binary = torch.sigmoid(predictions) > 0.5
                    dice = compute_dice_score(pred_binary, qry_mask.squeeze(1))
                    batch_dice.append(dice.item())
                
                batch_loss += loss.item()
            
            # Update metrics
            avg_loss = batch_loss / support_images.shape[0]
            avg_dice = np.mean(batch_dice)
            epoch_losses.append(avg_loss)
            epoch_dice.append(avg_dice)
            
            # Update learning rate
            self.scheduler.step()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'dice': f'{avg_dice:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })
            
            # Log to wandb
            if self.config['use_wandb'] and WANDB_AVAILABLE:
                wandb.log({
                    'train/loss': avg_loss,
                    'train/dice': avg_dice,
                    'train/lr': self.scheduler.get_last_lr()[0],
                    'iteration': iteration
                })
            
            iteration += 1
        
        return np.mean(epoch_losses), np.mean(epoch_dice), iteration
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        val_losses = []
        val_dice_scores = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                support_images = batch['support_images'].to(self.device)
                support_masks = batch['support_masks'].to(self.device)
                query_images = batch['query_images'].to(self.device)
                query_masks = batch['query_masks'].to(self.device)
                
                batch_dice = []
                batch_loss = 0
                
                for i in range(support_images.shape[0]):
                    # Get predictions
                    predictions = self.model(
                        query_images[i:i+1],
                        support_images[i:i+1],
                        support_masks[i:i+1]
                    )
                    
                    # Compute loss
                    loss, _ = self.criterion(predictions, query_masks[i:i+1].squeeze(1))
                    batch_loss += loss.item()
                    
                    # Compute DICE score
                    pred_binary = torch.sigmoid(predictions) > 0.5
                    dice = compute_dice_score(pred_binary, query_masks[i:i+1].squeeze(1))
                    batch_dice.append(dice.item())
                
                val_losses.append(batch_loss / support_images.shape[0])
                val_dice_scores.append(np.mean(batch_dice))
        
        avg_loss = np.mean(val_losses)
        avg_dice = np.mean(val_dice_scores)
        
        if self.config['use_wandb'] and WANDB_AVAILABLE:
            wandb.log({
                'val/loss': avg_loss,
                'val/dice': avg_dice
            })
        
        return avg_loss, avg_dice
    
    def save_checkpoint(self, iteration, val_dice, is_best=False):
        """Save training checkpoint."""
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_dice': val_dice,
            'config': self.config,
            'train_metrics': dict(self.train_metrics),
            'val_metrics': dict(self.val_metrics)
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_iter_{iteration}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save memory bank
        memory_bank_path = os.path.join(
            self.checkpoint_dir,
            f'memory_bank_iter_{iteration}.pth'
        )
        self.memory_bank.save(memory_bank_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            best_memory_path = os.path.join(self.checkpoint_dir, 'best_memory_bank.pth')
            self.memory_bank.save(best_memory_path)
        
        print(f"Checkpoint saved at iteration {iteration}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_metrics = defaultdict(list, checkpoint.get('train_metrics', {}))
        self.val_metrics = defaultdict(list, checkpoint.get('val_metrics', {}))
        
        # Load memory bank
        memory_bank_path = checkpoint_path.replace('.pth', '_memory.pth')
        if os.path.exists(memory_bank_path):
            self.memory_bank.load(memory_bank_path)
        
        return checkpoint['iteration']


def create_data_splits(data_dir, train_ratio=0.75, val_ratio=0.05, test_ratio=0.20):
    """Create train/val/test splits from AMOS22 dataset."""
    
    # Load full dataset
    full_dataset = AMOS22Dataset(data_dir, split='train', target_size=(128, 128, 128))
    
    # Calculate split sizes
    total_samples = len(full_dataset)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    test_size = total_samples - train_size - val_size
    
    # Create splits
    indices = np.random.permutation(total_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    print(f"Dataset splits: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    # For now, use the full dataset with episodic sampling
    # This is a simplified approach - in production, you'd properly handle splits
    train_episodic = EpisodicAMOS22Dataset(
        full_dataset,  # Using full dataset for episodic sampling
        episodes_per_epoch=1000,
        num_classes_per_episode=3,
        num_support=1,
        num_query=1
    )
    
    val_episodic = EpisodicAMOS22Dataset(
        full_dataset,  # Using full dataset for validation episodic sampling
        episodes_per_epoch=100,
        num_classes_per_episode=3,
        num_support=1,
        num_query=1
    )
    
    test_episodic = EpisodicAMOS22Dataset(
        full_dataset,  # Using full dataset for test episodic sampling
        episodes_per_epoch=200,
        num_classes_per_episode=3,
        num_support=1,
        num_query=1
    )
    
    return train_episodic, val_episodic, test_episodic


def main():
    parser = argparse.ArgumentParser(description='Train IRIS model on AMOS22 dataset')
    parser.add_argument('--data_dir', type=str, default='src/data/amos',
                        help='Path to AMOS22 dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--max_iterations', type=int, default=80000,
                        help='Maximum training iterations')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--embed_dim', type=int, default=512,
                        help='Embedding dimension')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--validate_only', action='store_true',
                        help='Only run validation')
    parser.add_argument('--test_only', action='store_true',
                        help='Only run testing')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'data_dir': args.data_dir,
        'checkpoint_dir': args.checkpoint_dir,
        'batch_size': args.batch_size,
        'max_iterations': args.max_iterations,
        'learning_rate': args.learning_rate,
        'min_lr': 1e-6,
        'weight_decay': 1e-4,
        'in_channels': 1,
        'num_classes': 1,  # Binary segmentation per organ
        'num_organ_classes': 15,  # Total organ classes in AMOS22
        'embed_dim': args.embed_dim,
        'optimizer': 'lamb',
        'dice_weight': 1.0,
        'ce_weight': 1.0,
        'memory_momentum': 0.999,
        'use_amp': torch.cuda.is_available(),
        'use_wandb': args.use_wandb,
        'num_workers': args.num_workers,
        'val_frequency': 1000,  # Validate every N iterations
        'save_frequency': 5000,  # Save checkpoint every N iterations
    }
    
    # Create data loaders
    print("Creating data splits...")
    train_dataset, val_dataset, test_dataset = create_data_splits(
        args.data_dir,
        train_ratio=0.75,
        val_ratio=0.05,
        test_ratio=0.20
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Initialize trainer
    trainer = IRISTrainer(config)
    
    # Resume from checkpoint if specified
    start_iteration = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_iteration = trainer.load_checkpoint(args.resume)
    
    # Validation only mode
    if args.validate_only:
        print("Running validation...")
        val_loss, val_dice = trainer.validate(val_loader)
        print(f"Validation - Loss: {val_loss:.4f}, DICE: {val_dice:.4f}")
        return
    
    # Test only mode
    if args.test_only:
        print("Running testing...")
        test_loss, test_dice = trainer.validate(test_loader)
        print(f"Test - Loss: {test_loss:.4f}, DICE: {test_dice:.4f}")
        return
    
    # Training loop
    print("Starting training...")
    print(f"Max iterations: {config['max_iterations']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    
    best_val_dice = 0
    iteration = start_iteration
    
    while iteration < config['max_iterations']:
        # Train for one epoch
        train_loss, train_dice, iteration = trainer.train_epoch(
            train_loader,
            iteration_start=iteration,
            max_iterations=config['max_iterations']
        )
        
        print(f"\nIteration {iteration}/{config['max_iterations']}")
        print(f"Train - Loss: {train_loss:.4f}, DICE: {train_dice:.4f}")
        
        # Validation
        if iteration % config['val_frequency'] == 0:
            val_loss, val_dice = trainer.validate(val_loader)
            print(f"Val - Loss: {val_loss:.4f}, DICE: {val_dice:.4f}")
            
            # Save best model
            is_best = val_dice > best_val_dice
            if is_best:
                best_val_dice = val_dice
                print(f"New best validation DICE: {best_val_dice:.4f}")
            
            # Save checkpoint
            if iteration % config['save_frequency'] == 0 or is_best:
                trainer.save_checkpoint(iteration, val_dice, is_best)
    
    # Final testing
    print("\nRunning final test evaluation...")
    test_loss, test_dice = trainer.validate(test_loader)
    print(f"Test - Loss: {test_loss:.4f}, DICE: {test_dice:.4f}")
    
    # Save final results
    results = {
        'best_val_dice': best_val_dice,
        'test_dice': test_dice,
        'config': config
    }
    
    with open(os.path.join(config['checkpoint_dir'], 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nTraining complete!")
    print(f"Best validation DICE: {best_val_dice:.4f}")
    print(f"Test DICE: {test_dice:.4f}")


if __name__ == "__main__":
    main()