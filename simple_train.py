"""
Simple Training Script for Quick Testing

This is a minimal training script to verify the pipeline works.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os

from src.models.iris_model import IRISModel
from src.data.amos22_loader import AMOS22Dataset
from src.losses.dice_loss import DiceLoss, compute_dice_score


def train_simple():
    """Simple training loop for testing."""
    
    print("="*60)
    print("SIMPLE TRAINING TEST")
    print("="*60)
    
    # Configuration
    data_dir = "src/data/amos"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load dataset
    print("\n1. Loading dataset...")
    dataset = AMOS22Dataset(data_dir, split='train', target_size=(64, 64, 64))
    print(f"   Loaded {len(dataset)} samples")
    
    # Create simple data loader
    train_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )
    
    # Create model
    print("\n2. Creating model...")
    model = IRISModel(
        in_channels=1,
        num_classes=1,
        embed_dim=256
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop
    print("\n3. Training for 5 iterations...")
    model.train()
    
    losses = []
    dice_scores = []
    
    progress = tqdm(range(5), desc="Training")
    data_iter = iter(train_loader)
    
    for iteration in progress:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        # Get data
        image = batch['image'].to(device)
        label = batch['label']
        
        if label is None or len(torch.unique(label)) < 2:
            continue
        
        # Create a simple binary mask (e.g., liver = organ 6)
        target_organ = 6
        mask = (label == target_organ).float()
        
        # Skip if organ not present
        if mask.sum() == 0:
            # Try a different organ
            unique_organs = torch.unique(label)
            unique_organs = unique_organs[unique_organs > 0]
            if len(unique_organs) > 0:
                target_organ = unique_organs[0].item()
                mask = (label == target_organ).float()
            else:
                continue
        
        mask = mask.unsqueeze(1).to(device)
        
        # Use same image as reference (self-supervised approach)
        predictions = model(image, image, mask)
        
        # Resize predictions to match mask size if needed
        if predictions.shape[-3:] != mask.shape[-3:]:
            predictions = torch.nn.functional.interpolate(
                predictions, 
                size=mask.shape[-3:],
                mode='trilinear',
                align_corners=False
            )
        
        # Compute loss
        loss = criterion(predictions, mask.squeeze(1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            pred_binary = (torch.sigmoid(predictions) > 0.5).float()
            # Simple DICE computation for binary case
            intersection = (pred_binary * mask).sum()
            union = pred_binary.sum() + mask.sum()
            dice = (2.0 * intersection / (union + 1e-8))
        
        losses.append(loss.item())
        dice_scores.append(dice.item())
        
        # Update progress
        progress.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice.item():.4f}'
        })
    
    # Results
    print("\n4. Training Results:")
    print(f"   Average Loss: {np.mean(losses):.4f}")
    print(f"   Average DICE: {np.mean(dice_scores):.4f}")
    
    # Save checkpoint
    checkpoint_dir = "checkpoints_test"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, "simple_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'dice_scores': dice_scores
    }, checkpoint_path)
    
    print(f"\n5. Checkpoint saved to: {checkpoint_path}")
    
    return True


def test_saved_model():
    """Test the saved model."""
    
    print("\n" + "="*60)
    print("TESTING SAVED MODEL")
    print("="*60)
    
    checkpoint_path = "checkpoints_test/simple_model.pth"
    
    if not os.path.exists(checkpoint_path):
        print("No checkpoint found. Run training first.")
        return False
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = IRISModel(
        in_channels=1,
        num_classes=1,
        embed_dim=256
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully!")
    
    # Test on one sample
    data_dir = "src/data/amos"
    dataset = AMOS22Dataset(data_dir, split='train', target_size=(64, 64, 64))
    
    sample = dataset[0]
    image = sample['image'].unsqueeze(0).to(device)
    
    # Create a dummy mask
    dummy_mask = torch.zeros_like(image)
    dummy_mask[:, :, 20:40, 20:40, 20:40] = 1.0
    
    with torch.no_grad():
        predictions = model(image, image, dummy_mask)
        pred_prob = torch.sigmoid(predictions)
    
    print(f"Prediction shape: {predictions.shape}")
    print(f"Prediction range: [{pred_prob.min():.3f}, {pred_prob.max():.3f}]")
    print(f"Mean prediction: {pred_prob.mean():.3f}")
    
    return True


def main():
    """Main function."""
    
    print("\n" + "="*60)
    print("SIMPLE TRAINING AND TESTING SCRIPT")
    print("="*60)
    
    # Check data
    if not os.path.exists("src/data/amos"):
        print("ERROR: AMOS22 data not found at src/data/amos/")
        return
    
    # Train
    success = train_simple()
    
    if success:
        # Test
        test_saved_model()
        
        print("\n" + "="*60)
        print("SUCCESS: Training and testing complete!")
        print("="*60)
        print("\nTo train with the full pipeline, use:")
        print("  python train_amos22.py --data_dir src/data/amos --batch_size 4 --max_iterations 1000")
        print("\nTo evaluate the model, use:")
        print("  python evaluate_amos22.py --model_path checkpoints_test/simple_model.pth")
    else:
        print("\nERROR: Training failed")


if __name__ == "__main__":
    main()