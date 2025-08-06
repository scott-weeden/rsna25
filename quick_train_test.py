"""
Quick Training and Testing Script for AMOS22

This script provides a simplified way to quickly test the training pipeline
with a small number of iterations for verification purposes.
"""

import torch
import os
import sys

def quick_test():
    """Quick test of the training pipeline."""
    
    print("="*60)
    print("QUICK TRAINING PIPELINE TEST")
    print("="*60)
    
    # Check if AMOS22 data exists
    data_dir = "src/data/amos"
    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory not found: {data_dir}")
        print("\nPlease ensure AMOS22 data is downloaded to src/data/amos/")
        print("Expected structure:")
        print("  src/data/amos/")
        print("    ├── imagesTr/  (training images)")
        print("    ├── labelsTr/  (training labels)")
        print("    ├── imagesVa/  (validation images)")
        print("    ├── labelsVa/  (validation labels)")
        print("    ├── imagesTs/  (test images)")
        print("    └── labelsTs/  (test labels)")
        return False
    
    # Test data loading
    print("\n1. Testing Data Loading...")
    try:
        from src.data.amos22_loader import AMOS22Dataset
        dataset = AMOS22Dataset(data_dir, split='train', target_size=(64, 64, 64))
        print(f"   SUCCESS: Loaded {len(dataset)} training samples")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"   Sample shape: {sample['image'].shape}")
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    # Test model creation
    print("\n2. Testing Model Creation...")
    try:
        from src.models.iris_model import IRISModel
        model = IRISModel(in_channels=1, num_classes=1, embed_dim=256)
        print(f"   SUCCESS: Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    # Test loss computation
    print("\n3. Testing Loss Computation...")
    try:
        from src.losses.dice_loss import DiceLoss
        criterion = DiceLoss()
        
        # Create dummy data for binary segmentation
        predictions = torch.randn(2, 1, 64, 64, 64)
        targets = torch.randint(0, 2, (2, 64, 64, 64)).float()  # Binary targets
        
        loss = criterion(predictions, targets)
        print(f"   SUCCESS: Loss computed = {loss.item():.4f}")
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    # Quick training test (just a few iterations)
    print("\n4. Testing Training Loop (5 iterations)...")
    print("   This will test the basic training functionality")
    
    cmd = (
        "python train_amos22.py "
        "--data_dir src/data/amos "
        "--checkpoint_dir checkpoints_test "
        "--batch_size 2 "
        "--max_iterations 5 "
        "--learning_rate 1e-4 "
        "--num_workers 0"
    )
    
    print(f"\n   Command: {cmd}")
    print("\n   Starting mini training run...")
    
    result = os.system(cmd)
    
    if result == 0:
        print("\n   SUCCESS: Training pipeline works!")
        
        # Check if checkpoint was created
        if os.path.exists("checkpoints_test"):
            print("   Checkpoint directory created successfully")
    else:
        print("\n   ERROR: Training failed")
        return False
    
    return True


def full_training_instructions():
    """Print instructions for full training."""
    
    print("\n" + "="*60)
    print("FULL TRAINING INSTRUCTIONS")
    print("="*60)
    
    print("\nTo train the model on AMOS22 dataset:")
    print("\n1. Basic Training (recommended for testing):")
    print("   python train_amos22.py \\")
    print("     --data_dir src/data/amos \\")
    print("     --checkpoint_dir checkpoints \\")
    print("     --batch_size 4 \\")
    print("     --max_iterations 1000 \\")
    print("     --learning_rate 1e-4")
    
    print("\n2. Full Training (as per paper - 80K iterations):")
    print("   python train_amos22.py \\")
    print("     --data_dir src/data/amos \\")
    print("     --checkpoint_dir checkpoints \\")
    print("     --batch_size 32 \\")
    print("     --max_iterations 80000 \\")
    print("     --learning_rate 1e-4 \\")
    print("     --use_wandb")
    
    print("\n3. Resume Training from Checkpoint:")
    print("   python train_amos22.py \\")
    print("     --resume checkpoints/checkpoint_iter_5000.pth \\")
    print("     --max_iterations 80000")
    
    print("\n4. Validation Only:")
    print("   python train_amos22.py \\")
    print("     --resume checkpoints/best_model.pth \\")
    print("     --validate_only")
    
    print("\n5. Testing Only:")
    print("   python train_amos22.py \\")
    print("     --resume checkpoints/best_model.pth \\")
    print("     --test_only")


def evaluation_instructions():
    """Print instructions for evaluation."""
    
    print("\n" + "="*60)
    print("EVALUATION INSTRUCTIONS")
    print("="*60)
    
    print("\nTo evaluate a trained model:")
    
    print("\n1. Basic Evaluation:")
    print("   python evaluate_amos22.py \\")
    print("     --model_path checkpoints/best_model.pth \\")
    print("     --data_dir src/data/amos \\")
    print("     --output_dir evaluation_results")
    
    print("\n2. Evaluation with Visualizations:")
    print("   python evaluate_amos22.py \\")
    print("     --model_path checkpoints/best_model.pth \\")
    print("     --data_dir src/data/amos \\")
    print("     --output_dir evaluation_results \\")
    print("     --visualize")
    
    print("\n3. Few-shot Evaluation (5-shot):")
    print("   python evaluate_amos22.py \\")
    print("     --model_path checkpoints/best_model.pth \\")
    print("     --data_dir src/data/amos \\")
    print("     --output_dir evaluation_results \\")
    print("     --n_shot 5")
    
    print("\n4. Quick Evaluation (10 samples only):")
    print("   python evaluate_amos22.py \\")
    print("     --model_path checkpoints/best_model.pth \\")
    print("     --data_dir src/data/amos \\")
    print("     --output_dir evaluation_results \\")
    print("     --num_samples 10")


def main():
    """Main function."""
    
    print("\n" + "="*60)
    print("IRIS FRAMEWORK - AMOS22 TRAINING & TESTING GUIDE")
    print("="*60)
    
    # Run quick test
    print("\nRunning quick pipeline test...")
    success = quick_test()
    
    if success:
        print("\n" + "="*60)
        print("SUCCESS: All components working correctly!")
        print("="*60)
        
        # Print full instructions
        full_training_instructions()
        evaluation_instructions()
        
        print("\n" + "="*60)
        print("KEY FEATURES IMPLEMENTED")
        print("="*60)
        print("✓ Episodic training for in-context learning")
        print("✓ Memory bank with exponential moving average")
        print("✓ 75/5/20 train/val/test splits")
        print("✓ Lamb optimizer support (falls back to AdamW if not available)")
        print("✓ Mixed precision training (automatic on GPU)")
        print("✓ Checkpoint saving and resumption")
        print("✓ Per-organ DICE evaluation")
        print("✓ Few-shot learning evaluation")
        print("✓ Visualization of predictions")
        
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("1. Install Lamb optimizer (optional):")
        print("   pip install torch-optimizer")
        print("\n2. Install Weights & Biases for logging (optional):")
        print("   pip install wandb")
        print("\n3. Start training with small iterations to verify")
        print("\n4. Once verified, run full 80K iteration training")
    else:
        print("\n" + "="*60)
        print("ERROR: Pipeline test failed")
        print("="*60)
        print("\nPlease check:")
        print("1. AMOS22 data is downloaded to src/data/amos/")
        print("2. All dependencies are installed (nibabel, torch, etc.)")
        print("3. Review error messages above for specific issues")


if __name__ == "__main__":
    main()