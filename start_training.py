#!/usr/bin/env python3
"""
Simple script to start multi-dataset training with reasonable defaults.
"""

import subprocess
import sys
import os

def main():
    print("🚀 Starting IRIS Multi-Dataset Training")
    print("=" * 60)
    
    # Training configuration
    config = {
        # Dataset paths
        '--amos_dir': 'src/data/amos',
        '--bcv_dir': 'src/data/bcv', 
        '--chaos_dir': 'src/data/chaos',
        '--kits19_dir': 'src/data/kits19',
        
        # Model configuration
        '--in_channels': '1',
        '--base_channels': '32',
        '--embed_dim': '256',  # Reduced from 512 for faster training
        '--num_tokens': '8',   # Reduced from 10
        '--num_classes': '1',
        
        # Training configuration
        '--epochs': '50',      # Reduced from 100 for initial training
        '--episodes_per_epoch': '500',  # Reduced from 1000
        '--k_shot': '1',
        '--learning_rate': '1e-4',
        '--weight_decay': '1e-5',
        '--dice_weight': '1.0',
        '--ce_weight': '0.5',
        
        # Other settings
        '--save_dir': 'checkpoints',
        '--save_every': '5',   # Save more frequently
        '--seed': '42',
        '--wandb_project': 'iris-medical-segmentation'
    }
    
    # Build command
    cmd = ['python', 'train_multi_dataset.py']
    for key, value in config.items():
        if key == '--target_size':
            cmd.extend([key, '96', '96', '96'])
        else:
            cmd.extend([key, value])
    
    print("📋 Training Configuration:")
    print(f"   Epochs: {config['--epochs']}")
    print(f"   Episodes per epoch: {config['--episodes_per_epoch']}")
    print(f"   Target size: 96x96x96")
    print(f"   Embedding dim: {config['--embed_dim']}")
    print(f"   Learning rate: {config['--learning_rate']}")
    print(f"   Save directory: {config['--save_dir']}")
    
    print(f"\n🏃 Running command:")
    print(f"   {' '.join(cmd)}")
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Run training
    try:
        result = subprocess.run(cmd, check=True)
        print("\n🎉 Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()
