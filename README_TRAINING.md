# IRIS Framework Training Guide for AMOS22 Dataset

## Overview

This guide explains how to train, validate, and test the IRIS framework using the AMOS22 medical imaging dataset.

## Prerequisites

1. **Install Dependencies**:
```bash
pip install torch torchvision nibabel
pip install torch-optimizer  # For Lamb optimizer (optional)
pip install wandb  # For experiment tracking (optional)
```

2. **Download AMOS22 Dataset**:
- Place the dataset in `src/data/amos/` with the following structure:
```
src/data/amos/
├── imagesTr/  # Training images (*.nii.gz)
├── labelsTr/  # Training labels (*.nii.gz)
├── imagesVa/  # Validation images
├── labelsVa/  # Validation labels
├── imagesTs/  # Test images
└── labelsTs/  # Test labels (if available)
```

## Training Scripts

### 1. Full Training Pipeline (`train_amos22.py`)

Complete training with episodic sampling, memory bank, and all features:

#### Basic Training (Quick Test)
```bash
python train_amos22.py \
  --data_dir src/data/amos \
  --checkpoint_dir checkpoints \
  --batch_size 4 \
  --max_iterations 1000 \
  --learning_rate 1e-4
```

#### Full Training (80K iterations as per paper)
```bash
python train_amos22.py \
  --data_dir src/data/amos \
  --checkpoint_dir checkpoints \
  --batch_size 32 \
  --max_iterations 80000 \
  --learning_rate 1e-4 \
  --use_wandb
```

#### Resume Training from Checkpoint
```bash
python train_amos22.py \
  --resume checkpoints/checkpoint_iter_5000.pth \
  --max_iterations 80000
```

### 2. Simple Training (`simple_train.py`)

Minimal training script for quick verification:

```bash
python simple_train.py
```

This will:
- Load AMOS22 data
- Train for 5 iterations
- Save a checkpoint to `checkpoints_test/simple_model.pth`

## Validation & Testing

### 1. Validation Only
```bash
python train_amos22.py \
  --resume checkpoints/best_model.pth \
  --validate_only
```

### 2. Test Only
```bash
python train_amos22.py \
  --resume checkpoints/best_model.pth \
  --test_only
```

## Evaluation (`evaluate_amos22.py`)

Comprehensive evaluation with multiple metrics:

### Basic Evaluation
```bash
python evaluate_amos22.py \
  --model_path checkpoints/best_model.pth \
  --data_dir src/data/amos \
  --output_dir evaluation_results
```

### Evaluation with Visualizations
```bash
python evaluate_amos22.py \
  --model_path checkpoints/best_model.pth \
  --data_dir src/data/amos \
  --output_dir evaluation_results \
  --visualize
```

### Few-shot Evaluation (N-shot learning)
```bash
python evaluate_amos22.py \
  --model_path checkpoints/best_model.pth \
  --data_dir src/data/amos \
  --output_dir evaluation_results \
  --n_shot 5
```

### Quick Evaluation (subset of samples)
```bash
python evaluate_amos22.py \
  --model_path checkpoints/best_model.pth \
  --data_dir src/data/amos \
  --output_dir evaluation_results \
  --num_samples 10
```

## Key Features Implemented

### Training Features
- **Episodic Sampling**: In-context learning with support/query pairs
- **Memory Bank**: Context ensemble with exponential moving average (α=0.999)
- **Data Splits**: 75%/5%/20% train/validation/test
- **Optimizer**: Lamb optimizer (falls back to AdamW if not available)
- **Mixed Precision**: Automatic mixed precision on GPU
- **Learning Rate Schedule**: Cosine annealing
- **Checkpointing**: Automatic saving and resumption

### Model Architecture
- **3D UNet Encoder**: 6 stages with channels [32, 32, 64, 128, 256, 512]
- **Query-Based Decoder**: Task-guided decoding with cross-attention
- **Task Encoding**: Dual-path (foreground + context) encoding
- **Fixed Channel Mismatch**: Proper skip connections between encoder-decoder

### Evaluation Metrics
- **Per-organ DICE scores**: Individual performance for 15 organ classes
- **Few-shot learning**: N-shot adaptation capability
- **Cross-patient generalization**: Performance on unseen patients
- **Visualization**: Prediction overlays and comparisons

## Training Parameters (Paper Specifications)

| Parameter | Value |
|-----------|-------|
| Max Iterations | 80,000 |
| Batch Size | 32 |
| Learning Rate | 1e-4 |
| Optimizer | Lamb |
| Input Size | 128×128×128 |
| Embedding Dim | 512 |
| Memory Momentum | 0.999 |
| Loss | DICE + Cross-Entropy |

## Expected Performance

After full training (80K iterations), expect:
- Average DICE score: >70% across organs
- Per-organ DICE: 60-85% depending on organ complexity
- Few-shot (1-shot) DICE: >60%
- Cross-dataset generalization: >55%

## Troubleshooting

### Out of Memory
- Reduce batch size: `--batch_size 2`
- Reduce input size in data loader
- Disable mixed precision if issues persist

### Slow Training
- Ensure GPU is available: `torch.cuda.is_available()`
- Reduce number of workers: `--num_workers 0`
- Use smaller input size: (64×64×64)

### No Lamb Optimizer
```bash
pip install torch-optimizer
```
Will automatically fall back to AdamW if not available.

### Data Loading Issues
- Verify AMOS22 data structure
- Check NIfTI files are valid (`.nii.gz`)
- Ensure sufficient disk space for caching

## Directory Structure After Training

```
checkpoints/
├── checkpoint_iter_1000.pth
├── checkpoint_iter_5000.pth
├── best_model.pth
├── best_memory_bank.pth
└── results.json

evaluation_results/
├── per_organ_results.json
├── 1_shot_results.json
└── visualizations/
    ├── sample_0.png
    ├── sample_1.png
    └── ...
```

## Citation

If using this implementation, please cite the original IRIS paper and this implementation.

## Support

For issues or questions:
1. Check existing GitHub issues
2. Review the error messages carefully
3. Ensure all dependencies are correctly installed
4. Verify AMOS22 dataset structure