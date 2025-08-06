"""
Evaluation Script for IRIS Framework on AMOS22 Dataset

This script provides comprehensive evaluation including:
- Per-organ DICE scores
- Cross-patient generalization
- In-context learning capabilities
- Visualization of predictions
"""

import os
import torch
import numpy as np
from tqdm import tqdm
import json
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

from src.models.iris_model import IRISModel
from src.data.amos22_loader import AMOS22Dataset
from src.losses.dice_loss import compute_dice_score
from train_amos22 import MemoryBank


class AMOS22Evaluator:
    """Comprehensive evaluator for AMOS22 dataset."""
    
    def __init__(self, model_path, data_dir, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['config']
        
        self.model = IRISModel(
            in_channels=config['in_channels'],
            num_classes=config['num_classes'],
            embed_dim=config['embed_dim']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load memory bank if available
        memory_bank_path = model_path.replace('.pth', '_memory.pth')
        if os.path.exists(memory_bank_path):
            self.memory_bank = MemoryBank(
                num_classes=config['num_organ_classes'],
                embed_dim=config['embed_dim']
            )
            self.memory_bank.load(memory_bank_path)
        else:
            self.memory_bank = None
        
        # Load dataset
        self.dataset = AMOS22Dataset(
            data_dir,
            split='test',
            target_size=(128, 128, 128)
        )
        
        # Organ labels
        self.organ_labels = self.dataset.organ_labels
    
    def evaluate_per_organ(self, num_samples=None):
        """Evaluate DICE scores per organ class."""
        print("Evaluating per-organ performance...")
        
        organ_scores = defaultdict(list)
        
        if num_samples is None:
            num_samples = len(self.dataset)
        
        for idx in tqdm(range(min(num_samples, len(self.dataset)))):
            sample = self.dataset[idx]
            
            if sample['label'] is None:
                continue
            
            image = sample['image'].unsqueeze(0).to(self.device)
            label = sample['label']
            
            # Process each organ
            for organ_id, organ_name in self.organ_labels.items():
                organ_mask = (label == organ_id).float()
                
                if organ_mask.sum() == 0:
                    continue  # Skip if organ not present
                
                # Use memory bank embedding if available
                if self.memory_bank and self.memory_bank.get(organ_id) is not None:
                    # Use memory bank for reference
                    with torch.no_grad():
                        task_embedding = self.memory_bank.get(organ_id).unsqueeze(0)
                        query_features = self.model.encoder(image)
                        predictions = self.model.decoder(query_features, task_embedding)
                else:
                    # Use same image as reference (self-supervised)
                    reference_mask = organ_mask.unsqueeze(0).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        predictions = self.model(image, image, reference_mask)
                
                # Compute DICE score
                pred_binary = (torch.sigmoid(predictions) > 0.5).cpu()
                dice = compute_dice_score(pred_binary.squeeze(), organ_mask.long())
                organ_scores[organ_name].append(dice.item())
        
        # Compute statistics
        results = {}
        for organ_name, scores in organ_scores.items():
            results[organ_name] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'n_samples': len(scores)
            }
        
        return results
    
    def evaluate_few_shot(self, n_shot=1, num_episodes=100):
        """Evaluate few-shot learning capability."""
        print(f"Evaluating {n_shot}-shot learning...")
        
        episode_scores = []
        
        for episode in tqdm(range(num_episodes)):
            # Randomly select support and query samples
            indices = np.random.choice(len(self.dataset), size=n_shot + 1, replace=False)
            support_indices = indices[:n_shot]
            query_idx = indices[-1]
            
            # Get query sample
            query_sample = self.dataset[query_idx]
            if query_sample['label'] is None:
                continue
            
            query_image = query_sample['image'].unsqueeze(0).to(self.device)
            query_label = query_sample['label']
            
            # Randomly select an organ present in query
            unique_organs = torch.unique(query_label)
            unique_organs = unique_organs[unique_organs > 0]  # Remove background
            
            if len(unique_organs) == 0:
                continue
            
            target_organ = unique_organs[np.random.randint(len(unique_organs))].item()
            query_mask = (query_label == target_organ).float()
            
            # Get support samples for the same organ
            support_images = []
            support_masks = []
            
            for sup_idx in support_indices:
                sup_sample = self.dataset[sup_idx]
                if sup_sample['label'] is None:
                    continue
                
                sup_mask = (sup_sample['label'] == target_organ).float()
                if sup_mask.sum() > 0:
                    support_images.append(sup_sample['image'].unsqueeze(0))
                    support_masks.append(sup_mask.unsqueeze(0).unsqueeze(0))
            
            if len(support_images) == 0:
                continue
            
            # Use first support as reference
            ref_image = support_images[0].to(self.device)
            ref_mask = support_masks[0].to(self.device)
            
            # Get prediction
            with torch.no_grad():
                predictions = self.model(query_image, ref_image, ref_mask)
            
            # Compute DICE
            pred_binary = (torch.sigmoid(predictions) > 0.5).cpu()
            dice = compute_dice_score(pred_binary.squeeze(), query_mask.long())
            episode_scores.append(dice.item())
        
        return {
            'mean': np.mean(episode_scores),
            'std': np.std(episode_scores),
            'min': np.min(episode_scores),
            'max': np.max(episode_scores),
            'n_episodes': len(episode_scores)
        }
    
    def visualize_predictions(self, sample_idx, save_path=None):
        """Visualize model predictions for a sample."""
        sample = self.dataset[sample_idx]
        
        if sample['label'] is None:
            print("No ground truth available for visualization")
            return
        
        image = sample['image'].unsqueeze(0).to(self.device)
        label = sample['label']
        
        # Get middle slice for visualization
        slice_idx = image.shape[-1] // 2
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Original image
        axes[0, 0].imshow(image[0, 0, :, :, slice_idx].cpu(), cmap='gray')
        axes[0, 0].set_title('Input Image')
        axes[0, 0].axis('off')
        
        # Ground truth
        axes[0, 1].imshow(label[:, :, slice_idx].cpu(), cmap='tab20')
        axes[0, 1].set_title('Ground Truth')
        axes[0, 1].axis('off')
        
        # Predictions for different organs
        organ_ids = [1, 6, 10]  # Spleen, Liver, Pancreas
        for i, organ_id in enumerate(organ_ids):
            organ_name = self.organ_labels.get(organ_id, f'Organ {organ_id}')
            organ_mask = (label == organ_id).float()
            
            if organ_mask.sum() == 0:
                axes[0, i+2].text(0.5, 0.5, f'{organ_name}\nNot Present', 
                                  ha='center', va='center')
                axes[0, i+2].axis('off')
                axes[1, i+1].axis('off')
                continue
            
            # Use organ as reference
            ref_mask = organ_mask.unsqueeze(0).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                predictions = self.model(image, image, ref_mask)
                pred_binary = (torch.sigmoid(predictions) > 0.5).cpu()
            
            # Show prediction
            axes[0, i+2].imshow(pred_binary[0, 0, :, :, slice_idx], cmap='binary')
            axes[0, i+2].set_title(f'Pred: {organ_name}')
            axes[0, i+2].axis('off')
            
            # Show overlay
            overlay = image[0, 0, :, :, slice_idx].cpu().numpy()
            overlay_rgb = np.stack([overlay, overlay, overlay], axis=-1)
            
            # Add predictions in red
            pred_slice = pred_binary[0, 0, :, :, slice_idx].numpy()
            overlay_rgb[pred_slice > 0, 0] = 1
            overlay_rgb[pred_slice > 0, 1] = 0
            overlay_rgb[pred_slice > 0, 2] = 0
            
            # Add ground truth in green
            gt_slice = organ_mask[:, :, slice_idx].numpy()
            overlay_rgb[gt_slice > 0, 0] = 0
            overlay_rgb[gt_slice > 0, 1] = 1
            overlay_rgb[gt_slice > 0, 2] = 0
            
            axes[1, i+1].imshow(overlay_rgb)
            axes[1, i+1].set_title(f'Overlay (R=Pred, G=GT)')
            axes[1, i+1].axis('off')
        
        # Compute overall DICE
        all_predictions = []
        all_labels = []
        
        for organ_id in self.organ_labels.keys():
            organ_mask = (label == organ_id).float()
            if organ_mask.sum() == 0:
                continue
            
            ref_mask = organ_mask.unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred = self.model(image, image, ref_mask)
                pred_binary = (torch.sigmoid(pred) > 0.5).cpu()
            
            all_predictions.append(pred_binary)
            all_labels.append(organ_mask.unsqueeze(0).unsqueeze(0))
        
        if len(all_predictions) > 0:
            combined_pred = torch.cat(all_predictions, dim=1).max(dim=1)[0]
            combined_label = torch.cat(all_labels, dim=1).max(dim=1)[0]
            overall_dice = compute_dice_score(combined_pred, combined_label.long())
            
            axes[1, 0].text(0.5, 0.5, f'Overall DICE: {overall_dice:.3f}', 
                           fontsize=14, ha='center', va='center')
        else:
            axes[1, 0].text(0.5, 0.5, 'No predictions', ha='center', va='center')
        
        axes[1, 0].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate IRIS model on AMOS22')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='src/data/amos',
                        help='Path to AMOS22 dataset')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to evaluate (None for all)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    parser.add_argument('--n_shot', type=int, default=1,
                        help='Number of support samples for few-shot evaluation')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    print("Loading model and data...")
    evaluator = AMOS22Evaluator(args.model_path, args.data_dir)
    
    # Per-organ evaluation
    print("\n" + "="*60)
    print("PER-ORGAN EVALUATION")
    print("="*60)
    organ_results = evaluator.evaluate_per_organ(args.num_samples)
    
    # Print results table
    df = pd.DataFrame(organ_results).T
    df = df.round(3)
    print("\nPer-Organ DICE Scores:")
    print(df.to_string())
    
    # Save results
    with open(os.path.join(args.output_dir, 'per_organ_results.json'), 'w') as f:
        json.dump(organ_results, f, indent=2)
    
    # Few-shot evaluation
    print("\n" + "="*60)
    print(f"{args.n_shot}-SHOT LEARNING EVALUATION")
    print("="*60)
    few_shot_results = evaluator.evaluate_few_shot(n_shot=args.n_shot)
    
    print(f"\n{args.n_shot}-shot DICE Score: {few_shot_results['mean']:.3f} ± {few_shot_results['std']:.3f}")
    print(f"Min: {few_shot_results['min']:.3f}, Max: {few_shot_results['max']:.3f}")
    print(f"Episodes: {few_shot_results['n_episodes']}")
    
    with open(os.path.join(args.output_dir, f'{args.n_shot}_shot_results.json'), 'w') as f:
        json.dump(few_shot_results, f, indent=2)
    
    # Generate visualizations
    if args.visualize:
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        vis_dir = os.path.join(args.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Visualize a few samples
        for i in range(min(5, len(evaluator.dataset))):
            print(f"Visualizing sample {i}...")
            save_path = os.path.join(vis_dir, f'sample_{i}.png')
            evaluator.visualize_predictions(i, save_path)
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_dice_scores = []
    for organ_name, scores in organ_results.items():
        all_dice_scores.extend([scores['mean']])
    
    if all_dice_scores:
        overall_mean = np.mean(all_dice_scores)
        overall_std = np.std(all_dice_scores)
        print(f"Overall Mean DICE: {overall_mean:.3f} ± {overall_std:.3f}")
    
    print("\nEvaluation complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()