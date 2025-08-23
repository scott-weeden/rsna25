"""
End-to-End Prediction Pipeline with ROC AUC Evaluation and Outlier Analysis

This script:
1. Loads or creates a trained IRIS model
2. Performs predictions on test data
3. Evaluates using sklearn.metrics.roc_auc_score
4. Identifies outliers in predictions
5. Provides recommendations for model adjustments
"""

import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.metrics import confusion_matrix, classification_report
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime
from collections import defaultdict

# Import model and data components
from src.models.iris_model import IRISModel
from src.losses.dice_loss import compute_dice_score


class PredictionEvaluator:
    """
    Comprehensive evaluation system for IRIS model predictions.
    Includes ROC AUC analysis and outlier detection.
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Storage for predictions and metrics
        self.predictions = []
        self.ground_truths = []
        self.probabilities = []
        self.dice_scores = []
        self.patient_ids = []
        self.outlier_cases = []
        
    def predict_batch(self, query_images, reference_images, reference_masks):
        """
        Generate predictions for a batch of query images.
        
        Returns:
            dict: Contains predictions, probabilities, and logits
        """
        with torch.no_grad():
            # Move to device
            query_images = query_images.to(self.device)
            reference_images = reference_images.to(self.device)
            reference_masks = reference_masks.to(self.device)
            
            # Forward pass
            logits = self.model(query_images, reference_images, reference_masks)
            
            # Convert to probabilities
            probabilities = torch.sigmoid(logits)
            
            # Binary predictions
            predictions = (probabilities > 0.5).float()
            
            return {
                'predictions': predictions.cpu().numpy(),
                'probabilities': probabilities.cpu().numpy(),
                'logits': logits.cpu().numpy()
            }
    
    def evaluate_test_set(self, test_loader, verbose=True):
        """
        Evaluate model on entire test set with comprehensive metrics.
        """
        print("🔍 Starting comprehensive evaluation...")
        
        all_predictions = []
        all_ground_truths = []
        all_probabilities = []
        all_dice_scores = []
        sample_metadata = []
        
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Handle different batch formats
            if isinstance(batch, dict):
                # Episodic format
                query_images = batch.get('query_images', batch.get('query_image'))
                reference_images = batch.get('support_images', batch.get('reference_image'))
                reference_masks = batch.get('support_masks', batch.get('reference_mask'))
                query_masks = batch.get('query_masks', batch.get('query_mask'))
                patient_ids = batch.get('patient_ids', [f"patient_{batch_idx}_{i}" for i in range(len(query_images))])
            else:
                # Assume tuple format (images, masks)
                query_images, query_masks = batch
                # Use first sample as reference (simplified)
                reference_images = query_images[:1].repeat(len(query_images), 1, 1, 1, 1)
                reference_masks = query_masks[:1].repeat(len(query_masks), 1, 1, 1, 1)
                patient_ids = [f"patient_{batch_idx}_{i}" for i in range(len(query_images))]
            
            # Get predictions
            results = self.predict_batch(query_images, reference_images, reference_masks)
            
            # Process each sample in batch
            for i in range(len(query_images)):
                pred = results['predictions'][i]
                prob = results['probabilities'][i]
                gt = query_masks[i].cpu().numpy()
                
                # Flatten for ROC AUC calculation
                pred_flat = pred.flatten()
                prob_flat = prob.flatten()
                gt_flat = gt.flatten()
                
                all_predictions.extend(pred_flat)
                all_probabilities.extend(prob_flat)
                all_ground_truths.extend(gt_flat)
                
                # Calculate DICE score
                dice = self._calculate_dice(pred, gt)
                all_dice_scores.append(dice)
                
                # Store metadata
                sample_metadata.append({
                    'patient_id': patient_ids[i] if i < len(patient_ids) else f"patient_{batch_idx}_{i}",
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'dice_score': dice,
                    'mean_probability': float(np.mean(prob)),
                    'std_probability': float(np.std(prob)),
                    'prediction_coverage': float(np.mean(pred))
                })
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_ground_truths = np.array(all_ground_truths)
        
        # Store for outlier analysis
        self.predictions = all_predictions
        self.probabilities = all_probabilities
        self.ground_truths = all_ground_truths
        self.dice_scores = all_dice_scores
        self.sample_metadata = sample_metadata
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_predictions, all_probabilities, all_ground_truths)
        
        # Identify outliers
        outliers = self._identify_outliers(sample_metadata, all_dice_scores)
        
        # Generate comprehensive report
        report = self._generate_evaluation_report(metrics, outliers, sample_metadata)
        
        if verbose:
            self._print_evaluation_summary(report)
        
        return report
    
    def _calculate_dice(self, prediction, ground_truth):
        """Calculate DICE coefficient."""
        intersection = np.sum(prediction * ground_truth)
        union = np.sum(prediction) + np.sum(ground_truth)
        
        if union == 0:
            return 1.0 if np.sum(prediction) == 0 else 0.0
        
        return 2.0 * intersection / union
    
    def _calculate_metrics(self, predictions, probabilities, ground_truths):
        """Calculate comprehensive metrics including ROC AUC."""
        metrics = {}
        
        # ROC AUC Score
        try:
            if len(np.unique(ground_truths)) > 1:  # Check if both classes present
                metrics['roc_auc'] = roc_auc_score(ground_truths, probabilities)
                
                # Calculate ROC curve
                fpr, tpr, thresholds = roc_curve(ground_truths, probabilities)
                metrics['roc_curve'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': thresholds.tolist()
                }
                
                # Find optimal threshold (Youden's J statistic)
                j_scores = tpr - fpr
                optimal_idx = np.argmax(j_scores)
                metrics['optimal_threshold'] = float(thresholds[optimal_idx])
                metrics['optimal_tpr'] = float(tpr[optimal_idx])
                metrics['optimal_fpr'] = float(fpr[optimal_idx])
            else:
                metrics['roc_auc'] = None
                metrics['roc_curve'] = None
                metrics['warning'] = "Only one class present in ground truth"
        except Exception as e:
            metrics['roc_auc'] = None
            metrics['roc_auc_error'] = str(e)
        
        # Precision-Recall metrics
        try:
            precision, recall, pr_thresholds = precision_recall_curve(ground_truths, probabilities)
            metrics['precision_recall'] = {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': pr_thresholds.tolist()
            }
        except:
            metrics['precision_recall'] = None
        
        # Confusion matrix
        cm = confusion_matrix(ground_truths, predictions)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification metrics
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) \
                              if (metrics['precision'] + metrics['recall']) > 0 else 0
        
        # DICE score statistics
        if self.dice_scores:
            metrics['dice_mean'] = float(np.mean(self.dice_scores))
            metrics['dice_std'] = float(np.std(self.dice_scores))
            metrics['dice_median'] = float(np.median(self.dice_scores))
            metrics['dice_min'] = float(np.min(self.dice_scores))
            metrics['dice_max'] = float(np.max(self.dice_scores))
        
        return metrics
    
    def _identify_outliers(self, sample_metadata, dice_scores):
        """
        Identify outliers using multiple methods.
        """
        outliers = {
            'statistical': [],
            'performance': [],
            'uncertainty': []
        }
        
        if not dice_scores:
            return outliers
        
        # 1. Statistical outliers (Z-score method)
        z_scores = stats.zscore(dice_scores)
        statistical_outlier_indices = np.where(np.abs(z_scores) > 3)[0]
        
        # 2. Performance outliers (bottom 5% DICE scores)
        threshold_5pct = np.percentile(dice_scores, 5)
        performance_outlier_indices = np.where(np.array(dice_scores) < threshold_5pct)[0]
        
        # 3. Uncertainty outliers (high prediction variance)
        uncertainty_scores = [m['std_probability'] for m in sample_metadata]
        if uncertainty_scores:
            uncertainty_threshold = np.percentile(uncertainty_scores, 95)
            uncertainty_outlier_indices = [i for i, m in enumerate(sample_metadata) 
                                          if m['std_probability'] > uncertainty_threshold]
        else:
            uncertainty_outlier_indices = []
        
        # Compile outlier information
        for idx in statistical_outlier_indices:
            if idx < len(sample_metadata):
                outliers['statistical'].append({
                    'index': int(idx),
                    'patient_id': sample_metadata[idx]['patient_id'],
                    'dice_score': sample_metadata[idx]['dice_score'],
                    'z_score': float(z_scores[idx])
                })
        
        for idx in performance_outlier_indices:
            if idx < len(sample_metadata):
                outliers['performance'].append({
                    'index': int(idx),
                    'patient_id': sample_metadata[idx]['patient_id'],
                    'dice_score': sample_metadata[idx]['dice_score'],
                    'threshold': float(threshold_5pct)
                })
        
        for idx in uncertainty_outlier_indices:
            if idx < len(sample_metadata):
                outliers['uncertainty'].append({
                    'index': int(idx),
                    'patient_id': sample_metadata[idx]['patient_id'],
                    'std_probability': sample_metadata[idx]['std_probability'],
                    'threshold': float(uncertainty_threshold)
                })
        
        return outliers
    
    def _generate_evaluation_report(self, metrics, outliers, sample_metadata):
        """Generate comprehensive evaluation report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'outliers': outliers,
            'summary_statistics': {
                'total_samples': len(sample_metadata),
                'total_outliers': {
                    'statistical': len(outliers['statistical']),
                    'performance': len(outliers['performance']),
                    'uncertainty': len(outliers['uncertainty'])
                }
            },
            'recommendations': self._generate_recommendations(metrics, outliers)
        }
        
        return report
    
    def _generate_recommendations(self, metrics, outliers):
        """
        Generate actionable recommendations based on evaluation results.
        """
        recommendations = []
        
        # ROC AUC based recommendations
        if metrics.get('roc_auc'):
            if metrics['roc_auc'] < 0.7:
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Model Performance',
                    'issue': f"Low ROC AUC score ({metrics['roc_auc']:.3f})",
                    'recommendations': [
                        "Increase model capacity (more channels or deeper architecture)",
                        "Add more diverse training data",
                        "Implement better data augmentation strategies",
                        "Consider using pre-trained weights from larger datasets"
                    ]
                })
            elif metrics['roc_auc'] < 0.85:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'category': 'Model Performance',
                    'issue': f"Moderate ROC AUC score ({metrics['roc_auc']:.3f})",
                    'recommendations': [
                        "Fine-tune learning rate schedule",
                        "Implement focal loss for hard examples",
                        "Add attention mechanisms to the model",
                        "Use ensemble methods"
                    ]
                })
        
        # Threshold optimization
        if metrics.get('optimal_threshold') and abs(metrics['optimal_threshold'] - 0.5) > 0.1:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Threshold Optimization',
                'issue': f"Optimal threshold ({metrics['optimal_threshold']:.3f}) differs from default (0.5)",
                'recommendations': [
                    f"Update prediction threshold to {metrics['optimal_threshold']:.3f}",
                    "Implement adaptive thresholding based on image characteristics",
                    "Consider class-specific thresholds for multi-organ segmentation"
                ]
            })
        
        # Outlier handling
        total_outliers = sum(len(outliers[k]) for k in outliers)
        if total_outliers > 0:
            outlier_percentage = (total_outliers / len(self.dice_scores)) * 100 if self.dice_scores else 0
            
            if outlier_percentage > 10:
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Outlier Management',
                    'issue': f"High outlier rate ({outlier_percentage:.1f}%)",
                    'recommendations': [
                        "Implement outlier-robust loss functions (e.g., Huber loss)",
                        "Add hard example mining to training",
                        "Review and potentially remove corrupted training samples",
                        "Implement confidence-based rejection for predictions",
                        "Add test-time augmentation for uncertain cases"
                    ]
                })
            
            # Specific outlier type recommendations
            if len(outliers['uncertainty']) > 5:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'category': 'Uncertainty Handling',
                    'issue': f"High uncertainty in {len(outliers['uncertainty'])} samples",
                    'recommendations': [
                        "Implement Monte Carlo dropout for uncertainty estimation",
                        "Add ensemble uncertainty quantification",
                        "Use temperature scaling for probability calibration",
                        "Implement selective prediction with rejection option"
                    ]
                })
        
        # Class imbalance
        if metrics.get('confusion_matrix'):
            cm = np.array(metrics['confusion_matrix'])
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
                positive_ratio = (tp + fn) / (tn + fp + fn + tp) if (tn + fp + fn + tp) > 0 else 0
                
                if positive_ratio < 0.1 or positive_ratio > 0.9:
                    recommendations.append({
                        'priority': 'HIGH',
                        'category': 'Class Imbalance',
                        'issue': f"Severe class imbalance (positive ratio: {positive_ratio:.3f})",
                        'recommendations': [
                            "Implement weighted loss functions",
                            "Use SMOTE or other oversampling techniques",
                            "Apply class-balanced sampling during training",
                            "Consider focal loss or dice loss instead of cross-entropy"
                        ]
                    })
        
        # DICE score recommendations
        if metrics.get('dice_mean'):
            if metrics['dice_mean'] < 0.5:
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Segmentation Quality',
                    'issue': f"Low average DICE score ({metrics['dice_mean']:.3f})",
                    'recommendations': [
                        "Increase training iterations",
                        "Implement multi-scale training",
                        "Add skip connections if not present",
                        "Use boundary-aware loss functions",
                        "Implement cascaded refinement networks"
                    ]
                })
            
            if metrics.get('dice_std') and metrics['dice_std'] > 0.2:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'category': 'Consistency',
                    'issue': f"High DICE score variance (std: {metrics['dice_std']:.3f})",
                    'recommendations': [
                        "Implement batch normalization for stability",
                        "Use gradient clipping during training",
                        "Add regularization (dropout, weight decay)",
                        "Implement consistency regularization",
                        "Use exponential moving average of model weights"
                    ]
                })
        
        return recommendations
    
    def _print_evaluation_summary(self, report):
        """Print formatted evaluation summary."""
        print("\n" + "="*80)
        print("📊 EVALUATION SUMMARY")
        print("="*80)
        
        metrics = report['metrics']
        
        # ROC AUC
        if metrics.get('roc_auc') is not None:
            print(f"\n🎯 ROC AUC Score: {metrics['roc_auc']:.4f}")
            if metrics.get('optimal_threshold'):
                print(f"   Optimal Threshold: {metrics['optimal_threshold']:.3f}")
                print(f"   TPR at Optimal: {metrics['optimal_tpr']:.3f}")
                print(f"   FPR at Optimal: {metrics['optimal_fpr']:.3f}")
        else:
            print("\n⚠️  ROC AUC could not be calculated")
            if metrics.get('warning'):
                print(f"   Reason: {metrics['warning']}")
        
        # Classification metrics
        print(f"\n📈 Classification Metrics:")
        print(f"   Accuracy:    {metrics.get('accuracy', 0):.4f}")
        print(f"   Precision:   {metrics.get('precision', 0):.4f}")
        print(f"   Recall:      {metrics.get('recall', 0):.4f}")
        print(f"   Specificity: {metrics.get('specificity', 0):.4f}")
        print(f"   F1 Score:    {metrics.get('f1_score', 0):.4f}")
        
        # DICE scores
        if metrics.get('dice_mean') is not None:
            print(f"\n🎲 DICE Scores:")
            print(f"   Mean:   {metrics['dice_mean']:.4f} ± {metrics['dice_std']:.4f}")
            print(f"   Median: {metrics['dice_median']:.4f}")
            print(f"   Range:  [{metrics['dice_min']:.4f}, {metrics['dice_max']:.4f}]")
        
        # Outliers
        print(f"\n🔍 Outlier Analysis:")
        outliers = report['outliers']
        print(f"   Statistical outliers: {len(outliers['statistical'])}")
        print(f"   Performance outliers: {len(outliers['performance'])}")
        print(f"   Uncertainty outliers: {len(outliers['uncertainty'])}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        recommendations = report['recommendations']
        
        if not recommendations:
            print("   ✅ No critical issues detected. Model performing well!")
        else:
            for i, rec in enumerate(recommendations, 1):
                print(f"\n   {i}. [{rec['priority']}] {rec['category']}")
                print(f"      Issue: {rec['issue']}")
                print(f"      Recommendations:")
                for action in rec['recommendations']:
                    print(f"        • {action}")
        
        print("\n" + "="*80)
    
    def save_report(self, report, output_path='evaluation_report.json'):
        """Save evaluation report to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📁 Report saved to: {output_path}")
    
    def plot_roc_curve(self, report, save_path='roc_curve.png'):
        """Plot and save ROC curve."""
        if not report['metrics'].get('roc_curve'):
            print("⚠️  Cannot plot ROC curve - data not available")
            return
        
        plt.figure(figsize=(8, 6))
        
        fpr = report['metrics']['roc_curve']['fpr']
        tpr = report['metrics']['roc_curve']['tpr']
        auc = report['metrics']['roc_auc']
        
        plt.plot(fpr, tpr, 'b-', linewidth=2, 
                label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
        
        # Mark optimal point
        if report['metrics'].get('optimal_fpr'):
            opt_fpr = report['metrics']['optimal_fpr']
            opt_tpr = report['metrics']['optimal_tpr']
            plt.plot(opt_fpr, opt_tpr, 'go', markersize=10, 
                    label=f'Optimal threshold = {report["metrics"]["optimal_threshold"]:.3f}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 ROC curve saved to: {save_path}")


def create_synthetic_test_loader(num_samples=50, batch_size=4):
    """
    Create synthetic test data for demonstration.
    In production, this should load real medical data.
    """
    from torch.utils.data import Dataset, DataLoader
    
    class SyntheticDataset(Dataset):
        def __init__(self, num_samples):
            self.num_samples = num_samples
            
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Create synthetic data
            query_image = torch.randn(1, 32, 64, 64)
            reference_image = torch.randn(1, 32, 64, 64)
            
            # Create more realistic masks with some structure
            mask_base = torch.randn(1, 32, 64, 64)
            reference_mask = (mask_base > 0.5).float()
            
            # Query mask similar but with some differences
            noise = torch.randn_like(mask_base) * 0.3
            query_mask = ((mask_base + noise) > 0.5).float()
            
            return {
                'query_images': query_image,
                'support_images': reference_image,
                'support_masks': reference_mask,
                'query_masks': query_mask,
                'patient_ids': f'synthetic_{idx}'
            }
    
    dataset = SyntheticDataset(num_samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


def main():
    """
    Main evaluation pipeline.
    """
    print("🚀 Starting End-to-End Evaluation Pipeline")
    print("="*80)
    
    # Configuration
    config = {
        'in_channels': 1,
        'base_channels': 32,
        'embed_dim': 256,
        'num_tokens': 8,
        'num_classes': 1,
        'checkpoint_path': None,  # Path to trained model if available
        'output_dir': 'evaluation_results'
    }
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Initialize model
    print("\n📦 Loading IRIS model...")
    model = IRISModel(
        in_channels=config['in_channels'],
        base_channels=config['base_channels'],
        embed_dim=config['embed_dim'],
        num_tokens=config['num_tokens'],
        num_classes=config['num_classes']
    )
    
    # Load checkpoint if available
    if config['checkpoint_path'] and os.path.exists(config['checkpoint_path']):
        print(f"📂 Loading checkpoint from: {config['checkpoint_path']}")
        checkpoint = torch.load(config['checkpoint_path'], map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Checkpoint loaded successfully")
    else:
        print("⚠️  No checkpoint found - using untrained model for demonstration")
    
    # Create evaluator
    evaluator = PredictionEvaluator(model)
    
    # Create test data loader
    print("\n📊 Preparing test data...")
    # In production, replace this with real data loader
    test_loader = create_synthetic_test_loader(num_samples=100, batch_size=4)
    
    # Run evaluation
    print("\n🔬 Running comprehensive evaluation...")
    report = evaluator.evaluate_test_set(test_loader, verbose=True)
    
    # Save results
    report_path = os.path.join(config['output_dir'], 'evaluation_report.json')
    evaluator.save_report(report, report_path)
    
    # Plot ROC curve if available
    if report['metrics'].get('roc_curve'):
        roc_path = os.path.join(config['output_dir'], 'roc_curve.png')
        evaluator.plot_roc_curve(report, roc_path)
    
    print("\n✅ Evaluation complete!")
    print(f"📁 Results saved to: {config['output_dir']}")
    
    return report


if __name__ == "__main__":
    report = main()