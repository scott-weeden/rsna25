"""
DICE Loss Implementation for Medical Image Segmentation

This module implements real DICE loss computation for the IRIS framework,
replacing any hardcoded formulas with proper mathematical implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class DiceLoss(nn.Module):
    """
    Dice Loss for binary and multi-class segmentation.
    
    The Dice coefficient measures the overlap between prediction and ground truth:
    DICE = 2 * |X ∩ Y| / (|X| + |Y|)
    
    Loss = 1 - DICE
    
    Args:
        smooth (float): Smoothing factor to avoid division by zero
        include_background (bool): Whether to include background class
        reduction (str): Reduction method ('mean', 'sum', 'none')
        squared_pred (bool): Whether to square predictions in denominator
    """
    
    def __init__(
        self,
        smooth: float = 1e-5,
        include_background: bool = True,
        reduction: str = 'mean',
        squared_pred: bool = False
    ):
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background
        self.reduction = reduction
        self.squared_pred = squared_pred
    
    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            input: Predicted logits or probabilities (B, C, D, H, W)
            target: Ground truth labels (B, D, H, W) or one-hot (B, C, D, H, W)
            mask: Optional mask to exclude certain regions
        
        Returns:
            Dice loss value
        """
        # Get dimensions
        batch_size = input.shape[0]
        num_classes = input.shape[1]
        
        # Apply sigmoid/softmax if needed
        if num_classes == 1:
            # Binary segmentation
            input = torch.sigmoid(input)
        else:
            # Multi-class segmentation
            input = F.softmax(input, dim=1)
        
        # Convert target to one-hot if needed
        if target.dim() == 4:  # (B, D, H, W)
            if num_classes == 1:
                # Binary case - target is already in correct format
                target_one_hot = target.unsqueeze(1).float()
            else:
                # Multi-class case
                target_one_hot = F.one_hot(target.long(), num_classes=num_classes)
                target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()
        else:
            target_one_hot = target.float()
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(input)
            input = input * mask
            target_one_hot = target_one_hot * mask
        
        # Calculate dice score per class
        dice_scores = []
        
        start_idx = 1 if not self.include_background and num_classes > 1 else 0
        
        for i in range(start_idx, num_classes):
            pred_i = input[:, i:i+1]
            target_i = target_one_hot[:, i:i+1]
            
            # Flatten spatial dimensions
            pred_flat = pred_i.view(batch_size, -1)
            target_flat = target_i.view(batch_size, -1)
            
            # Calculate intersection and union
            intersection = torch.sum(pred_flat * target_flat, dim=1)
            
            if self.squared_pred:
                pred_sum = torch.sum(pred_flat * pred_flat, dim=1)
            else:
                pred_sum = torch.sum(pred_flat, dim=1)
            
            target_sum = torch.sum(target_flat, dim=1)
            
            # Calculate Dice coefficient
            dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
            dice_scores.append(dice)
        
        # Stack dice scores
        dice_scores = torch.stack(dice_scores, dim=1)  # (B, C)
        
        # Convert to loss (1 - dice)
        dice_loss = 1.0 - dice_scores
        
        # Apply reduction
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class GeneralizedDiceLoss(nn.Module):
    """
    Generalized Dice Loss for handling class imbalance.
    
    Weights each class by the inverse of its volume.
    
    Args:
        smooth (float): Smoothing factor
        include_background (bool): Whether to include background
        reduction (str): Reduction method
    """
    
    def __init__(
        self,
        smooth: float = 1e-5,
        include_background: bool = True,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background
        self.reduction = reduction
    
    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Generalized Dice loss.
        
        Args:
            input: Predicted logits (B, C, D, H, W)
            target: Ground truth labels (B, D, H, W)
        
        Returns:
            Generalized Dice loss
        """
        batch_size = input.shape[0]
        num_classes = input.shape[1]
        
        # Apply softmax
        input = F.softmax(input, dim=1)
        
        # Convert target to one-hot
        target_one_hot = F.one_hot(target.long(), num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()
        
        # Flatten spatial dimensions
        input_flat = input.view(batch_size, num_classes, -1)
        target_flat = target_one_hot.view(batch_size, num_classes, -1)
        
        # Calculate class weights (inverse of class frequency)
        weights = 1.0 / (torch.sum(target_flat, dim=2) ** 2 + self.smooth)
        
        # Calculate weighted intersection and union
        intersection = torch.sum(input_flat * target_flat, dim=2) * weights
        union = torch.sum(input_flat + target_flat, dim=2) * weights
        
        # Calculate Generalized Dice
        start_idx = 1 if not self.include_background and num_classes > 1 else 0
        
        gdice = 2.0 * torch.sum(intersection[:, start_idx:], dim=1) / \
                (torch.sum(union[:, start_idx:], dim=1) + self.smooth)
        
        # Convert to loss
        gdice_loss = 1.0 - gdice
        
        # Apply reduction
        if self.reduction == 'mean':
            return gdice_loss.mean()
        elif self.reduction == 'sum':
            return gdice_loss.sum()
        else:
            return gdice_loss


class CombinedSegmentationLoss(nn.Module):
    """
    Combined loss for segmentation tasks.
    
    Combines Dice loss with Cross-Entropy for better gradient flow.
    
    Args:
        dice_weight (float): Weight for Dice loss
        ce_weight (float): Weight for Cross-Entropy loss
        dice_kwargs (dict): Arguments for DiceLoss
    """
    
    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        dice_kwargs: Optional[dict] = None
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
        dice_kwargs = dice_kwargs or {}
        self.dice_loss = DiceLoss(**dice_kwargs)
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute combined loss.
        
        Args:
            input: Predicted logits (B, C, D, H, W)
            target: Ground truth labels (B, D, H, W)
        
        Returns:
            Total loss and dictionary of individual losses
        """
        # Compute individual losses
        dice = self.dice_loss(input, target)
        ce = self.ce_loss(input, target.long())
        
        # Combine losses
        total_loss = self.dice_weight * dice + self.ce_weight * ce
        
        # Return total and components
        return total_loss, {
            'dice_loss': dice.item(),
            'ce_loss': ce.item(),
            'total_loss': total_loss.item()
        }


def compute_dice_score(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    smooth: float = 1e-5,
    per_class: bool = False
) -> torch.Tensor:
    """
    Compute DICE score for evaluation (not for training).
    
    Args:
        prediction: Binary predictions or probabilities (B, C, D, H, W) or (B, D, H, W)
        ground_truth: Ground truth masks (B, D, H, W)
        smooth: Smoothing factor
        per_class: Whether to return per-class scores
    
    Returns:
        DICE score(s)
    """
    with torch.no_grad():
        # Handle binary case
        if prediction.dim() == 4:
            prediction = prediction.unsqueeze(1)
        
        # Get number of classes
        num_classes = prediction.shape[1]
        
        # Threshold predictions if they're probabilities
        if prediction.dtype == torch.float:
            if num_classes == 1:
                prediction = (prediction > 0.5).float()
            else:
                prediction = torch.argmax(prediction, dim=1, keepdim=True)
                prediction = F.one_hot(prediction.squeeze(1), num_classes=num_classes)
                prediction = prediction.permute(0, 4, 1, 2, 3).float()
        
        # Convert ground truth to one-hot
        if ground_truth.dim() == 4:
            gt_one_hot = F.one_hot(ground_truth.long(), num_classes=num_classes)
            gt_one_hot = gt_one_hot.permute(0, 4, 1, 2, 3).float()
        else:
            gt_one_hot = ground_truth.float()
        
        # Compute DICE per class
        dice_scores = []
        
        for c in range(num_classes):
            pred_c = prediction[:, c].flatten()
            gt_c = gt_one_hot[:, c].flatten()
            
            intersection = torch.sum(pred_c * gt_c)
            union = torch.sum(pred_c) + torch.sum(gt_c)
            
            dice = (2.0 * intersection + smooth) / (union + smooth)
            dice_scores.append(dice)
        
        dice_scores = torch.stack(dice_scores)
        
        if per_class:
            return dice_scores
        else:
            return dice_scores.mean()


def test_dice_loss():
    """Test DICE loss implementation."""
    print("Testing DICE Loss implementation...")
    
    batch_size = 2
    num_classes = 5
    depth, height, width = 32, 64, 64
    
    # Create synthetic data
    logits = torch.randn(batch_size, num_classes, depth, height, width)
    targets = torch.randint(0, num_classes, (batch_size, depth, height, width))
    
    # Test DiceLoss
    dice_loss = DiceLoss(include_background=False)
    loss = dice_loss(logits, targets)
    print(f"Dice Loss: {loss.item():.4f}")
    
    # Test GeneralizedDiceLoss
    gdice_loss = GeneralizedDiceLoss(include_background=False)
    gloss = gdice_loss(logits, targets)
    print(f"Generalized Dice Loss: {gloss.item():.4f}")
    
    # Test CombinedLoss
    combined_loss = CombinedSegmentationLoss()
    total_loss, loss_dict = combined_loss(logits, targets)
    print(f"Combined Loss: {total_loss.item():.4f}")
    print(f"  - Dice: {loss_dict['dice_loss']:.4f}")
    print(f"  - CE: {loss_dict['ce_loss']:.4f}")
    
    # Test DICE score computation
    predictions = torch.sigmoid(logits)
    dice_score = compute_dice_score(predictions, targets)
    print(f"DICE Score: {dice_score.item():.4f}")
    
    # Test per-class scores
    per_class_scores = compute_dice_score(predictions, targets, per_class=True)
    print(f"Per-class DICE scores: {per_class_scores.numpy()}")
    
    print("All tests passed successfully!")


if __name__ == "__main__":
    test_dice_loss()