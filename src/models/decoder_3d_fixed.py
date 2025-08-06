"""
Fixed Query-Based Decoder for IRIS Framework

This module fixes the channel mismatch issue by properly aligning
encoder and decoder channels with correct skip connections.

Key fixes:
1. Proper skip connection channel mapping
2. Correct decoder channel progression
3. Symmetric U-Net architecture
4. Task embedding integration at each scale
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskGuidedBlock3D(nn.Module):
    """
    Fixed decoder block that integrates task embeddings via cross-attention.
    
    Args:
        in_channels (int): Number of input channels from previous decoder stage
        skip_channels (int): Number of channels from encoder skip connection
        out_channels (int): Number of output channels
        embed_dim (int): Dimension of task embeddings
        num_heads (int): Number of attention heads (default: 8)
    """
    
    def __init__(self, in_channels, skip_channels, out_channels, embed_dim, num_heads=8):
        super().__init__()
        
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        
        # Upsampling (if needed)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        
        # Channel adjustment for upsampled features
        self.up_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        
        # Channel adjustment for skip connection
        if skip_channels != out_channels:
            self.skip_conv = nn.Conv3d(skip_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.skip_conv = nn.Identity()
        
        # Feature processing after combining upsampled and skip
        combined_channels = out_channels * 2  # upsampled + skip (both adjusted to out_channels)
        self.pre_attention = nn.Sequential(
            nn.Conv3d(combined_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Project spatial features to embedding dimension for attention
        self.feature_proj = nn.Conv3d(out_channels, embed_dim, kernel_size=1)
        
        # Cross-attention between spatial features and task embeddings
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Project back to feature space
        self.feature_unproj = nn.Conv3d(embed_dim, out_channels, kernel_size=1)
        
        # Final processing
        self.post_attention = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Layer normalization for attention
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x, skip, task_embedding, upsample=True):
        """
        Forward pass of task-guided decoder block.
        
        Args:
            x: Input features from previous decoder stage (B, C_in, D, H, W)
            skip: Skip connection from encoder (B, C_skip, D_skip, H_skip, W_skip)
            task_embedding: Task embeddings (B, num_tokens, embed_dim)
            upsample: Whether to upsample input (default: True)
        
        Returns:
            Output features (B, C_out, D_out, H_out, W_out)
        """
        # Upsample input features if needed
        if upsample:
            x_up = self.upsample(x)
        else:
            x_up = x
        
        # Adjust channels for upsampled features
        x_up = self.up_conv(x_up)
        
        # Process skip connection
        skip_processed = self.skip_conv(skip)
        
        # Ensure spatial dimensions match
        if x_up.shape[2:] != skip_processed.shape[2:]:
            # Resize skip to match upsampled features
            skip_processed = F.interpolate(
                skip_processed, 
                size=x_up.shape[2:], 
                mode='trilinear', 
                align_corners=False
            )
        
        # Combine upsampled and skip features
        combined = torch.cat([x_up, skip_processed], dim=1)
        features = self.pre_attention(combined)
        
        # Apply cross-attention with task embeddings
        if task_embedding is not None:
            features = self._apply_task_attention(features, task_embedding)
        
        # Final processing
        output = self.post_attention(features)
        
        return output
    
    def _apply_task_attention(self, features, task_embedding):
        """
        Apply cross-attention between spatial features and task embeddings.
        
        Args:
            features: Spatial features (B, C, D, H, W)
            task_embedding: Task embeddings (B, num_tokens, embed_dim)
        
        Returns:
            Attended features (B, C, D, H, W)
        """
        B, C, D, H, W = features.shape
        
        # Project features to embedding dimension
        feat_proj = self.feature_proj(features)  # (B, embed_dim, D, H, W)
        
        # Reshape for attention: (B, D*H*W, embed_dim)
        feat_flat = feat_proj.view(B, self.embed_dim, -1).transpose(1, 2)
        
        # Cross-attention: spatial features attend to task embeddings
        attended_feat, _ = self.cross_attention(
            query=feat_flat,
            key=task_embedding,
            value=task_embedding
        )
        
        # Apply layer normalization and residual connection
        attended_feat = self.layer_norm(attended_feat + feat_flat)
        
        # Reshape back to spatial dimensions
        attended_feat = attended_feat.transpose(1, 2).view(B, self.embed_dim, D, H, W)
        
        # Project back to feature space
        attended_feat = self.feature_unproj(attended_feat)
        
        return attended_feat


class QueryBasedDecoderFixed(nn.Module):
    """
    Fixed query-based decoder with proper channel alignment.
    
    This decoder fixes the channel mismatch issue by:
    1. Properly mapping encoder channels to decoder channels
    2. Correct skip connection indexing
    3. Symmetric U-Net architecture
    
    Args:
        encoder_channels (list): List of encoder channel dimensions [32, 32, 64, 128, 256, 512]
        embed_dim (int): Dimension of task embeddings (default: 512)
        num_classes (int): Number of output classes (default: 1)
        num_heads (int): Number of attention heads (default: 8)
    """
    
    def __init__(self, encoder_channels, embed_dim=512, num_classes=1, num_heads=8):
        super().__init__()
        
        self.encoder_channels = encoder_channels  # [32, 32, 64, 128, 256, 512]
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Create symmetric decoder channels
        # Encoder: [32, 32, 64, 128, 256, 512] (stages 0,1,2,3,4,5)
        # Decoder: [256, 128, 64, 32, 32] (reverse path: 4->3->2->1->0)
        self.decoder_channels = [
            encoder_channels[4],  # 256 (from stage 4)
            encoder_channels[3],  # 128 (from stage 3)
            encoder_channels[2],  # 64  (from stage 2)
            encoder_channels[1],  # 32  (from stage 1)
            encoder_channels[0],  # 32  (from stage 0)
        ]
        
        print(f"Encoder channels: {encoder_channels}")
        print(f"Decoder channels: {self.decoder_channels}")
        
        # Decoder blocks with proper skip connections
        self.decoder_blocks = nn.ModuleList()
        
        # Block 0: 512 -> 256, skip from stage 4 (256 channels)
        self.decoder_blocks.append(
            TaskGuidedBlock3D(
                in_channels=encoder_channels[5],    # 512 (bottleneck)
                skip_channels=encoder_channels[4],  # 256 (stage 4)
                out_channels=self.decoder_channels[0],  # 256
                embed_dim=embed_dim,
                num_heads=num_heads
            )
        )
        
        # Block 1: 256 -> 128, skip from stage 3 (128 channels)
        self.decoder_blocks.append(
            TaskGuidedBlock3D(
                in_channels=self.decoder_channels[0],  # 256
                skip_channels=encoder_channels[3],     # 128 (stage 3)
                out_channels=self.decoder_channels[1], # 128
                embed_dim=embed_dim,
                num_heads=num_heads
            )
        )
        
        # Block 2: 128 -> 64, skip from stage 2 (64 channels)
        self.decoder_blocks.append(
            TaskGuidedBlock3D(
                in_channels=self.decoder_channels[1],  # 128
                skip_channels=encoder_channels[2],     # 64 (stage 2)
                out_channels=self.decoder_channels[2], # 64
                embed_dim=embed_dim,
                num_heads=num_heads
            )
        )
        
        # Block 3: 64 -> 32, skip from stage 1 (32 channels)
        self.decoder_blocks.append(
            TaskGuidedBlock3D(
                in_channels=self.decoder_channels[2],  # 64
                skip_channels=encoder_channels[1],     # 32 (stage 1)
                out_channels=self.decoder_channels[3], # 32
                embed_dim=embed_dim,
                num_heads=num_heads
            )
        )
        
        # Block 4: 32 -> 32, skip from stage 0 (32 channels)
        self.decoder_blocks.append(
            TaskGuidedBlock3D(
                in_channels=self.decoder_channels[3],  # 32
                skip_channels=encoder_channels[0],     # 32 (stage 0)
                out_channels=self.decoder_channels[4], # 32
                embed_dim=embed_dim,
                num_heads=num_heads
            )
        )
        
        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv3d(self.decoder_channels[-1], self.decoder_channels[-1], 
                     kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(self.decoder_channels[-1]),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.decoder_channels[-1], num_classes, kernel_size=1)
        )
        
        # Initialize weights
        self._init_weights()
        
        # Print architecture summary
        self._print_architecture()
    
    def _print_architecture(self):
        """Print decoder architecture for debugging."""
        print("\n=== Fixed Decoder Architecture ===")
        print("Skip connection mapping:")
        skip_stages = [4, 3, 2, 1, 0]  # Encoder stages used for skip connections
        for i, (block, skip_stage) in enumerate(zip(self.decoder_blocks, skip_stages)):
            print(f"  Block {i}: {block.in_channels} -> {block.out_channels}, "
                  f"skip from encoder stage {skip_stage} ({block.skip_channels} channels)")
        print(f"Final conv: {self.decoder_channels[-1]} -> {self.num_classes}")
        print("===================================\n")
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.InstanceNorm3d):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, encoder_features, task_embedding=None):
        """
        Forward pass of the fixed decoder.
        
        Args:
            encoder_features: List of encoder features [stage0, stage1, stage2, stage3, stage4, stage5]
                             Channels: [32, 32, 64, 128, 256, 512]
            task_embedding: Task embeddings (B, num_tokens, embed_dim) or None
        
        Returns:
            Segmentation logits (B, num_classes, D, H, W)
        """
        # Start from the bottleneck (stage 5: 512 channels)
        x = encoder_features[5]  # Bottleneck features
        
        # Skip connection stages in reverse order: [4, 3, 2, 1, 0]
        skip_stages = [4, 3, 2, 1, 0]
        
        # Process through decoder blocks
        for i, (decoder_block, skip_stage) in enumerate(zip(self.decoder_blocks, skip_stages)):
            skip = encoder_features[skip_stage]
            
            # Apply decoder block with task guidance
            x = decoder_block(x, skip, task_embedding, upsample=(i > 0))
            
            print(f"  Decoder block {i}: {x.shape} (skip from stage {skip_stage}: {skip.shape})")
        
        # Final output
        output = self.final_conv(x)
        
        return output


def test_fixed_decoder():
    """Test function to verify the fixed decoder works correctly."""
    print("Testing Fixed Query-Based Decoder...")
    
    # Test parameters
    batch_size = 2
    embed_dim = 512
    num_classes = 1
    num_tokens = 10
    
    # Encoder channel configuration (matching actual encoder)
    encoder_channels = [32, 32, 64, 128, 256, 512]
    
    # Create mock encoder features with correct shapes
    base_depth, base_height, base_width = 64, 128, 128
    encoder_features = []
    
    # Spatial scales for each encoder stage
    spatial_scales = [1.0, 1.0, 0.5, 0.25, 0.125, 0.0625]
    
    print("Creating mock encoder features:")
    for i, (channels, scale) in enumerate(zip(encoder_channels, spatial_scales)):
        d = int(base_depth * scale)
        h = int(base_height * scale)
        w = int(base_width * scale)
        feat = torch.randn(batch_size, channels, d, h, w)
        encoder_features.append(feat)
        print(f"  Stage {i}: {feat.shape} (channels: {channels}, scale: {scale})")
    
    # Create mock task embedding
    task_embedding = torch.randn(batch_size, num_tokens, embed_dim)
    print(f"Task embedding: {task_embedding.shape}")
    
    # Create fixed decoder
    print("\nCreating fixed decoder...")
    decoder = QueryBasedDecoderFixed(
        encoder_channels=encoder_channels,
        embed_dim=embed_dim,
        num_classes=num_classes,
        num_heads=8
    )
    
    param_count = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder parameters: {param_count:,}")
    
    # Forward pass
    print("\nTesting forward pass...")
    try:
        with torch.no_grad():
            output = decoder(encoder_features, task_embedding)
        
        expected_shape = (batch_size, num_classes, base_depth, base_height, base_width)
        print(f"Output shape: {output.shape} (expected: {expected_shape})")
        
        if output.shape == expected_shape:
            print("✅ Fixed decoder forward pass successful!")
        else:
            print(f"❌ Shape mismatch: got {output.shape}, expected {expected_shape}")
            return False
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    # Test without task embedding
    print("\nTesting without task embedding...")
    try:
        with torch.no_grad():
            output_no_task = decoder(encoder_features, task_embedding=None)
        
        assert output_no_task.shape == expected_shape, "Output shape mismatch without task embedding"
        print("✅ Works without task embedding")
        
    except Exception as e:
        print(f"❌ Failed without task embedding: {e}")
        return False
    
    # Test gradient flow
    print("\nTesting gradient flow...")
    try:
        encoder_features[0].requires_grad_(True)
        task_embedding.requires_grad_(True)
        
        output = decoder(encoder_features, task_embedding)
        loss = output.sum()
        loss.backward()
        
        assert encoder_features[0].grad is not None, "Gradients not flowing to encoder features"
        assert task_embedding.grad is not None, "Gradients not flowing to task embedding"
        print("✅ Gradients flow correctly")
        
    except Exception as e:
        print(f"❌ Gradient flow test failed: {e}")
        return False
    
    print("\n🎉 All fixed decoder tests passed!")
    print("✅ Channel mismatch issue resolved!")
    
    return decoder


if __name__ == "__main__":
    test_fixed_decoder()
