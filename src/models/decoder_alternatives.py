"""
Alternative Decoder Architectures for IRIS Framework

This module explores different decoder architectures that can work
within the IRIS framework, addressing the channel mismatch issue
with various approaches:

1. FPN-style decoder with lateral connections
2. Progressive upsampling decoder
3. Dense skip connection decoder
4. Attention-based feature fusion decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPNDecoder(nn.Module):
    """
    Feature Pyramid Network (FPN) style decoder.
    
    Uses lateral connections and top-down pathway for feature fusion.
    Good for handling multi-scale features with different channel dimensions.
    """
    
    def __init__(self, encoder_channels, embed_dim=512, num_classes=1, feature_dim=256):
        super().__init__()
        
        self.encoder_channels = encoder_channels  # [32, 32, 64, 128, 256, 512]
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.feature_dim = feature_dim  # Unified feature dimension
        
        # Lateral connections: convert encoder features to unified dimension
        self.lateral_convs = nn.ModuleList()
        for channels in encoder_channels:
            self.lateral_convs.append(
                nn.Conv3d(channels, feature_dim, kernel_size=1, bias=False)
            )
        
        # Top-down pathway: smooth the upsampled features
        self.fpn_convs = nn.ModuleList()
        for _ in range(len(encoder_channels) - 1):  # Skip the highest resolution
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv3d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
                    nn.InstanceNorm3d(feature_dim),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Task attention modules for each scale
        self.task_attentions = nn.ModuleList()
        for _ in encoder_channels:
            self.task_attentions.append(
                nn.MultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=8,
                    dropout=0.1,
                    batch_first=True
                )
            )
        
        # Feature projection for attention
        self.feat_proj = nn.Conv3d(feature_dim, embed_dim, kernel_size=1)
        self.feat_unproj = nn.Conv3d(embed_dim, feature_dim, kernel_size=1)
        
        # Final prediction head
        self.prediction_head = nn.Sequential(
            nn.Conv3d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_dim, num_classes, kernel_size=1)
        )
        
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, encoder_features, task_embedding=None):
        """
        Forward pass of FPN decoder.
        
        Args:
            encoder_features: List of encoder features
            task_embedding: Task embeddings (B, num_tokens, embed_dim)
        
        Returns:
            Segmentation logits (B, num_classes, D, H, W)
        """
        # Apply lateral connections
        laterals = []
        for feat, lateral_conv in zip(encoder_features, self.lateral_convs):
            laterals.append(lateral_conv(feat))
        
        # Top-down pathway
        fpn_features = [laterals[-1]]  # Start from the deepest feature
        
        for i in range(len(laterals) - 2, -1, -1):
            # Upsample previous FPN feature
            upsampled = F.interpolate(
                fpn_features[-1], 
                size=laterals[i].shape[2:], 
                mode='trilinear', 
                align_corners=False
            )
            
            # Add lateral connection
            fpn_feat = upsampled + laterals[i]
            
            # Apply smoothing convolution
            if i < len(self.fpn_convs):
                fpn_feat = self.fpn_convs[len(laterals) - 2 - i](fpn_feat)
            
            fpn_features.append(fpn_feat)
        
        # Reverse to get features from high to low resolution
        fpn_features = fpn_features[::-1]
        
        # Apply task attention to the highest resolution feature
        final_feature = fpn_features[0]
        if task_embedding is not None:
            final_feature = self._apply_task_attention(final_feature, task_embedding)
        
        # Generate prediction
        output = self.prediction_head(final_feature)
        
        return output
    
    def _apply_task_attention(self, features, task_embedding):
        """Apply task attention to features."""
        B, C, D, H, W = features.shape
        
        # Project to embedding dimension
        feat_proj = self.feat_proj(features)
        feat_flat = feat_proj.view(B, self.embed_dim, -1).transpose(1, 2)
        
        # Apply attention
        attended_feat, _ = self.task_attentions[0](
            query=feat_flat,
            key=task_embedding,
            value=task_embedding
        )
        
        # Residual connection and layer norm
        attended_feat = self.layer_norm(attended_feat + feat_flat)
        
        # Project back
        attended_feat = attended_feat.transpose(1, 2).view(B, self.embed_dim, D, H, W)
        attended_feat = self.feat_unproj(attended_feat)
        
        return attended_feat


class ProgressiveDecoder(nn.Module):
    """
    Progressive upsampling decoder.
    
    Gradually upsamples features while maintaining consistent channel dimensions.
    Uses progressive refinement at each scale.
    """
    
    def __init__(self, encoder_channels, embed_dim=512, num_classes=1, base_dim=64):
        super().__init__()
        
        self.encoder_channels = encoder_channels
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.base_dim = base_dim
        
        # Input projection from bottleneck
        self.input_proj = nn.Conv3d(encoder_channels[-1], base_dim * 8, kernel_size=1)
        
        # Progressive upsampling blocks
        self.up_blocks = nn.ModuleList()
        dims = [base_dim * 8, base_dim * 4, base_dim * 2, base_dim, base_dim]
        
        for i in range(len(dims) - 1):
            self.up_blocks.append(
                self._make_up_block(dims[i], dims[i + 1], encoder_channels[-(i + 2)])
            )
        
        # Task attention
        self.task_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.feat_proj = nn.Conv3d(base_dim, embed_dim, kernel_size=1)
        self.feat_unproj = nn.Conv3d(embed_dim, base_dim, kernel_size=1)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Final output
        self.final_conv = nn.Conv3d(base_dim, num_classes, kernel_size=1)
    
    def _make_up_block(self, in_dim, out_dim, skip_dim):
        """Create an upsampling block."""
        return nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(in_dim, out_dim, kernel_size=1, bias=False),
            nn.Conv3d(skip_dim, out_dim, kernel_size=1, bias=False),
            nn.Sequential(
                nn.Conv3d(out_dim * 2, out_dim, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm3d(out_dim),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm3d(out_dim),
                nn.ReLU(inplace=True)
            )
        ])
    
    def forward(self, encoder_features, task_embedding=None):
        """Forward pass of progressive decoder."""
        # Start from bottleneck
        x = self.input_proj(encoder_features[-1])
        
        # Progressive upsampling
        for i, up_block in enumerate(self.up_blocks):
            upsample, up_conv, skip_conv, refine = up_block
            
            # Upsample
            x = upsample(x)
            x = up_conv(x)
            
            # Process skip connection
            skip_idx = len(encoder_features) - 2 - i
            skip = encoder_features[skip_idx]
            skip = skip_conv(skip)
            
            # Resize if needed
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='trilinear', align_corners=False)
            
            # Combine and refine
            x = torch.cat([x, skip], dim=1)
            x = refine(x)
        
        # Apply task attention
        if task_embedding is not None:
            x = self._apply_task_attention(x, task_embedding)
        
        # Final output
        output = self.final_conv(x)
        
        return output
    
    def _apply_task_attention(self, features, task_embedding):
        """Apply task attention."""
        B, C, D, H, W = features.shape
        
        feat_proj = self.feat_proj(features)
        feat_flat = feat_proj.view(B, self.embed_dim, -1).transpose(1, 2)
        
        attended_feat, _ = self.task_attention(
            query=feat_flat,
            key=task_embedding,
            value=task_embedding
        )
        
        attended_feat = self.layer_norm(attended_feat + feat_flat)
        attended_feat = attended_feat.transpose(1, 2).view(B, self.embed_dim, D, H, W)
        attended_feat = self.feat_unproj(attended_feat)
        
        return attended_feat


class DenseSkipDecoder(nn.Module):
    """
    Dense skip connection decoder.
    
    Uses dense connections between all encoder and decoder features,
    similar to DenseNet but for U-Net architecture.
    """
    
    def __init__(self, encoder_channels, embed_dim=512, num_classes=1, growth_rate=32):
        super().__init__()
        
        self.encoder_channels = encoder_channels
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.growth_rate = growth_rate
        
        # Dense blocks for each decoder stage
        self.dense_blocks = nn.ModuleList()
        
        # Calculate input channels for each dense block
        current_channels = encoder_channels[-1]  # Start from bottleneck
        
        for i in range(len(encoder_channels) - 1):
            # Add skip connection channels
            skip_idx = len(encoder_channels) - 2 - i
            skip_channels = encoder_channels[skip_idx]
            
            dense_block = self._make_dense_block(
                current_channels + skip_channels, 
                growth_rate, 
                num_layers=4
            )
            self.dense_blocks.append(dense_block)
            
            # Update current channels
            current_channels = current_channels + skip_channels + growth_rate * 4
            
            # Add transition layer (upsampling + channel reduction)
            if i < len(encoder_channels) - 2:
                transition = nn.Sequential(
                    nn.Conv3d(current_channels, current_channels // 2, kernel_size=1, bias=False),
                    nn.InstanceNorm3d(current_channels // 2),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
                )
                self.dense_blocks.append(transition)
                current_channels = current_channels // 2
        
        # Task attention
        self.task_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.feat_proj = nn.Conv3d(current_channels, embed_dim, kernel_size=1)
        self.feat_unproj = nn.Conv3d(embed_dim, current_channels, kernel_size=1)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Final classifier
        self.classifier = nn.Conv3d(current_channels, num_classes, kernel_size=1)
    
    def _make_dense_block(self, in_channels, growth_rate, num_layers):
        """Create a dense block."""
        layers = []
        for i in range(num_layers):
            layers.append(
                nn.Sequential(
                    nn.InstanceNorm3d(in_channels + i * growth_rate),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(in_channels + i * growth_rate, growth_rate, 
                             kernel_size=3, padding=1, bias=False)
                )
            )
        return nn.ModuleList(layers)
    
    def forward(self, encoder_features, task_embedding=None):
        """Forward pass of dense skip decoder."""
        x = encoder_features[-1]  # Start from bottleneck
        
        block_idx = 0
        for i in range(len(encoder_features) - 1):
            # Get skip connection
            skip_idx = len(encoder_features) - 2 - i
            skip = encoder_features[skip_idx]
            
            # Resize skip if needed
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='trilinear', align_corners=False)
            
            # Concatenate with skip
            x = torch.cat([x, skip], dim=1)
            
            # Apply dense block
            dense_block = self.dense_blocks[block_idx]
            block_idx += 1
            
            dense_features = [x]
            for layer in dense_block:
                new_features = layer(torch.cat(dense_features, dim=1))
                dense_features.append(new_features)
            
            x = torch.cat(dense_features, dim=1)
            
            # Apply transition if not last stage
            if i < len(encoder_features) - 2:
                transition = self.dense_blocks[block_idx]
                block_idx += 1
                x = transition(x)
        
        # Apply task attention
        if task_embedding is not None:
            x = self._apply_task_attention(x, task_embedding)
        
        # Final classification
        output = self.classifier(x)
        
        return output
    
    def _apply_task_attention(self, features, task_embedding):
        """Apply task attention."""
        B, C, D, H, W = features.shape
        
        feat_proj = self.feat_proj(features)
        feat_flat = feat_proj.view(B, self.embed_dim, -1).transpose(1, 2)
        
        attended_feat, _ = self.task_attention(
            query=feat_flat,
            key=task_embedding,
            value=task_embedding
        )
        
        attended_feat = self.layer_norm(attended_feat + feat_flat)
        attended_feat = attended_feat.transpose(1, 2).view(B, self.embed_dim, D, H, W)
        attended_feat = self.feat_unproj(attended_feat)
        
        return attended_feat


def test_alternative_decoders():
    """Test all alternative decoder architectures."""
    print("Testing Alternative Decoder Architectures...")
    
    # Test parameters
    batch_size = 1
    embed_dim = 512
    num_classes = 1
    num_tokens = 10
    
    # Encoder configuration
    encoder_channels = [32, 32, 64, 128, 256, 512]
    
    # Create mock encoder features
    base_depth, base_height, base_width = 32, 64, 64  # Smaller for testing
    encoder_features = []
    
    spatial_scales = [1.0, 1.0, 0.5, 0.25, 0.125, 0.0625]
    
    for i, (channels, scale) in enumerate(zip(encoder_channels, spatial_scales)):
        d = int(base_depth * scale)
        h = int(base_height * scale)
        w = int(base_width * scale)
        feat = torch.randn(batch_size, channels, d, h, w)
        encoder_features.append(feat)
    
    task_embedding = torch.randn(batch_size, num_tokens, embed_dim)
    expected_shape = (batch_size, num_classes, base_depth, base_height, base_width)
    
    # Test decoders
    decoders = {
        'FPN': FPNDecoder(encoder_channels, embed_dim, num_classes),
        'Progressive': ProgressiveDecoder(encoder_channels, embed_dim, num_classes),
        'DenseSkip': DenseSkipDecoder(encoder_channels, embed_dim, num_classes)
    }
    
    results = {}
    
    for name, decoder in decoders.items():
        print(f"\n=== Testing {name} Decoder ===")
        
        try:
            param_count = sum(p.numel() for p in decoder.parameters())
            print(f"Parameters: {param_count:,}")
            
            with torch.no_grad():
                output = decoder(encoder_features, task_embedding)
            
            print(f"Output shape: {output.shape}")
            
            if output.shape == expected_shape:
                print(f"✅ {name} decoder works correctly!")
                results[name] = True
            else:
                print(f"❌ {name} decoder shape mismatch")
                results[name] = False
                
        except Exception as e:
            print(f"❌ {name} decoder failed: {e}")
            results[name] = False
    
    # Summary
    print(f"\n=== Results Summary ===")
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name} Decoder: {status}")
    
    successful_decoders = [name for name, success in results.items() if success]
    print(f"\nSuccessful decoders: {successful_decoders}")
    
    return results


if __name__ == "__main__":
    test_alternative_decoders()
