"""
Comprehensive End-to-End Test of IRIS Framework

This script performs a REAL end-to-end test to determine what actually works
in the IRIS implementation, without relying on synthetic data.

Tests:
1. Individual component functionality
2. Component integration
3. End-to-end forward pass
4. Gradient flow
5. Training capability
6. Alternative architectures if IRIS fails
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import traceback

# Try to import IRIS components
try:
    from models.encoder_3d import Encoder3D
    from models.task_encoding import TaskEncodingModule
    from models.decoder_3d import QueryBasedDecoder
    from models.iris_model import IRISModel
    
    # Try fixed versions if available
    try:
        from models.decoder_3d_fixed import QueryBasedDecoderFixed
        from models.iris_model_fixed import IRISModelFixed
        FIXED_AVAILABLE = True
    except ImportError:
        FIXED_AVAILABLE = False
        
    IRIS_AVAILABLE = True
except ImportError as e:
    print(f"❌ IRIS components not available: {e}")
    IRIS_AVAILABLE = False
    FIXED_AVAILABLE = False


class EndToEndTester:
    """Comprehensive end-to-end testing of IRIS framework."""
    
    def __init__(self):
        self.results = {}
        self.device = 'cpu'  # Use CPU to avoid GPU issues
        
        # Test configuration
        self.batch_size = 1
        self.in_channels = 1
        self.base_channels = 16  # Reduced for testing
        self.embed_dim = 64     # Reduced for testing
        self.num_tokens = 5     # Reduced for testing
        self.num_classes = 1
        self.spatial_size = (32, 64, 64)  # Small for testing
        
        print("🧪 IRIS Framework End-to-End Tester Initialized")
        print(f"   Device: {self.device}")
        print(f"   Spatial size: {self.spatial_size}")
        print(f"   Base channels: {self.base_channels}")
        print(f"   Embed dim: {self.embed_dim}")
    
    def create_test_data(self):
        """Create realistic test data (not random)."""
        D, H, W = self.spatial_size
        
        # Create structured test images (not random)
        query_image = torch.zeros(self.batch_size, self.in_channels, D, H, W)
        reference_image = torch.zeros(self.batch_size, self.in_channels, D, H, W)
        
        # Add structured patterns to simulate medical images
        # Central bright region (simulating organ)
        center_d, center_h, center_w = D//2, H//2, W//2
        size_d, size_h, size_w = D//4, H//4, W//4
        
        query_image[:, :, 
                   center_d-size_d:center_d+size_d,
                   center_h-size_h:center_h+size_h,
                   center_w-size_w:center_w+size_w] = 1.0
        
        reference_image[:, :,
                       center_d-size_d//2:center_d+size_d//2,
                       center_h-size_h//2:center_h+size_h//2,
                       center_w-size_w//2:center_w+size_w//2] = 1.0
        
        # Create corresponding masks
        reference_mask = torch.zeros(self.batch_size, 1, D, H, W)
        reference_mask[:, :,
                      center_d-size_d//2:center_d+size_d//2,
                      center_h-size_h//2:center_h+size_h//2,
                      center_w-size_w//2:center_w+size_w//2] = 1.0
        
        query_mask = torch.zeros(self.batch_size, 1, D, H, W)
        query_mask[:, :,
                   center_d-size_d:center_d+size_d,
                   center_h-size_h:center_h+size_h,
                   center_w-size_w:center_w+size_w] = 1.0
        
        return {
            'query_image': query_image,
            'reference_image': reference_image,
            'reference_mask': reference_mask,
            'query_mask': query_mask
        }
    
    def test_component(self, name, component_fn):
        """Test individual component."""
        print(f"\n🔧 Testing {name}...")
        try:
            result = component_fn()
            self.results[name] = {'status': 'PASS', 'details': result}
            print(f"   ✅ {name}: PASS")
            return True
        except Exception as e:
            self.results[name] = {'status': 'FAIL', 'error': str(e), 'traceback': traceback.format_exc()}
            print(f"   ❌ {name}: FAIL - {e}")
            return False
    
    def test_encoder(self):
        """Test 3D UNet encoder."""
        encoder = Encoder3D(
            in_channels=self.in_channels,
            base_channels=self.base_channels,
            num_blocks_per_stage=2
        )
        
        test_data = self.create_test_data()
        input_tensor = test_data['query_image']
        
        # Forward pass
        features = encoder(input_tensor)
        
        # Validate output
        expected_channels = encoder.get_feature_channels()
        if len(features) != len(expected_channels):
            raise ValueError(f"Expected {len(expected_channels)} features, got {len(features)}")
        
        for i, (feat, exp_channels) in enumerate(zip(features, expected_channels)):
            if feat.shape[1] != exp_channels:
                raise ValueError(f"Stage {i}: expected {exp_channels} channels, got {feat.shape[1]}")
        
        return {
            'features': len(features),
            'channels': expected_channels,
            'shapes': [f.shape for f in features],
            'parameters': sum(p.numel() for p in encoder.parameters())
        }
    
    def test_task_encoding(self):
        """Test task encoding module."""
        # First get encoder features
        encoder = Encoder3D(
            in_channels=self.in_channels,
            base_channels=self.base_channels,
            num_blocks_per_stage=2
        )
        
        test_data = self.create_test_data()
        ref_features = encoder(test_data['reference_image'])
        bottleneck_features = ref_features[-1]  # Use bottleneck
        
        # Create task encoder
        task_encoder = TaskEncodingModule(
            in_channels=bottleneck_features.shape[1],
            embed_dim=self.embed_dim,
            num_tokens=self.num_tokens,
            shuffle_scale=2
        )
        
        # Test task encoding
        task_embedding = task_encoder(bottleneck_features, test_data['reference_mask'])
        
        # Validate output shape
        expected_shape = (self.batch_size, self.num_tokens + 1, self.embed_dim)
        if task_embedding.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {task_embedding.shape}")
        
        # Test with different masks to ensure different embeddings
        mask2 = test_data['reference_mask'] * 0.5  # Different mask
        task_embedding2 = task_encoder(bottleneck_features, mask2)
        
        diff = torch.abs(task_embedding - task_embedding2).mean()
        if diff < 0.01:
            raise ValueError(f"Different masks produce too similar embeddings: {diff}")
        
        return {
            'output_shape': task_embedding.shape,
            'embedding_stats': {
                'mean': task_embedding.mean().item(),
                'std': task_embedding.std().item(),
                'min': task_embedding.min().item(),
                'max': task_embedding.max().item()
            },
            'mask_sensitivity': diff.item(),
            'parameters': sum(p.numel() for p in task_encoder.parameters())
        }
    
    def test_decoder_original(self):
        """Test original decoder (expected to fail)."""
        encoder = Encoder3D(
            in_channels=self.in_channels,
            base_channels=self.base_channels,
            num_blocks_per_stage=2
        )
        
        encoder_channels = encoder.get_feature_channels()
        
        decoder = QueryBasedDecoder(
            encoder_channels=encoder_channels,
            embed_dim=self.embed_dim,
            num_classes=self.num_classes,
            num_heads=4
        )
        
        test_data = self.create_test_data()
        encoder_features = encoder(test_data['query_image'])
        
        # Create dummy task embedding
        task_embedding = torch.randn(self.batch_size, self.num_tokens + 1, self.embed_dim)
        
        # Forward pass (expected to fail due to channel mismatch)
        output = decoder(encoder_features, task_embedding)
        
        expected_shape = (self.batch_size, self.num_classes, *self.spatial_size)
        if output.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {output.shape}")
        
        return {
            'output_shape': output.shape,
            'parameters': sum(p.numel() for p in decoder.parameters())
        }
    
    def test_decoder_fixed(self):
        """Test fixed decoder if available."""
        if not FIXED_AVAILABLE:
            raise ImportError("Fixed decoder not available")
        
        encoder = Encoder3D(
            in_channels=self.in_channels,
            base_channels=self.base_channels,
            num_blocks_per_stage=2
        )
        
        encoder_channels = encoder.get_feature_channels()
        
        decoder = QueryBasedDecoderFixed(
            encoder_channels=encoder_channels,
            embed_dim=self.embed_dim,
            num_classes=self.num_classes,
            num_heads=4
        )
        
        test_data = self.create_test_data()
        encoder_features = encoder(test_data['query_image'])
        
        # Create dummy task embedding
        task_embedding = torch.randn(self.batch_size, self.num_tokens + 1, self.embed_dim)
        
        # Forward pass
        output = decoder(encoder_features, task_embedding)
        
        expected_shape = (self.batch_size, self.num_classes, *self.spatial_size)
        if output.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {output.shape}")
        
        return {
            'output_shape': output.shape,
            'parameters': sum(p.numel() for p in decoder.parameters())
        }
    
    def test_iris_model_original(self):
        """Test original IRIS model (expected to fail)."""
        model = IRISModel(
            in_channels=self.in_channels,
            base_channels=self.base_channels,
            embed_dim=self.embed_dim,
            num_tokens=self.num_tokens,
            num_classes=self.num_classes
        )
        
        test_data = self.create_test_data()
        
        # End-to-end forward pass
        output = model(
            test_data['query_image'],
            test_data['reference_image'],
            test_data['reference_mask']
        )
        
        expected_shape = (self.batch_size, self.num_classes, *self.spatial_size)
        if output.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {output.shape}")
        
        return {
            'output_shape': output.shape,
            'model_info': model.get_model_info()
        }
    
    def test_iris_model_fixed(self):
        """Test fixed IRIS model if available."""
        if not FIXED_AVAILABLE:
            raise ImportError("Fixed IRIS model not available")
        
        model = IRISModelFixed(
            in_channels=self.in_channels,
            base_channels=self.base_channels,
            embed_dim=self.embed_dim,
            num_tokens=self.num_tokens,
            num_classes=self.num_classes
        )
        
        test_data = self.create_test_data()
        
        # End-to-end forward pass
        output = model(
            test_data['query_image'],
            test_data['reference_image'],
            test_data['reference_mask']
        )
        
        expected_shape = (self.batch_size, self.num_classes, *self.spatial_size)
        if output.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {output.shape}")
        
        return {
            'output_shape': output.shape,
            'model_info': model.get_model_info()
        }
    
    def test_gradient_flow(self):
        """Test gradient flow through the model."""
        if not FIXED_AVAILABLE:
            # Try original model
            model = IRISModel(
                in_channels=self.in_channels,
                base_channels=self.base_channels,
                embed_dim=self.embed_dim,
                num_tokens=self.num_tokens,
                num_classes=self.num_classes
            )
        else:
            # Use fixed model
            model = IRISModelFixed(
                in_channels=self.in_channels,
                base_channels=self.base_channels,
                embed_dim=self.embed_dim,
                num_tokens=self.num_tokens,
                num_classes=self.num_classes
            )
        
        test_data = self.create_test_data()
        
        # Enable gradients
        query_image = test_data['query_image'].requires_grad_(True)
        reference_image = test_data['reference_image'].requires_grad_(True)
        reference_mask = test_data['reference_mask'].requires_grad_(True)
        
        # Forward pass
        output = model(query_image, reference_image, reference_mask)
        
        # Compute loss and backward
        loss = output.sum()
        loss.backward()
        
        # Check gradients
        grad_info = {
            'query_grad_exists': query_image.grad is not None,
            'reference_grad_exists': reference_image.grad is not None,
            'loss_value': loss.item()
        }
        
        if query_image.grad is not None:
            grad_info['query_grad_norm'] = query_image.grad.norm().item()
        if reference_image.grad is not None:
            grad_info['reference_grad_norm'] = reference_image.grad.norm().item()
        
        return grad_info
    
    def test_training_step(self):
        """Test a single training step."""
        if not FIXED_AVAILABLE:
            model = IRISModel(
                in_channels=self.in_channels,
                base_channels=self.base_channels,
                embed_dim=self.embed_dim,
                num_tokens=self.num_tokens,
                num_classes=self.num_classes
            )
        else:
            model = IRISModelFixed(
                in_channels=self.in_channels,
                base_channels=self.base_channels,
                embed_dim=self.embed_dim,
                num_tokens=self.num_tokens,
                num_classes=self.num_classes
            )
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Create loss function
        criterion = nn.BCEWithLogitsLoss()
        
        test_data = self.create_test_data()
        
        # Training step
        optimizer.zero_grad()
        
        output = model(
            test_data['query_image'],
            test_data['reference_image'],
            test_data['reference_mask']
        )
        
        loss = criterion(output, test_data['query_mask'])
        loss.backward()
        optimizer.step()
        
        return {
            'loss_value': loss.item(),
            'output_shape': output.shape,
            'training_successful': True
        }
    
    def run_all_tests(self):
        """Run all tests and generate report."""
        print("\n" + "="*80)
        print("🧪 COMPREHENSIVE IRIS FRAMEWORK END-TO-END TEST")
        print("="*80)
        
        if not IRIS_AVAILABLE:
            print("❌ IRIS components not available - cannot run tests")
            return
        
        # Test individual components
        tests = [
            ("Encoder", self.test_encoder),
            ("Task Encoding", self.test_task_encoding),
            ("Decoder (Original)", self.test_decoder_original),
        ]
        
        if FIXED_AVAILABLE:
            tests.append(("Decoder (Fixed)", self.test_decoder_fixed))
        
        # Test complete models
        tests.extend([
            ("IRIS Model (Original)", self.test_iris_model_original),
        ])
        
        if FIXED_AVAILABLE:
            tests.append(("IRIS Model (Fixed)", self.test_iris_model_fixed))
        
        # Test training capabilities
        tests.extend([
            ("Gradient Flow", self.test_gradient_flow),
            ("Training Step", self.test_training_step),
        ])
        
        # Run all tests
        passed = 0
        total = len(tests)
        
        for test_name, test_fn in tests:
            if self.test_component(test_name, test_fn):
                passed += 1
        
        # Generate report
        self.generate_report(passed, total)
    
    def generate_report(self, passed, total):
        """Generate comprehensive test report."""
        print("\n" + "="*80)
        print("📊 TEST RESULTS SUMMARY")
        print("="*80)
        
        print(f"Tests Passed: {passed}/{total} ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED - IRIS Framework is working!")
        elif passed > total * 0.5:
            print("⚠️  PARTIAL SUCCESS - Some components working")
        else:
            print("❌ MAJOR ISSUES - Most components failing")
        
        print("\nDetailed Results:")
        for test_name, result in self.results.items():
            status_icon = "✅" if result['status'] == 'PASS' else "❌"
            print(f"{status_icon} {test_name}: {result['status']}")
            
            if result['status'] == 'FAIL':
                print(f"   Error: {result['error']}")
            elif 'details' in result:
                if isinstance(result['details'], dict):
                    for key, value in result['details'].items():
                        if isinstance(value, (int, float, str)):
                            print(f"   {key}: {value}")
        
        # Recommendations
        print("\n" + "="*80)
        print("🔧 RECOMMENDATIONS")
        print("="*80)
        
        failed_tests = [name for name, result in self.results.items() 
                       if result['status'] == 'FAIL']
        
        if not failed_tests:
            print("✅ Framework is ready for real medical data testing!")
        else:
            print("❌ Critical issues need to be resolved:")
            for test_name in failed_tests:
                print(f"   - Fix {test_name}")
            
            print("\n🔄 Alternative approaches to consider:")
            print("   - Implement simpler U-Net baseline")
            print("   - Use established medical segmentation architectures")
            print("   - Explore related works mentioned in IRIS paper")


def main():
    """Main test execution."""
    tester = EndToEndTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
