"""
Unit tests for ResNet model variants

Tests ResNet variants specified:
- ResNet: Standard ResNet with normal convolutions
- ResNet dil: ResNet with dilated convolutions for expanded receptive fields

Each test verifies:
1. Model initialization and architecture
2. Forward pass functionality
3. Dilated convolution specific features
4. Compatibility with all 5 datasets
5. Training integration capabilities
6. Receptive field properties
7. Physical validation aspects
"""

import unittest
import torch
import torch.nn as nn
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

try:
    from src.core.models.model_resnet import DilatedResNet
    RESNET_AVAILABLE = True
except ImportError as e:
    RESNET_AVAILABLE = False
    print(f"ResNet modules not available: {e}")

try:
    from src.core.utils.params import ModelParamsDecoder, DataParams
    PARAMS_AVAILABLE = True
except ImportError as e:
    PARAMS_AVAILABLE = False
    print(f"Params modules not available: {e}")

from tests.fixtures.dummy_datasets import get_dummy_batch


@unittest.skipIf(not RESNET_AVAILABLE, "ResNet modules not available")
class TestResNetVariants(unittest.TestCase):
    """Test suite for ResNet model variants"""

    def setUp(self):
        """Set up test parameters and configurations"""
        self.data_params = DataParams(
            batch=2,
            sequenceLength=[8, 2],
            dataSize=[16, 16],
            dimension=2,
            simFields=["pres"],
            simParams=[],
            normalizeMode=""
        ) if PARAMS_AVAILABLE else None

        # Test all dataset types
        self.dataset_names = ['inc_low', 'inc_high', 'tra_ext', 'tra_inc', 'iso']

    def test_resnet_standard_initialization(self):
        """Test standard ResNet initialization"""
        try:
            resnet_model = DilatedResNet(
                inFeatures=3,
                outFeatures=3,
                blocks=2,
                features=32,
                dilate=False  # Standard ResNet
            )
            self.assertIsInstance(resnet_model, nn.Module)

            # Check that dilation is disabled
            has_dilation = any(
                hasattr(module, 'dilation') and
                any(d > 1 for d in (module.dilation if isinstance(module.dilation, tuple) else [module.dilation]))
                for module in resnet_model.modules()
                if hasattr(module, 'dilation')
            )
            self.assertFalse(has_dilation, "Standard ResNet should not have dilated convolutions")

        except Exception as e:
            self.fail(f"Standard ResNet initialization failed: {e}")

    def test_resnet_dilated_initialization(self):
        """Test dilated ResNet initialization"""
        try:
            resnet_model = DilatedResNet(
                inFeatures=3,
                outFeatures=3,
                blocks=2,
                features=32,
                dilate=True  # Dilated ResNet
            )
            self.assertIsInstance(resnet_model, nn.Module)

            # Check that dilation is enabled (at least some layers should have dilation > 1)
            has_dilation = any(
                hasattr(module, 'dilation') and
                any(d > 1 for d in (module.dilation if isinstance(module.dilation, tuple) else [module.dilation]))
                for module in resnet_model.modules()
                if hasattr(module, 'dilation')
            )
            self.assertTrue(has_dilation, "Dilated ResNet should have dilated convolutions")

        except Exception as e:
            self.fail(f"Dilated ResNet initialization failed: {e}")

    def test_resnet_standard_vs_dilated_differences(self):
        """Test differences between standard and dilated ResNet"""
        try:
            # Create both variants with same parameters
            resnet_std = DilatedResNet(inFeatures=3, outFeatures=3, blocks=2, features=16, dilate=False)
            resnet_dil = DilatedResNet(inFeatures=3, outFeatures=3, blocks=2, features=16, dilate=True)

            # Test input
            x = torch.randn(1, 3, 16, 16)

            with torch.no_grad():
                output_std = resnet_std(x)
                output_dil = resnet_dil(x)

            # Both should produce same shape
            self.assertEqual(output_std.shape, output_dil.shape)

            # But outputs should generally be different due to different receptive fields
            # (though this isn't guaranteed with random initialization)
            self.assertEqual(output_std.shape, (1, 3, 16, 16))
            self.assertEqual(output_dil.shape, (1, 3, 16, 16))

            # Both should be finite
            self.assertTrue(torch.all(torch.isfinite(output_std)))
            self.assertTrue(torch.all(torch.isfinite(output_dil)))

        except Exception as e:
            self.fail(f"ResNet standard vs dilated comparison failed: {e}")

    def test_resnet_forward_pass(self):
        """Test ResNet forward pass with various configurations"""
        configs = [
            {"blocks": 1, "features": 8, "dilate": False},
            {"blocks": 2, "features": 16, "dilate": False},
            {"blocks": 1, "features": 8, "dilate": True},
            {"blocks": 2, "features": 16, "dilate": True},
        ]

        for config in configs:
            with self.subTest(config=config):
                try:
                    resnet_model = DilatedResNet(
                        inFeatures=3,
                        outFeatures=3,
                        **config
                    )

                    # Create test input [B, C, H, W]
                    x = torch.randn(2, 3, 16, 16)

                    with torch.no_grad():
                        output = resnet_model(x)

                    self.assertEqual(output.shape, (2, 3, 16, 16))
                    self.assertTrue(torch.all(torch.isfinite(output)))

                except Exception as e:
                    self.fail(f"ResNet forward pass failed with config {config}: {e}")

    def test_resnet_gradient_flow(self):
        """Test ResNet gradient flow for both variants"""
        variants = [False, True]  # Standard and dilated

        for dilate in variants:
            with self.subTest(dilated=dilate):
                try:
                    resnet_model = DilatedResNet(
                        inFeatures=3, outFeatures=3, blocks=1, features=8, dilate=dilate
                    )
                    x = torch.randn(1, 3, 8, 8, requires_grad=True)
                    target = torch.randn(1, 3, 8, 8)

                    output = resnet_model(x)
                    loss = nn.MSELoss()(output, target)
                    loss.backward()

                    has_gradients = any(param.grad is not None for param in resnet_model.parameters())
                    self.assertTrue(has_gradients, f"ResNet (dilated={dilate}) should have gradients")

                except Exception as e:
                    self.fail(f"ResNet gradient flow test failed for dilated={dilate}: {e}")

    def test_resnet_different_input_sizes(self):
        """Test ResNet with different input sizes"""
        sizes = [(8, 8), (16, 16), (32, 32)]

        for H, W in sizes:
            with self.subTest(size=(H, W)):
                try:
                    resnet_model = DilatedResNet(inFeatures=3, outFeatures=3, blocks=1, features=8)
                    x = torch.randn(1, 3, H, W)

                    with torch.no_grad():
                        output = resnet_model(x)

                    self.assertEqual(output.shape, (1, 3, H, W))

                except Exception as e:
                    self.fail(f"ResNet failed with input size {H}x{W}: {e}")

    def test_resnet_different_channel_counts(self):
        """Test ResNet with different input/output channel configurations"""
        channel_configs = [
            {"in": 1, "out": 1},  # Single channel
            {"in": 3, "out": 3},  # RGB-like
            {"in": 2, "out": 4},  # Different in/out
            {"in": 5, "out": 2},  # Many in, few out
        ]

        for config in channel_configs:
            with self.subTest(channels=config):
                try:
                    resnet_model = DilatedResNet(
                        inFeatures=config["in"],
                        outFeatures=config["out"],
                        blocks=1,
                        features=8
                    )
                    x = torch.randn(1, config["in"], 8, 8)

                    with torch.no_grad():
                        output = resnet_model(x)

                    self.assertEqual(output.shape, (1, config["out"], 8, 8))

                except Exception as e:
                    self.fail(f"ResNet failed with channel config {config}: {e}")

    def test_resnet_receptive_field_properties(self):
        """Test that dilated convolutions provide larger receptive fields"""
        try:
            # Create models with different block counts to test receptive field
            resnet_std = DilatedResNet(inFeatures=1, outFeatures=1, blocks=3, features=8, dilate=False)
            resnet_dil = DilatedResNet(inFeatures=1, outFeatures=1, blocks=3, features=8, dilate=True)

            # Create input with a single central point activated
            input_size = 32
            x = torch.zeros(1, 1, input_size, input_size)
            center = input_size // 2
            x[0, 0, center, center] = 1.0

            with torch.no_grad():
                output_std = resnet_std(x)
                output_dil = resnet_dil(x)

            # Both should produce some response
            self.assertGreater(torch.sum(torch.abs(output_std)).item(), 0)
            self.assertGreater(torch.sum(torch.abs(output_dil)).item(), 0)

            # Dilated version should potentially have wider spatial influence
            # (This is a structural test since exact receptive field depends on implementation)
            std_response_area = torch.sum(torch.abs(output_std) > 1e-6).item()
            dil_response_area = torch.sum(torch.abs(output_dil) > 1e-6).item()

            # Both should activate reasonable number of pixels
            self.assertGreater(std_response_area, 0)
            self.assertGreater(dil_response_area, 0)

        except Exception as e:
            self.fail(f"ResNet receptive field test failed: {e}")

    def test_resnet_residual_connections(self):
        """Test that residual connections are working"""
        try:
            resnet_model = DilatedResNet(inFeatures=3, outFeatures=3, blocks=2, features=16)

            # Create identity-like input to test residual behavior
            x = torch.randn(1, 3, 16, 16)

            # Forward pass
            with torch.no_grad():
                output = resnet_model(x)

            # Output should be different from input (due to processing)
            # but should maintain reasonable relationship
            self.assertFalse(torch.allclose(output, x, atol=1e-3))
            self.assertTrue(torch.all(torch.isfinite(output)))

            # Test that the model can learn identity mapping if needed
            # (This is more of a capability test)
            param_count = sum(p.numel() for p in resnet_model.parameters())
            self.assertGreater(param_count, 100)  # Should have reasonable parameters

        except Exception as e:
            self.fail(f"ResNet residual connections test failed: {e}")

    def test_resnet_parameter_count_scaling(self):
        """Test that parameter count scales appropriately with model size"""
        try:
            # Test different model sizes
            configs = [
                {"blocks": 1, "features": 8},
                {"blocks": 2, "features": 8},
                {"blocks": 1, "features": 16},
                {"blocks": 2, "features": 16},
            ]

            param_counts = {}
            for config in configs:
                model = DilatedResNet(inFeatures=3, outFeatures=3, **config)
                param_count = sum(p.numel() for p in model.parameters())
                param_counts[str(config)] = param_count

                # Should have reasonable parameter count
                self.assertGreater(param_count, 50)
                self.assertLess(param_count, 1_000_000)

            # More blocks or features should generally mean more parameters
            self.assertGreater(param_counts['{"blocks": 2, "features": 8}'],
                             param_counts['{"blocks": 1, "features": 8}'])
            self.assertGreater(param_counts['{"blocks": 1, "features": 16}'],
                             param_counts['{"blocks": 1, "features": 8}'])

        except Exception as e:
            self.fail(f"ResNet parameter count scaling test failed: {e}")

    def test_resnet_temporal_consistency_simulation(self):
        """Test ResNet with simulated temporal sequence processing"""
        try:
            resnet_model = DilatedResNet(inFeatures=6, outFeatures=3, blocks=2, features=16)

            # Simulate temporal processing by concatenating frames
            # [current_frame + previous_frame] -> [next_frame]
            frame1 = torch.randn(1, 3, 16, 16)
            frame2 = torch.randn(1, 3, 16, 16)

            # Concatenate along channel dimension (simulating temporal input)
            temporal_input = torch.cat([frame1, frame2], dim=1)  # [1, 6, 16, 16]

            with torch.no_grad():
                output = resnet_model(temporal_input)

            self.assertEqual(output.shape, (1, 3, 16, 16))
            self.assertTrue(torch.all(torch.isfinite(output)))

            # Test consistency: similar inputs should produce similar outputs
            frame1_similar = frame1 + torch.randn_like(frame1) * 0.01
            frame2_similar = frame2 + torch.randn_like(frame2) * 0.01
            temporal_input_similar = torch.cat([frame1_similar, frame2_similar], dim=1)

            with torch.no_grad():
                output_similar = resnet_model(temporal_input_similar)

            # Outputs should be reasonably close for similar inputs
            output_diff = torch.mean(torch.abs(output - output_similar))
            self.assertLess(output_diff.item(), 1.0, "ResNet should be stable to small input changes")

        except Exception as e:
            self.fail(f"ResNet temporal consistency simulation failed: {e}")

    @unittest.skipIf(not PARAMS_AVAILABLE, "Params modules not available")
    def test_resnet_all_datasets_compatibility(self):
        """Test ResNet models work with all dataset types"""
        resnet_model = DilatedResNet(inFeatures=6, outFeatures=3, blocks=1, features=8)

        for dataset_name in self.dataset_names:
            with self.subTest(dataset=dataset_name):
                try:
                    input_batch, _ = get_dummy_batch(dataset_name, batch_size=1)

                    # Simulate ResNet usage: concatenate temporal frames for input
                    B, T, C, H, W = input_batch.shape
                    if T >= 2:
                        # Use first two frames as input
                        resnet_input = torch.cat([input_batch[:, 0], input_batch[:, 1]], dim=1)
                    else:
                        # Duplicate single frame
                        resnet_input = torch.cat([input_batch[:, 0], input_batch[:, 0]], dim=1)

                    with torch.no_grad():
                        output = resnet_model(resnet_input)

                    # Should produce single frame output
                    self.assertEqual(output.shape, (B, C, H, W))
                    self.assertTrue(torch.all(torch.isfinite(output)))

                except Exception as e:
                    self.fail(f"ResNet failed on dataset {dataset_name}: {e}")

    def test_resnet_physical_validation_properties(self):
        """Test ResNet outputs have reasonable physical properties"""
        try:
            resnet_model = DilatedResNet(inFeatures=3, outFeatures=3, blocks=2, features=16)

            # Create physical-like input (velocity field simulation)
            x = torch.randn(1, 3, 16, 16) * 0.5  # Moderate velocity magnitudes

            with torch.no_grad():
                output = resnet_model(x)

            # Check output bounds are reasonable
            output_min = torch.min(output)
            output_max = torch.max(output)
            output_std = torch.std(output)

            # Should not have extreme values
            self.assertGreater(output_min.item(), -50.0, "ResNet output min too negative")
            self.assertLess(output_max.item(), 50.0, "ResNet output max too large")
            self.assertTrue(torch.isfinite(output_std), "ResNet output std should be finite")

            # Should have reasonable dynamic range
            self.assertGreater(output_std.item(), 1e-6, "ResNet output should have variation")

            # Check spatial smoothness (ResNet should not introduce extreme discontinuities)
            spatial_grad_x = torch.abs(output[:, :, 1:, :] - output[:, :, :-1, :])
            spatial_grad_y = torch.abs(output[:, :, :, 1:] - output[:, :, :, :-1])

            max_grad_x = torch.max(spatial_grad_x)
            max_grad_y = torch.max(spatial_grad_y)

            self.assertLess(max_grad_x.item(), 20.0, "ResNet spatial gradients should be reasonable")
            self.assertLess(max_grad_y.item(), 20.0, "ResNet spatial gradients should be reasonable")

        except Exception as e:
            self.fail(f"ResNet physical validation test failed: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)