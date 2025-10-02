"""
Unit tests for DeepONet Format Adapter

Tests the conversion utilities between Gen Stabilised format [B,T,C,H,W]
and DeepONet format (branch_input, trunk_input).
"""

import unittest
import torch
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

try:
    from src.core.models.deeponet_format_adapter import DeepONetFormatAdapter, DeepONetWrapper
    from deepxde.nn.pytorch.deeponet import DeepONet
    ADAPTER_AVAILABLE = True
except ImportError as e:
    ADAPTER_AVAILABLE = False
    print(f"DeepONet adapter modules not available: {e}")

from tests.fixtures.dummy_datasets import get_dummy_batch


@unittest.skipIf(not ADAPTER_AVAILABLE, "DeepONet adapter modules not available")
class TestDeepONetFormatAdapter(unittest.TestCase):
    """Test suite for DeepONet format conversion adapter"""

    def setUp(self):
        """Set up test parameters"""
        self.H, self.W = 16, 16
        self.coordinate_dim = 2
        self.adapter = DeepONetFormatAdapter((self.H, self.W), self.coordinate_dim)

    def test_adapter_initialization(self):
        """Test adapter initializes correctly"""
        self.assertEqual(self.adapter.H, self.H)
        self.assertEqual(self.adapter.W, self.W)
        self.assertEqual(self.adapter.coordinate_dim, self.coordinate_dim)
        self.assertEqual(self.adapter.spatial_points, self.H * self.W)

        # Check coordinate grid template
        coords = self.adapter._trunk_input_template
        self.assertEqual(coords.shape, (self.H * self.W, self.coordinate_dim))
        self.assertTrue(torch.all(coords >= 0) and torch.all(coords <= 1))

    def test_to_deeponet_format_last_timestep(self):
        """Test conversion to DeepONet format using last timestep"""
        # Create test input
        B, T, C = 2, 8, 3
        gen_input = torch.randn(B, T, C, self.H, self.W)

        # Convert to DeepONet format
        branch_input, trunk_input = self.adapter.to_deeponet_format(gen_input, use_last_timestep=True)

        # Check shapes
        expected_branch_size = C * self.H * self.W
        self.assertEqual(branch_input.shape, (B, expected_branch_size))
        self.assertEqual(trunk_input.shape, (B, self.H * self.W, self.coordinate_dim))

        # Check that branch input matches last timestep
        expected_branch = gen_input[:, -1].flatten(1)
        self.assertTrue(torch.allclose(branch_input, expected_branch))

        # Check trunk input is coordinate grid
        self.assertTrue(torch.all(trunk_input >= 0) and torch.all(trunk_input <= 1))

    def test_to_deeponet_format_all_timesteps(self):
        """Test conversion using all timesteps"""
        B, T, C = 2, 8, 3
        gen_input = torch.randn(B, T, C, self.H, self.W)

        branch_input, trunk_input = self.adapter.to_deeponet_format(gen_input, use_last_timestep=False)

        # Check shapes
        expected_branch_size = T * C * self.H * self.W
        self.assertEqual(branch_input.shape, (B, expected_branch_size))
        self.assertEqual(trunk_input.shape, (B, self.H * self.W, self.coordinate_dim))

        # Check that branch input matches flattened input
        expected_branch = gen_input.flatten(1)
        self.assertTrue(torch.allclose(branch_input, expected_branch))

    def test_from_deeponet_format(self):
        """Test conversion from DeepONet format back to Gen Stabilised"""
        B, T, C = 2, 4, 3
        target_shape = (B, T, C, self.H, self.W)

        # Create mock DeepONet output
        deeponet_output = torch.randn(B, self.H * self.W, C)

        # Convert back to Gen Stabilised format
        gen_output = self.adapter.from_deeponet_format(deeponet_output, target_shape)

        # Check shape
        self.assertEqual(gen_output.shape, target_shape)

        # Check that spatial structure is preserved across time
        for t in range(1, T):
            self.assertTrue(torch.allclose(gen_output[:, 0], gen_output[:, t]))

        # Check that output matches reshaped DeepONet output
        expected_spatial = deeponet_output.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        self.assertTrue(torch.allclose(gen_output[:, 0], expected_spatial))

    def test_format_consistency(self):
        """Test round-trip conversion consistency"""
        B, T, C = 1, 4, 2
        original_input = torch.randn(B, T, C, self.H, self.W)

        # Convert to DeepONet format and back
        branch_input, trunk_input = self.adapter.to_deeponet_format(original_input)

        # Create a proper mock DeepONet output with the expected structure
        # branch_input is [B, C*H*W], we want output as [B, H*W, C]
        # Rearrange the branch input to simulate realistic DeepONet output
        reshaped_branch = branch_input.reshape(B, C, self.H, self.W)  # [B, C, H, W]
        mock_deeponet_output = reshaped_branch.permute(0, 2, 3, 1).reshape(B, self.H * self.W, C)  # [B, H*W, C]

        # Convert back
        reconstructed = self.adapter.from_deeponet_format(mock_deeponet_output, original_input.shape)

        # Check that last timestep spatial structure is preserved
        # Note: We expect exact match since we're using the last timestep data
        last_timestep_original = original_input[:, -1]  # [B, C, H, W]
        last_timestep_reconstructed = reconstructed[:, -1]  # [B, C, H, W]
        self.assertTrue(torch.allclose(last_timestep_original, last_timestep_reconstructed, atol=1e-6))

    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Wrong input shape
        with self.assertRaises(ValueError):
            self.adapter.to_deeponet_format(torch.randn(2, 3, 4))  # Not 5D

        # Wrong spatial dimensions
        with self.assertRaises(ValueError):
            self.adapter.to_deeponet_format(torch.randn(2, 4, 3, 8, 8))  # Wrong H, W

        # Wrong DeepONet output shape
        with self.assertRaises(ValueError):
            self.adapter.from_deeponet_format(torch.randn(2, 3), (2, 4, 3, 16, 16))  # Not 3D

        # Wrong spatial points
        with self.assertRaises(ValueError):
            self.adapter.from_deeponet_format(
                torch.randn(2, 100, 3), (2, 4, 3, 16, 16)  # Wrong spatial points
            )

    def test_utility_methods(self):
        """Test utility methods for size calculation"""
        C, T = 3, 8

        # Branch input size
        size_last = self.adapter.get_branch_input_size(C, use_last_timestep=True)
        self.assertEqual(size_last, C * self.H * self.W)

        size_all = self.adapter.get_branch_input_size(C, use_last_timestep=False, T=T)
        self.assertEqual(size_all, T * C * self.H * self.W)

        # Trunk input size
        trunk_size = self.adapter.get_trunk_input_size()
        self.assertEqual(trunk_size, (self.H * self.W, self.coordinate_dim))

        # Expected output size
        output_size = self.adapter.get_expected_output_size(C)
        self.assertEqual(output_size, (self.H * self.W, C))

    def test_3d_coordinates(self):
        """Test adapter with 3D coordinates"""
        adapter_3d = DeepONetFormatAdapter((self.H, self.W), coordinate_dim=3)

        B, T, C = 1, 2, 1
        gen_input = torch.randn(B, T, C, self.H, self.W)

        branch_input, trunk_input = adapter_3d.to_deeponet_format(gen_input)

        # Check 3D coordinate shape
        self.assertEqual(trunk_input.shape, (B, self.H * self.W, 3))

        # Check that z coordinate is zero (dummy for 2D spatial data)
        self.assertTrue(torch.allclose(trunk_input[:, :, 2], torch.zeros_like(trunk_input[:, :, 2])))


@unittest.skipIf(not ADAPTER_AVAILABLE, "DeepONet adapter modules not available")
class TestDeepONetWrapper(unittest.TestCase):
    """Test suite for DeepONet wrapper with automatic format conversion"""

    def setUp(self):
        """Set up test parameters"""
        self.H, self.W = 16, 16
        self.C = 3

        # Create a DeepXDE DeepONet model (matches test setup)
        input_features = self.C * self.H * self.W  # 3*16*16 = 768
        coordinate_dim = 2  # x, y coordinates
        latent_dim = 64

        branch_layers = [input_features, 128, 64, latent_dim]
        trunk_layers = [coordinate_dim, 128, 64, latent_dim]

        self.deeponet_model = DeepONet(
            layer_sizes_branch=branch_layers,
            layer_sizes_trunk=trunk_layers,
            activation="tanh",
            kernel_initializer="Glorot normal",
            num_outputs=self.C,  # 3 outputs for 3 channels
            multi_output_strategy="independent"
        )

        # Create wrapper
        self.wrapper = DeepONetWrapper(self.deeponet_model, (self.H, self.W))

    def test_wrapper_forward(self):
        """Test wrapper forward pass with format conversion"""
        B, T = 2, 4
        gen_input = torch.randn(B, T, self.C, self.H, self.W)

        with torch.no_grad():
            output = self.wrapper(gen_input)

        # Check output shape
        self.assertEqual(output.shape, gen_input.shape)

        # Check output is finite
        self.assertTrue(torch.all(torch.isfinite(output)))

    def test_wrapper_integration_with_dummy_data(self):
        """Test wrapper with dummy dataset"""
        input_batch, _ = get_dummy_batch("inc_low", batch_size=1)
        B, T, C, H, W = input_batch.shape

        # Create a DeepONet model with correct input size for this data
        input_features = C * H * W
        coordinate_dim = 2
        latent_dim = 64

        branch_layers = [input_features, 128, 64, latent_dim]
        trunk_layers = [coordinate_dim, 128, 64, latent_dim]

        deeponet_model = DeepONet(
            layer_sizes_branch=branch_layers,
            layer_sizes_trunk=trunk_layers,
            activation="tanh",
            kernel_initializer="Glorot normal",
            num_outputs=C,  # outputs for channels
            multi_output_strategy="independent"
        )

        # Create wrapper with correct dimensions
        wrapper = DeepONetWrapper(deeponet_model, (H, W))

        with torch.no_grad():
            output = wrapper(input_batch)

        # Should maintain shape
        self.assertEqual(output.shape, input_batch.shape)
        self.assertTrue(torch.all(torch.isfinite(output)))

    def test_wrapper_format_info(self):
        """Test wrapper format information"""
        info = self.wrapper.get_format_info()

        expected_info = {
            'spatial_dims': (self.H, self.W),
            'coordinate_dim': 2,
            'spatial_points': self.H * self.W,
        }

        self.assertEqual(info, expected_info)

    def test_wrapper_gradient_flow(self):
        """Test gradient flow through wrapper"""
        B, T = 1, 2
        gen_input = torch.randn(B, T, self.C, self.H, self.W, requires_grad=True)
        target = torch.randn(B, T, self.C, self.H, self.W)

        output = self.wrapper(gen_input)
        loss = torch.nn.MSELoss()(output, target)
        loss.backward()

        # Check gradients exist
        self.assertIsNotNone(gen_input.grad)
        self.assertTrue(torch.any(gen_input.grad != 0))

        # Check model parameters have gradients
        has_gradients = any(param.grad is not None for param in self.wrapper.parameters())
        self.assertTrue(has_gradients)


class TestFormatAdapterCompatibility(unittest.TestCase):
    """Test compatibility between manual conversion and adapter"""

    def setUp(self):
        """Set up test parameters"""
        self.H, self.W = 16, 16
        self.adapter = DeepONetFormatAdapter((self.H, self.W))

    def test_compatibility_with_existing_tests(self):
        """Test that adapter produces same results as manual conversion"""
        # Use the same conversion logic from existing tests
        input_batch, _ = get_dummy_batch("inc_low", batch_size=2)
        B, T, C, H, W = input_batch.shape

        # Manual conversion (from existing tests)
        branch_input_manual = input_batch[:, -1].flatten(1)
        x = torch.linspace(0, 1, H)
        y = torch.linspace(0, 1, W)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        trunk_input_manual = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        trunk_input_manual = trunk_input_manual.unsqueeze(0).expand(B, -1, -1)

        # Adapter conversion
        adapter = DeepONetFormatAdapter((H, W))
        branch_input_adapter, trunk_input_adapter = adapter.to_deeponet_format(input_batch)

        # Compare results
        self.assertTrue(torch.allclose(branch_input_manual, branch_input_adapter))
        self.assertTrue(torch.allclose(trunk_input_manual, trunk_input_adapter))


if __name__ == '__main__':
    unittest.main(verbosity=2)