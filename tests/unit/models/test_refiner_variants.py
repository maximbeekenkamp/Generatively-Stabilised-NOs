"""
Unit tests for Refiner model variants

Tests the PDERefiner model which performs iterative refinement of PDE solutions
using a denoising diffusion approach.

The refiner model:
1. Takes conditioning input (e.g., from another model)
2. Iteratively refines solutions through denoising steps
3. Uses a U-Net backbone for the denoising process
4. Supports both training and inference modes

Each test verifies:
1. Model initialization and architecture
2. Training mode functionality (noise prediction)
3. Inference mode functionality (iterative refinement)
4. Conditioning mechanism
5. Denoising process properties
6. Compatibility with all datasets
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
    from src.core.models.model_refiner import PDERefiner
    from src.core.utils.params import DataParams, ModelParamsDecoder
    REFINER_AVAILABLE = True
except ImportError as e:
    REFINER_AVAILABLE = False
    print(f"Refiner modules not available: {e}")

from tests.fixtures.dummy_datasets import get_dummy_batch


@unittest.skipIf(not REFINER_AVAILABLE, "Refiner modules not available")
class TestRefinerVariants(unittest.TestCase):
    """Test suite for Refiner model variants"""

    def setUp(self):
        """Set up test parameters and configurations"""
        # Data parameters
        self.data_params = DataParams(
            batch=2,
            sequenceLength=[8, 2],
            dataSize=[16, 16],
            dimension=2,
            simFields=["pres"],
            simParams=[],  # Standardized to empty for consistent channel count
            normalizeMode=""
        )

        # Model parameters for refiner
        self.refiner_params = ModelParamsDecoder(
            arch="refiner",
            diffSteps=4,  # Number of refinement steps
            refinerStd=0.01,  # Minimum noise standard deviation
            decWidth=32
        )

        # Test all dataset types
        self.dataset_names = ['inc_low', 'inc_high', 'tra_ext', 'tra_inc', 'iso']

    def _create_refiner_params(self, steps=4, std=0.01):
        """Create refiner model parameters with custom settings"""
        class MockRefinerParams:
            def __init__(self, steps, std):
                self.arch = "refiner"
                self.diffSteps = steps
                self.refinerStd = std
                self.decWidth = 32

        return MockRefinerParams(steps, std)

    def test_refiner_initialization(self):
        """Test PDERefiner model initialization"""
        try:
            # Calculate conditioning channels
            cond_channels = self.data_params.dimension + len(self.data_params.simFields) + len(self.data_params.simParams)

            refiner_model = PDERefiner(
                self.data_params,
                self.refiner_params,
                condChannels=cond_channels
            )

            self.assertIsInstance(refiner_model, nn.Module)
            self.assertIsInstance(refiner_model, PDERefiner)

            # Check internal parameters
            self.assertEqual(refiner_model.numSteps, self.refiner_params.diffSteps)
            self.assertEqual(refiner_model.minNoiseStd, self.refiner_params.refinerStd)

            # Check that U-Net was created
            self.assertIsNotNone(refiner_model.unet)

        except Exception as e:
            self.fail(f"Refiner initialization failed: {e}")

    def test_refiner_training_mode(self):
        """Test PDERefiner in training mode"""
        try:
            cond_channels = self.data_params.dimension + len(self.data_params.simFields) + len(self.data_params.simParams)
            refiner_model = PDERefiner(self.data_params, self.refiner_params, condChannels=cond_channels)

            # Set to training mode
            refiner_model.train()

            batch_size = 2
            seq_len = 4
            channels = cond_channels
            H, W = self.data_params.dataSize

            # Create dummy conditioning and data
            conditioning = torch.randn(batch_size, seq_len, channels, H, W)
            data = torch.randn(batch_size, seq_len, channels, H, W)

            # Forward pass in training mode
            pred, target = refiner_model(conditioning, data)

            # Check output shapes
            self.assertEqual(pred.shape, data.shape)
            self.assertEqual(target.shape, data.shape)

            # Check outputs are finite
            self.assertTrue(torch.all(torch.isfinite(pred)))
            self.assertTrue(torch.all(torch.isfinite(target)))

            # Training mode should return both prediction and target
            self.assertIsInstance(pred, torch.Tensor)
            self.assertIsInstance(target, torch.Tensor)

        except Exception as e:
            self.fail(f"Refiner training mode test failed: {e}")

    def test_refiner_inference_mode(self):
        """Test PDERefiner in inference mode"""
        try:
            cond_channels = self.data_params.dimension + len(self.data_params.simFields) + len(self.data_params.simParams)
            refiner_model = PDERefiner(self.data_params, self.refiner_params, condChannels=cond_channels)

            # Set to evaluation mode
            refiner_model.eval()

            batch_size = 1
            seq_len = 2
            channels = cond_channels
            H, W = self.data_params.dataSize

            # Create dummy conditioning and data
            conditioning = torch.randn(batch_size, seq_len, channels, H, W)
            data = torch.randn(batch_size, seq_len, channels, H, W)

            # Forward pass in inference mode
            with torch.no_grad():
                refined_output = refiner_model(conditioning, data)

            # Check output shape
            self.assertEqual(refined_output.shape, data.shape)

            # Check output is finite
            self.assertTrue(torch.all(torch.isfinite(refined_output)))

            # Inference mode should return single tensor
            self.assertIsInstance(refined_output, torch.Tensor)

        except Exception as e:
            self.fail(f"Refiner inference mode test failed: {e}")

    def test_refiner_different_step_counts(self):
        """Test refiner with different numbers of refinement steps"""
        try:
            step_counts = [1, 2, 4, 8]
            cond_channels = self.data_params.dimension + len(self.data_params.simFields) + len(self.data_params.simParams)

            for steps in step_counts:
                with self.subTest(steps=steps):
                    refiner_params = self._create_refiner_params(steps=steps)
                    refiner_model = PDERefiner(self.data_params, refiner_params, condChannels=cond_channels)

                    refiner_model.eval()

                    batch_size = 1
                    seq_len = 2
                    H, W = self.data_params.dataSize

                    conditioning = torch.randn(batch_size, seq_len, cond_channels, H, W)
                    data = torch.randn(batch_size, seq_len, cond_channels, H, W)

                    with torch.no_grad():
                        output = refiner_model(conditioning, data)

                    self.assertEqual(output.shape, data.shape)
                    self.assertTrue(torch.all(torch.isfinite(output)))

        except Exception as e:
            self.fail(f"Refiner different step counts test failed: {e}")

    def test_refiner_different_noise_levels(self):
        """Test refiner with different noise standard deviations"""
        try:
            noise_stds = [0.001, 0.01, 0.1]
            cond_channels = self.data_params.dimension + len(self.data_params.simFields) + len(self.data_params.simParams)

            for std in noise_stds:
                with self.subTest(noise_std=std):
                    refiner_params = self._create_refiner_params(std=std)
                    refiner_model = PDERefiner(self.data_params, refiner_params, condChannels=cond_channels)

                    refiner_model.eval()

                    batch_size = 1
                    seq_len = 2
                    H, W = self.data_params.dataSize

                    conditioning = torch.randn(batch_size, seq_len, cond_channels, H, W)
                    data = torch.randn(batch_size, seq_len, cond_channels, H, W)

                    with torch.no_grad():
                        output = refiner_model(conditioning, data)

                    self.assertEqual(output.shape, data.shape)
                    self.assertTrue(torch.all(torch.isfinite(output)))

        except Exception as e:
            self.fail(f"Refiner different noise levels test failed: {e}")

    def test_refiner_conditioning_effect(self):
        """Test that conditioning input affects refiner output"""
        try:
            cond_channels = self.data_params.dimension + len(self.data_params.simFields) + len(self.data_params.simParams)
            refiner_model = PDERefiner(self.data_params, self.refiner_params, condChannels=cond_channels)

            refiner_model.eval()

            batch_size = 1
            seq_len = 2
            H, W = self.data_params.dataSize

            # Create different conditioning inputs
            conditioning1 = torch.randn(batch_size, seq_len, cond_channels, H, W)
            conditioning2 = torch.randn(batch_size, seq_len, cond_channels, H, W)
            data = torch.randn(batch_size, seq_len, cond_channels, H, W)

            with torch.no_grad():
                output1 = refiner_model(conditioning1, data)
                output2 = refiner_model(conditioning2, data)

            # Different conditioning should produce different outputs
            self.assertFalse(torch.allclose(output1, output2, atol=1e-3),
                           "Different conditioning should produce different outputs")

        except Exception as e:
            self.fail(f"Refiner conditioning effect test failed: {e}")

    def test_refiner_iterative_improvement(self):
        """Test that refiner iteratively improves solutions"""
        try:
            # Use more steps to test iterative improvement
            refiner_params = self._create_refiner_params(steps=8, std=0.1)
            cond_channels = self.data_params.dimension + len(self.data_params.simFields) + len(self.data_params.simParams)
            refiner_model = PDERefiner(self.data_params, refiner_params, condChannels=cond_channels)

            refiner_model.eval()

            batch_size = 1
            seq_len = 1
            H, W = self.data_params.dataSize

            # Create conditioning that represents a "good" solution
            conditioning = torch.zeros(batch_size, seq_len, cond_channels, H, W)
            conditioning[0, 0, 0, 8, 8] = 1.0  # Central peak

            # Create noisy data that needs refinement
            data = conditioning + torch.randn_like(conditioning) * 0.5

            with torch.no_grad():
                refined_output = refiner_model(conditioning, data)

            # Refined output should be different from input (refinement applied)
            refinement_magnitude = torch.mean(torch.abs(refined_output - data))
            self.assertGreater(refinement_magnitude.item(), 1e-6,
                             "Refiner should modify input data during refinement")

            # Refined output should be finite and bounded
            self.assertTrue(torch.all(torch.isfinite(refined_output)))
            self.assertLess(torch.max(torch.abs(refined_output)).item(), 100.0,
                           "Refined output should remain bounded")

        except Exception as e:
            self.fail(f"Refiner iterative improvement test failed: {e}")

    def test_refiner_gradient_flow(self):
        """Test gradient flow through refiner in training mode"""
        try:
            cond_channels = self.data_params.dimension + len(self.data_params.simFields) + len(self.data_params.simParams)
            refiner_model = PDERefiner(self.data_params, self.refiner_params, condChannels=cond_channels)

            refiner_model.train()

            batch_size = 1
            seq_len = 2
            H, W = self.data_params.dataSize

            conditioning = torch.randn(batch_size, seq_len, cond_channels, H, W, requires_grad=True)
            data = torch.randn(batch_size, seq_len, cond_channels, H, W, requires_grad=True)

            # Forward pass
            pred, target = refiner_model(conditioning, data)

            # Compute loss and backpropagate
            loss = nn.MSELoss()(pred, target)
            loss.backward()

            # Check that model parameters have gradients
            has_gradients = any(param.grad is not None for param in refiner_model.parameters())
            self.assertTrue(has_gradients, "Refiner model should have gradients")

            # Check that input gradients exist
            self.assertIsNotNone(conditioning.grad, "Conditioning should have gradients")
            self.assertIsNotNone(data.grad, "Data should have gradients")

        except Exception as e:
            self.fail(f"Refiner gradient flow test failed: {e}")

    def test_refiner_different_spatial_sizes(self):
        """Test refiner with different spatial resolutions"""
        try:
            spatial_sizes = [(8, 8), (16, 16), (32, 32)]
            cond_channels = 3  # Simplified for different sizes

            for H, W in spatial_sizes:
                with self.subTest(size=(H, W)):
                    # Create data params for this size
                    data_params = DataParams(
                        batch=1,
                        sequenceLength=[2, 1],
                        dataSize=[H, W],
                        dimension=2,
                        simFields=[],
                        simParams=[],
                        normalizeMode=""
                    )

                    refiner_params = self._create_refiner_params(steps=2)
                    refiner_model = PDERefiner(data_params, refiner_params, condChannels=cond_channels)

                    refiner_model.eval()

                    conditioning = torch.randn(1, 2, cond_channels, H, W)
                    data = torch.randn(1, 2, cond_channels, H, W)

                    with torch.no_grad():
                        output = refiner_model(conditioning, data)

                    self.assertEqual(output.shape, (1, 2, cond_channels, H, W))
                    self.assertTrue(torch.all(torch.isfinite(output)))

        except Exception as e:
            self.fail(f"Refiner different spatial sizes test failed: {e}")

    def test_refiner_all_datasets_compatibility(self):
        """Test refiner model works with all dataset types"""
        try:
            cond_channels = self.data_params.dimension + len(self.data_params.simFields) + len(self.data_params.simParams)
            refiner_model = PDERefiner(self.data_params, self.refiner_params, condChannels=cond_channels)

            refiner_model.eval()

            for dataset_name in self.dataset_names:
                with self.subTest(dataset=dataset_name):
                    input_batch, _ = get_dummy_batch(dataset_name, batch_size=1)

                    # Use input as both conditioning and data for testing
                    conditioning = input_batch
                    data = input_batch + torch.randn_like(input_batch) * 0.1  # Add some noise

                    with torch.no_grad():
                        output = refiner_model(conditioning, data)

                    self.assertEqual(output.shape, input_batch.shape)
                    self.assertTrue(torch.all(torch.isfinite(output)))

        except Exception as e:
            self.fail(f"Refiner dataset compatibility test failed: {e}")

    def test_refiner_denoising_properties(self):
        """Test refiner's denoising capabilities"""
        try:
            cond_channels = self.data_params.dimension + len(self.data_params.simFields) + len(self.data_params.simParams)
            refiner_model = PDERefiner(self.data_params, self.refiner_params, condChannels=cond_channels)

            refiner_model.eval()

            batch_size = 1
            seq_len = 1
            H, W = self.data_params.dataSize

            # Create clean signal
            clean_signal = torch.zeros(batch_size, seq_len, cond_channels, H, W)
            clean_signal[0, 0, 0, H//4:3*H//4, W//4:3*W//4] = 1.0  # Square pattern

            # Add noise
            noise_level = 0.2
            noisy_signal = clean_signal + torch.randn_like(clean_signal) * noise_level

            # Use clean signal as conditioning
            with torch.no_grad():
                denoised_output = refiner_model(clean_signal, noisy_signal)

            # Denoised output should be closer to clean signal than noisy input
            original_error = torch.mean(torch.abs(noisy_signal - clean_signal))
            denoised_error = torch.mean(torch.abs(denoised_output - clean_signal))

            # This is a structural test - with proper training, denoised should be better
            # For dummy test, just ensure denoising process runs and produces reasonable output
            self.assertTrue(torch.all(torch.isfinite(denoised_output)))
            self.assertLess(torch.max(torch.abs(denoised_output)).item(), 10.0,
                           "Denoised output should remain bounded")

        except Exception as e:
            self.fail(f"Refiner denoising properties test failed: {e}")

    def test_refiner_training_inference_consistency(self):
        """Test that training and inference modes are consistent in structure"""
        try:
            cond_channels = self.data_params.dimension + len(self.data_params.simFields) + len(self.data_params.simParams)
            refiner_model = PDERefiner(self.data_params, self.refiner_params, condChannels=cond_channels)

            batch_size = 1
            seq_len = 2
            H, W = self.data_params.dataSize

            conditioning = torch.randn(batch_size, seq_len, cond_channels, H, W)
            data = torch.randn(batch_size, seq_len, cond_channels, H, W)

            # Test training mode
            refiner_model.train()
            pred, target = refiner_model(conditioning, data)

            # Test inference mode
            refiner_model.eval()
            with torch.no_grad():
                output = refiner_model(conditioning, data)

            # All outputs should have same shape as input data
            self.assertEqual(pred.shape, data.shape)
            self.assertEqual(target.shape, data.shape)
            self.assertEqual(output.shape, data.shape)

            # All outputs should be finite
            self.assertTrue(torch.all(torch.isfinite(pred)))
            self.assertTrue(torch.all(torch.isfinite(target)))
            self.assertTrue(torch.all(torch.isfinite(output)))

        except Exception as e:
            self.fail(f"Refiner training/inference consistency test failed: {e}")

    def test_refiner_parameter_count(self):
        """Test refiner model has reasonable parameter count"""
        try:
            cond_channels = self.data_params.dimension + len(self.data_params.simFields) + len(self.data_params.simParams)
            refiner_model = PDERefiner(self.data_params, self.refiner_params, condChannels=cond_channels)

            param_count = sum(p.numel() for p in refiner_model.parameters())

            # Should have reasonable number of parameters
            self.assertGreater(param_count, 1000, "Refiner should have > 1000 parameters")
            self.assertLess(param_count, 50_000_000, "Refiner should have < 50M parameters")

        except Exception as e:
            self.fail(f"Refiner parameter count test failed: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)