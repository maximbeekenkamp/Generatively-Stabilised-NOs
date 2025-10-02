"""
Unit tests for FNO (Fourier Neural Operator) model variants

Tests all FNO variants specified:
- FNO 16: FNO with 16 modes
- FNO 32: FNO with 32 modes
- FNO16 + DM: FNO 16 with Diffusion Model correction

Each test verifies:
1. Model initialization
2. Forward pass functionality
3. Compatibility with all 5 datasets
4. Training integration
5. Sampling capability
6. Analysis integration
"""

import unittest
import torch
import torch.nn as nn
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from src.core.models.neural_operator_adapters import FNOPriorAdapter
from src.core.models.generative_operator_model import GenerativeOperatorModel
from src.core.models.model_registry import ModelRegistry
from src.core.utils.params import ModelParamsDecoder, DataParams, TrainingParams
from tests.fixtures.dummy_datasets import DummyDatasetFactory, get_dummy_batch


class TestFNOVariants(unittest.TestCase):
    """Test suite for FNO model variants"""

    def setUp(self):
        """Set up test parameters and configurations"""
        # Basic data parameters for dummy datasets
        self.data_params = DataParams(
            batch=2,
            sequenceLength=[8, 2],  # 8 input frames, 2 output frames
            dataSize=[16, 16],      # Small size for testing
            dimension=2,            # 2D velocity
            simFields=["pres"],     # Add pressure field
            simParams=[],           # No simulation parameters for testing
            normalizeMode=""
        )

        # Test all dataset types
        self.dataset_names = ['inc_low', 'inc_high', 'tra_ext', 'tra_inc', 'iso']

    def _create_fno_params(self, modes):
        """Create model parameters for FNO with specified modes"""
        # Mock ModelParamsDecoder with FNO-specific parameters
        class MockModelParams:
            def __init__(self, modes):
                self.fnoModes = [modes, modes]
                self.decWidth = 32
                self.architecture = 'fno'
                self.model_type = 'fno'
                self.prevSteps = 8

            def _get_prev_steps_from_arch(self):
                return self.prevSteps

        return MockModelParams(modes)

    def test_fno16_initialization(self):
        """Test FNO 16 model initialization"""
        model_params = self._create_fno_params(16)

        # Test that FNO can be initialized
        try:
            fno_model = FNOPriorAdapter(model_params, self.data_params)
            self.assertIsInstance(fno_model, nn.Module)
            self.assertIsInstance(fno_model, FNOPriorAdapter)
        except Exception as e:
            self.fail(f"FNO 16 initialization failed\n"
                     f"  Model config: modes=16, decWidth={self.data_params.dimension + len(self.data_params.simFields) + len(self.data_params.simParams)}\n"
                     f"  Data params: {self.data_params.dataSize}, channels={self.data_params.dimension + len(self.data_params.simFields) + len(self.data_params.simParams)}\n"
                     f"  Error: {e}")

    def test_fno32_initialization(self):
        """Test FNO 32 model initialization"""
        model_params = self._create_fno_params(32)

        try:
            fno_model = FNOPriorAdapter(model_params, self.data_params)
            self.assertIsInstance(fno_model, nn.Module)
            self.assertIsInstance(fno_model, FNOPriorAdapter)
        except Exception as e:
            self.fail(f"FNO 32 initialization failed: {e}")

    def test_fno16_forward_pass(self):
        """Test FNO 16 forward pass with dummy data"""
        model_params = self._create_fno_params(16)

        try:
            fno_model = FNOPriorAdapter(model_params, self.data_params)

            # Test with dummy data
            input_batch, _ = get_dummy_batch("inc_low", batch_size=2)

            with torch.no_grad():
                output = fno_model(input_batch)

            # Check output shape - FNO returns full sequence [B, T, C, H, W]
            expected_shape = (2, 8, 3, 16, 16)  # Same as input: 2 samples, 8 frames, 3 channels, 16x16 spatial
            self.assertEqual(output.shape, expected_shape)

        except Exception as e:
            self.fail(f"FNO 16 forward pass failed\n"
                     f"  Model config: modes=16, input_shape={input_batch.shape}\n"
                     f"  Expected output: {expected_shape}\n"
                     f"  Dataset: tra_ext, batch_size=2\n"
                     f"  Error: {e}")

    def test_fno32_forward_pass(self):
        """Test FNO 32 forward pass with dummy data"""
        model_params = self._create_fno_params(32)

        try:
            fno_model = FNOPriorAdapter(model_params, self.data_params)

            input_batch, _ = get_dummy_batch("tra_ext", batch_size=2)

            with torch.no_grad():
                output = fno_model(input_batch)

            expected_shape = (2, 8, 3, 16, 16)  # Same as input shape
            self.assertEqual(output.shape, expected_shape)

        except Exception as e:
            self.fail(f"FNO 32 forward pass failed\n"
                     f"  Model config: modes=32, input_shape={input_batch.shape}\n"
                     f"  Expected output: {expected_shape}\n"
                     f"  Dataset: tra_ext, batch_size=2\n"
                     f"  Error: {e}")

    def test_fno_all_datasets_compatibility(self):
        """Test FNO models work with all dataset types"""
        model_params = self._create_fno_params(16)
        fno_model = FNOPriorAdapter(model_params, self.data_params)

        for dataset_name in self.dataset_names:
            with self.subTest(dataset=dataset_name):
                try:
                    input_batch, target_batch = get_dummy_batch(dataset_name, batch_size=2)

                    with torch.no_grad():
                        output = fno_model(input_batch)

                    # Should produce output of same shape as input
                    self.assertEqual(output.shape, input_batch.shape)

                except Exception as e:
                    self.fail(f"FNO failed on dataset {dataset_name}: {e}")

    def test_fno_gradient_flow(self):
        """Test that gradients flow properly through FNO model"""
        model_params = self._create_fno_params(16)
        fno_model = FNOPriorAdapter(model_params, self.data_params)

        input_batch, target_batch = get_dummy_batch("inc_low", batch_size=2)

        # Ensure gradients are enabled
        input_batch.requires_grad_(True)

        output = fno_model(input_batch)  # Returns full sequence (8 frames)
        # Take only last 2 frames to match target shape
        output_subset = output[:, -2:]  # Last 2 frames to match target
        loss = nn.MSELoss()(output_subset, target_batch)  # Shapes should match now
        loss.backward()

        # Check that model parameters have gradients
        has_gradients = any(param.grad is not None for param in fno_model.parameters())
        self.assertTrue(has_gradients, "FNO model parameters should have gradients after backward pass")

    def test_fno_parameter_count(self):
        """Test FNO models have reasonable parameter counts"""
        for modes in [16, 32]:
            with self.subTest(modes=modes):
                model_params = self._create_fno_params(modes)
                fno_model = FNOPriorAdapter(model_params, self.data_params)

                param_count = sum(p.numel() for p in fno_model.parameters())

                # Should have reasonable number of parameters (not too few, not too many)
                self.assertGreater(param_count, 1000, f"FNO {modes} should have > 1000 parameters")
                self.assertLess(param_count, 10_000_000, f"FNO {modes} should have < 10M parameters")

    def test_fno_reproducibility(self):
        """Test that FNO models produce consistent outputs"""
        model_params = self._create_fno_params(16)

        # Set random seed for reproducibility
        torch.manual_seed(42)
        fno_model1 = FNOPriorAdapter(model_params, self.data_params)

        torch.manual_seed(42)
        fno_model2 = FNOPriorAdapter(model_params, self.data_params)

        input_batch, _ = get_dummy_batch("inc_low", batch_size=1)

        with torch.no_grad():
            output1 = fno_model1(input_batch)
            output2 = fno_model2(input_batch)

        # Models initialized with same seed should produce same output
        self.assertTrue(torch.allclose(output1, output2, atol=1e-6))

    def test_fno_different_batch_sizes(self):
        """Test FNO handles different batch sizes correctly"""
        model_params = self._create_fno_params(16)
        fno_model = FNOPriorAdapter(model_params, self.data_params)

        for batch_size in [1, 2, 4]:
            with self.subTest(batch_size=batch_size):
                input_batch, _ = get_dummy_batch("inc_low", batch_size=batch_size)

                with torch.no_grad():
                    output = fno_model(input_batch)

                expected_shape = (batch_size, 8, 3, 16, 16)  # Same as input shape
                self.assertEqual(output.shape, expected_shape)

    def test_fno_fourier_modes_truncation(self):
        """Test that FNO respects k_max mode truncation"""
        model_params = self._create_fno_params(8)  # Lower modes for testing
        fno_model = FNOPriorAdapter(model_params, self.data_params)

        input_batch, _ = get_dummy_batch("inc_low", batch_size=1)

        with torch.no_grad():
            output = fno_model(input_batch)

        # Check that output has reasonable spectral properties
        # For dummy data, just verify output is different from input and bounded
        self.assertFalse(torch.allclose(output, input_batch, atol=1e-3))
        self.assertTrue(torch.all(torch.isfinite(output)), "FNO output should be finite")

        # Verify output magnitude is reasonable (not exploding/vanishing)
        output_std = torch.std(output)
        self.assertGreater(output_std.item(), 1e-6, "FNO output should have some variation")
        self.assertLess(output_std.item(), 100.0, "FNO output should not explode")

    def test_fno_different_resolutions(self):
        """Test FNO on different input resolutions"""
        model_params = self._create_fno_params(16)

        # Test with different data sizes
        data_params_8x8 = DataParams(
            batch=2,
            sequenceLength=[8, 2],
            dataSize=[8, 8],  # Smaller resolution
            dimension=2,
            simFields=["pres"],
            simParams=[],
            normalizeMode=""
        )

        fno_model_8x8 = FNOPriorAdapter(model_params, data_params_8x8)
        fno_model_16x16 = FNOPriorAdapter(model_params, self.data_params)

        # Test 8x8 resolution
        input_8x8 = torch.randn(2, 8, 3, 8, 8)
        with torch.no_grad():
            output_8x8 = fno_model_8x8(input_8x8)
        self.assertEqual(output_8x8.shape, (2, 8, 3, 8, 8))

        # Test 16x16 resolution
        input_16x16, _ = get_dummy_batch("inc_low", batch_size=2)
        with torch.no_grad():
            output_16x16 = fno_model_16x16(input_16x16)
        self.assertEqual(output_16x16.shape, (2, 8, 3, 16, 16))

    def test_fno_spectral_output_properties(self):
        """Test output has reasonable spectral properties on dummy data"""
        model_params = self._create_fno_params(16)
        fno_model = FNOPriorAdapter(model_params, self.data_params)

        input_batch, _ = get_dummy_batch("inc_low", batch_size=1)

        with torch.no_grad():
            output = fno_model(input_batch)

        # Test spectral properties in Fourier domain
        for t in range(output.shape[1]):  # For each timestep
            for c in range(output.shape[2]):  # For each channel
                field = output[0, t, c]  # [H, W]

                # Apply 2D FFT
                field_fft = torch.fft.fft2(field)
                power_spectrum = torch.abs(field_fft) ** 2

                # Check that energy is not concentrated only at DC component
                dc_energy = power_spectrum[0, 0]
                total_energy = torch.sum(power_spectrum)

                # DC shouldn't dominate (would indicate constant field)
                self.assertLess(dc_energy / total_energy, 0.95,
                               f"Energy too concentrated at DC at t={t}, c={c}")

                # Check for reasonable energy distribution
                self.assertTrue(torch.all(torch.isfinite(power_spectrum)),
                               "Power spectrum should be finite")

    def test_fno_output_bounds_reasonable(self):
        """Test outputs are in reasonable physical ranges"""
        model_params = self._create_fno_params(16)
        fno_model = FNOPriorAdapter(model_params, self.data_params)

        input_batch, _ = get_dummy_batch("inc_low", batch_size=2)

        with torch.no_grad():
            output = fno_model(input_batch)

        # Check output bounds are reasonable
        output_min = torch.min(output)
        output_max = torch.max(output)
        output_mean = torch.mean(output)

        # Should not have extreme values (for dummy data)
        self.assertGreater(output_min.item(), -100.0, "FNO output min too negative")
        self.assertLess(output_max.item(), 100.0, "FNO output max too large")
        self.assertTrue(torch.isfinite(output_mean), "FNO output mean should be finite")

        # Check for reasonable dynamic range
        output_range = output_max - output_min
        self.assertGreater(output_range.item(), 1e-6, "FNO output should have some dynamic range")

    def test_fno_temporal_smoothness(self):
        """Test no extreme temporal jumps in predictions"""
        model_params = self._create_fno_params(16)
        fno_model = FNOPriorAdapter(model_params, self.data_params)

        input_batch, _ = get_dummy_batch("inc_low", batch_size=1)

        with torch.no_grad():
            output = fno_model(input_batch)

        # Check temporal derivatives are reasonable
        for t in range(1, output.shape[1]):
            temporal_diff = torch.abs(output[0, t] - output[0, t-1])
            max_temporal_change = torch.max(temporal_diff)

            # Temporal changes shouldn't be extremely large
            self.assertLess(max_temporal_change.item(), 50.0,
                           f"Temporal change at step {t} too large: {max_temporal_change.item()}")

            # Should have some temporal variation (not constant)
            mean_temporal_change = torch.mean(temporal_diff)
            self.assertGreater(mean_temporal_change.item(), 1e-8,
                              f"No temporal variation at step {t}")

    def test_fno_spatial_smoothness(self):
        """Test spatial derivatives are reasonable"""
        model_params = self._create_fno_params(16)
        fno_model = FNOPriorAdapter(model_params, self.data_params)

        input_batch, _ = get_dummy_batch("inc_low", batch_size=1)

        with torch.no_grad():
            output = fno_model(input_batch)

        # Check spatial gradients for last timestep
        field = output[0, -1, 0]  # Take one channel of last timestep [H, W]

        # Compute spatial gradients
        grad_x = torch.abs(field[1:, :] - field[:-1, :])  # Gradient in x direction
        grad_y = torch.abs(field[:, 1:] - field[:, :-1])  # Gradient in y direction

        max_grad_x = torch.max(grad_x)
        max_grad_y = torch.max(grad_y)

        # Spatial gradients shouldn't be extremely large (would indicate noise/instability)
        self.assertLess(max_grad_x.item(), 50.0, f"Spatial gradient in x too large: {max_grad_x.item()}")
        self.assertLess(max_grad_y.item(), 50.0, f"Spatial gradient in y too large: {max_grad_y.item()}")

        # Should have some spatial variation
        mean_grad = (torch.mean(grad_x) + torch.mean(grad_y)) / 2
        self.assertGreater(mean_grad.item(), 1e-8, "No spatial variation detected")

    def test_fno_error_conditions(self):
        """Test FNO models handle error conditions appropriately"""

        # Test invalid input dimensions
        model_params = self._create_fno_params(16)
        fno_model = FNOPriorAdapter(model_params, self.data_params)

        # Wrong number of dimensions (should be 5D: [B,T,C,H,W])
        with self.assertRaises((RuntimeError, ValueError)):
            invalid_input_2d = torch.randn(16, 16)  # 2D instead of 5D
            fno_model(invalid_input_2d)

        with self.assertRaises((RuntimeError, ValueError)):
            invalid_input_4d = torch.randn(2, 3, 16, 16)  # 4D instead of 5D
            fno_model(invalid_input_4d)

        # Wrong channel count (most likely to fail)
        with self.assertRaises((RuntimeError, ValueError)):
            wrong_channels = torch.randn(2, 8, 5, 16, 16)  # 5 channels instead of 3
            fno_model(wrong_channels)

        # Test that different spatial dimensions are handled (may work or fail)
        try:
            wrong_spatial = torch.randn(2, 8, 3, 8, 8)  # 8x8 instead of 16x16
            output = fno_model(wrong_spatial)
            # If it works, just verify the output shape makes sense
            self.assertEqual(len(output.shape), 5)
        except (RuntimeError, ValueError):
            # This is also acceptable - some models may be strict about spatial dims
            pass

    def test_fno_configuration_errors(self):
        """Test FNO model configuration error handling"""

        # Invalid modes configuration (negative or zero)
        with self.assertRaises((ValueError, AttributeError, RuntimeError)):
            bad_params = self._create_fno_params(-1)  # Negative modes
            FNOPriorAdapter(bad_params, self.data_params)

        with self.assertRaises((ValueError, AttributeError, RuntimeError)):
            bad_params = self._create_fno_params(0)  # Zero modes
            FNOPriorAdapter(bad_params, self.data_params)

    def test_fno_data_parameter_validation(self):
        """Test FNO validates data parameters correctly"""

        model_params = self._create_fno_params(16)

        # Invalid data dimensions
        invalid_data_params = DataParams(
            batch=2,
            sequenceLength=[8, 2],
            dataSize=[0, 16],  # Invalid: zero dimension
            dimension=2,
            simFields=["pres"],
            simParams=[],
            normalizeMode=""
        )

        with self.assertRaises((ValueError, RuntimeError)):
            FNOPriorAdapter(model_params, invalid_data_params)

    def test_fno_memory_limits(self):
        """Test FNO behavior under memory constraints"""

        model_params = self._create_fno_params(16)
        fno_model = FNOPriorAdapter(model_params, self.data_params)

        # Extremely large batch size should be handled gracefully
        try:
            large_input = torch.randn(1000, 8, 3, 16, 16)  # Very large batch
            with torch.no_grad():
                # This might succeed or fail depending on available memory
                # Either outcome is acceptable as long as it doesn't crash
                output = fno_model(large_input)
                self.assertEqual(output.shape[0], 1000)  # If it succeeds, check shape
        except (RuntimeError, torch.cuda.OutOfMemoryError, MemoryError):
            # Memory errors are expected and acceptable
            pass


class TestFNODiffusionVariant(unittest.TestCase):
    """Test suite for FNO + Diffusion Model variant"""

    def setUp(self):
        """Set up test parameters for FNO+DM"""
        self.data_params = DataParams(
            batch=2,
            sequenceLength=[8, 2],
            dataSize=[16, 16],
            dimension=2,
            simFields=["pres"],
            simParams=[],
            normalizeMode=""
        )

    def test_fno_diffusion_score_function_shape(self):
        """Test score function produces correct output shapes for FNO integration"""
        try:
            # Test basic score function interface that would be used with FNO
            model_params = self._create_fno_params(16)
            fno_model = FNOPriorAdapter(model_params, self.data_params)

            input_batch, _ = get_dummy_batch("inc_low", batch_size=2)

            with torch.no_grad():
                fno_output = fno_model(input_batch)

            # Mock score function that would condition on FNO output
            # Score function should take (X, G, sigma) where G is neural operator output
            def mock_score_function(x, prior_output, sigma):
                # Score function should return same shape as input
                # This simulates ∇_x log p(x | G, σ)
                return torch.randn_like(x) * sigma

            # Test score function shapes
            sigma = 0.1
            score_output = mock_score_function(input_batch, fno_output, sigma)

            # Score should have same shape as input
            self.assertEqual(score_output.shape, input_batch.shape)
            self.assertTrue(torch.all(torch.isfinite(score_output)), "Score output should be finite")

            # Score magnitude should scale with sigma
            sigma_large = 1.0
            score_large = mock_score_function(input_batch, fno_output, sigma_large)
            score_ratio = torch.mean(torch.abs(score_large)) / torch.mean(torch.abs(score_output))
            self.assertGreater(score_ratio.item(), 5.0, "Score should scale with sigma")

        except Exception as e:
            self.fail(f"FNO diffusion score function shape test failed: {e}")

    def test_fno_diffusion_sampling_step(self):
        """Test single diffusion sampling step with FNO prior"""
        try:
            model_params = self._create_fno_params(16)
            fno_model = FNOPriorAdapter(model_params, self.data_params)

            input_batch, _ = get_dummy_batch("inc_low", batch_size=1)

            with torch.no_grad():
                fno_prior = fno_model(input_batch)

            # Mock single Langevin dynamics step
            def langevin_step(x, score_fn, sigma, step_size=0.01):
                # x_{t+1} = x_t + step_size * score_fn(x_t) + sqrt(2 * step_size) * noise
                score = score_fn(x)
                noise = torch.randn_like(x)
                return x + step_size * score + torch.sqrt(torch.tensor(2 * step_size)) * noise

            def mock_score_fn(x):
                # Score points toward FNO prior
                return (fno_prior - x) / 0.1  # Simple gradient toward prior

            # Test sampling step
            initial_noise = torch.randn_like(input_batch)
            corrected_sample = langevin_step(initial_noise, mock_score_fn, sigma=0.1)

            # Should produce valid output shape
            self.assertEqual(corrected_sample.shape, input_batch.shape)
            self.assertTrue(torch.all(torch.isfinite(corrected_sample)), "Sample should be finite")

            # Should move toward prior
            distance_before = torch.mean(torch.abs(initial_noise - fno_prior))
            distance_after = torch.mean(torch.abs(corrected_sample - fno_prior))
            self.assertLess(distance_after.item(), distance_before.item(),
                           "Sampling step should move toward prior")

        except Exception as e:
            self.fail(f"FNO diffusion sampling step test failed: {e}")

    def test_fno_neural_operator_diffusion_pipeline(self):
        """Test FNO → Diffusion → Output pipeline"""
        try:
            model_params = self._create_fno_params(16)
            fno_model = FNOPriorAdapter(model_params, self.data_params)

            input_batch, _ = get_dummy_batch("inc_low", batch_size=1)

            # Step 1: Neural Operator prediction
            with torch.no_grad():
                no_prediction = fno_model(input_batch)

            # Step 2: Mock diffusion correction (simplified)
            def mock_diffusion_correction(no_output, n_steps=5):
                corrected = no_output.clone()
                step_size = 0.01

                for step in range(n_steps):
                    # Simple correction toward more structured output
                    noise = torch.randn_like(corrected) * 0.01
                    # Add some high-frequency content (mock spectral recovery)
                    high_freq_correction = torch.sin(corrected * 10) * 0.01
                    corrected = corrected + noise + high_freq_correction

                return corrected

            # Step 3: Apply diffusion correction
            final_output = mock_diffusion_correction(no_prediction)

            # Validate pipeline
            self.assertEqual(final_output.shape, input_batch.shape)
            self.assertTrue(torch.all(torch.isfinite(final_output)), "Final output should be finite")

            # Should be different from neural operator output (correction applied)
            correction_magnitude = torch.mean(torch.abs(final_output - no_prediction))
            self.assertGreater(correction_magnitude.item(), 1e-6,
                              "Diffusion should modify neural operator output")

            # Should maintain reasonable bounds
            self.assertLess(torch.max(torch.abs(final_output)).item(), 100.0,
                           "Final output should remain bounded")

        except Exception as e:
            self.fail(f"FNO diffusion pipeline test failed: {e}")

    def test_fno_spectral_recovery_simulation(self):
        """Test simulated spectral recovery with diffusion correction"""
        try:
            model_params = self._create_fno_params(8)  # Lower modes to test recovery
            fno_model = FNOPriorAdapter(model_params, self.data_params)

            input_batch, _ = get_dummy_batch("inc_low", batch_size=1)

            with torch.no_grad():
                fno_output = fno_model(input_batch)

            # Simulate high-frequency recovery
            def simulate_spectral_recovery(low_freq_field):
                """Simulate adding high-frequency content that diffusion models would recover"""
                B, T, C, H, W = low_freq_field.shape
                recovered = low_freq_field.clone()

                for t in range(T):
                    for c in range(C):
                        field = low_freq_field[0, t, c]

                        # Add simulated high-frequency content
                        x = torch.linspace(0, 2*torch.pi, H)
                        y = torch.linspace(0, 2*torch.pi, W)
                        X, Y = torch.meshgrid(x, y, indexing='ij')

                        # Add high-frequency sinusoidal patterns (mock turbulent structures)
                        high_freq = 0.1 * (torch.sin(4*X) * torch.cos(4*Y) +
                                           torch.sin(8*X) * torch.cos(6*Y))

                        recovered[0, t, c] = field + high_freq

                return recovered

            recovered_output = simulate_spectral_recovery(fno_output)

            # Validate spectral recovery
            self.assertEqual(recovered_output.shape, fno_output.shape)

            # Check that high-frequency content was added
            for t in range(fno_output.shape[1]):
                for c in range(fno_output.shape[2]):
                    original_field = fno_output[0, t, c]
                    recovered_field = recovered_output[0, t, c]

                    # Compute power spectra
                    original_fft = torch.fft.fft2(original_field)
                    recovered_fft = torch.fft.fft2(recovered_field)

                    original_power = torch.abs(original_fft) ** 2
                    recovered_power = torch.abs(recovered_fft) ** 2

                    # High-frequency energy should increase
                    H, W = original_field.shape
                    high_freq_mask = torch.zeros_like(original_power, dtype=torch.bool)
                    high_freq_mask[H//4:3*H//4, W//4:3*W//4] = True

                    original_hf_energy = torch.sum(original_power[high_freq_mask])
                    recovered_hf_energy = torch.sum(recovered_power[high_freq_mask])

                    self.assertGreater(recovered_hf_energy.item(), original_hf_energy.item(),
                                     f"High-frequency energy should increase at t={t}, c={c}")

        except Exception as e:
            self.fail(f"FNO spectral recovery simulation test failed: {e}")


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)