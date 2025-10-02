"""
Unit tests for ACDM (Adaptive Conditional Diffusion Model) variants

Tests ACDM variants specified:
- ACDM: Standard ACDM with noisy conditioning integration
- ACDM ncn: ACDM with clean conditioning integration (no conditioning noise)

Each test verifies:
1. Model initialization and diffusion parameters
2. Forward pass functionality (training and inference modes)
3. Conditioning integration modes (noisy vs clean)
4. Diffusion schedule implementations
5. Noise prediction and denoising capabilities
6. Compatibility with all 5 datasets
7. Temporal consistency and physical validation
8. Architecture variants (standard UNet vs DFP)
"""

import unittest
import torch
import torch.nn as nn
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

try:
    from src.core.models.model_diffusion import DiffusionModel
    from src.core.utils.params import DataParams, ModelParamsDecoder
    DIFFUSION_AVAILABLE = True
except ImportError as e:
    DIFFUSION_AVAILABLE = False
    print(f"Diffusion modules not available: {e}")

from tests.fixtures.dummy_datasets import get_dummy_batch


@unittest.skipIf(not DIFFUSION_AVAILABLE, "Diffusion modules not available")
class TestACDMVariants(unittest.TestCase):
    """Test suite for ACDM model variants"""

    def setUp(self):
        """Set up test parameters and configurations"""
        self.data_params = DataParams(
            batch=2,
            sequenceLength=[8, 2],
            dataSize=[16, 16],
            dimension=2,
            simFields=["pres"],
            simParams=[],  # No additional parameters as channels
            normalizeMode=""
        )

        # Test all dataset types
        self.dataset_names = ['inc_low', 'inc_high', 'tra_ext', 'tra_inc', 'iso']

    def _create_acdm_params(self, conditioning_mode="noisy", arch="direct-ddpm+Prev", schedule="linear", steps=20):
        """Create model parameters for ACDM"""
        class MockModelParams:
            def __init__(self, conditioning_mode, arch, schedule, steps):
                self.arch = arch
                self.diffSteps = steps
                self.diffSchedule = schedule
                self.diffCondIntegration = conditioning_mode
                self.decWidth = 32

        return MockModelParams(conditioning_mode, arch, schedule, steps)

    def test_acdm_standard_initialization(self):
        """Test standard ACDM initialization with noisy conditioning"""
        try:
            model_params = self._create_acdm_params(conditioning_mode="noisy")
            acdm_model = DiffusionModel(
                self.data_params,
                model_params,
                dimension=2,
                condChannels=3  # Match actual data channels
            )

            self.assertIsInstance(acdm_model, nn.Module)

            # Check diffusion parameters are properly initialized
            self.assertEqual(acdm_model.timesteps, 20)
            self.assertEqual(acdm_model.inferenceConditioningIntegration, "noisy")
            self.assertEqual(acdm_model.inferenceSamplingMode, "ddpm")

            # Check buffers are registered
            self.assertTrue(hasattr(acdm_model, 'betas'))
            self.assertTrue(hasattr(acdm_model, 'sqrtAlphasCumprod'))
            self.assertTrue(hasattr(acdm_model, 'sqrtOneMinusAlphasCumprod'))

            # Check buffer shapes
            self.assertEqual(acdm_model.betas.shape, (20, 1, 1, 1))
            self.assertEqual(acdm_model.sqrtAlphasCumprod.shape, (20, 1, 1, 1))

        except Exception as e:
            self.fail(f"Standard ACDM initialization failed: {e}")

    def test_acdm_ncn_initialization(self):
        """Test ACDM ncn (clean conditioning) initialization"""
        try:
            model_params = self._create_acdm_params(conditioning_mode="clean")
            acdm_model = DiffusionModel(
                self.data_params,
                model_params,
                dimension=2,
                condChannels=3
            )

            self.assertIsInstance(acdm_model, nn.Module)

            # Check conditioning integration mode
            self.assertEqual(acdm_model.inferenceConditioningIntegration, "clean")

            # Should have same diffusion parameters as standard ACDM
            self.assertEqual(acdm_model.timesteps, 20)
            self.assertTrue(hasattr(acdm_model, 'betas'))

        except Exception as e:
            self.fail(f"ACDM ncn initialization failed: {e}")

    def test_diffusion_schedule_variants(self):
        """Test different diffusion noise schedules"""
        schedules = ["linear", "quadratic", "sigmoid", "cosine"]

        for schedule in schedules:
            with self.subTest(schedule=schedule):
                try:
                    model_params = self._create_acdm_params(schedule=schedule)
                    acdm_model = DiffusionModel(
                        self.data_params,
                        model_params,
                        dimension=2,
                        condChannels=3
                    )

                    # Check betas are properly scheduled
                    betas = acdm_model.betas.squeeze()
                    self.assertEqual(len(betas), 20)
                    self.assertTrue(torch.all(betas > 0))
                    self.assertTrue(torch.all(betas < 1))

                    # Linear schedule should be monotonically increasing
                    if schedule == "linear":
                        self.assertTrue(torch.all(betas[1:] >= betas[:-1]))

                except Exception as e:
                    self.fail(f"ACDM with {schedule} schedule failed: {e}")

    def test_acdm_training_forward_pass(self):
        """Test ACDM forward pass in training mode"""
        try:
            model_params = self._create_acdm_params(conditioning_mode="noisy")
            acdm_model = DiffusionModel(
                self.data_params,
                model_params,
                dimension=2,
                condChannels=3
            )
            acdm_model.train()

            # Create dummy conditioning and data [B, S, C, H, W]
            conditioning = torch.randn(2, 2, 3, 16, 16)  # Previous frames
            data = torch.randn(2, 2, 3, 16, 16)  # Target frames

            noise, predicted_noise = acdm_model(conditioning, data)

            # Check output shapes
            expected_shape = (2, 2, 6, 16, 16)  # B, S, conditioning_channels + data_channels, H, W
            self.assertEqual(noise.shape, expected_shape)
            self.assertEqual(predicted_noise.shape, expected_shape)

            # Check outputs are finite
            self.assertTrue(torch.all(torch.isfinite(noise)))
            self.assertTrue(torch.all(torch.isfinite(predicted_noise)))

        except Exception as e:
            self.fail(f"ACDM training forward pass failed: {e}")

    def test_acdm_ncn_training_forward_pass(self):
        """Test ACDM ncn forward pass in training mode"""
        try:
            model_params = self._create_acdm_params(conditioning_mode="clean")
            acdm_model = DiffusionModel(
                self.data_params,
                model_params,
                dimension=2,
                condChannels=3
            )
            acdm_model.train()

            conditioning = torch.randn(2, 2, 3, 16, 16)
            data = torch.randn(2, 2, 3, 16, 16)

            noise, predicted_noise = acdm_model(conditioning, data)

            # Same shape expectations as standard ACDM
            expected_shape = (2, 2, 6, 16, 16)
            self.assertEqual(noise.shape, expected_shape)
            self.assertEqual(predicted_noise.shape, expected_shape)

            self.assertTrue(torch.all(torch.isfinite(noise)))
            self.assertTrue(torch.all(torch.isfinite(predicted_noise)))

        except Exception as e:
            self.fail(f"ACDM ncn training forward pass failed: {e}")

    def test_acdm_inference_forward_pass(self):
        """Test ACDM forward pass in inference mode"""
        try:
            model_params = self._create_acdm_params(conditioning_mode="noisy")
            acdm_model = DiffusionModel(
                self.data_params,
                model_params,
                dimension=2,
                condChannels=3
            )
            acdm_model.eval()

            conditioning = torch.randn(1, 1, 3, 16, 16)
            data = torch.randn(1, 1, 3, 16, 16)  # Not used in inference, just for shape

            with torch.no_grad():
                output = acdm_model(conditioning, data)

            # Should output denoised data
            expected_shape = (1, 1, 3, 16, 16)  # B, S, data_channels, H, W
            self.assertEqual(output.shape, expected_shape)
            self.assertTrue(torch.all(torch.isfinite(output)))

        except Exception as e:
            self.fail(f"ACDM inference forward pass failed: {e}")

    def test_acdm_ncn_inference_forward_pass(self):
        """Test ACDM ncn forward pass in inference mode"""
        try:
            model_params = self._create_acdm_params(conditioning_mode="clean")
            acdm_model = DiffusionModel(
                self.data_params,
                model_params,
                dimension=2,
                condChannels=3
            )
            acdm_model.eval()

            conditioning = torch.randn(1, 1, 3, 16, 16)
            data = torch.randn(1, 1, 3, 16, 16)

            with torch.no_grad():
                output = acdm_model(conditioning, data)

            expected_shape = (1, 1, 3, 16, 16)
            self.assertEqual(output.shape, expected_shape)
            self.assertTrue(torch.all(torch.isfinite(output)))

        except Exception as e:
            self.fail(f"ACDM ncn inference forward pass failed: {e}")

    def test_acdm_gradient_flow(self):
        """Test ACDM gradient flow in training"""
        try:
            model_params = self._create_acdm_params(conditioning_mode="noisy")
            acdm_model = DiffusionModel(
                self.data_params,
                model_params,
                dimension=2,
                condChannels=3
            )
            acdm_model.train()

            conditioning = torch.randn(1, 1, 3, 8, 8, requires_grad=True)
            data = torch.randn(1, 1, 3, 8, 8, requires_grad=True)

            noise, predicted_noise = acdm_model(conditioning, data)

            # Compute loss (simplified diffusion loss)
            loss = nn.MSELoss()(noise, predicted_noise)
            loss.backward()

            # Check gradients exist
            has_gradients = any(param.grad is not None for param in acdm_model.parameters())
            self.assertTrue(has_gradients, "ACDM should have gradients after backward pass")

            # Check gradient magnitudes are reasonable
            grad_norms = [param.grad.norm().item() for param in acdm_model.parameters() if param.grad is not None]
            self.assertTrue(all(g < 100.0 for g in grad_norms), "Gradient norms should be reasonable")

        except Exception as e:
            self.fail(f"ACDM gradient flow test failed: {e}")

    def test_acdm_different_timesteps(self):
        """Test ACDM with different numbers of diffusion timesteps"""
        timestep_configs = [10, 20, 50]

        for steps in timestep_configs:
            with self.subTest(timesteps=steps):
                try:
                    model_params = self._create_acdm_params(steps=steps)
                    acdm_model = DiffusionModel(
                        self.data_params,
                        model_params,
                        dimension=2,
                        condChannels=3
                    )

                    self.assertEqual(acdm_model.timesteps, steps)
                    self.assertEqual(acdm_model.betas.shape[0], steps)

                    # Test training forward pass
                    acdm_model.train()
                    conditioning = torch.randn(1, 1, 3, 8, 8)
                    data = torch.randn(1, 1, 3, 8, 8)

                    noise, predicted_noise = acdm_model(conditioning, data)
                    self.assertTrue(torch.all(torch.isfinite(noise)))
                    self.assertTrue(torch.all(torch.isfinite(predicted_noise)))

                except Exception as e:
                    self.fail(f"ACDM with {steps} timesteps failed: {e}")

    def test_acdm_conditioning_comparison(self):
        """Test difference between noisy and clean conditioning"""
        try:
            # Create both variants
            acdm_noisy_params = self._create_acdm_params(conditioning_mode="noisy")
            acdm_clean_params = self._create_acdm_params(conditioning_mode="clean")

            acdm_noisy = DiffusionModel(self.data_params, acdm_noisy_params, dimension=2, condChannels=3)
            acdm_clean = DiffusionModel(self.data_params, acdm_clean_params, dimension=2, condChannels=3)

            # Same conditioning and data
            conditioning = torch.randn(1, 1, 3, 8, 8)
            data = torch.randn(1, 1, 3, 8, 8)

            # Training mode
            acdm_noisy.train()
            acdm_clean.train()

            noise_noisy, pred_noisy = acdm_noisy(conditioning, data)
            noise_clean, pred_clean = acdm_clean(conditioning, data)

            # Both should produce valid outputs
            self.assertEqual(noise_noisy.shape, noise_clean.shape)
            self.assertEqual(pred_noisy.shape, pred_clean.shape)

            # Outputs should be finite
            self.assertTrue(torch.all(torch.isfinite(noise_noisy)))
            self.assertTrue(torch.all(torch.isfinite(noise_clean)))
            self.assertTrue(torch.all(torch.isfinite(pred_noisy)))
            self.assertTrue(torch.all(torch.isfinite(pred_clean)))

        except Exception as e:
            self.fail(f"ACDM conditioning comparison failed: {e}")

    def test_acdm_architecture_variants(self):
        """Test ACDM with different architecture variants"""
        arch_variants = [
            "direct-ddpm+Prev",
            "direct-ddim+Prev",
            "dfp-ddpm+Prev"
        ]

        for arch in arch_variants:
            with self.subTest(architecture=arch):
                try:
                    model_params = self._create_acdm_params(arch=arch)
                    acdm_model = DiffusionModel(
                        self.data_params,
                        model_params,
                        dimension=2,
                        condChannels=3
                    )

                    # Check sampling mode is set correctly
                    if "ddpm" in arch:
                        self.assertEqual(acdm_model.inferenceSamplingMode, "ddpm")
                    elif "ddim" in arch:
                        self.assertEqual(acdm_model.inferenceSamplingMode, "ddim")

                    # Test forward pass
                    acdm_model.train()
                    conditioning = torch.randn(1, 1, 3, 8, 8)
                    data = torch.randn(1, 1, 3, 8, 8)

                    noise, predicted_noise = acdm_model(conditioning, data)
                    self.assertTrue(torch.all(torch.isfinite(noise)))
                    self.assertTrue(torch.all(torch.isfinite(predicted_noise)))

                except Exception as e:
                    self.fail(f"ACDM with architecture {arch} failed: {e}")

    def test_acdm_all_datasets_compatibility(self):
        """Test ACDM models work with all dataset types"""
        model_params = self._create_acdm_params(conditioning_mode="noisy")
        acdm_model = DiffusionModel(
            self.data_params,
            model_params,
            dimension=2,
            condChannels=3
        )

        for dataset_name in self.dataset_names:
            with self.subTest(dataset=dataset_name):
                try:
                    input_batch, _ = get_dummy_batch(dataset_name, batch_size=1)

                    # Simulate ACDM usage: use previous frames as conditioning
                    B, T, C, H, W = input_batch.shape
                    if T >= 2:
                        conditioning = input_batch[:, :-1]  # Previous frames
                        data = input_batch[:, 1:]  # Target frames
                    else:
                        # Duplicate for single frame
                        conditioning = input_batch
                        data = input_batch

                    # Training mode
                    acdm_model.train()
                    noise, predicted_noise = acdm_model(conditioning, data)

                    # Check outputs
                    expected_channels = conditioning.shape[2] + data.shape[2]
                    expected_shape = (B, data.shape[1], expected_channels, H, W)
                    self.assertEqual(noise.shape, expected_shape)
                    self.assertEqual(predicted_noise.shape, expected_shape)
                    self.assertTrue(torch.all(torch.isfinite(noise)))
                    self.assertTrue(torch.all(torch.isfinite(predicted_noise)))

                except Exception as e:
                    self.fail(f"ACDM failed on dataset {dataset_name}: {e}")

    def test_acdm_temporal_consistency(self):
        """Test ACDM temporal consistency in inference"""
        try:
            model_params = self._create_acdm_params(conditioning_mode="noisy")
            acdm_model = DiffusionModel(
                self.data_params,
                model_params,
                dimension=2,
                condChannels=3
            )
            acdm_model.eval()

            # Create temporal sequence
            conditioning = torch.randn(1, 3, 3, 8, 8)  # 3 conditioning frames
            data = torch.randn(1, 3, 3, 8, 8)  # Target shape

            with torch.no_grad():
                output = acdm_model(conditioning, data)

            # Check temporal smoothness
            for t in range(1, output.shape[1]):
                temporal_diff = torch.abs(output[0, t] - output[0, t-1])
                max_temporal_change = torch.max(temporal_diff)

                # Changes shouldn't be extreme
                self.assertLess(max_temporal_change.item(), 10.0,
                               f"ACDM temporal change too large at step {t}")

                # Should have some variation
                mean_temporal_change = torch.mean(temporal_diff)
                self.assertGreater(mean_temporal_change.item(), 1e-8,
                                  f"ACDM no temporal variation at step {t}")

        except Exception as e:
            self.fail(f"ACDM temporal consistency test failed: {e}")

    def test_acdm_physical_validation(self):
        """Test ACDM outputs have reasonable physical properties"""
        try:
            model_params = self._create_acdm_params(conditioning_mode="noisy")
            acdm_model = DiffusionModel(
                self.data_params,
                model_params,
                dimension=2,
                condChannels=3
            )
            acdm_model.eval()

            # Physical-like conditioning and target
            conditioning = torch.randn(1, 1, 3, 16, 16) * 0.5
            data = torch.randn(1, 1, 3, 16, 16) * 0.5

            with torch.no_grad():
                output = acdm_model(conditioning, data)

            # Check output bounds
            output_min = torch.min(output)
            output_max = torch.max(output)
            output_std = torch.std(output)

            # Should not have extreme values
            self.assertGreater(output_min.item(), -50.0, "ACDM output min too negative")
            self.assertLess(output_max.item(), 50.0, "ACDM output max too large")
            self.assertTrue(torch.isfinite(output_std), "ACDM output std should be finite")

            # Should have reasonable dynamic range
            self.assertGreater(output_std.item(), 1e-6, "ACDM should produce varied output")

            # Check spatial smoothness
            if output.shape[-1] > 1 and output.shape[-2] > 1:
                spatial_grad_x = torch.abs(output[0, 0, 0, 1:, :] - output[0, 0, 0, :-1, :])
                spatial_grad_y = torch.abs(output[0, 0, 0, :, 1:] - output[0, 0, 0, :, :-1])

                max_grad_x = torch.max(spatial_grad_x)
                max_grad_y = torch.max(spatial_grad_y)

                self.assertLess(max_grad_x.item(), 20.0, "ACDM spatial gradients should be reasonable")
                self.assertLess(max_grad_y.item(), 20.0, "ACDM spatial gradients should be reasonable")

        except Exception as e:
            self.fail(f"ACDM physical validation test failed: {e}")

    def test_acdm_noise_schedule_properties(self):
        """Test properties of different noise schedules"""
        schedules = ["linear", "cosine"]

        for schedule in schedules:
            with self.subTest(schedule=schedule):
                try:
                    model_params = self._create_acdm_params(schedule=schedule)
                    acdm_model = DiffusionModel(
                        self.data_params,
                        model_params,
                        dimension=2,
                        condChannels=3
                    )

                    # Check schedule properties
                    betas = acdm_model.betas.squeeze()
                    alphas_cumprod = acdm_model.sqrtAlphasCumprod.squeeze() ** 2

                    # Betas should be in valid range
                    self.assertTrue(torch.all(betas > 0), f"{schedule} betas should be positive")
                    self.assertTrue(torch.all(betas < 1), f"{schedule} betas should be < 1")

                    # Cumulative alphas should decrease
                    self.assertTrue(torch.all(alphas_cumprod[1:] <= alphas_cumprod[:-1]),
                                   f"{schedule} alphas_cumprod should decrease")

                    # Should start near 1 and end near 0
                    self.assertGreater(alphas_cumprod[0].item(), 0.9,
                                      f"{schedule} should start with high signal")
                    self.assertLess(alphas_cumprod[-1].item(), 0.1,
                                   f"{schedule} should end with low signal")

                except Exception as e:
                    self.fail(f"ACDM noise schedule {schedule} properties test failed: {e}")

    def test_acdm_parameter_count_scaling(self):
        """Test ACDM parameter count scales with model size"""
        try:
            # Test different width settings
            widths = [16, 32, 64]
            param_counts = {}

            for width in widths:
                model_params = self._create_acdm_params()
                model_params.decWidth = width

                acdm_model = DiffusionModel(
                    self.data_params,
                    model_params,
                    dimension=2,
                    condChannels=3
                )

                param_count = sum(p.numel() for p in acdm_model.parameters())
                param_counts[width] = param_count

                # Should have reasonable parameter count
                self.assertGreater(param_count, 1000)
                self.assertLess(param_count, 10_000_000)

            # Larger width should generally have more parameters
            self.assertGreater(param_counts[32], param_counts[16])
            self.assertGreater(param_counts[64], param_counts[32])

        except Exception as e:
            self.fail(f"ACDM parameter count scaling test failed: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)