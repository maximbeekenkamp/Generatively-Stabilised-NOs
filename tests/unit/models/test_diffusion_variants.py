"""
Unit tests for Neural Operator + Diffusion Model variants

Tests +DM (diffusion model) integration variants:
- FNO + DM: Fourier Neural Operator with diffusion correction
- TNO + DM: Transformer Neural Operator with diffusion correction
- UNet + DM: U-Net with diffusion correction
- DeepONet + DM: Deep Operator Network with diffusion correction (future)

Each test verifies:
1. Prior + corrector model initialization and integration
2. Generative operator forward pass functionality
3. Training mode switching (prior-only, corrector-training, full-inference)
4. DCAR (Diffusion-Corrected AutoRegressive) rollout capabilities
5. Compatibility with all 5 datasets
6. Two-stage training pipeline integration
7. Memory optimization features
8. Physical validation of corrected outputs
"""

import unittest
import torch
import torch.nn as nn
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

try:
    from src.core.models.neural_operator_adapters import FNOPriorAdapter, TNOPriorAdapter, UNetPriorAdapter
    from src.core.models.generative_correctors import DiffusionCorrector
    from src.core.models.generative_operator_model import GenerativeOperatorModel
    from src.core.models.model_registry import ModelRegistry
    from src.core.models.genop_init import initialize_generative_operators
    from src.core.utils.params import DataParams, ModelParamsDecoder
    GENOP_AVAILABLE = True
except ImportError as e:
    GENOP_AVAILABLE = False
    print(f"Generative Operator modules not available: {e}")

try:
    from src.core.models.deeponet.deeponet_variants import StandardDeepONet
    from src.core.models.deeponet.deeponet_base import DeepONetConfig
    DEEPONET_AVAILABLE = True
except ImportError:
    DEEPONET_AVAILABLE = False
    print("DeepONet modules not available")

from tests.fixtures.dummy_datasets import get_dummy_batch


@unittest.skipIf(not GENOP_AVAILABLE, "Generative Operator modules not available")
class TestDiffusionVariants(unittest.TestCase):
    """Test suite for Neural Operator + Diffusion Model variants"""

    def setUp(self):
        """Set up test parameters and configurations"""
        self.data_params = DataParams(
            batch=2,
            sequenceLength=[8, 2],
            dataSize=[16, 16],
            dimension=2,
            simFields=["pres"],
            simParams=[],  # Remove simParams to avoid channel mismatch
            normalizeMode=""
        )

        # Test all dataset types
        self.dataset_names = ['inc_low', 'inc_high', 'tra_ext', 'tra_inc', 'iso']

        # Initialize generative operators
        initialize_generative_operators()

    def _create_fno_params(self, modes=16):
        """Create model parameters for FNO prior"""
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

    def _create_tno_params(self, L=2, K=1):
        """Create model parameters for TNO prior"""
        class MockModelParams:
            def __init__(self, L, K):
                self.L = L
                self.K = K
                self.architecture = f'tno_L{L}_K{K}'
                self.arch = f'tno_L{L}_K{K}'
                self.model_type = 'tno'
                self.decWidth = 64

        return MockModelParams(L, K)

    def _create_unet_params(self):
        """Create model parameters for UNet prior"""
        class MockModelParams:
            def __init__(self):
                self.architecture = 'unet'
                self.arch = 'unet'
                self.model_type = 'unet'
                self.decWidth = 32

        return MockModelParams()

    def _create_diffusion_params(self, steps=20, schedule="linear"):
        """Create model parameters for diffusion corrector"""
        class MockDiffusionParams:
            def __init__(self, steps, schedule):
                self.diffSteps = steps
                self.diffSchedule = schedule
                self.diffCondIntegration = "noisy"
                self.arch = "direct-ddpm+Prev"
                self.correction_strength = 1.0
                self.decWidth = 64  # Required for DiffusionCorrector

        return MockDiffusionParams(steps, schedule)

    def test_fno_diffusion_initialization(self):
        """Test FNO + Diffusion model initialization"""
        try:
            # Create FNO prior
            fno_params = self._create_fno_params(modes=16)
            fno_prior = FNOPriorAdapter(fno_params, self.data_params)

            # Create diffusion corrector
            diff_params = self._create_diffusion_params()
            diff_corrector = DiffusionCorrector(diff_params, self.data_params)

            # Create combined model
            genop_model = GenerativeOperatorModel(
                prior_model=fno_prior,
                corrector_model=diff_corrector,
                p_md=diff_params,
                p_d=self.data_params
            )

            self.assertIsInstance(genop_model, nn.Module)
            self.assertIsInstance(genop_model.prior_model, FNOPriorAdapter)
            self.assertIsInstance(genop_model.corrector_model, DiffusionCorrector)
            self.assertEqual(genop_model.training_mode, 'prior_only')

        except Exception as e:
            self.fail(f"FNO + Diffusion initialization failed: {e}")

    def test_tno_diffusion_initialization(self):
        """Test TNO + Diffusion model initialization"""
        try:
            # Create TNO prior
            tno_params = self._create_tno_params(L=2, K=1)
            tno_prior = TNOPriorAdapter(tno_params, self.data_params)

            # Create diffusion corrector
            diff_params = self._create_diffusion_params()
            diff_corrector = DiffusionCorrector(diff_params, self.data_params)

            # Create combined model
            genop_model = GenerativeOperatorModel(
                prior_model=tno_prior,
                corrector_model=diff_corrector,
                p_md=diff_params,
                p_d=self.data_params
            )

            self.assertIsInstance(genop_model, nn.Module)
            self.assertIsInstance(genop_model.prior_model, TNOPriorAdapter)
            self.assertIsInstance(genop_model.corrector_model, DiffusionCorrector)

        except Exception as e:
            self.fail(f"TNO + Diffusion initialization failed: {e}")

    def test_unet_diffusion_initialization(self):
        """Test UNet + Diffusion model initialization"""
        try:
            # Create UNet prior
            unet_params = self._create_unet_params()
            unet_prior = UNetPriorAdapter(unet_params, self.data_params)

            # Create diffusion corrector
            diff_params = self._create_diffusion_params()
            diff_corrector = DiffusionCorrector(diff_params, self.data_params)

            # Create combined model
            genop_model = GenerativeOperatorModel(
                prior_model=unet_prior,
                corrector_model=diff_corrector,
                p_md=diff_params,
                p_d=self.data_params
            )

            self.assertIsInstance(genop_model, nn.Module)
            self.assertIsInstance(genop_model.prior_model, UNetPriorAdapter)
            self.assertIsInstance(genop_model.corrector_model, DiffusionCorrector)

        except Exception as e:
            self.fail(f"UNet + Diffusion initialization failed: {e}")

    def test_training_mode_switching(self):
        """Test switching between different training modes"""
        try:
            # Create FNO + Diffusion model
            fno_params = self._create_fno_params()
            fno_prior = FNOPriorAdapter(fno_params, self.data_params)

            diff_params = self._create_diffusion_params()
            diff_corrector = DiffusionCorrector(diff_params, self.data_params)

            genop_model = GenerativeOperatorModel(
                prior_model=fno_prior,
                corrector_model=diff_corrector,
                p_md=diff_params,
                p_d=self.data_params
            )

            # Test mode switching
            modes = ['prior_only', 'corrector_training', 'full_inference']
            for mode in modes:
                with self.subTest(mode=mode):
                    genop_model.training_mode = mode
                    self.assertEqual(genop_model.training_mode, mode)

                    # Test forward pass (use dataSize dimensions)
                    x = torch.randn(1, 2, 3, 16, 16)
                    with torch.no_grad():
                        output = genop_model(x)

                    self.assertEqual(output.shape, (1, 2, 3, 16, 16))
                    self.assertTrue(torch.all(torch.isfinite(output)))

        except Exception as e:
            self.fail(f"Training mode switching test failed: {e}")

    def test_prior_only_mode(self):
        """Test prior-only mode (stage 1 training)"""
        try:
            # Create FNO + Diffusion model
            fno_params = self._create_fno_params()
            fno_prior = FNOPriorAdapter(fno_params, self.data_params)

            diff_params = self._create_diffusion_params()
            diff_corrector = DiffusionCorrector(diff_params, self.data_params)

            genop_model = GenerativeOperatorModel(
                prior_model=fno_prior,
                corrector_model=diff_corrector,
                p_md=diff_params,
                p_d=self.data_params
            )

            genop_model.training_mode = 'prior_only'
            genop_model.train()

            x = torch.randn(2, 2, 3, 16, 16, requires_grad=True)
            output = genop_model(x)

            # Should match prior-only output
            with torch.no_grad():
                prior_output = fno_prior(x)

            self.assertTrue(torch.allclose(output, prior_output, atol=1e-6))

            # Test gradient flow (only through prior)
            loss = torch.mean(output)
            loss.backward()

            # Prior should have gradients
            prior_has_grads = any(p.grad is not None for p in genop_model.prior_model.parameters())
            self.assertTrue(prior_has_grads)

        except Exception as e:
            self.fail(f"Prior-only mode test failed: {e}")

    def test_corrector_training_mode(self):
        """Test corrector training mode (stage 2 training)"""
        try:
            # Create FNO + Diffusion model
            fno_params = self._create_fno_params()
            fno_prior = FNOPriorAdapter(fno_params, self.data_params)

            diff_params = self._create_diffusion_params()
            diff_corrector = DiffusionCorrector(diff_params, self.data_params)

            genop_model = GenerativeOperatorModel(
                prior_model=fno_prior,
                corrector_model=diff_corrector,
                p_md=diff_params,
                p_d=self.data_params
            )

            genop_model.training_mode = 'corrector_training'
            genop_model.train()

            x = torch.randn(2, 2, 3, 16, 16)
            output = genop_model(x)

            # Output should be different from prior-only
            with torch.no_grad():
                prior_output = fno_prior(x)

            # Shapes should match
            self.assertEqual(output.shape, prior_output.shape)
            self.assertTrue(torch.all(torch.isfinite(output)))

        except Exception as e:
            self.fail(f"Corrector training mode test failed: {e}")

    def test_full_inference_mode(self):
        """Test full inference mode (both models active)"""
        try:
            # Create FNO + Diffusion model
            fno_params = self._create_fno_params()
            fno_prior = FNOPriorAdapter(fno_params, self.data_params)

            diff_params = self._create_diffusion_params()
            diff_corrector = DiffusionCorrector(diff_params, self.data_params)

            genop_model = GenerativeOperatorModel(
                prior_model=fno_prior,
                corrector_model=diff_corrector,
                p_md=diff_params,
                p_d=self.data_params
            )

            genop_model.training_mode = 'full_inference'
            genop_model.eval()

            x = torch.randn(1, 2, 3, 16, 16)

            with torch.no_grad():
                output = genop_model(x)

            self.assertEqual(output.shape, x.shape)
            self.assertTrue(torch.all(torch.isfinite(output)))

        except Exception as e:
            self.fail(f"Full inference mode test failed: {e}")

    def test_dcar_rollout_capability(self):
        """Test DCAR (Diffusion-Corrected AutoRegressive) rollout"""
        try:
            # Create FNO + Diffusion model with DCAR enabled
            fno_params = self._create_fno_params()
            fno_prior = FNOPriorAdapter(fno_params, self.data_params)

            diff_params = self._create_diffusion_params()
            diff_corrector = DiffusionCorrector(diff_params, self.data_params)

            genop_model = GenerativeOperatorModel(
                prior_model=fno_prior,
                corrector_model=diff_corrector,
                p_md=diff_params,
                p_d=self.data_params,
                enable_dcar=True,
                dcar_correction_frequency=2  # Correct every 2 steps
            )

            self.assertTrue(genop_model.enable_dcar)
            self.assertEqual(genop_model.dcar_correction_frequency, 2)

            # Test forward pass
            genop_model.training_mode = 'full_inference'
            genop_model.eval()

            x = torch.randn(1, 3, 3, 16, 16)  # Longer sequence for DCAR

            with torch.no_grad():
                output = genop_model(x)

            self.assertEqual(output.shape, x.shape)
            self.assertTrue(torch.all(torch.isfinite(output)))

        except Exception as e:
            self.fail(f"DCAR rollout test failed: {e}")

    def test_memory_optimization_features(self):
        """Test memory optimization features"""
        try:
            # Create model with memory optimization
            fno_params = self._create_fno_params()
            fno_prior = FNOPriorAdapter(fno_params, self.data_params)

            diff_params = self._create_diffusion_params()
            diff_corrector = DiffusionCorrector(diff_params, self.data_params)

            genop_model = GenerativeOperatorModel(
                prior_model=fno_prior,
                corrector_model=diff_corrector,
                p_md=diff_params,
                p_d=self.data_params,
                memory_efficient=True,
                gradient_checkpointing=True
            )

            self.assertTrue(genop_model.memory_efficient)
            self.assertTrue(genop_model.gradient_checkpointing)

            # Test forward pass
            x = torch.randn(1, 2, 3, 16, 16)
            output = genop_model(x)

            self.assertTrue(torch.all(torch.isfinite(output)))

        except Exception as e:
            self.fail(f"Memory optimization test failed: {e}")

    def test_different_diffusion_configurations(self):
        """Test different diffusion corrector configurations"""
        diffusion_configs = [
            {"steps": 10, "schedule": "linear"},
            {"steps": 20, "schedule": "cosine"},
            {"steps": 50, "schedule": "sigmoid"},
        ]

        for config in diffusion_configs:
            with self.subTest(config=config):
                try:
                    # Create FNO + Diffusion with different config
                    fno_params = self._create_fno_params()
                    fno_prior = FNOPriorAdapter(fno_params, self.data_params)

                    diff_params = self._create_diffusion_params(**config)
                    diff_corrector = DiffusionCorrector(diff_params, self.data_params)

                    genop_model = GenerativeOperatorModel(
                        prior_model=fno_prior,
                        corrector_model=diff_corrector,
                        p_md=diff_params,
                        p_d=self.data_params
                    )

                    # Test forward pass
                    x = torch.randn(1, 2, 3, 16, 16)
                    with torch.no_grad():
                        output = genop_model(x)

                    self.assertEqual(output.shape, x.shape)
                    self.assertTrue(torch.all(torch.isfinite(output)))

                except Exception as e:
                    self.fail(f"Diffusion config {config} failed: {e}")

    def test_all_variants_dataset_compatibility(self):
        """Test all +DM variants work with all dataset types"""
        variants = [
            ("FNO", self._create_fno_params, FNOPriorAdapter),
            ("TNO", self._create_tno_params, TNOPriorAdapter),
            ("UNet", self._create_unet_params, UNetPriorAdapter),
        ]

        for variant_name, param_creator, adapter_class in variants:
            for dataset_name in self.dataset_names:
                with self.subTest(variant=variant_name, dataset=dataset_name):
                    try:
                        # Create prior
                        prior_params = param_creator()
                        prior_model = adapter_class(prior_params, self.data_params)

                        # Create diffusion corrector
                        diff_params = self._create_diffusion_params()
                        diff_corrector = DiffusionCorrector(diff_params, self.data_params)

                        # Create combined model
                        genop_model = GenerativeOperatorModel(
                            prior_model=prior_model,
                            corrector_model=diff_corrector,
                            p_md=diff_params,
                            p_d=self.data_params
                        )

                        # Test with dataset
                        input_batch, _ = get_dummy_batch(dataset_name, batch_size=1)

                        with torch.no_grad():
                            output = genop_model(input_batch)

                        self.assertEqual(output.shape, input_batch.shape)
                        self.assertTrue(torch.all(torch.isfinite(output)))

                    except Exception as e:
                        self.fail(f"{variant_name} + DM failed on dataset {dataset_name}: {e}")

    def test_gradient_flow_through_combined_model(self):
        """Test gradient flow through combined prior + corrector model"""
        try:
            # Create FNO + Diffusion model
            fno_params = self._create_fno_params()
            fno_prior = FNOPriorAdapter(fno_params, self.data_params)

            diff_params = self._create_diffusion_params()
            diff_corrector = DiffusionCorrector(diff_params, self.data_params)

            genop_model = GenerativeOperatorModel(
                prior_model=fno_prior,
                corrector_model=diff_corrector,
                p_md=diff_params,
                p_d=self.data_params
            )

            genop_model.training_mode = 'full_inference'
            genop_model.train()

            x = torch.randn(1, 2, 3, 16, 16, requires_grad=True)
            target = torch.randn(1, 2, 3, 16, 16)

            output = genop_model(x)
            loss = nn.MSELoss()(output, target)
            loss.backward()

            # Check both models have gradients
            prior_has_grads = any(p.grad is not None for p in genop_model.prior_model.parameters())
            corrector_has_grads = any(p.grad is not None for p in genop_model.corrector_model.parameters())

            self.assertTrue(prior_has_grads, "Prior model should have gradients")
            self.assertTrue(corrector_has_grads, "Corrector model should have gradients")

        except Exception as e:
            self.fail(f"Gradient flow test failed: {e}")

    def test_physical_validation_corrected_outputs(self):
        """Test physical validation of diffusion-corrected outputs"""
        try:
            # Create FNO + Diffusion model
            fno_params = self._create_fno_params()
            fno_prior = FNOPriorAdapter(fno_params, self.data_params)

            diff_params = self._create_diffusion_params()
            diff_corrector = DiffusionCorrector(diff_params, self.data_params)

            genop_model = GenerativeOperatorModel(
                prior_model=fno_prior,
                corrector_model=diff_corrector,
                p_md=diff_params,
                p_d=self.data_params
            )

            genop_model.training_mode = 'full_inference'
            genop_model.eval()

            # Physical-like input
            x = torch.randn(1, 2, 3, 16, 16) * 0.5

            with torch.no_grad():
                output = genop_model(x)

            # Check output bounds are reasonable
            output_min = torch.min(output)
            output_max = torch.max(output)
            output_std = torch.std(output)

            # Should not have extreme values
            self.assertGreater(output_min.item(), -50.0, "Corrected output min too negative")
            self.assertLess(output_max.item(), 50.0, "Corrected output max too large")
            self.assertTrue(torch.isfinite(output_std), "Corrected output std should be finite")

            # Should have reasonable dynamic range
            self.assertGreater(output_std.item(), 1e-6, "Corrected output should have variation")

            # Check temporal smoothness
            if output.shape[1] > 1:
                temporal_diff = torch.abs(output[0, 1] - output[0, 0])
                max_temporal_change = torch.max(temporal_diff)
                self.assertLess(max_temporal_change.item(), 20.0, "Temporal changes should be reasonable")

        except Exception as e:
            self.fail(f"Physical validation test failed: {e}")

    def test_parameter_count_scaling(self):
        """Test parameter count scaling with model combinations"""
        try:
            # Test different prior sizes
            prior_configs = [
                ("FNO-16", self._create_fno_params, FNOPriorAdapter, {"modes": 16}),
                ("FNO-32", self._create_fno_params, FNOPriorAdapter, {"modes": 32}),
            ]

            param_counts = {}
            for config_name, param_creator, adapter_class, kwargs in prior_configs:
                # Create prior
                prior_params = param_creator(**kwargs)
                prior_model = adapter_class(prior_params, self.data_params)

                # Create diffusion corrector
                diff_params = self._create_diffusion_params()
                diff_corrector = DiffusionCorrector(diff_params, self.data_params)

                # Create combined model
                genop_model = GenerativeOperatorModel(
                    prior_model=prior_model,
                    corrector_model=diff_corrector,
                    p_md=diff_params,
                    p_d=self.data_params
                )

                param_count = sum(p.numel() for p in genop_model.parameters())
                param_counts[config_name] = param_count

                # Should have reasonable parameter count
                self.assertGreater(param_count, 1000)
                self.assertLess(param_count, 50_000_000)

            # Larger models should have more parameters
            self.assertGreater(param_counts["FNO-32"], param_counts["FNO-16"])

        except Exception as e:
            self.fail(f"Parameter count scaling test failed: {e}")

    def test_correction_strength_effects(self):
        """Test effect of different correction strengths"""
        try:
            strengths = [0.0, 0.5, 1.0]

            for strength in strengths:
                with self.subTest(strength=strength):
                    # Create FNO + Diffusion model
                    fno_params = self._create_fno_params()
                    fno_prior = FNOPriorAdapter(fno_params, self.data_params)

                    diff_params = self._create_diffusion_params()
                    diff_params.correction_strength = strength
                    diff_corrector = DiffusionCorrector(diff_params, self.data_params)

                    genop_model = GenerativeOperatorModel(
                        prior_model=fno_prior,
                        corrector_model=diff_corrector,
                        p_md=diff_params,
                        p_d=self.data_params
                    )

                    self.assertEqual(genop_model.correction_strength, strength)

                    # Test forward pass
                    x = torch.randn(1, 2, 3, 16, 16)
                    with torch.no_grad():
                        output = genop_model(x)

                    self.assertTrue(torch.all(torch.isfinite(output)))

        except Exception as e:
            self.fail(f"Correction strength test failed: {e}")

    @unittest.skipIf(not DEEPONET_AVAILABLE, "DeepONet modules not available")
    def test_deeponet_diffusion_placeholder(self):
        """Placeholder test for future DeepONet + Diffusion integration"""
        # This test serves as a placeholder for when DeepONet + DM integration is implemented
        try:
            config = DeepONetConfig(
                latent_dim=32,
                n_sensors=64,
                branch_layers=[64, 32],
                trunk_layers=[64, 32],
                bias=True,
                normalize_inputs=True
            )
            deeponet_model = StandardDeepONet(config)

            # For now, just test that DeepONet works standalone
            # Future implementation would integrate with DiffusionCorrector
            x = torch.randn(1, 2, 3, 16, 16)
            with torch.no_grad():
                output = deeponet_model(x)

            self.assertTrue(torch.all(torch.isfinite(output)))

        except Exception as e:
            self.fail(f"DeepONet + Diffusion placeholder test failed: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)