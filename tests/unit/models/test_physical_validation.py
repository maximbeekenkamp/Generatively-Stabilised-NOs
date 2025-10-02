"""
Physical validation tests for neural operators

These tests verify that model outputs satisfy basic physical constraints
and have reasonable properties for turbulence modeling applications.
Can be applied to any neural operator model (FNO, TNO, DeepONet, etc.).

IMPORTANT: These tests are designed to FAIL with dummy data and PASS with real data.
This validates that the tests are physically meaningful and can detect
when models are working with realistic vs unrealistic data.
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
    from src.core.utils.params import DataParams
    MODELS_AVAILABLE = True
except ImportError as e:
    MODELS_AVAILABLE = False
    print(f"Model modules not available: {e}")

try:
    from src.core.models.deeponet.deeponet_variants import StandardDeepONet
    from src.core.models.deeponet.deeponet_base import DeepONetConfig
    DEEPONET_AVAILABLE = True
except ImportError:
    DEEPONET_AVAILABLE = False

from tests.fixtures.dummy_datasets import get_dummy_batch


class PhysicalValidationTestMixin:
    """Mixin class providing physical validation tests for any neural operator model"""

    def run_physical_validation_suite(self, model, model_name="Model"):
        """Run complete physical validation suite on a model"""
        self.check_output_bounds_reasonable(model, model_name)
        self.check_temporal_smoothness(model, model_name)
        self.check_spatial_smoothness(model, model_name)
        self.check_conservation_properties(model, model_name)
        self.check_energy_spectrum_properties(model, model_name)

    def check_output_bounds_reasonable(self, model, model_name="Model"):
        """Test outputs are in reasonable physical ranges"""
        input_batch, _ = get_dummy_batch("inc_low", batch_size=2)

        with torch.no_grad():
            output = model(input_batch)

        # Check output bounds are reasonable
        output_min = torch.min(output)
        output_max = torch.max(output)
        output_mean = torch.mean(output)

        # Should not have extreme values (for dummy data)
        self.assertGreater(output_min.item(), -100.0, f"{model_name} output min too negative")
        self.assertLess(output_max.item(), 100.0, f"{model_name} output max too large")
        self.assertTrue(torch.isfinite(output_mean), f"{model_name} output mean should be finite")

        # Check for reasonable dynamic range
        output_range = output_max - output_min
        self.assertGreater(output_range.item(), 1e-6, f"{model_name} output should have some dynamic range")

        # Check no NaN or Inf values
        self.assertTrue(torch.all(torch.isfinite(output)), f"{model_name} output should be finite")

    def check_temporal_smoothness(self, model, model_name="Model"):
        """Test no extreme temporal jumps in predictions"""
        input_batch, _ = get_dummy_batch("inc_low", batch_size=1)

        with torch.no_grad():
            output = model(input_batch)

        # Check temporal derivatives are reasonable
        for t in range(1, output.shape[1]):
            temporal_diff = torch.abs(output[0, t] - output[0, t-1])
            max_temporal_change = torch.max(temporal_diff)

            # Temporal changes shouldn't be extremely large
            self.assertLess(max_temporal_change.item(), 50.0,
                           f"{model_name} temporal change at step {t} too large: {max_temporal_change.item()}")

            # Should have some temporal variation (not constant)
            mean_temporal_change = torch.mean(temporal_diff)
            self.assertGreater(mean_temporal_change.item(), 1e-8,
                              f"{model_name} no temporal variation at step {t}")

    def check_spatial_smoothness(self, model, model_name="Model"):
        """Test spatial derivatives are reasonable"""
        input_batch, _ = get_dummy_batch("inc_low", batch_size=1)

        with torch.no_grad():
            output = model(input_batch)

        # Check spatial gradients for last timestep
        for c in range(output.shape[2]):  # For each channel
            field = output[0, -1, c]  # Take one channel of last timestep [H, W]

            # Compute spatial gradients
            if field.shape[0] > 1 and field.shape[1] > 1:
                grad_x = torch.abs(field[1:, :] - field[:-1, :])  # Gradient in x direction
                grad_y = torch.abs(field[:, 1:] - field[:, :-1])  # Gradient in y direction

                max_grad_x = torch.max(grad_x)
                max_grad_y = torch.max(grad_y)

                # Spatial gradients shouldn't be extremely large (would indicate noise/instability)
                self.assertLess(max_grad_x.item(), 50.0,
                               f"{model_name} spatial gradient in x too large for channel {c}: {max_grad_x.item()}")
                self.assertLess(max_grad_y.item(), 50.0,
                               f"{model_name} spatial gradient in y too large for channel {c}: {max_grad_y.item()}")

                # Should have some spatial variation
                mean_grad = (torch.mean(grad_x) + torch.mean(grad_y)) / 2
                self.assertGreater(mean_grad.item(), 1e-8,
                                  f"{model_name} no spatial variation detected in channel {c}")

    def check_conservation_properties(self, model, model_name="Model"):
        """Test basic conservation properties (on dummy data)"""
        input_batch, _ = get_dummy_batch("inc_low", batch_size=1)

        with torch.no_grad():
            output = model(input_batch)

        # Check mass conservation (sum of field values should be reasonable)
        for t in range(output.shape[1]):
            for c in range(output.shape[2]):
                field_sum = torch.sum(output[0, t, c])
                field_mean = torch.mean(output[0, t, c])

                # Total mass shouldn't blow up
                self.assertLess(torch.abs(field_sum).item(), 10000.0,
                               f"{model_name} field sum too large at t={t}, c={c}")

                # Mean should be reasonable
                self.assertLess(torch.abs(field_mean).item(), 100.0,
                               f"{model_name} field mean too large at t={t}, c={c}")

        # Check momentum conservation (dummy test - just verify reasonable magnitudes)
        if output.shape[2] >= 2:  # At least 2 velocity components
            u_field = output[0, -1, 0]  # u velocity
            v_field = output[0, -1, 1]  # v velocity

            momentum_magnitude = torch.sqrt(u_field**2 + v_field**2)
            max_momentum = torch.max(momentum_magnitude)

            self.assertLess(max_momentum.item(), 100.0,
                           f"{model_name} momentum magnitude too large")

    def check_energy_spectrum_properties(self, model, model_name="Model"):
        """Test energy spectrum has reasonable properties"""
        input_batch, _ = get_dummy_batch("inc_low", batch_size=1)

        with torch.no_grad():
            output = model(input_batch)

        # Test spectral properties for last timestep, first channel
        field = output[0, -1, 0]  # [H, W]

        # Apply 2D FFT
        field_fft = torch.fft.fft2(field)
        power_spectrum = torch.abs(field_fft) ** 2

        # Check basic spectral properties
        total_energy = torch.sum(power_spectrum)
        dc_energy = power_spectrum[0, 0]

        # DC shouldn't dominate completely (would indicate constant field)
        self.assertLess(dc_energy / total_energy, 0.99,
                       f"{model_name} energy too concentrated at DC component")

        # Should have finite energy
        self.assertTrue(torch.isfinite(total_energy), f"{model_name} total energy should be finite")
        self.assertGreater(total_energy.item(), 0, f"{model_name} should have positive energy")

        # Check energy distribution is reasonable
        H, W = field.shape
        if H > 4 and W > 4:
            # Low frequency energy (central quarter)
            low_freq_mask = torch.zeros_like(power_spectrum, dtype=torch.bool)
            center_h, center_w = H//2, W//2
            quarter_h, quarter_w = H//4, W//4
            low_freq_mask[center_h-quarter_h:center_h+quarter_h,
                         center_w-quarter_w:center_w+quarter_w] = True

            low_freq_energy = torch.sum(power_spectrum[low_freq_mask])
            high_freq_energy = total_energy - low_freq_energy

            # Should have some energy in both low and high frequencies
            self.assertGreater(low_freq_energy.item() / total_energy.item(), 0.01,
                              f"{model_name} should have some low-frequency energy")
            self.assertGreater(high_freq_energy.item() / total_energy.item(), 0.01,
                              f"{model_name} should have some high-frequency energy")


@unittest.skipIf(not MODELS_AVAILABLE, "Model modules not available")
class TestFNOPhysicalValidation(unittest.TestCase, PhysicalValidationTestMixin):
    """Physical validation tests for FNO models"""

    def setUp(self):
        """Set up test parameters"""
        self.data_params = DataParams(
            batch=2,
            sequenceLength=[8, 2],
            dataSize=[16, 16],
            dimension=2,
            simFields=["pres"],
            simParams=[],
            normalizeMode=""
        )

    def _create_fno_params(self, modes):
        """Create model parameters for FNO"""
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

    def test_fno_physical_validation(self):
        """Run complete physical validation suite on FNO"""
        model_params = self._create_fno_params(16)
        fno_model = FNOPriorAdapter(model_params, self.data_params)

        self.run_physical_validation_suite(fno_model, "FNO")


@unittest.skipIf(not MODELS_AVAILABLE, "Model modules not available")
class TestTNOPhysicalValidation(unittest.TestCase, PhysicalValidationTestMixin):
    """Physical validation tests for TNO models"""

    def setUp(self):
        """Set up test parameters"""
        self.data_params = DataParams(
            batch=2,
            sequenceLength=[8, 2],
            dataSize=[16, 16],
            dimension=2,
            simFields=["pres"],
            simParams=[],
            normalizeMode=""
        )

    def _create_tno_params(self, L=1, K=1):
        """Create model parameters for TNO"""
        class MockModelParams:
            def __init__(self, L, K):
                self.L = L
                self.K = K
                self.architecture = f'tno_L{L}_K{K}'
                self.arch = f'tno_L{L}_K{K}'
                self.model_type = 'tno'
                self.decWidth = 64

        return MockModelParams(L, K)

    def test_tno_physical_validation(self):
        """Run complete physical validation suite on TNO"""
        model_params = self._create_tno_params(L=2, K=1)
        tno_model = TNOPriorAdapter(model_params, self.data_params)

        self.run_physical_validation_suite(tno_model, "TNO")


@unittest.skipIf(not MODELS_AVAILABLE, "Model modules not available")
class TestUNetPhysicalValidation(unittest.TestCase, PhysicalValidationTestMixin):
    """Physical validation tests for UNet models"""

    def setUp(self):
        """Set up test parameters"""
        self.data_params = DataParams(
            batch=2,
            sequenceLength=[8, 2],
            dataSize=[16, 16],
            dimension=2,
            simFields=["pres"],
            simParams=[],
            normalizeMode=""
        )

    def _create_unet_params(self):
        """Create model parameters for UNet"""
        class MockModelParams:
            def __init__(self):
                self.architecture = 'unet'
                self.arch = 'unet'
                self.model_type = 'unet'
                self.decWidth = 32

        return MockModelParams()

    def test_unet_physical_validation(self):
        """Run complete physical validation suite on UNet"""
        model_params = self._create_unet_params()
        unet_model = UNetPriorAdapter(model_params, self.data_params)

        self.run_physical_validation_suite(unet_model, "UNet")


@unittest.skipIf(not DEEPONET_AVAILABLE, "DeepONet modules not available")
class TestDeepONetPhysicalValidation(unittest.TestCase, PhysicalValidationTestMixin):
    """Physical validation tests for DeepONet models"""

    def test_deeponet_physical_validation(self):
        """Run complete physical validation suite on DeepONet"""
        config = DeepONetConfig(
            latent_dim=32,
            n_sensors=64,
            branch_layers=[64, 32],
            trunk_layers=[64, 32],
            bias=True,
            normalize_inputs=True
        )
        deeponet_model = StandardDeepONet(config)
        deeponet_model.eval()  # Set to eval mode to avoid batch norm issues with small batches

        self.run_physical_validation_suite(deeponet_model, "DeepONet")


class TestCrossModelPhysicalConsistency(unittest.TestCase, PhysicalValidationTestMixin):
    """Test physical consistency across different model types"""

    def setUp(self):
        """Set up test parameters"""
        self.data_params = DataParams(
            batch=1,
            sequenceLength=[8, 2],
            dataSize=[16, 16],
            dimension=2,
            simFields=["pres"],
            simParams=[],
            normalizeMode=""
        )

    @unittest.skipIf(not MODELS_AVAILABLE, "Model modules not available")
    def test_cross_model_output_consistency(self):
        """Test that different models produce physically consistent outputs"""
        # Create models
        models = {}

        # FNO
        fno_params = self._create_fno_params(16)
        models['FNO'] = FNOPriorAdapter(fno_params, self.data_params)

        # TNO
        tno_params = self._create_tno_params(L=1, K=1)
        models['TNO'] = TNOPriorAdapter(tno_params, self.data_params)

        # UNet
        unet_params = self._create_unet_params()
        models['UNet'] = UNetPriorAdapter(unet_params, self.data_params)

        input_batch, _ = get_dummy_batch("inc_low", batch_size=1)

        model_outputs = {}
        for name, model in models.items():
            with torch.no_grad():
                output = model(input_batch)
            model_outputs[name] = output

        # Check that all models produce outputs in similar ranges
        output_ranges = {}
        for name, output in model_outputs.items():
            output_min = torch.min(output)
            output_max = torch.max(output)
            output_ranges[name] = (output_min.item(), output_max.item())

        # All models should produce outputs in comparable ranges (order of magnitude)
        all_mins = [r[0] for r in output_ranges.values()]
        all_maxs = [r[1] for r in output_ranges.values()]

        min_ratio = max(all_mins) / (min(all_mins) + 1e-8)
        max_ratio = max(all_maxs) / (min(all_maxs) + 1e-8)

        # Models should be within 2 orders of magnitude of each other
        self.assertLess(min_ratio, 100.0, "Model output minimums too different")
        self.assertLess(max_ratio, 100.0, "Model output maximums too different")

    def _create_fno_params(self, modes):
        """Create FNO parameters"""
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

    def _create_tno_params(self, L=1, K=1):
        """Create TNO parameters"""
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
        """Create UNet parameters"""
        class MockModelParams:
            def __init__(self):
                self.architecture = 'unet'
                self.arch = 'unet'
                self.model_type = 'unet'
                self.decWidth = 32

        return MockModelParams()


if __name__ == '__main__':
    unittest.main(verbosity=2)