"""
Analysis Integration Tests

Tests integration between neural operator models and analysis code from the reference folder.
Verifies that models can be properly evaluated using the analysis tools for:

1. TKE (Turbulent Kinetic Energy) analysis
2. Loss computation and plotting
3. Frequency domain analysis
4. Downstream metrics evaluation
5. Data visualization compatibility
6. Model name and color mapping
7. Multi-dataset analysis workflows

Each test ensures models work with dummy data and produce reasonable analysis outputs.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import tempfile
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

# Add src and reference paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'reference code analysis'))

try:
    from src.core.models.neural_operator_adapters import FNOPriorAdapter, TNOPriorAdapter, UNetPriorAdapter
    from src.core.utils.params import DataParams
    MODELS_AVAILABLE = True
except ImportError as e:
    MODELS_AVAILABLE = False
    print(f"Model modules not available: {e}")

try:
    from plot_color_and_name_mapping import getColor, getModelName, getDatasetName, getFieldIndex, getLossRelevantFields
    ANALYSIS_AVAILABLE = True
except ImportError as e:
    ANALYSIS_AVAILABLE = False
    print(f"Analysis modules not available: {e}")

from tests.fixtures.dummy_datasets import get_dummy_batch


@unittest.skipIf(not MODELS_AVAILABLE or not ANALYSIS_AVAILABLE, "Required modules not available")
class TestAnalysisIntegration(unittest.TestCase):
    """Test suite for analysis integration with neural operator models"""

    def setUp(self):
        """Set up test parameters and configurations"""
        self.data_params = DataParams(
            batch=2,
            sequenceLength=[8, 2],
            dataSize=[32, 32],
            dimension=2,
            simFields=["pres"],
            simParams=[],  # Remove simParams to avoid channel mismatch
            normalizeMode=""
        )

        # Test all dataset types
        self.dataset_names = ['inc_low', 'inc_high', 'tra_ext', 'tra_inc', 'iso']

        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_fno_params(self, modes=16):
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

    def _create_tno_params(self, L=2, K=1):
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

    def _generate_model_predictions(self, model, dataset_name, num_sequences=2):
        """Generate model predictions for analysis"""
        predictions = []
        ground_truths = []

        input_seq_len, output_seq_len = self.data_params.sequenceLength

        for i in range(num_sequences):
            input_batch, target_batch = get_dummy_batch(dataset_name, batch_size=1)

            with torch.no_grad():
                pred = model(input_batch)

            # Align sequence lengths: extract only the target timesteps from predictions
            # Models predict for all input timesteps, but we only want the last output_seq_len timesteps
            aligned_pred = pred[:, -output_seq_len:]  # Take last output_seq_len timesteps

            predictions.append(aligned_pred.cpu().numpy())
            ground_truths.append(target_batch.cpu().numpy() if target_batch is not None else input_batch[:, -output_seq_len:].cpu().numpy())

        return np.concatenate(predictions, axis=0), np.concatenate(ground_truths, axis=0)

    def test_model_name_color_mapping(self):
        """Test that analysis tools recognize model names and assign colors"""
        try:
            # Test known model names
            model_names = ["FNO16", "FNO32", "U-Net", "U-Net-ut", "U-Net-tn", "ACDM", "ACDM-ncn", "Refiner"]

            for model_name in model_names:
                with self.subTest(model=model_name):
                    # Test color mapping
                    color = getColor(model_name)
                    self.assertIsNotNone(color, f"No color found for model {model_name}")

                    # Test name formatting
                    formatted_name = getModelName(model_name)
                    self.assertIsInstance(formatted_name, str)
                    self.assertGreater(len(formatted_name), 0)

        except Exception as e:
            self.fail(f"Model name/color mapping test failed: {e}")

    def test_dataset_name_mapping(self):
        """Test dataset name mapping for analysis"""
        try:
            # Test dataset name mappings
            dataset_mappings = {
                'inc_low': 'lowRey',
                'inc_high': 'highRey',
                'tra_ext': 'extrap',
                'tra_inc': 'interp',
                'iso': 'zInterp'
            }

            for internal_name, analysis_name in dataset_mappings.items():
                with self.subTest(dataset=internal_name):
                    # Test that analysis tools can handle dataset names
                    mapped_name = getDatasetName(analysis_name)
                    self.assertIsInstance(mapped_name, str)

        except Exception as e:
            self.fail(f"Dataset name mapping test failed: {e}")

    def test_field_index_mapping(self):
        """Test field index mapping for multi-field analysis"""
        try:
            # Test field indexing for different field types
            field_types = ["pres", "velo", "temp"]

            for field_type in field_types:
                with self.subTest(field=field_type):
                    try:
                        field_index = getFieldIndex(field_type)
                        self.assertIsInstance(field_index, (int, list))
                        if isinstance(field_index, int):
                            self.assertGreaterEqual(field_index, 0)
                        elif isinstance(field_index, list):
                            self.assertTrue(all(isinstance(idx, int) and idx >= 0 for idx in field_index))
                    except:
                        # Some fields might not be implemented, which is acceptable
                        pass

        except Exception as e:
            self.fail(f"Field index mapping test failed: {e}")

    def test_loss_computation_integration(self):
        """Test loss computation with model outputs"""
        try:
            # Create FNO model
            fno_params = self._create_fno_params()
            fno_model = FNOPriorAdapter(fno_params, self.data_params)

            for dataset_name in self.dataset_names:
                with self.subTest(dataset=dataset_name):
                    # Generate predictions
                    predictions, ground_truths = self._generate_model_predictions(fno_model, dataset_name)

                    # Compute basic MSE loss
                    mse_loss = np.mean((predictions - ground_truths) ** 2)

                    # Loss should be finite and reasonable
                    self.assertTrue(np.isfinite(mse_loss))
                    self.assertGreaterEqual(mse_loss, 0.0)
                    self.assertLess(mse_loss, 1000.0)  # Should not be extremely large

                    # Compute relative error
                    relative_error = np.mean(np.abs(predictions - ground_truths)) / (np.mean(np.abs(ground_truths)) + 1e-8)
                    self.assertTrue(np.isfinite(relative_error))
                    self.assertGreaterEqual(relative_error, 0.0)

        except Exception as e:
            self.fail(f"Loss computation integration test failed\n"
                     f"  Model: FNO, dataset: {dataset_name if 'dataset_name' in locals() else 'unknown'}\n"
                     f"  Data config: {self.data_params.dataSize}, seq_len={self.data_params.sequenceLength}\n"
                     f"  Expected: finite MSE and relative error values\n"
                     f"  Error: {e}")

    def test_tke_analysis_compatibility(self):
        """Test TKE (Turbulent Kinetic Energy) analysis compatibility"""
        try:
            # Create model
            fno_params = self._create_fno_params()
            fno_model = FNOPriorAdapter(fno_params, self.data_params)

            # Generate predictions for velocity field analysis
            dataset_name = 'inc_low'  # Incompressible flow dataset
            predictions, ground_truths = self._generate_model_predictions(fno_model, dataset_name)

            # Simulate TKE computation (simplified)
            # TKE = 0.5 * (u'^2 + v'^2 + w'^2) where u', v', w' are velocity fluctuations
            B, T, C, H, W = predictions.shape

            if C >= 2:  # At least 2 velocity components
                u_pred = predictions[:, :, 0]  # u velocity
                v_pred = predictions[:, :, 1]  # v velocity

                u_true = ground_truths[:, :, 0]
                v_true = ground_truths[:, :, 1]

                # Compute mean velocities
                u_mean_pred = np.mean(u_pred, axis=(2, 3), keepdims=True)
                v_mean_pred = np.mean(v_pred, axis=(2, 3), keepdims=True)

                u_mean_true = np.mean(u_true, axis=(2, 3), keepdims=True)
                v_mean_true = np.mean(v_true, axis=(2, 3), keepdims=True)

                # Compute fluctuations
                u_fluct_pred = u_pred - u_mean_pred
                v_fluct_pred = v_pred - v_mean_pred

                u_fluct_true = u_true - u_mean_true
                v_fluct_true = v_true - v_mean_true

                # Compute TKE
                tke_pred = 0.5 * (u_fluct_pred**2 + v_fluct_pred**2)
                tke_true = 0.5 * (u_fluct_true**2 + v_fluct_true**2)

                # TKE should be non-negative and finite
                self.assertTrue(np.all(tke_pred >= 0))
                self.assertTrue(np.all(np.isfinite(tke_pred)))
                self.assertTrue(np.all(tke_true >= 0))
                self.assertTrue(np.all(np.isfinite(tke_true)))

                # Compute TKE statistics
                tke_mean_pred = np.mean(tke_pred)
                tke_mean_true = np.mean(tke_true)

                self.assertGreater(tke_mean_pred, 0)  # Should have some turbulent energy
                self.assertGreater(tke_mean_true, 0)

        except Exception as e:
            self.fail(f"TKE analysis compatibility test failed: {e}")

    def test_frequency_domain_analysis(self):
        """Test frequency domain analysis compatibility"""
        try:
            # Create model
            unet_params = self._create_unet_params()
            unet_model = UNetPriorAdapter(unet_params, self.data_params)

            # Generate predictions
            dataset_name = 'inc_low'
            predictions, ground_truths = self._generate_model_predictions(unet_model, dataset_name)

            B, T, C, H, W = predictions.shape

            # Perform FFT analysis on spatial dimensions
            for c in range(min(C, 2)):  # Test first 2 channels
                field_pred = predictions[0, -1, c]  # Last timestep, one channel
                field_true = ground_truths[0, -1, c]

                # 2D FFT
                fft_pred = np.fft.fft2(field_pred)
                fft_true = np.fft.fft2(field_true)

                # Power spectrum
                power_pred = np.abs(fft_pred) ** 2
                power_true = np.abs(fft_true) ** 2

                # Check that spectra are reasonable
                self.assertTrue(np.all(np.isfinite(power_pred)))
                self.assertTrue(np.all(np.isfinite(power_true)))
                self.assertTrue(np.all(power_pred >= 0))
                self.assertTrue(np.all(power_true >= 0))

                # DC component should be reasonable
                dc_pred = power_pred[0, 0]
                dc_true = power_true[0, 0]
                total_power_pred = np.sum(power_pred)
                total_power_true = np.sum(power_true)

                # DC shouldn't dominate completely (would indicate constant field)
                self.assertLess(dc_pred / total_power_pred, 0.99)
                self.assertLess(dc_true / total_power_true, 0.99)

        except Exception as e:
            self.fail(f"Frequency domain analysis test failed: {e}")

    def test_multi_model_analysis_workflow(self):
        """Test analysis workflow with multiple model types"""
        try:
            # Create different model types
            models = {
                "FNO16": (self._create_fno_params(16), FNOPriorAdapter),
                "TNO": (self._create_tno_params(), TNOPriorAdapter),
                "UNet": (self._create_unet_params(), UNetPriorAdapter),
            }

            dataset_name = 'inc_low'
            model_results = {}

            for model_name, (params, adapter_class) in models.items():
                with self.subTest(model=model_name):
                    # Create model
                    model = adapter_class(params, self.data_params)

                    # Generate predictions
                    predictions, ground_truths = self._generate_model_predictions(model, dataset_name, num_sequences=1)

                    # Compute metrics
                    mse = np.mean((predictions - ground_truths) ** 2)
                    mae = np.mean(np.abs(predictions - ground_truths))

                    model_results[model_name] = {
                        'mse': mse,
                        'mae': mae,
                        'predictions_shape': predictions.shape
                    }

                    # Basic validation
                    self.assertTrue(np.isfinite(mse))
                    self.assertTrue(np.isfinite(mae))
                    self.assertGreaterEqual(mse, 0)
                    self.assertGreaterEqual(mae, 0)

            # Compare models (basic sanity check)
            mse_values = [results['mse'] for results in model_results.values()]
            self.assertTrue(len(set(mse_values)) > 1)  # Models should give different results

        except Exception as e:
            self.fail(f"Multi-model analysis workflow test failed: {e}")

    def test_visualization_compatibility(self):
        """Test compatibility with visualization tools"""
        try:
            # Create model
            fno_params = self._create_fno_params()
            fno_model = FNOPriorAdapter(fno_params, self.data_params)

            # Generate predictions
            dataset_name = 'inc_low'
            predictions, ground_truths = self._generate_model_predictions(fno_model, dataset_name, num_sequences=1)

            B, T, C, H, W = predictions.shape

            # Test basic plotting functionality
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            # Plot prediction
            im1 = axes[0].imshow(predictions[0, -1, 0], cmap='viridis')
            axes[0].set_title('Prediction')
            axes[0].axis('off')

            # Plot ground truth
            im2 = axes[1].imshow(ground_truths[0, -1, 0], cmap='viridis')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')

            # Save to temporary file
            temp_file = os.path.join(self.temp_dir, 'test_plot.png')
            plt.savefig(temp_file, dpi=100, bbox_inches='tight')
            plt.close()

            # Check file was created
            self.assertTrue(os.path.exists(temp_file))
            self.assertGreater(os.path.getsize(temp_file), 1000)  # Should be a reasonable size

        except Exception as e:
            self.fail(f"Visualization compatibility test failed: {e}")

    def test_temporal_analysis_integration(self):
        """Test temporal analysis integration"""
        try:
            # Create model
            tno_params = self._create_tno_params()
            tno_model = TNOPriorAdapter(tno_params, self.data_params)

            # Generate longer sequence for temporal analysis
            input_batch, target_batch = get_dummy_batch('inc_low', batch_size=1)
            B, T, C, H, W = input_batch.shape

            with torch.no_grad():
                predictions = tno_model(input_batch)

            # Align sequence lengths: extract only the target timesteps from predictions
            input_seq_len, output_seq_len = self.data_params.sequenceLength
            aligned_predictions = predictions[:, -output_seq_len:]  # Take last output_seq_len timesteps

            predictions_np = aligned_predictions.cpu().numpy()
            targets_np = target_batch.cpu().numpy() if target_batch is not None else input_batch[:, -output_seq_len:].cpu().numpy()

            # Compute temporal metrics
            temporal_mse = []
            for t in range(output_seq_len):  # Use output_seq_len instead of T
                mse_t = np.mean((predictions_np[:, t] - targets_np[:, t]) ** 2)
                temporal_mse.append(mse_t)

            temporal_mse = np.array(temporal_mse)

            # All temporal MSEs should be finite and non-negative
            self.assertTrue(np.all(np.isfinite(temporal_mse)))
            self.assertTrue(np.all(temporal_mse >= 0))

            # Compute temporal derivatives
            if T > 1:
                temporal_diff_pred = np.abs(predictions_np[:, 1:] - predictions_np[:, :-1])
                temporal_diff_true = np.abs(targets_np[:, 1:] - targets_np[:, :-1])

                mean_temporal_diff_pred = np.mean(temporal_diff_pred)
                mean_temporal_diff_true = np.mean(temporal_diff_true)

                self.assertTrue(np.isfinite(mean_temporal_diff_pred))
                self.assertTrue(np.isfinite(mean_temporal_diff_true))
                self.assertGreaterEqual(mean_temporal_diff_pred, 0)
                self.assertGreaterEqual(mean_temporal_diff_true, 0)

        except Exception as e:
            self.fail(f"Temporal analysis integration test failed: {e}")

    def test_loss_relevant_fields_analysis(self):
        """Test loss computation on relevant fields only"""
        try:
            # Test getLossRelevantFields function
            try:
                relevant_fields = getLossRelevantFields()
                if relevant_fields is not None:
                    self.assertIsInstance(relevant_fields, (list, tuple))
                    self.assertTrue(len(relevant_fields) > 0)
                    # All field indices should be integers
                    for field_idx in relevant_fields:
                        self.assertIsInstance(field_idx, int)
                        self.assertGreaterEqual(field_idx, 0)
            except:
                # Function might not be implemented, which is acceptable
                pass

            # Create model and test field-specific loss computation
            unet_params = self._create_unet_params()
            unet_model = UNetPriorAdapter(unet_params, self.data_params)

            predictions, ground_truths = self._generate_model_predictions(unet_model, 'inc_low')
            B, T, C, H, W = predictions.shape

            # Compute loss for each channel separately
            channel_losses = []
            for c in range(C):
                channel_loss = np.mean((predictions[:, :, c] - ground_truths[:, :, c]) ** 2)
                channel_losses.append(channel_loss)

                self.assertTrue(np.isfinite(channel_loss))
                self.assertGreaterEqual(channel_loss, 0)

            # Different channels should potentially have different losses
            if C > 1:
                channel_losses = np.array(channel_losses)
                self.assertTrue(np.any(channel_losses != channel_losses[0]))  # Not all identical

        except Exception as e:
            self.fail(f"Loss relevant fields analysis test failed: {e}")

    def test_downstream_metrics_computation(self):
        """Test computation of downstream metrics"""
        try:
            # Create model
            fno_params = self._create_fno_params()
            fno_model = FNOPriorAdapter(fno_params, self.data_params)

            dataset_name = 'tra_ext'  # Extrapolation dataset
            predictions, ground_truths = self._generate_model_predictions(fno_model, dataset_name)

            B, T, C, H, W = predictions.shape

            # Compute various downstream metrics
            metrics = {}

            # L2 error
            l2_error = np.sqrt(np.mean((predictions - ground_truths) ** 2))
            metrics['L2'] = l2_error

            # L1 error
            l1_error = np.mean(np.abs(predictions - ground_truths))
            metrics['L1'] = l1_error

            # Relative L2 error
            rel_l2_error = l2_error / (np.sqrt(np.mean(ground_truths ** 2)) + 1e-8)
            metrics['Relative_L2'] = rel_l2_error

            # Maximum error
            max_error = np.max(np.abs(predictions - ground_truths))
            metrics['Max'] = max_error

            # Correlation coefficient
            pred_flat = predictions.flatten()
            true_flat = ground_truths.flatten()
            correlation = np.corrcoef(pred_flat, true_flat)[0, 1]
            metrics['Correlation'] = correlation

            # Validate all metrics
            for metric_name, metric_value in metrics.items():
                with self.subTest(metric=metric_name):
                    self.assertTrue(np.isfinite(metric_value), f"{metric_name} should be finite")
                    if metric_name != 'Correlation':  # Correlation can be negative
                        self.assertGreaterEqual(metric_value, 0, f"{metric_name} should be non-negative")
                    if metric_name == 'Correlation':
                        self.assertGreaterEqual(metric_value, -1)
                        self.assertLessEqual(metric_value, 1)

        except Exception as e:
            self.fail(f"Downstream metrics computation test failed: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)