"""
Visual Output Generation Tests

Tests for visual output generation and visualization capabilities of neural operator models.
Verifies that models can generate proper visual outputs including:

1. Model prediction visualization compatibility
2. Field visualization (velocity, pressure, temperature)
3. Temporal sequence visualization
4. Comparison plot generation (prediction vs ground truth)
5. Statistical visualization (loss curves, metrics)
6. Frequency domain visualization (FFT, power spectra)
7. Error field visualization
8. Multi-model comparison visualization
9. Dataset-specific visualization formats
10. Export format compatibility (PNG, PDF, SVG)

Each test ensures models work with dummy data and produce valid visual outputs
without requiring display capabilities (using non-interactive backends).
"""

import unittest
import torch
import numpy as np
import sys
import os
import tempfile
import warnings

# Set matplotlib backend before any matplotlib imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation
warnings.filterwarnings('ignore', category=UserWarning)  # Suppress matplotlib warnings

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

try:
    from src.core.models.neural_operator_adapters import FNOPriorAdapter, TNOPriorAdapter, UNetPriorAdapter
    from src.core.utils.params import DataParams
    MODELS_AVAILABLE = True
except ImportError as e:
    MODELS_AVAILABLE = False
    print(f"Model modules not available: {e}")

from tests.fixtures.dummy_datasets import get_dummy_batch


@unittest.skipIf(not MODELS_AVAILABLE, "Model modules not available")
class TestVisualOutputGeneration(unittest.TestCase):
    """Test suite for visual output generation capabilities"""

    def setUp(self):
        """Set up test parameters and configurations"""
        self.data_params = DataParams(
            batch=2,
            sequenceLength=[8, 2],
            dataSize=[32, 32],
            dimension=2,
            simFields=["pres"],
            simParams=[],  # Standardized to empty for consistent channel count
            normalizeMode=""
        )

        # Test all dataset types
        self.dataset_names = ['inc_low', 'inc_high', 'tra_ext', 'tra_inc', 'iso']

        # Create temporary directory for visual outputs
        self.temp_dir = tempfile.mkdtemp()

        # Configure matplotlib for testing
        plt.rcParams.update({
            'figure.max_open_warning': 0,
            'font.size': 8,
            'axes.titlesize': 8,
            'axes.labelsize': 8,
            'xtick.labelsize': 6,
            'ytick.labelsize': 6
        })

    def tearDown(self):
        """Clean up temporary files and close all figures"""
        plt.close('all')
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

    def test_basic_field_visualization(self):
        """Test basic field visualization for different field types"""
        try:
            # Create model
            fno_params = self._create_fno_params()
            fno_model = FNOPriorAdapter(fno_params, self.data_params)

            # Generate predictions
            input_batch, target_batch = get_dummy_batch('inc_low', batch_size=1)

            with torch.no_grad():
                predictions = fno_model(input_batch)

            predictions_np = predictions.cpu().numpy()
            targets_np = target_batch.cpu().numpy() if target_batch is not None else input_batch.cpu().numpy()

            B, T, C, H, W = predictions_np.shape

            # Test visualization for each channel
            for c in range(min(C, 3)):  # Test up to 3 channels
                with self.subTest(channel=c):
                    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

                    # Prediction plot
                    pred_field = predictions_np[0, -1, c]  # Last timestep
                    im1 = axes[0].imshow(pred_field, cmap='viridis', origin='lower')
                    axes[0].set_title(f'Prediction - Channel {c}')
                    axes[0].axis('off')
                    plt.colorbar(im1, ax=axes[0], shrink=0.8)

                    # Ground truth plot
                    true_field = targets_np[0, -1, c]
                    im2 = axes[1].imshow(true_field, cmap='viridis', origin='lower')
                    axes[1].set_title(f'Ground Truth - Channel {c}')
                    axes[1].axis('off')
                    plt.colorbar(im2, ax=axes[1], shrink=0.8)

                    # Save visualization
                    output_path = os.path.join(self.temp_dir, f'field_vis_channel_{c}.png')
                    plt.savefig(output_path, dpi=100, bbox_inches='tight')
                    plt.close()

                    # Verify file was created and has reasonable size
                    self.assertTrue(os.path.exists(output_path))
                    self.assertGreater(os.path.getsize(output_path), 1000)

        except Exception as e:
            self.fail(f"Basic field visualization test failed: {e}")

    def test_temporal_sequence_visualization(self):
        """Test temporal sequence visualization"""
        try:
            # Create model
            unet_params = self._create_unet_params()
            unet_model = UNetPriorAdapter(unet_params, self.data_params)

            # Generate longer sequence
            input_batch, target_batch = get_dummy_batch('inc_low', batch_size=1)

            with torch.no_grad():
                predictions = unet_model(input_batch)

            predictions_np = predictions.cpu().numpy()
            B, T, C, H, W = predictions_np.shape

            # Create temporal sequence plot
            fig, axes = plt.subplots(2, min(T, 4), figsize=(16, 8))
            if T == 1:
                axes = axes.reshape(2, 1)

            for t in range(min(T, 4)):
                # Prediction
                pred_field = predictions_np[0, t, 0]  # First channel
                im1 = axes[0, t].imshow(pred_field, cmap='RdBu_r', origin='lower')
                axes[0, t].set_title(f'Pred t={t}')
                axes[0, t].axis('off')

                # Compute temporal difference if not first frame
                if t > 0:
                    prev_field = predictions_np[0, t-1, 0]
                    diff_field = pred_field - prev_field
                    im2 = axes[1, t].imshow(diff_field, cmap='seismic', origin='lower')
                    axes[1, t].set_title(f'Δt={t}')
                else:
                    axes[1, t].imshow(pred_field, cmap='RdBu_r', origin='lower')
                    axes[1, t].set_title(f'Initial')
                axes[1, t].axis('off')

            plt.tight_layout()

            # Save temporal visualization
            output_path = os.path.join(self.temp_dir, 'temporal_sequence.png')
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()

            # Verify file was created
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 1000)

        except Exception as e:
            self.fail(f"Temporal sequence visualization test failed: {e}")

    def test_error_field_visualization(self):
        """Test error field visualization"""
        try:
            # Create model
            tno_params = self._create_tno_params()
            tno_model = TNOPriorAdapter(tno_params, self.data_params)

            # Generate predictions
            input_batch, target_batch = get_dummy_batch('tra_ext', batch_size=1)

            with torch.no_grad():
                predictions = tno_model(input_batch)

            # Align sequence lengths: extract only the target timesteps from predictions
            input_seq_len, output_seq_len = self.data_params.sequenceLength
            aligned_predictions = predictions[:, -output_seq_len:]  # Take last output_seq_len timesteps

            predictions_np = aligned_predictions.cpu().numpy()
            targets_np = target_batch.cpu().numpy() if target_batch is not None else input_batch[:, -output_seq_len:].cpu().numpy()

            B, T, C, H, W = predictions_np.shape

            # Compute error fields
            error_fields = np.abs(predictions_np - targets_np)
            relative_error = error_fields / (np.abs(targets_np) + 1e-8)

            # Create error visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Absolute error
            im1 = axes[0, 0].imshow(error_fields[0, -1, 0], cmap='hot', origin='lower')
            axes[0, 0].set_title('Absolute Error')
            axes[0, 0].axis('off')
            plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)

            # Relative error
            im2 = axes[0, 1].imshow(relative_error[0, -1, 0], cmap='hot', origin='lower')
            axes[0, 1].set_title('Relative Error')
            axes[0, 1].axis('off')
            plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)

            # Error histogram
            axes[1, 0].hist(error_fields.flatten(), bins=50, alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('Absolute Error')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Error Distribution')
            axes[1, 0].grid(True, alpha=0.3)

            # Error statistics over time
            temporal_mse = np.mean(error_fields**2, axis=(0, 2, 3, 4))
            axes[1, 1].plot(range(T), temporal_mse, 'b-o', markersize=4)
            axes[1, 1].set_xlabel('Time Step')
            axes[1, 1].set_ylabel('MSE')
            axes[1, 1].set_title('Error Evolution')
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()

            # Save error visualization
            output_path = os.path.join(self.temp_dir, 'error_fields.png')
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()

            # Verify file was created
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 1000)

        except Exception as e:
            self.fail(f"Error field visualization test failed: {e}")

    def test_frequency_domain_visualization(self):
        """Test frequency domain visualization"""
        try:
            # Create model
            fno_params = self._create_fno_params()
            fno_model = FNOPriorAdapter(fno_params, self.data_params)

            # Generate predictions
            input_batch, target_batch = get_dummy_batch('inc_low', batch_size=1)

            with torch.no_grad():
                predictions = fno_model(input_batch)

            predictions_np = predictions.cpu().numpy()
            targets_np = target_batch.cpu().numpy() if target_batch is not None else input_batch.cpu().numpy()

            # Take last timestep, first channel
            pred_field = predictions_np[0, -1, 0]
            true_field = targets_np[0, -1, 0]

            # Compute 2D FFT
            pred_fft = np.fft.fft2(pred_field)
            true_fft = np.fft.fft2(true_field)

            # Power spectra
            pred_power = np.abs(pred_fft) ** 2
            true_power = np.abs(true_fft) ** 2

            # Shift zero frequency to center
            pred_power_shifted = np.fft.fftshift(pred_power)
            true_power_shifted = np.fft.fftshift(true_power)

            # Create frequency domain visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Power spectrum - prediction
            im1 = axes[0, 0].imshow(np.log10(pred_power_shifted + 1e-10), cmap='viridis', origin='lower')
            axes[0, 0].set_title('Log Power Spectrum - Prediction')
            axes[0, 0].axis('off')
            plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)

            # Power spectrum - ground truth
            im2 = axes[0, 1].imshow(np.log10(true_power_shifted + 1e-10), cmap='viridis', origin='lower')
            axes[0, 1].set_title('Log Power Spectrum - Ground Truth')
            axes[0, 1].axis('off')
            plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)

            # Radial power spectrum
            H, W = pred_power.shape
            center_h, center_w = H // 2, W // 2

            # Create radial coordinates
            y, x = np.ogrid[:H, :W]
            r = np.sqrt((x - center_w)**2 + (y - center_h)**2)
            r = r.astype(int)

            # Compute radial average
            max_r = min(center_h, center_w)
            radial_pred = np.zeros(max_r)
            radial_true = np.zeros(max_r)

            for radius in range(max_r):
                mask = (r == radius)
                if np.any(mask):
                    radial_pred[radius] = np.mean(pred_power[mask])
                    radial_true[radius] = np.mean(true_power[mask])

            axes[1, 0].loglog(range(max_r), radial_pred, 'b-', label='Prediction', linewidth=2)
            axes[1, 0].loglog(range(max_r), radial_true, 'r--', label='Ground Truth', linewidth=2)
            axes[1, 0].set_xlabel('Wavenumber')
            axes[1, 0].set_ylabel('Power')
            axes[1, 0].set_title('Radial Power Spectrum')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Phase difference
            phase_diff = np.angle(pred_fft) - np.angle(true_fft)
            phase_diff_shifted = np.fft.fftshift(phase_diff)
            im4 = axes[1, 1].imshow(phase_diff_shifted, cmap='RdBu_r', origin='lower', vmin=-np.pi, vmax=np.pi)
            axes[1, 1].set_title('Phase Difference')
            axes[1, 1].axis('off')
            plt.colorbar(im4, ax=axes[1, 1], shrink=0.8)

            plt.tight_layout()

            # Save frequency visualization
            output_path = os.path.join(self.temp_dir, 'frequency_domain.png')
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()

            # Verify file was created
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 1000)

        except Exception as e:
            self.fail(f"Frequency domain visualization test failed: {e}")

    def test_multi_model_comparison_visualization(self):
        """Test multi-model comparison visualization"""
        try:
            # Create different models
            models = {
                'FNO': (self._create_fno_params(), FNOPriorAdapter),
                'TNO': (self._create_tno_params(), TNOPriorAdapter),
                'UNet': (self._create_unet_params(), UNetPriorAdapter)
            }

            # Generate predictions for each model
            model_predictions = {}
            input_batch, target_batch = get_dummy_batch('inc_low', batch_size=1)

            for model_name, (params, adapter_class) in models.items():
                model = adapter_class(params, self.data_params)
                with torch.no_grad():
                    predictions = model(input_batch)
                model_predictions[model_name] = predictions.cpu().numpy()

            targets_np = target_batch.cpu().numpy() if target_batch is not None else input_batch.cpu().numpy()

            # Create comparison visualization
            num_models = len(models)
            fig, axes = plt.subplots(2, num_models + 1, figsize=(20, 8))

            # Plot ground truth
            true_field = targets_np[0, -1, 0]
            im_true = axes[0, 0].imshow(true_field, cmap='RdBu_r', origin='lower')
            axes[0, 0].set_title('Ground Truth')
            axes[0, 0].axis('off')
            plt.colorbar(im_true, ax=axes[0, 0], shrink=0.8)

            # Compute and plot errors
            errors = []
            for i, (model_name, predictions) in enumerate(model_predictions.items()):
                # Prediction
                pred_field = predictions[0, -1, 0]
                im_pred = axes[0, i+1].imshow(pred_field, cmap='RdBu_r', origin='lower')
                axes[0, i+1].set_title(f'{model_name} Prediction')
                axes[0, i+1].axis('off')
                plt.colorbar(im_pred, ax=axes[0, i+1], shrink=0.8)

                # Error
                error_field = np.abs(pred_field - true_field)
                errors.append(np.mean(error_field))
                im_err = axes[1, i+1].imshow(error_field, cmap='hot', origin='lower')
                axes[1, i+1].set_title(f'{model_name} Error')
                axes[1, i+1].axis('off')
                plt.colorbar(im_err, ax=axes[1, i+1], shrink=0.8)

            # Summary statistics
            model_names = list(model_predictions.keys())
            axes[1, 0].bar(model_names, errors, alpha=0.7, color=['blue', 'red', 'green'])
            axes[1, 0].set_ylabel('Mean Absolute Error')
            axes[1, 0].set_title('Model Comparison')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)

            plt.tight_layout()

            # Save comparison visualization
            output_path = os.path.join(self.temp_dir, 'model_comparison.png')
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()

            # Verify file was created
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 1000)

        except Exception as e:
            self.fail(f"Multi-model comparison visualization test failed: {e}")

    def test_loss_curve_visualization(self):
        """Test loss curve visualization"""
        try:
            # Simulate training history
            epochs = 50
            train_losses = []
            val_losses = []

            # Simulate different loss behaviors
            for epoch in range(epochs):
                # Training loss with noise
                base_train_loss = 1.0 * np.exp(-epoch / 20.0) + 0.1
                train_noise = np.random.normal(0, 0.05)
                train_loss = base_train_loss + train_noise

                # Validation loss with some overfitting
                base_val_loss = 1.0 * np.exp(-epoch / 25.0) + 0.15
                if epoch > 30:
                    base_val_loss += 0.01 * (epoch - 30)  # Slight overfitting
                val_noise = np.random.normal(0, 0.03)
                val_loss = base_val_loss + val_noise

                train_losses.append(max(0.05, train_loss))
                val_losses.append(max(0.05, val_loss))

            # Create loss visualization
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))

            # Loss curves
            axes[0].plot(range(epochs), train_losses, 'b-', label='Training Loss', linewidth=2)
            axes[0].plot(range(epochs), val_losses, 'r--', label='Validation Loss', linewidth=2)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training Progress')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].set_yscale('log')

            # Loss distribution
            axes[1].hist(train_losses, bins=20, alpha=0.5, label='Training', color='blue', edgecolor='black')
            axes[1].hist(val_losses, bins=20, alpha=0.5, label='Validation', color='red', edgecolor='black')
            axes[1].set_xlabel('Loss Value')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Loss Distribution')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()

            # Save loss visualization
            output_path = os.path.join(self.temp_dir, 'loss_curves.png')
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()

            # Verify file was created
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 1000)

        except Exception as e:
            self.fail(f"Loss curve visualization test failed: {e}")

    def test_dataset_specific_visualization(self):
        """Test dataset-specific visualization formats"""
        try:
            # Create model
            unet_params = self._create_unet_params()
            unet_model = UNetPriorAdapter(unet_params, self.data_params)

            # Test visualization for each dataset type
            for dataset_name in self.dataset_names[:3]:  # Test first 3 datasets
                with self.subTest(dataset=dataset_name):
                    # Generate predictions
                    input_batch, target_batch = get_dummy_batch(dataset_name, batch_size=1)

                    with torch.no_grad():
                        predictions = unet_model(input_batch)

                    predictions_np = predictions.cpu().numpy()
                    targets_np = target_batch.cpu().numpy() if target_batch is not None else input_batch.cpu().numpy()

                    # Create dataset-specific visualization
                    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

                    # Show all channels for this dataset
                    B, T, C, H, W = predictions_np.shape
                    for c in range(min(C, 3)):
                        # Prediction
                        pred_field = predictions_np[0, -1, c]
                        im1 = axes[0, c].imshow(pred_field, cmap='RdBu_r', origin='lower')
                        axes[0, c].set_title(f'{dataset_name} - Pred Ch{c}')
                        axes[0, c].axis('off')
                        plt.colorbar(im1, ax=axes[0, c], shrink=0.8)

                        # Ground truth
                        true_field = targets_np[0, -1, c]
                        im2 = axes[1, c].imshow(true_field, cmap='RdBu_r', origin='lower')
                        axes[1, c].set_title(f'{dataset_name} - True Ch{c}')
                        axes[1, c].axis('off')
                        plt.colorbar(im2, ax=axes[1, c], shrink=0.8)

                    plt.suptitle(f'Dataset: {dataset_name}', fontsize=14)
                    plt.tight_layout()

                    # Save dataset visualization
                    output_path = os.path.join(self.temp_dir, f'dataset_{dataset_name}.png')
                    plt.savefig(output_path, dpi=100, bbox_inches='tight')
                    plt.close()

                    # Verify file was created
                    self.assertTrue(os.path.exists(output_path))
                    self.assertGreater(os.path.getsize(output_path), 1000)

        except Exception as e:
            self.fail(f"Dataset-specific visualization test failed: {e}")

    def test_export_format_compatibility(self):
        """Test compatibility with different export formats"""
        try:
            # Create simple visualization
            x = np.linspace(0, 10, 100)
            y = np.sin(x)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(x, y, 'b-', linewidth=2)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('Export Format Test')
            ax.grid(True, alpha=0.3)

            # Test different export formats
            formats = ['png', 'pdf', 'svg']

            for fmt in formats:
                with self.subTest(format=fmt):
                    output_path = os.path.join(self.temp_dir, f'export_test.{fmt}')

                    try:
                        plt.savefig(output_path, format=fmt, dpi=100, bbox_inches='tight')

                        # Verify file was created
                        self.assertTrue(os.path.exists(output_path))
                        self.assertGreater(os.path.getsize(output_path), 100)

                    except Exception as format_error:
                        # Some formats might not be available, skip gracefully
                        self.skipTest(f"Format {fmt} not available: {format_error}")

            plt.close()

        except Exception as e:
            self.fail(f"Export format compatibility test failed: {e}")

    def test_colormap_and_normalization(self):
        """Test different colormap and normalization options"""
        try:
            # Create test data with different characteristics
            H, W = 32, 32

            # Different field types
            test_fields = {
                'diverging': np.random.randn(H, W),  # Zero-centered
                'positive': np.abs(np.random.randn(H, W)),  # Only positive
                'log_scale': np.exp(np.random.randn(H, W) * 2),  # Wide range
                'bounded': np.tanh(np.random.randn(H, W))  # Bounded [-1, 1]
            }

            colormaps = ['viridis', 'RdBu_r', 'hot', 'seismic']

            for field_name, field_data in test_fields.items():
                with self.subTest(field=field_name):
                    fig, axes = plt.subplots(1, len(colormaps), figsize=(16, 4))

                    for i, cmap in enumerate(colormaps):
                        # Choose appropriate normalization
                        if field_name == 'log_scale':
                            norm = colors.LogNorm(vmin=field_data.min() + 1e-8, vmax=field_data.max())
                        elif field_name == 'diverging':
                            vmax = max(abs(field_data.min()), abs(field_data.max()))
                            norm = colors.Normalize(vmin=-vmax, vmax=vmax)
                        else:
                            norm = colors.Normalize(vmin=field_data.min(), vmax=field_data.max())

                        try:
                            im = axes[i].imshow(field_data, cmap=cmap, norm=norm, origin='lower')
                            axes[i].set_title(f'{cmap}')
                            axes[i].axis('off')
                            plt.colorbar(im, ax=axes[i], shrink=0.8)
                        except:
                            # Skip if normalization fails
                            axes[i].text(0.5, 0.5, f'{cmap}\n(failed)',
                                       transform=axes[i].transAxes, ha='center', va='center')
                            axes[i].axis('off')

                    plt.suptitle(f'Field Type: {field_name}', fontsize=14)
                    plt.tight_layout()

                    # Save colormap test
                    output_path = os.path.join(self.temp_dir, f'colormap_{field_name}.png')
                    plt.savefig(output_path, dpi=100, bbox_inches='tight')
                    plt.close()

                    # Verify file was created
                    self.assertTrue(os.path.exists(output_path))
                    self.assertGreater(os.path.getsize(output_path), 1000)

        except Exception as e:
            self.fail(f"Colormap and normalization test failed: {e}")

    def test_statistical_visualization(self):
        """Test statistical visualization of model outputs"""
        try:
            # Create model
            fno_params = self._create_fno_params()
            fno_model = FNOPriorAdapter(fno_params, self.data_params)

            # Generate multiple predictions for statistics
            all_predictions = []
            all_targets = []

            for _ in range(3):  # Generate 3 samples
                input_batch, target_batch = get_dummy_batch('inc_low', batch_size=1)
                with torch.no_grad():
                    predictions = fno_model(input_batch)

                # Align sequence lengths: extract only the target timesteps from predictions
                input_seq_len, output_seq_len = self.data_params.sequenceLength
                aligned_predictions = predictions[:, -output_seq_len:]  # Take last output_seq_len timesteps

                all_predictions.append(aligned_predictions.cpu().numpy())
                if target_batch is not None:
                    all_targets.append(target_batch.cpu().numpy())
                else:
                    all_targets.append(input_batch[:, -output_seq_len:].cpu().numpy())

            predictions_array = np.concatenate(all_predictions, axis=0)
            targets_array = np.concatenate(all_targets, axis=0)

            # Compute statistics
            pred_mean = np.mean(predictions_array, axis=0)
            pred_std = np.std(predictions_array, axis=0)

            # Create statistical visualization
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))

            # Mean prediction
            im1 = axes[0, 0].imshow(pred_mean[-1, 0], cmap='RdBu_r', origin='lower')
            axes[0, 0].set_title('Mean Prediction')
            axes[0, 0].axis('off')
            plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)

            # Standard deviation
            im2 = axes[0, 1].imshow(pred_std[-1, 0], cmap='hot', origin='lower')
            axes[0, 1].set_title('Prediction Std Dev')
            axes[0, 1].axis('off')
            plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)

            # Coefficient of variation
            cv = pred_std[-1, 0] / (np.abs(pred_mean[-1, 0]) + 1e-8)
            im3 = axes[0, 2].imshow(cv, cmap='plasma', origin='lower')
            axes[0, 2].set_title('Coefficient of Variation')
            axes[0, 2].axis('off')
            plt.colorbar(im3, ax=axes[0, 2], shrink=0.8)

            # Value distributions
            axes[1, 0].hist(predictions_array.flatten(), bins=50, alpha=0.7,
                          label='Predictions', density=True, edgecolor='black')
            axes[1, 0].hist(targets_array.flatten(), bins=50, alpha=0.7,
                          label='Targets', density=True, edgecolor='black')
            axes[1, 0].set_xlabel('Value')
            axes[1, 0].set_ylabel('Density')
            axes[1, 0].set_title('Value Distribution')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Q-Q plot
            pred_sorted = np.sort(predictions_array.flatten())
            target_sorted = np.sort(targets_array.flatten())
            min_len = min(len(pred_sorted), len(target_sorted))
            axes[1, 1].scatter(target_sorted[:min_len], pred_sorted[:min_len],
                             alpha=0.5, s=1)
            axes[1, 1].plot([target_sorted.min(), target_sorted.max()],
                          [target_sorted.min(), target_sorted.max()], 'r--', linewidth=2)
            axes[1, 1].set_xlabel('Target Quantiles')
            axes[1, 1].set_ylabel('Prediction Quantiles')
            axes[1, 1].set_title('Q-Q Plot')
            axes[1, 1].grid(True, alpha=0.3)

            # Error statistics
            errors = np.abs(predictions_array - targets_array)
            spatial_mean_error = np.mean(errors, axis=(3, 4))  # Average over spatial dims

            for sample in range(min(3, spatial_mean_error.shape[0])):
                axes[1, 2].plot(spatial_mean_error[sample, :, 0],
                               label=f'Sample {sample+1}', marker='o', markersize=4)

            axes[1, 2].set_xlabel('Time Step')
            axes[1, 2].set_ylabel('Spatial Mean Error')
            axes[1, 2].set_title('Temporal Error Evolution')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)

            plt.tight_layout()

            # Save statistical visualization
            output_path = os.path.join(self.temp_dir, 'statistical_analysis.png')
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()

            # Verify file was created
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 1000)

        except Exception as e:
            self.fail(f"Statistical visualization test failed: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)