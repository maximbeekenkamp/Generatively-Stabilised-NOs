"""
Unit tests for spectral visualization module.

Tests cover:
- Energy spectrum computation (FFT, radial binning)
- POD analysis (SVD, eigenvalues, modes)
- Plotting functions (energy spectrum, POD comparison)
- Edge cases and numerical stability

Run:
    python -m pytest tests/unit/analysis/test_spectral_visualization.py -v
    python tests/unit/analysis/test_spectral_visualization.py  # Direct execution
"""

import torch
import numpy as np
import pytest
from pathlib import Path
import tempfile

from src.analysis.spectral_visualization import (
    compute_energy_spectrum,
    plot_energy_spectrum_comparison,
    compute_pod_analysis,
    plot_pod_comparison
)


class TestEnergySpectrum:
    """Test energy spectrum computation."""

    def test_compute_spectrum_2d_input(self):
        """Energy spectrum should work with 2D input."""
        H, W = 64, 64
        field = torch.randn(H, W)

        k, E = compute_energy_spectrum(field, wavenumber_range=(1, 32))

        assert len(k) == len(E), f"k and E lengths should match: {len(k)} vs {len(E)}"
        assert len(k) == 32, f"Expected 32 bins for range (1,32), got {len(k)}"
        # Some bins might be empty (giving NaN), filter them out
        valid_E = E[~np.isnan(E)]
        assert len(valid_E) > 0, "Should have some valid energy values"
        assert (valid_E >= 0).all(), "Valid energy should be non-negative"

    def test_compute_spectrum_5d_input(self):
        """Energy spectrum should average over batch, time, channels for 5D input."""
        B, T, C, H, W = 2, 4, 3, 64, 64
        field = torch.randn(B, T, C, H, W)

        k, E = compute_energy_spectrum(field, wavenumber_range=(1, 32))

        assert len(k) == len(E)
        assert E.min() >= 0
        assert not np.isnan(E).any()

    def test_spectrum_sinusoidal_field(self):
        """Energy spectrum should detect dominant frequency."""
        H, W = 64, 64
        x = torch.linspace(0, 2*np.pi, W)
        y = torch.linspace(0, 2*np.pi, H)
        X, Y = torch.meshgrid(x, y, indexing='ij')

        # Single frequency: k=4
        field = torch.sin(4 * X)

        k, E = compute_energy_spectrum(field, wavenumber_range=(1, 32))

        # Energy should peak near k=4
        peak_idx = np.argmax(E)
        peak_k = k[peak_idx]
        assert 3 <= peak_k <= 5, f"Peak should be near k=4, got k={peak_k}"

    def test_spectrum_different_ranges(self):
        """Energy spectrum should work with different wavenumber ranges."""
        field = torch.randn(64, 64)

        k1, E1 = compute_energy_spectrum(field, wavenumber_range=(1, 16))
        k2, E2 = compute_energy_spectrum(field, wavenumber_range=(1, 32))

        assert len(k1) < len(k2), "Larger range should have more bins"
        assert k1.max() < k2.max()


class TestPlotEnergySpectrum:
    """Test energy spectrum plotting."""

    def test_plot_single_model(self):
        """Plot should work with single model prediction."""
        B, T, C, H, W = 2, 4, 2, 64, 64
        ground_truth = torch.randn(B, T, C, H, W)
        predictions = {'model1': torch.randn(B, T, C, H, W)}

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'spectrum.png'

            spectra = plot_energy_spectrum_comparison(
                ground_truth, predictions, str(save_path)
            )

            assert save_path.exists(), "Plot file should be created"
            assert 'ground_truth' in spectra
            assert 'model1' in spectra

    def test_plot_multiple_models(self):
        """Plot should work with multiple model predictions."""
        B, T, C, H, W = 2, 4, 2, 64, 64
        ground_truth = torch.randn(B, T, C, H, W)
        predictions = {
            'model1': torch.randn(B, T, C, H, W),
            'model2': torch.randn(B, T, C, H, W),
            'model3': torch.randn(B, T, C, H, W)
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'spectrum.png'

            spectra = plot_energy_spectrum_comparison(
                ground_truth, predictions, str(save_path)
            )

            assert save_path.exists()
            assert len(spectra) == 4  # GT + 3 models

    def test_plot_subset_of_models(self):
        """Plot should work with model_names parameter."""
        B, T, C, H, W = 2, 4, 2, 64, 64
        ground_truth = torch.randn(B, T, C, H, W)
        predictions = {
            'model1': torch.randn(B, T, C, H, W),
            'model2': torch.randn(B, T, C, H, W),
            'model3': torch.randn(B, T, C, H, W)
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'spectrum.png'

            spectra = plot_energy_spectrum_comparison(
                ground_truth, predictions, str(save_path),
                model_names=['model1', 'model2']  # Only plot 2 of 3
            )

            assert 'model1' in spectra
            assert 'model2' in spectra
            assert 'model3' not in spectra  # Should not be plotted


class TestPODAnalysis:
    """Test POD analysis computation."""

    def test_pod_basic_computation(self):
        """POD should compute eigenvalues and modes correctly."""
        B, T, C, H, W = 1, 10, 1, 32, 32
        ground_truth = torch.randn(B, T, C, H, W)
        prediction = torch.randn(B, T, C, H, W)

        pod = compute_pod_analysis(ground_truth, prediction, n_modes=5)

        assert 'eigenvalues' in pod
        assert 'modes' in pod
        assert 'coefficients' in pod
        assert 'reconstruction_error' in pod
        assert 'prediction_error' in pod
        assert 'cumulative_energy' in pod

        assert len(pod['eigenvalues']) == 5
        assert pod['modes'].shape == (5, H, W)
        assert pod['coefficients'].shape == (5, T)

    def test_pod_eigenvalue_decay(self):
        """POD eigenvalues should be in decreasing order."""
        B, T, C, H, W = 1, 10, 1, 32, 32
        ground_truth = torch.randn(B, T, C, H, W)

        pod = compute_pod_analysis(ground_truth, ground_truth, n_modes=5)

        eigenvalues = pod['eigenvalues']
        for i in range(len(eigenvalues) - 1):
            assert eigenvalues[i] >= eigenvalues[i+1], \
                f"Eigenvalues should decrease: {eigenvalues[i]} < {eigenvalues[i+1]}"

    def test_pod_cumulative_energy(self):
        """POD cumulative energy should increase monotonically."""
        B, T, C, H, W = 1, 10, 1, 32, 32
        ground_truth = torch.randn(B, T, C, H, W)

        pod = compute_pod_analysis(ground_truth, ground_truth, n_modes=5)

        cum_energy = pod['cumulative_energy']
        for i in range(len(cum_energy) - 1):
            assert cum_energy[i] <= cum_energy[i+1], \
                "Cumulative energy should increase"
        assert 0 <= cum_energy[-1] <= 1, "Cumulative energy should be in [0, 1]"

    def test_pod_identical_input(self):
        """POD should have low prediction error for identical inputs."""
        B, T, C, H, W = 1, 10, 1, 32, 32
        ground_truth = torch.randn(B, T, C, H, W)

        pod = compute_pod_analysis(ground_truth, ground_truth, n_modes=5)

        # With truncation to n_modes < timesteps, some error is expected
        # But it should still be reasonable
        assert pod['prediction_error'] < 1.0, \
            f"Expected small error for identical inputs, got {pod['prediction_error']}"
        assert pod['reconstruction_error'] < 1.0, \
            f"Expected small reconstruction error, got {pod['reconstruction_error']}"

    def test_pod_different_n_modes(self):
        """POD should work with different numbers of modes."""
        B, T, C, H, W = 1, 10, 1, 32, 32
        ground_truth = torch.randn(B, T, C, H, W)

        for n_modes in [3, 5, 8]:
            pod = compute_pod_analysis(ground_truth, ground_truth, n_modes=n_modes)
            assert len(pod['eigenvalues']) == n_modes
            assert pod['modes'].shape[0] == n_modes


class TestPlotPODComparison:
    """Test POD comparison plotting."""

    def test_plot_pod_single_model(self):
        """POD plot should work with single model."""
        B, T, C, H, W = 1, 10, 1, 32, 32
        ground_truth = torch.randn(B, T, C, H, W)
        prediction = torch.randn(B, T, C, H, W)

        gt_pod = compute_pod_analysis(ground_truth, ground_truth, n_modes=5)
        pred_pod = {'model1': compute_pod_analysis(ground_truth, prediction, n_modes=5)}

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'pod.png'

            plot_pod_comparison(gt_pod, pred_pod, str(save_path))

            assert save_path.exists(), "POD plot file should be created"

    def test_plot_pod_multiple_models(self):
        """POD plot should work with multiple models."""
        B, T, C, H, W = 1, 10, 1, 32, 32
        ground_truth = torch.randn(B, T, C, H, W)

        gt_pod = compute_pod_analysis(ground_truth, ground_truth, n_modes=5)
        pred_pod = {
            'model1': compute_pod_analysis(ground_truth, torch.randn(B, T, C, H, W), n_modes=5),
            'model2': compute_pod_analysis(ground_truth, torch.randn(B, T, C, H, W), n_modes=5)
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'pod.png'

            plot_pod_comparison(gt_pod, pred_pod, str(save_path))

            assert save_path.exists()


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_constant_field(self):
        """Spectrum and POD should handle constant fields."""
        H, W = 32, 32
        field = torch.ones(H, W) * 5.0

        # Spectrum should not crash
        k, E = compute_energy_spectrum(field, wavenumber_range=(1, 16))
        assert not np.isnan(E).any()
        assert not np.isinf(E).any()

    def test_zero_field(self):
        """Spectrum and POD should handle zero fields gracefully."""
        B, T, C, H, W = 1, 10, 1, 32, 32
        field = torch.zeros(B, T, C, H, W)

        # Spectrum should not crash
        k, E = compute_energy_spectrum(field, wavenumber_range=(1, 16))
        assert not np.isnan(E).any()
        assert not np.isinf(E).any()

        # POD should not crash
        pod = compute_pod_analysis(field, field, n_modes=5)
        assert not np.isnan(pod['eigenvalues']).any()
        assert not np.isinf(pod['eigenvalues']).any()

    def test_large_values(self):
        """Spectrum and POD should handle large values without overflow."""
        B, T, C, H, W = 1, 10, 1, 32, 32
        field = torch.randn(B, T, C, H, W) * 1000

        k, E = compute_energy_spectrum(field, wavenumber_range=(1, 16))
        assert not np.isnan(E).any()
        assert not np.isinf(E).any()

        pod = compute_pod_analysis(field, field, n_modes=5)
        assert not np.isnan(pod['eigenvalues']).any()
        assert not np.isinf(pod['eigenvalues']).any()

    def test_small_spatial_dims(self):
        """Should work with small spatial dimensions."""
        B, T, C, H, W = 1, 10, 1, 16, 16
        field = torch.randn(B, T, C, H, W)

        k, E = compute_energy_spectrum(field, wavenumber_range=(1, 8))
        assert len(k) > 0

        pod = compute_pod_analysis(field, field, n_modes=3)
        assert len(pod['eigenvalues']) == 3


def run_all_tests():
    """Run all tests manually (for direct execution)."""
    print("Running spectral visualization tests...")

    test_classes = [
        TestEnergySpectrum,
        TestPlotEnergySpectrum,
        TestPODAnalysis,
        TestPlotPODComparison,
        TestEdgeCases
    ]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Testing: {test_class.__name__}")
        print(f"{'='*60}")

        test_instance = test_class()
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]

        for method_name in test_methods:
            total_tests += 1
            try:
                print(f"  {method_name}...", end=" ")
                getattr(test_instance, method_name)()
                print("✓ PASS")
                passed_tests += 1
            except AssertionError as e:
                print(f"✗ FAIL: {e}")
            except Exception as e:
                print(f"✗ ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed_tests}/{total_tests} tests passed")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_all_tests()
