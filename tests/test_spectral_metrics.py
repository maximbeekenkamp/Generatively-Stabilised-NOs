"""
Unit tests for spectral metrics (field error and spectrum error).

Tests cover:
- Field error computation and correctness
- Spectrum error computation with radial binning
- Gradient flow for both losses
- Edge cases and numerical stability
- Validation versions (no gradients)

Run:
    python -m pytest tests/test_spectral_metrics.py -v
    python tests/test_spectral_metrics.py  # Direct execution
"""

import torch
import torch.nn as nn
import numpy as np
import pytest

from src.core.training.spectral_metrics import (
    compute_field_error_loss,
    compute_spectrum_error_loss,
    compute_field_error_validation,
    compute_spectrum_error_validation,
    radial_bin_differentiable,
    validate_gradients
)


class TestFieldError:
    """Test field error computation."""

    def test_field_error_identical_inputs(self):
        """Field error should be zero for identical inputs."""
        B, T, C, H, W = 2, 4, 3, 32, 32
        pred = torch.randn(B, T, C, H, W)
        true = pred.clone()

        loss = compute_field_error_loss(pred, true)
        assert loss.item() < 1e-6, f"Expected ~0, got {loss.item()}"

    def test_field_error_per_frame_normalization(self):
        """Field error should normalize per-frame, not globally."""
        B, T, C, H, W = 1, 2, 1, 16, 16

        # Frame 1: small values, Frame 2: large values
        true = torch.cat([
            torch.ones(B, 1, C, H, W) * 0.1,
            torch.ones(B, 1, C, H, W) * 10.0
        ], dim=1)

        # Same relative error in both frames
        pred = true * 1.1  # 10% error

        loss = compute_field_error_loss(pred, true)

        # Should be similar despite magnitude difference
        # Relative error = ((0.11-0.1)^2) / (0.1^2) = 0.01/0.01 = 1.0
        # and ((11-10)^2) / (10^2) = 1/100 = 0.01
        # Average should be ~0.505
        expected = 0.5 * (1.0 + 0.01)
        assert abs(loss.item() - expected) < 0.01, f"Expected {expected:.3f}, got {loss.item():.3f}"

    def test_field_error_4d_input(self):
        """Field error should handle 4D input [B, C, H, W]."""
        B, C, H, W = 2, 3, 32, 32
        pred = torch.randn(B, C, H, W, requires_grad=True)
        true = torch.randn(B, C, H, W)

        loss = compute_field_error_loss(pred, true)
        assert loss.ndim == 0, "Loss should be scalar"
        assert loss.requires_grad, "Loss should require grad"

        # Check gradient flow
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0

    def test_field_error_5d_input(self):
        """Field error should handle 5D input [B, T, C, H, W]."""
        B, T, C, H, W = 2, 4, 3, 32, 32
        pred = torch.randn(B, T, C, H, W, requires_grad=True)
        true = torch.randn(B, T, C, H, W)

        loss = compute_field_error_loss(pred, true)
        assert loss.ndim == 0, "Loss should be scalar"
        assert loss.requires_grad, "Loss should require grad"

        # Check gradient flow
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0

    def test_field_error_validation(self):
        """Validation version should return float without gradients."""
        B, T, C, H, W = 2, 4, 3, 32, 32
        pred = torch.randn(B, T, C, H, W, requires_grad=True)
        true = torch.randn(B, T, C, H, W)

        loss_val = compute_field_error_validation(pred, true)

        assert isinstance(loss_val, float), "Validation should return float"
        assert pred.grad is None, "Should not create gradients"


class TestSpectrumError:
    """Test spectrum error computation."""

    def test_spectrum_error_identical_inputs(self):
        """Spectrum error should be zero for identical inputs."""
        B, T, C, H, W = 2, 4, 3, 32, 32
        pred = torch.randn(B, T, C, H, W)
        true = pred.clone()

        loss = compute_spectrum_error_loss(pred, true)
        assert loss.item() < 1e-5, f"Expected ~0, got {loss.item()}"

    def test_spectrum_error_fft_computation(self):
        """Spectrum error should compute FFT correctly."""
        B, T, C, H, W = 1, 1, 1, 64, 64

        # Create sinusoidal pattern (known frequency content)
        x = torch.linspace(0, 2*np.pi, W)
        y = torch.linspace(0, 2*np.pi, H)
        X, Y = torch.meshgrid(x, y, indexing='ij')

        # Single frequency: k=2
        true = torch.sin(2 * X).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # Slightly different: k=2 with phase shift
        pred = torch.sin(2 * X + 0.1).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        loss = compute_spectrum_error_loss(pred, true)

        # Should detect the difference in phase
        assert loss.item() > 0, "Should detect difference"
        assert loss.item() < 1.0, "Error should be reasonable"

    def test_spectrum_error_multi_channel(self):
        """Spectrum error should average over channels."""
        B, T, C, H, W = 1, 2, 3, 32, 32
        pred = torch.randn(B, T, C, H, W, requires_grad=True)
        true = torch.randn(B, T, C, H, W)

        loss = compute_spectrum_error_loss(pred, true)

        assert loss.ndim == 0, "Loss should be scalar"
        assert loss.requires_grad, "Loss should require grad"

        # Check gradient flow
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0

    def test_spectrum_error_radial_binning(self):
        """Test radial binning produces correct shape."""
        B, T, H, W = 2, 4, 64, 64
        device = 'cpu'

        # Create test field
        field = torch.randn(B, T, H, W)

        # Create wavenumber grid
        kfreq_y = torch.fft.fftfreq(H, device=device) * H
        kfreq_x = torch.fft.fftfreq(W, device=device) * W
        kfreq2D_x, kfreq2D_y = torch.meshgrid(kfreq_x, kfreq_y, indexing='ij')
        knrm = torch.sqrt(kfreq2D_x**2 + kfreq2D_y**2)

        # Define bins
        kbins = torch.arange(0.5, min(H, W)//2 + 1, 1.0, device=device)

        # Bin the field
        binned = radial_bin_differentiable(field, knrm, kbins)

        # Check shape
        expected_shape = (B, T, len(kbins) - 1)
        assert binned.shape == expected_shape, f"Expected {expected_shape}, got {binned.shape}"

        # Check all values are non-negative (power spectrum)
        assert (binned >= 0).all(), "Binned values should be non-negative"

    def test_spectrum_error_validation(self):
        """Validation version should return float without gradients."""
        B, T, C, H, W = 2, 4, 3, 32, 32
        pred = torch.randn(B, T, C, H, W, requires_grad=True)
        true = torch.randn(B, T, C, H, W)

        loss_val = compute_spectrum_error_validation(pred, true)

        assert isinstance(loss_val, float), "Validation should return float"
        assert pred.grad is None, "Should not create gradients"


class TestGradientFlow:
    """Test gradient flow through both losses."""

    def test_field_error_gradients_exist(self):
        """Field error should produce non-zero gradients."""
        B, T, C, H, W = 2, 4, 3, 32, 32
        pred = torch.randn(B, T, C, H, W, requires_grad=True)
        true = torch.randn(B, T, C, H, W)

        loss = compute_field_error_loss(pred, true)
        loss.backward()

        assert pred.grad is not None, "Gradients should exist"
        assert pred.grad.abs().sum() > 0, "Gradients should be non-zero"

    def test_spectrum_error_gradients_exist(self):
        """Spectrum error should produce non-zero gradients."""
        B, T, C, H, W = 2, 4, 3, 32, 32
        pred = torch.randn(B, T, C, H, W, requires_grad=True)
        true = torch.randn(B, T, C, H, W)

        loss = compute_spectrum_error_loss(pred, true)
        loss.backward()

        assert pred.grad is not None, "Gradients should exist"
        assert pred.grad.abs().sum() > 0, "Gradients should be non-zero"

    def test_validate_gradients_helper(self):
        """Test the gradient validation helper function."""
        B, T, C, H, W = 2, 4, 3, 32, 32
        pred = torch.randn(B, T, C, H, W, requires_grad=True)
        true = torch.randn(B, T, C, H, W)

        results = validate_gradients(pred, true)

        assert 'field_error_gradients' in results
        assert 'spectrum_error_gradients' in results
        assert results['field_error_gradients'], "Field error should have gradients"
        assert results['spectrum_error_gradients'], "Spectrum error should have gradients"


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_constant_field(self):
        """Losses should handle constant fields (zero variance)."""
        B, T, C, H, W = 2, 4, 3, 32, 32

        # Constant prediction and truth
        pred = torch.ones(B, T, C, H, W) * 5.0
        true = torch.ones(B, T, C, H, W) * 5.0

        # Field error should be zero (epsilon prevents division by zero)
        field_loss = compute_field_error_loss(pred, true)
        assert field_loss.item() < 1e-5, "Field error should be ~0 for identical constants"

        # Spectrum error should be zero
        spec_loss = compute_spectrum_error_loss(pred, true)
        assert spec_loss.item() < 1e-5, "Spectrum error should be ~0 for identical constants"

    def test_zero_field(self):
        """Losses should handle zero fields gracefully."""
        B, T, C, H, W = 2, 4, 3, 32, 32

        pred = torch.zeros(B, T, C, H, W)
        true = torch.zeros(B, T, C, H, W)

        # Field error with epsilon should not crash
        field_loss = compute_field_error_loss(pred, true)
        assert not torch.isnan(field_loss), "Field error should not be NaN"
        assert not torch.isinf(field_loss), "Field error should not be Inf"

        # Spectrum error with epsilon should not crash
        spec_loss = compute_spectrum_error_loss(pred, true)
        assert not torch.isnan(spec_loss), "Spectrum error should not be NaN"
        assert not torch.isinf(spec_loss), "Spectrum error should not be Inf"

    def test_large_values(self):
        """Losses should handle large values without overflow."""
        B, T, C, H, W = 2, 4, 3, 32, 32

        pred = torch.randn(B, T, C, H, W) * 1000  # Large values
        true = torch.randn(B, T, C, H, W) * 1000

        field_loss = compute_field_error_loss(pred, true)
        assert not torch.isnan(field_loss), "Field error should not overflow"
        assert not torch.isinf(field_loss), "Field error should not be Inf"

        spec_loss = compute_spectrum_error_loss(pred, true)
        assert not torch.isnan(spec_loss), "Spectrum error should not overflow"
        assert not torch.isinf(spec_loss), "Spectrum error should not be Inf"

    def test_small_spatial_dims(self):
        """Losses should work with small spatial dimensions."""
        B, T, C, H, W = 2, 4, 3, 8, 8  # Small spatial dims

        pred = torch.randn(B, T, C, H, W, requires_grad=True)
        true = torch.randn(B, T, C, H, W)

        # Both losses should work
        field_loss = compute_field_error_loss(pred, true)
        spec_loss = compute_spectrum_error_loss(pred, true)

        assert field_loss.item() >= 0, "Field loss should be non-negative"
        assert spec_loss.item() >= 0, "Spectrum loss should be non-negative"

        # Gradients should flow
        (field_loss + spec_loss).backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0


def run_all_tests():
    """Run all tests manually (for direct execution)."""
    print("Running spectral metrics tests...")

    test_classes = [
        TestFieldError,
        TestSpectrumError,
        TestGradientFlow,
        TestEdgeCases
    ]

    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Testing: {test_class.__name__}")
        print(f"{'='*60}")

        test_instance = test_class()
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]

        for method_name in test_methods:
            try:
                print(f"  {method_name}...", end=" ")
                getattr(test_instance, method_name)()
                print("✓ PASS")
            except AssertionError as e:
                print(f"✗ FAIL: {e}")
            except Exception as e:
                print(f"✗ ERROR: {e}")

    print(f"\n{'='*60}")
    print("All tests completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_all_tests()
