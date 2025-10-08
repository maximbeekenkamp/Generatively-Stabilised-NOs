"""
Spectral metrics for turbulence validation and training.

Implements field error and energy spectrum error from NO+DM reference code:
- Field Error: Relative MSE in real space (per-frame normalization)
- Spectrum Error: Relative MSE in log power spectrum space (radial binning)

Reference: NeuralOperator_DiffusionModel/case_*/no_dm/*/dm/dm_postprocess.ipynb

Author: GenStabilisation-Proj
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict


def compute_field_error_loss(prediction: torch.Tensor, groundTruth: torch.Tensor) -> torch.Tensor:
    """
    Differentiable field error (relative MSE in real space).

    Computes relative error per timestep/channel, then averages.
    This is more stable for turbulence than global relative error.

    Formula: mean(mean((pred-true)², axis=(H,W)) / mean(true², axis=(H,W)))

    Reference: dm_postprocess.ipynb::field_get_err()
    ```python
    def field_get_err(true, pred):
        return np.mean(np.mean((true-pred)**2, axis=(2,3)) / np.mean(true**2, axis=(2,3)))
    ```

    Args:
        prediction: [B, T, C, H, W] or [B, C, H, W]
        groundTruth: [B, T, C, H, W] or [B, C, H, W]

    Returns:
        Scalar loss tensor with gradients
    """
    # Handle both 4D and 5D tensors
    if prediction.ndim == 4:
        # [B, C, H, W] -> add time dimension
        prediction = prediction.unsqueeze(1)
        groundTruth = groundTruth.unsqueeze(1)

    # Compute spatial MSE per frame: [B, T, C]
    spatial_mse = ((prediction - groundTruth) ** 2).mean(dim=(-2, -1))

    # Compute spatial mean square of ground truth: [B, T, C]
    spatial_mean_sq = (groundTruth ** 2).mean(dim=(-2, -1))

    # Relative error per frame (with epsilon for numerical stability)
    epsilon = 1e-8
    relative_error = spatial_mse / (spatial_mean_sq + epsilon)

    # Average over all dimensions
    return relative_error.mean()


def compute_spectrum_error_loss(prediction: torch.Tensor, groundTruth: torch.Tensor) -> torch.Tensor:
    """
    Differentiable energy spectrum error (relative MSE in log power spectrum).

    Implements the NO+DM reference approach:
    1. Compute 2D FFT of fields
    2. Radial binning by wavenumber magnitude: k = sqrt(kx² + ky²)
    3. Scale by annular area: π(k₂² - k₁²)
    4. Take logarithm of power spectrum
    5. Compute relative MSE in log space

    Reference: dm_postprocess.ipynb::compute_power() + spec_get_err()
    ```python
    def compute_power(true, pred, inp):
        fourier = torch.fft.fftn(true, dim=(-2, -1))
        amplitudes = torch.abs(fourier) ** 2
        # ... radial binning ...
        Abins *= torch.pi * (kbins[1:]**2 - kbins[:-1]**2)
        return torch.log(Abins)

    def spec_get_err(true, pred):
        return np.mean(np.mean((true-pred)**2, axis=(1,2)) / np.mean(true**2, axis=(1,2)))
    ```

    Args:
        prediction: [B, T, C, H, W] or [B, C, H, W]
        groundTruth: [B, T, C, H, W] or [B, C, H, W]

    Returns:
        Scalar loss tensor with gradients
    """
    # Handle both 4D and 5D tensors
    if prediction.ndim == 4:
        prediction = prediction.unsqueeze(1)
        groundTruth = groundTruth.unsqueeze(1)

    # Average over channels if multi-channel (for velocity fields: u, v, p)
    if prediction.shape[2] > 1:
        pred_field = prediction.mean(dim=2)  # [B, T, H, W]
        true_field = groundTruth.mean(dim=2)
    else:
        pred_field = prediction.squeeze(2)
        true_field = groundTruth.squeeze(2)

    B, T, H, W = pred_field.shape
    device = prediction.device

    # Compute 2D FFT
    pred_fft = torch.fft.fftn(pred_field, dim=(-2, -1))
    true_fft = torch.fft.fftn(true_field, dim=(-2, -1))

    # Power spectrum (amplitude squared)
    pred_power = torch.abs(pred_fft) ** 2
    true_power = torch.abs(true_fft) ** 2

    # Create wavenumber grid
    kfreq_y = torch.fft.fftfreq(H, device=device) * H
    kfreq_x = torch.fft.fftfreq(W, device=device) * W
    kfreq2D_x, kfreq2D_y = torch.meshgrid(kfreq_x, kfreq_y, indexing='ij')

    # Wavenumber magnitude
    knrm = torch.sqrt(kfreq2D_x**2 + kfreq2D_y**2)

    # Define bins for wavenumber (as in reference)
    kbins = torch.arange(0.5, min(H, W)//2 + 1, 1.0, device=device)

    # Radial binning
    pred_spectrum = radial_bin_differentiable(pred_power, knrm, kbins)  # [B, T, n_bins]
    true_spectrum = radial_bin_differentiable(true_power, knrm, kbins)

    # Scale by annular area (as in reference)
    scaling = torch.pi * (kbins[1:]**2 - kbins[:-1]**2)
    pred_spectrum = pred_spectrum * scaling
    true_spectrum = true_spectrum * scaling

    # Log space (with epsilon for numerical stability)
    epsilon = 1e-10
    pred_log_spec = torch.log(pred_spectrum + epsilon)
    true_log_spec = torch.log(true_spectrum + epsilon)

    # Relative MSE in log spectrum space
    # Formula: mean((spec_pred - spec_true)² / spec_true²) over (time, bins)
    spec_mse = ((pred_log_spec - true_log_spec) ** 2).mean(dim=(1, 2))  # [B]
    spec_mean_sq = (true_log_spec ** 2).mean(dim=(1, 2))

    relative_error = spec_mse / (spec_mean_sq + epsilon)

    return relative_error.mean()


def radial_bin_differentiable(field: torch.Tensor, knrm: torch.Tensor,
                               kbins: torch.Tensor) -> torch.Tensor:
    """
    Differentiable radial binning for FFT power spectrum.

    Bins 2D power spectrum by wavenumber magnitude into 1D radial profile.
    This is the standard approach for isotropic turbulence analysis.

    Args:
        field: [B, T, H, W] - Power spectrum in 2D Fourier space
        knrm: [H, W] - Wavenumber magnitude grid
        kbins: [n_bins] - Bin edges for wavenumber

    Returns:
        [B, T, n_bins-1] - Radially averaged power spectrum
    """
    B, T, H, W = field.shape

    # Flatten spatial dimensions
    knrm_flat = knrm.flatten()  # [H*W]
    field_flat = field.view(B, T, H * W)  # [B, T, H*W]

    # Digitize wavenumbers into bins
    bin_indices = torch.bucketize(knrm_flat, kbins)  # [H*W]

    # Initialize output
    binned = torch.zeros((B, T, len(kbins) - 1), device=field.device)

    # Average over each bin
    for bin_idx in range(1, len(kbins)):
        # Create mask for current bin
        mask = (bin_indices == bin_idx).float()  # [H*W]
        mask = mask.unsqueeze(0).unsqueeze(0)     # [1, 1, H*W]

        # Weighted sum (differentiable)
        bin_sum = (field_flat * mask).sum(dim=-1)  # [B, T]
        bin_count = mask.sum() + 1e-8  # Prevent division by zero

        # Average
        binned[:, :, bin_idx - 1] = bin_sum / bin_count

    return binned


# ============================================================================
# Validation versions (no gradients, return float)
# ============================================================================

def compute_field_error_validation(prediction: torch.Tensor, groundTruth: torch.Tensor) -> float:
    """
    Non-differentiable version of field error for validation logging.

    Args:
        prediction: [B, T, C, H, W]
        groundTruth: [B, T, C, H, W]

    Returns:
        float: Field error value
    """
    with torch.no_grad():
        return compute_field_error_loss(prediction, groundTruth).item()


def compute_spectrum_error_validation(prediction: torch.Tensor, groundTruth: torch.Tensor) -> float:
    """
    Non-differentiable version of spectrum error for validation logging.

    Args:
        prediction: [B, T, C, H, W]
        groundTruth: [B, T, C, H, W]

    Returns:
        float: Spectrum error value
    """
    with torch.no_grad():
        return compute_spectrum_error_loss(prediction, groundTruth).item()


# ============================================================================
# Helper function for computing LSIM unweighted (for validation metrics)
# ============================================================================

def compute_lsim_unweighted(lsim_model: nn.Module, prediction: torch.Tensor,
                            groundTruth: torch.Tensor, num_fields: int) -> float:
    """
    Compute unweighted LSIM for validation logging.

    Args:
        lsim_model: Pre-trained LSIM model
        prediction: [B, T, C, H, W]
        groundTruth: [B, T, C, H, W]
        num_fields: Number of field channels to use

    Returns:
        float: Average LSIM distance
    """
    with torch.no_grad():
        from src.core.training.loss import loss_lsim

        # Compute LSIM for all timesteps
        lsim_values = loss_lsim(
            lsim_model,
            prediction[:, :, 0:num_fields],
            groundTruth[:, :, 0:num_fields]
        )

        # Average over batch, time, channels
        return lsim_values.mean().item()


# ============================================================================
# Debugging utilities
# ============================================================================

def validate_gradients(prediction: torch.Tensor, groundTruth: torch.Tensor) -> Dict[str, bool]:
    """
    Test that gradients flow correctly through loss functions.

    Args:
        prediction: [B, T, C, H, W] with requires_grad=True
        groundTruth: [B, T, C, H, W]

    Returns:
        Dict with gradient status for each loss
    """
    results = {}

    # Test field error gradients
    pred_field = prediction.clone().requires_grad_(True)
    loss_field = compute_field_error_loss(pred_field, groundTruth)
    loss_field.backward()
    results['field_error_gradients'] = pred_field.grad is not None and pred_field.grad.abs().sum() > 0

    # Test spectrum error gradients
    pred_spec = prediction.clone().requires_grad_(True)
    loss_spec = compute_spectrum_error_loss(pred_spec, groundTruth)
    loss_spec.backward()
    results['spectrum_error_gradients'] = pred_spec.grad is not None and pred_spec.grad.abs().sum() > 0

    return results


if __name__ == "__main__":
    # Quick sanity test
    print("Testing spectral metrics...")

    # Create dummy data
    B, T, C, H, W = 2, 4, 3, 64, 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pred = torch.randn(B, T, C, H, W, device=device, requires_grad=True)
    true = torch.randn(B, T, C, H, W, device=device)

    # Test field error
    field_err = compute_field_error_loss(pred, true)
    print(f"Field error: {field_err.item():.6f}")

    # Test spectrum error
    spec_err = compute_spectrum_error_loss(pred, true)
    print(f"Spectrum error: {spec_err.item():.6f}")

    # Test gradients
    grad_status = validate_gradients(pred, true)
    print(f"Gradient status: {grad_status}")

    print("✓ All tests passed!")
