"""
Spectral Visualization for Turbulence Validation

This module implements energy spectrum and POD analysis tools extracted from the
NO+DM reference code (NeuralOperator_DiffusionModel/dm_postprocess.ipynb).

Energy Spectrum Analysis:
    - Computes E(k) via 2D FFT and radial binning
    - Plots wavenumber vs energy on log-log scale
    - Includes Kolmogorov -5/3 reference line for turbulence

POD (Proper Orthogonal Decomposition) Analysis:
    - Singular Value Decomposition of temporal sequences
    - Eigenvalue decay comparison
    - Spatial mode shape visualization

Reference: NeuralOperator_DiffusionModel/case_*/no_dm/*/dm/dm_postprocess.ipynb

Author: GenStabilisation-Proj
Date: 2025-10-08
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional, Any
from pathlib import Path


def compute_energy_spectrum(
    field: torch.Tensor,
    wavenumber_range: Tuple[int, int] = (1, 64)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute radially-averaged energy spectrum E(k) from 2D spatial field.

    This implements the standard turbulence analysis approach:
    1. 2D FFT of spatial field
    2. Compute power spectrum |F{u}(k)|²
    3. Radial binning by wavenumber magnitude k = sqrt(kx² + ky²)
    4. Scale by annular area π(k₂² - k₁²)

    Reference: dm_postprocess.ipynb::compute_power()

    Args:
        field: Spatial field [H, W] or [B, T, C, H, W] (will be averaged)
        wavenumber_range: (k_min, k_max) for binning

    Returns:
        wavenumbers: Array of bin centers [n_bins]
        spectrum: Radially-averaged energy E(k) [n_bins]
    """
    # Handle multi-dimensional input
    if field.ndim == 5:  # [B, T, C, H, W]
        # Average over batch, time, channels
        field = field.mean(dim=(0, 1, 2))  # [H, W]
    elif field.ndim == 4:  # [B, C, H, W]
        field = field.mean(dim=(0, 1))  # [H, W]
    elif field.ndim == 3:  # [T, H, W] or [C, H, W]
        field = field.mean(dim=0)  # [H, W]
    elif field.ndim != 2:
        raise ValueError(f"Expected 2D-5D tensor, got {field.ndim}D")

    # Ensure numpy array
    if isinstance(field, torch.Tensor):
        field = field.detach().cpu().numpy()

    H, W = field.shape

    # 2D FFT
    field_fft = np.fft.fft2(field)
    # Power spectrum
    power = np.abs(field_fft) ** 2

    # Wavenumber grid
    kfreq_y = np.fft.fftfreq(H) * H
    kfreq_x = np.fft.fftfreq(W) * W
    kfreq2D_x, kfreq2D_y = np.meshgrid(kfreq_x, kfreq_y, indexing='ij')

    # Wavenumber magnitude
    knrm = np.sqrt(kfreq2D_x**2 + kfreq2D_y**2)

    # Define bins
    kmin, kmax = wavenumber_range
    kbins = np.arange(kmin - 0.5, kmax + 1.5, 1.0)

    # Radial binning
    bin_indices = np.digitize(knrm.flatten(), kbins)
    power_flat = power.flatten()

    spectrum = np.zeros(len(kbins) - 1)
    for bin_idx in range(1, len(kbins)):
        mask = (bin_indices == bin_idx)
        if mask.sum() > 0:
            spectrum[bin_idx - 1] = power_flat[mask].mean()

    # Scale by annular area (as in reference)
    scaling = np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    spectrum = spectrum * scaling

    # Bin centers
    wavenumbers = (kbins[:-1] + kbins[1:]) / 2

    return wavenumbers, spectrum


def plot_energy_spectrum_comparison(
    ground_truth: torch.Tensor,
    predictions: Dict[str, torch.Tensor],
    save_path: str,
    wavenumber_range: Tuple[int, int] = (1, 64),
    model_names: Optional[List[str]] = None,
    title: str = "Energy Spectrum Comparison",
    kolmogorov_slope: bool = True
) -> Dict[str, np.ndarray]:
    """
    Plot energy spectrum E(k) comparison for multiple models.

    Creates log-log plot with:
    - Ground truth spectrum
    - Model prediction spectra
    - Kolmogorov -5/3 reference line (optional)

    Args:
        ground_truth: Ground truth data [B, T, C, H, W]
        predictions: Dict mapping model names to predictions [B, T, C, H, W]
        save_path: Output file path
        wavenumber_range: (k_min, k_max) for analysis
        model_names: Subset of models to plot (None = all)
        title: Plot title
        kolmogorov_slope: Add -5/3 reference line

    Returns:
        Dictionary of (wavenumbers, spectrum) arrays for each model
    """
    if model_names is None:
        model_names = list(predictions.keys())

    # Compute spectra
    k_gt, E_gt = compute_energy_spectrum(ground_truth, wavenumber_range)

    spectra = {'ground_truth': (k_gt, E_gt)}
    for name in model_names:
        if name in predictions:
            k, E = compute_energy_spectrum(predictions[name], wavenumber_range)
            spectra[name] = (k, E)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Ground truth
    ax.loglog(k_gt, E_gt, 'k-', linewidth=2.5, label='Ground Truth', alpha=0.9)

    # Model predictions
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    for idx, name in enumerate(model_names):
        if name in spectra and name != 'ground_truth':
            k, E = spectra[name]
            ax.loglog(k, E, linestyle='--', color=colors[idx],
                     linewidth=2, label=name.upper(), alpha=0.8)

    # Kolmogorov -5/3 slope reference
    if kolmogorov_slope:
        k_ref = k_gt[k_gt > 5]  # Show reference in inertial range
        E_ref = k_ref**(-5/3) * (E_gt[5] / (k_gt[5]**(-5/3)))  # Scale to match data
        ax.loglog(k_ref, E_ref, 'r:', linewidth=1.5,
                 label='Kolmogorov -5/3', alpha=0.6)

    ax.set_xlabel('Wavenumber k', fontsize=14)
    ax.set_ylabel('Energy E(k)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, which='both', alpha=0.3, linestyle=':')
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Energy spectrum plot saved to: {save_path}")

    return spectra


def compute_pod_analysis(
    ground_truth: torch.Tensor,
    prediction: torch.Tensor,
    n_modes: int = 10
) -> Dict[str, Any]:
    """
    Compute Proper Orthogonal Decomposition (POD) modes and eigenvalues.

    POD is performed via SVD on the temporal sequence of spatial snapshots:
    U = [u(t₁), u(t₂), ..., u(tₙ)] ∈ ℝ^(space × time)

    SVD: U = Φ Σ Ψᵀ
    - Φ: Spatial modes (eigenvectors)
    - Σ: Singular values (√eigenvalues)
    - Ψ: Temporal coefficients

    Args:
        ground_truth: Ground truth [B, T, C, H, W]
        prediction: Model prediction [B, T, C, H, W]
        n_modes: Number of POD modes to compute

    Returns:
        Dictionary with:
            - 'eigenvalues': Energy of each mode [n_modes]
            - 'modes': Spatial POD modes [n_modes, H, W]
            - 'coefficients': Temporal coefficients [n_modes, T]
            - 'reconstruction_error': Relative error using n_modes
    """
    # Average over batch and channels, keep time
    if ground_truth.ndim == 5:  # [B, T, C, H, W]
        gt_seq = ground_truth.mean(dim=(0, 2))  # [T, H, W]
        pred_seq = prediction.mean(dim=(0, 2))  # [T, H, W]
    else:
        raise ValueError(f"Expected 5D tensor [B,T,C,H,W], got {ground_truth.ndim}D")

    # Convert to numpy and flatten spatial dimensions
    if isinstance(gt_seq, torch.Tensor):
        gt_seq = gt_seq.detach().cpu().numpy()
        pred_seq = pred_seq.detach().cpu().numpy()

    T, H, W = gt_seq.shape
    gt_matrix = gt_seq.reshape(T, H*W).T  # [space, time]
    pred_matrix = pred_seq.reshape(T, H*W).T

    # Center data (remove temporal mean)
    gt_mean = gt_matrix.mean(axis=1, keepdims=True)
    gt_centered = gt_matrix - gt_mean

    # SVD
    U, S, Vt = np.linalg.svd(gt_centered, full_matrices=False)

    # Eigenvalues (energy)
    eigenvalues = (S ** 2) / T  # Normalize by time steps

    # Extract modes
    modes = U[:, :n_modes].T.reshape(n_modes, H, W)  # [n_modes, H, W]
    coefficients = Vt[:n_modes, :]  # [n_modes, T]

    # Reconstruction error using n_modes
    reconstruction = (U[:, :n_modes] @ np.diag(S[:n_modes]) @ Vt[:n_modes, :]) + gt_mean
    reconstruction_error = np.linalg.norm(gt_matrix - reconstruction, 'fro') / np.linalg.norm(gt_matrix, 'fro')

    # Compute relative error for prediction
    pred_centered = pred_matrix - pred_matrix.mean(axis=1, keepdims=True)
    pred_coeffs = U[:, :n_modes].T @ pred_centered  # Project onto GT modes
    pred_reconstruction = (U[:, :n_modes] @ pred_coeffs) + pred_matrix.mean(axis=1, keepdims=True)
    pred_error = np.linalg.norm(pred_matrix - pred_reconstruction, 'fro') / np.linalg.norm(pred_matrix, 'fro')

    return {
        'eigenvalues': eigenvalues[:n_modes],
        'modes': modes,
        'coefficients': coefficients,
        'reconstruction_error': reconstruction_error,
        'prediction_error': pred_error,
        'cumulative_energy': np.cumsum(eigenvalues[:n_modes]) / eigenvalues.sum()
    }


def plot_pod_comparison(
    ground_truth_pod: Dict[str, Any],
    predictions_pod: Dict[str, Dict[str, Any]],
    save_path: str,
    model_names: Optional[List[str]] = None,
    title: str = "POD Analysis Comparison"
):
    """
    Plot POD eigenvalue decay and mode shape comparison.

    Creates multi-panel figure with:
    1. Eigenvalue decay (log scale)
    2. Cumulative energy
    3. First 3 POD mode shapes

    Args:
        ground_truth_pod: POD dict from compute_pod_analysis()
        predictions_pod: Dict mapping model names to POD dicts
        save_path: Output file path
        model_names: Subset of models to plot (None = all)
        title: Plot title
    """
    if model_names is None:
        model_names = list(predictions_pod.keys())

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

    # 1. Eigenvalue decay
    ax_eigen = fig.add_subplot(gs[0, :2])
    n_modes = len(ground_truth_pod['eigenvalues'])
    mode_indices = np.arange(1, n_modes + 1)

    ax_eigen.semilogy(mode_indices, ground_truth_pod['eigenvalues'],
                     'ko-', linewidth=2.5, markersize=8, label='Ground Truth', alpha=0.9)

    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    for idx, name in enumerate(model_names):
        if name in predictions_pod:
            pod = predictions_pod[name]
            ax_eigen.semilogy(mode_indices, pod['eigenvalues'],
                            linestyle='--', marker='s', color=colors[idx],
                            linewidth=2, markersize=6, label=name.upper(), alpha=0.8)

    ax_eigen.set_xlabel('Mode Number', fontsize=12)
    ax_eigen.set_ylabel('Eigenvalue (Energy)', fontsize=12)
    ax_eigen.set_title('POD Eigenvalue Decay', fontsize=14, fontweight='bold')
    ax_eigen.legend(fontsize=10)
    ax_eigen.grid(True, alpha=0.3)

    # 2. Cumulative energy
    ax_cumul = fig.add_subplot(gs[0, 2:])
    ax_cumul.plot(mode_indices, ground_truth_pod['cumulative_energy'] * 100,
                 'ko-', linewidth=2.5, markersize=8, label='Ground Truth', alpha=0.9)

    for idx, name in enumerate(model_names):
        if name in predictions_pod:
            pod = predictions_pod[name]
            ax_cumul.plot(mode_indices, pod['cumulative_energy'] * 100,
                        linestyle='--', marker='s', color=colors[idx],
                        linewidth=2, markersize=6, label=name.upper(), alpha=0.8)

    ax_cumul.set_xlabel('Number of Modes', fontsize=12)
    ax_cumul.set_ylabel('Cumulative Energy (%)', fontsize=12)
    ax_cumul.set_title('Cumulative Energy Capture', fontsize=14, fontweight='bold')
    ax_cumul.legend(fontsize=10)
    ax_cumul.grid(True, alpha=0.3)
    ax_cumul.set_ylim([0, 105])

    # 3. First 3 POD mode shapes (only ground truth for clarity)
    for mode_idx in range(min(3, ground_truth_pod['modes'].shape[0])):
        ax_mode = fig.add_subplot(gs[1, mode_idx])
        mode_shape = ground_truth_pod['modes'][mode_idx]

        im = ax_mode.imshow(mode_shape, cmap='RdBu_r', aspect='auto')
        ax_mode.set_title(f'Mode {mode_idx + 1}', fontsize=11, fontweight='bold')
        ax_mode.axis('off')
        plt.colorbar(im, ax=ax_mode, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ POD comparison plot saved to: {save_path}")


if __name__ == "__main__":
    # Self-test with dummy data
    print("Testing spectral visualization module...")
    print()

    # Create dummy turbulence-like data
    B, T, C, H, W = 2, 10, 2, 64, 64
    torch.manual_seed(42)

    # Ground truth: sum of multiple frequencies
    x = torch.linspace(0, 2*np.pi, W)
    y = torch.linspace(0, 2*np.pi, H)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    gt = (torch.sin(2*X) * torch.cos(2*Y) +
          0.5 * torch.sin(4*X) * torch.cos(4*Y) +
          0.1 * torch.randn(H, W))
    ground_truth = gt.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, T, C, H, W)

    # Predictions: slightly smoothed versions
    pred_no = ground_truth * 0.95 + 0.05 * torch.randn_like(ground_truth)
    pred_dm = ground_truth * 0.98 + 0.02 * torch.randn_like(ground_truth)

    predictions = {
        'no': pred_no,
        'no_dm': pred_dm
    }

    # Test 1: Energy spectrum
    print("Test 1: Energy spectrum")
    spectra = plot_energy_spectrum_comparison(
        ground_truth, predictions,
        save_path='test_energy_spectrum.png',
        title='Test: Energy Spectrum'
    )
    assert 'ground_truth' in spectra
    assert 'no' in spectra
    print(f"  ✓ Generated spectrum plot with {len(spectra)} curves")
    print()

    # Test 2: POD analysis
    print("Test 2: POD analysis")
    gt_pod = compute_pod_analysis(ground_truth, ground_truth, n_modes=5)
    no_pod = compute_pod_analysis(ground_truth, pred_no, n_modes=5)
    dm_pod = compute_pod_analysis(ground_truth, pred_dm, n_modes=5)

    assert len(gt_pod['eigenvalues']) == 5
    assert gt_pod['modes'].shape == (5, H, W)
    print(f"  ✓ Computed POD with {len(gt_pod['eigenvalues'])} modes")
    print(f"  ✓ Cumulative energy (5 modes): {gt_pod['cumulative_energy'][-1]*100:.1f}%")
    print()

    # Test 3: POD comparison plot
    print("Test 3: POD comparison plot")
    plot_pod_comparison(
        gt_pod, {'no': no_pod, 'no_dm': dm_pod},
        save_path='test_pod_comparison.png',
        title='Test: POD Comparison'
    )
    print()

    print("✓ All tests passed!")
    print("Test plots saved: test_energy_spectrum.png, test_pod_comparison.png")
