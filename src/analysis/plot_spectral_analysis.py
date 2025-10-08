#!/usr/bin/env python3
"""
Spectral Analysis Plotting Script

Generates energy spectrum and POD analysis plots for model groups.
This script is called by plot_data_groups.py for each group of models.

Usage:
    python src/analysis/plot_spectral_analysis.py \
        --progress-dir results/tra_verification \
        --models fno,tno,unet,deeponet \
        --group-name neural_operators \
        --output-dir analysis/generated_plots/neural_operators

Input:
    - Ground truth: {progress_dir}/sampling/groundTruth.dict
    - Predictions: {progress_dir}/sampling/{model}.dict for each model

Output:
    - {output_dir}/energy_spectrum.png
    - {output_dir}/pod_comparison.png

Author: GenStabilisation-Proj
Date: 2025-10-08
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.spectral_visualization import (
    plot_energy_spectrum_comparison,
    compute_pod_analysis,
    plot_pod_comparison
)


def load_predictions(file_path: Path) -> torch.Tensor:
    """
    Load prediction data from pickle file.

    Args:
        file_path: Path to .dict file containing predictions

    Returns:
        Predictions tensor [B, T, C, H, W]
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {file_path}")

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # Handle different data formats
    if isinstance(data, dict):
        # Try common keys
        for key in ['predictions', 'data', 'output', 'samples']:
            if key in data:
                data = data[key]
                break

    # Convert to tensor if needed
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    elif not isinstance(data, torch.Tensor):
        raise ValueError(f"Unsupported data type: {type(data)}")

    # Ensure 5D format [B, T, C, H, W]
    if data.ndim == 4:  # [B, C, H, W]
        data = data.unsqueeze(1)  # Add time dimension
    elif data.ndim != 5:
        raise ValueError(f"Expected 4D or 5D tensor, got {data.ndim}D")

    return data


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate spectral analysis plots for model groups'
    )

    parser.add_argument(
        '--progress-dir',
        type=str,
        required=True,
        help='Directory containing sampling results'
    )

    parser.add_argument(
        '--models',
        type=str,
        required=True,
        help='Comma-separated list of model names (e.g., fno,tno,unet)'
    )

    parser.add_argument(
        '--group-name',
        type=str,
        required=True,
        help='Name of the model group (e.g., neural_operators, no_dm)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save plots'
    )

    parser.add_argument(
        '--wavenumber-range',
        type=str,
        default='1,64',
        help='Wavenumber range for spectrum analysis (e.g., 1,64)'
    )

    parser.add_argument(
        '--n-modes',
        type=int,
        default=10,
        help='Number of POD modes to compute (default: 10)'
    )

    parser.add_argument(
        '--skip-missing',
        action='store_true',
        help='Skip models with missing prediction files instead of erroring'
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    # Parse paths
    progress_dir = Path(args.progress_dir)
    output_dir = Path(args.output_dir)
    sampling_dir = progress_dir / 'sampling'

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse model list
    models = [m.strip() for m in args.models.split(',')]

    # Parse wavenumber range
    k_min, k_max = map(int, args.wavenumber_range.split(','))
    wavenumber_range = (k_min, k_max)

    print("=" * 60)
    print(f"Spectral Analysis: {args.group_name}")
    print("=" * 60)
    print(f"Progress dir: {progress_dir}")
    print(f"Models: {', '.join(models)}")
    print(f"Wavenumber range: k ∈ [{k_min}, {k_max}]")
    print(f"POD modes: {args.n_modes}")
    print()

    # Load ground truth
    gt_path = sampling_dir / 'groundTruth.dict'
    if not gt_path.exists():
        print(f"✗ Ground truth file not found: {gt_path}")
        sys.exit(1)

    print(f"Loading ground truth from: {gt_path}")
    ground_truth = load_predictions(gt_path)
    print(f"  Shape: {ground_truth.shape}")
    print()

    # Load model predictions
    predictions = {}
    missing_models = []

    for model in models:
        pred_path = sampling_dir / f'{model}.dict'

        if not pred_path.exists():
            if args.skip_missing:
                print(f"⚠ Skipping {model}: file not found")
                missing_models.append(model)
                continue
            else:
                print(f"✗ Prediction file not found: {pred_path}")
                sys.exit(1)

        print(f"Loading {model} predictions from: {pred_path}")
        try:
            predictions[model] = load_predictions(pred_path)
            print(f"  Shape: {predictions[model].shape}")
        except Exception as e:
            if args.skip_missing:
                print(f"⚠ Skipping {model}: {e}")
                missing_models.append(model)
            else:
                print(f"✗ Error loading {model}: {e}")
                sys.exit(1)

    if not predictions:
        print("✗ No valid predictions loaded")
        sys.exit(1)

    print()
    print(f"✓ Loaded predictions for {len(predictions)} models")
    if missing_models:
        print(f"⚠ Skipped {len(missing_models)} models: {', '.join(missing_models)}")
    print()

    # ========================================================================
    # 1. Energy Spectrum Analysis
    # ========================================================================
    print("📊 Generating energy spectrum plot...")

    spectrum_path = output_dir / 'energy_spectrum.png'
    title = f"Energy Spectrum Comparison - {args.group_name.replace('_', ' ').title()}"

    try:
        spectra = plot_energy_spectrum_comparison(
            ground_truth=ground_truth,
            predictions=predictions,
            save_path=str(spectrum_path),
            wavenumber_range=wavenumber_range,
            model_names=list(predictions.keys()),
            title=title,
            kolmogorov_slope=True
        )

        # Compute relative spectrum errors
        k_gt, E_gt = spectra['ground_truth']
        print()
        print("Spectrum Errors (relative MSE in log space):")
        for model, (k, E) in spectra.items():
            if model != 'ground_truth':
                # Relative error in log spectrum
                log_error = np.mean((np.log(E + 1e-10) - np.log(E_gt + 1e-10))**2)
                print(f"  {model:12s}: {log_error:.6f}")

    except Exception as e:
        print(f"✗ Error generating energy spectrum plot: {e}")
        import traceback
        traceback.print_exc()

    print()

    # ========================================================================
    # 2. POD Analysis
    # ========================================================================
    print("📊 Generating POD analysis plot...")

    # Compute POD for ground truth
    print("  Computing ground truth POD...")
    gt_pod = compute_pod_analysis(
        ground_truth=ground_truth,
        prediction=ground_truth,
        n_modes=args.n_modes
    )

    # Compute POD for predictions
    predictions_pod = {}
    for model, pred in predictions.items():
        print(f"  Computing {model} POD...")
        try:
            predictions_pod[model] = compute_pod_analysis(
                ground_truth=ground_truth,
                prediction=pred,
                n_modes=args.n_modes
            )
        except Exception as e:
            print(f"    ⚠ Error computing POD for {model}: {e}")

    # Generate comparison plot
    pod_path = output_dir / 'pod_comparison.png'
    title = f"POD Analysis - {args.group_name.replace('_', ' ').title()}"

    try:
        plot_pod_comparison(
            ground_truth_pod=gt_pod,
            predictions_pod=predictions_pod,
            save_path=str(pod_path),
            model_names=list(predictions_pod.keys()),
            title=title
        )

        # Print POD statistics
        print()
        print(f"POD Statistics ({args.n_modes} modes):")
        print(f"  Ground Truth:")
        print(f"    Cumulative energy: {gt_pod['cumulative_energy'][-1]*100:.2f}%")
        print(f"    Reconstruction error: {gt_pod['reconstruction_error']:.6f}")
        print()

        for model, pod in predictions_pod.items():
            print(f"  {model}:")
            print(f"    Cumulative energy: {pod['cumulative_energy'][-1]*100:.2f}%")
            print(f"    Prediction error: {pod['prediction_error']:.6f}")

    except Exception as e:
        print(f"✗ Error generating POD plot: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("=" * 60)
    print("✓ Spectral analysis complete")
    print(f"✓ Plots saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
