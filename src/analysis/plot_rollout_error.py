"""
Plot Rollout Error vs Frame Number

Visualizes prediction error across autoregressive rollout to analyze model stability.
Shows which models maintain accuracy over long rollouts vs which degrade quickly.

Usage:
    python src/analysis/plot_rollout_error.py \
        --prediction-folder local_progress/sampling \
        --output-path local_progress/plots/rollout_error.png \
        --models unet fno tno
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Dict, List, Tuple


def compute_frame_errors(predictions: np.ndarray, ground_truth: np.ndarray,
                         metric: str = 'mse') -> np.ndarray:
    """
    Compute error for each frame in the rollout.

    Args:
        predictions: [N, T, C, H, W] prediction array
        ground_truth: [N, T, C, H, W] ground truth array
        metric: 'mse' or 'mae'

    Returns:
        errors: [T] array of per-frame errors (averaged over sequences, channels, spatial dims)
    """
    if metric == 'mse':
        # MSE per frame: mean over (sequences, channels, height, width)
        errors = np.mean((predictions - ground_truth) ** 2, axis=(0, 2, 3, 4))
    elif metric == 'mae':
        # MAE per frame
        errors = np.mean(np.abs(predictions - ground_truth), axis=(0, 2, 3, 4))
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return errors


def load_model_predictions(prediction_folder: Path, model_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load predictions and ground truth for a model.

    Args:
        prediction_folder: Path to folder containing .npz files
        model_name: Name of model (e.g., 'unet', 'fno')

    Returns:
        predictions: [N, T, C, H, W] array
        ground_truth: [N, T, C, H, W] array
    """
    # Load predictions
    pred_path = prediction_folder / f"{model_name}_tra.npz"
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_path}")

    predictions = np.load(pred_path)['arr_0']

    # Load ground truth
    gt_path = prediction_folder / "groundTruth.dict"
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")

    ground_truth = torch.load(gt_path, map_location='cpu')['data'].numpy()

    return predictions, ground_truth


def plot_rollout_error(
    prediction_folder: Path,
    model_names: List[str],
    output_path: Path,
    metric: str = 'mse',
    title: str = "Autoregressive Rollout Error",
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Create rollout error plot for multiple models.

    Args:
        prediction_folder: Path to folder containing predictions
        model_names: List of model names to plot
        output_path: Where to save the plot
        metric: 'mse' or 'mae'
        title: Plot title
        figsize: Figure size (width, height)
    """
    # Model display names and colors
    model_display_names = {
        'unet': 'U-Net',
        'fno': 'FNO',
        'tno': 'TNO',
        'deeponet': 'DeepONet',
        'fno_dm': 'FNO+DM',
        'tno_dm': 'TNO+DM',
        'unet_dm': 'U-Net+DM',
        'deeponet_dm': 'DeepONet+DM',
        'resnet': 'ResNet',
        'dil_resnet': 'Dilated ResNet',
        'latent_mgn': 'Latent MGN',
        'acdm': 'ACDM',
        'refiner': 'Refiner'
    }

    model_colors = {
        'unet': '#1f77b4',
        'fno': '#ff7f0e',
        'tno': '#2ca02c',
        'deeponet': '#d62728',
        'fno_dm': '#9467bd',
        'tno_dm': '#8c564b',
        'unet_dm': '#e377c2',
        'deeponet_dm': '#7f7f7f',
        'resnet': '#bcbd22',
        'dil_resnet': '#17becf',
        'latent_mgn': '#ff9896',
        'acdm': '#9edae5',
        'refiner': '#c5b0d5'
    }

    fig, ax = plt.subplots(figsize=figsize)

    # Plot each model
    for model_name in model_names:
        try:
            # Load data
            predictions, ground_truth = load_model_predictions(prediction_folder, model_name)

            # Compute per-frame errors
            errors = compute_frame_errors(predictions, ground_truth, metric=metric)

            # Get display name and color
            display_name = model_display_names.get(model_name, model_name.upper())
            color = model_colors.get(model_name, None)

            # Plot
            frames = np.arange(len(errors))
            ax.plot(frames, errors, label=display_name, linewidth=2, color=color, marker='o', markersize=3)

        except FileNotFoundError as e:
            print(f"  ⚠️  Skipping {model_name}: {e}")
            continue
        except Exception as e:
            print(f"  ❌ Error plotting {model_name}: {str(e)[:100]}")
            continue

    # Formatting
    ax.set_xlabel('Frame Number', fontsize=12)
    metric_label = 'MSE' if metric == 'mse' else 'MAE'
    ax.set_ylabel(f'{metric_label} Error', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale for better visualization

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved rollout error plot: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot autoregressive rollout error")
    parser.add_argument('--prediction-folder', type=str, required=True,
                       help='Path to folder containing prediction .npz files')
    parser.add_argument('--output-path', type=str, required=True,
                       help='Path to save output plot')
    parser.add_argument('--models', nargs='+', required=True,
                       help='List of model names to plot')
    parser.add_argument('--metric', type=str, default='mse', choices=['mse', 'mae'],
                       help='Error metric to use')
    parser.add_argument('--title', type=str, default='Autoregressive Rollout Error',
                       help='Plot title')

    args = parser.parse_args()

    prediction_folder = Path(args.prediction_folder)
    output_path = Path(args.output_path)

    print(f"\n📊 Plotting rollout error for {len(args.models)} models...")
    plot_rollout_error(
        prediction_folder=prediction_folder,
        model_names=args.models,
        output_path=output_path,
        metric=args.metric,
        title=args.title
    )
    print(f"✅ Complete!\n")


if __name__ == "__main__":
    main()
