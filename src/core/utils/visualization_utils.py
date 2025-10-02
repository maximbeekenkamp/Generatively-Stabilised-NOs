"""
Visualization Utilities for Neural Operators

Provides common plotting functions, data loading, and visualization helpers
to reduce code duplication across visualization scripts.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# Configure matplotlib for publication-quality plots
plt.rcParams['pdf.fonttype'] = 42  # prevent type3 fonts
plt.rcParams['ps.fonttype'] = 42


class PlotConfig:
    """Configuration for plot styling and settings."""

    def __init__(self, **kwargs):
        self.figsize = kwargs.get('figsize', (10, 6))
        self.dpi = kwargs.get('dpi', 300)
        self.format = kwargs.get('format', 'pdf')
        self.legend = kwargs.get('legend', True)
        self.grid = kwargs.get('grid', False)
        self.tight_layout = kwargs.get('tight_layout', True)


class DataPathConfig:
    """Configuration for data paths (configurable, not hardcoded)."""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.path.join(os.getcwd(), "results")
        self.sampling_dir = os.path.join(self.base_dir, "sampling")
        self.output_dir = self.base_dir

    def get_prediction_folder(self, dataset_name: str) -> str:
        """Get prediction folder for a dataset."""
        return os.path.join(self.sampling_dir, dataset_name)

    def set_custom_paths(self, sampling_dir: str = None, output_dir: str = None):
        """Update paths to custom locations."""
        if sampling_dir:
            self.sampling_dir = sampling_dir
        if output_dir:
            self.output_dir = output_dir


def setup_matplotlib_backend(interactive: bool = False):
    """Set matplotlib backend (non-interactive for headless environments)."""
    if not interactive:
        matplotlib.use('Agg')
        plt.ioff()
    else:
        plt.ion()


def load_prediction_data(prediction_folder: str,
                        model_names: List[str],
                        validate_exists: bool = True) -> Dict:
    """
    Load prediction data for multiple models from a folder.

    Args:
        prediction_folder: Path to folder containing predictions
        model_names: List of model names to load
        validate_exists: Whether to validate files exist before loading

    Returns:
        Dictionary with loaded prediction data
    """
    results = {}

    for model_name in model_names:
        pred_file = os.path.join(prediction_folder, f"{model_name}.dict")

        if validate_exists and not os.path.exists(pred_file):
            print(f"Warning: Prediction file not found: {pred_file}")
            continue

        try:
            results[model_name] = torch.load(pred_file)
        except Exception as e:
            print(f"Error loading {pred_file}: {e}")

    return results


def load_ground_truth(prediction_folder: str) -> Optional[Dict]:
    """
    Load ground truth data from prediction folder.

    Args:
        prediction_folder: Path to folder containing ground truth

    Returns:
        Ground truth dictionary or None if not found
    """
    gt_file = os.path.join(prediction_folder, "groundTruth.dict")

    if not os.path.exists(gt_file):
        print(f"Warning: Ground truth file not found: {gt_file}")
        return None

    try:
        return torch.load(gt_file)
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        return None


def save_figure(fig, output_path: str, config: PlotConfig = None):
    """
    Save matplotlib figure with consistent settings.

    Args:
        fig: Matplotlib figure object
        output_path: Path to save figure
        config: Plot configuration
    """
    if config is None:
        config = PlotConfig()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if config.tight_layout:
        fig.tight_layout()

    fig.savefig(output_path, dpi=config.dpi, format=config.format, bbox_inches='tight')
    print(f"Saved: {output_path}")


def create_field_visualization(field_data: np.ndarray,
                               colormap: str = 'RdBu_r',
                               vmin: float = None,
                               vmax: float = None,
                               title: str = None,
                               ax = None) -> plt.Figure:
    """
    Create visualization for a single field.

    Args:
        field_data: 2D array to visualize
        colormap: Matplotlib colormap name
        vmin, vmax: Color scale limits
        title: Plot title
        ax: Existing axes to plot on (creates new if None)

    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    im = ax.imshow(field_data, cmap=colormap, vmin=vmin, vmax=vmax, origin='lower')
    plt.colorbar(im, ax=ax)

    if title:
        ax.set_title(title)

    ax.axis('off')

    return fig


def create_temporal_sequence_plot(sequence_data: np.ndarray,
                                  timesteps: List[int],
                                  config: PlotConfig = None) -> plt.Figure:
    """
    Create visualization of temporal sequence.

    Args:
        sequence_data: Shape (T, H, W) temporal sequence
        timesteps: Indices of timesteps to plot
        config: Plot configuration

    Returns:
        Matplotlib figure
    """
    if config is None:
        config = PlotConfig()

    n_steps = len(timesteps)
    fig, axes = plt.subplots(1, n_steps, figsize=(4*n_steps, 4))

    if n_steps == 1:
        axes = [axes]

    for idx, t in enumerate(timesteps):
        create_field_visualization(
            sequence_data[t],
            title=f"t={t}",
            ax=axes[idx]
        )

    return fig


def cleanup_matplotlib_memory():
    """Clean up matplotlib figures to prevent memory leaks."""
    plt.close('all')


# Stub for missing turbpred module
class LSIMStub:
    """
    Stub for missing turbpred.loss.loss_lsim function.

    This provides a placeholder when the external turbpred package is not available.
    Replace with actual implementation if turbpred becomes available.
    """

    @staticmethod
    def loss_lsim(*args, **kwargs):
        raise NotImplementedError(
            "LSIM metric requires the external 'turbpred' package which is not available. "
            "Please install turbpred or use MSE/MAE metrics instead."
        )


# Make stub available as if it were from turbpred
def get_lsim_loss_function():
    """Get LSIM loss function (stub if turbpred not available)."""
    try:
        from turbpred.loss import loss_lsim
        return loss_lsim
    except ImportError:
        print("Warning: turbpred module not found. Using stub implementation.")
        return LSIMStub.loss_lsim
