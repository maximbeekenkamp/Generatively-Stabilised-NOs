"""
Comprehensive Visual Output Generation Demo
Runs the entire codebase on dummy datasets and generates all visualizations.
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.core.models.neural_operator_adapters import FNOPriorAdapter, UNetPriorAdapter, TNOPriorAdapter
    from src.core.utils.params import DataParams
    MODELS_AVAILABLE = True
except ImportError as e:
    MODELS_AVAILABLE = False
    print(f"Model modules not available: {e}")

from tests.fixtures.dummy_datasets import get_dummy_batch


def create_model_params(model_type):
    """Create model parameters for each model type"""
    class MockModelParams:
        def __init__(self, model_type):
            self.model_type = model_type
            self.architecture = model_type
            self.arch = model_type
            self.fnoModes = [8, 8] if model_type == 'fno' else None
            self.decWidth = 32
            self.prevSteps = 8

    return MockModelParams(model_type)


def generate_field_visualizations(output_dir):
    """Generate field visualizations for all datasets and models"""
    print("\n" + "="*60)
    print("GENERATING FIELD VISUALIZATIONS")
    print("="*60)

    if not MODELS_AVAILABLE:
        print("Models not available. Skipping.")
        return

    data_params = DataParams(
        batch=1,
        sequenceLength=[8, 2],
        dataSize=[32, 32],
        dimension=2,
        simFields=["pres"],
        simParams=[],
        normalizeMode=""
    )

    # Create models
    models = {
        'FNO': FNOPriorAdapter(create_model_params('fno'), data_params),
        'UNet': UNetPriorAdapter(create_model_params('unet'), data_params),
        'TNO': TNOPriorAdapter(create_model_params('tno'), data_params)
    }

    datasets = ['inc_low', 'inc_high', 'tra_ext', 'tra_inc', 'iso']

    for dataset_name in datasets:
        print(f"\nProcessing dataset: {dataset_name}")

        try:
            input_batch, target_batch = get_dummy_batch(dataset_name, batch_size=1)

            # Generate predictions from all models
            predictions = {}
            for model_name, model in models.items():
                model.eval()
                with torch.no_grad():
                    output = model(input_batch)
                    predictions[model_name] = output[0]  # [T, C, H, W]

            # Create comparison visualization
            fig, axes = plt.subplots(len(models) + 1, 3, figsize=(12, 4*(len(models)+1)))

            # Ground truth row
            gt = target_batch[0].cpu().numpy()  # [T, C, H, W]
            axes[0, 0].imshow(gt[0, 0], cmap='viridis')
            axes[0, 0].set_title('Ground Truth - t=0')
            axes[0, 0].axis('off')

            axes[0, 1].imshow(gt[-1, 0], cmap='viridis')
            axes[0, 1].set_title('Ground Truth - t=final')
            axes[0, 1].axis('off')

            axes[0, 2].text(0.5, 0.5, f'Dataset: {dataset_name}\nShape: {gt.shape}',
                           ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].axis('off')

            # Model predictions rows
            for i, (model_name, pred) in enumerate(predictions.items(), 1):
                pred_np = pred.cpu().numpy()

                axes[i, 0].imshow(pred_np[0, 0], cmap='viridis')
                axes[i, 0].set_title(f'{model_name} - t=0')
                axes[i, 0].axis('off')

                axes[i, 1].imshow(pred_np[-1, 0], cmap='viridis')
                axes[i, 1].set_title(f'{model_name} - t=final')
                axes[i, 1].axis('off')

                # Error visualization (only compare overlapping timesteps)
                min_t = min(pred_np.shape[0], gt.shape[0])
                error = np.abs(pred_np[:min_t] - gt[:min_t])
                axes[i, 2].imshow(error[-1, 0], cmap='hot')
                axes[i, 2].set_title(f'{model_name} - Error')
                axes[i, 2].axis('off')

            plt.suptitle(f'Field Visualization: {dataset_name}', fontsize=16)
            plt.tight_layout()

            save_path = os.path.join(output_dir, f'field_viz_{dataset_name}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"  ✓ Saved: {save_path}")

        except Exception as e:
            print(f"  ✗ Error: {e}")


def generate_temporal_sequences(output_dir):
    """Generate temporal sequence visualizations"""
    print("\n" + "="*60)
    print("GENERATING TEMPORAL SEQUENCE VISUALIZATIONS")
    print("="*60)

    if not MODELS_AVAILABLE:
        print("Models not available. Skipping.")
        return

    data_params = DataParams(
        batch=1,
        sequenceLength=[8, 2],
        dataSize=[32, 32],
        dimension=2,
        simFields=["pres"],
        simParams=[],
        normalizeMode=""
    )

    model = FNOPriorAdapter(create_model_params('fno'), data_params)
    model.eval()

    dataset_name = 'inc_low'
    input_batch, target_batch = get_dummy_batch(dataset_name, batch_size=1)

    with torch.no_grad():
        output = model(input_batch)

    # Create temporal sequence plot
    pred_np = output[0].cpu().numpy()  # [T, C, H, W]
    T = pred_np.shape[0]

    fig, axes = plt.subplots(2, T, figsize=(3*T, 6))

    for t in range(T):
        # Prediction
        axes[0, t].imshow(pred_np[t, 0], cmap='viridis')
        axes[0, t].set_title(f'Pred t={t}')
        axes[0, t].axis('off')

        # Ground truth
        if t < target_batch.shape[1]:
            axes[1, t].imshow(target_batch[0, t, 0].cpu().numpy(), cmap='viridis')
            axes[1, t].set_title(f'GT t={t}')
        else:
            axes[1, t].text(0.5, 0.5, 'N/A', ha='center', va='center',
                           transform=axes[1, t].transAxes)
        axes[1, t].axis('off')

    plt.suptitle('Temporal Sequence Evolution', fontsize=16)
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'temporal_sequence.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {save_path}")


def generate_statistical_plots(output_dir):
    """Generate statistical plots (loss curves, metrics)"""
    print("\n" + "="*60)
    print("GENERATING STATISTICAL PLOTS")
    print("="*60)

    if not MODELS_AVAILABLE:
        print("Models not available. Skipping.")
        return

    data_params = DataParams(
        batch=1,
        sequenceLength=[8, 2],
        dataSize=[32, 32],
        dimension=2,
        simFields=["pres"],
        simParams=[],
        normalizeMode=""
    )

    models = {
        'FNO': FNOPriorAdapter(create_model_params('fno'), data_params),
        'UNet': UNetPriorAdapter(create_model_params('unet'), data_params),
        'TNO': TNOPriorAdapter(create_model_params('tno'), data_params)
    }

    dataset_name = 'inc_low'
    input_batch, target_batch = get_dummy_batch(dataset_name, batch_size=1)

    # Compute metrics for each model
    metrics = {}
    for model_name, model in models.items():
        model.eval()
        with torch.no_grad():
            output = model(input_batch)

        # Compute MSE over time
        mse_per_timestep = []
        for t in range(min(output.shape[1], target_batch.shape[1])):
            mse = torch.mean((output[0, t] - target_batch[0, t])**2).item()
            mse_per_timestep.append(mse)

        metrics[model_name] = {
            'mse_per_timestep': mse_per_timestep,
            'avg_mse': np.mean(mse_per_timestep)
        }

    # Plot metrics
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # MSE over time
    for model_name, model_metrics in metrics.items():
        axes[0].plot(model_metrics['mse_per_timestep'], marker='o', label=model_name)
    axes[0].set_xlabel('Timestep')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('MSE Evolution Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Average MSE comparison
    model_names = list(metrics.keys())
    avg_mses = [metrics[name]['avg_mse'] for name in model_names]
    axes[1].bar(model_names, avg_mses, color=['blue', 'green', 'orange'])
    axes[1].set_ylabel('Average MSE')
    axes[1].set_title('Model Comparison - Average MSE')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'statistical_metrics.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {save_path}")


def generate_frequency_domain_plots(output_dir):
    """Generate frequency domain visualizations"""
    print("\n" + "="*60)
    print("GENERATING FREQUENCY DOMAIN PLOTS")
    print("="*60)

    if not MODELS_AVAILABLE:
        print("Models not available. Skipping.")
        return

    data_params = DataParams(
        batch=1,
        sequenceLength=[8, 2],
        dataSize=[32, 32],
        dimension=2,
        simFields=["pres"],
        simParams=[],
        normalizeMode=""
    )

    model = FNOPriorAdapter(create_model_params('fno'), data_params)
    model.eval()

    dataset_name = 'inc_low'
    input_batch, target_batch = get_dummy_batch(dataset_name, batch_size=1)

    with torch.no_grad():
        output = model(input_batch)

    # Compute FFT
    pred_np = output[0, -1, 0].cpu().numpy()  # Last timestep, first channel
    gt_np = target_batch[0, -1, 0].cpu().numpy()

    pred_fft = np.fft.fft2(pred_np)
    gt_fft = np.fft.fft2(gt_np)

    pred_power = np.abs(np.fft.fftshift(pred_fft))**2
    gt_power = np.abs(np.fft.fftshift(gt_fft))**2

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Spatial domain
    im0 = axes[0, 0].imshow(pred_np, cmap='viridis')
    axes[0, 0].set_title('Prediction (Spatial)')
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(gt_np, cmap='viridis')
    axes[0, 1].set_title('Ground Truth (Spatial)')
    plt.colorbar(im1, ax=axes[0, 1])

    # Frequency domain
    im2 = axes[1, 0].imshow(np.log10(pred_power + 1e-10), cmap='hot')
    axes[1, 0].set_title('Prediction (Frequency)')
    plt.colorbar(im2, ax=axes[1, 0])

    im3 = axes[1, 1].imshow(np.log10(gt_power + 1e-10), cmap='hot')
    axes[1, 1].set_title('Ground Truth (Frequency)')
    plt.colorbar(im3, ax=axes[1, 1])

    plt.suptitle('Spatial vs Frequency Domain Analysis', fontsize=16)
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'frequency_domain.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {save_path}")


def main():
    """Main execution function"""
    print("="*60)
    print("COMPREHENSIVE VISUAL OUTPUT GENERATION")
    print("="*60)
    print(f"Models available: {MODELS_AVAILABLE}")

    # Create output directory
    output_dir = "tests/outputs/comprehensive_demo"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Run all visualization generations
    generate_field_visualizations(output_dir)
    generate_temporal_sequences(output_dir)
    generate_statistical_plots(output_dir)
    generate_frequency_domain_plots(output_dir)

    # Summary
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)

    # List all generated files
    output_files = sorted(Path(output_dir).glob('*.png'))
    print(f"\nGenerated {len(output_files)} visualizations:")
    for f in output_files:
        print(f"  ✓ {f.name}")

    print(f"\nAll outputs saved to: {output_dir}")
    print("\nYou can now verify these work locally before running on Colab!")


if __name__ == "__main__":
    main()
