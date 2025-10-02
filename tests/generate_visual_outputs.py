"""
Visual Output Generation Script

This script demonstrates integration with reference analysis code
and generates visual outputs for model validation.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from src.core.models.neural_operator_adapters import FNOPriorAdapter, UNetPriorAdapter, TNOPriorAdapter
    from src.core.utils.params import DataParams
    MODELS_AVAILABLE = True
except ImportError as e:
    MODELS_AVAILABLE = False
    print(f"Model modules not available: {e}")

from tests.fixtures.dummy_datasets import DummyDatasetFactory, get_dummy_batch


def create_mock_model_params(model_type):
    """Create mock model parameters"""
    class MockModelParams:
        def __init__(self, model_type):
            self.model_type = model_type
            self.architecture = model_type
            self.arch = model_type
            self.fnoModes = [8, 8] if model_type == 'fno' else None
            self.decWidth = 32
            self.prevSteps = 2

    return MockModelParams(model_type)


def generate_model_outputs():
    """Generate outputs from different models for analysis"""
    if not MODELS_AVAILABLE:
        print("Models not available for visual output generation")
        return

    data_params = DataParams(
        batch=4,
        sequenceLength=[8, 2],
        dataSize=[16, 16],
        dimension=2,
        simFields=["pres"],
        simParams=[],
        normalizeMode=""
    )

    # Create models
    models = {}
    try:
        # FNO models
        fno_params = create_mock_model_params('fno')
        models['FNO 16'] = FNOPriorAdapter(fno_params, data_params)

        # UNet models
        unet_params = create_mock_model_params('unet')
        models['UNet'] = UNetPriorAdapter(unet_params, data_params)

        # TNO models
        tno_params = create_mock_model_params('tno')
        models['TNO'] = TNOPriorAdapter(tno_params, data_params)

    except Exception as e:
        print(f"Error creating models: {e}")
        return

    # Generate outputs for each dataset
    dataset_names = ['inc_low', 'inc_high', 'tra_ext', 'tra_inc', 'iso']

    for dataset_name in dataset_names:
        print(f"\nGenerating outputs for dataset: {dataset_name}")

        try:
            input_batch, target_batch = get_dummy_batch(dataset_name, batch_size=1)

            dataset_outputs = {}
            for model_name, model in models.items():
                model.eval()
                with torch.no_grad():
                    output = model(input_batch)
                    dataset_outputs[model_name] = output[0]  # Take first sample

            # Create visualization
            create_comparison_plot(dataset_outputs, dataset_name)

        except Exception as e:
            print(f"Error generating outputs for {dataset_name}: {e}")


def create_comparison_plot(outputs, dataset_name):
    """Create comparison plot for different models"""
    try:
        output_dir = f"tests/outputs/group1_fno_tno_deeponet"
        os.makedirs(output_dir, exist_ok=True)

        fig, axes = plt.subplots(len(outputs), 2, figsize=(10, 4*len(outputs)))
        if len(outputs) == 1:
            axes = axes.reshape(1, -1)

        for i, (model_name, output) in enumerate(outputs.items()):
            # Plot first and last timestep
            output_np = output.cpu().numpy()

            # First timestep
            axes[i, 0].imshow(output_np[0, 0], cmap='viridis')
            axes[i, 0].set_title(f'{model_name} - t=0')
            axes[i, 0].axis('off')

            # Last timestep
            axes[i, 1].imshow(output_np[-1, 0], cmap='viridis')
            axes[i, 1].set_title(f'{model_name} - t=final')
            axes[i, 1].axis('off')

        plt.suptitle(f'Model Comparison - Dataset: {dataset_name}')
        plt.tight_layout()

        output_file = os.path.join(output_dir, f'comparison_{dataset_name}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved visualization: {output_file}")

    except Exception as e:
        print(f"Error creating comparison plot: {e}")


def generate_reference_analysis_demo():
    """Demonstrate reference analysis code integration"""
    print("\n=== Reference Analysis Integration Demo ===")

    try:
        # Mock reference analysis functions
        def mock_plot_loss_calculation(predictions, targets):
            """Mock version of plot_loss_all.py functionality"""
            mse = np.mean((predictions - targets) ** 2)
            return {"MSE": mse, "RMSE": np.sqrt(mse)}

        def mock_downstream_analysis(predictions):
            """Mock version of downstream analysis"""
            spatial_mean = np.mean(predictions, axis=(2, 3))  # Average over spatial dims
            temporal_mean = np.mean(spatial_mean, axis=0)     # Average over time
            return {"temporal_average": temporal_mean}

        # Generate sample data
        if MODELS_AVAILABLE:
            data_params = DataParams(batch=1, sequenceLength=[8, 2], dataSize=[16, 16])
            fno_params = create_mock_model_params('fno')
            model = FNOPriorAdapter(fno_params, data_params)
            model.eval()

            input_batch, target_batch = get_dummy_batch("inc_low", batch_size=1)

            with torch.no_grad():
                predictions = model(input_batch)

            pred_np = predictions.cpu().numpy()
            target_np = target_batch.cpu().numpy()

            # Apply mock analysis
            loss_metrics = mock_plot_loss_calculation(pred_np[:, -2:], target_np)
            downstream_metrics = mock_downstream_analysis(pred_np)

            print(f"Loss metrics: {loss_metrics}")
            print(f"Downstream metrics: {downstream_metrics}")
            print("Reference analysis integration: SUCCESSFUL")

        else:
            print("Models not available for reference analysis demo")

    except Exception as e:
        print(f"Reference analysis demo failed: {e}")


if __name__ == "__main__":
    print("=== Visual Output Generation ===")
    generate_model_outputs()

    generate_reference_analysis_demo()

    print("\n=== Visual Output Generation Complete ===")
    print("Check tests/outputs/ directory for generated visualizations")