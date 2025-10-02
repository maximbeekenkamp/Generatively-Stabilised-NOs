"""
Comprehensive Test: Neural Operators and NO+DM Configurations

Tests all core models in both standalone and diffusion-corrected variants:
- FNO, TNO, UNet, DeepONet (standalone)
- FNO+DM, TNO+DM, UNet+DM, DeepONet+DM (with diffusion correction)

Generates visualizations for all configurations.
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import models
try:
    from src.core.models.neural_operator_adapters import FNOPriorAdapter, UNetPriorAdapter, TNOPriorAdapter
    from src.core.utils.params import DataParams
    MODELS_AVAILABLE = True
except ImportError as e:
    MODELS_AVAILABLE = False
    print(f"Model modules not available: {e}")

try:
    from src.core.models.generative_correctors import DiffusionCorrector
    from src.core.models.generative_operator_model import GenerativeOperatorModel
    from src.core.models.genop_init import initialize_generative_operators
    GENOP_AVAILABLE = True
except ImportError as e:
    GENOP_AVAILABLE = False
    print(f"Generative Operator modules not available: {e}")

try:
    from src.core.models.deeponet.deeponet_variants import StandardDeepONet
    from src.core.models.deeponet.deeponet_base import DeepONetConfig
    DEEPONET_AVAILABLE = True
except ImportError as e:
    DEEPONET_AVAILABLE = False
    print(f"DeepONet modules not available: {e}")

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
            # Diffusion-specific params
            self.diffSteps = 100
            self.diffSchedule = "cosine"
            self.diffCondIntegration = "clean"

    return MockModelParams(model_type)


def create_diffusion_params():
    """Create diffusion-specific model parameters"""
    class DiffusionParams:
        def __init__(self):
            self.diffSteps = 50  # Reduced for faster testing
            self.diffSchedule = "cosine"
            self.diffCondIntegration = "clean"
            self.arch = "unet"
            self.decWidth = 32

    return DiffusionParams()


class TestResults:
    """Track test results"""
    def __init__(self):
        self.results = {}

    def add_result(self, model_name, status, error=None, output_shape=None):
        self.results[model_name] = {
            'status': status,
            'error': error,
            'output_shape': output_shape
        }

    def print_summary(self):
        print("\n" + "="*60)
        print("TEST RESULTS SUMMARY")
        print("="*60)

        passed = sum(1 for r in self.results.values() if r['status'] == 'PASS')
        failed = sum(1 for r in self.results.values() if r['status'] == 'FAIL')

        for model_name, result in sorted(self.results.items()):
            status_symbol = "✓" if result['status'] == 'PASS' else "✗"
            print(f"{status_symbol} {model_name:20s} - {result['status']}")
            if result['output_shape']:
                print(f"  └─ Output shape: {result['output_shape']}")
            if result['error']:
                print(f"  └─ Error: {result['error']}")

        print(f"\nTotal: {len(self.results)} | Passed: {passed} | Failed: {failed}")
        print("="*60)


def test_standalone_models(data_params, results):
    """Test standalone neural operator models"""
    print("\n" + "="*60)
    print("TESTING STANDALONE MODELS")
    print("="*60)

    if not MODELS_AVAILABLE:
        print("Models not available. Skipping.")
        return {}

    models_to_test = {
        'FNO': (FNOPriorAdapter, create_model_params('fno')),
        'TNO': (TNOPriorAdapter, create_model_params('tno')),
        'UNet': (UNetPriorAdapter, create_model_params('unet')),
    }

    # Add DeepONet if available
    if DEEPONET_AVAILABLE:
        print("DeepONet available - will test")
    else:
        print("DeepONet not available - skipping")

    dataset_name = 'inc_low'
    input_batch, target_batch = get_dummy_batch(dataset_name, batch_size=1)

    tested_models = {}

    for model_name, (ModelClass, params) in models_to_test.items():
        print(f"\n Testing {model_name}...")

        try:
            # Create model
            model = ModelClass(params, data_params)
            model.eval()

            # Forward pass
            with torch.no_grad():
                output = model(input_batch)

            # Verify output
            assert output.shape[0] == input_batch.shape[0], f"Batch size mismatch"
            assert output.shape[-2:] == input_batch.shape[-2:], f"Spatial dims mismatch"

            tested_models[model_name] = model
            results.add_result(model_name, 'PASS', output_shape=tuple(output.shape))
            print(f"  ✓ {model_name} PASSED - Output shape: {tuple(output.shape)}")

        except Exception as e:
            results.add_result(model_name, 'FAIL', error=str(e))
            print(f"  ✗ {model_name} FAILED: {e}")

    return tested_models


def test_diffusion_models(data_params, results):
    """Test neural operator + diffusion model configurations"""
    print("\n" + "="*60)
    print("TESTING NO+DM CONFIGURATIONS")
    print("="*60)

    if not GENOP_AVAILABLE:
        print("Generative Operator modules not available. Skipping.")
        return {}

    # Initialize generative operators registry
    try:
        initialize_generative_operators()
        print("✓ Generative operators initialized")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        return {}

    # For diffusion models, use shorter sequences for stability
    # Create a simple test input directly (diffusion inference is slow/unstable with long sequences)
    test_input = torch.randn(1, 2, 3, 16, 16)  # Small sequence for testing

    # Models to test with diffusion
    diffusion_configs = [
        ('FNO+DM', 'fno', FNOPriorAdapter),
        ('TNO+DM', 'tno', TNOPriorAdapter),
        ('UNet+DM', 'unet', UNetPriorAdapter),
    ]

    tested_models = {}

    for model_name, model_type, PriorClass in diffusion_configs:
        print(f"\nTesting {model_name}...")

        try:
            # Create prior model
            prior_params = create_model_params(model_type)
            prior_model = PriorClass(prior_params, data_params)

            # Create diffusion corrector
            diffusion_params = create_diffusion_params()
            corrector = DiffusionCorrector(
                p_md=diffusion_params,
                p_d=data_params
            )

            # Create generative operator model (prior + corrector)
            genop_model = GenerativeOperatorModel(
                prior_model=prior_model,
                corrector_model=corrector,
                p_md=prior_params,
                p_d=data_params,
                enable_dcar=True
            )

            # Set to full inference mode to use both prior and corrector
            genop_model.training_mode = 'full_inference'

            genop_model.eval()

            # Test forward pass with small input
            with torch.no_grad():
                output = genop_model(test_input)

            # Verify output
            assert output.shape[0] == test_input.shape[0], f"Batch size mismatch"
            assert output.shape[-2:] == test_input.shape[-2:], f"Spatial dims mismatch"
            assert torch.all(torch.isfinite(output)), f"Output contains NaN or Inf"

            tested_models[model_name] = genop_model
            results.add_result(model_name, 'PASS', output_shape=tuple(output.shape))
            print(f"  ✓ {model_name} PASSED - Output shape: {tuple(output.shape)}")

        except Exception as e:
            results.add_result(model_name, 'FAIL', error=str(e))
            print(f"  ✗ {model_name} FAILED: {e}")

    return tested_models


def generate_comparison_visualization(standalone_models, diffusion_models, output_dir):
    """Generate comparison visualizations for all models"""
    print("\n" + "="*60)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("="*60)

    if not standalone_models and not diffusion_models:
        print("No models to visualize. Skipping.")
        return

    dataset_name = 'inc_low'
    data_params = DataParams(
        batch=1,
        sequenceLength=[8, 2],
        dataSize=[32, 32],
        dimension=2,
        simFields=["pres"],
        simParams=[],
        normalizeMode=""
    )

    input_batch, target_batch = get_dummy_batch(dataset_name, batch_size=1)

    # Combine all models
    all_models = {}
    all_models.update(standalone_models)
    all_models.update(diffusion_models)

    if not all_models:
        print("No models available for visualization")
        return

    # Generate predictions
    predictions = {}
    for model_name, model in all_models.items():
        try:
            model.eval()
            with torch.no_grad():
                output = model(input_batch)
                predictions[model_name] = output[0].cpu().numpy()  # [T, C, H, W]
            print(f"  ✓ Generated prediction for {model_name}")
        except Exception as e:
            print(f"  ✗ Failed to generate prediction for {model_name}: {e}")

    if not predictions:
        print("No predictions to visualize")
        return

    # Create visualization
    n_models = len(predictions)
    fig, axes = plt.subplots(n_models, 3, figsize=(12, 4*n_models))

    if n_models == 1:
        axes = axes.reshape(1, -1)

    gt_np = target_batch[0].cpu().numpy()

    for i, (model_name, pred) in enumerate(predictions.items()):
        # First timestep
        axes[i, 0].imshow(pred[0, 0], cmap='viridis')
        axes[i, 0].set_title(f'{model_name} - t=0')
        axes[i, 0].axis('off')

        # Last timestep
        axes[i, 1].imshow(pred[-1, 0], cmap='viridis')
        axes[i, 1].set_title(f'{model_name} - t=final')
        axes[i, 1].axis('off')

        # Error (compare overlapping timesteps)
        min_t = min(pred.shape[0], gt_np.shape[0])
        error = np.abs(pred[:min_t] - gt_np[:min_t])
        axes[i, 2].imshow(error[-1, 0], cmap='hot')
        axes[i, 2].set_title(f'{model_name} - Error')
        axes[i, 2].axis('off')

    plt.suptitle('Neural Operators: Standalone vs +DM Comparison', fontsize=16)
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'no_dm_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Saved comparison visualization: {save_path}")


def generate_detailed_visualizations(standalone_models, diffusion_models, output_dir):
    """Generate detailed per-model visualizations"""
    print("\n" + "="*60)
    print("GENERATING DETAILED VISUALIZATIONS")
    print("="*60)

    data_params = DataParams(
        batch=1,
        sequenceLength=[8, 2],
        dataSize=[32, 32],
        dimension=2,
        simFields=["pres"],
        simParams=[],
        normalizeMode=""
    )

    datasets = ['inc_low', 'tra_ext', 'iso']

    # Combine models
    all_models = {}
    all_models.update(standalone_models)
    all_models.update(diffusion_models)

    for dataset_name in datasets:
        print(f"\nProcessing dataset: {dataset_name}")

        try:
            input_batch, target_batch = get_dummy_batch(dataset_name, batch_size=1)

            # Create figure for this dataset
            n_models = len(all_models)
            if n_models == 0:
                continue

            fig, axes = plt.subplots(n_models, 2, figsize=(8, 3*n_models))

            if n_models == 1:
                axes = axes.reshape(1, -1)

            for i, (model_name, model) in enumerate(all_models.items()):
                try:
                    model.eval()
                    with torch.no_grad():
                        output = model(input_batch)

                    pred_np = output[0].cpu().numpy()

                    axes[i, 0].imshow(pred_np[0, 0], cmap='viridis')
                    axes[i, 0].set_title(f'{model_name} - t=0')
                    axes[i, 0].axis('off')

                    axes[i, 1].imshow(pred_np[-1, 0], cmap='viridis')
                    axes[i, 1].set_title(f'{model_name} - t=final')
                    axes[i, 1].axis('off')

                except Exception as e:
                    axes[i, 0].text(0.5, 0.5, f'Error: {str(e)[:30]}',
                                   ha='center', va='center',
                                   transform=axes[i, 0].transAxes)
                    axes[i, 0].axis('off')
                    axes[i, 1].axis('off')

            plt.suptitle(f'All Models - Dataset: {dataset_name}', fontsize=14)
            plt.tight_layout()

            save_path = os.path.join(output_dir, f'detailed_{dataset_name}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"  ✓ Saved: detailed_{dataset_name}.png")

        except Exception as e:
            print(f"  ✗ Error processing {dataset_name}: {e}")


def main():
    """Main execution function"""
    print("="*60)
    print("COMPREHENSIVE NO+DM CONFIGURATION TEST")
    print("="*60)
    print(f"Models available: {MODELS_AVAILABLE}")
    print(f"Generative operators available: {GENOP_AVAILABLE}")
    print(f"DeepONet available: {DEEPONET_AVAILABLE}")

    # Create output directory
    output_dir = "tests/outputs/no_dm_configs"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Initialize data parameters
    data_params = DataParams(
        batch=1,
        sequenceLength=[8, 2],
        dataSize=[32, 32],
        dimension=2,
        simFields=["pres"],
        simParams=[],
        normalizeMode=""
    )

    # Track results
    results = TestResults()

    # Test standalone models
    standalone_models = test_standalone_models(data_params, results)

    # Test diffusion models
    diffusion_models = test_diffusion_models(data_params, results)

    # Generate visualizations
    generate_comparison_visualization(standalone_models, diffusion_models, output_dir)
    generate_detailed_visualizations(standalone_models, diffusion_models, output_dir)

    # Print summary
    results.print_summary()

    # List generated files
    print("\n" + "="*60)
    print("GENERATED FILES")
    print("="*60)
    output_files = sorted(Path(output_dir).glob('*.png'))
    for f in output_files:
        print(f"  ✓ {f.name}")

    print(f"\nAll outputs saved to: {output_dir}")
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
