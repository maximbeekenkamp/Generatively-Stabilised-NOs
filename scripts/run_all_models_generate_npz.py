"""
Run All Models on Dummy Data and Generate NPZ Files

This script:
1. Runs all Neural Operators (FNO, TNO, UNet, DeepONet) on dummy data
2. Runs all NO+DM variants (FNO+DM, TNO+DM, UNet+DM, DeepONet+DM)
3. Generates .npz files in autoreg format for plotting
4. Creates visualization-ready outputs

NPZ Format: [num_models, num_evals, num_sequences, timesteps, channels, H, W]
"""

import torch
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.append(str(Path(__file__).parent / "src"))

from src.core.utils.params import DataParams, ModelParamsDecoder
from src.core.models.model import PredictionModel
from tests.fixtures.dummy_datasets import get_dummy_batch


class ModelConfig:
    """Configuration for a model to test"""
    def __init__(self, name: str, arch: str, diffusion: bool = False):
        self.name = name
        self.arch = arch
        self.diffusion = diffusion


# Define all models to test
ALL_MODELS = [
    # Standalone Neural Operators
    ModelConfig("FNO", "fno", diffusion=False),
    ModelConfig("TNO", "tno", diffusion=False),
    ModelConfig("UNet", "unet", diffusion=False),
    ModelConfig("DeepONet", "deeponet", diffusion=False),

    # NO+DM Variants
    ModelConfig("FNO+DM", "genop-fno-diffusion", diffusion=True),
    ModelConfig("TNO+DM", "genop-tno-diffusion", diffusion=True),
    ModelConfig("UNet+DM", "genop-unet-diffusion", diffusion=True),
    ModelConfig("DeepONet+DM", "genop-deeponet-diffusion", diffusion=True),
]

# Datasets to test on
DATASETS = ['inc_low', 'tra_ext', 'iso']


def create_model_params(arch: str, diffusion: bool = False):
    """Create model parameters for given architecture"""
    class MockParams:
        def __init__(self):
            self.arch = arch
            self.decWidth = 32
            self.pretrained = False
            self.frozen = False
            self.trainingNoise = 0.0
            self.fnoModes = (8, 8)  # FNO Fourier modes
            self.vae = False
            self.refinerStd = 0.0

            # Diffusion parameters (if applicable)
            if diffusion:
                self.diffSteps = 10  # Few steps for fast testing
                self.diffSchedule = "cosine"
                self.diffCondIntegration = "clean"
            else:
                self.diffSteps = 500
                self.diffSchedule = "linear"
                self.diffCondIntegration = "noisy"

    return MockParams()


def run_model_on_dataset(
    model_config: ModelConfig,
    dataset_name: str,
    data_params: DataParams
) -> np.ndarray:
    """
    Run a single model on a dataset and return predictions.

    Returns:
        predictions: numpy array [num_sequences, timesteps, channels, H, W]
    """
    print(f"  Running {model_config.name} on {dataset_name}...")

    try:
        # Create model
        model_params = create_model_params(model_config.arch, model_config.diffusion)
        model = PredictionModel(
            p_d=data_params,
            p_t=None,
            p_l=None,
            p_me=None,
            p_md=model_params,
            p_ml=None,
            useGPU=False
        )
        model.eval()

        # Get dummy data
        # get_dummy_batch returns (input_batch, target_batch)
        input_batch, target_batch = get_dummy_batch(
            dataset_name=dataset_name,
            batch_size=data_params.batch,
            sequence_length=data_params.sequenceLength[0],
            spatial_size=data_params.dataSize[0]
        )
        data = input_batch  # [B, T, C, H, W]

        # Run model
        with torch.no_grad():
            output = model.forwardDirect(data)

        # Convert to numpy and reshape
        # From [B, T, C, H, W] to [num_sequences, timesteps, channels, H, W]
        predictions = output.cpu().numpy()

        print(f"    ✓ Output shape: {predictions.shape}")
        print(f"    ✓ Is finite: {np.isfinite(predictions).all()}")
        print(f"    ✓ Range: [{predictions.min():.4f}, {predictions.max():.4f}]")

        return predictions

    except Exception as e:
        print(f"    ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_npz_autoreg_format(
    predictions_dict: Dict[str, Dict[str, np.ndarray]],
    output_dir: Path
):
    """
    Save predictions in autoreg .npz format.

    Format: [num_models, num_evals, num_sequences, timesteps, channels, H, W]

    Args:
        predictions_dict: {dataset_name: {model_name: predictions}}
        output_dir: Directory to save .npz files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name, model_preds in predictions_dict.items():
        # Filter out failed models
        valid_models = {k: v for k, v in model_preds.items() if v is not None}

        if not valid_models:
            print(f"  ⚠️ No valid predictions for {dataset_name}, skipping...")
            continue

        # Stack all model predictions
        # Each prediction is [num_sequences, timesteps, channels, H, W]
        # We need [num_models, num_evals, num_sequences, timesteps, channels, H, W]

        model_names = list(valid_models.keys())
        predictions_list = []

        for model_name in model_names:
            pred = valid_models[model_name]
            # Add num_evals dimension (set to 1 for single evaluation)
            pred_expanded = np.expand_dims(pred, axis=0)  # [1, num_sequences, timesteps, channels, H, W]
            predictions_list.append(pred_expanded)

        # Stack along model dimension
        stacked_predictions = np.stack(predictions_list, axis=0)  # [num_models, 1, num_sequences, ...]

        # Save to .npz
        output_file = output_dir / f"predictions_{dataset_name}.npz"
        np.savez_compressed(output_file, predFull=stacked_predictions)

        print(f"\n  ✓ Saved {output_file}")
        print(f"    Shape: {stacked_predictions.shape}")
        print(f"    Models: {model_names}")

        # Also save metadata
        metadata = {
            'model_names': model_names,
            'dataset': dataset_name,
            'shape_description': '[num_models, num_evals, num_sequences, timesteps, channels, H, W]'
        }
        metadata_file = output_dir / f"metadata_{dataset_name}.txt"
        with open(metadata_file, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")


def main():
    """Main execution"""
    print("="*80)
    print("Running All Models on Dummy Data")
    print("="*80)

    # Create data parameters
    # Use shorter sequences for diffusion models to be faster
    data_params = DataParams(
        batch=1,
        sequenceLength=[4, 2],  # 4 input steps, 2 output steps
        dataSize=[16, 16],  # Must be list [H, W]
        dimension=2,
        simFields=["pres"],
        simParams=[],
        normalizeMode=""
    )

    print(f"Data params:")
    print(f"  Batch: {data_params.batch}")
    print(f"  Sequence length: {data_params.sequenceLength}")
    print(f"  Data size: {data_params.dataSize} (type: {type(data_params.dataSize)})")
    print(f"  Dimension: {data_params.dimension}")
    print(f"  Sim fields: {data_params.simFields}")

    # Dictionary to store all predictions
    # Structure: {dataset_name: {model_name: predictions}}
    all_predictions = {}

    # Run all models on all datasets
    for dataset_name in DATASETS:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*80}")

        dataset_predictions = {}

        for model_config in ALL_MODELS:
            predictions = run_model_on_dataset(model_config, dataset_name, data_params)
            dataset_predictions[model_config.name] = predictions

        all_predictions[dataset_name] = dataset_predictions

    # Save all predictions in autoreg format
    print(f"\n{'='*80}")
    print("Saving NPZ Files")
    print(f"{'='*80}")

    output_dir = Path("tests/outputs/model_predictions")
    save_npz_autoreg_format(all_predictions, output_dir)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    total_models = len(ALL_MODELS)
    total_datasets = len(DATASETS)

    print(f"Models tested: {total_models}")
    print(f"  Standalone NOs: FNO, TNO, UNet, DeepONet")
    print(f"  NO+DM variants: FNO+DM, TNO+DM, UNet+DM, DeepONet+DM")
    print(f"\nDatasets tested: {total_datasets}")
    print(f"  {', '.join(DATASETS)}")
    print(f"\nOutput directory: {output_dir}")
    print(f"  ✓ NPZ files saved in autoreg format")
    print(f"  ✓ Ready for plotting scripts")

    print(f"\n{'='*80}")
    print("ALL MODELS RUN SUCCESSFULLY ✅")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
