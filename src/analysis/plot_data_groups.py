"""
Plot Data Groups - Grouped Model Comparison Plotting

This script wraps plot_data.py to support multiple model comparison groups:
- legacy: All original autoreg models (ResNet, Refiner, ACDM, etc.)
- neural_operators: All standalone Neural Operators (FNO, TNO, UNet, DeepONet)
- no_dm: All NO+DM models (FNO+DM, TNO+DM, UNet+DM, DeepONet+DM)
- fno_comparison: FNO vs FNO+DM
- tno_comparison: TNO vs TNO+DM
- unet_comparison: UNet vs UNet+DM
- deeponet_comparison: DeepONet vs DeepONet+DM

Usage:
    python plot_data_groups.py --groups legacy neural_operators
    python plot_data_groups.py --groups fno_comparison tno_comparison
    python plot_data_groups.py --groups all  # Run all groups
"""

import os
import sys
import argparse
from typing import List, Dict
from pathlib import Path

# Model groups definition
MODEL_GROUPS = {
    "legacy": {
        "models": {
            "Simulation": "groundTruth.dict",
            "ResNet": "resnet-m2.npz",
            "Dil-ResNet": "dil-resnet-m2.npz",
            "FNO32": "fno-32modes-m2.npz",
            "U-Net": "unet-m2.npz",
            "U-Net-ut": "unet-m8.npz",
            "Refiner": "refiner-r4_std{std}.npz",  # Template - needs std substitution
            "ACDM": "acdm{steps}.npz",  # Template - needs steps substitution
        },
        "description": "Original AutoReg Models"
    },

    "neural_operators": {
        "models": {
            "Simulation": "groundTruth.dict",
            "FNO": "fno_tra.npz",
            "TNO": "tno_tra.npz",
            "U-Net": "unet_tra.npz",
            "DeepONet": "deeponet_tra.npz",
        },
        "description": "Neural Operators Comparison"
    },

    "no_dm": {
        "models": {
            "Simulation": "groundTruth.dict",
            "FNO+DM": "fno_dm_tra.npz",
            "TNO+DM": "tno_dm_tra.npz",
            "UNet+DM": "unet_dm_tra.npz",
            "DeepONet+DM": "deeponet_dm_tra.npz",
        },
        "description": "NO+DM Models Comparison"
    },

    "fno_comparison": {
        "models": {
            "Simulation": "groundTruth.dict",
            "FNO": "fno_tra.npz",
            "FNO+DM": "fno_dm_tra.npz",
        },
        "description": "FNO vs FNO+DM"
    },

    "tno_comparison": {
        "models": {
            "Simulation": "groundTruth.dict",
            "TNO": "tno_tra.npz",
            "TNO+DM": "tno_dm_tra.npz",
        },
        "description": "TNO vs TNO+DM"
    },

    "unet_comparison": {
        "models": {
            "Simulation": "groundTruth.dict",
            "U-Net": "unet_tra.npz",
            "UNet+DM": "unet_dm_tra.npz",
        },
        "description": "UNet vs UNet+DM"
    },

    "deeponet_comparison": {
        "models": {
            "Simulation": "groundTruth.dict",
            "DeepONet": "deeponet_tra.npz",
            "DeepONet+DM": "deeponet_dm_tra.npz",
        },
        "description": "DeepONet vs DeepONet+DM"
    },
}


def substitute_template_params(models: Dict[str, str], dataset_name: str) -> Dict[str, str]:
    """
    Substitute template parameters in model paths based on dataset.

    Args:
        models: Dictionary of model_name -> file_path
        dataset_name: Dataset identifier (e.g., "zInterp", "extrap", etc.)

    Returns:
        Updated models dictionary with substituted paths
    """
    result = {}

    for model_name, file_path in models.items():
        # Substitute {std} for Refiner models
        if "{std}" in file_path:
            if dataset_name in ["zInterp"]:
                std = "0.00001"
            else:
                std = "0.000001"
            file_path = file_path.format(std=std)

        # Substitute {steps} for ACDM models
        if "{steps}" in file_path:
            if dataset_name in ["zInterp"]:
                steps = "100"
            else:
                steps = "20"
            file_path = file_path.format(steps=steps)

        result[model_name] = file_path

    return result


def create_plot_script_for_group(
    group_name: str,
    models: Dict[str, str],
    dataset_name: str = "zInterp",
    field: str = "vort",
    output_folder: str = "results",
    prediction_folder: str = None
) -> str:
    """
    Generate a Python script that calls plot_data.py for a specific group.

    Returns:
        Path to generated script
    """
    if prediction_folder is None:
        prediction_folder = f"results/sampling/{dataset_name}"

    # Substitute template parameters
    models = substitute_template_params(models, dataset_name)

    script_content = f'''"""
Auto-generated plot script for group: {group_name}
Dataset: {dataset_name}
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Setup project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.plot_color_and_name_mapping import getColor, getModelName, getDatasetName, getFieldIndex, getLossRelevantFields, getColormapAndNorm

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Configuration
datasetName = "{dataset_name}"
modelMinMax = (0,1)
evalMinMax = (0,1)
sequenceMinMax = (0,1)  # Select first sequence for plotting
timeSteps = [14,29,44,59] if datasetName in ["lowRey", "highRey", "varReyIn"] else \\
            [9,49,129,209] if datasetName in ["interp", "extrap", "longer"] else \\
            [9,19,39,59]  # zInterp: 60 timesteps total, indices 0-59
spatialZoom = [[20,84], [0,64]] if datasetName in ["lowRey", "highRey", "varReyIn"] else \\
              [[6,70], [0,64]] if datasetName in ["interp", "extrap", "longer"] else \\
              [[0,40], [0,40]]
field = "{field}"

predictionFolder = "{prediction_folder}"
outputFolder = "{output_folder}"

models = {models}

# Load and process predictions
modelNames = []
frameData = []

for modelName, modelPath in models.items():
    try:
        if modelPath == "groundTruth.dict":
            groundTruthDict = torch.load(os.path.join(predictionFolder, "groundTruth.dict"))
            groundTruth = groundTruthDict["data"]  # [N, T, C, H, W]
            print("Original ground truth shape: %s" % (str(list(groundTruth.shape))))
            # Index: [sequence, :, channels, H, W] then select timesteps
            prediction = groundTruth[sequenceMinMax[0]:sequenceMinMax[1],
                                    :,
                                    :,
                                    spatialZoom[0][0]:spatialZoom[0][1],
                                    spatialZoom[1][0]:spatialZoom[1][1]]
            # Select specific timesteps using advanced indexing
            prediction = prediction[:, timeSteps]  # [sequence, len(timeSteps), C, H, W]
            print("Loaded ground truth with shape: %s" % (str(list(prediction.shape))))

        else:
            fullPath = os.path.join(predictionFolder, modelPath)
            if not os.path.exists(fullPath):
                print(f"⚠️  Skipping {{modelName}}: File not found: {{modelPath}}")
                continue
            prediction = torch.from_numpy(np.load(fullPath)["arr_0"])  # [N, T, C, H, W]
            # Index: [sequence, :, channels, H, W] then select timesteps
            prediction = prediction[sequenceMinMax[0]:sequenceMinMax[1],
                                :,
                                :,
                                spatialZoom[0][0]:spatialZoom[0][1],
                                spatialZoom[1][0]:spatialZoom[1][1]]
            # Select specific timesteps using advanced indexing
            prediction = prediction[:, timeSteps]  # [sequence, len(timeSteps), C, H, W]
            print("Loaded prediction from model %s with shape: %s" % (modelName, str(list(prediction.shape))))

        # Only add to modelNames if loading succeeded
        modelNames += [modelName]
    except Exception as e:
        print(f"⚠️  Skipping {{modelName}}: Error loading data: {{e}}")
        continue

    # Compute vorticity or select field channel
    if field == "vort":
        # prediction shape: [sequence, timesteps, channels, H, W]
        # Extract vx and vy (first two channels)
        vx = prediction[:, :, 0]  # [sequence, timesteps, H, W]
        vy = prediction[:, :, 1]  # [sequence, timesteps, H, W]
        # Compute vorticity for each sequence and timestep
        vort_list = []
        for s in range(vx.shape[0]):
            vort_timesteps = []
            for t in range(vx.shape[1]):
                vxDx, vxDy = torch.gradient(vx[s, t], dim=[0, 1])
                vyDx, vyDy = torch.gradient(vy[s, t], dim=[0, 1])
                vort_timesteps.append(vyDx - vxDy)
            vort_list.append(torch.stack(vort_timesteps))
        prediction = torch.stack(vort_list)  # [sequence, timesteps, H, W]
    else:
        # Select the requested field channel
        prediction = prediction[:, :, getFieldIndex(datasetName, field)]  # [sequence, timesteps, H, W]

    # Reshape for plotting: [sequence, timesteps, H, W] -> [sequence, timesteps, W, H] for imshow
    frameData += [prediction.permute(0, 1, 3, 2).numpy()]

# Check if any models were loaded successfully
if len(modelNames) == 0:
    print("⚠️  No models loaded successfully - skipping plot generation")
    exit(1)

# Compute global min/max across all data for consistent normalization
all_data = np.concatenate([fd.flatten() for fd in frameData])
vmin_data, vmax_data = np.percentile(all_data, [1, 99])  # Use percentiles to avoid outliers

# Create plot
# frameData[i] has shape [sequence=1, timesteps, W, H]
# Extract first sequence for plotting
fig, axs = plt.subplots(nrows=len(modelNames), ncols=len(timeSteps), figsize=(4.5,6.6), dpi=250, squeeze=False)
for i in range(len(modelNames)):
    for j in range(len(timeSteps)):
        ax = axs[i][j]
        cmap, _ = getColormapAndNorm(datasetName, field)
        # Use data-driven normalization for better visibility
        # Index: [model_i][sequence=0, timestep_j, W, H]
        im = ax.imshow(frameData[i][0, j], cmap=cmap, vmin=vmin_data, vmax=vmax_data, interpolation='bilinear', origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(getModelName(modelNames[i]), fontsize=8)
        if i == 0:
            ax.set_title(f"t={{timeSteps[j]}}", fontsize=8)

plt.tight_layout()
output_file = os.path.join(outputFolder, f"plot_{group_name}_{{datasetName}}.png")
os.makedirs(outputFolder, exist_ok=True)
plt.savefig(output_file, bbox_inches='tight', dpi=250)
print(f"Saved plot to {{output_file}}")
plt.close()
'''

    # Write script
    script_dir = Path("src/analysis/generated_plots")
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / f"plot_{group_name}.py"

    with open(script_path, 'w') as f:
        f.write(script_content)

    return str(script_path)


def generate_spectral_analysis(
    group_name: str,
    models: Dict[str, str],
    prediction_folder: str,
    output_folder: str
) -> bool:
    """
    Generate spectral analysis (energy spectrum + POD) for a model group.

    Args:
        group_name: Name of the model group
        models: Dictionary of model names to file paths
        prediction_folder: Path to prediction files
        output_folder: Output directory for plots

    Returns:
        True if successful, False otherwise
    """
    # Extract model names (excluding "Simulation" / ground truth)
    model_names = [name.lower().replace("-", "_").replace("+", "_").replace(" ", "_")
                   for name in models.keys() if name.lower() != "simulation"]

    if not model_names:
        print(f"  No models to analyze for spectral analysis")
        return False

    print(f"\n📊 Generating spectral analysis for {group_name}...")
    print(f"  Models: {', '.join(model_names)}")

    # Build command
    spectral_script = Path("src/analysis/plot_spectral_analysis.py")
    if not spectral_script.exists():
        print(f"  ⚠ Spectral analysis script not found: {spectral_script}")
        return False

    spectral_output = Path(output_folder) / group_name

    cmd = [
        sys.executable,
        str(spectral_script),
        "--progress-dir", str(Path(prediction_folder).parent),  # Parent of sampling/ dir
        "--models", ",".join(model_names),
        "--group-name", group_name,
        "--output-dir", str(spectral_output),
        "--skip-missing"  # Don't fail if some models are missing
    ]

    # Execute spectral analysis
    import subprocess
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  ✓ Spectral analysis complete")
        return True
    else:
        print(f"  ⚠ Spectral analysis encountered issues:")
        print(result.stderr)
        return False


def run_plot_group(
    group_name: str,
    dataset_name: str = "zInterp",
    field: str = "vort",
    output_folder: str = "results",
    prediction_folder: str = None
):
    """Generate and execute plotting script for a specific group."""
    if group_name not in MODEL_GROUPS:
        raise ValueError(f"Unknown group: {group_name}. Available: {list(MODEL_GROUPS.keys())}")

    group_info = MODEL_GROUPS[group_name]
    models = group_info["models"]
    description = group_info["description"]

    print(f"\n{'='*80}")
    print(f"Plotting Group: {group_name}")
    print(f"Description: {description}")
    print(f"Dataset: {dataset_name}")
    print(f"Models: {', '.join(models.keys())}")
    print(f"{'='*80}")

    # Create and execute plot script
    script_path = create_plot_script_for_group(
        group_name, models, dataset_name, field, output_folder, prediction_folder
    )

    print(f"Generated plot script: {script_path}")
    print(f"Executing...")

    import subprocess
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✓ Successfully generated plots for {group_name}")
        print(result.stdout)

        # Generate spectral analysis plots (energy spectrum + POD)
        spectral_success = generate_spectral_analysis(
            group_name=group_name,
            models=models,
            prediction_folder=prediction_folder or f"results/sampling/{dataset_name}",
            output_folder=output_folder
        )
        if not spectral_success:
            print(f"⚠ Spectral analysis skipped or failed for {group_name}")
    else:
        print(f"✗ Error generating plots for {group_name}")
        print(result.stderr)

    return result.returncode == 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Plot data in model comparison groups")
    parser.add_argument("--groups", nargs="+", default=["all"],
                       help=f"Groups to plot: {list(MODEL_GROUPS.keys())} or 'all'")
    parser.add_argument("--dataset", default="zInterp",
                       help="Dataset name (e.g., zInterp, extrap, interp)")
    parser.add_argument("--field", default="vort",
                       help="Field to plot (e.g., vort, pres, velX)")
    parser.add_argument("--output-folder", default="results",
                       help="Output folder for plots")
    parser.add_argument("--prediction-folder", default=None,
                       help="Folder containing prediction .npz files")

    args = parser.parse_args()

    # Determine which groups to plot
    if "all" in args.groups:
        groups_to_plot = list(MODEL_GROUPS.keys())
    else:
        groups_to_plot = args.groups

    print(f"\n{'='*80}")
    print(f"Plot Data Groups")
    print(f"{'='*80}")
    print(f"Groups to plot: {', '.join(groups_to_plot)}")
    print(f"Dataset: {args.dataset}")
    print(f"Field: {args.field}")

    # Run each group
    results = {}
    for group in groups_to_plot:
        success = run_plot_group(
            group,
            dataset_name=args.dataset,
            field=args.field,
            output_folder=args.output_folder,
            prediction_folder=args.prediction_folder
        )
        results[group] = success

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    successful = [g for g, s in results.items() if s]
    failed = [g for g, s in results.items() if not s]

    print(f"Successful: {len(successful)}/{len(results)}")
    if successful:
        print(f"  ✓ {', '.join(successful)}")
    if failed:
        print(f"Failed: {len(failed)}/{len(results)}")
        print(f"  ✗ {', '.join(failed)}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
