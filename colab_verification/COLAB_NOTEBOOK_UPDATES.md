# Colab Notebook Update Instructions

This document describes the cells to add to `colab_tra_only_verification.ipynb` for:
1. Centralized seed management
2. Spectral visualization (energy spectrum + POD analysis)

## 1. Add Seed Initialization Cell

**Location:** After the imports cell (cell 3), before any model operations

**Cell Type:** Code

**Content:**
```python
# ============================================================================
# REPRODUCIBILITY - Centralized Seed Management
# ============================================================================

from src.core.utils.reproducibility import set_global_seed

# Initialize all random seeds for reproducibility
set_global_seed(42)
print("✓ Random seeds initialized (seed=42)")
print("  - Python random, NumPy, PyTorch (CPU/GPU)")
print("  - cuDNN deterministic mode enabled")
```

---

## 2. Add Spectral Visualization Section

**Location:** In the visualization phase, after the existing plotting cells

**Cell Type:** Markdown

**Content:**
```markdown
## Spectral Analysis

Generate energy spectrum E(k) and POD analysis for model groups:
- **Energy Spectrum**: Validates frequency content preservation
- **POD Analysis**: Checks eigenvalue decay and mode structure
```

---

**Cell Type:** Code

**Content:**
```python
# ============================================================================
# SPECTRAL VISUALIZATION - Energy Spectrum & POD
# ============================================================================

from src.analysis.spectral_visualization import (
    plot_energy_spectrum_comparison,
    compute_pod_analysis,
    plot_pod_comparison
)
import pickle
from IPython.display import Image, display

# Define model groups for spectral analysis
model_groups = {
    'Neural Operators': ['fno', 'tno', 'unet', 'deeponet'],
    'NO+DM': ['fno_dm', 'tno_dm', 'unet_dm', 'deeponet_dm']
}

# Helper function to load prediction data
def load_prediction_dict(path):
    """Load prediction from .dict file"""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and 'predictions' in data:
        data = data['predictions']
    return torch.from_numpy(data) if isinstance(data, np.ndarray) else data

# Load ground truth (once)
print("Loading ground truth...")
gt_path = progress_dir / 'sampling/groundTruth.dict'
ground_truth = load_prediction_dict(gt_path)
print(f"  Shape: {ground_truth.shape}")
print()

# Generate spectral analysis for each model group
for group_name, model_list in model_groups.items():
    print("=" * 80)
    print(f"Spectral Analysis: {group_name}")
    print("=" * 80)

    # Load predictions for this group
    predictions = {}
    for model in model_list:
        pred_path = progress_dir / f'sampling/{model}.dict'
        if pred_path.exists():
            try:
                predictions[model] = load_prediction_dict(pred_path)
                print(f"✓ Loaded {model}: {predictions[model].shape}")
            except Exception as e:
                print(f"⚠ Error loading {model}: {e}")
        else:
            print(f"⚠ {model} predictions not found")

    if not predictions:
        print(f"⚠ No predictions found for {group_name}, skipping")
        continue

    print()

    # --------------------------------------------------------------------
    # Energy Spectrum E(k)
    # --------------------------------------------------------------------
    print(f"📊 Generating energy spectrum plot for {group_name}...")

    spectrum_path = progress_dir / f'plots/energy_spectrum_{group_name.lower().replace(" ", "_")}.png'
    spectrum_path.parent.mkdir(parents=True, exist_ok=True)

    spectra = plot_energy_spectrum_comparison(
        ground_truth=ground_truth,
        predictions=predictions,
        save_path=str(spectrum_path),
        wavenumber_range=(1, 64),
        model_names=list(predictions.keys()),
        title=f'Energy Spectrum - {group_name}',
        kolmogorov_slope=True
    )

    # Display inline
    display(Image(filename=str(spectrum_path)))

    # Print spectrum errors
    k_gt, E_gt = spectra['ground_truth']
    print()
    print("Spectrum Errors (relative MSE in log space):")
    for model, (k, E) in spectra.items():
        if model != 'ground_truth':
            log_error = np.mean((np.log(E + 1e-10) - np.log(E_gt + 1e-10))**2)
            print(f"  {model:15s}: {log_error:.6f}")
    print()

    # --------------------------------------------------------------------
    # POD Analysis
    # --------------------------------------------------------------------
    print(f"📊 Generating POD analysis for {group_name}...")

    # Compute POD for ground truth
    gt_pod = compute_pod_analysis(ground_truth, ground_truth, n_modes=10)

    # Compute POD for predictions
    predictions_pod = {}
    for model, pred in predictions.items():
        try:
            predictions_pod[model] = compute_pod_analysis(ground_truth, pred, n_modes=10)
        except Exception as e:
            print(f"  ⚠ Error computing POD for {model}: {e}")

    # Generate plot
    pod_path = progress_dir / f'plots/pod_analysis_{group_name.lower().replace(" ", "_")}.png'

    plot_pod_comparison(
        ground_truth_pod=gt_pod,
        predictions_pod=predictions_pod,
        save_path=str(pod_path),
        model_names=list(predictions_pod.keys()),
        title=f'POD Analysis - {group_name}'
    )

    # Display inline
    display(Image(filename=str(pod_path)))

    # Print POD statistics
    print()
    print("POD Statistics (10 modes):")
    print(f"  Ground Truth:")
    print(f"    Cumulative energy: {gt_pod['cumulative_energy'][-1]*100:.2f}%")
    print()
    for model, pod in predictions_pod.items():
        print(f"  {model}:")
        print(f"    Cumulative energy: {pod['cumulative_energy'][-1]*100:.2f}%")
        print(f"    Prediction error: {pod['prediction_error']:.6f}")
    print()

print("=" * 80)
print("✓ Spectral analysis complete for all groups")
print("=" * 80)
```

---

## Instructions to Add Cells

1. Open `colab_tra_only_verification.ipynb` in Jupyter or Colab
2. Find the imports cell (approximately cell 3)
3. **Insert new code cell after imports** - paste the seed initialization code
4. Find the visualization section (after model training/sampling)
5. **Insert markdown cell** - paste the "Spectral Analysis" header
6. **Insert code cell** - paste the spectral visualization code
7. Save the notebook
8. Run all cells to verify functionality

---

## Expected Output

After adding these cells and running the notebook, you should see:

1. **Seed Initialization:**
   ```
   ✓ Random seeds initialized (seed=42)
     - Python random, NumPy, PyTorch (CPU/GPU)
     - cuDNN deterministic mode enabled
   ```

2. **Spectral Analysis:**
   - Energy spectrum plots for each model group (log-log scale)
   - POD eigenvalue decay and mode shapes
   - Quantitative error metrics (spectrum MSE, POD error)
   - Inline plot display in notebook

3. **Generated Files:**
   - `progress_dir/plots/energy_spectrum_neural_operators.png`
   - `progress_dir/plots/energy_spectrum_no_dm.png`
   - `progress_dir/plots/pod_analysis_neural_operators.png`
   - `progress_dir/plots/pod_analysis_no_dm.png`

---

## Verification

To verify the updates work correctly:

1. Run seed initialization cell → should print confirmation
2. Train/sample a few models (e.g., FNO, FNO+DM)
3. Run spectral analysis cell → should generate plots
4. Check that plots display inline and files are saved

If you encounter issues:
- Ensure `src.analysis.spectral_visualization` module is importable
- Check that prediction `.dict` files exist in `progress_dir/sampling/`
- Verify ground truth is loaded correctly (should be 5D tensor [B, T, C, H, W])
