# Scripts Directory

This directory contains executable scripts for model testing, data generation, and visualization.

## Available Scripts

### `run_all_models_generate_npz.py`
**Purpose:** Run all Neural Operators (FNO, TNO, UNet, DeepONet) and NO+DM variants on dummy data to generate NPZ files for plotting.

**Usage:**
```bash
python scripts/run_all_models_generate_npz.py
```

**Output:**
- Generates `.npz` files in `tests/outputs/model_predictions/`
- Format: `[num_models, num_evals, num_sequences, timesteps, channels, H, W]` (7D tensors)
- Creates metadata files with model names and configuration

**Models Tested:**
- Standalone: FNO, TNO, UNet, DeepONet
- NO+DM variants: FNO+DM, TNO+DM, UNet+DM, DeepONet+DM

**Datasets:** inc_low, tra_ext, iso

---

### `run_full_visual_demo.py`
**Purpose:** Generate comprehensive visualizations demonstrating all model capabilities.

**Usage:**
```bash
python scripts/run_full_visual_demo.py
```

**Output:**
- Creates visualization outputs in `tests/outputs/comprehensive_demo/`
- Generates comparison plots across all models
- Creates statistical analysis plots
- Produces publication-quality figures

**Features:**
- Model comparison visualizations
- Temporal evolution plots
- Statistical metrics (MSE, spectral analysis)
- Error distribution plots

---

## Requirements

Both scripts require:
- PyTorch with CUDA support (recommended)
- All dependencies from `requirements.txt`
- Sufficient memory (recommend 16GB+ RAM, 8GB+ VRAM)

## Environment Setup

```bash
# From repository root
export PYTHONPATH=.
python scripts/<script_name>.py
```

## Troubleshooting

**Issue:** `ModuleNotFoundError: No module named 'src'`
**Solution:** Ensure `PYTHONPATH=.` is set and you're running from repository root

**Issue:** CUDA out of memory
**Solution:** Reduce batch size or use smaller spatial resolution in the script configuration
