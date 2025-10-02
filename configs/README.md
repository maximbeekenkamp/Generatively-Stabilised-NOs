# Generative Operator Configuration System

This directory contains a comprehensive configuration system for training NO+DM (Neural Operator + Diffusion Model) generative operators across different datasets and architectures.

## Directory Structure

```
configs/
├── README.md                          # This file
├── templates/                         # Configuration templates
│   ├── base_config_template.json     # Base template for all configurations
│   ├── fno_diffusion_optimized.json  # FNO + Diffusion optimized settings
│   ├── tno_diffusion_optimized.json  # TNO + Diffusion optimized settings
│   └── unet_diffusion_optimized.json # U-Net + Diffusion optimized settings
├── inc/                              # Incompressible flow configurations
│   ├── genop_fno_diffusion_inc.json
│   ├── genop_tno_diffusion_inc.json
│   └── genop_unet_diffusion_inc.json
├── tra/                              # Transonic flow configurations
│   ├── genop_fno_diffusion_tra.json
│   ├── genop_tno_diffusion_tra.json
│   └── genop_unet_diffusion_tra.json
├── iso/                              # Isotropic turbulence configurations
│   ├── genop_fno_diffusion_iso.json
│   ├── genop_tno_diffusion_iso.json
│   └── genop_unet_diffusion_iso.json
└── generated/                        # Auto-generated configurations
    └── comparison/                   # Standardized configs for comparison
```

## Configuration Templates

### Base Template
The `base_config_template.json` provides the foundation for all configurations with:
- Common training parameters
- Standard hardware settings
- Default optimization parameters
- Template variables for customization

### Model-Specific Templates
Each model type has an optimized template:

#### FNO + Diffusion (`fno_diffusion_optimized.json`)
- **Strengths**: High-frequency resolution, spectral domain processing
- **Best for**: Periodic boundary conditions, smooth dynamics
- **Optimizations**: Spectral normalization, frequency-domain losses

#### TNO + Diffusion (`tno_diffusion_optimized.json`)
- **Strengths**: Long-sequence modeling, attention mechanisms
- **Best for**: Complex temporal dependencies, variable-length sequences
- **Optimizations**: Attention optimization, gradient checkpointing

#### U-Net + Diffusion (`unet_diffusion_optimized.json`)
- **Strengths**: Multi-scale spatial processing, local feature preservation
- **Best for**: Spatial detail preservation, skip connections
- **Optimizations**: Multi-scale training, feature map optimization

## Dataset-Specific Optimizations

### Incompressible Flows (Inc)
- **Resolution**: 64×64 or 128×64
- **Channels**: 3 (velocity components + pressure)
- **Sequence Length**: 10 frames
- **Augmentation**: Standard (flips, rotations, noise)
- **Special**: Physics-informed losses for conservation

### Transonic Flows (Tra)
- **Resolution**: 128×128 (higher for shock resolution)
- **Channels**: 4 (density, momentum, energy, pressure)
- **Sequence Length**: 8 frames (computationally intensive)
- **Augmentation**: Conservative (shock preservation)
- **Special**: Shock detection, gradient clipping for stability

### Isotropic Turbulence (Iso)
- **Resolution**: 64×64
- **Channels**: 3 (3D velocity components)
- **Sequence Length**: 12 frames (turbulence evolution)
- **Augmentation**: Aggressive (turbulence robustness)
- **Special**: Energy spectrum preservation, Kolmogorov scaling

## Usage Guide

### Quick Start
Use pre-configured settings for standard training:

```bash
# Train FNO + Diffusion on incompressible flows
python scripts/train_generative_operator_inc.py --config configs/inc/genop_fno_diffusion_inc.json

# Train TNO + Diffusion on transonic flows
python scripts/train_generative_operator_tra.py --config configs/tra/genop_tno_diffusion_tra.json

# Train U-Net + Diffusion on isotropic turbulence
python scripts/train_generative_operator_iso.py --config configs/iso/genop_unet_diffusion_iso.json
```

### Generate Custom Configurations
Use the configuration generator for custom settings:

```bash
# Generate all possible combinations
python scripts/generate_configs.py --all

# Generate specific model-dataset combination
python scripts/generate_configs.py --model fno --dataset inc

# Generate multiple combinations
python scripts/generate_configs.py --combinations fno-inc,tno-tra,unet-iso

# Generate standardized comparison configurations
python scripts/generate_configs.py --comparison

# Generate training scripts automatically
python scripts/generate_configs.py --all --scripts
```

## Configuration Parameters

### Core Model Parameters
```json
{
    "model_type": "genop-fno-diffusion",        # Architecture type
    "correction_strength": 1.0,                 # Diffusion correction strength
    "batch_size": 4,                           # Training batch size
    "input_length": 10,                        # Input sequence length
    "output_length": 10                        # Output sequence length
}
```

### Training Schedule
```json
{
    "stage1_epochs": 100,                      # Prior training epochs
    "stage2_epochs": 50,                       # Corrector training epochs
    "stage1_lr": 0.001,                        # Prior learning rate
    "stage2_lr": 0.0005,                       # Corrector learning rate
    "enable_joint_finetuning": true,           # Enable stage 3
    "joint_finetuning_epochs": 20              # Joint fine-tuning epochs
}
```

### Loss Configuration
```json
{
    "use_l1_loss": true,                       # Enable L1 loss component
    "l1_loss_weight": 0.1,                     # L1 loss weight
    "use_perceptual_loss": false,              # Enable perceptual loss
    "perceptual_loss_weight": 0.1              # Perceptual loss weight
}
```

### Memory Optimization
```json
{
    "enable_gradient_checkpointing": true,     # Memory-compute tradeoff
    "use_mixed_precision": true,               # FP16 training
    "adaptive_batch_sizing": true              # Dynamic batch size adjustment
}
```

## Model Comparison Guidelines

For fair model comparison, use configurations from `configs/generated/comparison/`:
- Standardized training parameters
- Identical batch sizes and learning rates
- Same early stopping criteria
- Consistent augmentation strategies

## Performance Tuning

### Memory Optimization
1. **Gradient Checkpointing**: Trade compute for memory
2. **Mixed Precision**: Reduce memory usage by ~50%
3. **Adaptive Batch Sizing**: Automatically adjust for available memory

### Training Speed
1. **Batch Size**: Larger batches for better GPU utilization
2. **Data Loading**: More workers and prefetching
3. **Model Architecture**: U-Net typically fastest, TNO slowest

### Quality Optimization
1. **Loss Functions**: Task-specific loss combinations
2. **Augmentation**: Dataset-appropriate strategies
3. **Two-Stage Training**: Prior → Corrector → Joint fine-tuning

## Troubleshooting

### Common Issues

#### Out of Memory (OOM)
- Reduce `batch_size`
- Enable `gradient_checkpointing`
- Use `mixed_precision`
- Reduce `input_length`

#### Poor Convergence
- Check `learning_rate` values
- Verify `loss_weights` are balanced
- Ensure appropriate `augmentation_strategy`
- Consider longer `warmup_epochs` for TNO

#### Slow Training
- Increase `batch_size` if memory allows
- Use more `num_workers` for data loading
- Enable `pin_memory` and `prefetch_factor`
- Consider `mixed_precision` training

### Hardware Recommendations

#### Minimum Requirements
- **GPU**: 8GB VRAM (RTX 3070, V100)
- **RAM**: 16GB system memory
- **Storage**: 100GB for datasets + checkpoints

#### Recommended Configuration
- **GPU**: 24GB VRAM (RTX 3090, A100)
- **RAM**: 64GB system memory
- **Storage**: 500GB NVMe SSD

#### High-Performance Setup
- **GPU**: 40GB+ VRAM (A100, H100)
- **RAM**: 128GB system memory
- **Storage**: 1TB+ NVMe SSD with high IOPS

## Best Practices

### Configuration Management
1. Always version control your configurations
2. Use meaningful experiment names
3. Document parameter changes and their rationale
4. Save configurations alongside checkpoints

### Training Workflow
1. Start with pre-configured settings
2. Monitor training curves closely
3. Use early stopping to prevent overfitting
4. Save intermediate checkpoints frequently

### Evaluation Strategy
1. Use consistent evaluation metrics
2. Test on multiple validation sets
3. Compare against baseline models
4. Visualize predictions for qualitative assessment

## Extension Guide

### Adding New Datasets
1. Create dataset-specific optimizations in templates
2. Add data loading logic in `generative_operator_loader.py`
3. Create configuration files following naming convention
4. Update the configuration generator

### Adding New Model Types
1. Create model-specific optimization template
2. Add model creation logic in model registry
3. Update configuration generator with new model type
4. Create training script template

### Custom Loss Functions
1. Implement loss in `GenerativeOperatorLoss` class
2. Add configuration parameters for loss weights
3. Update templates with new loss options
4. Test with ablation studies

For more detailed information, see the individual README files in each subdirectory.