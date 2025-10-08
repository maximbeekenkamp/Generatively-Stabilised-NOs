# Example Configurations: Field Error & Flexible Loss Scheduler

This directory contains example configurations demonstrating the new field error loss architecture and flexible loss scheduler for adaptive spectral bias mitigation.

## Quick Start

```python
from configs.examples.field_error_with_scheduler_example import create_example_config

# Create configuration
p_d, p_t, p_l, p_s, p_me, p_md, p_ml = create_example_config(dataset="iso")

# Use in training
from src.core.training.flexible_loss_scheduler import FlexibleLossScheduler
scheduler = FlexibleLossScheduler.from_params(p_s)
```

## What's New?

### 1. **Field Error Loss** (Replaces MSE)

Per-frame relative MSE - more stable for turbulence with varying magnitudes:

```python
p_l = LossParams(
    recFieldError=1.0,      # Reconstruction: per-frame relative MSE
    predFieldError=1.0,     # Prediction: per-frame relative MSE
    spectrumError=0.0,      # Spectrum error (for high-freq details)
)
```

**Formula:**
```
field_error = mean(mean((pred-true)², H,W) / mean(true², H,W))
```

### 2. **Spectrum Error Loss** (Optional)

Relative MSE in log power spectrum space - captures high-frequency turbulence details:

```python
p_l = LossParams(
    spectrumError=0.5  # Weight for spectrum error
)
```

**Formula:**
```
spectrum_error = mean((log(P_pred) - log(P_true))² / log(P_true)²)
where P = FFT → radial binning → scale by annular area
```

### 3. **Flexible Loss Scheduler** (Adaptive)

Smoothly transitions between loss formulations when validation plateaus:

```python
p_s = SchedulerParams(
    enabled=True,

    # What to transition
    source_loss="field_error",
    target_loss="spectrum_error",

    # Weight progression (conservative)
    final_weight_source=0.3,    # Keep 30% field error
    final_weight_target=0.7,    # Max 70% spectrum error
    max_weight_target=0.8,      # Safety cap

    # When to adapt
    monitor_metric="spectrum_error",
    patience=5,                 # Wait 5 epochs before adapting
    warmup_epochs=5,            # No adaptation in first 5 epochs
)
```

## Example Configurations

### Example 1: Field Error → Spectrum Error (Default)

**Use case:** Mitigate spectral bias in turbulence forecasting

```python
from configs.examples.field_error_with_scheduler_example import create_example_config

p_d, p_t, p_l, p_s, ... = create_example_config()
```

**Training progression:**
1. **Epochs 1-5:** Warmup, 100% field error
2. **Epochs 6+:** Adapt when spectrum_error plateaus
3. **Final:** 30% field error + 70% spectrum error (learns fine structures)

### Example 2: LSIM → Field Error

**Use case:** Start with perceptual quality, refine physics accuracy

```python
from configs.examples.field_error_with_scheduler_example import create_lsim_to_field_error_config

p_d, p_t, p_l, p_s, ... = create_lsim_to_field_error_config()
```

**Training progression:**
1. **Early:** 100% LSIM (perceptual similarity)
2. **Late:** Transition to field error (physics accuracy)

### Example 3: Static 50/50 Mix (No Scheduler)

**Use case:** Fixed spectral bias mitigation without adaptation

```python
from configs.examples.field_error_with_scheduler_example import create_static_spectrum_config

p_d, p_t, p_l, p_s, ... = create_static_spectrum_config()
```

## Dataset-Specific Recommendations

Different datasets have different spectral characteristics:

```python
from configs.examples.field_error_with_scheduler_example import get_recommended_config_for_dataset

# Isotropic turbulence: Balanced scheduling
p_d, p_t, p_l, p_s, ... = get_recommended_config_for_dataset("iso")

# Transitional flow: Aggressive scheduling (high spectral bias)
p_d, p_t, p_l, p_s, ... = get_recommended_config_for_dataset("tra")

# Incompressible flow: Gentle scheduling (low spectral bias)
p_d, p_t, p_l, p_s, ... = get_recommended_config_for_dataset("inc")
```

**Recommendations:**
- **TRA:** `final_weight_target=0.8`, `patience=3` (aggressive)
- **ISO:** `final_weight_target=0.7`, `patience=5` (balanced)
- **INC:** `final_weight_target=0.5`, `patience=8` (gentle)

## Integration with Training Loop

Update your training script to use the scheduler:

```python
from src.core.training.flexible_loss_scheduler import FlexibleLossScheduler
from configs.examples.field_error_with_scheduler_example import create_example_config

# 1. Create configuration
p_d, p_t, p_l, p_s, p_me, p_md, p_ml = create_example_config()

# 2. Create scheduler (if enabled)
loss_scheduler = None
if p_s.enabled:
    loss_scheduler = FlexibleLossScheduler.from_params(p_s)

# 3. Create trainer with scheduler
trainer = Trainer(
    model, trainLoader, optimizer, lrScheduler,
    criterion, trainHistory, writer, p_d, p_t,
    loss_scheduler=loss_scheduler  # NEW parameter
)

# 4. Training loop
for epoch in range(p_t.epochs):
    trainer.trainingStep(epoch)
    tester.testStep(epoch)

    # Update scheduler with validation metrics (after validation)
    if loss_scheduler is not None:
        validation_metrics = testHistory.get_validation_metrics()
        trainer.update_loss_scheduler(epoch, validation_metrics)
```

## Monitoring & Debugging

### TensorBoard Logs

The scheduler automatically logs to TensorBoard:

```bash
tensorboard --logdir=logs/
```

**Metrics to watch:**
- `LossScheduler/recFieldError` - Field error weight over time
- `LossScheduler/spectrumError` - Spectrum error weight over time
- `val/epoch_lossSpectrumError` - Validation spectrum error (triggers adaptation)

### Console Output

When scheduler adapts, you'll see:

```
[Loss Scheduler] Epoch 42: Adapted loss weights (plateau in spectrum_error)
  Current progress: field_error 0.60 → spectrum_error 0.40
```

## Troubleshooting

### Issue: Scheduler never adapts

**Cause:** Validation metric not improving (no plateau detected)

**Solution:** Check patience and warmup settings:
```python
p_s.patience = 3           # Reduce patience
p_s.warmup_epochs = 5      # Ensure warmup passed
```

### Issue: Training unstable after adaptation

**Cause:** Too aggressive transition

**Solution:** Increase safety caps:
```python
p_s.max_weight_target = 0.6              # Lower cap
p_s.max_consecutive_adaptations = 5      # Stop earlier
```

### Issue: Using old MSE parameters

**Cause:** Trying to use deprecated `recMSE` or `predMSE` parameters

**Solution:** Use the current field error parameters:
```python
# Correct usage
p_l = LossParams(recFieldError=1.0, predFieldError=1.0)
```

Note: `recMSE` and `predMSE` parameters have been removed. Update any legacy configs to use `recFieldError` and `predFieldError`.

## References

- **Field Error:** `src/core/training/spectral_metrics.py::compute_field_error_loss()`
- **Spectrum Error:** `src/core/training/spectral_metrics.py::compute_spectrum_error_loss()`
- **Scheduler:** `src/core/training/flexible_loss_scheduler.py`
- **Loss:** `src/core/training/loss.py`

---

*For more details, see the implementation plan in the project documentation.*
