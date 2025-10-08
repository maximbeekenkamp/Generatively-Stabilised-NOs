# Quick Start: Flexible Loss Scheduler

This guide shows how to use the new flexible loss scheduler in both **local** and **Colab** environments.

## ✅ Backward Compatibility

**Your existing code still works!** The scheduler is optional and defaults to OFF.

```python
# Old code (still works - no scheduler)
p_l = LossParams(recMSE=1.0, predMSE=1.0)  # Auto-converts to recFieldError
trainer = Trainer(model, train_loader, optimizer, lr_scheduler, ...)

# New code (with scheduler)
p_l = LossParams(recFieldError=1.0, predFieldError=1.0, spectrumError=0.0)
p_s = SchedulerParams(enabled=True, ...)
loss_scheduler = FlexibleLossScheduler(p_s)
trainer = Trainer(..., loss_scheduler=loss_scheduler)
```

---

## 🖥️ Local Setup

### 1. Verify Installation

All files are already in your repository (no installation needed):

```bash
# Verify files exist
ls src/core/training/spectral_metrics.py           # ✓
ls src/core/training/flexible_loss_scheduler.py    # ✓
ls configs/examples/field_error_with_scheduler_example.py  # ✓
```

### 2. Run Tests (Optional)

```bash
# Test spectral metrics
PYTHONPATH=. python tests/test_spectral_metrics.py

# Test scheduler
PYTHONPATH=. python tests/test_flexible_loss_scheduler.py

# Run integration demo
PYTHONPATH=. python tests/integration_test_scheduler.py --epochs 30
```

### 3. Use in Your Training Script

```python
# Example: train_my_model.py
from configs.examples.field_error_with_scheduler_example import create_example_config
from src.core.training.flexible_loss_scheduler import FlexibleLossScheduler

# Create configuration (choose dataset: "iso", "tra", or "inc")
p_d, p_t, p_l, p_s, p_me, p_md, p_ml = create_example_config(dataset="tra")

# Create scheduler
loss_scheduler = FlexibleLossScheduler(p_s) if p_s.enabled else None

# Create trainer with scheduler
trainer = Trainer(
    model, train_loader, optimizer, lr_scheduler,
    criterion, train_history, writer, p_d, p_t,
    loss_scheduler=loss_scheduler  # NEW parameter
)

# Training loop
for epoch in range(p_t.epochs):
    trainer.trainingStep(epoch)
    tester.testStep(epoch)

    # Update scheduler after validation
    if loss_scheduler:
        val_metrics = test_history.get_validation_metrics()
        trainer.update_loss_scheduler(epoch, val_metrics)
```

### 4. Monitor with TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir=logs/

# Watch for scheduler adaptations
# Look for: LossScheduler/recFieldError, LossScheduler/spectrumError
```

---

## ☁️ Colab Setup

### 1. Add Scheduler Demo Cell

In your Colab notebook, add this cell **after environment setup** and **before training**:

```python
# =============================================================================
# NEW CELL: Flexible Loss Scheduler Setup
# =============================================================================

from src.core.training.flexible_loss_scheduler import FlexibleLossScheduler
from src.core.utils.params import LossParams, SchedulerParams

# Configuration for TRA (high spectral bias)
p_l = LossParams(
    recFieldError=1.0,
    predFieldError=1.0,
    spectrumError=0.0,     # Scheduler will increase this
    predLSIM=0.5           # Optional perceptual loss
)

p_s = SchedulerParams(
    enabled=True,
    source_loss="field_error",
    target_loss="spectrum_error",
    source_components={"recFieldError": True, "predFieldError": True},
    target_components={"spectrumError": True},
    static_components={"predLSIM": 0.5},  # Keep LSIM fixed

    # TRA-specific: Aggressive scheduling
    final_weight_source=0.2,
    final_weight_target=0.8,
    max_weight_target=0.9,
    patience=3,
    warmup_epochs=5,
)

# Create scheduler
loss_scheduler = FlexibleLossScheduler(p_s)
print(f"✅ Scheduler enabled: {p_s.source_loss} → {p_s.target_loss}")
```

### 2. Modify Training Cell

Update your existing training loop:

```python
# BEFORE (old code)
# trainer = Trainer(model, train_loader, optimizer, lr_scheduler,
#                   criterion, train_history, writer, p_d, p_t)

# AFTER (with scheduler)
trainer = Trainer(
    model, train_loader, optimizer, lr_scheduler,
    criterion, train_history, writer, p_d, p_t,
    loss_scheduler=loss_scheduler  # ← ADD THIS
)

# Training loop
for epoch in range(p_t.epochs):
    trainer.trainingStep(epoch)
    tester.testStep(epoch)

    # ← ADD THIS: Update scheduler
    if loss_scheduler:
        val_metrics = test_history.get_validation_metrics()
        trainer.update_loss_scheduler(epoch, val_metrics)
```

### 3. Pre-configured Cell (Copy-Paste Ready)

See `colab_verification/flexible_scheduler_demo_cell.py` for a complete, copy-paste ready cell with:
- TRA configuration (aggressive)
- ISO configuration (balanced)
- Full integration example

---

## 📊 Dataset-Specific Recommendations

Different datasets have different spectral characteristics:

### TRA (Transitional Flow) - High Spectral Bias
```python
p_s = SchedulerParams(
    enabled=True,
    final_weight_source=0.2,     # Only 20% field at end
    final_weight_target=0.8,     # 80% spectrum (aggressive)
    patience=3,                  # Fast adaptation
)
```

### ISO (Isotropic Turbulence) - Moderate Bias
```python
p_s = SchedulerParams(
    enabled=True,
    final_weight_source=0.3,     # 30% field (conservative)
    final_weight_target=0.7,     # 70% spectrum (balanced)
    patience=5,                  # Moderate adaptation
)
```

### INC (Incompressible Flow) - Low Bias
```python
p_s = SchedulerParams(
    enabled=True,
    final_weight_source=0.5,     # 50% field (gentle)
    final_weight_target=0.5,     # 50% spectrum (gentle)
    patience=8,                  # Slow adaptation
)
```

---

## 🔍 Monitoring & Debugging

### Console Output

When scheduler adapts, you'll see:

```
[Loss Scheduler] Epoch 42: Adapted loss weights (plateau in spectrum_error)
  Current progress: field_error 0.60 → spectrum_error 0.40
```

### TensorBoard Metrics

Watch these metrics in TensorBoard:

- `LossScheduler/recFieldError` - Field error weight over time
- `LossScheduler/spectrumError` - Spectrum error weight over time
- `val/epoch_lossSpectrumError` - Validation spectrum error (triggers adaptation)
- `val/epoch_lossRecFieldError` - Validation field error

### Validation Metrics

Check validation history for scheduler inputs:

```python
# After validation
val_metrics = test_history.get_validation_metrics()
print(val_metrics)
# {'field_error': 0.1234, 'spectrum_error': 0.0567, 'lsim': 0.0890}
```

---

## 🛠️ Troubleshooting

### Issue: "Module not found: flexible_loss_scheduler"

**Solution:** Make sure you're running from project root:

```bash
# Local
cd /path/to/Generatively-Stabilised-NOs
PYTHONPATH=. python your_script.py

# Colab
%cd /content/Generatively-Stabilised-NOs
sys.path.insert(0, '/content/Generatively-Stabilised-NOs')
```

### Issue: Scheduler never adapts

**Cause:** Validation metric not improving (no plateau detected)

**Solution:**
```python
p_s.patience = 3           # Reduce patience
p_s.warmup_epochs = 5      # Ensure warmup has passed
p_s.use_ema = True         # Use EMA smoothing
```

### Issue: Training unstable after adaptation

**Cause:** Too aggressive transition

**Solution:**
```python
p_s.max_weight_target = 0.6              # Lower safety cap
p_s.max_consecutive_adaptations = 5      # Stop earlier
p_s.adaptive_step = True                 # Use adaptive step sizing
```

### Issue: Legacy configs break

**Cause:** Using old `recMSE` parameter

**Solution:** Auto-conversion works, but update for clarity:
```python
# Old (deprecated, but still works)
p_l = LossParams(recMSE=1.0, predMSE=1.0)

# New (recommended)
p_l = LossParams(recFieldError=1.0, predFieldError=1.0)
```

---

## 📚 Additional Resources

- **Complete Examples:** `configs/examples/field_error_with_scheduler_example.py`
- **Documentation:** `configs/examples/README.md`
- **Unit Tests:** `tests/test_spectral_metrics.py`, `tests/test_flexible_loss_scheduler.py`
- **Integration Test:** `tests/integration_test_scheduler.py`
- **Colab Demo Cell:** `colab_verification/flexible_scheduler_demo_cell.py`

---

## 🎯 Quick Reference

| Component | Path | Purpose |
|-----------|------|---------|
| `spectral_metrics.py` | `src/core/training/` | Field error + spectrum error losses |
| `flexible_loss_scheduler.py` | `src/core/training/` | Adaptive scheduler with plateau detection |
| `params.py` | `src/core/utils/` | LossParams + SchedulerParams configuration |
| `loss.py` | `src/core/training/` | Integrated loss with weight management |
| `trainer.py` | `src/core/training/` | Scheduler integration + validation |
| `loss_history.py` | `src/core/training/` | Enhanced logging for new metrics |

---

**Need Help?** See `configs/examples/README.md` for comprehensive documentation and troubleshooting.
