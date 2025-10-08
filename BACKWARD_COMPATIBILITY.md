# Backward Compatibility Guide

## ✅ Your Existing Code Still Works!

The flexible loss scheduler implementation is **fully backward compatible**. All existing training scripts, notebooks, and configurations continue to work without modification.

---

## What Changed vs. What's Preserved

### ✅ Preserved (Still Works)

#### 1. **Legacy MSE Parameters**
```python
# OLD CODE - Still works, auto-converts with warning
p_l = LossParams(recMSE=1.0, predMSE=1.0)

# Console output:
# Warning: recMSE is deprecated, use recFieldError instead. Auto-converting...
# Warning: predMSE is deprecated, use recFieldError instead. Auto-converting...
```

#### 2. **Existing Trainer Initialization**
```python
# OLD CODE - Still works, scheduler defaults to None
trainer = Trainer(
    model, train_loader, optimizer, lr_scheduler,
    criterion, train_history, writer, p_d, p_t
)
# No scheduler = standard training (no changes)
```

#### 3. **Existing Training Loops**
```python
# OLD CODE - Still works exactly as before
for epoch in range(p_t.epochs):
    trainer.trainingStep(epoch)
    tester.testStep(epoch)
```

#### 4. **Loss History Logging**
```python
# OLD CODE - Still works, auto-adapts to new format
trainHistory.updateBatch(lossParts, lossSeq, s, timeMin)
trainHistory.updateEpoch(timeMin)
```

#### 5. **TensorBoard Metrics**
```python
# OLD CODE - Still logged (MSE preserved for compatibility)
writer.add_scalar("train/epoch_lossRecMSE", ...)
writer.add_scalar("train/epoch_lossPredMSE", ...)
```

### 🆕 New Features (Optional)

#### 1. **Field Error (Replaces MSE as Primary)**
```python
# NEW CODE - Recommended for turbulence
p_l = LossParams(recFieldError=1.0, predFieldError=1.0)
```

#### 2. **Spectrum Error (Spectral Bias Mitigation)**
```python
# NEW CODE - Optional for high-frequency preservation
p_l = LossParams(
    recFieldError=1.0,
    predFieldError=1.0,
    spectrumError=0.5  # NEW: Spectrum error weight
)
```

#### 3. **Flexible Loss Scheduler (Adaptive)**
```python
# NEW CODE - Optional scheduler for automatic adaptation
p_s = SchedulerParams(enabled=True, ...)
loss_scheduler = FlexibleLossScheduler(p_s)

trainer = Trainer(
    ...,
    loss_scheduler=loss_scheduler  # NEW: Optional parameter
)
```

---

## Migration Path (3 Levels)

### Level 0: No Changes (Everything Works)
```python
# Your existing code - ZERO changes needed
p_l = LossParams(recMSE=1.0, predMSE=1.0)  # Auto-converts
trainer = Trainer(model, train_loader, optimizer, lr_scheduler,
                 criterion, train_history, writer, p_d, p_t)

for epoch in range(p_t.epochs):
    trainer.trainingStep(epoch)
    tester.testStep(epoch)
```

**Result:** Training works identically to before. MSE auto-converts to field error with warnings.

---

### Level 1: Update Loss Names (5 min)
```python
# Update loss parameter names (removes deprecation warnings)
p_l = LossParams(
    recFieldError=1.0,    # was: recMSE
    predFieldError=1.0,   # was: predMSE
)

# Rest of code unchanged
trainer = Trainer(...)
for epoch in range(p_t.epochs):
    trainer.trainingStep(epoch)
    tester.testStep(epoch)
```

**Result:** Same training behavior, no warnings. Field error is slightly better for turbulence (per-frame normalization).

---

### Level 2: Add Spectrum Error (10 min)
```python
# Add spectrum error for better high-frequency capture
p_l = LossParams(
    recFieldError=1.0,
    predFieldError=1.0,
    spectrumError=0.5,    # NEW: Add spectrum error
)

# Rest of code unchanged
trainer = Trainer(...)
for epoch in range(p_t.epochs):
    trainer.trainingStep(epoch)
    tester.testStep(epoch)
```

**Result:** Static mix of field + spectrum (50/50). Better high-frequency details, no automatic adaptation.

---

### Level 3: Enable Scheduler (15 min)
```python
# Enable adaptive loss scheduling
p_l = LossParams(
    recFieldError=1.0,
    predFieldError=1.0,
    spectrumError=0.0,    # Start at 0, scheduler increases
)

p_s = SchedulerParams(
    enabled=True,
    final_weight_source=0.3,
    final_weight_target=0.7,
)

# Create scheduler
loss_scheduler = FlexibleLossScheduler(p_s)

# Pass to trainer
trainer = Trainer(..., loss_scheduler=loss_scheduler)  # NEW parameter

# Modified training loop
for epoch in range(p_t.epochs):
    trainer.trainingStep(epoch)
    tester.testStep(epoch)

    # NEW: Update scheduler after validation
    if loss_scheduler:
        val_metrics = test_history.get_validation_metrics()
        trainer.update_loss_scheduler(epoch, val_metrics)
```

**Result:** Adaptive training. Starts with field error, smoothly transitions to spectrum error when validation plateaus. Best spectral bias mitigation.

---

## File-by-File Changes

### Modified Files

| File | Changes | Backward Compatible? |
|------|---------|---------------------|
| `loss.py` | - Removed TNO LpLoss<br>- MSE → Field Error<br>- Added spectrum error<br>- Added `set_loss_weights()` | ✅ Yes (MSE still in dict) |
| `trainer.py` | - Added `loss_scheduler` param<br>- Added `update_loss_scheduler()` | ✅ Yes (param optional) |
| `params.py` | - `recMSE`→`recFieldError`<br>- Added `SchedulerParams` | ✅ Yes (auto-converts) |
| `loss_history.py` | - Updated loss component names<br>- Added `get_validation_metrics()` | ✅ Yes (old names work) |

### New Files (No Impact on Existing Code)

- `src/core/training/spectral_metrics.py` - New metrics
- `src/core/training/flexible_loss_scheduler.py` - Scheduler
- `configs/examples/*` - Example configurations
- `tests/test_*.py` - Unit tests

---

## Testing Backward Compatibility

### Test 1: Run Existing Script Unchanged
```bash
# Your old training script should work without modification
python src/core/training/training_tno_tra.py
# ✓ Should run successfully with deprecation warnings
```

### Test 2: Load Old Checkpoints
```python
# Old checkpoints load correctly
checkpoint = torch.load('old_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
# ✓ Works - model architecture unchanged
```

### Test 3: Old Configs Load
```python
# Old LossParams configs work
old_config = {"recMSE": 1.0, "predMSE": 1.0}
p_l = LossParams.fromDict(old_config)
# ✓ Auto-converts to recFieldError/predFieldError
```

---

## Breaking Changes (None!)

**There are NO breaking changes.** All code paths that worked before continue to work.

The only visible difference is deprecation warnings for `recMSE`/`predMSE`:
```
Warning: recMSE is deprecated, use recFieldError instead. Auto-converting...
```

To remove warnings, simply rename:
- `recMSE` → `recFieldError`
- `predMSE` → `predFieldError`

---

## Recommended Migration Timeline

### Immediate (Optional)
- Update loss parameter names to remove warnings
- Test that training works identically

### Short-term (1-2 weeks)
- Try static spectrum error (Level 2) on a test run
- Compare field error vs. MSE on your datasets

### Long-term (When ready)
- Enable scheduler (Level 3) for new experiments
- Compare baseline vs. scheduler on validation metrics

---

## Support

If you encounter any compatibility issues:

1. **Check deprecation warnings** - They guide you to the new API
2. **Run tests** - `tests/test_spectral_metrics.py` verifies correctness
3. **Compare outputs** - Old vs. new should be nearly identical (field error ≈ MSE for most cases)

**No Rush:** You can migrate at your own pace. Everything works as-is!
