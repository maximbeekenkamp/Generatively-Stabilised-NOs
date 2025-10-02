# Manual Test Scripts

This directory contains manual integration test scripts for quick validation and development testing. These scripts are NOT part of the automated pytest test suite.

## Available Scripts

### `quick_test_deeponet.py`
**Purpose:** Quick standalone test for DeepONet integration (both NO and NO+DM variants).

**Usage:**
```bash
cd /path/to/repository
export PYTHONPATH=.
python tests/manual/quick_test_deeponet.py
```

**What it tests:**
- DeepONet standalone model creation and forward pass
- DeepONet+DM (with diffusion model) creation and forward pass
- Format compatibility ([B,T,C,H,W] tensors)
- Basic shape validation

**Expected output:**
```
Testing DeepONet Integration
Creating DeepONet model...
✓ DeepONet forward pass successful
✓ Output shape: torch.Size([2, 2, 3, 16, 16])

Testing DeepONet+DM...
✓ DeepONet+DM forward pass successful
✓ Output shape: torch.Size([2, 2, 3, 16, 16])
```

---

### `test_single_model.py`
**Purpose:** Test a single model configuration quickly without running the full suite.

**Usage:**
```bash
export PYTHONPATH=.
python tests/manual/test_single_model.py
```

**What it tests:**
- Single model instantiation
- Basic forward pass
- Output validation

**Customization:**
Edit the script to change:
- Model architecture (`arch` parameter)
- Data configuration
- Batch size and dimensions

---

## When to Use Manual Tests

Use these scripts when:
1. **Rapid development iteration** - Testing a specific model without full test suite overhead
2. **Debugging specific issues** - Isolating a particular model or configuration
3. **Quick validation** - Verifying a change didn't break basic functionality
4. **Development** - Testing new features before adding formal pytest tests

## When to Use Automated Tests Instead

Use `pytest tests/` when:
1. **Comprehensive validation** - Testing all models and configurations
2. **CI/CD** - Automated testing in continuous integration
3. **Regression testing** - Ensuring no functionality breaks
4. **Coverage analysis** - Measuring test coverage

## Differences from Integration Tests

| Manual Tests | Integration Tests (`tests/integration/`) |
|--------------|------------------------------------------|
| Quick, focused | Comprehensive, multi-component |
| Development/debugging | Validation and regression |
| Not automated | Run by pytest/CI |
| No coverage tracking | Coverage metrics |
| Ad-hoc execution | Structured test suites |

## Adding New Manual Tests

When creating new manual test scripts:

1. Add them to this directory
2. Use descriptive names (e.g., `quick_test_<feature>.py`)
3. Include docstrings explaining purpose
4. Add simple print-based output (not pytest assertions)
5. Update this README with description

## Environment Requirements

- Python 3.8+
- PyTorch with CUDA (optional but recommended)
- All dependencies from `requirements.txt`
- Set `PYTHONPATH=.` from repository root
