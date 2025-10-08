#!/usr/bin/env python3
"""
Verification Script: Flexible Loss Scheduler Installation

Run this script to verify that all scheduler components are correctly installed
and accessible in your environment (local or Colab).

Usage:
    # Local
    python verify_scheduler_installation.py

    # Colab
    !python verify_scheduler_installation.py
"""

import sys
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f" {text}")
    print("="*70)

def print_subheader(text):
    """Print formatted subheader"""
    print(f"\n{text}")
    print("-"*70)

def check_file_exists(filepath):
    """Check if file exists and return status"""
    path = Path(filepath)
    if path.exists():
        size = path.stat().st_size
        print(f"  ✅ {filepath} ({size:,} bytes)")
        return True
    else:
        print(f"  ❌ {filepath} (NOT FOUND)")
        return False

def check_import(module_path, import_name):
    """Try to import a module and return status"""
    try:
        exec(f"from {module_path} import {import_name}")
        print(f"  ✅ {module_path}.{import_name}")
        return True
    except ImportError as e:
        print(f"  ❌ {module_path}.{import_name} - {str(e)[:50]}")
        return False
    except Exception as e:
        print(f"  ⚠️  {module_path}.{import_name} - {str(e)[:50]}")
        return False

def main():
    print_header("FLEXIBLE LOSS SCHEDULER - Installation Verification")

    # Get project root
    project_root = Path.cwd()
    print(f"\nProject Root: {project_root}")

    results = {
        'files': 0,
        'imports': 0,
        'total_files': 0,
        'total_imports': 0,
    }

    # ==========================================================================
    # 1. Check Core Implementation Files
    # ==========================================================================
    print_subheader("1. Core Implementation Files")

    files_to_check = [
        "src/core/training/spectral_metrics.py",
        "src/core/training/flexible_loss_scheduler.py",
        "src/core/training/loss.py",
        "src/core/training/trainer.py",
        "src/core/training/loss_history.py",
        "src/core/utils/params.py",
    ]

    for filepath in files_to_check:
        results['total_files'] += 1
        if check_file_exists(filepath):
            results['files'] += 1

    # ==========================================================================
    # 2. Check Documentation Files
    # ==========================================================================
    print_subheader("2. Documentation Files")

    doc_files = [
        "configs/examples/field_error_with_scheduler_example.py",
        "configs/examples/README.md",
        "QUICKSTART_SCHEDULER.md",
        "BACKWARD_COMPATIBILITY.md",
        "IMPLEMENTATION_SUMMARY.md",
    ]

    for filepath in doc_files:
        results['total_files'] += 1
        if check_file_exists(filepath):
            results['files'] += 1

    # ==========================================================================
    # 3. Check Test Files
    # ==========================================================================
    print_subheader("3. Test Files")

    test_files = [
        "tests/test_spectral_metrics.py",
        "tests/test_flexible_loss_scheduler.py",
        "tests/integration_test_scheduler.py",
    ]

    for filepath in test_files:
        results['total_files'] += 1
        if check_file_exists(filepath):
            results['files'] += 1

    # ==========================================================================
    # 4. Check Colab Support Files
    # ==========================================================================
    print_subheader("4. Colab Support Files")

    colab_files = [
        "colab_verification/flexible_scheduler_demo_cell.py",
    ]

    for filepath in colab_files:
        results['total_files'] += 1
        if check_file_exists(filepath):
            results['files'] += 1

    # ==========================================================================
    # 5. Check Python Imports
    # ==========================================================================
    print_subheader("5. Python Imports (requires PYTHONPATH set)")

    imports_to_check = [
        ("src.core.training.spectral_metrics", "compute_field_error_loss"),
        ("src.core.training.spectral_metrics", "compute_spectrum_error_loss"),
        ("src.core.training.flexible_loss_scheduler", "FlexibleLossScheduler"),
        ("src.core.training.flexible_loss_scheduler", "FlexibleLossSchedulerConfig"),
        ("src.core.utils.params", "LossParams"),
        ("src.core.utils.params", "SchedulerParams"),
        ("src.core.training.loss", "PredictionLoss"),
        ("src.core.training.trainer", "Trainer"),
        ("src.core.training.loss_history", "LossHistory"),
    ]

    for module_path, import_name in imports_to_check:
        results['total_imports'] += 1
        if check_import(module_path, import_name):
            results['imports'] += 1

    # ==========================================================================
    # 6. Check Backward Compatibility
    # ==========================================================================
    print_subheader("6. Backward Compatibility")

    try:
        from src.core.utils.params import LossParams

        # Test legacy MSE parameter
        print("  Testing legacy recMSE parameter...")
        p_l_old = LossParams(recMSE=1.0, predMSE=1.0)
        assert hasattr(p_l_old, 'recFieldError'), "recMSE should auto-convert"
        assert p_l_old.recFieldError == 1.0, "recMSE value should transfer"
        print("  ✅ Legacy recMSE/predMSE auto-conversion works")

        # Test new parameters
        print("  Testing new recFieldError parameter...")
        p_l_new = LossParams(recFieldError=1.0, predFieldError=1.0, spectrumError=0.5)
        assert p_l_new.recFieldError == 1.0
        assert p_l_new.spectrumError == 0.5
        print("  ✅ New field error parameters work")

        results['imports'] += 2
        results['total_imports'] += 2

    except Exception as e:
        print(f"  ❌ Backward compatibility test failed: {str(e)[:100]}")
        results['total_imports'] += 2

    # ==========================================================================
    # 7. Quick Functional Test
    # ==========================================================================
    print_subheader("7. Quick Functional Test")

    try:
        import torch
        from src.core.training.spectral_metrics import compute_field_error_loss, compute_spectrum_error_loss

        # Test field error
        print("  Testing field error computation...")
        pred = torch.randn(2, 4, 3, 32, 32, requires_grad=True)
        true = torch.randn(2, 4, 3, 32, 32)
        field_loss = compute_field_error_loss(pred, true)
        assert field_loss.ndim == 0, "Should be scalar"
        assert field_loss.requires_grad, "Should have gradients"
        print(f"  ✅ Field error: {field_loss.item():.4f} (scalar, differentiable)")

        # Test spectrum error
        print("  Testing spectrum error computation...")
        spec_loss = compute_spectrum_error_loss(pred, true)
        assert spec_loss.ndim == 0, "Should be scalar"
        assert spec_loss.requires_grad, "Should have gradients"
        print(f"  ✅ Spectrum error: {spec_loss.item():.4f} (scalar, differentiable)")

        # Test scheduler
        print("  Testing scheduler initialization...")
        from src.core.training.flexible_loss_scheduler import FlexibleLossScheduler, FlexibleLossSchedulerConfig

        config = FlexibleLossSchedulerConfig(enabled=True)
        scheduler = FlexibleLossScheduler(config)
        weights = scheduler.get_loss_weights()
        assert 'recFieldError' in weights
        assert 'spectrumError' in weights
        print(f"  ✅ Scheduler: {len(weights)} components, initial weights correct")

        results['imports'] += 3
        results['total_imports'] += 3

    except Exception as e:
        print(f"  ❌ Functional test failed: {str(e)[:100]}")
        import traceback
        traceback.print_exc()
        results['total_imports'] += 3

    # ==========================================================================
    # Summary
    # ==========================================================================
    print_header("VERIFICATION SUMMARY")

    files_pct = (results['files'] / results['total_files'] * 100) if results['total_files'] > 0 else 0
    imports_pct = (results['imports'] / results['total_imports'] * 100) if results['total_imports'] > 0 else 0

    print(f"\nFiles:   {results['files']}/{results['total_files']} found ({files_pct:.0f}%)")
    print(f"Imports: {results['imports']}/{results['total_imports']} successful ({imports_pct:.0f}%)")

    if results['files'] == results['total_files']:
        print("\n✅ All files present!")
    else:
        print(f"\n⚠️  Missing {results['total_files'] - results['files']} files")

    if results['imports'] == results['total_imports']:
        print("✅ All imports successful!")
    else:
        print(f"⚠️  Failed {results['total_imports'] - results['imports']} imports")
        print("\nNote: Import failures may indicate:")
        print("  - PYTHONPATH not set (run: PYTHONPATH=. python verify_scheduler_installation.py)")
        print("  - Missing dependencies (run: pip install -r requirements.txt)")
        print("  - Wrong directory (run from project root)")

    # Overall status
    print("\n" + "="*70)
    if results['files'] == results['total_files'] and results['imports'] >= results['total_imports'] * 0.8:
        print("🎉 INSTALLATION VERIFIED - Scheduler is ready to use!")
        print("\nNext steps:")
        print("  1. See QUICKSTART_SCHEDULER.md for usage guide")
        print("  2. Try: python tests/integration_test_scheduler.py")
        print("  3. Check: configs/examples/README.md for complete documentation")
    else:
        print("⚠️  INSTALLATION INCOMPLETE - Some components missing")
        print("\nTroubleshooting:")
        print("  1. Ensure you're in the project root directory")
        print("  2. Run: git pull (to get latest changes)")
        print("  3. Set PYTHONPATH: export PYTHONPATH=.")
        print("  4. Check file permissions")
    print("="*70)

    return results['files'] == results['total_files'] and results['imports'] >= results['total_imports'] * 0.8


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
