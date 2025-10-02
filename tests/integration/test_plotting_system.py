"""
Test Plotting System with Generated NPZ Files

This script validates that the plotting scripts work with our generated model predictions.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

def test_npz_files():
    """Test that NPZ files are correctly formatted"""
    print("="*80)
    print("Testing NPZ File Format")
    print("="*80)

    npz_dir = Path("tests/outputs/model_predictions")

    if not npz_dir.exists():
        print(f"❌ Directory not found: {npz_dir}")
        return False

    npz_files = list(npz_dir.glob("predictions_*.npz"))

    if not npz_files:
        print(f"❌ No NPZ files found in {npz_dir}")
        return False

    print(f"\nFound {len(npz_files)} NPZ files:")

    all_valid = True
    for npz_file in npz_files:
        print(f"\n📁 {npz_file.name}")

        try:
            data = np.load(npz_file)

            if 'predFull' not in data:
                print(f"   ❌ Missing 'predFull' key")
                all_valid = False
                continue

            pred = data['predFull']
            print(f"   ✓ Shape: {pred.shape}")
            print(f"   ✓ Expected: [num_models, num_evals, num_sequences, timesteps, channels, H, W]")

            # Verify shape is 7D
            if len(pred.shape) != 7:
                print(f"   ❌ Expected 7D tensor, got {len(pred.shape)}D")
                all_valid = False
                continue

            # Check for finite values
            if not np.isfinite(pred).all():
                print(f"   ❌ Contains NaN or Inf values")
                all_valid = False
                continue

            print(f"   ✓ All values finite")
            print(f"   ✓ Range: [{pred.min():.4f}, {pred.max():.4f}]")

            # Read metadata if available
            metadata_file = npz_file.parent / f"metadata_{npz_file.stem.replace('predictions_', '')}.txt"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    content = f.read()
                    if 'model_names' in content:
                        models_line = [l for l in content.split('\n') if 'model_names' in l][0]
                        print(f"   ✓ {models_line}")

        except Exception as e:
            print(f"   ❌ Error loading file: {e}")
            all_valid = False

    if all_valid:
        print(f"\n✅ All NPZ files valid and ready for plotting!")
    else:
        print(f"\n❌ Some NPZ files have issues")

    return all_valid


def test_color_mappings():
    """Test that color mappings exist for our models"""
    print("\n" + "="*80)
    print("Testing Color Mappings")
    print("="*80)

    try:
        from src.analysis.plot_color_and_name_mapping import colorRemap, modelRemap

        our_models = ['FNO', 'TNO', 'UNet', 'DeepONet',
                      'FNO+DM', 'TNO+DM', 'UNet+DM', 'DeepONet+DM']

        print(f"\nChecking mappings for {len(our_models)} models:")

        missing_colors = []
        missing_names = []

        for model in our_models:
            has_color = model in colorRemap
            has_name = model in modelRemap

            status_color = "✓" if has_color else "❌"
            status_name = "✓" if has_name else "❌"

            print(f"  {model:15} Color: {status_color}  Name: {status_name}")

            if not has_color:
                missing_colors.append(model)
            if not has_name:
                missing_names.append(model)

        if missing_colors:
            print(f"\n❌ Missing color mappings: {missing_colors}")
        if missing_names:
            print(f"❌ Missing name mappings: {missing_names}")

        if not missing_colors and not missing_names:
            print(f"\n✅ All model mappings present!")
            return True
        else:
            return False

    except Exception as e:
        print(f"❌ Error loading mappings: {e}")
        return False


def create_simple_comparison_plot():
    """Create a simple comparison plot to verify plotting works"""
    print("\n" + "="*80)
    print("Creating Test Comparison Plot")
    print("="*80)

    try:
        import matplotlib.pyplot as plt
        from src.analysis.plot_color_and_name_mapping import colorRemap, modelRemap

        # Load one NPZ file
        npz_file = Path("tests/outputs/model_predictions/predictions_inc_low.npz")

        if not npz_file.exists():
            print(f"❌ File not found: {npz_file}")
            return False

        data = np.load(npz_file)
        pred = data['predFull']  # [num_models, num_evals, num_sequences, timesteps, channels, H, W]

        # Read model names from metadata
        metadata_file = npz_file.parent / "metadata_inc_low.txt"
        model_names = []
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                for line in f:
                    if 'model_names' in line:
                        # Extract list from string representation
                        names_str = line.split(':', 1)[1].strip()
                        model_names = eval(names_str)  # Safe here since we created the file

        print(f"\nLoaded data shape: {pred.shape}")
        print(f"Model names: {model_names}")

        # Create comparison plot
        num_models = pred.shape[0]
        t_final = pred.shape[3] - 1  # Last timestep

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Model Comparison - inc_low Dataset (Final Timestep)', fontsize=16)

        axes = axes.flatten()

        for i in range(min(num_models, 6)):  # Plot up to 6 models
            model_name = model_names[i] if i < len(model_names) else f"Model {i}"

            # Get prediction: [1, 1, timesteps, channels, H, W]
            # Take first eval, first sequence, last timestep, first channel
            img = pred[i, 0, 0, t_final, 0, :, :]

            # Get color for this model
            color = colorRemap.get(model_name, 'gray')
            display_name = modelRemap.get(model_name, model_name)

            # Plot
            im = axes[i].imshow(img, cmap='RdBu_r', aspect='auto')
            axes[i].set_title(f'{display_name}', color=color if isinstance(color, str) else 'black')
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046)

        # Hide unused subplots
        for i in range(num_models, 6):
            axes[i].axis('off')

        plt.tight_layout()

        # Save plot
        output_file = Path("tests/outputs/model_predictions/comparison_plot.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n✅ Comparison plot saved: {output_file}")
        print(f"   Models plotted: {num_models}")
        print(f"   Resolution: {pred.shape[-2]}x{pred.shape[-1]}")

        return True

    except Exception as e:
        print(f"❌ Error creating plot: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("PLOTTING SYSTEM VALIDATION")
    print("="*80 + "\n")

    results = []

    # Test 1: NPZ files
    results.append(("NPZ Files", test_npz_files()))

    # Test 2: Color mappings
    results.append(("Color Mappings", test_color_mappings()))

    # Test 3: Create comparison plot
    results.append(("Comparison Plot", create_simple_comparison_plot()))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20} {status}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "="*80)
    if all_passed:
        print("ALL TESTS PASSED ✅")
        print("Plotting system is ready to use!")
    else:
        print("SOME TESTS FAILED ❌")
        print("Check errors above")
    print("="*80 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
