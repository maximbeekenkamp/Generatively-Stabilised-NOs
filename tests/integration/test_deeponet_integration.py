"""
Test DeepONet integration with PredictionModel and GenerativeOperatorModel

This script verifies:
1. DeepONet works as standalone decoder in PredictionModel
2. DeepONet works as prior in GenerativeOperatorModel (DeepONet+DM)
3. Both configurations produce valid outputs
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.utils.params import DataParams, ModelParamsDecoder
from src.core.models.model import PredictionModel
from tests.fixtures.dummy_datasets import get_dummy_batch


def test_deeponet_standalone():
    """Test DeepONet as standalone decoder in PredictionModel"""
    print("\n" + "="*80)
    print("TEST 1: DeepONet Standalone in PredictionModel")
    print("="*80)

    # Create parameters
    data_params = DataParams(
        batch=2,
        sequenceLength=[8, 2],
        dataSize=[16, 16],
        dimension=2,
        simFields=["pres"],
        simParams=[],
        normalizeMode=""
    )

    # Mock model parameters for DeepONet
    class MockModelParams:
        def __init__(self):
            self.arch = "deeponet"
            self.decWidth = 64  # Latent dimension for DeepONet
            self.pretrained = False
            self.frozen = False

    model_params = MockModelParams()

    try:
        # Create PredictionModel with DeepONet decoder
        print("\n1. Initializing PredictionModel with DeepONet decoder...")
        model = PredictionModel(
            p_d=data_params,
            p_t=None,
            p_l=None,
            p_me=None,
            p_md=model_params,
            p_ml=None,
            useGPU=False
        )
        print("   ✓ Model initialized successfully")
        print(f"   ✓ Decoder type: {type(model.modelDecoder).__name__}")

        # Get dummy data
        print("\n2. Loading dummy data...")
        batch = get_dummy_batch(dataset_name='inc_low',
                               batch_size=data_params.batch,
                               sequence_length=data_params.sequenceLength[0],
                               spatial_size=data_params.dataSize[0])
        data = batch['data']
        print(f"   ✓ Data shape: {data.shape}")

        # Forward pass
        print("\n3. Running forward pass...")
        with torch.no_grad():
            output = model.forwardDecoder(data)

        print(f"   ✓ Output shape: {output.shape}")
        print(f"   ✓ Output is finite: {torch.isfinite(output).all().item()}")
        print(f"   ✓ Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

        # Verify shape
        expected_shape = data.shape
        assert output.shape == expected_shape, f"Shape mismatch: {output.shape} vs {expected_shape}"
        print(f"   ✓ Output shape matches input shape")

        print("\n✅ DeepONet standalone test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ DeepONet standalone test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_deeponet_plus_dm():
    """Test DeepONet as prior in GenerativeOperatorModel (DeepONet+DM)"""
    print("\n" + "="*80)
    print("TEST 2: DeepONet+DM in GenerativeOperatorModel")
    print("="*80)

    # Create parameters
    data_params = DataParams(
        batch=1,
        sequenceLength=[2, 2],  # Shorter sequence for diffusion
        dataSize=[16, 16],
        dimension=2,
        simFields=["pres"],
        simParams=[],
        normalizeMode=""
    )

    # Mock model parameters for DeepONet+DM
    class MockModelParams:
        def __init__(self):
            self.arch = "genop-deeponet-diffusion"
            self.decWidth = 32
            self.diffSteps = 10  # Few steps for fast testing
            self.diffSchedule = "cosine"
            self.diffCondIntegration = "clean"
            self.pretrained = False
            self.frozen = False

    model_params = MockModelParams()

    try:
        # Create PredictionModel with GenOp-DeepONet-Diffusion
        print("\n1. Initializing PredictionModel with DeepONet+DM...")
        model = PredictionModel(
            p_d=data_params,
            p_t=None,
            p_l=None,
            p_me=None,
            p_md=model_params,
            p_ml=None,
            useGPU=False
        )
        print("   ✓ Model initialized successfully")
        print(f"   ✓ Decoder type: {type(model.modelDecoder).__name__}")

        # Get dummy data
        print("\n2. Loading dummy data...")
        batch = get_dummy_batch(dataset_name='inc_low',
                               batch_size=data_params.batch,
                               sequence_length=data_params.sequenceLength[0],
                               spatial_size=data_params.dataSize[0])
        data = batch['data']
        print(f"   ✓ Data shape: {data.shape}")

        # Forward pass
        print("\n3. Running forward pass with diffusion...")
        with torch.no_grad():
            output = model.forwardDecoder(data)

        print(f"   ✓ Output shape: {output.shape}")
        print(f"   ✓ Output is finite: {torch.isfinite(output).all().item()}")
        print(f"   ✓ Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

        # Verify shape
        expected_shape = data.shape
        assert output.shape == expected_shape, f"Shape mismatch: {output.shape} vs {expected_shape}"
        print(f"   ✓ Output shape matches input shape")

        print("\n✅ DeepONet+DM test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ DeepONet+DM test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_deeponet_all_datasets():
    """Test DeepONet on all dataset types"""
    print("\n" + "="*80)
    print("TEST 3: DeepONet on All Dataset Types")
    print("="*80)

    datasets = ['inc_low', 'inc_high', 'tra_ext', 'tra_inc', 'iso']

    data_params = DataParams(
        batch=1,
        sequenceLength=[4, 2],
        dataSize=[16, 16],
        dimension=2,
        simFields=["pres"],
        simParams=[],
        normalizeMode=""
    )

    class MockModelParams:
        def __init__(self):
            self.arch = "deeponet"
            self.decWidth = 32
            self.pretrained = False
            self.frozen = False

    model_params = MockModelParams()

    try:
        # Create model once
        print("\n1. Initializing model...")
        model = PredictionModel(
            p_d=data_params,
            p_t=None,
            p_l=None,
            p_me=None,
            p_md=model_params,
            p_ml=None,
            useGPU=False
        )

        all_passed = True
        print("\n2. Testing on all datasets...")
        for dataset_name in datasets:
            print(f"\n   Testing {dataset_name}...")
            batch = get_dummy_batch(dataset_name=dataset_name,
                                   batch_size=data_params.batch,
                                   sequence_length=data_params.sequenceLength[0],
                                   spatial_size=data_params.dataSize[0])
            data = batch['data']

            with torch.no_grad():
                output = model.forwardDecoder(data)

            is_finite = torch.isfinite(output).all().item()
            shape_match = output.shape == data.shape

            if is_finite and shape_match:
                print(f"   ✓ {dataset_name}: PASS")
            else:
                print(f"   ❌ {dataset_name}: FAIL (finite={is_finite}, shape={shape_match})")
                all_passed = False

        if all_passed:
            print("\n✅ All dataset tests PASSED")
        else:
            print("\n❌ Some dataset tests FAILED")

        return all_passed

    except Exception as e:
        print(f"\n❌ Dataset tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("DeepONet Integration Tests")
    print("="*80)

    results = []

    # Run tests
    results.append(("DeepONet Standalone", test_deeponet_standalone()))
    results.append(("DeepONet+DM", test_deeponet_plus_dm()))
    results.append(("DeepONet All Datasets", test_deeponet_all_datasets()))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name}: {status}")

    all_passed = all(passed for _, passed in results)
    print("\n" + "="*80)
    if all_passed:
        print("ALL TESTS PASSED ✅")
    else:
        print("SOME TESTS FAILED ❌")
    print("="*80 + "\n")
