"""Quick test to verify DeepONet integration works"""

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.utils.params import DataParams, ModelParamsDecoder
from src.core.models.model import PredictionModel

print("="*80)
print("Quick DeepONet Integration Test")
print("="*80)

# Test 1: DeepONet standalone
print("\nTEST 1: DeepONet Standalone")
print("-"*40)

data_params = DataParams(
    batch=1, sequenceLength=[4, 2], dataSize=[8, 8],
    dimension=2, simFields=["pres"], simParams=[], normalizeMode=""
)

class MockParams:
    def __init__(self):
        self.arch = "deeponet"
        self.decWidth = 32
        self.pretrained = False
        self.frozen = False
        self.trainingNoise = 0.0

model_params = MockParams()

try:
    print("1. Creating model...")
    model = PredictionModel(p_d=data_params, p_t=None, p_l=None, p_me=None,
                           p_md=model_params, p_ml=None, useGPU=False)
    print(f"   ✓ Model type: {type(model.modelDecoder).__name__}")

    print("2. Creating dummy input...")
    x = torch.randn(1, 4, 3, 8, 8)  # [B, T, C, H, W]
    print(f"   ✓ Input shape: {x.shape}")

    print("3. Forward pass...")
    with torch.no_grad():
        out = model.forwardDirect(x)
    print(f"   ✓ Output shape: {out.shape}")
    print(f"   ✓ Is finite: {torch.isfinite(out).all().item()}")
    print("   ✅ DeepONet standalone WORKS!")

except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: DeepONet+DM
print("\nTEST 2: DeepONet+DM")
print("-"*40)

model_params2 = MockParams()
model_params2.arch = "genop-deeponet-diffusion"
model_params2.diffSteps = 5
model_params2.diffSchedule = "cosine"
model_params2.diffCondIntegration = "clean"
model_params2.trainingNoise = 0.0

data_params2 = DataParams(
    batch=1, sequenceLength=[2, 2], dataSize=[8, 8],
    dimension=2, simFields=["pres"], simParams=[], normalizeMode=""
)

try:
    print("1. Creating model...")
    model2 = PredictionModel(p_d=data_params2, p_t=None, p_l=None, p_me=None,
                            p_md=model_params2, p_ml=None, useGPU=False)
    print(f"   ✓ Model type: {type(model2.modelDecoder).__name__}")

    print("2. Creating dummy input...")
    x2 = torch.randn(1, 2, 3, 8, 8)
    print(f"   ✓ Input shape: {x2.shape}")

    print("3. Forward pass...")
    with torch.no_grad():
        out2 = model2.forwardDirect(x2)
    print(f"   ✓ Output shape: {out2.shape}")
    print(f"   ✓ Is finite: {torch.isfinite(out2).all().item()}")
    print("   ✅ DeepONet+DM WORKS!")

except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("Integration tests complete!")
print("="*80)
