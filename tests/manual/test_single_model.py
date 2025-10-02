"""Quick test of single model"""
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.utils.params import DataParams, ModelParamsDecoder
from src.core.models.model import PredictionModel
from tests.fixtures.dummy_datasets import get_dummy_batch

data_params = DataParams(
    batch=1,
    sequenceLength=[4, 2],
    dataSize=[16, 16],
    dimension=2,
    simFields=["pres"],
    simParams=[],
    normalizeMode=""
)

model_params = ModelParamsDecoder(
    arch="fno",
    decWidth=32,
    fnoModes=(8, 8),
    trainingNoise=0.0
)

print("Creating model...")
model = PredictionModel(
    p_d=data_params,
    p_t=None,
    p_l=None,
    p_me=None,
    p_md=model_params,
    p_ml=None,
    useGPU=False
)

print("Getting dummy data...")
batch = get_dummy_batch(
    dataset_name='inc_low',
    batch_size=1,
    sequence_length=4,
    spatial_size=16
)
data = batch['data']
print(f"Data shape: {data.shape}")
print(f"Data type: {type(batch)}, keys: {batch.keys()}")

print("Running model...")
with torch.no_grad():
    output = model.forwardDirect(data)

print(f"Output shape: {output.shape}")
print("Success!")
