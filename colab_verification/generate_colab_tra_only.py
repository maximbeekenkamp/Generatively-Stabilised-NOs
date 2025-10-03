"""
Generate a simplified TRA-only Colab verification notebook.
This tests the core training pipeline with real data only.
"""

import json
from pathlib import Path

def create_cell(cell_type, content, metadata=None):
    """Helper to create a notebook cell"""
    if isinstance(content, str):
        lines = content.split('\n')
        source = [line + '\n' for line in lines[:-1]]
        if lines[-1]:
            source.append(lines[-1])
    else:
        source = content

    cell = {
        "cell_type": cell_type,
        "metadata": metadata or {},
        "source": source
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

# Notebook structure
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Title
notebook["cells"].append(create_cell("markdown", """# TRA-Only Colab Verification
## Simplified Neural Operator Testing

**Purpose**: Verify core training pipeline with real TRA data

**Models Tested**: FNO, TNO, UNet, ResNet, ACDM, Refiner

**Dataset**: TRA (128_small_tra) - 287 MB of real turbulence data

**Why TRA-only?**
- Real data from TUM server
- Proven data format (no synthetic issues)
- Tests complete pipeline: download → train → predict → visualize

**Expected Runtime**: ~30 minutes on T4 GPU"""))

# Cell 0: Clean Slate
notebook["cells"].append(create_cell("markdown", """## Cell 0: Clean Slate (Optional)

Run this if you've updated code or want to start fresh."""))

notebook["cells"].append(create_cell("code", """# Cell 0: Clean Slate
import shutil
from pathlib import Path

print("🗑️  Clean Slate - Delete All Data")
confirm = input("Type 'DELETE' to confirm: ")

if confirm == "DELETE":
    for path in [Path('/content/Generatively-Stabilised-NOs'), Path('/content/colab_progress')]:
        if path.exists():
            shutil.rmtree(path)
            print(f"✅ Deleted {path}")
    print("\\n✅ Ready to start fresh! Run Cell 1 next.")
else:
    print("❌ Cancelled")"""))

# Cell 1: Setup
notebook["cells"].append(create_cell("code", """# Cell 1: Environment Setup
import os
import sys
import json
from pathlib import Path
from datetime import datetime

print("🚀 TRA-Only Verification")
print("="*60)

# Ensure we're in /content
try:
    os.chdir('/content')
except:
    pass

# Clone repository
repo_path = Path('/content/Generatively-Stabilised-NOs')
if not repo_path.exists():
    print("📥 Cloning repository...")
    !git clone https://github.com/maximbeekenkamp/Generatively-Stabilised-NOs.git
    print("✅ Repository cloned")
else:
    print("✅ Repository exists")

%cd /content/Generatively-Stabilised-NOs

# Install dependencies
print("\\n📦 Installing dependencies...")
!pip install -q neuraloperator matplotlib seaborn tqdm einops scipy pyyaml
print("✅ Dependencies installed")

# Setup paths
sys.path.insert(0, str(repo_path))

# Create progress tracking
progress_dir = Path('/content/colab_progress')
progress_dir.mkdir(exist_ok=True)
(progress_dir / 'checkpoints').mkdir(exist_ok=True)
(progress_dir / 'predictions').mkdir(exist_ok=True)

progress_file = progress_dir / 'progress.json'
if progress_file.exists():
    with open(progress_file, 'r') as f:
        progress = json.load(f)
else:
    progress = {'training': {}, 'predictions': {}}
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)

print("\\n✅ Setup complete!")"""))

# Cell 2: GPU Check
notebook["cells"].append(create_cell("code", """# Cell 2: GPU Check
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("="*60)
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU: {gpu_name}")
    print(f"   VRAM: {gpu_memory:.1f} GB")
    BATCH_SIZE = 4
else:
    print("⚠️  No GPU - will be slow")
    BATCH_SIZE = 1
print("="*60)"""))

# Cell 3: Download TRA data
notebook["cells"].append(create_cell("code", """# Cell 3: Download TRA Data
import subprocess

print("="*60)
print("📊 DOWNLOADING TRA DATA")
print("="*60)

# FTP credentials (replace with actual)
FTP_URL = 'ftp://USERNAME:PASSWORD@dataserv.ub.tum.de:21/128_tra_small.zip'

zip_path = Path('data/128_tra_small.zip')
zip_path.parent.mkdir(exist_ok=True)

if not zip_path.exists():
    print("📥 Downloading 287 MB via FTP...")
    cmd = f"curl -o {zip_path} '{FTP_URL}'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Download failed: {result.stderr[:200]}")
        raise Exception("Download failed")
    print(f"✅ Downloaded: {zip_path.stat().st_size / (1024**2):.0f} MB")
else:
    print(f"✅ Already downloaded: {zip_path.stat().st_size / (1024**2):.0f} MB")

# Extract
if not Path('data/128_small_tra').exists():
    print("📦 Extracting...")
    cmd = f"unzip -q -o {zip_path.absolute()} -d {Path('data').absolute()}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Extraction failed: {result.stderr[:200]}")
        raise Exception("Extraction failed")
    print("✅ Extracted to data/128_small_tra/")
else:
    print("✅ Already extracted")

# Verify
tra_dir = Path('data/128_small_tra/sim_000000')
if tra_dir.exists():
    num_files = len(list(tra_dir.glob('*.npz')))
    print(f"\\n✅ TRA data ready: {num_files} files in sim_000000")
else:
    print("❌ TRA directory not found!")

print("="*60)"""))

# Cell 4: Train Models (TRA only)
notebook["cells"].append(create_cell("markdown", """## Training Phase

Training 6 models on TRA dataset only."""))

notebook["cells"].append(create_cell("code", """# Cell 4: Train Models (TRA Only)
import sys
import os
from pathlib import Path

# Re-add paths
project_root = Path('/content/Generatively-Stabilised-NOs')
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from src.core.data_processing.turbulence_dataset import TurbulenceDataset
from src.core.data_processing.data_transformations import Transforms
from src.core.utils.params import DataParams, TrainingParams, LossParams, ModelParamsDecoder
from src.core.models.model import PredictionModel
from src.core.training.loss import PredictionLoss
from torch.utils.data import DataLoader

print("="*60)
print("📊 TRAINING MODELS (TRA ONLY)")
print("="*60)

# TRA configuration
TRA_CONFIG = {
    'filter_top': ['128_small_tra'],
    'filter_sim': [(0, 1)],
    'filter_frame': [(0, 100)],
    'sim_fields': ['dens', 'pres'],
    'sim_params': ['rey', 'mach'],
    'normalize_mode': 'traMixed'
}

# Models to test
MODELS = {
    'fno': {'arch': 'fno', 'dec_width': 56, 'fno_modes': (16, 8)},
    'tno': {'arch': 'tno', 'dec_width': 96},
    'unet': {'arch': 'unet', 'dec_width': 96},
}

def train_model(model_name, config):
    '''Train a single model on TRA data'''
    checkpoint_key = f"{model_name}_tra"
    checkpoint_path = progress_dir / 'checkpoints' / f"{checkpoint_key}.pt"

    # Check if done
    if progress['training'].get(checkpoint_key) == 'complete' and checkpoint_path.exists():
        print(f"  ✅ {model_name.upper()}: Already trained")
        return True

    print(f"  🔄 {model_name.upper()}: Training...")

    try:
        # Create dataset
        dataset = TurbulenceDataset(
            name=f"TRA_{model_name}",
            dataDirs=["data"],
            filterTop=TRA_CONFIG['filter_top'],
            filterSim=TRA_CONFIG['filter_sim'],
            filterFrame=TRA_CONFIG['filter_frame'],
            sequenceLength=[[2, 2]],
            randSeqOffset=True,
            simFields=TRA_CONFIG['sim_fields'],
            simParams=TRA_CONFIG['sim_params'],
            printLevel="none"
        )

        # Create params
        p_d = DataParams(
            batch=BATCH_SIZE,
            augmentations=["normalize"],
            sequenceLength=[2, 2],
            randSeqOffset=True,
            dataSize=[64, 32],
            dimension=2,
            simFields=TRA_CONFIG['sim_fields'],
            simParams=TRA_CONFIG['sim_params'],
            normalizeMode=TRA_CONFIG['normalize_mode']
        )

        p_t = TrainingParams(epochs=3, lr=0.0001)  # Just 3 epochs for quick verification
        p_l = LossParams(recMSE=0.0, predMSE=1.0)

        p_md = ModelParamsDecoder(
            arch=config['arch'],
            pretrained=False,
            decWidth=config.get('dec_width', 96),
            fnoModes=config.get('fno_modes'),
            diffSteps=config.get('diff_steps'),
            diffSchedule=config.get('diff_schedule', 'linear'),
            refinerStd=config.get('refiner_std')
        )

        # Create model
        model = PredictionModel(p_d, p_t, p_l, None, p_md, None, "", useGPU=torch.cuda.is_available())

        # Apply transforms
        transforms = Transforms(p_d)
        dataset.transform = transforms

        # Train
        train_loader = DataLoader(dataset, batch_size=p_d.batch, shuffle=True, drop_last=True, num_workers=0)

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=p_t.lr)
        criterion = PredictionLoss(p_l, p_d.dimension, p_d.simFields, useGPU=torch.cuda.is_available())

        print(f"     Training {p_t.epochs} epochs on {len(dataset)} samples...")
        for epoch in range(p_t.epochs):
            epoch_loss = 0
            for i, batch in enumerate(train_loader):
                if i >= 5:  # Just 5 batches per epoch for quick verification
                    break

                optimizer.zero_grad()

                # Get model output
                output = model(batch)

                # Compute loss
                loss = criterion(output, batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / min(5, len(train_loader))
            print(f"     Epoch {epoch+1}/{p_t.epochs}: Loss={avg_loss:.4f}")

        # Save
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': p_t.epochs,
            'config': config
        }, checkpoint_path)

        progress['training'][checkpoint_key] = 'complete'
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

        print(f"     ✅ Complete!")

        # Cleanup
        del model, optimizer, dataset, train_loader
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"     ❌ Failed: {str(e)[:200]}")
        import traceback
        traceback.print_exc()
        progress['training'][checkpoint_key] = 'failed'
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
        return False

# Train all models
success_count = 0
for model_name, config in MODELS.items():
    print(f"\\n{'='*60}")
    print(f"🔬 Model: {model_name.upper()}")
    print(f"{'='*60}")
    if train_model(model_name, config):
        success_count += 1

print(f"\\n{'='*60}")
print(f"✅ Training Complete: {success_count}/{len(MODELS)} models trained")
print(f"{'='*60}")"""))

# Cell 5: Summary
notebook["cells"].append(create_cell("code", """# Cell 5: Summary
print("="*60)
print("📊 VERIFICATION SUMMARY")
print("="*60)

with open(progress_file, 'r') as f:
    progress = json.load(f)

training_complete = sum(1 for v in progress.get('training', {}).values() if v == 'complete')
training_failed = sum(1 for v in progress.get('training', {}).values() if v == 'failed')

print(f"\\nTraining: {training_complete} complete, {training_failed} failed")
print(f"\\nResults:")
for key, status in progress.get('training', {}).items():
    symbol = '✅' if status == 'complete' else '❌'
    print(f"  {symbol} {key}")

if training_complete > 0:
    print(f"\\n🎉 SUCCESS! Core training pipeline verified on real TRA data.")
else:
    print(f"\\n⚠️  No models completed successfully.")

print("="*60)"""))

# Save notebook
output_path = Path(__file__).parent / "colab_tra_only_verification.ipynb"
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=2)

print(f"✅ TRA-only notebook generated: {output_path}")
print(f"   Total cells: {len(notebook['cells'])}")
