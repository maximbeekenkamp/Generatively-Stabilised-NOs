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
## Complete Model Testing on Real Data

**Purpose**: Verify ALL models work with real TRA turbulence data

**Models Tested** (13 total):
- **Neural Operators (4)**: FNO, TNO, UNet, DeepONet
- **NO + Diffusion (4)**: FNO+DM, TNO+DM, UNet+DM, DeepONet+DM
- **Legacy Deterministic (3)**: ResNet, Dil-ResNet, Latent-MGN
- **Legacy Diffusion (2)**: ACDM, Refiner

**Dataset**: TRA (128_small_tra) - 287 MB of real turbulence data from TUM server

**Why TRA-only?**
- Real data with proven format (no synthetic data issues)
- Tests complete pipeline: download → train → checkpoints
- Verifies all model architectures work

**Expected Runtime**: ~4-6 hours on T4 GPU (13 models × 50 epochs)"""))

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
# Fix protobuf compatibility with Python 3.13+ and tensorboard
!pip install -q neuraloperator matplotlib seaborn rich einops scipy pyyaml "protobuf>=3.20.0,<4.0.0" tensorboard
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

Training ALL 13 models on TRA dataset:
- **Neural Operators (4)**: FNO, TNO, UNet, DeepONet
- **NO + Diffusion (4)**: FNO+DM, TNO+DM, UNet+DM, DeepONet+DM
- **Legacy Deterministic (3)**: ResNet, Dil-ResNet, Latent-MGN
- **Legacy Diffusion (2)**: ACDM, Refiner

Each model trains for 50 epochs on full dataset with batch size 4.

**Total**: 13 models × 50 epochs ≈ 4-6 hours on T4 GPU"""))

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
from src.core.utils.params import DataParams, TrainingParams, LossParams, ModelParamsDecoder, ModelParamsEncoder, ModelParamsLatent
from src.core.models.model import PredictionModel
from src.core.training.loss import PredictionLoss
from src.core.training.trainer import Trainer
from src.core.training.loss_history import LossHistory
from torch.utils.data import DataLoader

print("="*60)
print("📊 TRAINING MODELS (TRA ONLY)")
print("="*60)

# GPU-friendly training configuration
EPOCHS = 50  # More epochs for GPU training (vs 3 for local CPU)
BATCH_SIZE = 4  # Higher batch size for GPU (vs 1 for local CPU)

# TRA configuration (matches local_tra_verification.py)
TRA_CONFIG = {
    'filter_top': ['128_small_tra'],
    'filter_sim': [(0, 1)],
    'filter_frame': [(0, 1000)],  # Full frame range
    'sim_fields': ['dens', 'pres'],
    'sim_params': ['mach'],  # CRITICAL: Only 'mach', not 'rey'! Affects channel count.
    'normalize_mode': 'machMixed'  # Autoreg uses machMixed, not traMixed
}

# Models to test (13 models matching local_tra_verification.py)
MODELS = {
    # Neural Operators (Standalone)
    'fno': {'arch': 'fno', 'dec_width': 56, 'fno_modes': (16, 8)},
    'tno': {'arch': 'tno', 'dec_width': 96},
    'unet': {'arch': 'unet', 'dec_width': 96},
    'deeponet': {'arch': 'deeponet', 'dec_width': 96, 'n_sensors': 392,
                 'branch_batch_norm': True, 'trunk_batch_norm': True},

    # Neural Operators + Diffusion Models (Generative Operators) - Stage 1: prior-only training
    'fno_dm': {'arch': 'genop-fno-diffusion', 'dec_width': 56, 'fno_modes': (16, 8), 'diff_steps': 20, 'training_stage': 1,
               'load_pretrained_prior': True, 'prior_checkpoint_key': 'fno_tra'},
    'tno_dm': {'arch': 'genop-tno-diffusion', 'dec_width': 96, 'diff_steps': 20, 'training_stage': 1,
               'load_pretrained_prior': True, 'prior_checkpoint_key': 'tno_tra'},
    'unet_dm': {'arch': 'genop-unet-diffusion', 'dec_width': 96, 'diff_steps': 20, 'training_stage': 1,
                'load_pretrained_prior': True, 'prior_checkpoint_key': 'unet_tra'},
    'deeponet_dm': {'arch': 'genop-deeponet-diffusion', 'dec_width': 96, 'diff_steps': 20, 'training_stage': 1, 'n_sensors': 392,
                    'load_pretrained_prior': True, 'prior_checkpoint_key': 'deeponet_tra',
                    'branch_batch_norm': True, 'trunk_batch_norm': True},

    # Legacy Deterministic
    'resnet': {'arch': 'resnet', 'dec_width': 144},
    'dil_resnet': {'arch': 'dil_resnet', 'dec_width': 144},
    'latent_mgn': {'arch': 'skip', 'dec_width': 96, 'enc_width': 32, 'latent_size': 32,
                   'requires_encoder': True, 'requires_latent': True, 'vae': False},

    # Legacy Diffusion (Standalone)
    'acdm': {'arch': 'direct-ddpm+Prev', 'diff_steps': 20, 'sequence_length': [3, 2],
             'is_diffusion_model': True, 'diff_cond_integration': 'noisy'},
    'refiner': {'arch': 'refiner', 'diff_steps': 4, 'refiner_std': 0.000001,
                'is_diffusion_model': True, 'dec_width': 96},
}

def create_model_params(config):
    '''Create model parameters matching local_tra_verification.py'''
    # Create encoder params if needed (for LatentMGN)
    p_me = None
    if config.get('requires_encoder'):
        p_me = ModelParamsEncoder(
            arch="skip",
            pretrained=False,
            encWidth=config.get('enc_width', 32),
            latentSize=config.get('latent_size', 32)
        )

    # Create latent params if needed (for LatentMGN)
    p_ml = None
    if config.get('requires_latent'):
        p_ml = ModelParamsLatent(
            arch="transformerMGN",
            pretrained=False,
            width=1024,
            layers=1,
            dropout=0.0,
            transTrainUnroll=True,
            transTargetFull=False,
            maxInputLen=30
        )

    # Create decoder params (required for all models)
    p_md = ModelParamsDecoder(
        arch=config['arch'],
        pretrained=False,
        decWidth=config.get('dec_width', 96),
        fnoModes=config.get('fno_modes'),
        diffSteps=config.get('diff_steps'),
        diffSchedule=config.get('diff_schedule', 'linear'),
        diffCondIntegration=config.get('diff_cond_integration', 'noisy'),
        refinerStd=config.get('refiner_std'),
        vae=config.get('vae', False),
        n_sensors=config.get('n_sensors'),
        training_stage=config.get('training_stage')
    )

    # Store DeepONet overrides in p_md for architecture matching
    deeponet_overrides = {}
    if 'deeponet' in config['arch'].lower():
        deeponet_keys = ['branch_batch_norm', 'trunk_batch_norm', 'branch_layers', 'trunk_layers',
                       'branch_activation', 'trunk_activation', 'branch_dropout', 'trunk_dropout']
        for key in deeponet_keys:
            if key in config:
                setattr(p_md, key, config[key])
                deeponet_overrides[key] = config[key]

    return p_me, p_md, p_ml, deeponet_overrides

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
        # Get sequence length for this model
        seq_len = config.get('sequence_length', [2, 2])

        # Create dataset
        dataset = TurbulenceDataset(
            name=f"TRA_{model_name}",
            dataDirs=["data"],
            filterTop=TRA_CONFIG['filter_top'],
            filterSim=TRA_CONFIG['filter_sim'],
            filterFrame=TRA_CONFIG['filter_frame'],
            sequenceLength=[seq_len],
            randSeqOffset=True,
            simFields=TRA_CONFIG['sim_fields'],
            simParams=TRA_CONFIG['sim_params'],
            printLevel="none"
        )

        # DeepONet models with BatchNorm require batch_size >= 2
        model_batch_size = BATCH_SIZE
        if 'deeponet' in config['arch'].lower() and BATCH_SIZE < 2:
            model_batch_size = 2
            print(f"     Note: Using batch_size={model_batch_size} for DeepONet (BatchNorm requirement)")

        # Create params
        p_d = DataParams(
            batch=model_batch_size,
            augmentations=["normalize"],
            sequenceLength=seq_len,
            randSeqOffset=True,
            dataSize=[128, 64],  # Match autoreg reference
            dimension=2,
            simFields=TRA_CONFIG['sim_fields'],
            simParams=TRA_CONFIG['sim_params'],
            normalizeMode=TRA_CONFIG['normalize_mode']
        )

        # GPU-friendly training parameters: higher epochs, batch size, and learning rate decay
        p_t = TrainingParams(epochs=EPOCHS, lr=0.0001, expLrGamma=0.995)

        # Configure loss with LSIM for better perceptual quality on GPU
        p_l = LossParams(recMSE=0.0, predMSE=1.0, predLSIM=1.0)
        # Optional: Enable TNO relative L2 loss
        # p_l = LossParams(recMSE=0.0, predMSE=1.0, predLSIM=1.0, tno_lp_loss=1.0)

        # Create model params using helper function
        p_me, p_md, p_ml, deeponet_overrides = create_model_params(config)

        # Handle pretrained prior loading for NO+DM models
        pretrain_path = ""
        if config.get('load_pretrained_prior') and config.get('prior_checkpoint_key'):
            prior_checkpoint = progress_dir / 'checkpoints' / f"{config['prior_checkpoint_key']}.pt"
            if prior_checkpoint.exists():
                pretrain_path = str(prior_checkpoint)
                print(f"     Loading pretrained prior from: {prior_checkpoint.name}")
                p_md.pretrained = True
            else:
                print(f"     Warning: Pretrained prior not found: {prior_checkpoint.name}")
                print(f"     Continuing with random initialization")

        # Create model
        model = PredictionModel(p_d, p_t, p_l, p_me, p_md, p_ml, pretrain_path, useGPU=torch.cuda.is_available())

        # Apply transforms
        transforms = Transforms(p_d)
        dataset.transform = transforms

        # Create data loader
        train_loader = DataLoader(dataset, batch_size=p_d.batch, shuffle=True,
                                  drop_last=True, num_workers=0)

        # Setup training using Trainer class from codebase
        from src.core.training.trainer import Trainer
        from src.core.training.loss_history import LossHistory

        # Try to import SummaryWriter, fall back to dummy if tensorboard has issues
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=str(progress_dir / 'logs' / model_name))
        except (ImportError, TypeError) as e:
            class DummyWriter:
                def add_scalar(self, *args, **kwargs): pass
                def flush(self): pass
                def close(self): pass
            writer = DummyWriter()

        # Setup optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=p_t.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=5,
            factor=p_t.expLrGamma
        )
        criterion = PredictionLoss(p_l, p_d.dimension, p_d.simFields, useGPU=torch.cuda.is_available())

        # Create training history tracker with Rich progress bar
        train_history = LossHistory(
            "_train", "Training", writer, len(train_loader),
            0, 1, printInterval=1, logInterval=1, simFields=p_d.simFields,
            use_rich_progress=True, total_epochs=p_t.epochs,
            model_name=model_name.upper(), loss_params=p_l
        )

        # Create Trainer with checkpoint support
        trainer = Trainer(
            model, train_loader, optimizer, lr_scheduler, criterion,
            train_history, writer, p_d, p_t,
            checkpoint_path=str(checkpoint_path),
            checkpoint_frequency=max(1, p_t.epochs // 5),  # Save 5 checkpoints during training
            min_epoch_for_scheduler=10  # Start LR scheduling earlier on Colab
        )

        print(f"     Training {p_t.epochs} epochs on {len(dataset)} samples using Trainer class...")

        # Training loop using Trainer.trainingStep()
        for epoch in range(p_t.epochs):
            trainer.trainingStep(epoch)

        # Save checkpoint with enhanced config for architecture reproduction
        enhanced_config = config.copy()

        # For DeepONet models, save the actual architecture config used
        if 'deeponet' in config['arch'].lower():
            enhanced_config.update(deeponet_overrides)

        # For NO+DM models, also capture prior architecture details if available
        if config.get('load_pretrained_prior') and config.get('prior_checkpoint_key'):
            prior_checkpoint_path = progress_dir / 'checkpoints' / f"{config['prior_checkpoint_key']}.pt"
            if prior_checkpoint_path.exists():
                try:
                    prior_checkpoint = torch.load(prior_checkpoint_path, map_location='cpu', weights_only=False)
                    prior_config = prior_checkpoint.get('config', {})
                    # Merge prior architecture details (e.g., DeepONet BatchNorm flags)
                    arch_keys = ['branch_batch_norm', 'trunk_batch_norm', 'branch_layers', 'trunk_layers',
                                'branch_activation', 'trunk_activation', 'branch_dropout', 'trunk_dropout',
                                'fno_modes', 'n_sensors']
                    for key in arch_keys:
                        if key in prior_config and key not in enhanced_config:
                            enhanced_config[key] = prior_config[key]
                except Exception as e:
                    print(f"     Warning: Could not load prior config: {e}")

        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': p_t.epochs,
            'config': enhanced_config
        }, checkpoint_path)

        progress['training'][checkpoint_key] = 'complete'
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

        print(f"     ✅ Complete!")

        # Cleanup
        train_history.cleanup()  # Stop Rich progress bar
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

# Cell 5: Sampling Phase
notebook["cells"].append(create_cell("code", """# Cell 5: Sampling Phase - Generate Predictions

def sample_model(model_name: str, config: dict):
    \"\"\"Generate predictions from a trained model\"\"\"
    checkpoint_key = f"{model_name}_tra"
    checkpoint_path = progress_dir / 'checkpoints' / f"{checkpoint_key}.pt"
    sample_output_path = progress_dir / 'sampling' / f"{checkpoint_key}.npz"

    # Check if model is trained
    if not checkpoint_path.exists():
        print(f"  ⚠️  {model_name.upper()}: No checkpoint found, skipping sampling")
        return False

    # Check if already sampled
    if progress.get('sampling', {}).get(checkpoint_key) == 'complete' and sample_output_path.exists():
        print(f"  ✅ {model_name.upper()}: Already sampled")
        return True

    print(f"  🔄 {model_name.upper()}: Generating predictions...")

    try:
        # Create test dataset
        test_dataset = TurbulenceDataset(
            name=f"TRA_test_{model_name}",
            dataDirs=["data"],
            filterTop=TRA_CONFIG['filter_top'],
            filterSim=[(0, 3)],  # Different sims for testing
            filterFrame=[(500, 750)],  # Different frames for testing
            sequenceLength=[[60, 2]],
            randSeqOffset=False,
            simFields=TRA_CONFIG['sim_fields'],
            simParams=TRA_CONFIG['sim_params'],
            printLevel="none"
        )

        # Create params
        p_d = DataParams(
            batch=1,
            augmentations=["normalize"],
            sequenceLength=[60, 2],
            randSeqOffset=False,
            dataSize=[128, 64],
            dimension=2,
            simFields=TRA_CONFIG['sim_fields'],
            simParams=TRA_CONFIG['sim_params'],
            normalizeMode=TRA_CONFIG['normalize_mode']
        )

        p_t = TrainingParams(epochs=1, lr=0.0001)
        p_l = LossParams(recMSE=0.0, predMSE=1.0)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)
        checkpoint_config = checkpoint.get('config', {})

        # Create model
        p_me, p_md, p_ml, deeponet_overrides = create_model_params(config, checkpoint_config)
        model = PredictionModel(p_d, p_t, p_l, p_me, p_md, p_ml, "", useGPU=torch.cuda.is_available())

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"     Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

        # Create test loader
        transforms = Transforms(p_d)
        test_dataset.transform = transforms
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

        # Create Tester
        from src.core.training.trainer import Tester

        class DummyWriter:
            def add_scalar(self, *args, **kwargs): pass
            def add_image(self, *args, **kwargs): pass
            def flush(self, *args, **kwargs): pass
            def close(self): pass
        writer = DummyWriter()

        criterion = PredictionLoss(p_l, p_d.dimension, p_d.simFields, useGPU=torch.cuda.is_available())
        test_history = LossHistory(
            "_test", "Testing", writer, len(test_loader),
            0, 1, printInterval=0, logInterval=0, simFields=p_d.simFields
        )

        tester = Tester(model, test_loader, criterion, test_history, p_t)

        # Generate predictions
        print(f"     Generating predictions on {len(test_dataset)} test samples...")
        predictions = tester.generatePredictions(output_path=str(sample_output_path), model_name=model_name.upper())

        # Save ground truth (only once)
        ground_truth_path = progress_dir / 'sampling' / 'groundTruth.dict'
        if not ground_truth_path.exists():
            print(f"     Saving ground truth data...")
            all_ground_truth = []
            with torch.no_grad():
                for sample in test_loader:
                    all_ground_truth.append(sample["data"])
            ground_truth_tensor = torch.cat(all_ground_truth, dim=0)
            torch.save({"data": ground_truth_tensor}, ground_truth_path)

        # Update progress
        if 'sampling' not in progress:
            progress['sampling'] = {}
        progress['sampling'][checkpoint_key] = 'complete'
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

        print(f"     ✅ Complete! Shape: {predictions.shape}")

        # Cleanup
        del model, test_dataset, test_loader
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"     ❌ Failed: {str(e)[:200]}")
        import traceback
        traceback.print_exc()
        return False

# Sample all models
print("\\n" + "="*60)
print("🔮 SAMPLING PHASE")
print("="*60)

sample_count = 0
for model_name in MODELS.keys():
    config = MODELS[model_name]
    if sample_model(model_name, config):
        sample_count += 1

print(f"\\n✅ Sampling complete: {sample_count}/{len(MODELS)} models")"""))

# Cell 6: Plotting Phase
notebook["cells"].append(create_cell("code", """# Cell 6: Plotting Phase - Generate Comparison Plots

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

print("\\n" + "="*60)
print("📊 PLOTTING PHASE")
print("="*60)

# Helper function to plot predictions vs ground truth
def plot_comparison(model_name: str, time_step: int = 30):
    \"\"\"Plot prediction vs ground truth for a specific model\"\"\"
    checkpoint_key = f"{model_name}_tra"
    sample_path = progress_dir / 'sampling' / f"{checkpoint_key}.npz"
    gt_path = progress_dir / 'sampling' / 'groundTruth.dict'

    if not sample_path.exists() or not gt_path.exists():
        print(f"  ⚠️  {model_name.upper()}: Missing data, skipping plot")
        return False

    try:
        # Load data
        predictions = np.load(sample_path)['arr_0']  # [N, T, C, H, W]
        ground_truth = torch.load(gt_path)['data'].numpy()  # [N, T, C, H, W]

        # Select first sequence and specific time step
        pred = predictions[0, time_step, :2]  # [C, H, W] - velocity fields only
        gt = ground_truth[0, time_step, :2]

        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f'{model_name.upper()} - Time Step {time_step}', fontsize=16)

        # Velocity X
        im0 = axes[0, 0].imshow(gt[0], cmap='RdBu_r', aspect='auto')
        axes[0, 0].set_title('Ground Truth - Velocity X')
        axes[0, 0].axis('off')
        plt.colorbar(im0, ax=axes[0, 0])

        im1 = axes[1, 0].imshow(pred[0], cmap='RdBu_r', aspect='auto')
        axes[1, 0].set_title('Prediction - Velocity X')
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0])

        # Velocity Y
        im2 = axes[0, 1].imshow(gt[1], cmap='RdBu_r', aspect='auto')
        axes[0, 1].set_title('Ground Truth - Velocity Y')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1])

        im3 = axes[1, 1].imshow(pred[1], cmap='RdBu_r', aspect='auto')
        axes[1, 1].set_title('Prediction - Velocity Y')
        axes[1, 1].axis('off')
        plt.colorbar(im3, ax=axes[1, 1])

        # Error maps
        error_x = np.abs(gt[0] - pred[0])
        error_y = np.abs(gt[1] - pred[1])

        im4 = axes[0, 2].imshow(error_x, cmap='hot', aspect='auto')
        axes[0, 2].set_title('Absolute Error - Velocity X')
        axes[0, 2].axis('off')
        plt.colorbar(im4, ax=axes[0, 2])

        im5 = axes[1, 2].imshow(error_y, cmap='hot', aspect='auto')
        axes[1, 2].set_title('Absolute Error - Velocity Y')
        axes[1, 2].axis('off')
        plt.colorbar(im5, ax=axes[1, 2])

        plt.tight_layout()

        # Save plot
        plot_dir = progress_dir / 'plots'
        plot_dir.mkdir(exist_ok=True)
        plot_path = plot_dir / f'{checkpoint_key}_comparison.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()

        print(f"  ✅ {model_name.upper()}: Plot saved to {plot_path}")
        return True

    except Exception as e:
        print(f"  ❌ {model_name.upper()}: Plotting failed - {str(e)[:100]}")
        return False

# Plot all models with progress bar
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

plot_count = 0
with Progress(
    SpinnerColumn(),
    TextColumn("[bold blue]{task.description}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TextColumn("•"),
    TextColumn("{task.completed}/{task.total} models"),
    TimeElapsedColumn(),
) as progress:
    task = progress.add_task("Generating plots", total=len(MODELS))

    for model_name in MODELS.keys():
        if plot_comparison(model_name):
            plot_count += 1
        progress.update(task, advance=1)

print(f"\\n✅ Plotting complete: {plot_count}/{len(MODELS)} models")"""))

# Cell 7: Summary
notebook["cells"].append(create_cell("code", """# Cell 7: Summary
print("="*60)
print("📊 VERIFICATION SUMMARY")
print("="*60)

with open(progress_file, 'r') as f:
    progress = json.load(f)

training_complete = sum(1 for v in progress.get('training', {}).values() if v == 'complete')
training_failed = sum(1 for v in progress.get('training', {}).values() if v == 'failed')
sampling_complete = sum(1 for v in progress.get('sampling', {}).values() if v == 'complete')

print(f"\\nTraining: {training_complete} complete, {training_failed} failed")
print(f"Sampling: {sampling_complete} complete")

print(f"\\nTraining Results:")
for key, status in progress.get('training', {}).items():
    symbol = '✅' if status == 'complete' else '❌'
    print(f"  {symbol} {key}")

print(f"\\nSampling Results:")
for key, status in progress.get('sampling', {}).items():
    symbol = '✅' if status == 'complete' else '❌'
    print(f"  {symbol} {key}")

if training_complete > 0 and sampling_complete > 0:
    print(f"\\n🎉 SUCCESS! Full pipeline verified: Training → Sampling → Plotting")
elif training_complete > 0:
    print(f"\\n✅ Training complete. Run sampling and plotting cells to generate visualizations.")
else:
    print(f"\\n⚠️  No models completed successfully.")

print("="*60)"""))

# Save notebook
output_path = Path(__file__).parent / "colab_tra_only_verification.ipynb"
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=2)

print(f"✅ TRA-only notebook generated: {output_path}")
print(f"   Total cells: {len(notebook['cells'])}")
