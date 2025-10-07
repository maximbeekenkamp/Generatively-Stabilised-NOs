#!/usr/bin/env python3
"""
Local TRA Verification Script - CPU-Friendly Version

This script tests all 13 models on real TRA turbulence data locally with reduced epochs.
Designed for CPU testing with optional LSIM loss and individual model selection.

Usage:
    python colab_verification/local_tra_verification.py --run-all
    python colab_verification/local_tra_verification.py --fno --tno --use-lsim
    python colab_verification/local_tra_verification.py --fno-dm --deeponet-dm
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List

import torch
from torch.utils.data import DataLoader

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# Import directly from modules to avoid __init__ dependencies
import importlib.util

# Load modules directly to avoid circular imports and unnecessary dependencies
def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import required classes
from src.core.data_processing.turbulence_dataset import TurbulenceDataset
from src.core.data_processing.data_transformations import Transforms
from src.core.utils.params import DataParams, TrainingParams, LossParams, ModelParamsDecoder, ModelParamsEncoder, ModelParamsLatent
from src.core.models.model import PredictionModel

# Import PredictionLoss directly from the file to avoid training.__init__ issues
loss_module = load_module_from_path('loss_standalone', PROJECT_ROOT / 'src/core/training/loss.py')
PredictionLoss = loss_module.PredictionLoss


# FTP Configuration
FTP_USERNAME = "m1734798.001"
FTP_PASSWORD = "m1734798.001"
FTP_HOST = "dataserv.ub.tum.de"
FTP_PORT = "21"
FTP_FILE = "128_tra_small.zip"
FTP_URL = f"ftp://{FTP_USERNAME}:{FTP_PASSWORD}@{FTP_HOST}:{FTP_PORT}/{FTP_FILE}"

# TRA Dataset Configuration - Matches autoreg reference (training_tra_legacy.py, training_diffusion_tra_legacy.py)
TRA_CONFIG = {
    'filter_top': ['128_small_tra'],  # Using small dataset for verification
    'filter_sim': [(0, 1)],  # Simplified filter for small dataset
    'filter_frame': [(0, 1000)],  # Full frame range
    'sim_fields': ['dens', 'pres'],
    'sim_params': ['mach'],  # CRITICAL: Only 'mach', not 'rey'! Affects channel count.
    'normalize_mode': 'machMixed'  # Autoreg uses machMixed, not traMixed
}

# Model Configurations (13 models total)
MODELS = {
    # Neural Operators (Standalone)
    'fno': {'arch': 'fno', 'dec_width': 56, 'fno_modes': (16, 8)},
    'tno': {'arch': 'tno', 'dec_width': 96},
    'unet': {'arch': 'unet', 'dec_width': 96},
    'deeponet': {'arch': 'deeponet', 'dec_width': 96, 'n_sensors': 392,
                 'branch_batch_norm': True, 'trunk_batch_norm': True},  # NOTE: Sampling may fail due to architecture drift (BatchNorm layer count mismatch)

    # Neural Operators + Diffusion Models (Generative Operators) - Stage 1: prior-only training
    'fno_dm': {'arch': 'genop-fno-diffusion', 'dec_width': 56, 'fno_modes': (16, 8), 'diff_steps': 20, 'training_stage': 1,
               'load_pretrained_prior': True, 'prior_checkpoint_key': 'fno_tra'},
    'tno_dm': {'arch': 'genop-tno-diffusion', 'dec_width': 96, 'diff_steps': 20, 'training_stage': 1,
               'load_pretrained_prior': True, 'prior_checkpoint_key': 'tno_tra'},
    'unet_dm': {'arch': 'genop-unet-diffusion', 'dec_width': 96, 'diff_steps': 20, 'training_stage': 1,
                'load_pretrained_prior': True, 'prior_checkpoint_key': 'unet_tra'},
    'deeponet_dm': {'arch': 'genop-deeponet-diffusion', 'dec_width': 96, 'diff_steps': 20, 'training_stage': 1, 'n_sensors': 392,
                    'branch_batch_norm': True, 'trunk_batch_norm': True,  # Must match prior architecture
                    'load_pretrained_prior': True, 'prior_checkpoint_key': 'deeponet_tra'},

    # Legacy Deterministic
    'resnet': {'arch': 'resnet', 'dec_width': 144},
    'dil_resnet': {'arch': 'dil_resnet', 'dec_width': 144},
    'latent_mgn': {'arch': 'skip', 'dec_width': 96, 'enc_width': 32, 'latent_size': 32,
                   'requires_encoder': True, 'requires_latent': True,
                   'vae': False},  # Decoder arch is 'skip', latent arch is 'transformerMGN'

    # Legacy Diffusion (Standalone) - Use diffusion-specific training loop
    'acdm': {'arch': 'direct-ddpm+Prev', 'diff_steps': 20, 'sequence_length': [3, 2],
             'is_diffusion_model': True, 'diff_cond_integration': 'noisy'},
    'refiner': {'arch': 'refiner', 'diff_steps': 4, 'refiner_std': 0.000001,
                'is_diffusion_model': True, 'sequence_length': [2, 2]},
}


def parse_args():
    """Parse command-line arguments for model selection"""
    parser = argparse.ArgumentParser(description='Local TRA Verification - Train models with reduced epochs')

    # Model selection flags
    parser.add_argument('--run-all', action='store_true', help='Train all 13 models')

    # Neural Operators
    parser.add_argument('--fno', action='store_true', help='Train FNO model')
    parser.add_argument('--tno', action='store_true', help='Train TNO model')
    parser.add_argument('--unet', action='store_true', help='Train UNet model')
    parser.add_argument('--deeponet', action='store_true', help='Train DeepONet model')

    # Neural Operators + Diffusion
    parser.add_argument('--fno-dm', action='store_true', help='Train FNO + Diffusion model')
    parser.add_argument('--tno-dm', action='store_true', help='Train TNO + Diffusion model')
    parser.add_argument('--unet-dm', action='store_true', help='Train UNet + Diffusion model')
    parser.add_argument('--deeponet-dm', action='store_true', help='Train DeepONet + Diffusion model')

    # Legacy Deterministic
    parser.add_argument('--resnet', action='store_true', help='Train ResNet model')
    parser.add_argument('--dil-resnet', action='store_true', help='Train Dilated ResNet model')
    parser.add_argument('--latent-mgn', action='store_true', help='Train Latent MGN model')

    # Legacy Diffusion
    parser.add_argument('--acdm', action='store_true', help='Train ACDM model')
    parser.add_argument('--refiner', action='store_true', help='Train Refiner model')

    # Options
    parser.add_argument('--use-lsim', action='store_true', help='Enable LSIM loss (slower on CPU)')
    parser.add_argument('--use-tno-loss', action='store_true', help='Enable TNO relative L2 loss')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs per model (default: 2)')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size (default: 1 for CPU)')

    # Phases
    parser.add_argument('--skip-training', action='store_true', help='Skip training phase (use existing checkpoints)')
    parser.add_argument('--sample', action='store_true', help='Run sampling phase (generate predictions)')
    parser.add_argument('--plot', action='store_true', help='Run plotting phase')
    parser.add_argument('--plot-groups', nargs='+', help='Plot groups to generate (e.g., neural_operators no_dm)')

    return parser.parse_args()


def get_selected_models(args):
    """Determine which models to train based on arguments"""
    # If run-all is specified or no individual flags, train all models
    model_flags = [
        args.fno, args.tno, args.unet, args.deeponet,
        args.fno_dm, args.tno_dm, args.unet_dm, args.deeponet_dm,
        args.resnet, args.dil_resnet, args.latent_mgn,
        args.acdm, args.refiner
    ]

    if args.run_all or not any(model_flags):
        return list(MODELS.keys())

    # Build list of selected models
    selected = []
    flag_to_model = {
        'fno': 'fno', 'tno': 'tno', 'unet': 'unet', 'deeponet': 'deeponet',
        'fno_dm': 'fno_dm', 'tno_dm': 'tno_dm', 'unet_dm': 'unet_dm', 'deeponet_dm': 'deeponet_dm',
        'resnet': 'resnet', 'dil_resnet': 'dil_resnet', 'latent_mgn': 'latent_mgn',
        'acdm': 'acdm', 'refiner': 'refiner'
    }

    for flag_name, model_name in flag_to_model.items():
        if getattr(args, flag_name.replace('-', '_')):
            selected.append(model_name)

    return selected


def download_and_extract_tra_data():
    """Download and extract TRA data if not already present"""
    print("=" * 60)
    print("📊 TRA DATA SETUP")
    print("=" * 60)

    zip_path = PROJECT_ROOT / 'data' / '128_tra_small.zip'
    tra_dir = PROJECT_ROOT / 'data' / '128_small_tra'

    # Create data directory if it doesn't exist
    zip_path.parent.mkdir(exist_ok=True)

    # Check if data already exists
    if tra_dir.exists() and (tra_dir / 'sim_000000').exists():
        num_files = len(list((tra_dir / 'sim_000000').glob('*.npz')))
        print(f"✅ TRA data already exists: {num_files} files in sim_000000")
        print("=" * 60)
        return

    # Download if zip doesn't exist
    if not zip_path.exists():
        print(f"📥 Downloading TRA data (287 MB) from FTP server...")
        print(f"   Server: {FTP_HOST}")
        try:
            cmd = f"curl -o {zip_path} '{FTP_URL}'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Download failed: {result.stderr[:200]}")
                raise Exception("FTP download failed")
            print(f"✅ Downloaded: {zip_path.stat().st_size / (1024**2):.0f} MB")
        except Exception as e:
            print(f"❌ Error downloading: {str(e)}")
            raise
    else:
        print(f"✅ Zip file already exists: {zip_path.stat().st_size / (1024**2):.0f} MB")

    # Extract
    if not tra_dir.exists():
        print("📦 Extracting zip file...")
        try:
            cmd = f"unzip -q -o {zip_path.absolute()} -d {zip_path.parent.absolute()}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Extraction failed: {result.stderr[:200]}")
                raise Exception("Extraction failed")
            print("✅ Extracted to data/128_small_tra/")
        except Exception as e:
            print(f"❌ Error extracting: {str(e)}")
            raise

    # Verify
    if (tra_dir / 'sim_000000').exists():
        num_files = len(list((tra_dir / 'sim_000000').glob('*.npz')))
        print(f"✅ TRA data ready: {num_files} files in sim_000000")
    else:
        raise Exception("❌ TRA directory not found after extraction!")

    print("=" * 60)


def setup_progress_tracking():
    """Setup progress tracking directory and JSON file"""
    progress_dir = PROJECT_ROOT / 'local_progress'
    progress_dir.mkdir(exist_ok=True)
    (progress_dir / 'checkpoints').mkdir(exist_ok=True)

    progress_file = progress_dir / 'progress.json'
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            progress = json.load(f)
    else:
        progress = {'training': {}}
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

    return progress_dir, progress_file, progress


def create_model_params(config: Dict[str, Any], checkpoint_config: Dict[str, Any] = None):
    """
    Create model parameter objects (p_me, p_md, p_ml) from config dictionary.
    This function is shared between training and sampling phases.

    Args:
        config: Model configuration dictionary with architecture details
        checkpoint_config: Optional checkpoint config with saved architecture details
                          (e.g., BatchNorm flags for DeepONet). Takes priority over config.

    Returns:
        Tuple of (p_me, p_md, p_ml, deeponet_overrides) parameter objects
        deeponet_overrides: Dict of DeepONet-specific config overrides from checkpoint
    """
    # Merge checkpoint config into config (checkpoint takes priority)
    if checkpoint_config:
        config = {**config, **checkpoint_config}

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
        n_sensors=config.get('n_sensors'),  # DeepONet sensor count
        training_stage=config.get('training_stage')  # Generative operator training stage
    )

    # Extract DeepONet-specific config overrides for architecture matching
    deeponet_overrides = {}
    deeponet_keys = ['branch_batch_norm', 'trunk_batch_norm', 'branch_layers', 'trunk_layers',
                     'branch_activation', 'trunk_activation', 'branch_dropout', 'trunk_dropout']
    for key in deeponet_keys:
        if key in config:
            deeponet_overrides[key] = config[key]

    # Store DeepONet overrides in p_md as attributes for model creation to use
    if 'deeponet' in config['arch'].lower() and deeponet_overrides:
        for key, value in deeponet_overrides.items():
            setattr(p_md, key, value)

    return p_me, p_md, p_ml, deeponet_overrides


def train_model(model_name: str, config: Dict[str, Any], args, progress_dir: Path,
                progress_file: Path, progress: Dict):
    """Train a single model on TRA data"""
    checkpoint_key = f"{model_name}_tra"
    checkpoint_path = progress_dir / 'checkpoints' / f"{checkpoint_key}.pt"

    # Check if already trained
    if progress['training'].get(checkpoint_key) == 'complete' and checkpoint_path.exists():
        print(f"  ✅ {model_name.upper()}: Already trained")
        return True

    print(f"  🔄 {model_name.upper()}: Training...")

    try:
        # Check for model-specific sequence length
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

        # Create params
        p_d = DataParams(
            batch=args.batch_size,
            augmentations=["normalize"],
            sequenceLength=seq_len,
            randSeqOffset=True,
            dataSize=[128, 64],  # Match autoreg reference
            dimension=2,
            simFields=TRA_CONFIG['sim_fields'],
            simParams=TRA_CONFIG['sim_params'],
            normalizeMode=TRA_CONFIG['normalize_mode']
        )

        p_t = TrainingParams(epochs=args.epochs, lr=0.0001, expLrGamma=0.995)

        # Configure loss params based on flags
        loss_kwargs = {'recMSE': 0.0, 'predMSE': 1.0}

        if args.use_lsim:
            loss_kwargs['predLSIM'] = 1.0
            print(f"     Using LSIM loss (enabled)")
        else:
            print(f"     LSIM loss disabled (faster CPU training)")

        if args.use_tno_loss:
            loss_kwargs['tno_lp_loss'] = 1.0  # Weight for TNO relative L2 loss
            print(f"     Using TNO relative L2 loss (enabled)")

        p_l = LossParams(**loss_kwargs)

        # Create model params using helper function
        p_me, p_md, p_ml, deeponet_overrides = create_model_params(config)

        # Log encoder/latent setup if present
        if p_me:
            print(f"     Encoder: skip, latentSize={config.get('latent_size', 32)}")
        if p_ml:
            print(f"     Latent: transformerMGN")

        # Create model (CPU only for local testing)
        use_gpu = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if use_gpu else "CPU"
        print(f"     Device: {device_name}")

        # Handle pretrained prior loading for NO+DM models
        pretrain_path = ""
        if config.get('load_pretrained_prior') and config.get('prior_checkpoint_key'):
            prior_checkpoint = progress_dir / 'checkpoints' / f"{config['prior_checkpoint_key']}.pt"
            if prior_checkpoint.exists():
                pretrain_path = str(prior_checkpoint)
                print(f"     Loading pretrained prior from: {prior_checkpoint.name}")
                # Set pretrained flag in model params
                p_md.pretrained = True
            else:
                print(f"     Warning: Pretrained prior not found: {prior_checkpoint.name}")
                print(f"     Continuing with random initialization")

        model = PredictionModel(p_d, p_t, p_l, p_me, p_md, p_ml, pretrain_path, useGPU=use_gpu)

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
            # Tensorboard/protobuf compatibility issue - create dummy writer
            print(f"     Warning: TensorBoard unavailable ({type(e).__name__}), using dummy writer")
            class DummySummaryWriter:
                def __init__(self, *args, **kwargs): pass
                def add_scalar(self, *args, **kwargs): pass
                def add_image(self, *args, **kwargs): pass
                def flush(self, *args, **kwargs): pass
                def close(self, *args, **kwargs): pass
            writer = DummySummaryWriter()

        optimizer = torch.optim.Adam(model.parameters(), lr=p_t.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=5,
            factor=p_t.expLrGamma
        )
        criterion = PredictionLoss(p_l, p_d.dimension, p_d.simFields, useGPU=use_gpu)

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
            min_epoch_for_scheduler=50
        )

        print(f"     Training {p_t.epochs} epochs on {len(dataset)} samples using Trainer class...")

        # Training loop using Trainer.trainingStep()
        for epoch in range(p_t.epochs):
            trainer.trainingStep(epoch)

        # Save checkpoint with enhanced config for architecture reproduction
        enhanced_config = config.copy()

        # For DeepONet models, save the actual architecture config used
        if 'deeponet' in config['arch'].lower():
            # Extract the actual DeepONetConfig that was used
            # The deeponet_overrides from create_model_params contains the explicit settings
            enhanced_config.update(deeponet_overrides)

        # For NO+DM models, also capture prior architecture details if available
        if config.get('load_pretrained_prior') and config.get('prior_checkpoint_key'):
            prior_checkpoint_path = progress_dir / 'checkpoints' / f"{config['prior_checkpoint_key']}.pt"
            if prior_checkpoint_path.exists():
                try:
                    prior_checkpoint = torch.load(prior_checkpoint_path, map_location='cpu', weights_only=False)
                    prior_config = prior_checkpoint.get('config', {})
                    # Merge prior architecture details (e.g., DeepONet BatchNorm flags)
                    # Only copy architecture-specific keys, not training params
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

        print(f"     ✅ Complete! Checkpoint saved.")

        # Cleanup
        train_history.cleanup()  # Stop Rich progress bar
        del model, optimizer, dataset, train_loader
        if use_gpu:
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


def sample_model(model_name: str, config: Dict[str, Any], args, progress_dir: Path,
                progress_file: Path, progress: Dict):
    """Generate predictions from a trained model"""
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
        # Create test dataset (same as training but different data split)
        seq_len = config.get('sequence_length', [2, 2])
        test_dataset = TurbulenceDataset(
            name=f"TRA_test_{model_name}",
            dataDirs=["data"],
            filterTop=TRA_CONFIG['filter_top'],
            filterSim=[(0, 3)],  # Different sims for testing
            filterFrame=[(500, 750)],  # Different frames for testing
            sequenceLength=[[60, 2]],  # Longer sequences for testing
            randSeqOffset=False,
            simFields=TRA_CONFIG['sim_fields'],
            simParams=TRA_CONFIG['sim_params'],
            printLevel="none"
        )

        # Create params matching training
        p_d = DataParams(
            batch=1,  # Batch size 1 for sampling
            augmentations=["normalize"],
            sequenceLength=[60, 2],
            randSeqOffset=False,
            dataSize=[128, 64],
            dimension=2,
            simFields=TRA_CONFIG['sim_fields'],
            simParams=TRA_CONFIG['sim_params'],
            normalizeMode=TRA_CONFIG['normalize_mode']
        )

        p_t = TrainingParams(epochs=1, lr=0.0001)  # Not used for sampling
        p_l = LossParams(recMSE=0.0, predMSE=1.0)  # Not used for sampling

        # Load checkpoint first to extract saved config
        use_gpu = torch.cuda.is_available()
        checkpoint = torch.load(checkpoint_path, map_location='cpu' if not use_gpu else None, weights_only=False)
        checkpoint_config = checkpoint.get('config', {})

        # Create model params with checkpoint config for exact architecture matching
        p_me, p_md, p_ml, deeponet_overrides = create_model_params(config, checkpoint_config)

        # Create model with architecture matching checkpoint
        model = PredictionModel(p_d, p_t, p_l, p_me, p_md, p_ml, "", useGPU=use_gpu)

        # Try to load state dict - handle architecture mismatches gracefully
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"     Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(f"     ⚠️  Architecture mismatch (skipping): {str(e)[:100]}...")
                return False
            raise

        # Apply transforms and create loader
        transforms = Transforms(p_d)
        test_dataset.transform = transforms
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

        # Create Tester
        from src.core.training.trainer import Tester
        from src.core.training.loss_history import LossHistory

        # Try to import SummaryWriter, fall back to dummy if tensorboard has issues
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=str(progress_dir / 'logs' / f"{model_name}_test"))
        except (ImportError, TypeError):
            class DummySummaryWriter:
                def __init__(self, *args, **kwargs): pass
                def add_scalar(self, *args, **kwargs): pass
                def add_image(self, *args, **kwargs): pass
                def flush(self, *args, **kwargs): pass
                def close(self, *args, **kwargs): pass
            writer = DummySummaryWriter()

        criterion = PredictionLoss(p_l, p_d.dimension, p_d.simFields, useGPU=use_gpu)
        test_history = LossHistory(
            "_test", "Testing", writer, len(test_loader),
            0, 1, printInterval=0, logInterval=0, simFields=p_d.simFields
        )

        tester = Tester(model, test_loader, criterion, test_history, p_t)

        # Generate predictions
        print(f"     Generating predictions on {len(test_dataset)} test samples...")
        predictions = tester.generatePredictions(output_path=str(sample_output_path), model_name=model_name.upper())

        # Save ground truth data for plotting (only once, not per model)
        sample_dir = progress_dir / 'sampling'
        ground_truth_path = sample_dir / 'groundTruth.dict'
        if not ground_truth_path.exists():
            print(f"     Saving ground truth data for plotting...")
            # Collect ground truth from test loader
            all_ground_truth = []
            with torch.no_grad():
                for sample in test_loader:
                    data = sample["data"]  # [B, T, C, H, W]
                    all_ground_truth.append(data)

            # Concatenate and save
            ground_truth_tensor = torch.cat(all_ground_truth, dim=0)  # [N, T, C, H, W]
            torch.save({"data": ground_truth_tensor}, ground_truth_path)
            print(f"     Saved ground truth: {ground_truth_tensor.shape}")

        # Update progress
        if 'sampling' not in progress:
            progress['sampling'] = {}
        progress['sampling'][checkpoint_key] = 'complete'
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

        print(f"     ✅ Complete! Saved predictions: {predictions.shape}")

        # Cleanup
        del model, test_dataset, test_loader
        if use_gpu:
            torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"     ❌ Failed: {str(e)[:200]}")
        import traceback
        traceback.print_exc()
        return False


def plot_models(args, progress_dir: Path, models_to_plot: List[str]):
    """Generate plots using plot_data_groups.py"""
    print("\n" + "=" * 60)
    print("📊 PLOTTING PHASE")
    print("=" * 60)

    # Determine which groups to plot
    if args.plot_groups:
        groups = args.plot_groups
    else:
        # Default: plot all comparison groups
        groups = ['neural_operators', 'no_dm', 'fno_comparison', 'tno_comparison',
                  'unet_comparison', 'deeponet_comparison']

    print(f"\nPlotting groups: {', '.join(groups)}")

    # Call plot_data_groups.py
    import subprocess
    import sys

    plot_script = Path("src/analysis/plot_data_groups.py")
    if not plot_script.exists():
        print(f"❌ Plot script not found: {plot_script}")
        return False

    # Run plotting script
    cmd = [
        sys.executable, str(plot_script),
        '--groups'] + groups + [
        '--dataset', 'zInterp',  # Can be made configurable
        '--output-folder', 'local_progress/plots',
        '--prediction-folder', 'local_progress/sampling'
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("\n✅ Plotting complete!")
        print(result.stdout)
        return True
    else:
        print("\n❌ Plotting failed!")
        print(result.stderr)
        return False


def compute_model_metrics(model_name: str, config: Dict[str, Any], progress_dir: Path):
    """
    Compute accuracy metrics for a trained model.

    Args:
        model_name: Name of the model
        config: Model configuration dictionary
        progress_dir: Directory containing checkpoints and samples

    Returns:
        Dictionary with metrics: {mse, lsim, time, size_mb} or None if failed
    """
    import time
    import numpy as np

    checkpoint_key = f"{model_name}_tra"
    checkpoint_path = progress_dir / 'checkpoints' / f"{checkpoint_key}.pt"
    sample_path = progress_dir / 'sampling' / f"{checkpoint_key}.npz"

    # Check if model and samples exist
    if not checkpoint_path.exists() or not sample_path.exists():
        return None

    try:
        # Get checkpoint size
        size_mb = checkpoint_path.stat().st_size / (1024 ** 2)

        # Load test data for ground truth comparison
        test_dataset = TurbulenceDataset(
            name=f"TRA_test_{model_name}",
            dataDirs=["data"],
            filterTop=TRA_CONFIG['filter_top'],
            filterSim=[(0, 3)],
            filterFrame=[(500, 750)],
            sequenceLength=[[60, 2]],
            randSeqOffset=False,
            simFields=TRA_CONFIG['sim_fields'],
            simParams=TRA_CONFIG['sim_params'],
            printLevel="none"
        )

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

        transforms = Transforms(p_d)
        test_dataset.transform = transforms

        # Load predictions
        predictions = np.load(sample_path)['arr_0']

        # Compute MSE and LSIM on a subset (first 10 sequences)
        num_samples = min(10, len(test_dataset), predictions.shape[0])
        mse_scores = []
        lsim_scores = []

        # Create loss criterion for metric computation
        p_l = LossParams(recMSE=0.0, predMSE=1.0)
        use_gpu = torch.cuda.is_available()
        criterion = PredictionLoss(p_l, p_d.dimension, p_d.simFields, useGPU=use_gpu)

        for i in range(num_samples):
            # Get ground truth
            sample = test_dataset[i]
            gt_data = sample['data']
            if isinstance(gt_data, np.ndarray):
                gt = torch.from_numpy(gt_data).unsqueeze(0)
            else:
                gt = gt_data.unsqueeze(0)

            # Get prediction
            pred = torch.from_numpy(predictions[i:i+1])

            # Compute MSE
            mse = torch.nn.functional.mse_loss(pred, gt).item()
            mse_scores.append(mse)

            # Compute LSIM if 2D
            if p_d.dimension == 2:
                numFields = p_d.dimension + len(p_d.simFields)
                with torch.no_grad():
                    from src.core.training.loss import loss_lsim
                    lsim = loss_lsim(criterion.lsim, pred[:,:,0:numFields], gt[:,:,0:numFields]).mean().item()
                    lsim_scores.append(lsim)

        avg_mse = np.mean(mse_scores)
        avg_lsim = np.mean(lsim_scores) if lsim_scores else 0.0

        # Estimate inference time (load model and run single prediction)
        # Load checkpoint first to extract saved config
        checkpoint = torch.load(checkpoint_path, map_location='cpu' if not use_gpu else None, weights_only=False)
        checkpoint_config = checkpoint.get('config', {})

        # Create model params with checkpoint config for exact architecture matching
        p_me, p_md, p_ml, deeponet_overrides = create_model_params(config, checkpoint_config)
        p_t = TrainingParams(epochs=1, lr=0.0001)

        model = PredictionModel(p_d, p_t, p_l, p_me, p_md, p_ml, "", useGPU=use_gpu)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()

        # Time single forward pass
        sample = test_dataset[0]
        data = torch.from_numpy(sample['data']) if isinstance(sample['data'], np.ndarray) else sample['data']
        data = data.unsqueeze(0)
        simParams = torch.from_numpy(sample['simParameters']) if isinstance(sample['simParameters'], np.ndarray) else sample['simParameters']
        simParams = simParams.unsqueeze(0)

        with torch.no_grad():
            start = time.time()
            _ = model(data, simParams, useLatent=False)
            inference_time = time.time() - start

        # Cleanup
        del model, test_dataset
        if use_gpu:
            torch.cuda.empty_cache()

        return {
            'mse': avg_mse,
            'lsim': avg_lsim,
            'time': inference_time,
            'size_mb': size_mb
        }

    except Exception as e:
        print(f"     Warning: Failed to compute metrics for {model_name}: {str(e)[:100]}")
        return None


def display_comparison_table(selected_models: List[str], progress_dir: Path):
    """
    Display comparison table of all models ranked by accuracy.

    Args:
        selected_models: List of model names to compare
        progress_dir: Directory containing checkpoints and samples
    """
    print("\n" + "=" * 60)
    print("📊 MODEL PERFORMANCE COMPARISON")
    print("=" * 60)

    # Collect metrics for all models
    results = []
    for model_name in selected_models:
        config = MODELS[model_name]
        metrics = compute_model_metrics(model_name, config, progress_dir)
        if metrics:
            results.append({
                'model': model_name.upper(),
                'mse': metrics['mse'],
                'lsim': metrics['lsim'],
                'time': metrics['time'],
                'size_mb': metrics['size_mb']
            })

    if not results:
        print("\n⚠️  No models available for comparison")
        return

    # Sort by MSE (lower is better)
    results.sort(key=lambda x: x['mse'])

    # Display ASCII table
    print("\n╔═════╦═════════════════╦═══════════╦═══════════╦══════════╦═══════╗")
    print("║ Rank│ Model           │ MSE ↓     │ LSIM ↓    │ Time (s) │ Size  ║")
    print("╠═════╬═════════════════╬═══════════╬═══════════╬══════════╬═══════╣")

    for rank, result in enumerate(results, 1):
        print(f"║ {rank:3d} │ {result['model']:15s} │ {result['mse']:9.6f} │ "
              f"{result['lsim']:9.6f} │ {result['time']:8.3f} │ {result['size_mb']:4.0f}MB ║")

    print("╚═════╩═════════════════╩═══════════╩═══════════╩══════════╩═══════╝")

    # Save to CSV
    csv_path = progress_dir / 'model_comparison.csv'
    import csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['rank', 'model', 'mse', 'lsim', 'time_s', 'size_mb'])
        writer.writeheader()
        for rank, result in enumerate(results, 1):
            writer.writerow({
                'rank': rank,
                'model': result['model'],
                'mse': result['mse'],
                'lsim': result['lsim'],
                'time_s': result['time'],
                'size_mb': result['size_mb']
            })

    print(f"\n✅ Comparison table saved to: {csv_path}")
    print("=" * 60)


def main():
    """Main execution function"""
    args = parse_args()

    print("=" * 60)
    print("🚀 LOCAL TRA VERIFICATION")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Epochs per model: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"LSIM loss: {'Enabled' if args.use_lsim else 'Disabled (faster)'}")
    print(f"TNO L2 loss: {'Enabled' if args.use_tno_loss else 'Disabled'}")
    print("=" * 60)

    # Download and extract data
    download_and_extract_tra_data()

    # Setup progress tracking
    progress_dir, progress_file, progress = setup_progress_tracking()

    # Get selected models
    selected_models = get_selected_models(args)
    print(f"\n📋 Selected {len(selected_models)} model(s):")
    for model_name in selected_models:
        print(f"   - {model_name}")
    print()

    # Phase 1: Training
    success_count = 0  # Initialize before conditional to avoid UnboundLocalError
    if not args.skip_training:
        print("\n" + "=" * 60)
        print("🏋️  TRAINING PHASE")
        print("=" * 60)
        for model_name in selected_models:
            print("\n" + "=" * 60)
            print(f"🔬 Model: {model_name.upper()}")
            print("=" * 60)

            config = MODELS[model_name]
            if train_model(model_name, config, args, progress_dir, progress_file, progress):
                success_count += 1

        print(f"\n✅ Training complete: {success_count}/{len(selected_models)} models")
    else:
        print("\n⏩ Skipping training phase (using existing checkpoints)")

    # Phase 2: Sampling
    if args.sample:
        print("\n" + "=" * 60)
        print("🔮 SAMPLING PHASE")
        print("=" * 60)

        # Try to import Rich for progress bar
        try:
            from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
            RICH_AVAILABLE = True
        except ImportError:
            RICH_AVAILABLE = False

        sample_count = 0
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("•"),
                TextColumn("{task.completed}/{task.total} models"),
                TimeElapsedColumn(),
            ) as progress_bar:
                task = progress_bar.add_task("Sampling models", total=len(selected_models))

                for model_name in selected_models:
                    config = MODELS[model_name]
                    if sample_model(model_name, config, args, progress_dir, progress_file, progress):
                        sample_count += 1
                    progress_bar.update(task, advance=1)
        else:
            for model_name in selected_models:
                config = MODELS[model_name]
                if sample_model(model_name, config, args, progress_dir, progress_file, progress):
                    sample_count += 1

        print(f"\n✅ Sampling complete: {sample_count}/{len(selected_models)} models")

        # Display comparison table after sampling
        if sample_count > 0:
            display_comparison_table(selected_models, progress_dir)

    # Phase 3: Plotting
    if args.plot:
        plot_models(args, progress_dir, selected_models)

        # Generate rollout error plot
        print("\n" + "=" * 60)
        print("📈 ROLLOUT ERROR ANALYSIS")
        print("=" * 60)
        from src.analysis.plot_rollout_error import plot_rollout_error
        rollout_plot_path = progress_dir / 'plots' / 'rollout_error.png'
        try:
            plot_rollout_error(
                prediction_folder=progress_dir / 'sampling',
                model_names=selected_models,
                output_path=rollout_plot_path,
                metric='mse',
                title='Autoregressive Rollout Stability (MSE vs Frame)'
            )
        except Exception as e:
            print(f"  ⚠️  Rollout error plot failed: {str(e)[:100]}")

    # Summary
    print("\n" + "=" * 60)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"\nModels trained: {success_count}/{len(selected_models)}")
    print(f"\nResults:")

    # Reload progress to get final state
    with open(progress_file, 'r') as f:
        progress = json.load(f)

    for model_name in selected_models:
        checkpoint_key = f"{model_name}_tra"
        status = progress['training'].get(checkpoint_key, 'not started')
        symbol = '✅' if status == 'complete' else '❌'
        print(f"  {symbol} {model_name}: {status}")

    if success_count == len(selected_models):
        print(f"\n🎉 SUCCESS! All {success_count} model(s) trained successfully.")
    elif success_count > 0:
        print(f"\n⚠️  Partial success: {success_count}/{len(selected_models)} models completed.")
    else:
        print(f"\n❌ No models completed successfully.")

    print("=" * 60)


if __name__ == "__main__":
    main()
