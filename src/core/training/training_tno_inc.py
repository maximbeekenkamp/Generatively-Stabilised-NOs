"""
Enhanced TNO Training Configuration for Incompressible Wake Flow (Inc) Dataset
Phase 2: Full implementation with advanced features

Integrates Phase 1.1-1.3 enhancements:
- Enhanced TNOModel with teacher forcing support
- TNOTrainer with gradual phase transitions
- LpLoss2 integration for better convergence
- Multi-phase training configurations
"""

import os
import copy
from typing import Dict
import torch
import argparse
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, SubsetRandomSampler

from src.core.models.model import PredictionModel
from src.core.utils.logger import Logger
from src.core.utils.params import DataParams, TrainingParams, LossParams, ModelParamsEncoder, ModelParamsDecoder, ModelParamsLatent
from src.core.data_processing.turbulence_dataset import TurbulenceDataset
from src.core.data_processing.data_transformations import Transforms
from src.core.training.loss import PredictionLoss
from src.core.training.loss_history import LossHistory
from src.core.training.trainer import Trainer
from src.core.training.trainer_tno import TNOTrainer


def create_tno_inc_config(phase: str = "full", test_mode: bool = False):
    """
    Create TNO configuration for Inc dataset - Phase 2 Enhanced
    
    Args:
        phase: Training phase ("full", "teacher_forcing", "fine_tuning")
        test_mode: Whether to use reduced settings for testing
        
    Returns:
        Configuration dictionaries for Gen Stabilised framework
    """
    
    # Model configuration based on phase
    if phase == "full":
        modelName = "2D_Inc/128_tno-phase2-L2K4"
        arch = "tno+Prev"  # L=2 teacher forcing
        epochs = 1000
        teacher_forcing_epochs = 500
        batch_size = 16
        lr = 0.001
    elif phase == "teacher_forcing":
        modelName = "2D_Inc/128_tno-teacher-only"
        arch = "tno+Prev"  # L=2 
        epochs = 800
        teacher_forcing_epochs = 10000  # Never switch
        batch_size = 12
        lr = 0.001
    else:  # fine_tuning
        modelName = "2D_Inc/128_tno-fine-tuning"
        arch = "tno"  # L=1
        epochs = 600
        teacher_forcing_epochs = 0  # Start in fine-tuning
        batch_size = 20
        lr = 0.0005
    
    # Adjust for test mode
    if test_mode:
        modelName += "-test"
        epochs = 5
        batch_size = 2
        teacher_forcing_epochs = min(teacher_forcing_epochs, 3)
    
    # Data parameters - compatible with Gen Stabilised
    p_d = DataParams(
        batch=batch_size,
        augmentations=["normalize"],
        sequenceLength=[5, 2] if not test_mode else [3, 0],  # [length, skip]
        randSeqOffset=True,
        dataSize=[128, 64] if not test_mode else [32, 32],
        dimension=2,
        simFields=["vx", "vy", "pres"],  # Velocity components + pressure
        simParams=["rey"],  # Reynolds number parameter
        normalizeMode="incMixed"  # Existing Inc normalization
    )
    
    # Training parameters
    p_t = TrainingParams(
        epochs=epochs,
        lr=lr,
        fadeInSeqLen=[-1, 0] if test_mode else [100, 400],  # Curriculum learning
        fadeInPredLoss=[50, 200] if not test_mode else [-1, 0],
        expLrGamma=0.995 if not test_mode else 1.0  # Learning rate decay
    )
    
    # Loss parameters with Phase 1.3 TNO enhancements
    p_l = LossParams(
        recMSE=0.0,  # No reconstruction for TNO
        predMSE=0.8,  # Main prediction loss
        regMeanStd=0.1,  # Regularization
        tno_lp_loss=0.2 if phase == "full" else 0.0,  # Phase 1.3: LpLoss2
        tno_transition_weight=0.1  # Phase 1.3: Gradual transitions
    )
    
    # Model parameters - TNO architecture
    p_md = ModelParamsDecoder(
        arch=arch,
        decWidth=360 if not test_mode else 64,  # TNO width parameter
        pretrained=False
    )
    
    return modelName, p_d, p_t, p_l, None, p_md, None, teacher_forcing_epochs


def run_tno_training(phase="full", test_mode=False, data_dir=None):
    """
    Run TNO training with specified configuration
    
    Args:
        phase: Training phase configuration
        test_mode: Whether to use test settings
        data_dir: Data directory override
    """
    
    useGPU = True and torch.cuda.is_available()
    gpuID = "0"
    
    print(f"\\n{'='*60}")
    print(f"Enhanced TNO Training - Inc Dataset (Phase 2)")
    print(f"Training Phase: {phase}")
    print(f"Test Mode: {test_mode}")
    print(f"GPU Available: {torch.cuda.is_available()}")
    print(f"{'='*60}\\n")
    
    # Get configuration
    modelName, p_d, p_t, p_l, p_me, p_md, p_ml, teacher_forcing_epochs = create_tno_inc_config(
        phase=phase, test_mode=test_mode
    )
    
    pretrainPath = ""
    
    # Print configuration summary
    print(f"Model: {modelName}")
    print(f"Architecture: {p_md.arch}")
    print(f"Batch size: {p_d.batch}")
    print(f"Epochs: {p_t.epochs}")
    print(f"Teacher forcing epochs: {teacher_forcing_epochs}")
    print(f"TNO LpLoss2 weight: {p_l.tno_lp_loss}")
    print()
    
    # Initialize model
    model = PredictionModel(p_d, p_t, p_l, p_me, p_md, p_ml, pretrainPath, useGPU)
    model.printModel()
    
    # Create dummy dataset for testing if no data available
    if test_mode:
        print("Creating dummy dataset for testing...")
        
        class DummyDataset:
            def __init__(self, num_samples=20):
                self.num_samples = num_samples
                
            def __len__(self):
                return self.num_samples
                
            def __getitem__(self, idx):
                # [T, C, H, W] format
                seq_len = p_d.sequenceLength[0]
                channels = len(p_d.simFields) + len(p_d.simParams)  # 3 fields + 1 param
                H, W = p_d.dataSize
                
                # Create structured dummy data
                data = torch.randn(seq_len, channels, H, W) * 0.1
                
                # Add physical structure
                data[:, 0] += 1.0    # Pressure baseline
                data[:, 1] += 0.5    # X-velocity with flow
                data[:, 2] += 0.1    # Y-velocity perturbations  
                data[:, 3] = 1000.0  # Reynolds number (constant)
                
                return {
                    "data": data,
                    "simParameters": torch.tensor([1000.0])  # Reynolds
                }
        
        trainDataset = DummyDataset(50)
        valDataset = DummyDataset(10)
        
        trainLoader = DataLoader(trainDataset, batch_size=p_d.batch, shuffle=True)
        valLoader = DataLoader(valDataset, batch_size=p_d.batch, shuffle=False)
        
    else:
        # Load real dataset
        print("Loading Inc dataset...")
        data_path = data_dir if data_dir else "data/inc"  # Default path
        
        dataTransforms = Transforms(params=p_d)
        
        trainDataset = TurbulenceDataset(
            p_d, dataTransforms, split="train", directory=data_path
        )
        valDataset = TurbulenceDataset(
            p_d, dataTransforms, split="test", directory=data_path
        )
        
        # Create data loaders
        if len(trainDataset) < 100:  # Small dataset
            trainLoader = DataLoader(trainDataset, batch_size=p_d.batch, shuffle=True)
        else:
            trainSampler = RandomSampler(trainDataset)
            trainLoader = DataLoader(trainDataset, batch_size=p_d.batch, sampler=trainSampler)
        
        valLoader = DataLoader(valDataset, batch_size=p_d.batch, shuffle=False)
    
    print(f"Training samples: {len(trainLoader.dataset)}")
    print(f"Validation samples: {len(valLoader.dataset) if hasattr(valLoader.dataset, '__len__') else 'N/A'}")
    print()
    
    # Loss and history
    criterion = PredictionLoss(p_l, p_d.dimension, p_d.simFields, useGPU)
    trainHistory = LossHistory()
    
    # Logger
    logger = Logger(model, None, useGPU, p_d, p_l)
    
    # Create enhanced TNO trainer
    if phase == "full":
        print("Using enhanced TNOTrainer with phase management...")
        trainer = TNOTrainer(
            model, trainLoader, 
            torch.optim.Adam(model.parameters(), lr=p_t.lr),
            None,  # No scheduler for now
            criterion, trainHistory, None,  # No tensorboard writer for now
            p_d, p_t,
            teacher_forcing_epochs=teacher_forcing_epochs,
            transition_epochs=50,  # Gradual transition
            phase_lr_factor=0.1    # Reduce LR for fine-tuning
        )
    else:
        print("Using standard Trainer...")
        trainer = Trainer(
            model, trainLoader, 
            torch.optim.Adam(model.parameters(), lr=p_t.lr),
            None, criterion, trainHistory, None, p_d, p_t
        )
    
    # Training loop
    print(f"Starting {phase} training...")
    for epoch in range(p_t.epochs):
        trainer.trainingStep(epoch)
        
        # Progress reporting
        if epoch % 10 == 0 or epoch < 5:
            if hasattr(trainer, 'get_training_summary'):
                summary = trainer.get_training_summary()
                print(f"Epoch {epoch:3d} - Phase: {summary.get('current_phase', 'standard')}")
            else:
                print(f"Epoch {epoch:3d} - Training step completed")
    
    print(f"\\nTraining completed!")
    
    # Print final summary
    if hasattr(trainer, 'get_training_summary'):
        summary = trainer.get_training_summary()
        print(f"Final training summary:")
        print(f"  Current phase: {summary.get('current_phase')}")
        print(f"  Phase transitions: {len(summary.get('phase_transitions', []))}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Enhanced TNO Training for Inc Dataset')
    parser.add_argument('--phase', type=str, default='full',
                       choices=['full', 'teacher_forcing', 'fine_tuning'],
                       help='Training phase configuration')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode with reduced settings')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Override data directory')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    try:
        run_tno_training(
            phase=args.phase,
            test_mode=args.test,
            data_dir=args.data_dir
        )
    except KeyboardInterrupt:
        print("\\nTraining interrupted by user")
    except Exception as e:
        print(f"\\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\\nScript completed.")