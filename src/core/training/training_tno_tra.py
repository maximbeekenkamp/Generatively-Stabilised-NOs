"""
Enhanced TNO Training Configuration for Transonic Flow (Tra) Dataset
Phase 2: Full implementation with advanced features

Integrates Phase 1.1-1.3 enhancements:
- Enhanced TNOModel with teacher forcing support
- TNOTrainer with gradual phase transitions
- LpLoss2 integration for better convergence
- Multi-phase training configurations
- Tra-specific field mappings and optimizations
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


def create_tno_tra_config(phase: str = "full", test_mode: bool = False):
    """
    Create TNO configuration for Tra dataset - Phase 2 Enhanced
    
    Args:
        phase: Training phase ("full", "teacher_forcing", "fine_tuning")
        test_mode: Whether to use reduced settings for testing
        
    Returns:
        Configuration dictionaries for Gen Stabilised framework
    """
    
    # Model configuration based on phase
    if phase == "full":
        modelName = "2D_Tra/128_tno-phase2-L2K4"
        arch = "tno+Prev"  # L=2 teacher forcing
        epochs = 1200
        teacher_forcing_epochs = 600
        batch_size = 16
        lr = 0.001
    elif phase == "teacher_forcing":
        modelName = "2D_Tra/128_tno-teacher-only"
        arch = "tno+Prev"  # L=2 
        epochs = 1000
        teacher_forcing_epochs = 10000  # Never switch
        batch_size = 12
        lr = 0.001
    else:  # fine_tuning
        modelName = "2D_Tra/128_tno-fine-tuning"
        arch = "tno"  # L=1
        epochs = 800
        teacher_forcing_epochs = 0  # Start in fine-tuning
        batch_size = 20
        lr = 0.0005
    
    # Adjust for test mode
    if test_mode:
        modelName += "-test"
        epochs = 5
        batch_size = 2
        teacher_forcing_epochs = min(teacher_forcing_epochs, 3)
    
    # Data parameters - Tra-specific fields and normalization
    p_d = DataParams(
        batch=batch_size,
        augmentations=["normalize"],
        sequenceLength=[4, 2] if not test_mode else [3, 0],  # Shorter for Tra stability
        randSeqOffset=True,
        dataSize=[128, 64] if not test_mode else [32, 32],
        dimension=2,
        simFields=["dens", "pres"],  # Density and pressure for Tra
        simParams=["mach"],  # Mach number parameter
        normalizeMode="machMixed"  # Existing Tra normalization
    )
    
    # Training parameters - Adjusted for Tra dataset
    p_t = TrainingParams(
        epochs=epochs,
        lr=lr,
        fadeInSeqLen=[-1, 0] if test_mode else [150, 500],  # Longer for Tra complexity
        fadeInPredLoss=[75, 250] if not test_mode else [-1, 0],
        expLrGamma=0.99 if not test_mode else 1.0  # Slower decay for Tra
    )
    
    # Loss parameters with Phase 1.3 TNO enhancements
    p_l = LossParams(
        recMSE=0.0,  # No reconstruction for TNO
        predMSE=0.85,  # Slightly higher weight for Tra prediction
        regMeanStd=0.15,  # Higher regularization for Tra
        tno_lp_loss=0.15 if phase == "full" else 0.0,  # Phase 1.3: LpLoss2
        tno_transition_weight=0.12  # Phase 1.3: Gradual transitions
    )
    
    # Model parameters - TNO architecture optimized for Tra
    p_md = ModelParamsDecoder(
        arch=arch,
        decWidth=400 if not test_mode else 64,  # Wider for Tra complexity
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
    
    print(f"\n{'='*60}")
    print(f"Enhanced TNO Training - Tra Dataset (Phase 2)")
    print(f"Training Phase: {phase}")
    print(f"Test Mode: {test_mode}")
    print(f"GPU Available: {torch.cuda.is_available()}")
    print(f"{'='*60}\n")
    
    # Get configuration
    modelName, p_d, p_t, p_l, p_me, p_md, p_ml, teacher_forcing_epochs = create_tno_tra_config(
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
    print(f"Fields: {p_d.simFields}")
    print(f"Mach parameter: {p_d.simParams}")
    print()
    
    # Initialize model
    model = PredictionModel(p_d, p_t, p_l, p_me, p_md, p_ml, pretrainPath, useGPU)
    model.printModel()
    
    # Create dummy dataset for testing if no data available
    if test_mode:
        print("Creating dummy Tra dataset for testing...")
        
        class DummyTraDataset:
            def __init__(self, num_samples=20):
                self.num_samples = num_samples
                
            def __len__(self):
                return self.num_samples
                
            def __getitem__(self, idx):
                # [T, C, H, W] format
                seq_len = p_d.sequenceLength[0]
                channels = len(p_d.simFields) + len(p_d.simParams)  # 2 fields + 1 param
                H, W = p_d.dataSize
                
                # Create structured dummy data for Tra
                data = torch.randn(seq_len, channels, H, W) * 0.1
                
                # Add Tra-specific physical structure
                data[:, 0] += 2.0    # Density baseline (higher for compressible)
                data[:, 1] += 3.0    # Pressure baseline (higher for transonic)
                data[:, 2] = 0.8     # Mach number (transonic range)
                
                return {
                    "data": data,
                    "simParameters": torch.tensor([0.8])  # Mach number
                }
        
        trainDataset = DummyTraDataset(50)
        valDataset = DummyTraDataset(10)
        
        trainLoader = DataLoader(trainDataset, batch_size=p_d.batch, shuffle=True)
        valLoader = DataLoader(valDataset, batch_size=p_d.batch, shuffle=False)
        
    else:
        # Load real Tra dataset
        print("Loading Tra dataset...")
        
        # Use existing Tra dataset filter patterns from training_tra.py
        trainSet = TurbulenceDataset("Training", ["data"], filterTop=["128_tra"], 
                                   filterSim=[[0,1,2,14,15,16,17,18]], excludefilterSim=True, 
                                   filterFrame=[(0,1000)],
                                   sequenceLength=[p_d.sequenceLength], randSeqOffset=p_d.randSeqOffset, 
                                   simFields=p_d.simFields, simParams=p_d.simParams, printLevel="sim")
        
        # Create test sets for evaluation
        testSets = {
            "extrap": TurbulenceDataset("Test Extrapolate Mach 0.50-0.52", ["data"], 
                                      filterTop=["128_tra"], filterSim=[(0,3)],
                                      filterFrame=[(500,750)], sequenceLength=[[60,2]], 
                                      simFields=p_d.simFields, simParams=p_d.simParams, printLevel="sim"),
            "interp": TurbulenceDataset("Test Interpolate Mach 0.66-0.68", ["data"], 
                                      filterTop=["128_tra"], filterSim=[(16,19)],
                                      filterFrame=[(500,750)], sequenceLength=[[60,2]], 
                                      simFields=p_d.simFields, simParams=p_d.simParams, printLevel="sim"),
        }
        
        # Create data transforms
        transTrain = Transforms(p_d)
        trainSet.transform = transTrain
        trainSet.printDatasetInfo()
        
        # Create data loaders
        trainSampler = RandomSampler(trainSet)
        trainLoader = DataLoader(trainSet, sampler=trainSampler,
                               batch_size=p_d.batch, drop_last=True, num_workers=4)
        
        # For simplicity, create a validation set from the training data
        valLoader = DataLoader(trainSet, batch_size=p_d.batch, shuffle=False, 
                             num_workers=2, sampler=SubsetRandomSampler(range(min(100, len(trainSet)))))
    
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
            transition_epochs=75,  # Gradual transition for Tra
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
    print(f"Starting {phase} training for Tra dataset...")
    for epoch in range(p_t.epochs):
        trainer.trainingStep(epoch)
        
        # Progress reporting
        if epoch % 15 == 0 or epoch < 5:  # Report every 15 epochs for Tra
            if hasattr(trainer, 'get_training_summary'):
                summary = trainer.get_training_summary()
                print(f"Epoch {epoch:3d} - Phase: {summary.get('current_phase', 'standard')}")
            else:
                print(f"Epoch {epoch:3d} - Training step completed")
    
    print(f"\nTra training completed!")
    
    # Print final summary
    if hasattr(trainer, 'get_training_summary'):
        summary = trainer.get_training_summary()
        print(f"Final training summary:")
        print(f"  Current phase: {summary.get('current_phase')}")
        print(f"  Phase transitions: {len(summary.get('phase_transitions', []))}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Enhanced TNO Training for Tra Dataset')
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
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nScript completed.")