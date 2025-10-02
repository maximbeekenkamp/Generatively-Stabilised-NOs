"""
DeepONet Training Configuration and Factory

Provides training setup and configuration for DeepONet models,
integrating with the existing framework's parameter system.
"""

from typing import Dict, Optional, Any, Tuple
import logging
import torch
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from .trainer_deeponet import DeepONetTrainer
from .deeponet_loss import DeepONetLoss, create_deeponet_loss
from .loss_history import LossHistory
from ..models.deeponet import DeepONetFactory
from ..models.deeponet.data_utils import (
    create_operator_dataloaders,
    DeepONetDataConfig,
    OperatorDataLoader
)
from ..utils.params import DataParams, TrainingParams, ModelParamsDecoder


class DeepONetTrainingConfig:
    """Configuration class for DeepONet training setup."""

    def __init__(self):
        # Model configuration
        self.model_variant = 'standard'  # 'standard', 'fourier', 'physics', 'multiscale'
        self.model_config = {
            'latent_dim': 256,
            'n_sensors': 100,
            'sensor_strategy': 'uniform',
            'n_query_train': 1000,
            'branch_layers': [128, 256],
            'trunk_layers': [64, 128],
            'activation': 'gelu',
            'dropout': 0.1,
            'batch_norm': True,
            'bias': True,
            'normalize_inputs': True,
            'positional_encoding': True
        }

        # Variant-specific configurations
        self.fourier_config = {
            'branch_fourier_modes': 32,
            'trunk_fourier_modes': 64
        }

        self.physics_config = {
            'physics_type': 'general',  # 'fluid', 'heat', 'wave', 'general'
            'use_physics_loss': True
        }

        self.multiscale_config = {
            'n_scales': 3,
            'scale_factors': [1.0, 2.0, 4.0]
        }

        # Data configuration
        self.data_config = DeepONetDataConfig(
            n_sensors=100,
            sensor_strategy='uniform',
            n_query_train=1000,
            n_query_eval=None,  # Use full grid
            query_strategy='random',
            normalize_inputs=True,
            normalize_outputs=True,
            add_noise=False,
            noise_level=0.01,
            random_sensor_dropout=False,
            sensor_dropout_rate=0.1
        )

        # Loss configuration
        self.loss_config = {
            'operator_loss_type': 'relative_l2',  # 'l2', 'relative_l2', 'huber'
            'operator_weight': 1.0,
            'physics_weight': 0.1,
            'gradient_weight': 0.01,
            'sensor_reg_weight': 0.001,
            'spectral_weight': 0.0
        }

        # Optimizer configuration
        self.optimizer_type = 'adam'  # 'adam', 'adamw', 'sgd'
        self.optimizer_config = {
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'betas': (0.9, 0.999),
            'eps': 1e-8
        }

        # Scheduler configuration
        self.scheduler_type = 'step'  # 'step', 'exponential', 'cosine', 'plateau'
        self.scheduler_config = {
            'step_size': 50,
            'gamma': 0.8
        }

        # Training configuration
        self.batch_size = 32
        self.num_epochs = 200
        self.val_frequency = 5
        self.save_frequency = 20
        self.log_frequency = 10
        self.num_workers = 4
        self.gradient_clip = 1.0
        self.adaptive_sensors = False

        # Early stopping
        self.early_stopping = True
        self.patience = 20
        self.min_delta = 1e-6

    def update_from_legacy_params(self, p_md: ModelParamsDecoder, p_t: TrainingParams, p_d: DataParams):
        """Update configuration from legacy parameter system."""
        # Update model parameters
        if hasattr(p_md, 'decWidth') and p_md.decWidth > 0:
            self.model_config['latent_dim'] = p_md.decWidth
            self.model_config['branch_layers'] = [p_md.decWidth//2, p_md.decWidth]
            self.model_config['trunk_layers'] = [p_md.decWidth//4, p_md.decWidth//2]

        # Update data parameters
        if hasattr(p_d, 'dataSize') and len(p_d.dataSize) >= 2:
            H, W = p_d.dataSize[-2], p_d.dataSize[-1]
            # Adjust sensor count based on spatial resolution
            self.data_config.n_sensors = min(max(50, (H * W) // 20), 1000)
            self.model_config['n_sensors'] = self.data_config.n_sensors

        # Update training parameters
        if hasattr(p_t, 'numEpochs'):
            self.num_epochs = p_t.numEpochs[0]

        if hasattr(p_t, 'batchSize'):
            self.batch_size = p_t.batchSize[0]

        if hasattr(p_t, 'learningRate'):
            self.optimizer_config['lr'] = p_t.learningRate[0]


def create_deeponet_training_setup(
    train_data: Tuple[torch.Tensor, torch.Tensor],
    val_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    test_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    config: Optional[DeepONetTrainingConfig] = None,
    p_md: Optional[ModelParamsDecoder] = None,
    p_t: Optional[TrainingParams] = None,
    p_d: Optional[DataParams] = None,
    device: str = 'cuda',
    log_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create complete DeepONet training setup.

    Args:
        train_data: Tuple of (input_functions, output_functions)
        val_data: Optional validation data tuple
        test_data: Optional test data tuple
        config: Training configuration
        p_md: Legacy model parameters
        p_t: Legacy training parameters
        p_d: Legacy data parameters
        device: Device for training
        log_dir: Directory for tensorboard logs

    Returns:
        training_setup: Dictionary with all training components
    """
    if config is None:
        config = DeepONetTrainingConfig()

    # Update configuration from legacy parameters if provided
    if p_md is not None and p_t is not None and p_d is not None:
        config.update_from_legacy_params(p_md, p_t, p_d)

    logging.info("Setting up DeepONet training pipeline")

    # Create data loaders
    train_input, train_output = train_data
    dataloaders = create_operator_dataloaders(
        train_input=train_input,
        train_output=train_output,
        val_input=val_data[0] if val_data else None,
        val_output=val_data[1] if val_data else None,
        test_input=test_data[0] if test_data else None,
        test_output=test_data[1] if test_data else None,
        config=config.data_config,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )

    # Create model
    model_config = config.model_config.copy()

    # Add variant-specific configuration
    if config.model_variant == 'fourier':
        model_config.update(config.fourier_config)
    elif config.model_variant == 'physics':
        model_config.update(config.physics_config)
    elif config.model_variant == 'multiscale':
        model_config.update(config.multiscale_config)

    model = DeepONetFactory.create(
        variant=config.model_variant,
        config=model_config,
        p_md=p_md,
        p_d=p_d
    )

    # Move model to device
    model = model.to(device)
    model.useGPU = (device == 'cuda')

    logging.info(f"Created {config.model_variant} DeepONet with {model.get_parameter_count():,} parameters")

    # Create optimizer
    optimizer = create_optimizer(model, config.optimizer_type, config.optimizer_config)

    # Create learning rate scheduler
    scheduler = create_scheduler(optimizer, config.scheduler_type, config.scheduler_config)

    # Create loss function
    criterion = create_deeponet_loss(config.loss_config)

    # Create training history
    train_history = LossHistory()

    # Create tensorboard writer
    writer = None
    if log_dir is not None:
        writer = SummaryWriter(log_dir=log_dir)

    # Create trainer
    trainer = DeepONetTrainer(
        model=model,
        trainLoader=dataloaders['train'],
        optimizer=optimizer,
        lrScheduler=scheduler,
        criterion=criterion,
        trainHistory=train_history,
        writer=writer,
        p_d=p_d or DataParams(),  # Provide default if None
        p_t=p_t or TrainingParams(),  # Provide default if None
        valLoader=dataloaders.get('val'),
        adaptive_sensors=config.adaptive_sensors
    )

    training_setup = {
        'model': model,
        'trainer': trainer,
        'dataloaders': dataloaders,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'criterion': criterion,
        'train_history': train_history,
        'writer': writer,
        'config': config
    }

    return training_setup


def create_optimizer(model: torch.nn.Module, optimizer_type: str, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Create optimizer for DeepONet training."""
    if optimizer_type.lower() == 'adam':
        return Adam(model.parameters(), **config)
    elif optimizer_type.lower() == 'adamw':
        return AdamW(model.parameters(), **config)
    elif optimizer_type.lower() == 'sgd':
        # Adjust config for SGD
        sgd_config = config.copy()
        sgd_config.pop('betas', None)
        sgd_config.pop('eps', None)
        sgd_config['momentum'] = sgd_config.get('momentum', 0.9)
        return SGD(model.parameters(), **sgd_config)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(optimizer: torch.optim.Optimizer, scheduler_type: str, config: Dict[str, Any]):
    """Create learning rate scheduler for DeepONet training."""
    if scheduler_type.lower() == 'step':
        return StepLR(optimizer, **config)
    elif scheduler_type.lower() == 'exponential':
        return ExponentialLR(optimizer, **config)
    elif scheduler_type.lower() == 'cosine':
        return CosineAnnealingLR(optimizer, **config)
    elif scheduler_type.lower() == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='min', **config)
    elif scheduler_type.lower() == 'none':
        return None
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def train_deeponet(
    train_data: Tuple[torch.Tensor, torch.Tensor],
    val_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    config: Optional[DeepONetTrainingConfig] = None,
    save_path: Optional[str] = None,
    device: str = 'cuda',
    log_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    High-level function to train a DeepONet model.

    Args:
        train_data: Training data tuple (input_functions, output_functions)
        val_data: Optional validation data tuple
        config: Training configuration
        save_path: Path to save trained model
        device: Device for training
        log_dir: Directory for tensorboard logs

    Returns:
        results: Dictionary with training results and model
    """
    if config is None:
        config = DeepONetTrainingConfig()

    # Create training setup
    setup = create_deeponet_training_setup(
        train_data=train_data,
        val_data=val_data,
        config=config,
        device=device,
        log_dir=log_dir
    )

    model = setup['model']
    trainer = setup['trainer']
    train_history = setup['train_history']

    logging.info(f"Starting DeepONet training for {config.num_epochs} epochs")

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config.num_epochs):
        # Training step
        trainer.trainingStep(epoch)

        # Validation step
        if val_data is not None and epoch % config.val_frequency == 0:
            trainer.validationStep(epoch)

            # Early stopping check
            if config.early_stopping:
                current_val_loss = train_history.get_latest_validation_loss()
                if current_val_loss is not None:
                    if current_val_loss < best_val_loss - config.min_delta:
                        best_val_loss = current_val_loss
                        patience_counter = 0

                        # Save best model
                        if save_path:
                            trainer.save_checkpoint(f"{save_path}_best.pth", epoch)
                    else:
                        patience_counter += 1

                    if patience_counter >= config.patience:
                        logging.info(f"Early stopping at epoch {epoch}")
                        break

        # Periodic checkpoint saving
        if save_path and epoch % config.save_frequency == 0:
            trainer.save_checkpoint(f"{save_path}_epoch_{epoch}.pth", epoch)

    # Save final model
    if save_path:
        trainer.save_checkpoint(f"{save_path}_final.pth", config.num_epochs)

    # Close writer
    if setup['writer']:
        setup['writer'].close()

    logging.info("DeepONet training completed")

    return {
        'model': model,
        'trainer': trainer,
        'train_history': train_history,
        'config': config,
        'best_val_loss': best_val_loss
    }