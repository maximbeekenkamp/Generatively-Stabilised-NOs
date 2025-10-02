# Training modules and configurations

from .trainer import Trainer
from .trainer_deeponet import DeepONetTrainer
from .deeponet_loss import DeepONetLoss, create_deeponet_loss
from .training_deeponet import (
    DeepONetTrainingConfig,
    create_deeponet_training_setup,
    train_deeponet
)
from .loss import PredictionLoss
from .loss_history import LossHistory

__all__ = [
    # Base training
    'Trainer',
    'PredictionLoss',
    'LossHistory',

    # DeepONet training
    'DeepONetTrainer',
    'DeepONetLoss',
    'create_deeponet_loss',
    'DeepONetTrainingConfig',
    'create_deeponet_training_setup',
    'train_deeponet'
]