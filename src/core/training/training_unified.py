#!/usr/bin/env python3
"""
Unified Training Script

This script replaces the separate training_inc.py, training_iso.py, and training_tra.py
files with a single configurable training script.

Usage:
    python training_unified.py --dataset inc --architecture unet
    python training_unified.py --dataset iso --architecture unet --variant unrolled --unroll_length 8
    python training_unified.py --dataset tra --architecture unet --variant noise --noise_level 0.01
"""

import argparse
import copy
from typing import Dict

# Use consolidated imports
from src.core.utils.common_imports import *
from src.core.utils.ml_imports import torch, DataLoader, SequentialSampler, RandomSampler
from src.core.utils.environment_setup import initialize_environment

from src.core.models.model import PredictionModel
from src.core.utils.logger import Logger
from src.core.data_processing.dataset_factory import DatasetFactory
from src.core.data_processing.data_transformations import Transforms
from src.core.training.loss import PredictionLoss
from src.core.training.loss_history import LossHistory
from src.core.training.trainer import Trainer, Tester
from src.core.training.training_config import TrainingConfigs, ConfigVariants, DatasetConfig


class UnifiedTrainer:
    """Unified training class that handles all dataset types."""

    def __init__(self, config: DatasetConfig, use_gpu: bool = True, gpu_id: str = "0"):
        self.config = config
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id

    def create_datasets(self):
        """Create training and test datasets using the dataset factory."""
        factory = DatasetFactory()

        # Create training dataset
        train_set = factory.create_training_dataset(
            self.config.name,
            sequence_length=[self.config.data_params.sequenceLength],
            sim_fields=self.config.data_params.simFields,
            sim_params=self.config.data_params.simParams
        )

        # Create test datasets
        test_sets = factory.create_test_datasets(
            self.config.name,
            sim_fields=self.config.data_params.simFields,
            sim_params=self.config.data_params.simParams
        )

        return train_set, test_sets

    def train(self):
        """Main training function."""
        # GPU setup
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_id

        # Create datasets
        train_set, test_sets = self.create_datasets()

        # Setup logging and model
        logger = Logger(self.config.model_name, addNumber=True)
        model = PredictionModel(
            self.config.data_params,
            self.config.training_params,
            self.config.loss_params,
            self.config.encoder_params,
            self.config.decoder_params,
            self.config.latent_params,
            self.config.pretrain_path,
            self.use_gpu
        )
        model.printModelInfo()

        # Training setup
        criterion = PredictionLoss(
            self.config.loss_params,
            self.config.data_params.dimension,
            self.config.data_params.simFields,
            self.use_gpu
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.training_params.lr,
            weight_decay=self.config.training_params.weightDecay
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=5,
            factor=self.config.training_params.expLrGamma
        )
        logger.setup(model, optimizer)

        # Data loading setup
        trans_train = Transforms(self.config.data_params)
        train_set.transform = trans_train
        train_set.printDatasetInfo()
        train_sampler = RandomSampler(train_set)
        train_loader = DataLoader(
            train_set,
            sampler=train_sampler,
            batch_size=self.config.data_params.batch,
            drop_last=True,
            num_workers=4
        )

        # Training history
        train_history = LossHistory(
            "_train", "Training", logger.tfWriter, len(train_loader),
            0, 1, printInterval=1, logInterval=1,
            simFields=self.config.data_params.simFields,
            use_rich_progress=True, total_epochs=self.config.training_params.epochs
        )

        # Trainer
        trainer = Trainer(
            model, train_loader, optimizer, lr_scheduler, criterion,
            train_history, logger.tfWriter,
            self.config.data_params, self.config.training_params,
            min_epoch_for_scheduler=50
        )

        # Test setup
        testers = []
        test_histories = []
        for short_name, test_set in test_sets.items():
            p_d_test = copy.deepcopy(self.config.data_params)
            p_d_test.augmentations = ["normalize"]

            trans_test = Transforms(p_d_test)
            test_set.transform = trans_test
            test_sampler = SequentialSampler(test_set)
            test_loader = DataLoader(
                test_set,
                sampler=test_sampler,
                batch_size=p_d_test.batch,
                drop_last=True,
                num_workers=4
            )

            test_history = LossHistory(
                f"_test_{short_name}", f"Test {short_name}", logger.tfWriter,
                len(test_loader), 0, self.config.training_params.testInterval,
                printInterval=1, logInterval=1,
                simFields=self.config.data_params.simFields
            )

            tester = Tester(model, test_loader, criterion, test_history, logger.tfWriter, p_d_test)
            testers.append(tester)
            test_histories.append(test_history)

        # Run training
        for epoch in range(self.config.training_params.epochs):
            trainer.doTraining(epoch)

            if epoch % self.config.training_params.testInterval == 0:
                for tester in testers:
                    tester.doTesting(epoch)

            if epoch % self.config.training_params.saveInterval == 0:
                logger.saveModel(model, epoch)

        # Final save
        logger.saveModel(model, self.config.training_params.epochs)
        print(f"Training completed for {self.config.name} dataset!")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Unified training script for all dataset types")

    parser.add_argument("--dataset", type=str, required=True, choices=["inc", "iso", "tra"],
                        help="Dataset type to train on")
    parser.add_argument("--architecture", type=str, default="unet", choices=["unet", "resnet", "fno"],
                        help="Neural network architecture")
    parser.add_argument("--variant", type=str, default="standard",
                        choices=["standard", "unrolled", "noise"],
                        help="Training variant")
    parser.add_argument("--unroll_length", type=int, default=8,
                        help="Sequence length for unrolled training")
    parser.add_argument("--noise_level", type=float, default=0.01,
                        help="Training noise level")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID to use")
    parser.add_argument("--no_gpu", action="store_true", help="Disable GPU usage")

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()

    # Get base configuration
    base_config = TrainingConfigs.get_config(args.dataset)

    # Apply variants
    if args.variant == "unrolled":
        config = ConfigVariants.get_unrolled_training_config(base_config, args.unroll_length)
    elif args.variant == "noise":
        config = ConfigVariants.get_noise_training_config(base_config, args.noise_level)
    else:
        config = base_config

    # Create and run trainer
    trainer = UnifiedTrainer(config, use_gpu=not args.no_gpu, gpu_id=args.gpu)
    trainer.train()


if __name__ == "__main__":
    main()