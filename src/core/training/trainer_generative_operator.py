"""
Two-Stage Trainer for Generative Operator Models

This module implements a specialized trainer that extends the existing Trainer class
to support two-stage training for generative operator models (NO+DM combinations).

Training Stages:
1. Stage 1: Train neural operator prior only
2. Stage 2: Freeze prior, train generative corrector
3. Optional: Joint fine-tuning of both components

The trainer maintains full compatibility with the existing Gen Stabilised training
infrastructure while adding generative operator specific functionality.
"""

import logging
import time
import math
import os
from typing import Dict, Optional, Any, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, StepLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from .trainer import Trainer
from src.core.models.generative_operator_model import GenerativeOperatorModel
from src.core.training.loss import PredictionLoss
from src.core.training.loss_history import LossHistory
from src.core.utils.params import DataParams, TrainingParams
from src.core.models.memory_optimization import MemoryOptimizer, AdaptiveBatchSampler


class GenerativeOperatorLoss:
    """
    Specialized loss functions for generative operator training.

    Implements MSE, perceptual, and consistency losses for both prior
    and corrector training stages.
    """

    def __init__(self, p_t: TrainingParams):
        self.p_t = p_t
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

        # Initialize perceptual loss if requested
        self.use_perceptual_loss = getattr(p_t, 'use_perceptual_loss', False)
        self.perceptual_weight = getattr(p_t, 'perceptual_loss_weight', 0.1)

        if self.use_perceptual_loss:
            self._init_perceptual_loss()

    def _init_perceptual_loss(self):
        """Initialize perceptual loss network (simplified VGG-like)."""
        try:
            import torchvision.models as models
            vgg = models.vgg16(pretrained=True).features[:16]  # Use first few layers
            vgg.eval()
            for param in vgg.parameters():
                param.requires_grad = False
            self.perceptual_net = vgg
            logging.info("Initialized perceptual loss with VGG16")
        except ImportError:
            logging.warning("torchvision not available, disabling perceptual loss")
            self.use_perceptual_loss = False

    def compute_prior_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute loss for prior training stage."""
        mse_loss = self.mse_loss(prediction, target)

        losses = {
            'mse_loss': mse_loss,
            'total_loss': mse_loss
        }

        # Add L1 regularization if requested
        if getattr(self.p_t, 'use_l1_loss', False):
            l1_loss = self.l1_loss(prediction, target)
            l1_weight = getattr(self.p_t, 'l1_loss_weight', 0.1)
            losses['l1_loss'] = l1_loss
            losses['total_loss'] = losses['total_loss'] + l1_weight * l1_loss

        return losses

    def compute_corrector_loss(self, prediction: torch.Tensor, target: torch.Tensor,
                              prior_prediction: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute loss for corrector training stage."""

        # Main reconstruction loss
        mse_loss = self.mse_loss(prediction, target)

        # Consistency loss (ensure corrector improves upon prior)
        prior_mse = self.mse_loss(prior_prediction, target)
        corrected_mse = self.mse_loss(prediction, target)
        consistency_loss = F.relu(corrected_mse - prior_mse + 0.01)  # Encourage improvement

        losses = {
            'mse_loss': mse_loss,
            'consistency_loss': consistency_loss,
            'total_loss': mse_loss + 0.1 * consistency_loss
        }

        # Add perceptual loss if enabled
        if self.use_perceptual_loss and hasattr(self, 'perceptual_net'):
            perceptual_loss = self._compute_perceptual_loss(prediction, target)
            losses['perceptual_loss'] = perceptual_loss
            losses['total_loss'] = losses['total_loss'] + self.perceptual_weight * perceptual_loss

        return losses

    def _compute_perceptual_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss using pre-trained network."""
        # Reshape for VGG input (expect 3-channel images)
        B, T, C, H, W = prediction.shape

        # Take last frame and adapt channels for VGG
        pred_frame = prediction[:, -1]  # [B, C, H, W]
        target_frame = target[:, -1] if target.shape[1] > 0 else target[:, 0]

        # Convert to 3-channel if needed
        if C == 1:
            pred_frame = pred_frame.repeat(1, 3, 1, 1)
            target_frame = target_frame.repeat(1, 3, 1, 1)
        elif C > 3:
            pred_frame = pred_frame[:, :3]  # Take first 3 channels
            target_frame = target_frame[:, :3]

        # Resize to minimum VGG input size if needed
        if H < 224 or W < 224:
            pred_frame = F.interpolate(pred_frame, size=(224, 224), mode='bilinear')
            target_frame = F.interpolate(target_frame, size=(224, 224), mode='bilinear')

        # Compute features
        pred_features = self.perceptual_net(pred_frame)
        target_features = self.perceptual_net(target_frame)

        return self.mse_loss(pred_features, target_features)


class GenerativeOperatorTrainer(Trainer):
    """
    Enhanced two-stage trainer for generative operator models.

    This trainer extends the base Trainer class to support the unique training
    requirements of generative operator models, including stage transitions,
    component freezing, specialized loss computation, and memory optimization.
    """

    def __init__(self, model: GenerativeOperatorModel, trainLoader: DataLoader,
                 optimizer: Optimizer, lrScheduler: _LRScheduler,
                 criterion: PredictionLoss, trainHistory: LossHistory,
                 writer: SummaryWriter, p_d: DataParams, p_t: TrainingParams):

        # Initialize base trainer
        super().__init__(model, trainLoader, optimizer, lrScheduler, criterion,
                        trainHistory, writer, p_d, p_t)

        # Generative operator specific attributes
        self.current_stage = 1  # Start with stage 1 (prior training)
        self.stage_transition_epoch = p_t.stage1_epochs
        self.total_stage1_epochs = p_t.stage1_epochs
        self.total_stage2_epochs = p_t.stage2_epochs
        self.enable_joint_finetuning = getattr(p_t, 'enable_joint_finetuning', False)
        self.joint_finetuning_epochs = getattr(p_t, 'joint_finetuning_epochs', 10)

        # Enhanced loss computation
        self.genop_loss = GenerativeOperatorLoss(p_t)

        # Memory optimization
        self.memory_optimizer = MemoryOptimizer()
        self.adaptive_batch_sampler = AdaptiveBatchSampler(
            initial_batch_size=p_d.batch_size,
            min_batch_size=max(1, p_d.batch_size // 4),
            max_batch_size=p_d.batch_size * 2
        )

        # Create stage-specific optimizers and schedulers
        self.stage1_optimizer = None
        self.stage2_optimizer = None
        self.joint_optimizer = None
        self.stage1_scheduler = None
        self.stage2_scheduler = None
        self.joint_scheduler = None
        self._setup_stage_optimizers()
        self._setup_stage_schedulers()

        # Enhanced training state tracking
        self.stage_start_epoch = 0
        self.validation_scores = {'stage1': [], 'stage2': [], 'joint': []}
        self.best_validation_loss = float('inf')
        self.epochs_without_improvement = 0
        self.losses_history = {
            'stage1': {'mse_loss': [], 'l1_loss': [], 'total_loss': []},
            'stage2': {'mse_loss': [], 'consistency_loss': [], 'perceptual_loss': [], 'total_loss': []},
            'joint': {'total_loss': [], 'prior_loss': [], 'corrector_loss': []}
        }

        # Checkpoint management
        self.checkpoint_dir = getattr(p_t, 'checkpoint_dir', './checkpoints')
        self.save_best_only = getattr(p_t, 'save_best_only', True)
        self.checkpoint_frequency = getattr(p_t, 'checkpoint_frequency', 10)

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        logging.info(f"Initialized Enhanced GenerativeOperatorTrainer:")
        logging.info(f"  Stage 1 epochs: {self.total_stage1_epochs}")
        logging.info(f"  Stage 2 epochs: {self.total_stage2_epochs}")
        logging.info(f"  Joint fine-tuning: {self.enable_joint_finetuning}")
        if self.enable_joint_finetuning:
            logging.info(f"  Joint fine-tuning epochs: {self.joint_finetuning_epochs}")
        logging.info(f"  Stage transition at epoch: {self.stage_transition_epoch}")
        logging.info(f"  Memory optimization enabled: {True}")
        logging.info(f"  Checkpoint directory: {self.checkpoint_dir}")

    def _setup_stage_optimizers(self):
        """Setup optimizers for different training stages."""
        if hasattr(self.model, 'prior_model') and hasattr(self.model, 'corrector_model'):
            # Stage 1: Only optimize prior model
            self.stage1_optimizer = torch.optim.Adam(
                self.model.prior_model.parameters(),
                lr=self.p_t.stage1_lr,
                weight_decay=self.p_t.weightDecay
            )

            # Stage 2: Only optimize corrector model
            self.stage2_optimizer = torch.optim.Adam(
                self.model.corrector_model.parameters(),
                lr=self.p_t.stage2_lr,
                weight_decay=self.p_t.weightDecay
            )

            logging.info(f"Created stage-specific optimizers:")
            logging.info(f"  Stage 1 LR: {self.p_t.stage1_lr}")
            logging.info(f"  Stage 2 LR: {self.p_t.stage2_lr}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train one epoch with stage-aware logic.

        Args:
            epoch: Current epoch number

        Returns:
            losses: Dictionary of loss values for this epoch
        """
        # Check for stage transition
        if epoch == self.stage_transition_epoch and self.current_stage == 1:
            self._transition_to_stage2(epoch)

        # Set training mode based on current stage
        if self.current_stage == 1:
            return self._train_stage1_epoch(epoch)
        elif self.current_stage == 2:
            return self._train_stage2_epoch(epoch)
        else:
            raise ValueError(f"Unknown training stage: {self.current_stage}")

    def _transition_to_stage2(self, epoch: int):
        """Transition from stage 1 to stage 2 training."""
        logging.info(f"\n" + "="*50)
        logging.info(f"STAGE TRANSITION AT EPOCH {epoch}")
        logging.info(f"Stage 1 → Stage 2")
        logging.info(f"Freezing neural operator prior, training generative corrector")
        logging.info(f"="*50)

        # Update stage tracking
        self.current_stage = 2
        self.stage_start_epoch = epoch

        # Set model training mode
        self.model.set_training_mode('corrector_training')

        # Freeze prior if configured
        if self.p_t.freeze_prior_after_stage1:
            self.model.freeze_prior()

        # Switch to stage 2 optimizer
        if self.stage2_optimizer:
            self.optimizer = self.stage2_optimizer

        # Update learning rate scheduler for new optimizer
        if hasattr(self.lrScheduler, 'optimizer'):
            self.lrScheduler.optimizer = self.optimizer

        # Log transition information
        prior_params = sum(p.numel() for p in self.model.prior_model.parameters() if p.requires_grad)
        corrector_params = sum(p.numel() for p in self.model.corrector_model.parameters() if p.requires_grad)

        logging.info(f"Trainable parameters after transition:")
        logging.info(f"  Prior: {prior_params:,}")
        logging.info(f"  Corrector: {corrector_params:,}")

    def _train_stage1_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train one epoch in stage 1 (neural operator prior only).

        Args:
            epoch: Current epoch number

        Returns:
            losses: Dictionary of loss values
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.trainLoader):
            # Extract data and target
            data = batch[0]  # [B, T, C, H, W]
            target = batch[1] if len(batch) > 1 else data[:, 1:]  # [B, T-1, C, H, W]

            # Move to device
            data = data.to(next(self.model.parameters()).device)
            target = target.to(next(self.model.parameters()).device)

            # Forward pass through prior only
            self.stage1_optimizer.zero_grad()
            prediction = self.model(data)

            # Compute loss (standard prediction loss)
            loss_dict = self.criterion(prediction, target, None, None)
            total_loss_value = loss_dict['total']

            # Backward pass
            total_loss_value.backward()
            self.stage1_optimizer.step()

            # Accumulate losses
            total_loss += total_loss_value.item()
            num_batches += 1

            # Log batch progress
            if batch_idx % 100 == 0:
                logging.debug(f"Stage 1 Epoch {epoch}, Batch {batch_idx}: Loss = {total_loss_value.item():.6f}")

        # Average losses
        avg_loss = total_loss / num_batches

        # Store stage 1 losses
        self.losses_history['stage1']['prior_loss'].append(avg_loss)
        self.losses_history['stage1']['total_loss'].append(avg_loss)

        # Log to tensorboard
        if self.writer:
            self.writer.add_scalar('Stage1/Prior_Loss', avg_loss, epoch)
            self.writer.add_scalar('Stage1/Learning_Rate', self.stage1_optimizer.param_groups[0]['lr'], epoch)

        return {
            'total_loss': avg_loss,
            'prior_loss': avg_loss,
            'stage': 1
        }

    def _train_stage2_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train one epoch in stage 2 (generative corrector only).

        Args:
            epoch: Current epoch number

        Returns:
            losses: Dictionary of loss values
        """
        self.model.train()
        total_loss = 0.0
        diffusion_loss_total = 0.0
        consistency_loss_total = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.trainLoader):
            # Extract data and target
            data = batch[0]  # [B, T, C, H, W]
            target = batch[1] if len(batch) > 1 else data[:, 1:]  # [B, T-1, C, H, W]

            # Move to device
            data = data.to(next(self.model.parameters()).device)
            target = target.to(next(self.model.parameters()).device)

            # Forward pass: get prior prediction (no gradients)
            with torch.no_grad():
                self.model.set_training_mode('prior_only')
                prior_prediction = self.model(data)
                self.model.set_training_mode('corrector_training')

            # Train corrector
            self.stage2_optimizer.zero_grad()

            # Use corrector's train_step method
            batch_dict = {'data': data, 'target': target}
            loss_dict = self.model.corrector_model.train_step(
                batch_dict, prior_prediction, self.stage2_optimizer
            )

            # Extract losses
            diffusion_loss = loss_dict.get('diffusion_loss', 0.0)
            consistency_loss = loss_dict.get('consistency_loss', 0.0)
            total_loss_value = loss_dict.get('total_loss', diffusion_loss + consistency_loss)

            # Accumulate losses
            total_loss += total_loss_value
            diffusion_loss_total += diffusion_loss
            consistency_loss_total += consistency_loss
            num_batches += 1

            # Log batch progress
            if batch_idx % 100 == 0:
                logging.debug(f"Stage 2 Epoch {epoch}, Batch {batch_idx}: "
                             f"Total = {total_loss_value:.6f}, "
                             f"Diffusion = {diffusion_loss:.6f}, "
                             f"Consistency = {consistency_loss:.6f}")

        # Average losses
        avg_total_loss = total_loss / num_batches
        avg_diffusion_loss = diffusion_loss_total / num_batches
        avg_consistency_loss = consistency_loss_total / num_batches

        # Store stage 2 losses
        self.losses_history['stage2']['diffusion_loss'].append(avg_diffusion_loss)
        self.losses_history['stage2']['consistency_loss'].append(avg_consistency_loss)
        self.losses_history['stage2']['total_loss'].append(avg_total_loss)

        # Log to tensorboard
        if self.writer:
            self.writer.add_scalar('Stage2/Total_Loss', avg_total_loss, epoch)
            self.writer.add_scalar('Stage2/Diffusion_Loss', avg_diffusion_loss, epoch)
            self.writer.add_scalar('Stage2/Consistency_Loss', avg_consistency_loss, epoch)
            self.writer.add_scalar('Stage2/Learning_Rate', self.stage2_optimizer.param_groups[0]['lr'], epoch)

        return {
            'total_loss': avg_total_loss,
            'diffusion_loss': avg_diffusion_loss,
            'consistency_loss': avg_consistency_loss,
            'stage': 2
        }

    def validate_epoch(self, validLoader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Validate one epoch with stage-aware evaluation.

        Args:
            validLoader: Validation data loader
            epoch: Current epoch number

        Returns:
            losses: Validation loss dictionary
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in validLoader:
                data = batch[0]
                target = batch[1] if len(batch) > 1 else data[:, 1:]

                # Move to device
                data = data.to(next(self.model.parameters()).device)
                target = target.to(next(self.model.parameters()).device)

                # Forward pass based on current stage
                if self.current_stage == 1:
                    # Stage 1: Evaluate prior only
                    self.model.set_training_mode('prior_only')
                    prediction = self.model(data)
                    loss_dict = self.criterion(prediction, target, None, None)
                    loss_value = loss_dict['total'].item()

                else:
                    # Stage 2: Evaluate full generative operator
                    self.model.set_training_mode('full_inference')
                    prediction = self.model(data)
                    loss_dict = self.criterion(prediction, target, None, None)
                    loss_value = loss_dict['total'].item()

                total_loss += loss_value
                num_batches += 1

        avg_loss = total_loss / num_batches

        # Log validation results
        if self.writer:
            stage_name = f"Stage{self.current_stage}"
            self.writer.add_scalar(f'{stage_name}/Validation_Loss', avg_loss, epoch)

        return {'validation_loss': avg_loss, 'stage': self.current_stage}

    def get_training_status(self) -> Dict[str, Any]:
        """
        Get comprehensive training status information.

        Returns:
            status: Dictionary with training status
        """
        base_status = {
            'current_stage': self.current_stage,
            'stage_transition_epoch': self.stage_transition_epoch,
            'stage_start_epoch': self.stage_start_epoch,
            'total_stage1_epochs': self.total_stage1_epochs,
            'total_stage2_epochs': self.total_stage2_epochs,
            'losses_history': self.losses_history
        }

        # Add model information
        if hasattr(self.model, 'get_model_info'):
            base_status['model_info'] = self.model.get_model_info()

        # Add optimizer information
        current_optimizer = self.stage1_optimizer if self.current_stage == 1 else self.stage2_optimizer
        if current_optimizer:
            base_status['current_lr'] = current_optimizer.param_groups[0]['lr']

        return base_status

    def save_checkpoint(self, filepath: str, epoch: int, additional_info: Optional[Dict] = None):
        """
        Save training checkpoint with stage information.

        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch
            additional_info: Additional information to save
        """
        checkpoint = {
            'epoch': epoch,
            'current_stage': self.current_stage,
            'stage_transition_epoch': self.stage_transition_epoch,
            'stage_start_epoch': self.stage_start_epoch,
            'losses_history': self.losses_history,
            'model_state_dict': self.model.save_state_dict(),
            'stage1_optimizer_state_dict': self.stage1_optimizer.state_dict() if self.stage1_optimizer else None,
            'stage2_optimizer_state_dict': self.stage2_optimizer.state_dict() if self.stage2_optimizer else None,
            'scheduler_state_dict': self.lrScheduler.state_dict() if self.lrScheduler else None,
        }

        if additional_info:
            checkpoint['additional_info'] = additional_info

        torch.save(checkpoint, filepath)
        logging.info(f"Saved checkpoint to {filepath}")

    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """
        Load training checkpoint and restore stage information.

        Args:
            filepath: Path to checkpoint file

        Returns:
            checkpoint_info: Information from loaded checkpoint
        """
        checkpoint = torch.load(filepath, map_location=next(self.model.parameters()).device)

        # Restore stage information
        self.current_stage = checkpoint.get('current_stage', 1)
        self.stage_transition_epoch = checkpoint.get('stage_transition_epoch', self.total_stage1_epochs)
        self.stage_start_epoch = checkpoint.get('stage_start_epoch', 0)
        self.losses_history = checkpoint.get('losses_history', self.losses_history)

        # Load model state
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer states
        if 'stage1_optimizer_state_dict' in checkpoint and self.stage1_optimizer:
            self.stage1_optimizer.load_state_dict(checkpoint['stage1_optimizer_state_dict'])

        if 'stage2_optimizer_state_dict' in checkpoint and self.stage2_optimizer:
            self.stage2_optimizer.load_state_dict(checkpoint['stage2_optimizer_state_dict'])

        # Load scheduler state
        if 'scheduler_state_dict' in checkpoint and self.lrScheduler:
            self.lrScheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Set current optimizer based on stage
        if self.current_stage == 1:
            self.optimizer = self.stage1_optimizer
        else:
            self.optimizer = self.stage2_optimizer
            if self.p_t.freeze_prior_after_stage1:
                self.model.freeze_prior()

        # Set appropriate training mode
        if self.current_stage == 1:
            self.model.set_training_mode('prior_only')
        else:
            self.model.set_training_mode('corrector_training')

        logging.info(f"Loaded checkpoint from {filepath}")
        logging.info(f"Restored to stage {self.current_stage}")

        return {
            'epoch': checkpoint.get('epoch', 0),
            'current_stage': self.current_stage,
            'additional_info': checkpoint.get('additional_info', {})
        }

    def finalize_training(self):
        """Finalize training and set model to inference mode."""
        logging.info("\n" + "="*60)
        logging.info("FINALIZING GENERATIVE OPERATOR TRAINING")
        logging.info("="*60)

        # Save final checkpoint
        final_checkpoint_path = os.path.join(self.checkpoint_dir, 'final_model.pt')
        self.save_checkpoint(final_checkpoint_path, -1, {'training_completed': True})

        # Unfreeze all parameters for final inference
        self.model.unfreeze_prior()
        self.model.set_training_mode('full_inference')

        # Log comprehensive training statistics
        logging.info("Final Training Statistics:")

        if self.losses_history['stage1']['total_loss']:
            final_stage1_loss = self.losses_history['stage1']['total_loss'][-1]
            logging.info(f"  Stage 1 (Prior) Final Loss: {final_stage1_loss:.6f}")

        if self.losses_history['stage2']['total_loss']:
            final_stage2_loss = self.losses_history['stage2']['total_loss'][-1]
            logging.info(f"  Stage 2 (Corrector) Final Loss: {final_stage2_loss:.6f}")

        if self.enable_joint_finetuning and self.losses_history['joint']['total_loss']:
            final_joint_loss = self.losses_history['joint']['total_loss'][-1]
            logging.info(f"  Joint Fine-tuning Final Loss: {final_joint_loss:.6f}")

        logging.info(f"  Best Validation Loss: {self.best_validation_loss:.6f}")

        # Get final model information
        model_info = self.model.get_model_info()
        logging.info(f"  Total Parameters: {model_info['total_parameters']:,}")
        logging.info(f"  Prior Parameters: {model_info.get('prior_parameters', 'N/A')}")
        logging.info(f"  Corrector Parameters: {model_info.get('corrector_parameters', 'N/A')}")

        # Memory stats
        memory_stats = self.adaptive_batch_sampler.get_memory_stats()
        if memory_stats['memory_available']:
            logging.info(f"  Peak Memory Usage: {memory_stats['usage_ratio']:.2%}")
            logging.info(f"  Final Batch Size: {memory_stats['current_batch_size']}")

        logging.info("\n" + "="*60)
        logging.info("GENERATIVE OPERATOR TRAINING COMPLETED SUCCESSFULLY!")
        logging.info(f"Model ready for inference with {self.get_stage_name()}")
        logging.info("="*60)


def create_generative_operator_trainer(model: GenerativeOperatorModel,
                                     trainLoader: DataLoader,
                                     p_d: DataParams,
                                     p_t: TrainingParams,
                                     validLoader: Optional[DataLoader] = None,
                                     writer: Optional[SummaryWriter] = None) -> GenerativeOperatorTrainer:
    """
    Factory function to create a configured generative operator trainer.

    Args:
        model: Generative operator model to train
        trainLoader: Training data loader
        p_d: Data parameters
        p_t: Training parameters
        validLoader: Optional validation data loader
        writer: Optional tensorboard writer

    Returns:
        trainer: Configured GenerativeOperatorTrainer
    """
    from src.core.training.loss import PredictionLoss
    from src.core.training.loss_history import LossHistory

    # Create basic components
    criterion = PredictionLoss(p_t)
    trainHistory = LossHistory()

    # Create placeholder optimizer (will be replaced by stage-specific optimizers)
    optimizer = torch.optim.Adam(model.parameters(), lr=p_t.stage1_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Create writer if not provided
    if writer is None:
        log_dir = getattr(p_t, 'log_dir', './logs')
        writer = SummaryWriter(log_dir)

    trainer = GenerativeOperatorTrainer(
        model=model,
        trainLoader=trainLoader,
        optimizer=optimizer,
        lrScheduler=scheduler,
        criterion=criterion,
        trainHistory=trainHistory,
        writer=writer,
        p_d=p_d,
        p_t=p_t
    )

    logging.info("Created GenerativeOperatorTrainer with automatic configuration")
    logging.info(f"  Training stages: {trainer.get_stage_name()}")
    logging.info(f"  Model type: {model.get_model_info().get('model_type', 'Unknown')}")

    return trainer


def run_two_stage_training(trainer: GenerativeOperatorTrainer,
                          validLoader: Optional[DataLoader] = None,
                          early_stopping_patience: int = 50) -> Dict[str, Any]:
    """
    Run complete two-stage training workflow.

    Args:
        trainer: Configured GenerativeOperatorTrainer
        validLoader: Optional validation data loader
        early_stopping_patience: Epochs to wait before early stopping

    Returns:
        training_results: Comprehensive training results
    """
    total_epochs = (trainer.total_stage1_epochs +
                   trainer.total_stage2_epochs +
                   (trainer.joint_finetuning_epochs if trainer.enable_joint_finetuning else 0))

    logging.info(f"Starting two-stage training for {total_epochs} total epochs")
    start_time = time.time()

    for epoch in range(total_epochs):
        epoch_start_time = time.time()

        # Training epoch
        train_losses = trainer.train_epoch(epoch)

        # Validation epoch
        val_losses = {}
        if validLoader is not None:
            val_losses = trainer.validate_epoch(validLoader, epoch)

        # Save periodic checkpoint
        trainer.save_periodic_checkpoint(epoch)

        # Log progress
        epoch_time = time.time() - epoch_start_time
        stage_name = trainer.get_stage_name()

        logging.info(f"Epoch {epoch:3d}/{total_epochs} [{stage_name}] "
                    f"Train Loss: {train_losses['total_loss']:.6f} "
                    f"Val Loss: {val_losses.get('validation_loss', 'N/A')} "
                    f"Time: {epoch_time:.2f}s")

        # Check early stopping
        if (validLoader is not None and
            trainer.should_early_stop(early_stopping_patience)):
            logging.info(f"Early stopping triggered after {trainer.epochs_without_improvement} epochs without improvement")
            break

    # Finalize training
    trainer.finalize_training()

    total_time = time.time() - start_time

    # Compile results
    results = {
        'total_training_time': total_time,
        'final_stage': trainer.current_stage,
        'best_validation_loss': trainer.best_validation_loss,
        'training_status': trainer.get_training_status(),
        'losses_history': trainer.losses_history,
        'validation_scores': trainer.validation_scores
    }

    logging.info(f"Training completed in {total_time/3600:.2f} hours")
    return results