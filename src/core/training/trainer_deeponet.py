"""
DeepONet-specific trainer for operator learning tasks.

Extends the base trainer to handle the specific requirements of DeepONet training,
including operator data loading, sensor-query point batching, and operator-specific
loss functions.
"""

import logging
import time
from typing import Dict, Optional, Any, Tuple

import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast

from .trainer import Trainer
from .loss_history import LossHistory
from .deeponet_loss import DeepONetLoss
from ..models.deeponet import DeepONet
from ..models.deeponet.data_utils import OperatorDataLoader, AdaptiveSensorSampler
from ..utils.params import DataParams, TrainingParams


class DeepONetTrainer(Trainer):
    """
    Trainer specifically designed for DeepONet operator learning.

    Handles the unique aspects of operator learning:
    - Function-operator pair training data
    - Sensor location optimization
    - Query point sampling strategies
    - Operator-specific losses and metrics
    """

    def __init__(self,
                 model: DeepONet,
                 trainLoader: OperatorDataLoader,
                 optimizer: Optimizer,
                 lrScheduler: _LRScheduler,
                 criterion: DeepONetLoss,
                 trainHistory: LossHistory,
                 writer: SummaryWriter,
                 p_d: DataParams,
                 p_t: TrainingParams,
                 valLoader: Optional[OperatorDataLoader] = None,
                 adaptive_sensors: bool = False):

        # Initialize base trainer
        super().__init__(model, trainLoader, optimizer, lrScheduler,
                        criterion, trainHistory, writer, p_d, p_t)

        self.valLoader = valLoader
        self.adaptive_sensors = adaptive_sensors

        # DeepONet-specific attributes
        self.operator_model = model  # Type-specific reference
        self.operator_criterion = criterion  # Type-specific reference

        # Initialize adaptive sensor sampler if requested
        if adaptive_sensors and hasattr(model, 'sensor_locations'):
            self.sensor_sampler = AdaptiveSensorSampler(
                initial_sensors=model.sensor_locations.clone(),
                update_frequency=100,  # Update every 100 batches
                learning_rate=0.01
            )
        else:
            self.sensor_sampler = None

        # Training state tracking
        self.current_epoch = 0
        self.global_step = 0

        logging.info(f"DeepONetTrainer initialized with {type(model).__name__}")
        logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logging.info(f"Adaptive sensors: {adaptive_sensors}")

    def trainingStep(self, epoch: int):
        """
        Execute one epoch of DeepONet training.

        Args:
            epoch: Current epoch number
        """
        assert len(self.trainLoader) > 0, "Not enough samples for one batch!"
        timerStart = time.perf_counter()

        self.current_epoch = epoch
        self.model.train()

        epoch_loss = 0.0
        epoch_operator_loss = 0.0
        epoch_physics_loss = 0.0
        num_batches = len(self.trainLoader)

        for batch_idx, batch in enumerate(self.trainLoader):
            self.optimizer.zero_grad()

            # Move batch to device
            device = "cuda" if self.model.useGPU else "cpu"
            batch = self._move_batch_to_device(batch, device)

            # MIXED PRECISION: Wrap forward pass in autocast
            with autocast('cuda', enabled=self.use_amp):
                loss_dict = self._forward_pass(batch, epoch)

            # Gradient clipping if specified
            # NOTE: Must unscale gradients before clipping when using mixed precision
            total_loss = loss_dict['total_loss']
            if hasattr(self.p_t, 'gradient_clip') and self.p_t.gradient_clip > 0:
                if self.use_amp and self.scaler is not None:
                    # Unscale gradients before clipping to avoid clipping on scaled gradients
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.p_t.gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.p_t.gradient_clip)
                    self.optimizer.step()
            else:
                # No gradient clipping: use parent's centralized method
                self._optimize_step(total_loss)

            # Update adaptive sensors if enabled
            if self.sensor_sampler is not None:
                with torch.no_grad():
                    # Get current batch data for sensor updates
                    input_funcs = self._reconstruct_input_functions(batch)
                    self.sensor_sampler.update_sensors(
                        self.model, input_funcs, batch['query_coords']
                    )

                    # Update model's sensor locations
                    updated_sensors = self.sensor_sampler.get_current_sensors()
                    self.model.sensor_locations.copy_(updated_sensors)

            # Accumulate losses
            epoch_loss += loss_dict['total_loss'].item()
            epoch_operator_loss += loss_dict.get('operator_loss', torch.tensor(0.0)).item()
            epoch_physics_loss += loss_dict.get('physics_loss', torch.tensor(0.0)).item()

            # Log batch-level metrics
            if batch_idx % 50 == 0:
                self._log_batch_metrics(epoch, batch_idx, loss_dict, num_batches)

            self.global_step += 1

        # Update learning rate
        if self.lrScheduler is not None:
            self.lrScheduler.step()

        # Calculate epoch averages
        avg_loss = epoch_loss / num_batches
        avg_operator_loss = epoch_operator_loss / num_batches
        avg_physics_loss = epoch_physics_loss / num_batches

        # Record training history
        self.trainHistory.recordTrainingStep(avg_loss)

        # Log epoch-level metrics
        elapsed = time.perf_counter() - timerStart
        self._log_epoch_metrics(epoch, avg_loss, avg_operator_loss,
                              avg_physics_loss, elapsed)

        # Validation step if validation loader provided
        if self.valLoader is not None and epoch % getattr(self.p_t, 'val_frequency', 5) == 0:
            self.validationStep(epoch)

    def validationStep(self, epoch: int):
        """
        Execute validation step for current epoch.

        Args:
            epoch: Current epoch number
        """
        if self.valLoader is None:
            return

        self.model.eval()
        val_loss = 0.0
        val_operator_loss = 0.0
        val_physics_loss = 0.0
        num_val_batches = len(self.valLoader)

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valLoader):
                device = "cuda" if self.model.useGPU else "cpu"
                batch = self._move_batch_to_device(batch, device)

                # Forward pass without gradient computation
                loss_dict = self._forward_pass(batch, epoch, is_training=False)

                val_loss += loss_dict['total_loss'].item()
                val_operator_loss += loss_dict.get('operator_loss', torch.tensor(0.0)).item()
                val_physics_loss += loss_dict.get('physics_loss', torch.tensor(0.0)).item()

        # Calculate validation averages
        avg_val_loss = val_loss / num_val_batches
        avg_val_operator_loss = val_operator_loss / num_val_batches
        avg_val_physics_loss = val_physics_loss / num_val_batches

        # Log validation metrics
        self._log_validation_metrics(epoch, avg_val_loss, avg_val_operator_loss,
                                   avg_val_physics_loss)

        # Record validation history
        self.trainHistory.recordValidationStep(avg_val_loss)

    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
        """Move batch tensors to specified device."""
        moved_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(device)
            else:
                moved_batch[key] = value
        return moved_batch

    def _forward_pass(self, batch: Dict[str, torch.Tensor], epoch: int,
                     is_training: bool = True) -> Dict[str, torch.Tensor]:
        """
        Execute forward pass and loss computation.

        Args:
            batch: Batch of operator data
            epoch: Current epoch number
            is_training: Whether in training mode

        Returns:
            loss_dict: Dictionary of computed losses
        """
        # Extract batch components
        input_sensors = batch['input_sensors']        # [B, T, n_sensors, C]
        query_coords = batch['query_coords']          # [B, n_query, 2]
        target_values = batch['target_values']        # [B, T, n_query, C]
        sensor_locations = batch['sensor_locations']   # [B, n_sensors, 2]

        B, T, n_sensors, C = input_sensors.shape
        n_query = query_coords.shape[1]

        # DeepONet expects input functions in [B, T, C, H, W] format for sensor sampling
        # Since we already have sensor values, we need to adapt the forward call

        # For now, use the DeepONet's forward method with pre-sampled sensor data
        # This requires modifying the forward call to accept sensor data directly
        predictions = self._deeponet_forward_with_sensors(
            input_sensors, query_coords, sensor_locations
        )

        # Compute losses
        loss_dict = self.operator_criterion(
            predictions=predictions,
            targets=target_values,
            input_sensors=input_sensors,
            query_coords=query_coords,
            model=self.model,
            epoch=epoch,
            is_training=is_training
        )

        return loss_dict

    def _deeponet_forward_with_sensors(self,
                                     input_sensors: torch.Tensor,
                                     query_coords: torch.Tensor,
                                     sensor_locations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DeepONet with pre-sampled sensor data.

        Args:
            input_sensors: Sensor values [B, T, n_sensors, C]
            query_coords: Query coordinates [B, n_query, 2]
            sensor_locations: Sensor locations [B, n_sensors, 2]

        Returns:
            predictions: Model predictions [B, T, n_query, C]
        """
        B, T, n_sensors, C = input_sensors.shape
        n_query = query_coords.shape[1]

        # Process each time step
        outputs = []
        for t in range(T):
            u_t = input_sensors[:, t]  # [B, n_sensors, C]

            # Branch network: encode function values at sensors
            branch_features = []
            for c in range(C):
                u_c = u_t[:, :, c]  # [B, n_sensors]
                if self.model.input_norm is not None:
                    u_c = self.model.input_norm(u_c)
                branch_out = self.model.branch_network(u_c)  # [B, latent_dim]
                branch_features.append(branch_out)

            # Average over channels
            branch_output = torch.stack(branch_features, dim=1).mean(dim=1)  # [B, latent_dim]

            # Trunk network: encode query coordinates
            coords_flat = query_coords.view(B * n_query, 2)  # [B*n_query, 2]
            if self.model.coord_norm is not None:
                coords_flat = self.model.coord_norm(coords_flat)
            trunk_output = self.model.trunk_network(coords_flat)  # [B*n_query, latent_dim]
            trunk_output = trunk_output.view(B, n_query, self.model.config.latent_dim)

            # Compute dot product
            branch_expanded = branch_output.unsqueeze(1)  # [B, 1, latent_dim]
            dot_product = (branch_expanded * trunk_output).sum(dim=2)  # [B, n_query]

            # Add bias if present
            if self.model.bias is not None:
                dot_product = dot_product + self.model.bias

            outputs.append(dot_product)

        # Stack time steps
        output = torch.stack(outputs, dim=1)  # [B, T, n_query]
        output = output.unsqueeze(3)  # [B, T, n_query, 1]

        # Expand to match input channels
        if C > 1:
            output = output.expand(-1, -1, -1, C)

        return output

    def _reconstruct_input_functions(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Reconstruct approximate input functions for adaptive sensor updates.

        This is a placeholder - in practice, you'd want to store the original
        function data or use a reconstruction method.
        """
        # For now, return a dummy tensor with correct shape
        B = batch['input_sensors'].shape[0]
        T = batch['input_sensors'].shape[1]
        C = batch['input_sensors'].shape[3]
        H, W = 64, 64  # Placeholder spatial dimensions

        return torch.zeros(B, T, C, H, W, device=batch['input_sensors'].device)

    def _log_batch_metrics(self, epoch: int, batch_idx: int,
                          loss_dict: Dict[str, torch.Tensor], num_batches: int):
        """Log batch-level training metrics."""
        if self.writer is not None:
            step = epoch * num_batches + batch_idx

            self.writer.add_scalar('Train/BatchLoss', loss_dict['total_loss'].item(), step)
            if 'operator_loss' in loss_dict:
                self.writer.add_scalar('Train/BatchOperatorLoss',
                                     loss_dict['operator_loss'].item(), step)
            if 'physics_loss' in loss_dict:
                self.writer.add_scalar('Train/BatchPhysicsLoss',
                                     loss_dict['physics_loss'].item(), step)

    def _log_epoch_metrics(self, epoch: int, avg_loss: float,
                          avg_operator_loss: float, avg_physics_loss: float,
                          elapsed_time: float):
        """Log epoch-level training metrics."""
        logging.info(f"Epoch {epoch:04d} - "
                    f"Loss: {avg_loss:.6f} "
                    f"(Op: {avg_operator_loss:.6f}, "
                    f"Phy: {avg_physics_loss:.6f}) - "
                    f"Time: {elapsed_time:.2f}s")

        if self.writer is not None:
            self.writer.add_scalar('Train/EpochLoss', avg_loss, epoch)
            self.writer.add_scalar('Train/EpochOperatorLoss', avg_operator_loss, epoch)
            self.writer.add_scalar('Train/EpochPhysicsLoss', avg_physics_loss, epoch)
            self.writer.add_scalar('Train/EpochTime', elapsed_time, epoch)

            # Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Train/LearningRate', current_lr, epoch)

            # Log adaptive sensor information
            if self.sensor_sampler is not None:
                info_history = self.sensor_sampler.get_information_history()
                if info_history:
                    self.writer.add_scalar('Train/SensorInformationGain',
                                         info_history[-1], epoch)

    def _log_validation_metrics(self, epoch: int, avg_val_loss: float,
                              avg_val_operator_loss: float, avg_val_physics_loss: float):
        """Log validation metrics."""
        logging.info(f"Validation - "
                    f"Loss: {avg_val_loss:.6f} "
                    f"(Op: {avg_val_operator_loss:.6f}, "
                    f"Phy: {avg_val_physics_loss:.6f})")

        if self.writer is not None:
            self.writer.add_scalar('Val/EpochLoss', avg_val_loss, epoch)
            self.writer.add_scalar('Val/EpochOperatorLoss', avg_val_operator_loss, epoch)
            self.writer.add_scalar('Val/EpochPhysicsLoss', avg_val_physics_loss, epoch)

    def save_checkpoint(self, filepath: str, epoch: int, additional_info: Dict[str, Any] = None):
        """
        Save training checkpoint including adaptive sensor state.

        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch number
            additional_info: Additional information to save
        """
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lrScheduler.state_dict() if self.lrScheduler else None,
            'train_history': self.trainHistory.get_history(),
            'model_config': self.operator_model.config.__dict__ if hasattr(self.operator_model, 'config') else None
        }

        # Save adaptive sensor state
        if self.sensor_sampler is not None:
            checkpoint['adaptive_sensors'] = {
                'sensor_locations': self.sensor_sampler.get_current_sensors(),
                'information_history': self.sensor_sampler.get_information_history(),
                'batch_count': self.sensor_sampler.batch_count
            }

        # Add additional info
        if additional_info:
            checkpoint.update(additional_info)

        torch.save(checkpoint, filepath)
        logging.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """
        Load training checkpoint including adaptive sensor state.

        Args:
            filepath: Path to checkpoint file

        Returns:
            checkpoint_info: Dictionary with checkpoint metadata
        """
        checkpoint = torch.load(filepath, map_location='cpu')

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if checkpoint.get('scheduler_state_dict') and self.lrScheduler:
            self.lrScheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'train_history' in checkpoint and hasattr(self.trainHistory, 'load_history'):
            self.trainHistory.load_history(checkpoint['train_history'])

        # Restore adaptive sensor state
        if 'adaptive_sensors' in checkpoint and self.sensor_sampler is not None:
            sensor_state = checkpoint['adaptive_sensors']
            self.sensor_sampler.sensors.data.copy_(sensor_state['sensor_locations'])
            self.sensor_sampler.information_history = sensor_state.get('information_history', [])
            self.sensor_sampler.batch_count = sensor_state.get('batch_count', 0)

            # Update model's sensor locations
            self.model.sensor_locations.copy_(sensor_state['sensor_locations'])

        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)

        logging.info(f"Checkpoint loaded from {filepath}, resuming from epoch {self.current_epoch}")

        return {
            'epoch': checkpoint.get('epoch', 0),
            'global_step': checkpoint.get('global_step', 0),
            'model_config': checkpoint.get('model_config', None)
        }