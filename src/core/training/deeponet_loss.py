"""
Loss functions and metrics specifically designed for DeepONet operator learning.

Implements various loss functions for training DeepONet models, including
operator losses, physics-informed losses, and regularization terms.
"""

from typing import Dict, Optional, List, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..models.deeponet.deeponet_variants import PhysicsInformedDeepONet


class OperatorL2Loss(nn.Module):
    """
    Standard L2 loss for operator learning.

    Computes mean squared error between predicted and target operator outputs
    at query points.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute L2 loss between predictions and targets.

        Args:
            predictions: Model predictions [B, T, n_query, C]
            targets: Target values [B, T, n_query, C]

        Returns:
            loss: L2 loss value
        """
        mse = F.mse_loss(predictions, targets, reduction='none')

        if self.reduction == 'mean':
            return torch.mean(mse)
        elif self.reduction == 'sum':
            return torch.sum(mse)
        elif self.reduction == 'none':
            return mse
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class RelativeL2Loss(nn.Module):
    """
    Relative L2 loss normalized by target magnitude.

    Useful for problems with varying solution magnitudes.
    """

    def __init__(self, eps: float = 1e-8, reduction: str = 'mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute relative L2 loss.

        Args:
            predictions: Model predictions [B, T, n_query, C]
            targets: Target values [B, T, n_query, C]

        Returns:
            loss: Relative L2 loss value
        """
        diff = predictions - targets
        rel_loss = torch.norm(diff, dim=(1, 2, 3)) / (torch.norm(targets, dim=(1, 2, 3)) + self.eps)

        if self.reduction == 'mean':
            return torch.mean(rel_loss)
        elif self.reduction == 'sum':
            return torch.sum(rel_loss)
        elif self.reduction == 'none':
            return rel_loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class HuberLoss(nn.Module):
    """
    Huber loss for robust operator learning.

    Less sensitive to outliers than L2 loss.
    """

    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Huber loss.

        Args:
            predictions: Model predictions [B, T, n_query, C]
            targets: Target values [B, T, n_query, C]

        Returns:
            loss: Huber loss value
        """
        return F.huber_loss(predictions, targets, delta=self.delta, reduction=self.reduction)


class SpectralLoss(nn.Module):
    """
    Spectral loss for frequency domain accuracy.

    Useful for problems with important frequency content.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral loss using FFT.

        Args:
            predictions: Model predictions [B, T, n_query, C]
            targets: Target values [B, T, n_query, C]

        Returns:
            loss: Spectral loss value
        """
        # Compute FFT along spatial dimensions (assuming square grids)
        pred_fft = torch.fft.fft2(predictions, dim=(2, 3))
        target_fft = torch.fft.fft2(targets, dim=(2, 3))

        # Compute L2 loss in frequency domain
        spectral_loss = F.mse_loss(pred_fft.real, target_fft.real, reduction='none') + \
                       F.mse_loss(pred_fft.imag, target_fft.imag, reduction='none')

        if self.reduction == 'mean':
            return torch.mean(spectral_loss)
        elif self.reduction == 'sum':
            return torch.sum(spectral_loss)
        elif self.reduction == 'none':
            return spectral_loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class GradientLoss(nn.Module):
    """
    Gradient-based loss for spatial consistency.

    Penalizes differences in spatial gradients between predictions and targets.
    """

    def __init__(self, weight: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                coords: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient loss.

        Args:
            predictions: Model predictions [B, T, n_query, C]
            targets: Target values [B, T, n_query, C]
            coords: Query coordinates [B, n_query, 2]

        Returns:
            loss: Gradient loss value
        """
        # Enable gradient computation for predictions
        predictions = predictions.clone().requires_grad_(True)
        targets = targets.clone().requires_grad_(True)
        coords = coords.clone().requires_grad_(True)

        # Compute gradients w.r.t. coordinates
        pred_grad = torch.autograd.grad(
            outputs=predictions.sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True
        )[0]

        target_grad = torch.autograd.grad(
            outputs=targets.sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True
        )[0]

        # Compute L2 loss between gradients
        grad_loss = F.mse_loss(pred_grad, target_grad, reduction=self.reduction)

        return self.weight * grad_loss


class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed loss for enforcing PDE constraints.

    Works with PhysicsInformedDeepONet to compute PDE residual losses.
    """

    def __init__(self, physics_weights: Dict[str, float] = None):
        super().__init__()
        self.physics_weights = physics_weights or {
            'continuity_loss': 1.0,
            'laplace_loss': 1.0,
            'helmholtz_loss': 1.0
        }

    def forward(self, model: PhysicsInformedDeepONet, predictions: torch.Tensor,
                targets: torch.Tensor, query_coords: torch.Tensor,
                input_functions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute physics-informed losses.

        Args:
            model: Physics-informed DeepONet model
            predictions: Model predictions [B, T, n_query, C]
            targets: Target values [B, T, n_query, C]
            query_coords: Query coordinates [B, n_query, 2]
            input_functions: Input functions (if available)

        Returns:
            losses: Dictionary of physics losses
        """
        if not isinstance(model, PhysicsInformedDeepONet):
            return {}

        # Compute physics losses using the model's built-in method
        physics_losses = model.compute_physics_loss(
            input_functions, predictions, query_coords
        )

        # Apply weights to losses
        weighted_losses = {}
        for loss_name, loss_value in physics_losses.items():
            weight = self.physics_weights.get(loss_name, 1.0)
            weighted_losses[loss_name] = weight * loss_value

        return weighted_losses


class SensorRegularizationLoss(nn.Module):
    """
    Regularization loss for sensor placement optimization.

    Encourages diverse and well-distributed sensor locations.
    """

    def __init__(self, weight: float = 0.01):
        super().__init__()
        self.weight = weight

    def forward(self, sensor_locations: torch.Tensor) -> torch.Tensor:
        """
        Compute sensor regularization loss.

        Args:
            sensor_locations: Sensor coordinates [B, n_sensors, 2]

        Returns:
            loss: Regularization loss encouraging sensor diversity
        """
        B, n_sensors, _ = sensor_locations.shape

        # Compute pairwise distances between sensors
        # Expand for broadcasting: [B, n_sensors, 1, 2] and [B, 1, n_sensors, 2]
        sensors_expanded1 = sensor_locations.unsqueeze(2)  # [B, n_sensors, 1, 2]
        sensors_expanded2 = sensor_locations.unsqueeze(1)  # [B, 1, n_sensors, 2]

        # Compute L2 distances
        distances = torch.norm(sensors_expanded1 - sensors_expanded2, dim=-1)  # [B, n_sensors, n_sensors]

        # Mask out diagonal elements (self-distances)
        mask = torch.eye(n_sensors, device=sensor_locations.device).unsqueeze(0).expand(B, -1, -1)
        distances = distances * (1 - mask)

        # Penalize sensors that are too close (encourage diversity)
        min_distance = 0.1  # Minimum desired distance
        close_penalty = F.relu(min_distance - distances)
        diversity_loss = torch.sum(close_penalty) / (B * n_sensors * (n_sensors - 1))

        # Penalize sensors near boundaries (encourage interior placement)
        boundary_penalty = (
            F.relu(0.1 - sensor_locations[:, :, 0]).sum() +  # Left boundary
            F.relu(sensor_locations[:, :, 0] - 0.9).sum() +  # Right boundary
            F.relu(0.1 - sensor_locations[:, :, 1]).sum() +  # Bottom boundary
            F.relu(sensor_locations[:, :, 1] - 0.9).sum()    # Top boundary
        ) / (B * n_sensors)

        total_loss = diversity_loss + 0.1 * boundary_penalty
        return self.weight * total_loss


class DeepONetLoss(nn.Module):
    """
    Composite loss function for DeepONet training.

    Combines multiple loss components with configurable weights.
    """

    def __init__(self,
                 operator_loss_type: str = 'l2',
                 operator_weight: float = 1.0,
                 physics_weight: float = 0.1,
                 gradient_weight: float = 0.01,
                 sensor_reg_weight: float = 0.001,
                 spectral_weight: float = 0.0,
                 **loss_kwargs):
        super().__init__()

        self.operator_weight = operator_weight
        self.physics_weight = physics_weight
        self.gradient_weight = gradient_weight
        self.sensor_reg_weight = sensor_reg_weight
        self.spectral_weight = spectral_weight

        # Initialize operator loss
        if operator_loss_type == 'l2':
            self.operator_loss = OperatorL2Loss(**loss_kwargs)
        elif operator_loss_type == 'relative_l2':
            self.operator_loss = RelativeL2Loss(**loss_kwargs)
        elif operator_loss_type == 'huber':
            self.operator_loss = HuberLoss(**loss_kwargs)
        else:
            raise ValueError(f"Unknown operator loss type: {operator_loss_type}")

        # Initialize additional losses
        if self.physics_weight > 0:
            self.physics_loss = PhysicsInformedLoss()

        if self.gradient_weight > 0:
            self.gradient_loss = GradientLoss()

        if self.sensor_reg_weight > 0:
            self.sensor_reg_loss = SensorRegularizationLoss()

        if self.spectral_weight > 0:
            self.spectral_loss = SpectralLoss()

    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor,
                input_sensors: torch.Tensor,
                query_coords: torch.Tensor,
                model: nn.Module,
                epoch: int = 0,
                is_training: bool = True,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute composite loss for DeepONet training.

        Args:
            predictions: Model predictions [B, T, n_query, C]
            targets: Target values [B, T, n_query, C]
            input_sensors: Sensor values [B, T, n_sensors, C]
            query_coords: Query coordinates [B, n_query, 2]
            model: DeepONet model
            epoch: Current training epoch
            is_training: Whether in training mode

        Returns:
            losses: Dictionary containing all loss components
        """
        losses = {}

        # Primary operator loss
        op_loss = self.operator_loss(predictions, targets)
        losses['operator_loss'] = op_loss

        # Physics-informed losses
        if (self.physics_weight > 0 and
            hasattr(self, 'physics_loss') and
            isinstance(model, PhysicsInformedDeepONet)):

            # Need to reconstruct input functions for physics loss
            # This is a placeholder - in practice, you'd want the original functions
            input_functions = torch.zeros(
                predictions.shape[0], predictions.shape[1],
                predictions.shape[3], 64, 64,  # Placeholder H, W
                device=predictions.device
            )

            physics_losses = self.physics_loss(
                model, predictions, targets, query_coords, input_functions
            )
            losses.update(physics_losses)

            # Sum all physics losses
            total_physics_loss = sum(physics_losses.values()) if physics_losses else torch.tensor(0.0)
            losses['physics_loss'] = total_physics_loss
        else:
            losses['physics_loss'] = torch.tensor(0.0)

        # Gradient loss
        if self.gradient_weight > 0 and hasattr(self, 'gradient_loss'):
            grad_loss = self.gradient_loss(predictions, targets, query_coords)
            losses['gradient_loss'] = grad_loss

        # Sensor regularization loss
        if (self.sensor_reg_weight > 0 and
            hasattr(self, 'sensor_reg_loss') and
            hasattr(model, 'sensor_locations')):
            sensor_locations = model.sensor_locations.unsqueeze(0).expand(
                predictions.shape[0], -1, -1
            )
            sensor_loss = self.sensor_reg_loss(sensor_locations)
            losses['sensor_reg_loss'] = sensor_loss

        # Spectral loss
        if self.spectral_weight > 0 and hasattr(self, 'spectral_loss'):
            # Only compute spectral loss if data represents spatial grid
            if predictions.shape[2] == targets.shape[2]:  # Same number of query points
                spec_loss = self.spectral_loss(predictions, targets)
                losses['spectral_loss'] = spec_loss

        # Compute total weighted loss
        total_loss = self.operator_weight * losses['operator_loss']

        if 'physics_loss' in losses:
            total_loss += self.physics_weight * losses['physics_loss']

        if 'gradient_loss' in losses:
            total_loss += self.gradient_weight * losses['gradient_loss']

        if 'sensor_reg_loss' in losses:
            total_loss += self.sensor_reg_weight * losses['sensor_reg_loss']

        if 'spectral_loss' in losses:
            total_loss += self.spectral_weight * losses['spectral_loss']

        losses['total_loss'] = total_loss

        return losses

    def get_loss_info(self) -> Dict[str, float]:
        """Get information about loss configuration."""
        return {
            'operator_weight': self.operator_weight,
            'physics_weight': self.physics_weight,
            'gradient_weight': self.gradient_weight,
            'sensor_reg_weight': self.sensor_reg_weight,
            'spectral_weight': self.spectral_weight,
            'operator_loss_type': type(self.operator_loss).__name__
        }


def create_deeponet_loss(loss_config: Dict[str, any]) -> DeepONetLoss:
    """
    Factory function to create DeepONet loss from configuration.

    Args:
        loss_config: Dictionary with loss configuration

    Returns:
        DeepONetLoss instance
    """
    return DeepONetLoss(**loss_config)