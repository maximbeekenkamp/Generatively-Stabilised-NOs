"""
DeepONet Variants and Specialized Architectures

This module provides pre-configured DeepONet variants combining different
branch and trunk networks for specific applications and physics domains.
"""

from typing import Dict, Optional, Any, List
import torch
import torch.nn as nn

from .deeponet_base import DeepONet, DeepONetConfig
from .branch_networks import DenseBranchNetwork, ConvolutionalBranchNetwork, FourierBranchNetwork
from .trunk_networks import DenseTrunkNetwork, FourierTrunkNetwork, PhysicsAwareTrunkNetwork
from src.core.utils.params import ModelParamsDecoder, DataParams


class StandardDeepONet(DeepONet):
    """
    Standard DeepONet with dense branch and trunk networks.

    Best for general operator learning tasks without specific
    domain constraints or periodic behavior.
    """

    def __init__(self,
                 config: DeepONetConfig,
                 p_md: Optional[ModelParamsDecoder] = None,
                 p_d: Optional[DataParams] = None):
        # Create dense branch network
        branch_network = DenseBranchNetwork(
            n_sensors=config.n_sensors,
            latent_dim=config.latent_dim,
            hidden_layers=config.branch_layers,
            activation=config.branch_activation,
            dropout=config.branch_dropout,
            batch_norm=config.branch_batch_norm
        )

        # Create dense trunk network
        trunk_network = DenseTrunkNetwork(
            spatial_dim=2,  # Assume 2D for now
            latent_dim=config.latent_dim,
            hidden_layers=config.trunk_layers,
            activation=config.trunk_activation,
            dropout=config.trunk_dropout,
            batch_norm=config.trunk_batch_norm,
            positional_encoding=config.positional_encoding
        )

        super().__init__(branch_network, trunk_network, config, p_md, p_d)

    @classmethod
    def from_config(cls, **kwargs) -> 'StandardDeepONet':
        """Create StandardDeepONet from configuration parameters."""
        config = DeepONetConfig(**kwargs)
        return cls(config)


class FourierDeepONet(DeepONet):
    """
    Fourier-enhanced DeepONet for periodic functions and domains.

    Uses Fourier features in both branch and trunk networks for
    improved handling of periodic boundary conditions and functions.
    """

    def __init__(self,
                 config: DeepONetConfig,
                 branch_fourier_modes: int = 32,
                 trunk_fourier_modes: int = 64,
                 p_md: Optional[ModelParamsDecoder] = None,
                 p_d: Optional[DataParams] = None):
        # Create Fourier branch network
        branch_network = FourierBranchNetwork(
            n_sensors=config.n_sensors,
            latent_dim=config.latent_dim,
            fourier_modes=branch_fourier_modes,
            hidden_layers=config.branch_layers,
            activation=config.branch_activation,
            dropout=config.branch_dropout,
            batch_norm=config.branch_batch_norm
        )

        # Create Fourier trunk network
        trunk_network = FourierTrunkNetwork(
            spatial_dim=2,
            latent_dim=config.latent_dim,
            fourier_modes=trunk_fourier_modes,
            hidden_layers=config.trunk_layers,
            activation=config.trunk_activation,
            dropout=config.trunk_dropout,
            batch_norm=config.trunk_batch_norm
        )

        super().__init__(branch_network, trunk_network, config, p_md, p_d)
        self.branch_fourier_modes = branch_fourier_modes
        self.trunk_fourier_modes = trunk_fourier_modes

    @classmethod
    def from_config(cls, branch_fourier_modes: int = 32, trunk_fourier_modes: int = 64, **kwargs) -> 'FourierDeepONet':
        """Create FourierDeepONet from configuration parameters."""
        config = DeepONetConfig(**kwargs)
        return cls(config, branch_fourier_modes, trunk_fourier_modes)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information including Fourier parameters."""
        info = super().get_model_info()
        info.update({
            'model_variant': 'FourierDeepONet',
            'branch_fourier_modes': self.branch_fourier_modes,
            'trunk_fourier_modes': self.trunk_fourier_modes
        })
        return info


class PhysicsInformedDeepONet(DeepONet):
    """
    Physics-informed DeepONet with domain-specific constraints.

    Incorporates physics-aware coordinate encoding and can be extended
    with physics-informed loss functions for PDE constraints.
    """

    def __init__(self,
                 config: DeepONetConfig,
                 physics_type: str = "general",
                 use_physics_loss: bool = True,
                 p_md: Optional[ModelParamsDecoder] = None,
                 p_d: Optional[DataParams] = None):
        # Create dense branch network (can be enhanced later)
        branch_network = DenseBranchNetwork(
            n_sensors=config.n_sensors,
            latent_dim=config.latent_dim,
            hidden_layers=config.branch_layers,
            activation=config.branch_activation,
            dropout=config.branch_dropout,
            batch_norm=config.branch_batch_norm
        )

        # Create physics-aware trunk network
        trunk_network = PhysicsAwareTrunkNetwork(
            spatial_dim=2,
            latent_dim=config.latent_dim,
            physics_type=physics_type,
            hidden_layers=config.trunk_layers,
            activation=config.trunk_activation,
            dropout=config.trunk_dropout,
            batch_norm=config.trunk_batch_norm
        )

        super().__init__(branch_network, trunk_network, config, p_md, p_d)
        self.physics_type = physics_type
        self.use_physics_loss = use_physics_loss

    def compute_physics_loss(self,
                           x: torch.Tensor,
                           output: torch.Tensor,
                           query_coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute physics-informed loss terms.

        Args:
            x: Input functions [B, T, C, H, W]
            output: Model output [B, T, n_query, C]
            query_coords: Query coordinates [B, n_query, 2]

        Returns:
            physics_losses: Dictionary of physics loss components
        """
        losses = {}

        if not self.use_physics_loss:
            return losses

        # Enable gradient computation for query coordinates
        query_coords = query_coords.clone().detach().requires_grad_(True)

        # Re-compute output with gradient tracking
        output_with_grad = self.forward(x, query_coords)

        if self.physics_type == "fluid":
            losses.update(self._compute_fluid_losses(output_with_grad, query_coords))
        elif self.physics_type == "heat":
            losses.update(self._compute_heat_losses(output_with_grad, query_coords))
        elif self.physics_type == "wave":
            losses.update(self._compute_wave_losses(output_with_grad, query_coords))

        return losses

    def _compute_fluid_losses(self, output: torch.Tensor, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute fluid dynamics PDE losses (Navier-Stokes)."""
        # Compute gradients
        grads = torch.autograd.grad(
            outputs=output.sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True
        )[0]

        # Extract velocity components (assuming output represents [u, v])
        u = output[..., 0]  # x-velocity
        v = output[..., 1] if output.shape[-1] > 1 else torch.zeros_like(u)  # y-velocity

        # Compute divergence (continuity equation): ∇·u = 0
        div_u = grads[..., 0, 0] + grads[..., 1, 1]  # ∂u/∂x + ∂v/∂y
        continuity_loss = torch.mean(div_u**2)

        return {
            'continuity_loss': continuity_loss
        }

    def _compute_heat_losses(self, output: torch.Tensor, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute heat equation PDE losses."""
        # Heat equation: ∂T/∂t = α∇²T
        # For steady state: ∇²T = 0 (Laplace equation)

        grads = torch.autograd.grad(
            outputs=output.sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True
        )[0]

        # Compute second derivatives (Laplacian)
        grad2 = torch.autograd.grad(
            outputs=grads.sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True
        )[0]

        # Laplacian: ∂²T/∂x² + ∂²T/∂y²
        laplacian = grad2[..., 0, 0] + grad2[..., 1, 1]
        laplace_loss = torch.mean(laplacian**2)

        return {
            'laplace_loss': laplace_loss
        }

    def _compute_wave_losses(self, output: torch.Tensor, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute wave equation PDE losses."""
        # Wave equation: ∂²u/∂t² = c²∇²u
        # For harmonic solutions: -ω²u = c²∇²u

        grads = torch.autograd.grad(
            outputs=output.sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True
        )[0]

        grad2 = torch.autograd.grad(
            outputs=grads.sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True
        )[0]

        # Helmholtz equation approximation
        wave_speed = 1.0  # Can be made learnable
        omega = 1.0      # Frequency parameter

        laplacian = grad2[..., 0, 0] + grad2[..., 1, 1]
        helmholtz_residual = laplacian + (omega/wave_speed)**2 * output[..., 0]
        helmholtz_loss = torch.mean(helmholtz_residual**2)

        return {
            'helmholtz_loss': helmholtz_loss
        }

    @classmethod
    def from_config(cls, physics_type: str = "general", use_physics_loss: bool = True, **kwargs) -> 'PhysicsInformedDeepONet':
        """Create PhysicsInformedDeepONet from configuration parameters."""
        config = DeepONetConfig(**kwargs)
        return cls(config, physics_type, use_physics_loss)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information including physics parameters."""
        info = super().get_model_info()
        info.update({
            'model_variant': 'PhysicsInformedDeepONet',
            'physics_type': self.physics_type,
            'use_physics_loss': self.use_physics_loss
        })

        # Add physics-specific parameters from trunk network
        if hasattr(self.trunk_network, 'get_physics_info'):
            info.update(self.trunk_network.get_physics_info())

        return info


class MultiScaleDeepONet(DeepONet):
    """
    Multi-scale DeepONet for handling multiple resolution levels.

    Uses multiple trunk networks operating at different scales
    to capture both fine and coarse features in the operator.
    """

    def __init__(self,
                 config: DeepONetConfig,
                 n_scales: int = 3,
                 scale_factors: List[float] = None,
                 p_md: Optional[ModelParamsDecoder] = None,
                 p_d: Optional[DataParams] = None):
        self.n_scales = n_scales
        self.scale_factors = scale_factors or [1.0, 2.0, 4.0]  # Different frequency scales

        # Create single branch network
        branch_network = DenseBranchNetwork(
            n_sensors=config.n_sensors,
            latent_dim=config.latent_dim,
            hidden_layers=config.branch_layers,
            activation=config.branch_activation,
            dropout=config.branch_dropout,
            batch_norm=config.branch_batch_norm
        )

        # Create multiple trunk networks at different scales
        trunk_networks = nn.ModuleList()
        for i, scale in enumerate(self.scale_factors[:n_scales]):
            trunk_net = FourierTrunkNetwork(
                spatial_dim=2,
                latent_dim=config.latent_dim // n_scales,  # Split latent dimension
                fourier_modes=32,
                hidden_layers=[layer // n_scales for layer in config.trunk_layers],
                activation=config.trunk_activation,
                dropout=config.trunk_dropout,
                batch_norm=config.trunk_batch_norm,
                learnable_frequencies=True
            )

            # Scale the frequencies
            with torch.no_grad():
                trunk_net.fourier_freqs.data *= scale

            trunk_networks.append(trunk_net)

        # Use first trunk network for base class initialization
        super().__init__(branch_network, trunk_networks[0], config, p_md, p_d)

        # Replace single trunk with multi-scale trunks
        self.trunk_networks = trunk_networks
        self.trunk_network = None  # Disable single trunk

    def forward(self, x: torch.Tensor, query_coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through multi-scale DeepONet.

        Args:
            x: Input tensor [B, T, C, H, W]
            query_coords: Optional query coordinates [B, n_query, 2]

        Returns:
            output: Operator output [B, T, C, H, W] or [B, T, n_query, C]
        """
        if not self.validate_input_shape(x):
            raise ValueError(f"Invalid input shape: {x.shape}")

        B, T, C, H, W = x.shape
        device = x.device

        # Sample input functions at sensor locations
        u_sensors = self.sample_functions_at_sensors(x)  # [B, T, n_sensors, C]

        # Prepare query coordinates
        if query_coords is None:
            h_coords = torch.linspace(0, 1, H, device=device)
            w_coords = torch.linspace(0, 1, W, device=device)
            hh, ww = torch.meshgrid(h_coords, w_coords, indexing='ij')
            query_coords = torch.stack([hh.flatten(), ww.flatten()], dim=1)
            query_coords = query_coords.unsqueeze(0).expand(B, -1, -1)
            n_query = H * W
            return_grid = True
        else:
            n_query = query_coords.shape[1]
            return_grid = False

        # Process each time step
        outputs = []
        for t in range(T):
            u_t = u_sensors[:, t]  # [B, n_sensors, C]

            # Branch network: encode function values at sensors
            branch_features = []
            for c in range(C):
                u_c = u_t[:, :, c]  # [B, n_sensors]
                if self.input_norm is not None:
                    u_c = self.input_norm(u_c)
                branch_out = self.branch_network(u_c)  # [B, latent_dim]
                branch_features.append(branch_out)

            # Average over channels
            branch_output = torch.stack(branch_features, dim=1).mean(dim=1)  # [B, latent_dim]

            # Multi-scale trunk networks
            trunk_outputs = []
            coords_flat = query_coords.view(B * n_query, 2)  # [B*n_query, 2]

            for trunk_net in self.trunk_networks:
                trunk_out = trunk_net(coords_flat)  # [B*n_query, latent_dim//n_scales]
                trunk_out = trunk_out.view(B, n_query, -1)  # [B, n_query, latent_dim//n_scales]
                trunk_outputs.append(trunk_out)

            # Concatenate multi-scale trunk outputs
            trunk_combined = torch.cat(trunk_outputs, dim=2)  # [B, n_query, latent_dim]

            # Compute dot product
            branch_expanded = branch_output.unsqueeze(1)  # [B, 1, latent_dim]
            dot_product = (branch_expanded * trunk_combined).sum(dim=2)  # [B, n_query]

            # Add bias if present
            if self.bias is not None:
                dot_product = dot_product + self.bias

            outputs.append(dot_product)

        # Stack time steps
        output = torch.stack(outputs, dim=1)  # [B, T, n_query]
        output = output.unsqueeze(3)  # [B, T, n_query, 1]

        # Expand to match input channels
        if C > 1:
            output = output.expand(-1, -1, -1, C)

        # Reshape to spatial grid if needed
        if return_grid:
            output = output.view(B, T, H, W, C)
            output = output.permute(0, 1, 4, 2, 3)  # [B, T, C, H, W]

        return output

    @classmethod
    def from_config(cls, n_scales: int = 3, scale_factors: List[float] = None, **kwargs) -> 'MultiScaleDeepONet':
        """Create MultiScaleDeepONet from configuration parameters."""
        config = DeepONetConfig(**kwargs)
        return cls(config, n_scales, scale_factors)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information including multi-scale parameters."""
        info = super().get_model_info()
        info.update({
            'model_variant': 'MultiScaleDeepONet',
            'n_scales': self.n_scales,
            'scale_factors': self.scale_factors
        })
        return info