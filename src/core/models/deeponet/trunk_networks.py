"""
Trunk Network Implementations for DeepONet

Trunk networks encode query coordinates to provide spatial/temporal basis functions.
Different variants handle different coordinate systems and physics-informed constraints.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class TrunkNetwork(nn.Module, ABC):
    """Abstract base class for DeepONet trunk networks."""

    @abstractmethod
    def forward(self, y_coords: torch.Tensor) -> torch.Tensor:
        """
        Encode query coordinates to basis functions.

        Args:
            y_coords: Query coordinates [B*n_query, spatial_dim]

        Returns:
            trunk_output: Basis functions [B*n_query, latent_dim]
        """
        pass

    @abstractmethod
    def get_input_dim(self) -> int:
        """Get expected input dimension (spatial dimension)."""
        pass

    @abstractmethod
    def get_output_dim(self) -> int:
        """Get output dimension (latent dimension)."""
        pass


class DenseTrunkNetwork(TrunkNetwork):
    """
    Dense (fully connected) trunk network.

    Standard implementation using multilayer perceptron to encode
    query coordinates with optional positional encoding.
    """

    def __init__(self,
                 spatial_dim: int,
                 latent_dim: int,
                 hidden_layers: List[int] = None,
                 activation: str = "gelu",
                 dropout: float = 0.1,
                 batch_norm: bool = True,
                 positional_encoding: bool = True,
                 pe_frequencies: int = 10):
        super().__init__()

        self.spatial_dim = spatial_dim
        self.latent_dim = latent_dim
        self.activation_name = activation
        self.dropout_rate = dropout
        self.use_batch_norm = batch_norm
        self.use_pe = positional_encoding
        self.pe_frequencies = pe_frequencies

        # Default hidden layers
        if hidden_layers is None:
            hidden_layers = [64, 128]

        # Determine input dimension with positional encoding
        if positional_encoding:
            # Each coordinate gets sin/cos encoding for each frequency
            input_dim = spatial_dim * (2 * pe_frequencies + 1)  # +1 for original coordinate
        else:
            input_dim = spatial_dim

        # Build layer dimensions
        layer_dims = [input_dim] + hidden_layers + [latent_dim]

        # Activation function
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "silu":
            self.activation = nn.SiLU()
        elif activation.lower() == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build layers
        layers = []
        for i in range(len(layer_dims) - 1):
            # Linear layer
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))

            # Skip activation and normalization for last layer
            if i < len(layer_dims) - 2:
                # Batch normalization
                if batch_norm:
                    layers.append(nn.BatchNorm1d(layer_dims[i + 1]))

                # Activation
                layers.append(self.activation)

                # Dropout
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def positional_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding to coordinates.

        Args:
            coords: Input coordinates [N, spatial_dim]

        Returns:
            encoded_coords: Positionally encoded coordinates [N, encoded_dim]
        """
        if not self.use_pe:
            return coords

        N = coords.shape[0]
        device = coords.device

        # Create frequency basis
        freqs = torch.arange(self.pe_frequencies, device=device, dtype=torch.float32)
        freqs = 2 ** freqs  # Exponential frequency scaling

        encoded_parts = [coords]  # Start with original coordinates

        for dim in range(self.spatial_dim):
            coord_dim = coords[:, dim:dim+1]  # [N, 1]

            # Apply each frequency
            for freq in freqs:
                encoded_parts.append(torch.sin(2 * math.pi * freq * coord_dim))
                encoded_parts.append(torch.cos(2 * math.pi * freq * coord_dim))

        return torch.cat(encoded_parts, dim=1)

    def forward(self, y_coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dense trunk network.

        Args:
            y_coords: Query coordinates [B*n_query, spatial_dim]

        Returns:
            trunk_output: Basis functions [B*n_query, latent_dim]
        """
        if y_coords.shape[-1] != self.spatial_dim:
            raise ValueError(f"Expected {self.spatial_dim} spatial dimensions, got {y_coords.shape[-1]}")

        # Apply positional encoding
        encoded_coords = self.positional_encoding(y_coords)

        # Forward through network
        return self.network(encoded_coords)

    def get_input_dim(self) -> int:
        return self.spatial_dim

    def get_output_dim(self) -> int:
        return self.latent_dim


class FourierTrunkNetwork(TrunkNetwork):
    """
    Fourier-enhanced trunk network for periodic coordinate systems.

    Uses learnable Fourier features to better capture periodic boundary
    conditions and coordinate dependencies.
    """

    def __init__(self,
                 spatial_dim: int,
                 latent_dim: int,
                 fourier_modes: int = 64,
                 hidden_layers: List[int] = None,
                 activation: str = "gelu",
                 dropout: float = 0.1,
                 batch_norm: bool = True,
                 learnable_frequencies: bool = True):
        super().__init__()

        self.spatial_dim = spatial_dim
        self.latent_dim = latent_dim
        self.fourier_modes = fourier_modes
        self.learnable_frequencies = learnable_frequencies

        # Default hidden layers
        if hidden_layers is None:
            hidden_layers = [128, 256]

        # Input dimension: original coords + Fourier features
        fourier_dim = spatial_dim * fourier_modes * 2  # cos + sin for each mode
        input_dim = spatial_dim + fourier_dim

        # Activation function
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.GELU()

        # Fourier frequency parameters
        if learnable_frequencies:
            # Learnable frequencies for each spatial dimension and mode
            self.fourier_freqs = nn.Parameter(
                torch.randn(spatial_dim, fourier_modes) * 2 * math.pi
            )
        else:
            # Fixed exponential frequencies
            freqs = torch.arange(fourier_modes, dtype=torch.float32)
            freqs = 2 ** freqs  # Exponential scaling
            freqs = freqs.unsqueeze(0).expand(spatial_dim, -1)  # [spatial_dim, fourier_modes]
            self.register_buffer('fourier_freqs', freqs)

        # Build dense network
        layer_dims = [input_dim] + hidden_layers + [latent_dim]
        layers = []

        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))

            if i < len(layer_dims) - 2:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(layer_dims[i + 1]))
                layers.append(self.activation)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def compute_fourier_features(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute Fourier features from coordinates.

        Args:
            coords: Coordinates [N, spatial_dim]

        Returns:
            fourier_features: Fourier features [N, spatial_dim * fourier_modes * 2]
        """
        N = coords.shape[0]
        features = []

        for dim in range(self.spatial_dim):
            coord_dim = coords[:, dim]  # [N]

            for mode in range(self.fourier_modes):
                freq = self.fourier_freqs[dim, mode]

                # Compute sine and cosine features
                sin_feature = torch.sin(freq * coord_dim)  # [N]
                cos_feature = torch.cos(freq * coord_dim)  # [N]

                features.extend([sin_feature, cos_feature])

        return torch.stack(features, dim=1)  # [N, spatial_dim * fourier_modes * 2]

    def forward(self, y_coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Fourier trunk network.

        Args:
            y_coords: Query coordinates [B*n_query, spatial_dim]

        Returns:
            trunk_output: Basis functions [B*n_query, latent_dim]
        """
        # Compute Fourier features
        fourier_features = self.compute_fourier_features(y_coords)

        # Concatenate original coordinates with Fourier features
        combined_features = torch.cat([y_coords, fourier_features], dim=1)

        # Forward through network
        return self.network(combined_features)

    def get_input_dim(self) -> int:
        return self.spatial_dim

    def get_output_dim(self) -> int:
        return self.latent_dim


class PhysicsAwareTrunkNetwork(TrunkNetwork):
    """
    Physics-aware trunk network with domain-specific coordinate handling.

    Incorporates physics-based coordinate transformations and constraints
    specific to different physical domains (fluid dynamics, heat transfer, etc.).
    """

    def __init__(self,
                 spatial_dim: int,
                 latent_dim: int,
                 physics_type: str = "general",
                 hidden_layers: List[int] = None,
                 activation: str = "gelu",
                 dropout: float = 0.1,
                 batch_norm: bool = True,
                 boundary_encoding: bool = True):
        super().__init__()

        self.spatial_dim = spatial_dim
        self.latent_dim = latent_dim
        self.physics_type = physics_type.lower()
        self.boundary_encoding = boundary_encoding

        # Default hidden layers
        if hidden_layers is None:
            hidden_layers = [64, 128, 256]

        # Physics-specific coordinate transformations
        self.coord_transform_dim = self._get_transform_dim()
        input_dim = self.coord_transform_dim

        # Activation function
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.GELU()

        # Physics-specific learnable parameters
        self._init_physics_parameters()

        # Build dense network
        layer_dims = [input_dim] + hidden_layers + [latent_dim]
        layers = []

        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))

            if i < len(layer_dims) - 2:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(layer_dims[i + 1]))
                layers.append(self.activation)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _get_transform_dim(self) -> int:
        """Get dimension after physics-specific coordinate transformation."""
        if self.physics_type == "fluid":
            # Fluid dynamics: add velocity potential, vorticity features
            return self.spatial_dim + 4  # x, y + u_potential, v_potential, vorticity, pressure_basis
        elif self.physics_type == "heat":
            # Heat transfer: add thermal features
            return self.spatial_dim + 2  # x, y + temperature_basis, heat_flux_basis
        elif self.physics_type == "wave":
            # Wave propagation: add wave-specific features
            return self.spatial_dim + 3  # x, y + wave_speed_basis, dispersion_basis, reflection_basis
        else:
            # General physics: minimal transformation
            return self.spatial_dim + 1  # x, y + distance_from_boundary

    def _init_physics_parameters(self):
        """Initialize physics-specific learnable parameters."""
        if self.physics_type == "fluid":
            # Reynolds number scaling, viscosity effects
            self.reynolds_scale = nn.Parameter(torch.tensor(1.0))
            self.viscosity_scale = nn.Parameter(torch.tensor(0.1))
        elif self.physics_type == "heat":
            # Thermal diffusivity, Prandtl number effects
            self.thermal_diffusivity = nn.Parameter(torch.tensor(0.1))
            self.prandtl_scale = nn.Parameter(torch.tensor(1.0))
        elif self.physics_type == "wave":
            # Wave speed, dispersion parameters
            self.wave_speed = nn.Parameter(torch.tensor(1.0))
            self.dispersion_coeff = nn.Parameter(torch.tensor(0.1))

    def _physics_transform(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Apply physics-specific coordinate transformation.

        Args:
            coords: Raw coordinates [N, spatial_dim]

        Returns:
            transformed_coords: Physics-aware coordinates [N, transform_dim]
        """
        N = coords.shape[0]

        if self.physics_type == "fluid":
            # Fluid dynamics transformations
            x, y = coords[:, 0], coords[:, 1]

            # Velocity potential basis (for irrotational flow)
            u_potential = torch.sin(2 * math.pi * x) * torch.cos(2 * math.pi * y)
            v_potential = -torch.cos(2 * math.pi * x) * torch.sin(2 * math.pi * y)

            # Vorticity basis
            vorticity = 4 * math.pi * torch.sin(2 * math.pi * x) * torch.sin(2 * math.pi * y)

            # Pressure gradient basis
            pressure_basis = torch.cos(math.pi * x) * torch.cos(math.pi * y)

            return torch.stack([
                x, y,
                u_potential * self.reynolds_scale,
                v_potential * self.reynolds_scale,
                vorticity * self.viscosity_scale,
                pressure_basis
            ], dim=1)

        elif self.physics_type == "heat":
            # Heat transfer transformations
            x, y = coords[:, 0], coords[:, 1]

            # Temperature basis functions
            temp_basis = torch.exp(-(x**2 + y**2) * self.thermal_diffusivity)

            # Heat flux basis
            flux_basis = (x**2 + y**2) * self.prandtl_scale

            return torch.stack([x, y, temp_basis, flux_basis], dim=1)

        elif self.physics_type == "wave":
            # Wave propagation transformations
            x, y = coords[:, 0], coords[:, 1]
            r = torch.sqrt(x**2 + y**2)

            # Wave speed basis
            speed_basis = torch.sin(self.wave_speed * r)

            # Dispersion basis
            dispersion_basis = torch.cos(self.dispersion_coeff * r**2)

            # Reflection basis (boundary effects)
            reflection_basis = torch.exp(-r / 0.1)  # Exponential decay from boundaries

            return torch.stack([x, y, speed_basis, dispersion_basis, reflection_basis], dim=1)

        else:
            # General physics: add distance from boundary
            x, y = coords[:, 0], coords[:, 1]

            # Simple distance from domain boundary [0,1]^2
            dist_from_boundary = torch.minimum(
                torch.minimum(x, 1-x),
                torch.minimum(y, 1-y)
            )

            return torch.stack([x, y, dist_from_boundary], dim=1)

    def _initialize_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use smaller initialization for physics networks
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, y_coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through physics-aware trunk network.

        Args:
            y_coords: Query coordinates [B*n_query, spatial_dim]

        Returns:
            trunk_output: Physics-aware basis functions [B*n_query, latent_dim]
        """
        # Apply physics-specific coordinate transformation
        transformed_coords = self._physics_transform(y_coords)

        # Forward through network
        return self.network(transformed_coords)

    def get_input_dim(self) -> int:
        return self.spatial_dim

    def get_output_dim(self) -> int:
        return self.latent_dim

    def get_physics_info(self) -> dict:
        """Get physics-specific parameter information."""
        info = {
            'physics_type': self.physics_type,
            'transform_dim': self.coord_transform_dim
        }

        if self.physics_type == "fluid":
            info.update({
                'reynolds_scale': self.reynolds_scale.item(),
                'viscosity_scale': self.viscosity_scale.item()
            })
        elif self.physics_type == "heat":
            info.update({
                'thermal_diffusivity': self.thermal_diffusivity.item(),
                'prandtl_scale': self.prandtl_scale.item()
            })
        elif self.physics_type == "wave":
            info.update({
                'wave_speed': self.wave_speed.item(),
                'dispersion_coeff': self.dispersion_coeff.item()
            })

        return info