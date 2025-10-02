"""
Base DeepONet Implementation

This module provides the core DeepONet architecture with branch-trunk decomposition
for learning operators between function spaces.
"""

from typing import Dict, Optional, Any, Tuple, Union, List
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

from ..base_classes import NeuralOperatorPrior, DataFormatHandler
try:
    from src.core.utils.params import ModelParamsDecoder, DataParams
except ImportError:
    try:
        from ...utils.params import ModelParamsDecoder, DataParams
    except ImportError:
        ModelParamsDecoder = None
        DataParams = None


@dataclass
class DeepONetConfig:
    """Configuration for DeepONet architecture."""

    # Architecture parameters
    latent_dim: int = 256
    bias: bool = True
    normalize_inputs: bool = True

    # Branch network configuration
    branch_type: str = "dense"  # "dense", "conv", "fourier"
    branch_layers: List[int] = None
    branch_activation: str = "gelu"
    branch_dropout: float = 0.1
    branch_batch_norm: bool = True

    # Trunk network configuration
    trunk_type: str = "dense"  # "dense", "fourier", "physics_aware"
    trunk_layers: List[int] = None
    trunk_activation: str = "gelu"
    trunk_dropout: float = 0.1
    trunk_batch_norm: bool = True
    positional_encoding: bool = True

    # Sensor configuration
    sensor_strategy: str = "uniform"  # "uniform", "random", "adaptive"
    n_sensors: int = 100
    sensor_locations: Optional[torch.Tensor] = None

    # Training configuration
    n_query_train: int = 1000
    query_sampling: str = "random"  # "random", "uniform", "grid"

    def __post_init__(self):
        """Set default layer configurations if not provided."""
        if self.branch_layers is None:
            self.branch_layers = [128, 256, self.latent_dim]
        if self.trunk_layers is None:
            self.trunk_layers = [64, 128, self.latent_dim]

    @classmethod
    def from_params(cls, p_md: ModelParamsDecoder, p_d: DataParams) -> 'DeepONetConfig':
        """Create configuration from existing parameter system."""

        # Map existing parameters to DeepONet configuration
        config = cls()

        # Use existing width parameters for network sizing
        if hasattr(p_md, 'decWidth') and p_md.decWidth > 0:
            config.latent_dim = p_md.decWidth
            config.branch_layers = [p_md.decWidth//2, p_md.decWidth, p_md.decWidth]
            config.trunk_layers = [p_md.decWidth//4, p_md.decWidth//2, p_md.decWidth]

        # Determine sensor count based on data size
        if hasattr(p_d, 'dataSize') and len(p_d.dataSize) >= 2:
            H, W = p_d.dataSize[-2], p_d.dataSize[-1]
            # Use roughly 1-5% of spatial points as sensors
            config.n_sensors = min(max(50, (H * W) // 20), 1000)

        return config


class DeepONet(NeuralOperatorPrior):
    """
    Deep Operator Network for learning operators between function spaces.

    DeepONet uses a branch-trunk architecture to approximate nonlinear operators:
    G(u)(y) = Σᵢ φᵢ(u) × ψᵢ(y) + b₀

    Where:
    - φᵢ(u): Branch network encoding of input function u at sensor locations
    - ψᵢ(y): Trunk network encoding of query coordinates y
    - b₀: Optional bias term
    """

    def __init__(self,
                 branch_network: nn.Module,
                 trunk_network: nn.Module,
                 config: DeepONetConfig,
                 p_md: Optional[ModelParamsDecoder] = None,
                 p_d: Optional[DataParams] = None):
        super().__init__()

        self.branch_network = branch_network
        self.trunk_network = trunk_network
        self.config = config
        self.p_md = p_md
        self.p_d = p_d
        self.data_handler = DataFormatHandler()

        # Optional bias term
        if config.bias:
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter('bias', None)

        # Input normalization
        if config.normalize_inputs:
            self.input_norm = nn.LayerNorm([config.n_sensors])
            self.coord_norm = nn.LayerNorm([2])  # Assume 2D coordinates for now
        else:
            self.input_norm = None
            self.coord_norm = None

        # Sensor locations (will be initialized during first forward pass)
        self.register_buffer('sensor_locations', torch.zeros(config.n_sensors, 2))
        self._sensors_initialized = False

        self._initialize_weights()

        logging.info(f"DeepONet initialized with {self.get_parameter_count()} parameters")

    def _initialize_weights(self):
        """Initialize network weights using Xavier/Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def initialize_sensors(self, spatial_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Initialize sensor locations based on strategy.

        Args:
            spatial_shape: (H, W) spatial dimensions

        Returns:
            sensor_locations: Tensor of shape [n_sensors, 2] with coordinates
        """
        H, W = spatial_shape
        device = next(self.parameters()).device

        if self.config.sensor_locations is not None:
            # Use provided sensor locations
            sensors = self.config.sensor_locations.to(device)
        elif self.config.sensor_strategy == "uniform":
            # Uniform grid sampling
            n_h = int(np.sqrt(self.config.n_sensors * H / W))
            n_w = int(np.sqrt(self.config.n_sensors * W / H))

            h_coords = torch.linspace(0, 1, n_h, device=device)
            w_coords = torch.linspace(0, 1, n_w, device=device)

            # Create grid and flatten
            hh, ww = torch.meshgrid(h_coords, w_coords, indexing='ij')
            sensors = torch.stack([hh.flatten(), ww.flatten()], dim=1)

            # Trim to exact number of sensors
            sensors = sensors[:self.config.n_sensors]

        elif self.config.sensor_strategy == "random":
            # Random sensor placement
            sensors = torch.rand(self.config.n_sensors, 2, device=device)

        else:
            raise ValueError(f"Unknown sensor strategy: {self.config.sensor_strategy}")

        self.sensor_locations.copy_(sensors)
        self._sensors_initialized = True

        return sensors

    def sample_functions_at_sensors(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sample input functions at sensor locations.

        Args:
            x: Input tensor [B, T, C, H, W]

        Returns:
            u_sensors: Function values at sensors [B, T, n_sensors, C]
        """
        B, T, C, H, W = x.shape

        # Initialize sensors if needed
        if not self._sensors_initialized:
            self.initialize_sensors((H, W))

        # Convert sensor coordinates to pixel indices
        sensor_coords = self.sensor_locations  # [n_sensors, 2], values in [0, 1]
        sensor_h = (sensor_coords[:, 0] * (H - 1)).long()  # [n_sensors]
        sensor_w = (sensor_coords[:, 1] * (W - 1)).long()  # [n_sensors]

        # Sample functions at sensor locations
        # x: [B, T, C, H, W], we want [B, T, C, n_sensors]
        u_sensors = x[:, :, :, sensor_h, sensor_w]  # [B, T, C, n_sensors]
        u_sensors = u_sensors.permute(0, 1, 3, 2)   # [B, T, n_sensors, C]

        return u_sensors

    def forward(self, x: torch.Tensor, query_coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through DeepONet.

        Args:
            x: Input tensor [B, T, C, H, W]
            query_coords: Optional query coordinates [B, n_query, 2].
                         If None, will evaluate on full spatial grid.

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
            # Evaluate on full spatial grid
            h_coords = torch.linspace(0, 1, H, device=device)
            w_coords = torch.linspace(0, 1, W, device=device)
            hh, ww = torch.meshgrid(h_coords, w_coords, indexing='ij')
            query_coords = torch.stack([hh.flatten(), ww.flatten()], dim=1)  # [H*W, 2]
            query_coords = query_coords.unsqueeze(0).expand(B, -1, -1)  # [B, H*W, 2]
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

            # Average over channels or concatenate (for now, average)
            branch_output = torch.stack(branch_features, dim=1).mean(dim=1)  # [B, latent_dim]

            # Trunk network: encode query coordinates
            coords_flat = query_coords.reshape(B * n_query, 2)  # [B*n_query, 2]
            if self.coord_norm is not None:
                coords_flat = self.coord_norm(coords_flat)
            trunk_output = self.trunk_network(coords_flat)  # [B*n_query, latent_dim]
            trunk_output = trunk_output.view(B, n_query, self.config.latent_dim)  # [B, n_query, latent_dim]

            # Compute dot product: branch [B, latent_dim] × trunk [B, n_query, latent_dim]
            branch_expanded = branch_output.unsqueeze(1)  # [B, 1, latent_dim]
            dot_product = (branch_expanded * trunk_output).sum(dim=2)  # [B, n_query]

            # Add bias if present
            if self.bias is not None:
                dot_product = dot_product + self.bias

            outputs.append(dot_product)

        # Stack time steps
        output = torch.stack(outputs, dim=1)  # [B, T, n_query]

        # Add channel dimension for compatibility
        output = output.unsqueeze(3)  # [B, T, n_query, 1]

        # Expand to match input channels
        if C > 1:
            output = output.expand(-1, -1, -1, C)  # [B, T, n_query, C]

        # Reshape to spatial grid if needed
        if return_grid:
            output = output.view(B, T, H, W, C)
            output = output.permute(0, 1, 4, 2, 3)  # [B, T, C, H, W]

        return output

    def get_prior_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features for conditioning generative correctors.

        Args:
            x: Input tensor [B, T, C, H, W]

        Returns:
            features: Dictionary of feature tensors
        """
        # Sample at sensors
        u_sensors = self.sample_functions_at_sensors(x)
        B, T, n_sensors, C = u_sensors.shape

        # Get branch network features for conditioning
        branch_features = []
        for t in range(T):
            t_features = []
            for c in range(C):
                u_c = u_sensors[:, t, :, c]  # [B, n_sensors]
                if self.input_norm is not None:
                    u_c = self.input_norm(u_c)
                branch_out = self.branch_network(u_c)  # [B, latent_dim]
                t_features.append(branch_out)
            branch_features.append(torch.stack(t_features, dim=1))  # [B, C, latent_dim]

        branch_features = torch.stack(branch_features, dim=1)  # [B, T, C, latent_dim]

        return {
            'branch_features': branch_features,
            'sensor_values': u_sensors,
            'sensor_locations': self.sensor_locations.unsqueeze(0).expand(B, -1, -1)
        }

    def validate_input_shape(self, x: torch.Tensor) -> bool:
        """Validate input tensor shape."""
        return (len(x.shape) == 5 and
                x.shape[1] > 0 and  # T > 0
                x.shape[2] > 0 and  # C > 0
                x.shape[3] > 0 and  # H > 0
                x.shape[4] > 0)     # W > 0

    def get_parameter_count(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging and debugging."""
        base_info = super().get_model_info()
        base_info.update({
            'model_class': 'DeepONet',
            'latent_dim': self.config.latent_dim,
            'n_sensors': self.config.n_sensors,
            'branch_type': self.config.branch_type,
            'trunk_type': self.config.trunk_type,
            'has_bias': self.config.bias,
            'sensor_strategy': self.config.sensor_strategy
        })
        return base_info