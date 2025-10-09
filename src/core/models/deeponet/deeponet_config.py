"""
DeepONet Configuration for Gen Stabilised Framework

This module defines the configuration class for DeepONet models, providing
integration with the Gen Stabilised parameter system while maintaining
compatibility with the reference implementation.
"""

from dataclasses import dataclass
from typing import Tuple
from src.core.utils.params import ModelParamsDecoder, DataParams


@dataclass
class DeepONetConfig:
    """
    Configuration for DeepONet models.

    This configuration supports per-channel processing to handle multi-field
    spatiotemporal data in the [B,T,C,H,W] format.

    Attributes:
        sensor_dim_per_channel: Number of spatial points (H×W) for one channel
        num_channels: Number of solution fields/channels
        coord_dim: Coordinate dimension (2 for 2D spatial, 3 for 3D)
        latent_features: Hidden dimension for branch-trunk combination
        branch_hidden: Hidden layer size for branch MLP
        branch_layers: Number of hidden layers in branch MLP
        trunk_hidden: Hidden layer size for trunk MLP
        trunk_layers: Number of hidden layers in trunk MLP
    """

    sensor_dim_per_channel: int
    num_channels: int
    coord_dim: int
    latent_features: int
    branch_hidden: int
    branch_layers: int
    trunk_hidden: int
    trunk_layers: int

    @classmethod
    def from_params(cls, p_md: ModelParamsDecoder, p_d: DataParams) -> 'DeepONetConfig':
        """
        Create DeepONet configuration from Gen Stabilised parameters.

        This method extracts relevant parameters from the existing parameter
        system and creates a DeepONet configuration. The configuration supports
        per-channel processing for handling multiple fields.

        Args:
            p_md: Model decoder parameters
            p_d: Data parameters

        Returns:
            DeepONetConfig: Configuration object

        Example:
            >>> config = DeepONetConfig.from_params(p_md, p_d)
            >>> print(f"Sensor dim: {config.sensor_dim_per_channel}")
            >>> print(f"Num channels: {config.num_channels}")
        """
        # Extract spatial dimensions
        H, W = p_d.dataSize[-2], p_d.dataSize[-1]

        # Determine number of solution fields (exclude parameter channels)
        num_channels = len(p_d.simFields) if p_d.simFields else 2

        # Get decoder width or use default
        decoder_width = p_md.decWidth if hasattr(p_md, 'decWidth') and p_md.decWidth else 128

        return cls(
            sensor_dim_per_channel=H * W,
            num_channels=num_channels,
            coord_dim=2,  # 2D spatial coordinates (x, y)
            latent_features=decoder_width,
            branch_hidden=decoder_width,
            branch_layers=4,  # Reference implementation uses 4 hidden layers
            trunk_hidden=decoder_width,
            trunk_layers=4,
        )

    def get_branch_config(self) -> Tuple[int, int, int, int]:
        """
        Get branch network configuration.

        Returns:
            Tuple of (in_features, out_features, hidden_features, num_hidden_layers)
        """
        return (
            self.sensor_dim_per_channel,  # For single channel processing
            self.latent_features,
            self.branch_hidden,
            self.branch_layers
        )

    def get_trunk_config(self) -> Tuple[int, int, int, int]:
        """
        Get trunk network configuration.

        Returns:
            Tuple of (in_features, out_features, hidden_features, num_hidden_layers)
        """
        return (
            self.coord_dim,
            self.latent_features,
            self.trunk_hidden,
            self.trunk_layers
        )

    def __repr__(self) -> str:
        """String representation for logging."""
        return (
            f"DeepONetConfig(\n"
            f"  sensor_dim_per_channel={self.sensor_dim_per_channel},\n"
            f"  num_channels={self.num_channels},\n"
            f"  coord_dim={self.coord_dim},\n"
            f"  latent_features={self.latent_features},\n"
            f"  branch: [{self.sensor_dim_per_channel} → {self.branch_hidden}×{self.branch_layers} → {self.latent_features}],\n"
            f"  trunk: [{self.coord_dim} → {self.trunk_hidden}×{self.trunk_layers} → {self.latent_features}]\n"
            f")"
        )
