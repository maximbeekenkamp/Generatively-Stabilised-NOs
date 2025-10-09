"""
DeepOKAN Configuration

This module defines the configuration for DeepOKAN models, matching the
reference implementation hyperparameters while integrating with the Gen
Stabilised parameter system.
"""

from dataclasses import dataclass
import torch
from typing import List
from src.core.utils.params import ModelParamsDecoder, DataParams


@dataclass
class DeepOKANConfig:
    """
    Configuration for DeepOKAN models matching reference implementation.

    DeepOKAN uses RBF-KAN networks with specific hyperparameters derived
    from the reference implementation. These values have been validated
    for operator learning tasks.

    Attributes:
        sensor_dim: H*W for single channel (spatial points)
        coord_dim: Coordinate dimension (2 for 2D spatial)
        HD: Hidden dimension for branch-trunk combination (from reference)
        hid_trunk: Trunk network hidden layer size
        num_layer_trunk: Number of trunk hidden layers
        hid_branch: Branch network hidden layer size
        num_layer_branch: Number of branch hidden layers
        grid_count: Number of RBF centers per input dimension
        grid_opt: Whether to optimize RBF center positions
        init_scale: Weight initialization scale
        dtype: Data type (float64 recommended for KAN stability)
        trunk_min_grid: Min RBF center for trunk (coordinates in [0,1])
        trunk_max_grid: Max RBF center for trunk
        branch_min_grid: Min RBF center for branch (data normalized)
        branch_max_grid: Max RBF center for branch
    """

    sensor_dim: int
    coord_dim: int
    HD: int = 4  # From reference implementation
    hid_trunk: int = 5
    num_layer_trunk: int = 1
    hid_branch: int = 5
    num_layer_branch: int = 1
    grid_count: int = 5
    grid_opt: bool = False
    init_scale: float = 0.01
    dtype: torch.dtype = torch.float64  # KAN networks benefit from higher precision
    trunk_min_grid: float = 0.0  # Coordinates normalized to [0,1]
    trunk_max_grid: float = 1.0
    branch_min_grid: float = -1.0  # Data will be normalized to this range
    branch_max_grid: float = 1.0

    @property
    def trunk_width(self) -> List[int]:
        """
        Trunk network layer sizes.

        Returns:
            List of layer dimensions [coord_dim, hidden, ..., HD]
        """
        return [self.coord_dim] + [self.hid_trunk] * self.num_layer_trunk + [self.HD]

    @property
    def branch_width(self) -> List[int]:
        """
        Branch network layer sizes.

        The output dimension is sensor_dim * HD because the branch output
        will be reshaped to [B, sensor_dim, HD] for einsum combination.

        Returns:
            List of layer dimensions [sensor_dim, hidden, ..., sensor_dim*HD]
        """
        return [self.sensor_dim] + [self.hid_branch] * self.num_layer_branch + [self.sensor_dim * self.HD]

    @classmethod
    def from_params(cls, p_md: ModelParamsDecoder, p_d: DataParams) -> 'DeepOKANConfig':
        """
        Create DeepOKAN configuration from Gen Stabilised parameters.

        This method extracts relevant parameters and creates a configuration
        matching the reference implementation's proven hyperparameters.

        Args:
            p_md: Model decoder parameters
            p_d: Data parameters

        Returns:
            DeepOKANConfig: Configuration object
        """
        # Extract spatial dimensions
        H, W = p_d.dataSize[-2], p_d.dataSize[-1]

        return cls(
            sensor_dim=H * W,  # Spatial points for single channel
            coord_dim=2  # 2D spatial coordinates (x, y)
        )

    def get_branch_config(self) -> dict:
        """
        Get branch network configuration dictionary.

        Returns:
            dict: Branch network parameters
        """
        return {
            'hidden_layers': self.branch_width,
            'min_grid': self.branch_min_grid,
            'max_grid': self.branch_max_grid,
            'grid_count': self.grid_count,
            'apply_base_update': False,
            'grid_opt': self.grid_opt,
            'init_scale': self.init_scale,
            'dtype': self.dtype
        }

    def get_trunk_config(self) -> dict:
        """
        Get trunk network configuration dictionary.

        Returns:
            dict: Trunk network parameters
        """
        return {
            'hidden_layers': self.trunk_width,
            'min_grid': self.trunk_min_grid,
            'max_grid': self.trunk_max_grid,
            'grid_count': self.grid_count,
            'apply_base_update': False,
            'grid_opt': self.grid_opt,
            'init_scale': self.init_scale,
            'dtype': self.dtype
        }

    def __repr__(self) -> str:
        """String representation for logging."""
        return (
            f"DeepOKANConfig(\n"
            f"  sensor_dim={self.sensor_dim} (H*W spatial points),\n"
            f"  coord_dim={self.coord_dim},\n"
            f"  HD={self.HD},\n"
            f"  branch_width={self.branch_width},\n"
            f"  trunk_width={self.trunk_width},\n"
            f"  grid_count={self.grid_count},\n"
            f"  init_scale={self.init_scale},\n"
            f"  dtype={self.dtype}\n"
            f")"
        )
