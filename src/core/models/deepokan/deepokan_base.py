"""
DeepOKAN: DeepONet with Kolmogorov-Arnold Networks

This module implements the core DeepOKAN architecture, which replaces standard
MLP networks with RBF-KAN networks in the DeepONet framework.

Key architectural difference from standard DeepONet:
- Trunk network is SHARED (not batched): [N_points, coord_dim] not [B, N_points, coord_dim]
- Uses einsum for combining branch and trunk outputs: 'bik,nk->bni'

Reference: DeepOKAN Example_3_transient/final_rbfkan_transient_poisson.py lines 129-148
"""

import torch
import torch.nn as nn
from typing import Optional
from .kan_layers import RadialBasisFunctionNetwork as RBFKAN
from .deepokan_config import DeepOKANConfig


class DeepOKAN(nn.Module):
    """
    DeepOKAN: Deep Operator Network with KAN layers.

    This architecture uses RBF-KAN networks for both branch and trunk instead
    of standard MLPs. The key difference from standard DeepONet is that the
    trunk network is shared across the batch (not batched).

    Mathematical formulation:
        Y[b,n,i] = sum_k Branch[b,i,k] * Trunk[n,k]

    where:
        - b: batch index
        - n: spatial point index
        - i: sensor/input dimension index
        - k: hidden dimension index

    Architecture flow:
    1. Branch: [B, sensor_dim] → RBF-KAN → [B, sensor_dim*HD]
    2. Reshape: [B, sensor_dim*HD] → [B, sensor_dim, HD]
    3. Trunk: [N_points, coord_dim] → RBF-KAN → [N_points, HD]
    4. Einsum: 'bik,nk->bni' → [B, N_points, sensor_dim]

    For per-channel processing (single channel at a time):
    - sensor_dim = H*W (spatial points)
    - Output: [B, H*W, H*W] which we take diagonal → [B, H*W, 1]

    Args:
        config: DeepOKANConfig instance
    """

    def __init__(self, config: DeepOKANConfig):
        super().__init__()
        self.config = config

        # Branch KAN network
        self.branch_net = RBFKAN(
            hidden_layers=config.branch_width,
            dtype=config.dtype,
            apply_base_update=False,
            init_scale=config.init_scale,
            min_grid=config.branch_min_grid,
            max_grid=config.branch_max_grid,
            grid_count=config.grid_count,
            grid_opt=config.grid_opt
        )

        # Trunk KAN network
        self.trunk_net = RBFKAN(
            hidden_layers=config.trunk_width,
            dtype=config.dtype,
            apply_base_update=False,
            init_scale=config.init_scale,
            min_grid=config.trunk_min_grid,
            max_grid=config.trunk_max_grid,
            grid_count=config.grid_count,
            grid_opt=config.grid_opt
        )

    def forward(self, x_branch: torch.Tensor, x_trunk: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DeepOKAN.

        This implementation follows the reference code exactly, using einsum
        to combine branch and trunk outputs.

        Args:
            x_branch: Branch input [B, sensor_dim] - discretized field for one channel
            x_trunk: Trunk input [N_points, coord_dim] - SHARED coordinates (NOT batched!)

        Returns:
            Y: Operator output [B, N_points, 1] - single channel prediction

        Implementation details:
            1. Branch processes the input field → [B, sensor_dim*HD]
            2. Reshape to [B, sensor_dim, HD] to separate sensors and hidden dims
            3. Trunk processes coordinates → [N_points, HD]
            4. Einsum combines them: [B,sensor_dim,HD] × [N_points,HD] → [B,N_points,sensor_dim]
            5. For single-channel: take diagonal to get [B, N_points, 1]
        """
        # Branch network: [B, sensor_dim] → [B, sensor_dim*HD]
        y_branch = self.branch_net(x_branch)

        # Reshape: [B, sensor_dim*HD] → [B, sensor_dim, HD]
        # This separates the sensor dimension from the hidden dimension
        y_branch = y_branch.view(-1, self.config.sensor_dim, self.config.HD)

        # Trunk network: [N_points, coord_dim] → [N_points, HD]
        # Note: Trunk is NOT batched - it's shared across all samples in the batch
        y_trunk = self.trunk_net(x_trunk)

        # Check dimension compatibility
        assert y_branch.shape[-1] == y_trunk.shape[-1], \
            f"Branch output HD={y_branch.shape[-1]} != Trunk output HD={y_trunk.shape[-1]}"

        # Einsum combination (from reference code line 146)
        # 'bik,nk->bni': [B,sensor_dim,HD] × [N_points,HD] → [B,N_points,sensor_dim]
        Y = torch.einsum('bik,nk->bni', y_branch, y_trunk)

        # For single-channel processing: extract diagonal
        # When sensor_dim == N_points, the diagonal gives us the prediction at each point
        if self.config.sensor_dim == Y.shape[1]:
            # Take diagonal: sensor i predicts output at point i
            # [B, N_points, N_points] → [B, N_points]
            Y = torch.diagonal(Y, dim1=1, dim2=2).unsqueeze(-1)  # [B, N_points, 1]
        else:
            # If dimensions don't match, keep full tensor
            # This would happen if using different sensor and query point counts
            pass

        return Y

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"DeepOKAN(\n"
            f"  sensor_dim={self.config.sensor_dim},\n"
            f"  coord_dim={self.config.coord_dim},\n"
            f"  HD={self.config.HD},\n"
            f"  branch_width={self.config.branch_width},\n"
            f"  trunk_width={self.config.trunk_width},\n"
            f"  grid_count={self.config.grid_count}\n"
            f")"
        )
