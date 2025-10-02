"""
DeepONet Format Adapter for Gen Stabilised Framework

This module provides utilities to convert between Gen Stabilised tensor format
[B,T,C,H,W] and DeepONet's expected input/output format.

DeepONet expects:
- Branch input: [B, feature_size] - flattened function values
- Trunk input: [B, spatial_points, coordinate_dim] - spatial coordinates
- Output: [B, spatial_points, output_channels]

Gen Stabilised format:
- Input/Output: [B, T, C, H, W] - batched spatio-temporal data
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn


class DeepONetFormatAdapter:
    """
    Adapter to convert between Gen Stabilised format and DeepONet format.

    This adapter handles the conversion logic that's currently repeated
    throughout the DeepONet tests, making it reusable and maintainable.
    """

    def __init__(self, spatial_dims: Tuple[int, int], coordinate_dim: int = 2):
        """
        Initialize the format adapter.

        Args:
            spatial_dims: (H, W) spatial dimensions
            coordinate_dim: Dimension of coordinates (2 for 2D, 3 for 3D)
        """
        self.H, self.W = spatial_dims
        self.coordinate_dim = coordinate_dim
        self.spatial_points = self.H * self.W

        # Pre-compute coordinate grid for efficiency
        self._trunk_input_template = self._create_coordinate_grid()

    def _create_coordinate_grid(self) -> torch.Tensor:
        """Create normalized coordinate grid for trunk input."""
        if self.coordinate_dim == 2:
            x = torch.linspace(0, 1, self.H)
            y = torch.linspace(0, 1, self.W)
            grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
            coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        elif self.coordinate_dim == 3:
            # For 3D case, add a dummy z coordinate
            x = torch.linspace(0, 1, self.H)
            y = torch.linspace(0, 1, self.W)
            z = torch.zeros(self.H, self.W)  # Dummy z for 2D spatial data
            grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
            coords = torch.stack([grid_x.flatten(), grid_y.flatten(), z.flatten()], dim=-1)
        else:
            raise ValueError(f"coordinate_dim {self.coordinate_dim} not supported")

        return coords  # [H*W, coordinate_dim]

    def to_deeponet_format(self,
                          gen_stabilised_input: torch.Tensor,
                          use_last_timestep: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert Gen Stabilised format [B,T,C,H,W] to DeepONet format.

        Args:
            gen_stabilised_input: Input tensor [B, T, C, H, W]
            use_last_timestep: If True, use last timestep for branch input.
                              If False, flatten all timesteps.

        Returns:
            branch_input: [B, feature_size] - flattened function values
            trunk_input: [B, spatial_points, coordinate_dim] - coordinates
        """
        if len(gen_stabilised_input.shape) != 5:
            raise ValueError(f"Expected [B,T,C,H,W] format, got shape {gen_stabilised_input.shape}")

        B, T, C, H, W = gen_stabilised_input.shape

        if H != self.H or W != self.W:
            raise ValueError(f"Spatial dims mismatch: expected ({self.H}, {self.W}), got ({H}, {W})")

        # Create branch input (flattened function values)
        if use_last_timestep:
            # Use last timestep: [B, C, H, W] → [B, C*H*W]
            branch_input = gen_stabilised_input[:, -1].flatten(1)
        else:
            # Use all timesteps: [B, T, C, H, W] → [B, T*C*H*W]
            branch_input = gen_stabilised_input.flatten(1)

        # Create trunk input (coordinate grid)
        trunk_input = self._trunk_input_template.unsqueeze(0).expand(B, -1, -1)
        trunk_input = trunk_input.to(gen_stabilised_input.device)

        return branch_input, trunk_input

    def from_deeponet_format(self,
                           deeponet_output: torch.Tensor,
                           target_shape: Tuple[int, int, int, int, int]) -> torch.Tensor:
        """
        Convert DeepONet output back to Gen Stabilised format.

        Args:
            deeponet_output: [B, spatial_points, output_channels]
            target_shape: Target shape (B, T, C, H, W)

        Returns:
            gen_stabilised_output: [B, T, C, H, W]
        """
        if len(deeponet_output.shape) != 3:
            raise ValueError(f"Expected [B, spatial_points, output_channels], got {deeponet_output.shape}")

        B_out, spatial_points, output_channels = deeponet_output.shape
        B, T, C, H, W = target_shape

        if spatial_points != self.spatial_points:
            raise ValueError(f"Spatial points mismatch: expected {self.spatial_points}, got {spatial_points}")

        if B_out != B:
            raise ValueError(f"Batch size mismatch: expected {B}, got {B_out}")

        if output_channels != C:
            raise ValueError(f"Channel mismatch: expected {C}, got {output_channels}")

        # Reshape from [B, H*W, C] to [B, T, C, H, W]
        # Note: DeepONet produces single timestep, so we replicate for T timesteps
        output_spatial = deeponet_output.reshape(B, H, W, C)  # [B, H, W, C]
        output_spatial = output_spatial.permute(0, 3, 1, 2)   # [B, C, H, W]

        # Replicate across time dimension
        gen_stabilised_output = output_spatial.unsqueeze(1).expand(-1, T, -1, -1, -1)

        return gen_stabilised_output

    def get_branch_input_size(self, C: int, use_last_timestep: bool = True, T: int = 1) -> int:
        """Get the expected branch input size for a given number of channels."""
        if use_last_timestep:
            return C * self.H * self.W
        else:
            return T * C * self.H * self.W

    def get_trunk_input_size(self) -> Tuple[int, int]:
        """Get trunk input dimensions: (spatial_points, coordinate_dim)."""
        return self.spatial_points, self.coordinate_dim

    def get_expected_output_size(self, C: int) -> Tuple[int, int]:
        """Get expected DeepONet output dimensions: (spatial_points, output_channels)."""
        return self.spatial_points, C


class DeepONetWrapper(nn.Module):
    """
    Wrapper to make DeepONet compatible with Gen Stabilised format.

    This wrapper automatically handles format conversion, allowing DeepONet
    to be used seamlessly in the Gen Stabilised framework.
    """

    def __init__(self, deeponet_model, spatial_dims: Tuple[int, int], coordinate_dim: int = 2):
        """
        Initialize wrapper.

        Args:
            deeponet_model: The actual DeepONet model
            spatial_dims: (H, W) spatial dimensions
            coordinate_dim: Coordinate dimension for trunk input
        """
        super().__init__()
        self.deeponet_model = deeponet_model
        self.adapter = DeepONetFormatAdapter(spatial_dims, coordinate_dim)

    def forward(self, x: torch.Tensor, unused=None) -> torch.Tensor:
        """
        Forward pass with automatic format conversion.

        Args:
            x: Input in Gen Stabilised format [B, T, C, H, W] or [B, C, H, W]
            unused: Unused parameter for compatibility with PredictionModel interface

        Returns:
            Output in Gen Stabilised format matching input (4D or 5D)
        """
        # Handle both [B, T, C, H, W] and [B, C, H, W] formats
        if len(x.shape) == 4:
            # [B, C, H, W] - add dummy time dimension
            x = x.unsqueeze(1)  # [B, 1, C, H, W]
            squeeze_output = True
        elif len(x.shape) == 5:
            squeeze_output = False
        else:
            raise ValueError(f"Expected 4D [B,C,H,W] or 5D [B,T,C,H,W] input, got {x.shape}")

        # DeepONet can directly handle [B, T, C, H, W] format
        # It internally samples at sensors and evaluates on query grid
        gen_stabilised_output = self.deeponet_model(x)

        # Remove time dimension if input was 4D
        if squeeze_output:
            gen_stabilised_output = gen_stabilised_output.squeeze(1)

        return gen_stabilised_output

    def get_format_info(self) -> dict:
        """Get information about format conversion."""
        return {
            'spatial_dims': (self.adapter.H, self.adapter.W),
            'coordinate_dim': self.adapter.coordinate_dim,
            'spatial_points': self.adapter.spatial_points,
        }