"""
DeepONet Format Adapter for Gen Stabilised Framework

This module provides the adapter layer that converts data from the Gen Stabilised
[B,T,C,H,W] format to DeepONet's branch-trunk input format and back.

The adapter uses per-channel processing to match the reference implementation,
where each field/channel is processed independently through DeepONet.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class DeepONetFormatAdapter(nn.Module):
    """
    Adapter for converting [B,T,C,H,W] format to DeepONet branch-trunk format.

    This adapter implements per-channel processing, which means each field
    (e.g., velocity_x, velocity_y) is processed independently through DeepONet.
    This matches the reference implementation pattern where the reference code
    processes single channels at a time (e.g., loads[:,:,1]).

    Key architectural decisions:
    1. Per-channel processing: Each of C channels processed separately
    2. Batched trunk: Trunk coordinates are [B, H*W, 2] (batched)
    3. Branch input: [B, 1, H*W] for each channel (the '1' enables broadcasting)
    4. Coordinate normalization: [0, 1] range (verified from reference)

    Args:
        deeponet_model: The DeepONet model instance
        spatial_dims: Tuple of (H, W) spatial dimensions
        num_channels: Number of solution fields to process
    """

    def __init__(self, deeponet_model: nn.Module, spatial_dims: Tuple[int, int], num_channels: int):
        super().__init__()
        self.deeponet = deeponet_model
        self.H, self.W = spatial_dims
        self.num_channels = num_channels
        self._coord_cache = None

    def _get_coordinates(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """
        Generate normalized [0,1] coordinate grid for trunk network.

        The coordinates are cached for efficiency since they don't change
        between forward passes (similar to TNO's coordinate caching).

        Args:
            H: Height of spatial grid
            W: Width of spatial grid
            device: Device to place coordinates on

        Returns:
            coords: Coordinate tensor [H*W, 2] containing (x, y) pairs
        """
        if self._coord_cache is None or self._coord_cache.device != device:
            # Normalize to [0, 1] range (verified from reference implementation)
            x = torch.linspace(0, 1, W, device=device)
            y = torch.linspace(0, 1, H, device=device)
            yy, xx = torch.meshgrid(y, x, indexing='ij')

            # Flatten to [H*W, 2]
            coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)
            self._coord_cache = coords

        return self._coord_cache

    def forward(self, x: torch.Tensor, simParams: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass: Convert [B,C,H,W] to branch-trunk format, process, and convert back.

        The forward pass processes each channel independently:
        1. For each channel c in [0, min(C, num_channels)):
           a. Extract channel: [B, C, H, W] → [B, 1, H*W]
           b. Generate trunk coords: [H*W, 2] → [B, H*W, 2]
           c. DeepONet forward: ([B, H*W, 2], [B, 1, H*W]) → [B, H*W, 1]
        2. Stack channel outputs: [B, C, H*W]
        3. Reshape: [B, C, H*W] → [B, C, H, W]

        Args:
            x: Input tensor [B, C, H, W] - single timestep
            simParams: Optional simulation parameters (not used by DeepONet, but required for interface compatibility)

        Returns:
            pred: Prediction tensor [B, C, H, W]
        """
        B, C, H, W = x.shape
        device = x.device

        # Generate trunk coordinates [H*W, 2] → [B, H*W, 2] (batched)
        trunk_coords = self._get_coordinates(H, W, device)
        trunk_input = trunk_coords.unsqueeze(0).expand(B, -1, -1)

        # Process each channel separately (matches reference pattern)
        channel_outputs = []
        for c in range(min(C, self.num_channels)):
            # Branch input: [B, C, H, W] → [B, 1, H*W] for this channel
            # The '1' dimension is INTENTIONAL (from reference implementation)
            branch_input = x[:, c:c+1, :, :].reshape(B, 1, -1)

            # DeepONet forward pass
            # Input: trunk [B, H*W, 2], branch [B, 1, H*W]
            # Output: [B, H*W, 1]
            output = self.deeponet(trunk_input, branch_input)

            # Remove output channel dimension: [B, H*W, 1] → [B, H*W]
            channel_outputs.append(output.squeeze(-1))

        # Stack channels: [B, C, H*W]
        output = torch.stack(channel_outputs, dim=1)

        # Reshape to spatial: [B, C, H*W] → [B, C, H, W]
        output = output.reshape(B, -1, H, W)

        # Handle extra channels (e.g., simulation parameters)
        # If input has more channels than we process, preserve them
        if C > self.num_channels:
            # Keep original parameter channels unchanged
            output = torch.cat([output, x[:, self.num_channels:]], dim=1)

        return output

    def get_format_info(self) -> dict:
        """
        Get information about the format conversion.

        Returns:
            dict: Format information including dimensions and processing details
        """
        return {
            'spatial_dims': (self.H, self.W),
            'num_channels': self.num_channels,
            'sensor_dim_per_channel': self.H * self.W,
            'coord_dim': 2,
            'processing_mode': 'per_channel',
            'trunk_batched': True,  # Standard DeepONet batches trunk
            'branch_shape': f'[B, 1, {self.H * self.W}]',
            'trunk_shape': f'[B, {self.H * self.W}, 2]',
            'output_shape': f'[B, {self.H * self.W}, 1]'
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"DeepONetFormatAdapter(\n"
            f"  spatial_dims=({self.H}, {self.W}),\n"
            f"  num_channels={self.num_channels},\n"
            f"  sensor_dim_per_channel={self.H * self.W},\n"
            f"  processing_mode='per_channel'\n"
            f")"
        )
