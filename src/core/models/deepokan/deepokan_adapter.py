"""
DeepOKAN Format Adapter for Gen Stabilised Framework

This module provides the adapter layer that converts data from Gen Stabilised
[B,T,C,H,W] format to DeepOKAN's branch-trunk input format.

Critical architectural difference from DeepONet:
- Trunk is SHARED across batch: [H*W, 2] not [B, H*W, 2]
- This matches the reference implementation where trunk processes coordinates
  independently of the batch dimension.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class DeepOKANFormatAdapter(nn.Module):
    """
    Adapter for DeepOKAN with per-channel processing and shared trunk.

    DeepOKAN's architecture requires special handling compared to standard DeepONet:
    1. Trunk is shared (NOT batched): [H*W, 2] for all samples
    2. Branch is per-sample: [B, H*W] for each channel
    3. Einsum combination produces [B, H*W, H*W] → extract diagonal → [B, H*W, 1]

    This design follows the reference implementation where the trunk network
    processes query coordinates once and reuses the output across all samples.

    Args:
        deepokan_model: The DeepOKAN model instance
        spatial_dims: Tuple of (H, W) spatial dimensions
        num_channels: Number of solution fields to process
    """

    def __init__(self, deepokan_model: nn.Module, spatial_dims: Tuple[int, int], num_channels: int):
        super().__init__()
        self.deepokan = deepokan_model
        self.H, self.W = spatial_dims
        self.num_channels = num_channels
        self._coord_cache = None

    def _get_coordinates(self, H: int, W: int, device: torch.device, dtype: torch.dtype = None) -> torch.Tensor:
        """
        Generate shared coordinate grid for trunk network.

        CRITICAL: Returns [H*W, 2] NOT [B, H*W, 2] - trunk is shared across batch!

        The coordinates are cached since they're reused across forward passes.
        Normalization is [0, 1] range (verified from reference lines 101-102).

        Args:
            H: Height of spatial grid
            W: Width of spatial grid
            device: Device to place coordinates on
            dtype: Data type for coordinates (defaults to float64 for DeepOKAN internal precision)

        Returns:
            coords: Coordinate tensor [H*W, 2] - SHARED, not batched!
        """
        # Use float64 by default for DeepOKAN (KAN benefits from higher precision)
        if dtype is None:
            dtype = torch.float64

        if self._coord_cache is None or self._coord_cache.device != device or self._coord_cache.dtype != dtype:
            # Normalize to [0, 1] range (matches reference implementation)
            x = torch.linspace(0, 1, W, device=device, dtype=dtype)
            y = torch.linspace(0, 1, H, device=device, dtype=dtype)
            yy, xx = torch.meshgrid(y, x, indexing='ij')

            # Flatten to [H*W, 2] - NO batch dimension!
            coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)
            self._coord_cache = coords

        return self._coord_cache

    def forward(self, x: torch.Tensor, simParams: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass: Convert [B,C,H,W] to branch-trunk format with shared trunk.

        Processing flow (per channel):
        1. Extract channel: [B, C, H, W] → [B, H*W] for channel c
        2. Generate shared trunk coords: [H*W, 2] (NOT batched!)
        3. DeepOKAN forward: ([B, H*W], [H*W, 2]) → [B, H*W, 1]
        4. Stack channels: [B, C, H*W]
        5. Reshape: [B, C, H*W] → [B, C, H, W]

        Args:
            x: Input tensor [B, C, H, W] - single timestep
            simParams: Optional simulation parameters (not used by DeepOKAN, but required for interface compatibility)

        Returns:
            pred: Prediction tensor [B, C, H, W]
        """
        B, C, H, W = x.shape
        device = x.device
        input_dtype = x.dtype  # Save input dtype for output conversion

        # Generate shared trunk coordinates [H*W, 2] - NOT batched!
        # Use float64 for DeepOKAN internal precision
        trunk_input = self._get_coordinates(H, W, device, dtype=torch.float64)

        # Process each channel separately
        channel_outputs = []
        for c in range(min(C, self.num_channels)):
            # Branch input: [B, C, H, W] → [B, H*W] for this channel
            # Convert to float64 for KAN internal processing
            branch_input = x[:, c, :, :].reshape(B, -1).to(torch.float64)

            # DeepOKAN forward pass
            # Input: branch [B, H*W], trunk [H*W, 2] (shared!)
            # Output: [B, H*W, 1] in float64
            output = self.deepokan(branch_input, trunk_input)

            # Remove output channel dimension: [B, H*W, 1] → [B, H*W]
            channel_outputs.append(output.squeeze(-1))

        # Stack channels: [B, C, H*W]
        output = torch.stack(channel_outputs, dim=1)

        # Reshape to spatial: [B, C, H*W] → [B, C, H, W]
        output = output.reshape(B, -1, H, W)

        # Handle extra channels (e.g., simulation parameters)
        if C > self.num_channels:
            # Keep original parameter channels unchanged
            output = torch.cat([output, x[:, self.num_channels:]], dim=1)

        # Ensure output matches input dtype for framework compatibility
        # DeepOKAN uses float64 internally for KAN stability, but output must match input
        if output.dtype != input_dtype:
            output = output.to(input_dtype)

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
            'sensor_dim': self.H * self.W,
            'coord_dim': 2,
            'processing_mode': 'per_channel',
            'trunk_batched': False,  # KEY DIFFERENCE: DeepOKAN trunk is SHARED!
            'branch_shape': f'[B, {self.H * self.W}]',
            'trunk_shape': f'[{self.H * self.W}, 2]',  # No batch dimension!
            'output_shape': f'[B, {self.H * self.W}, 1]',
            'einsum_formula': 'bik,nk->bni'
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"DeepOKANFormatAdapter(\n"
            f"  spatial_dims=({self.H}, {self.W}),\n"
            f"  num_channels={self.num_channels},\n"
            f"  sensor_dim={self.H * self.W},\n"
            f"  processing_mode='per_channel',\n"
            f"  trunk_mode='shared' (NOT batched!)\n"
            f")"
        )
