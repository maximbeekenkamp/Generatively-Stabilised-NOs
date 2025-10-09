"""
Radial Basis Function Kolmogorov-Arnold Network (RBF-KAN) Layers

This module implements RBF-KAN layers for operator learning in DeepOKAN.
The implementation is preserved exactly from the reference code with only
documentation additions.

KAN networks replace traditional MLPs with learnable basis functions, providing
improved expressiveness for operator learning tasks.

Reference: DeepOKAN Example_3_transient/KAN_RBF6.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class RadialBasisFunctionNetwork(nn.Module):
    """
    RBF-KAN Network: Stack of RBF-KAN layers with tanh activations.

    This network uses radial basis functions (Gaussian RBFs) as the learnable
    basis instead of standard linear transformations. The first layer uses
    custom grid bounds, while subsequent layers use [-1, 1].

    Args:
        hidden_layers: List of layer dimensions [input_dim, hidden1, ..., output_dim]
        min_grid: Minimum grid value for first layer RBF centers
        max_grid: Maximum grid value for first layer RBF centers
        grid_count: Number of RBF centers per input dimension
        apply_base_update: Whether to add a learnable base transformation
        activation: Activation function for base updates
        grid_opt: Whether to optimize RBF center positions
        init_scale: Scale for weight initialization
        dtype: Data type for parameters
    """

    def __init__(self, hidden_layers: List[int], min_grid: float = -1., max_grid: float = 1.,
                 grid_count: int = 5, apply_base_update: bool = False, activation: nn.Module = nn.SiLU(),
                 grid_opt: bool = False, init_scale: float = 0.1, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.layers = nn.ModuleList()

        # First hidden layer with custom grid bounds
        self.layers.append(RadialBasisFunctionLayer(
            hidden_layers[0], hidden_layers[1], min_grid, max_grid,
            grid_count, apply_base_update, activation, grid_opt, init_scale, dtype))

        # Remaining hidden layers with standard [-1, 1] grid
        for in_dim, out_dim in zip(hidden_layers[1:-1], hidden_layers[2:]):
            self.layers.append(RadialBasisFunctionLayer(
                in_dim, out_dim, -1., 1., grid_count, apply_base_update,
                activation, grid_opt, init_scale, dtype))

        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through RBF-KAN network.

        Args:
            x: Input tensor

        Returns:
            output: Transformed tensor
        """
        x = x.to(self.dtype)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply tanh activation after all hidden layers except the last one
            if i < len(self.layers) - 1:
                x = torch.tanh(x)
        return x


class RadialBasisFunctionLayer(nn.Module):
    """
    Single RBF-KAN Layer.

    This layer replaces a standard linear transformation with a radial basis
    function (RBF) expansion. Each input dimension is expanded using Gaussian
    RBFs centered on a learnable grid, then linearly combined to produce outputs.

    Mathematical formulation:
        y = W * φ(x) + b * σ(W_b * x)  [if apply_base_update=True]
        y = W * φ(x)                    [if apply_base_update=False]

    where φ(x) are Gaussian RBF basis functions:
        φ_i(x) = exp(-((x - c_i) / σ)²)

    Args:
        in_features: Input dimension
        out_features: Output dimension
        min_grid: Minimum RBF center position
        max_grid: Maximum RBF center position
        grid_count: Number of RBF centers per input dimension
        apply_base_update: Add learnable base transformation
        activation: Activation for base updates
        grid_opt: Optimize RBF center positions
        init_scale: Weight initialization scale
        dtype: Data type for parameters
    """

    def __init__(self, in_features: int, out_features: int, min_grid: float = -1., max_grid: float = 1.,
                 grid_count: int = 5, apply_base_update: bool = False, activation: nn.Module = nn.SiLU(),
                 grid_opt: bool = False, init_scale: float = 0.1, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.apply_base_update = apply_base_update
        self.activation = activation
        self.min_grid = min_grid
        self.max_grid = max_grid
        self.grid_count = grid_count

        # RBF centers (grid points)
        self.grid = nn.Parameter(
            torch.linspace(min_grid, max_grid, grid_count, dtype=dtype),
            requires_grad=grid_opt
        )

        # RBF weights: [in_features * grid_count, out_features]
        self.rbf_weight = nn.Parameter(
            torch.randn(in_features * grid_count, out_features, dtype=dtype) * init_scale
        )

        # Optional base linear layer
        self.base_linear = nn.Linear(in_features, out_features, dtype=dtype) if apply_base_update else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through RBF-KAN layer.

        Args:
            x: Input tensor [B, in_features]

        Returns:
            output: Transformed tensor [B, out_features]
        """
        x = x.to(self.rbf_weight.dtype)

        # Compute RBF basis functions
        x_unsqueezed = x.unsqueeze(-1)  # [B, in_features, 1]

        # Gaussian RBF: exp(-((x - c) / σ)²)
        # σ = (max_grid - min_grid) / (grid_count - 1)
        rbf_basis = torch.exp(
            -((x_unsqueezed - self.grid) / ((self.max_grid - self.min_grid) / (self.grid_count - 1))) ** 2
        )  # [B, in_features, grid_count]

        # Flatten: [B, in_features * grid_count]
        rbf_basis = rbf_basis.view(rbf_basis.size(0), -1)

        # Linear combination: [B, in_features * grid_count] × [in_features * grid_count, out_features]
        rbf_output = torch.einsum('bi,ij->bj', rbf_basis, self.rbf_weight)  # [B, out_features]

        # Optional base update
        if self.apply_base_update:
            base_output = self.base_linear(self.activation(x))
            rbf_output += base_output

        return rbf_output
