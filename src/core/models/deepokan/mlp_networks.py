"""
MLP Network Baseline for DeepOKAN

This module implements a standard MLP network as a baseline for comparison
with RBF-KAN networks. The implementation is preserved from the reference code.

Reference: DeepOKAN Example_3_transient/MLPNet.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union


class MLPNetwork(nn.Module):
    """
    Standard Multi-Layer Perceptron network.

    This network serves as a baseline for comparing against RBF-KAN networks
    in operator learning tasks. It uses standard linear layers with activation
    functions and Xavier initialization.

    Args:
        input_dim: Input dimension
        hidden_layers: List of hidden layer dimensions
        output_dim: Output dimension
        activation: Activation function (default: SiLU)
        dtype: Data type for parameters

    Example:
        >>> mlp = MLPNetwork(100, [128, 128, 64], 50)
        >>> x = torch.randn(32, 100)
        >>> y = mlp(x)  # Output shape: [32, 50]
    """

    def __init__(self, input_dim: int, hidden_layers: List[int], output_dim: int,
                 activation: nn.Module = nn.SiLU(), dtype: torch.dtype = torch.float32):
        super(MLPNetwork, self).__init__()
        layers = []
        in_dim = input_dim

        # Build hidden layers
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim, dtype=dtype))
            layers.append(activation)
            in_dim = hidden_dim

        # Output layer (no activation)
        layers.append(nn.Linear(in_dim, output_dim, dtype=dtype))

        self.network = nn.Sequential(*layers)

        # Apply Xavier initialization
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP.

        Args:
            x: Input tensor

        Returns:
            output: Transformed tensor
        """
        return self.network(x)

    def _init_weights(self, module):
        """
        Initialize weights using Xavier normal initialization.

        Args:
            module: Module to initialize
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
