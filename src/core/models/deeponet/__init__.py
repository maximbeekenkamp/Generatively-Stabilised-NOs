"""
DeepONet: Deep Operator Network

This module implements DeepONet (Deep Operator Network) for operator learning
in the Gen Stabilised framework. DeepONet learns mappings between function spaces
using a branch-trunk architecture.

Reference: Lu et al., "Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators"

Components:
- MLP: Multi-layer perceptron with tanh activations
- DeepONet: Branch-trunk operator network
- DeepONetConfig: Configuration for model instantiation
- DeepONetFormatAdapter: Converts [B,T,C,H,W] to branch-trunk format
"""

from .mlp_networks import MLP, DeepONet
from .deeponet_config import DeepONetConfig
from .deeponet_adapter import DeepONetFormatAdapter

__all__ = [
    'MLP',
    'DeepONet',
    'DeepONetConfig',
    'DeepONetFormatAdapter'
]
