"""
DeepOKAN: Deep Operator Network with Kolmogorov-Arnold Networks

This module implements DeepOKAN, which replaces the standard MLP networks in
DeepONet with Kolmogorov-Arnold Networks (KAN) using radial basis functions (RBF).
KAN networks have shown improved performance in operator learning tasks.

Reference: DeepOKAN original implementation (Example_3_transient)

Key differences from standard DeepONet:
1. RBF-KAN networks instead of MLPs for branch and trunk
2. Shared (non-batched) trunk network: [H*W, 2] not [B, H*W, 2]
3. Einsum combination: 'bik,nk->bni' instead of element-wise multiply

Components:
- RadialBasisFunctionNetwork: RBF-KAN network implementation
- RadialBasisFunctionLayer: Individual RBF-KAN layer
- MLPNetwork: MLP baseline for comparison
- DeepOKAN: Core architecture combining branch-KAN and trunk-KAN
- DeepOKANConfig: Configuration for model instantiation
- DeepOKANFormatAdapter: Converts [B,T,C,H,W] to branch-trunk format
"""

from .kan_layers import RadialBasisFunctionNetwork, RadialBasisFunctionLayer
from .mlp_networks import MLPNetwork
from .deepokan_base import DeepOKAN
from .deepokan_config import DeepOKANConfig
from .deepokan_adapter import DeepOKANFormatAdapter

__all__ = [
    'RadialBasisFunctionNetwork',
    'RadialBasisFunctionLayer',
    'MLPNetwork',
    'DeepOKAN',
    'DeepOKANConfig',
    'DeepOKANFormatAdapter'
]
