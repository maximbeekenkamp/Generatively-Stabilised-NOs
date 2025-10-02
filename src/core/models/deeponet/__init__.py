"""
DeepONet (Deep Operator Network) Implementation for Generative Operators

This module implements DeepONet architecture for learning operators in the
Generative Operator Analysis Framework. DeepONet learns mappings between
function spaces using a branch-trunk architecture.

Key Components:
- DeepONet: Main operator learning model
- BranchNetwork: Encodes input functions at sensor locations
- TrunkNetwork: Encodes query coordinates
- Various DeepONet variants (Fourier, Physics-Informed, Multi-Scale)
- DeepONetFactory: Factory for creating configured models

Architecture:
    G(u)(y) = Σᵢ φᵢ(u) × ψᵢ(y) + b₀

Where:
- G: Nonlinear operator to be learned
- u: Input function sampled at sensor locations
- y: Query coordinates where operator is evaluated
- φᵢ(u): Branch network outputs (function encoding)
- ψᵢ(y): Trunk network outputs (coordinate encoding)
- b₀: Optional bias term

Usage:
    # Create standard DeepONet
    model = DeepONetFactory.create('standard', n_sensors=100, latent_dim=256)

    # Create physics-informed variant
    model = DeepONetFactory.create('physics', physics_type='fluid')

    # Create Fourier-enhanced variant
    model = DeepONetFactory.create('fourier', trunk_fourier_modes=64)

References:
- Chen, T. & Chen, H. (2021). Universal approximation to nonlinear operators
  by neural networks with arbitrary activation functions and its application
  to dynamical systems. arXiv:1911.03967
- Lu, L. et al. (2021). Learning nonlinear operators via DeepONet based on
  the universal approximation theorem of operators. Nature Machine Intelligence.
"""

from .deeponet_base import DeepONet, DeepONetConfig
from .branch_networks import (
    BranchNetwork,
    DenseBranchNetwork,
    ConvolutionalBranchNetwork,
    FourierBranchNetwork
)
from .trunk_networks import (
    TrunkNetwork,
    DenseTrunkNetwork,
    FourierTrunkNetwork,
    PhysicsAwareTrunkNetwork
)
from .deeponet_variants import (
    StandardDeepONet,
    FourierDeepONet,
    PhysicsInformedDeepONet,
    MultiScaleDeepONet
)
from .model_factory import DeepONetFactory

__all__ = [
    # Core classes
    'DeepONet',
    'DeepONetConfig',

    # Branch networks
    'BranchNetwork',
    'DenseBranchNetwork',
    'ConvolutionalBranchNetwork',
    'FourierBranchNetwork',

    # Trunk networks
    'TrunkNetwork',
    'DenseTrunkNetwork',
    'FourierTrunkNetwork',
    'PhysicsAwareTrunkNetwork',

    # Model variants
    'StandardDeepONet',
    'FourierDeepONet',
    'PhysicsInformedDeepONet',
    'MultiScaleDeepONet',

    # Factory
    'DeepONetFactory'
]

# Version info
__version__ = "1.0.0"
__author__ = "Generative Operators Research Team"