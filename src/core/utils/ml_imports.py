"""
Machine Learning Imports Module

Centralizes frequently used ML framework imports to reduce duplication.
These imports appeared in 40+ files each.
"""

# PyTorch imports (most common ML framework in the codebase)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, RandomSampler, SequentialSampler, SubsetRandomSampler

# Additional common ML imports
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False

try:
    import scipy
    SCIPY_AVAILABLE = True
except ImportError:
    scipy = None
    SCIPY_AVAILABLE = False

# Re-export commonly used items
__all__ = [
    'torch', 'nn', 'F',
    'DataLoader', 'Dataset', 'Sampler', 'RandomSampler', 'SequentialSampler', 'SubsetRandomSampler',
    'plt', 'MATPLOTLIB_AVAILABLE',
    'scipy', 'SCIPY_AVAILABLE'
]