"""
TNO LpLoss Implementation
Relative Lp-norm loss function from the original TNO paper

Adapted from TNO_Codes/NCEP-NCAR/lploss.py for Gen Stabilised integration
"""

import torch
import torch.nn as nn
import numpy as np
import operator
from functools import reduce
from functools import partial


class LpLoss2(object):
    """
    Relative Lp-norm loss function
    
    Computes relative error: ||u - u_pred||_p / ||u||_p
    where p is the norm order (typically p=2 for L2 norm)
    
    Args:
        d: Dimension (typically 2 for 2D problems)
        p: Norm order (typically 2 for L2 norm) 
        size_average: Whether to average over batch
        reduction: Whether to reduce to scalar
    """
    
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss2, self).__init__()

        # Dimension and Lp-norm type are positive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        """
        Compute relative Lp error
        
        Args:
            x: Predictions [B, ...]
            y: Ground truth [B, ...]
            
        Returns:
            Relative Lp errors [B] or scalar
        """
        num_examples = x.size()[0]

        # Compute Lp norms of differences and targets
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        # Avoid division by zero
        y_norms = torch.clamp(y_norms, min=1e-8)
        
        relative_errors = diff_norms / y_norms

        if self.reduction:
            if self.size_average:
                return torch.mean(relative_errors)
            else:
                return torch.sum(relative_errors)

        return relative_errors

    def __call__(self, x, y):
        return self.rel(x, y)


class LpLoss2Adaptive(LpLoss2):
    """
    Adaptive LpLoss2 that can handle different tensor shapes and dimensions
    Specifically designed for TNO integration with Gen Stabilised framework
    """
    
    def __init__(self, d=2, p=2, size_average=True, reduction=True, spatial_dims=(-2, -1)):
        super().__init__(d, p, size_average, reduction)
        self.spatial_dims = spatial_dims
    
    def rel(self, x, y):
        """
        Adaptive relative Lp error computation
        
        Handles tensors of shape [B, T, C, H, W] or [B, C, H, W]
        """
        # Ensure tensors have same shape
        if x.shape != y.shape:
            raise ValueError(f"Input shapes don't match: {x.shape} vs {y.shape}")
        
        batch_size = x.size(0)
        
        # Flatten spatial dimensions while keeping batch and possibly time/channel dims
        if len(x.shape) == 5:  # [B, T, C, H, W]
            x_flat = x.reshape(batch_size, -1)
            y_flat = y.reshape(batch_size, -1)
        elif len(x.shape) == 4:  # [B, C, H, W]
            x_flat = x.reshape(batch_size, -1)
            y_flat = y.reshape(batch_size, -1)
        else:
            # For other shapes, flatten everything except batch dimension
            x_flat = x.reshape(batch_size, -1)
            y_flat = y.reshape(batch_size, -1)

        # Compute Lp norms
        diff_norms = torch.norm(x_flat - y_flat, self.p, dim=1)
        y_norms = torch.norm(y_flat, self.p, dim=1)

        # Avoid division by zero with small epsilon
        y_norms = torch.clamp(y_norms, min=1e-8)
        
        relative_errors = diff_norms / y_norms

        if self.reduction:
            if self.size_average:
                return torch.mean(relative_errors)
            else:
                return torch.sum(relative_errors)

        return relative_errors


# Convenience functions for common use cases
def create_tno_loss(p=2, reduction=True):
    """Create standard TNO LpLoss2 with common settings"""
    return LpLoss2Adaptive(d=2, p=p, size_average=True, reduction=reduction)


def create_relative_l2_loss():
    """Create relative L2 loss (most common for TNO)"""
    return create_tno_loss(p=2)


def create_relative_l1_loss():
    """Create relative L1 loss"""
    return create_tno_loss(p=1)