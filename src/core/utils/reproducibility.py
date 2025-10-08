"""
Centralized Seed Management for Reproducibility

This module provides a single source of truth for random seed management across:
- Python's random module
- NumPy
- PyTorch (CPU and GPU)
- cuDNN determinism
- Other libraries via PYTHONHASHSEED

Usage:
    from src.core.utils.reproducibility import set_global_seed, set_seed_from_config

    # Method 1: Use default global seed (42)
    set_global_seed()

    # Method 2: Set specific seed
    set_global_seed(123)

    # Method 3: Load from config
    config = {"random_seed": 42, "deterministic": True}
    set_seed_from_config(config)

Note on Performance:
    Setting torch.backends.cudnn.deterministic=True may reduce performance
    but ensures reproducible results. This is controlled by the 'deterministic'
    config flag.

Limitations:
    Some PyTorch operations are non-deterministic even with seeds set
    (e.g., certain atomicAdd operations on GPU). This provides best-effort
    reproducibility.

Author: GenStabilisation-Proj
Date: 2025-10-08
"""

import torch
import numpy as np
import random
import os
from typing import Optional, Dict, Any


# Global seed value - single source of truth
# Change this value to change the seed across the entire codebase
GLOBAL_SEED = 42


def set_global_seed(seed: Optional[int] = None, verbose: bool = True) -> None:
    """
    Set all random seeds for reproducibility.

    This function sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU)
    - PyTorch (GPU - all CUDA devices)
    - cuDNN (deterministic mode)
    - Environment variable for other libraries

    Args:
        seed: Random seed value. If None, uses GLOBAL_SEED (42)
        verbose: If True, prints confirmation message

    Example:
        >>> set_global_seed(42)
        ✓ Global seed set to 42

        >>> set_global_seed()  # Uses GLOBAL_SEED
        ✓ Global seed set to 42
    """
    if seed is None:
        seed = GLOBAL_SEED

    # Python random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch GPU (all CUDA devices)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # cuDNN determinism
    # Note: This may reduce performance but ensures reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Environment variable for additional libraries (e.g., PYTHONHASHSEED)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if verbose:
        print(f"✓ Global seed set to {seed}")


def get_global_seed() -> int:
    """
    Get the current global seed value.

    Returns:
        The global seed value (default: 42)

    Example:
        >>> get_global_seed()
        42
    """
    return GLOBAL_SEED


def set_seed_from_config(config: Dict[str, Any], verbose: bool = True) -> None:
    """
    Set seed from a configuration dictionary.

    Args:
        config: Configuration dictionary containing:
            - 'random_seed' (int, optional): Seed value (default: GLOBAL_SEED)
            - 'deterministic' (bool, optional): Enable deterministic mode (default: True)
        verbose: If True, prints confirmation message

    Example:
        >>> config = {"random_seed": 123, "deterministic": True}
        >>> set_seed_from_config(config)
        ✓ Global seed set to 123

        >>> config = {}  # Uses GLOBAL_SEED
        >>> set_seed_from_config(config)
        ✓ Global seed set to 42
    """
    seed = config.get('random_seed', GLOBAL_SEED)
    deterministic = config.get('deterministic', True)

    # Set seed
    set_global_seed(seed, verbose=verbose)

    # Optionally disable determinism for performance
    if not deterministic:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        if verbose:
            print("⚠ Deterministic mode disabled for performance (results may vary)")


def enable_deterministic_mode(verbose: bool = True) -> None:
    """
    Enable deterministic mode for maximum reproducibility.

    This sets cuDNN to deterministic mode and disables benchmarking.
    May reduce performance but ensures reproducible results.

    Args:
        verbose: If True, prints confirmation message
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if verbose:
        print("✓ Deterministic mode enabled (may impact performance)")


def disable_deterministic_mode(verbose: bool = True) -> None:
    """
    Disable deterministic mode for better performance.

    This allows cuDNN to use non-deterministic algorithms for faster
    computation. Results may vary slightly between runs.

    Args:
        verbose: If True, prints confirmation message
    """
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    if verbose:
        print("⚠ Deterministic mode disabled (results may vary)")


if __name__ == "__main__":
    # Self-test
    print("Testing reproducibility module...")
    print()

    # Test 1: Default seed
    print("Test 1: Default seed")
    set_global_seed()
    assert get_global_seed() == 42
    print()

    # Test 2: Custom seed
    print("Test 2: Custom seed")
    set_global_seed(123)
    print()

    # Test 3: Config loading
    print("Test 3: Config loading")
    config = {"random_seed": 456, "deterministic": True}
    set_seed_from_config(config)
    print()

    # Test 4: Deterministic mode toggle
    print("Test 4: Deterministic mode")
    enable_deterministic_mode()
    disable_deterministic_mode()
    enable_deterministic_mode()  # Re-enable for safety
    print()

    # Test 5: Verify reproducibility
    print("Test 5: Verify reproducibility")
    set_global_seed(42, verbose=False)
    x1 = torch.randn(5)
    y1 = np.random.randn(5)

    set_global_seed(42, verbose=False)
    x2 = torch.randn(5)
    y2 = np.random.randn(5)

    assert torch.allclose(x1, x2), "PyTorch random values don't match!"
    assert np.allclose(y1, y2), "NumPy random values don't match!"
    print("✓ Reproducibility verified: identical random values after reseeding")
    print()

    print("✓ All tests passed!")
