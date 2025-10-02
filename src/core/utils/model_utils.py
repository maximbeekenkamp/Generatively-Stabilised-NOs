"""
Model utility functions shared across neural operator implementations.

This module provides common utility functions used by multiple models
to avoid code duplication.
"""

from typing import Union, Any


def get_prev_steps_from_arch(model_params: Any) -> int:
    """
    Extract the number of previous timesteps from architecture string.

    This function parses the architecture string in model parameters to determine
    how many previous timesteps the model uses for prediction. This is commonly
    used in autoregressive neural operators.

    Args:
        model_params: Model parameters object containing an 'arch' attribute

    Returns:
        Number of previous timesteps:
            - 4 if "+3Prev" in architecture
            - 3 if "+2Prev" in architecture
            - 2 if "+Prev" in architecture
            - 1 otherwise (single previous step)

    Examples:
        >>> params.arch = "FNO16+Prev"
        >>> get_prev_steps_from_arch(params)
        2
        >>> params.arch = "TNO+2Prev+DM"
        >>> get_prev_steps_from_arch(params)
        3
    """
    arch = getattr(model_params, 'arch', '')

    if "+3Prev" in arch:
        return 4
    elif "+2Prev" in arch:
        return 3
    elif "+Prev" in arch:
        return 2
    else:
        return 1


def calculate_input_channels(p_d: Any, prev_steps: int) -> int:
    """
    Calculate the number of input channels based on data parameters.

    Args:
        p_d: Data parameters object
        prev_steps: Number of previous timesteps

    Returns:
        Total number of input channels
    """
    return prev_steps * (p_d.dimension + len(p_d.simFields) + len(p_d.simParams))


def calculate_output_channels(p_d: Any) -> int:
    """
    Calculate the number of output channels based on data parameters.

    Args:
        p_d: Data parameters object

    Returns:
        Total number of output channels
    """
    return p_d.dimension + len(p_d.simFields) + len(p_d.simParams)
