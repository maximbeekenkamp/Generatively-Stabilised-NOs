"""
Environment Setup Utilities

Consolidates common environment setup code that was duplicated across 32+ files.
Includes project path configuration, logging setup, and other common initialization tasks.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Union


def setup_project_paths(current_file: Optional[Union[str, Path]] = None) -> Path:
    """
    Setup project paths consistently across all scripts.

    This replaces 32+ instances of duplicate path setup code throughout the codebase.

    Args:
        current_file: Path to the current file (__file__). If None, uses caller's location.

    Returns:
        Path object pointing to the project root directory.
    """
    if current_file is None:
        # Try to determine from call stack
        import inspect
        current_file = Path(inspect.getfile(inspect.currentframe().f_back))
    else:
        current_file = Path(current_file)

    # Navigate to project root (adjust levels as needed)
    if 'src' in current_file.parts:
        # Find src directory and go one level up
        src_index = current_file.parts.index('src')
        project_root = Path(*current_file.parts[:src_index])
    else:
        # Default: assume we're in a subdirectory, go up until we find src/
        project_root = current_file.parent
        while project_root.parent != project_root:  # Not at filesystem root
            if (project_root / 'src').exists():
                break
            project_root = project_root.parent

    # Add src to Python path if not already there
    src_dir = project_root / 'src'
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    return project_root


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    include_timestamp: bool = True
) -> logging.Logger:
    """
    Setup logging configuration consistently.

    This consolidates 4 duplicate setup_logging functions found in:
    - examples/quick_start_example.py
    - scripts/train_generative_operator_*.py files

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
        format_string: Custom format string for log messages
        include_timestamp: Whether to include timestamp in log format

    Returns:
        Configured logger instance
    """
    if format_string is None:
        if include_timestamp:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        else:
            format_string = '%(name)s - %(levelname)s - %(message)s'

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )

    return logging.getLogger(__name__)


def setup_gpu_environment(gpu_id: str = "0", use_gpu: bool = True) -> bool:
    """
    Setup GPU environment consistently.

    Args:
        gpu_id: GPU ID to use (e.g., "0", "1", or "0,1")
        use_gpu: Whether to attempt GPU usage

    Returns:
        True if GPU is available and configured, False otherwise
    """
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return False

    import torch

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        return False

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        # Verify GPU is accessible
        device = torch.device(f'cuda:{gpu_id.split(",")[0]}')
        torch.cuda.set_device(device)
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        return True
    except Exception as e:
        print(f"WARNING: Failed to setup GPU {gpu_id}: {e}")
        return False


def print_environment_info(include_gpu: bool = True, include_packages: bool = False):
    """
    Print environment information for debugging.

    Args:
        include_gpu: Whether to include GPU information
        include_packages: Whether to include package version information
    """
    print("=== Environment Information ===")
    print(f"Python: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    print(f"Python Path: {sys.path[:3]}...")  # Show first 3 entries

    if include_gpu:
        try:
            import torch
            print(f"PyTorch: {torch.__version__}")
            print(f"CUDA Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA Version: {torch.version.cuda}")
                print(f"GPU Count: {torch.cuda.device_count()}")
        except ImportError:
            print("PyTorch: Not installed")

    if include_packages:
        try:
            import numpy as np
            print(f"NumPy: {np.__version__}")
        except ImportError:
            print("NumPy: Not installed")

    print("================================")


# Convenience function that combines common setup steps
def initialize_environment(
    current_file: Optional[Union[str, Path]] = None,
    gpu_id: str = "0",
    use_gpu: bool = True,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    verbose: bool = True
) -> dict:
    """
    Complete environment initialization combining all setup steps.

    Args:
        current_file: Path to current file for project path setup
        gpu_id: GPU ID to use
        use_gpu: Whether to use GPU
        log_level: Logging level
        log_file: Optional log file path
        verbose: Whether to print environment information

    Returns:
        Dictionary with setup results: {'project_root', 'gpu_available', 'logger'}
    """
    # Setup paths
    project_root = setup_project_paths(current_file)

    # Setup logging
    logger = setup_logging(log_level, log_file)

    # Setup GPU
    gpu_available = setup_gpu_environment(gpu_id, use_gpu)

    if verbose:
        print_environment_info(include_gpu=True, include_packages=False)

    return {
        'project_root': project_root,
        'gpu_available': gpu_available,
        'logger': logger
    }