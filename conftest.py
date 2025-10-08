"""
Pytest configuration file for neural operator testing framework.

This file configures pytest to properly handle:
- Python path setup for imports
- Test fixtures and dependencies
- Matplotlib backend configuration
- Random seed initialization (centralized)
"""

import sys
import os
from pathlib import Path
import pytest

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add src directory to Python path
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

# Add tests directory to Python path
TESTS_DIR = PROJECT_ROOT / "tests"
if TESTS_DIR.exists():
    sys.path.insert(0, str(TESTS_DIR))


def pytest_configure(config):
    """Configure pytest environment."""
    # Set matplotlib to non-interactive backend
    import matplotlib
    matplotlib.use('Agg')

    # Disable matplotlib interactive mode
    import matplotlib.pyplot as plt
    plt.ioff()


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """
    Reset random seeds before each test for reproducibility.

    Uses centralized seed management from src.core.utils.reproducibility.
    """
    from src.core.utils.reproducibility import set_global_seed
    set_global_seed(verbose=False)  # Suppress output for cleaner test logs
    yield


@pytest.fixture(autouse=True)
def cleanup_matplotlib():
    """Clean up matplotlib figures after each test."""
    yield
    import matplotlib.pyplot as plt
    plt.close('all')


@pytest.fixture(autouse=True)
def reset_torch_state():
    """Reset PyTorch state after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide path to test data directory."""
    return PROJECT_ROOT / "tests" / "data"


@pytest.fixture(scope="session")
def temp_output_dir(tmp_path_factory):
    """Provide temporary directory for test outputs."""
    return tmp_path_factory.mktemp("test_outputs")
