"""
Test fixtures for neural operator testing framework.

This module provides dummy datasets and test utilities for comprehensive
neural operator model testing.
"""

from .dummy_datasets import (
    get_dummy_batch,
    get_dummy_batch_for_model,
    DummyDatasetFactory,
    DatasetConfig
)

__all__ = [
    'get_dummy_batch',
    'get_dummy_batch_for_model',
    'DummyDatasetFactory',
    'DatasetConfig'
]
