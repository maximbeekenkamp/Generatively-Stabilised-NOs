"""
Test Utilities Package

This package contains utility functions, classes, and decorators for testing
neural operator models efficiently and comprehensively.
"""

from .test_utilities import (
    ModelPerformanceMetrics,
    TestResult,
    PerformanceProfiler,
    ModelTestSuite,
    ErrorInjector,
    ModelComparator,
    TestDataValidator,
    ExtendedAssertions,
    skip_if_no_cuda,
    timeout,
    repeat_test,
    profile_test
)

from .common_patterns import (
    BaseModelTestCase,
    ModelVariantTestPattern,
    DatasetCompatibilityTestPattern,
    PhysicalValidationTestPattern,
    PerformanceTestPattern,
    create_model_test_class
)

__all__ = [
    # Test utilities
    'ModelPerformanceMetrics',
    'TestResult',
    'PerformanceProfiler',
    'ModelTestSuite',
    'ErrorInjector',
    'ModelComparator',
    'TestDataValidator',
    'ExtendedAssertions',
    'skip_if_no_cuda',
    'timeout',
    'repeat_test',
    'profile_test',
    # Common patterns
    'BaseModelTestCase',
    'ModelVariantTestPattern',
    'DatasetCompatibilityTestPattern',
    'PhysicalValidationTestPattern',
    'PerformanceTestPattern',
    'create_model_test_class'
]