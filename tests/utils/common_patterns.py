"""
Common Testing Patterns

Provides commonly used testing patterns and templates for neural operator
model testing. These patterns can be easily reused across different test files.
"""

import torch
import torch.nn as nn
import unittest
from typing import Dict, List, Any, Optional, Callable
from .test_utilities import ModelTestSuite, ExtendedAssertions


class BaseModelTestCase(unittest.TestCase, ExtendedAssertions):
    """Base test case with common functionality for model testing"""

    def setUp(self):
        """Set up common test fixtures"""
        from src.core.utils.reproducibility import set_global_seed
        set_global_seed(verbose=False)  # For reproducible tests
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def tearDown(self):
        """Clean up after tests"""
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def run_standard_model_tests(self, model: nn.Module, input_tensor: torch.Tensor, model_name: str = "Model"):
        """Run standard suite of model tests"""
        test_suite = ModelTestSuite(model, model_name)

        # Basic functionality tests
        basic_results = test_suite.run_basic_tests(input_tensor)
        for result in basic_results:
            with self.subTest(test=result.test_name):
                if not result.passed:
                    self.fail(f"Test {result.test_name} failed: {result.error_message}")

        # Robustness tests
        robustness_results = test_suite.run_robustness_tests(input_tensor)
        for result in robustness_results:
            with self.subTest(test=result.test_name):
                if not result.passed:
                    self.fail(f"Test {result.test_name} failed: {result.error_message}")

    def assert_valid_model_output(self, model: nn.Module, input_tensor: torch.Tensor, expected_shape: Optional[tuple] = None):
        """Assert model produces valid output"""
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)

        # Check output is finite
        self.assert_tensor_finite(output, "Model output contains non-finite values")

        # Check shape if provided
        if expected_shape is not None:
            self.assert_tensor_shape(output, expected_shape, f"Output shape mismatch")

        # Check output is not all zeros or constant
        output_std = torch.std(output)
        self.assertGreater(output_std.item(), 1e-8, "Model output appears to be constant")

        return output

    def assert_gradient_flow(self, model: nn.Module, input_tensor: torch.Tensor, loss_fn: Callable = None):
        """Assert proper gradient flow through model"""
        if loss_fn is None:
            loss_fn = lambda x: torch.mean(x)

        model.train()
        input_tensor = input_tensor.clone().requires_grad_(True)

        output = model(input_tensor)
        loss = loss_fn(output)
        loss.backward()

        # Check model has gradients
        has_gradients = any(p.grad is not None for p in model.parameters())
        self.assertTrue(has_gradients, "No gradients computed for model parameters")

        # Check gradients are finite
        self.assert_gradient_finite(model, "Model gradients are not finite")

        # Check input has gradients (if required)
        if input_tensor.requires_grad:
            self.assertIsNotNone(input_tensor.grad, "Input tensor should have gradients")

    def assert_model_consistency(self, model: nn.Module, input_tensor: torch.Tensor, num_runs: int = 3):
        """Assert model produces consistent outputs"""
        model.eval()
        outputs = []

        for _ in range(num_runs):
            with torch.no_grad():
                output = model(input_tensor)
            outputs.append(output)

        # Check all outputs are close
        for i in range(1, len(outputs)):
            self.assert_tensors_close(
                outputs[0], outputs[i],
                rtol=1e-5, atol=1e-8,
                msg=f"Model outputs inconsistent between runs 0 and {i}"
            )

    def benchmark_model_performance(self, model: nn.Module, input_tensor: torch.Tensor, num_trials: int = 10) -> Dict[str, float]:
        """Benchmark model performance"""
        import time
        import gc

        model.eval()
        times = []

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model(input_tensor)

        # Actual timing
        for _ in range(num_trials):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            gc.collect()

            start_time = time.time()
            with torch.no_grad():
                output = model(input_tensor)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()

            times.append(end_time - start_time)

        return {
            "mean_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "std_time": torch.std(torch.tensor(times)).item()
        }


class ModelVariantTestPattern:
    """Pattern for testing multiple variants of a model type"""

    def __init__(self, base_test_case: unittest.TestCase):
        self.test_case = base_test_case

    def test_all_variants(self, model_factory: Callable, variant_configs: Dict[str, Dict], input_tensor: torch.Tensor):
        """Test all variants of a model type"""
        for variant_name, config in variant_configs.items():
            with self.test_case.subTest(variant=variant_name):
                try:
                    model = model_factory(**config)
                    self._test_single_variant(model, input_tensor, variant_name)
                except Exception as e:
                    self.test_case.fail(f"Variant {variant_name} failed: {e}")

    def _test_single_variant(self, model: nn.Module, input_tensor: torch.Tensor, variant_name: str):
        """Test a single model variant"""
        # Basic forward pass
        output = model(input_tensor)
        self.test_case.assertTrue(torch.all(torch.isfinite(output)), f"{variant_name} output not finite")

        # Parameter count check
        param_count = sum(p.numel() for p in model.parameters())
        self.test_case.assertGreater(param_count, 0, f"{variant_name} has no parameters")

        # Gradient flow check
        if hasattr(self.test_case, 'assert_gradient_flow'):
            self.test_case.assert_gradient_flow(model, input_tensor)


class DatasetCompatibilityTestPattern:
    """Pattern for testing model compatibility with different datasets"""

    def __init__(self, base_test_case: unittest.TestCase):
        self.test_case = base_test_case

    def test_dataset_compatibility(self, model: nn.Module, dataset_names: List[str], batch_creator: Callable):
        """Test model compatibility with multiple datasets"""
        for dataset_name in dataset_names:
            with self.test_case.subTest(dataset=dataset_name):
                try:
                    input_batch, target_batch = batch_creator(dataset_name)
                    self._test_single_dataset(model, input_batch, target_batch, dataset_name)
                except Exception as e:
                    self.test_case.fail(f"Dataset {dataset_name} compatibility failed: {e}")

    def _test_single_dataset(self, model: nn.Module, input_batch: torch.Tensor,
                           target_batch: Optional[torch.Tensor], dataset_name: str):
        """Test model with a single dataset"""
        # Forward pass
        output = model(input_batch)
        self.test_case.assertTrue(torch.all(torch.isfinite(output)),
                                f"Model output not finite for dataset {dataset_name}")

        # Shape compatibility
        if target_batch is not None:
            expected_shape = target_batch.shape
            self.test_case.assertEqual(output.shape, expected_shape,
                                     f"Output shape mismatch for dataset {dataset_name}")

        # Value range check
        output_max = torch.max(torch.abs(output))
        self.test_case.assertLess(output_max.item(), 1000.0,
                                f"Output values too large for dataset {dataset_name}")


class PhysicalValidationTestPattern:
    """Pattern for testing physical properties of model outputs"""

    def __init__(self, base_test_case: unittest.TestCase):
        self.test_case = base_test_case

    def test_physical_properties(self, model: nn.Module, input_tensor: torch.Tensor):
        """Test physical properties of model output"""
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)

        self._test_conservation_properties(output)
        self._test_continuity_properties(output)
        self._test_stability_properties(output)

    def _test_conservation_properties(self, output: torch.Tensor):
        """Test conservation properties"""
        # Mass conservation (simplified - total values shouldn't blow up)
        total_mass = torch.sum(output, dim=(-2, -1))  # Sum over spatial dimensions
        max_mass = torch.max(torch.abs(total_mass))
        self.test_case.assertLess(max_mass.item(), 10000.0, "Mass conservation violation")

    def _test_continuity_properties(self, output: torch.Tensor):
        """Test continuity properties"""
        if output.shape[1] > 1:  # Temporal sequence
            # Temporal continuity
            temporal_diff = torch.abs(output[:, 1:] - output[:, :-1])
            max_temporal_change = torch.max(temporal_diff)
            self.test_case.assertLess(max_temporal_change.item(), 50.0, "Temporal discontinuity")

        # Spatial continuity
        if output.shape[-1] > 1 and output.shape[-2] > 1:
            spatial_grad_x = torch.abs(output[..., 1:, :] - output[..., :-1, :])
            spatial_grad_y = torch.abs(output[..., :, 1:] - output[..., :, :-1])
            max_spatial_grad = max(torch.max(spatial_grad_x), torch.max(spatial_grad_y))
            self.test_case.assertLess(max_spatial_grad.item(), 50.0, "Spatial discontinuity")

    def _test_stability_properties(self, output: torch.Tensor):
        """Test numerical stability properties"""
        # Check for reasonable dynamic range
        output_std = torch.std(output)
        output_mean = torch.mean(torch.abs(output))

        # Coefficient of variation should be reasonable
        cv = output_std / (output_mean + 1e-8)
        self.test_case.assertLess(cv.item(), 100.0, "Unreasonable dynamic range")

        # Check for outliers
        output_flat = output.flatten()
        q75, q25 = torch.quantile(output_flat, 0.75), torch.quantile(output_flat, 0.25)
        iqr = q75 - q25
        outlier_threshold = q75 + 3 * iqr

        outlier_fraction = torch.sum(torch.abs(output_flat) > outlier_threshold).float() / len(output_flat)
        self.test_case.assertLess(outlier_fraction.item(), 0.05, "Too many outliers in output")


class PerformanceTestPattern:
    """Pattern for performance testing"""

    def __init__(self, base_test_case: unittest.TestCase):
        self.test_case = base_test_case

    def test_performance_requirements(self, model: nn.Module, input_tensor: torch.Tensor,
                                    max_time: float = 1.0, max_memory_mb: float = 1000.0):
        """Test performance requirements"""
        import time
        import gc
        import psutil
        import os

        model.eval()

        # Memory test
        gc.collect()
        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

        # Time test
        start_time = time.time()
        with torch.no_grad():
            output = model(input_tensor)
        end_time = time.time()

        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

        # Assertions
        execution_time = end_time - start_time
        memory_usage = final_memory - initial_memory

        self.test_case.assertLess(execution_time, max_time,
                                f"Model too slow: {execution_time:.3f}s > {max_time}s")
        self.test_case.assertLess(memory_usage, max_memory_mb,
                                f"Model uses too much memory: {memory_usage:.1f}MB > {max_memory_mb}MB")

        return {
            "execution_time": execution_time,
            "memory_usage": memory_usage,
            "output_shape": output.shape
        }


def create_model_test_class(model_name: str, model_factory: Callable,
                          input_shape: tuple, **test_configs) -> type:
    """Factory function to create model test classes dynamically"""

    class DynamicModelTest(BaseModelTestCase):
        def setUp(self):
            super().setUp()
            self.model = model_factory()
            self.input_tensor = torch.randn(*input_shape)

        def test_basic_functionality(self):
            """Test basic model functionality"""
            self.run_standard_model_tests(self.model, self.input_tensor, model_name)

        def test_output_validity(self):
            """Test output validity"""
            output = self.assert_valid_model_output(self.model, self.input_tensor)
            self.assertIsNotNone(output)

        def test_gradient_flow(self):
            """Test gradient flow"""
            self.assert_gradient_flow(self.model, self.input_tensor)

        def test_consistency(self):
            """Test model consistency"""
            self.assert_model_consistency(self.model, self.input_tensor)

        def test_performance(self):
            """Test basic performance"""
            performance_pattern = PerformanceTestPattern(self)
            results = performance_pattern.test_performance_requirements(
                self.model, self.input_tensor,
                max_time=test_configs.get('max_time', 5.0),
                max_memory_mb=test_configs.get('max_memory_mb', 2000.0)
            )
            self.assertIsNotNone(results)

    # Set class name dynamically
    DynamicModelTest.__name__ = f"Test{model_name}"
    DynamicModelTest.__qualname__ = f"Test{model_name}"

    return DynamicModelTest


if __name__ == "__main__":
    # Test the patterns
    print("Testing common patterns...")

    # Create a simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 3, padding=1)

        def forward(self, x):
            B, T, C, H, W = x.shape
            x = x.view(B*T, C, H, W)
            x = self.conv(x)
            return x.view(B, T, C, H, W)

    # Test base test case
    class SimpleTest(BaseModelTestCase):
        def test_model(self):
            model = TestModel()
            input_tensor = torch.randn(1, 2, 3, 8, 8)
            self.run_standard_model_tests(model, input_tensor, "TestModel")

    # Run a simple test
    test_case = SimpleTest()
    test_case.setUp()
    try:
        test_case.test_model()
        print("✅ Common patterns test passed!")
    except Exception as e:
        print(f"❌ Common patterns test failed: {e}")
    finally:
        test_case.tearDown()