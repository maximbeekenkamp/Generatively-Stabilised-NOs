"""
Comprehensive Test Utilities

Provides utility functions, classes, and decorators to make testing neural operator
models easier and more efficient. Includes:

1. Test decorators for common patterns
2. Model comparison utilities
3. Performance profiling helpers
4. Memory usage monitoring
5. Test data validation
6. Error injection for robustness testing
7. Parallel test execution helpers
8. Test result aggregation
9. Mock object factories
10. Common assertion helpers
"""

import time
import torch
import torch.nn as nn
import numpy as np
import functools
import warnings
import gc
import psutil
import os
import sys
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from contextlib import contextmanager
from dataclasses import dataclass
from collections import defaultdict
import unittest


@dataclass
class ModelPerformanceMetrics:
    """Container for model performance metrics"""
    forward_time: float
    memory_usage_mb: float
    parameter_count: int
    flops_estimate: Optional[int] = None
    gradient_norm: Optional[float] = None
    numerical_stability: bool = True


@dataclass
class TestResult:
    """Container for test results"""
    test_name: str
    passed: bool
    execution_time: float
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class PerformanceProfiler:
    """Profile model performance during testing"""

    def __init__(self):
        self.results = {}
        self.start_time = None
        self.start_memory = None

    @contextmanager
    def profile(self, name: str):
        """Context manager for profiling code blocks"""
        self.start_time = time.time()
        self.start_memory = self._get_memory_usage()

        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()

            self.results[name] = {
                'execution_time': end_time - self.start_time,
                'memory_delta': end_memory - self.start_memory,
                'peak_memory': end_memory
            }

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def get_results(self) -> Dict[str, Dict[str, float]]:
        """Get profiling results"""
        return self.results.copy()

    def clear_results(self):
        """Clear profiling results"""
        self.results.clear()


class ModelTestSuite:
    """Comprehensive test suite for neural operator models"""

    def __init__(self, model: nn.Module, model_name: str = "Model"):
        self.model = model
        self.model_name = model_name
        self.profiler = PerformanceProfiler()
        self.test_results = []

    def run_basic_tests(self, input_tensor: torch.Tensor) -> List[TestResult]:
        """Run basic functionality tests"""
        tests = [
            ("forward_pass", self._test_forward_pass),
            ("gradient_flow", self._test_gradient_flow),
            ("parameter_count", self._test_parameter_count),
            ("numerical_stability", self._test_numerical_stability),
            ("memory_efficiency", self._test_memory_efficiency)
        ]

        results = []
        for test_name, test_func in tests:
            result = self._run_single_test(test_name, test_func, input_tensor)
            results.append(result)

        return results

    def run_robustness_tests(self, input_tensor: torch.Tensor) -> List[TestResult]:
        """Run robustness tests"""
        tests = [
            ("input_perturbation", self._test_input_perturbation),
            ("batch_size_variation", self._test_batch_size_variation),
            ("device_compatibility", self._test_device_compatibility),
            ("precision_consistency", self._test_precision_consistency)
        ]

        results = []
        for test_name, test_func in tests:
            result = self._run_single_test(test_name, test_func, input_tensor)
            results.append(result)

        return results

    def _run_single_test(self, name: str, test_func: Callable, *args) -> TestResult:
        """Run a single test and capture results"""
        start_time = time.time()

        try:
            test_func(*args)
            passed = True
            error_message = None
        except Exception as e:
            passed = False
            error_message = str(e)

        execution_time = time.time() - start_time

        return TestResult(
            test_name=f"{self.model_name}_{name}",
            passed=passed,
            execution_time=execution_time,
            error_message=error_message
        )

    def _test_forward_pass(self, input_tensor: torch.Tensor):
        """Test basic forward pass"""
        with self.profiler.profile("forward_pass"):
            output = self.model(input_tensor)

        if not torch.all(torch.isfinite(output)):
            raise ValueError("Model output contains NaN or Inf values")

    def _test_gradient_flow(self, input_tensor: torch.Tensor):
        """Test gradient flow through model"""
        input_tensor = input_tensor.clone().requires_grad_(True)
        output = self.model(input_tensor)
        loss = torch.mean(output)
        loss.backward()

        has_gradients = any(p.grad is not None for p in self.model.parameters())
        if not has_gradients:
            raise ValueError("No gradients computed")

        # Check for gradient explosion
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in self.model.parameters()]))
        if total_norm > 1000:
            raise ValueError(f"Gradient explosion detected: norm = {total_norm}")

    def _test_parameter_count(self, input_tensor: torch.Tensor):
        """Test parameter count is reasonable"""
        param_count = sum(p.numel() for p in self.model.parameters())
        if param_count == 0:
            raise ValueError("Model has no parameters")
        if param_count > 100_000_000:
            warnings.warn(f"Large model: {param_count} parameters")

    def _test_numerical_stability(self, input_tensor: torch.Tensor):
        """Test numerical stability"""
        # Test with different input magnitudes
        for scale in [0.1, 1.0, 10.0]:
            scaled_input = input_tensor * scale
            output = self.model(scaled_input)

            if not torch.all(torch.isfinite(output)):
                raise ValueError(f"Numerical instability at scale {scale}")

    def _test_memory_efficiency(self, input_tensor: torch.Tensor):
        """Test memory efficiency"""
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

        initial_memory = self.profiler._get_memory_usage()

        for _ in range(3):  # Multiple forward passes
            output = self.model(input_tensor)
            del output

        final_memory = self.profiler._get_memory_usage()
        memory_increase = final_memory - initial_memory

        if memory_increase > 1000:  # More than 1GB increase
            raise ValueError(f"Memory leak detected: {memory_increase:.2f} MB increase")

    def _test_input_perturbation(self, input_tensor: torch.Tensor):
        """Test robustness to input perturbations"""
        original_output = self.model(input_tensor)

        # Add small noise
        noise_scale = 0.01 * torch.std(input_tensor)
        noisy_input = input_tensor + torch.randn_like(input_tensor) * noise_scale
        noisy_output = self.model(noisy_input)

        # Check output doesn't change drastically
        output_diff = torch.mean(torch.abs(original_output - noisy_output))
        input_diff = torch.mean(torch.abs(input_tensor - noisy_input))

        sensitivity = output_diff / (input_diff + 1e-8)
        if sensitivity > 1000:
            raise ValueError(f"Model too sensitive to input perturbations: {sensitivity}")

    def _test_batch_size_variation(self, input_tensor: torch.Tensor):
        """Test with different batch sizes"""
        original_batch_size = input_tensor.shape[0]

        # Test with different batch sizes
        for new_batch_size in [1, 2, original_batch_size * 2]:
            if new_batch_size <= original_batch_size:
                test_input = input_tensor[:new_batch_size]
            else:
                # Repeat samples to get larger batch
                repeat_factor = (new_batch_size + original_batch_size - 1) // original_batch_size
                test_input = input_tensor.repeat(repeat_factor, *([1] * (input_tensor.ndim - 1)))[:new_batch_size]

            output = self.model(test_input)
            expected_shape = (new_batch_size,) + output.shape[1:]

            if output.shape != expected_shape:
                raise ValueError(f"Incorrect output shape for batch size {new_batch_size}")

    def _test_device_compatibility(self, input_tensor: torch.Tensor):
        """Test device compatibility"""
        if torch.cuda.is_available():
            # Test CUDA
            cuda_model = self.model.cuda()
            cuda_input = input_tensor.cuda()
            cuda_output = cuda_model(cuda_input)

            if not cuda_output.is_cuda:
                raise ValueError("Model output not on CUDA device")

            # Move back to CPU
            self.model.cpu()

    def _test_precision_consistency(self, input_tensor: torch.Tensor):
        """Test precision consistency"""
        # Test float32
        float32_model = self.model.float()
        float32_input = input_tensor.float()
        float32_output = float32_model(float32_input)

        # Test float64
        float64_model = self.model.double()
        float64_input = input_tensor.double()
        float64_output = float64_model(float64_input)

        # Check outputs are reasonable
        if not torch.all(torch.isfinite(float32_output)):
            raise ValueError("Float32 precision issues")

        if not torch.all(torch.isfinite(float64_output)):
            raise ValueError("Float64 precision issues")


class ErrorInjector:
    """Inject controlled errors for robustness testing"""

    @staticmethod
    def add_noise_to_weights(model: nn.Module, noise_scale: float = 0.01):
        """Add noise to model weights"""
        for param in model.parameters():
            if param.requires_grad:
                noise = torch.randn_like(param) * noise_scale
                param.data += noise

    @staticmethod
    def corrupt_input(input_tensor: torch.Tensor, corruption_type: str = "gaussian_noise", **kwargs):
        """Corrupt input tensor in various ways"""
        corrupted = input_tensor.clone()

        if corruption_type == "gaussian_noise":
            noise_scale = kwargs.get("scale", 0.1)
            corrupted += torch.randn_like(input_tensor) * noise_scale

        elif corruption_type == "salt_pepper":
            prob = kwargs.get("prob", 0.05)
            mask = torch.rand_like(input_tensor) < prob
            corrupted[mask] = torch.randint_like(corrupted[mask], 0, 2).float()

        elif corruption_type == "dropout":
            prob = kwargs.get("prob", 0.1)
            mask = torch.rand_like(input_tensor) > prob
            corrupted *= mask

        elif corruption_type == "blur":
            # Simple blur using convolution
            kernel_size = kwargs.get("kernel_size", 3)
            kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size**2)

            if input_tensor.dim() == 5:  # [B, T, C, H, W]
                B, T, C, H, W = input_tensor.shape
                reshaped = input_tensor.view(B*T*C, 1, H, W)
                blurred = torch.nn.functional.conv2d(reshaped, kernel, padding=kernel_size//2)
                corrupted = blurred.view(B, T, C, H, W)

        return corrupted

    @staticmethod
    def create_adversarial_example(model: nn.Module, input_tensor: torch.Tensor, epsilon: float = 0.01):
        """Create adversarial example using FGSM"""
        input_tensor = input_tensor.clone().requires_grad_(True)

        output = model(input_tensor)
        loss = torch.mean(output)
        loss.backward()

        # FGSM attack
        sign_data_grad = input_tensor.grad.data.sign()
        perturbed_input = input_tensor + epsilon * sign_data_grad

        return perturbed_input.detach()


class ModelComparator:
    """Compare multiple models on the same tasks"""

    def __init__(self):
        self.models = {}
        self.results = defaultdict(dict)

    def add_model(self, name: str, model: nn.Module):
        """Add a model to comparison"""
        self.models[name] = model

    def compare_forward_time(self, input_tensor: torch.Tensor, num_trials: int = 10) -> Dict[str, float]:
        """Compare forward pass times"""
        times = {}

        for name, model in self.models.items():
            model.eval()
            total_time = 0

            for _ in range(num_trials):
                start_time = time.time()
                with torch.no_grad():
                    _ = model(input_tensor)
                total_time += time.time() - start_time

            times[name] = total_time / num_trials

        return times

    def compare_accuracy(self, input_tensor: torch.Tensor, ground_truth: torch.Tensor) -> Dict[str, Dict[str, float]]:
        """Compare model accuracy on given data"""
        metrics = {}

        for name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                prediction = model(input_tensor)

            # Compute various metrics
            mse = torch.mean((prediction - ground_truth)**2).item()
            mae = torch.mean(torch.abs(prediction - ground_truth)).item()

            pred_flat = prediction.flatten()
            truth_flat = ground_truth.flatten()
            correlation = torch.corrcoef(torch.stack([pred_flat, truth_flat]))[0, 1].item()

            metrics[name] = {
                "mse": mse,
                "mae": mae,
                "correlation": correlation
            }

        return metrics

    def compare_memory_usage(self, input_tensor: torch.Tensor) -> Dict[str, float]:
        """Compare memory usage"""
        memory_usage = {}

        for name, model in self.models.items():
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

            initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

            _ = model(input_tensor)

            final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            memory_usage[name] = final_memory - initial_memory

        return memory_usage

    def generate_comparison_report(self, input_tensor: torch.Tensor, ground_truth: Optional[torch.Tensor] = None) -> str:
        """Generate comprehensive comparison report"""
        report = ["Model Comparison Report", "=" * 50, ""]

        # Forward time comparison
        forward_times = self.compare_forward_time(input_tensor)
        report.append("Forward Pass Times:")
        for name, time_val in sorted(forward_times.items(), key=lambda x: x[1]):
            report.append(f"  {name}: {time_val:.4f}s")
        report.append("")

        # Memory usage comparison
        memory_usage = self.compare_memory_usage(input_tensor)
        report.append("Memory Usage:")
        for name, mem in sorted(memory_usage.items(), key=lambda x: x[1]):
            report.append(f"  {name}: {mem:.2f} MB")
        report.append("")

        # Accuracy comparison (if ground truth provided)
        if ground_truth is not None:
            accuracy_metrics = self.compare_accuracy(input_tensor, ground_truth)
            report.append("Accuracy Metrics:")
            for name, metrics in accuracy_metrics.items():
                report.append(f"  {name}:")
                for metric, value in metrics.items():
                    report.append(f"    {metric}: {value:.6f}")
            report.append("")

        # Parameter count comparison
        report.append("Parameter Counts:")
        for name, model in self.models.items():
            param_count = sum(p.numel() for p in model.parameters())
            report.append(f"  {name}: {param_count:,} parameters")

        return "\n".join(report)


class TestDataValidator:
    """Validate test data quality and properties"""

    @staticmethod
    def validate_tensor_properties(tensor: torch.Tensor, name: str = "tensor") -> Dict[str, bool]:
        """Validate basic tensor properties"""
        results = {}

        # Check for finite values
        results["finite_values"] = torch.all(torch.isfinite(tensor))

        # Check for reasonable bounds
        tensor_max = torch.max(torch.abs(tensor))
        results["reasonable_bounds"] = tensor_max < 1e6

        # Check for sufficient variation
        tensor_std = torch.std(tensor)
        results["sufficient_variation"] = tensor_std > 1e-8

        # Check for proper normalization (optional)
        tensor_mean = torch.mean(tensor)
        results["approximately_centered"] = torch.abs(tensor_mean) < 10.0

        return results

    @staticmethod
    def validate_batch_consistency(batch_tensor: torch.Tensor) -> Dict[str, bool]:
        """Validate batch consistency"""
        results = {}

        if batch_tensor.shape[0] <= 1:
            results["batch_variation"] = True  # Can't check with single sample
            return results

        # Check variation across batch dimension
        batch_std = torch.std(batch_tensor, dim=0)
        mean_batch_std = torch.mean(batch_std)
        results["batch_variation"] = mean_batch_std > 1e-8

        # Check no duplicate samples
        batch_size = batch_tensor.shape[0]
        unique_samples = 0
        for i in range(batch_size):
            is_unique = True
            for j in range(i+1, batch_size):
                if torch.allclose(batch_tensor[i], batch_tensor[j], atol=1e-6):
                    is_unique = False
                    break
            if is_unique:
                unique_samples += 1

        results["no_duplicates"] = unique_samples == batch_size

        return results

    @staticmethod
    def validate_temporal_consistency(temporal_tensor: torch.Tensor) -> Dict[str, bool]:
        """Validate temporal consistency for sequence data"""
        results = {}

        if temporal_tensor.shape[1] <= 1:
            results["temporal_continuity"] = True
            return results

        # Check temporal continuity
        temporal_diff = torch.abs(temporal_tensor[:, 1:] - temporal_tensor[:, :-1])
        max_temporal_change = torch.max(temporal_diff)
        results["temporal_continuity"] = max_temporal_change < 100.0

        # Check temporal evolution (should have some change)
        mean_temporal_change = torch.mean(temporal_diff)
        results["temporal_evolution"] = mean_temporal_change > 1e-8

        return results


# Test decorators
def skip_if_no_cuda(test_func):
    """Skip test if CUDA is not available"""
    @functools.wraps(test_func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        return test_func(*args, **kwargs)
    return wrapper


def timeout(seconds: int):
    """Timeout decorator for tests"""
    def decorator(test_func):
        @functools.wraps(test_func)
        def wrapper(*args, **kwargs):
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError(f"Test timed out after {seconds} seconds")

            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)

            try:
                result = test_func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

            return result
        return wrapper
    return decorator


def repeat_test(num_runs: int):
    """Repeat test multiple times to check consistency"""
    def decorator(test_func):
        @functools.wraps(test_func)
        def wrapper(*args, **kwargs):
            results = []
            for i in range(num_runs):
                try:
                    result = test_func(*args, **kwargs)
                    results.append(result)
                except Exception as e:
                    raise AssertionError(f"Test failed on run {i+1}/{num_runs}: {e}")

            # Check consistency of results
            if results and all(isinstance(r, (int, float)) for r in results):
                mean_result = np.mean(results)
                std_result = np.std(results)
                cv = std_result / (abs(mean_result) + 1e-8)

                if cv > 0.1:  # Coefficient of variation > 10%
                    warnings.warn(f"High variability across runs: CV = {cv:.3f}")

            return results[-1]  # Return last result
        return wrapper
    return decorator


def profile_test(test_func):
    """Profile test execution"""
    @functools.wraps(test_func)
    def wrapper(*args, **kwargs):
        profiler = PerformanceProfiler()

        with profiler.profile(test_func.__name__):
            result = test_func(*args, **kwargs)

        # Print profiling results
        results = profiler.get_results()
        if test_func.__name__ in results:
            metrics = results[test_func.__name__]
            print(f"Test {test_func.__name__} performance:")
            print(f"  Execution time: {metrics['execution_time']:.4f}s")
            print(f"  Memory delta: {metrics['memory_delta']:.2f}MB")

        return result
    return wrapper


# Assertion helpers
class ExtendedAssertions:
    """Extended assertion methods for neural operator testing"""

    @staticmethod
    def assert_tensor_finite(tensor: torch.Tensor, msg: str = "Tensor contains non-finite values"):
        """Assert tensor contains only finite values"""
        if not torch.all(torch.isfinite(tensor)):
            raise AssertionError(msg)

    @staticmethod
    def assert_tensor_shape(tensor: torch.Tensor, expected_shape: Tuple[int, ...], msg: str = None):
        """Assert tensor has expected shape"""
        if tensor.shape != expected_shape:
            if msg is None:
                msg = f"Expected shape {expected_shape}, got {tensor.shape}"
            raise AssertionError(msg)

    @staticmethod
    def assert_tensors_close(tensor1: torch.Tensor, tensor2: torch.Tensor,
                           rtol: float = 1e-5, atol: float = 1e-8, msg: str = None):
        """Assert tensors are close"""
        if not torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol):
            if msg is None:
                max_diff = torch.max(torch.abs(tensor1 - tensor2))
                msg = f"Tensors not close. Max difference: {max_diff:.6e}"
            raise AssertionError(msg)

    @staticmethod
    def assert_gradient_finite(model: nn.Module, msg: str = "Model gradients contain non-finite values"):
        """Assert model gradients are finite"""
        for name, param in model.named_parameters():
            if param.grad is not None and not torch.all(torch.isfinite(param.grad)):
                raise AssertionError(f"{msg} (parameter: {name})")

    @staticmethod
    def assert_model_deterministic(model: nn.Module, input_tensor: torch.Tensor,
                                 msg: str = "Model is not deterministic"):
        """Assert model produces deterministic outputs"""
        model.eval()
        from src.core.utils.reproducibility import set_global_seed; set_global_seed(verbose=False)
        with torch.no_grad():
            output1 = model(input_tensor)

        from src.core.utils.reproducibility import set_global_seed; set_global_seed(verbose=False)
        with torch.no_grad():
            output2 = model(input_tensor)

        if not torch.allclose(output1, output2, rtol=1e-5, atol=1e-8):
            raise AssertionError(msg)


if __name__ == "__main__":
    # Test the utilities
    print("Testing utilities...")

    # Create a simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 3, padding=1)
            self.relu = nn.ReLU()

        def forward(self, x):
            B, T, C, H, W = x.shape
            x = x.view(B*T, C, H, W)
            x = self.relu(self.conv(x))
            return x.view(B, T, C, H, W)

    model = SimpleModel()
    input_tensor = torch.randn(2, 4, 3, 8, 8)

    # Test model suite
    suite = ModelTestSuite(model, "SimpleModel")
    basic_results = suite.run_basic_tests(input_tensor)
    print(f"Basic tests passed: {sum(r.passed for r in basic_results)}/{len(basic_results)}")

    # Test error injection
    corrupted_input = ErrorInjector.corrupt_input(input_tensor, "gaussian_noise", scale=0.1)
    print(f"Input corruption test: {corrupted_input.shape}")

    # Test data validation
    validation_results = TestDataValidator.validate_tensor_properties(input_tensor)
    print(f"Data validation: {validation_results}")

    print("✅ Test utilities working correctly!")


class EnhancedErrorMessages:
    """Utility class for generating descriptive error messages in tests."""

    @staticmethod
    def model_initialization_error(model_name: str, model_config: dict, data_params, error: Exception) -> str:
        """Generate enhanced error message for model initialization failures."""
        channels = data_params.dimension + len(data_params.simFields) + len(data_params.simParams)
        return (f"{model_name} initialization failed\n"
                f"  Model config: {model_config}\n"
                f"  Data params: size={data_params.dataSize}, channels={channels}\n"
                f"  Seq length: {data_params.sequenceLength}\n"
                f"  Error: {error}")

    @staticmethod
    def forward_pass_error(model_name: str, model_config: dict, input_shape: tuple,
                          expected_shape: tuple, dataset_name: str, error: Exception) -> str:
        """Generate enhanced error message for forward pass failures."""
        return (f"{model_name} forward pass failed\n"
                f"  Model config: {model_config}\n"
                f"  Input shape: {input_shape}\n"
                f"  Expected output: {expected_shape}\n"
                f"  Dataset: {dataset_name}\n"
                f"  Error: {error}")

    @staticmethod
    def shape_mismatch_error(expected: tuple, actual: tuple, context: str = "") -> str:
        """Generate enhanced error message for shape mismatches."""
        context_str = f" in {context}" if context else ""
        return (f"Shape mismatch{context_str}\n"
                f"  Expected: {expected}\n"
                f"  Actual: {actual}\n"
                f"  Difference: {[f'{e} vs {a}' for e, a in zip(expected, actual) if e != a]}")

    @staticmethod
    def analysis_error(test_name: str, model_name: str, dataset_name: str,
                      data_config: dict, expected_behavior: str, error: Exception) -> str:
        """Generate enhanced error message for analysis test failures."""
        return (f"{test_name} failed\n"
                f"  Model: {model_name}, Dataset: {dataset_name}\n"
                f"  Data config: {data_config}\n"
                f"  Expected: {expected_behavior}\n"
                f"  Error: {error}")

    @staticmethod
    def sequence_alignment_error(pred_shape: tuple, target_shape: tuple,
                               seq_config: tuple, error: Exception) -> str:
        """Generate enhanced error message for sequence alignment issues."""
        return (f"Sequence alignment failed\n"
                f"  Prediction shape: {pred_shape}\n"
                f"  Target shape: {target_shape}\n"
                f"  Sequence config: input_len={seq_config[0]}, output_len={seq_config[1]}\n"
                f"  Hint: Use pred[:, -{seq_config[1]}:] to align sequences\n"
                f"  Error: {error}")

    @staticmethod
    def channel_mismatch_error(expected_channels: int, actual_channels: int,
                             data_params, error: Exception) -> str:
        """Generate enhanced error message for channel mismatches."""
        return (f"Channel count mismatch\n"
                f"  Expected: {expected_channels} channels\n"
                f"  Actual: {actual_channels} channels\n"
                f"  Data config: dim={data_params.dimension}, fields={len(data_params.simFields)}, params={len(data_params.simParams)}\n"
                f"  Hint: Check simParams configuration - should be [] for tests\n"
                f"  Error: {error}")


def enhanced_fail(test_case, error_type: str, **kwargs):
    """
    Enhanced test failure function with context-aware error messages.

    Args:
        test_case: unittest.TestCase instance
        error_type: Type of error ('init', 'forward', 'shape', 'analysis', 'sequence', 'channel')
        **kwargs: Context-specific parameters for error message generation
    """
    error_generators = {
        'init': EnhancedErrorMessages.model_initialization_error,
        'forward': EnhancedErrorMessages.forward_pass_error,
        'shape': EnhancedErrorMessages.shape_mismatch_error,
        'analysis': EnhancedErrorMessages.analysis_error,
        'sequence': EnhancedErrorMessages.sequence_alignment_error,
        'channel': EnhancedErrorMessages.channel_mismatch_error,
    }

    if error_type not in error_generators:
        test_case.fail(f"Unknown error type: {error_type}. Error: {kwargs.get('error', 'Unknown')}")

    error_message = error_generators[error_type](**kwargs)
    test_case.fail(error_message)