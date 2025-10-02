#!/usr/bin/env python3
"""
Performance Optimization System for Generative Operators

This module provides comprehensive performance optimization for generative operator
sampling, including batched rollout processing, mixed precision training,
memory optimization, and adaptive batch sizing.

Author: Phase 3 Implementation
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import time
import psutil
import gc
from contextlib import contextmanager
from dataclasses import dataclass
import numpy as np


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization tracking."""
    throughput_steps_per_second: float
    memory_usage_gb: float
    gpu_utilization_percent: float
    batch_processing_time_seconds: float
    total_processing_time_seconds: float
    effective_batch_size: int
    mixed_precision_enabled: bool
    memory_optimization_level: str


class MemoryMonitor:
    """Monitor system and GPU memory usage."""

    def __init__(self):
        self.cpu_memory_history = []
        self.gpu_memory_history = []

    def get_cpu_memory_usage(self) -> float:
        """Get current CPU memory usage in GB."""
        return psutil.Process().memory_info().rss / (1024**3)

    def get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        return 0.0

    def get_gpu_memory_cached(self) -> float:
        """Get cached GPU memory in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_reserved() / (1024**3)
        return 0.0

    def log_memory_usage(self):
        """Log current memory usage."""
        cpu_mem = self.get_cpu_memory_usage()
        gpu_mem = self.get_gpu_memory_usage()
        gpu_cached = self.get_gpu_memory_cached()

        self.cpu_memory_history.append(cpu_mem)
        self.gpu_memory_history.append(gpu_mem)

        logging.debug(f"Memory - CPU: {cpu_mem:.2f}GB, GPU: {gpu_mem:.2f}GB, Cached: {gpu_cached:.2f}GB")

    def clear_gpu_cache(self):
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


class AdaptiveBatchSizer:
    """Adaptive batch sizing based on memory constraints."""

    def __init__(self, initial_batch_size: int = 4, memory_threshold: float = 0.85):
        self.initial_batch_size = initial_batch_size
        self.memory_threshold = memory_threshold  # Fraction of total GPU memory
        self.current_batch_size = initial_batch_size
        self.min_batch_size = 1
        self.max_batch_size = 32
        self.memory_monitor = MemoryMonitor()

        # Get total GPU memory
        if torch.cuda.is_available():
            self.total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            self.total_gpu_memory = 8.0  # Default assumption

    def adjust_batch_size(self, current_memory_usage: float) -> int:
        """
        Adjust batch size based on current memory usage.

        Args:
            current_memory_usage: Current GPU memory usage in GB

        Returns:
            int: Adjusted batch size
        """
        memory_fraction = current_memory_usage / self.total_gpu_memory

        if memory_fraction > self.memory_threshold:
            # Reduce batch size
            new_batch_size = max(self.min_batch_size, self.current_batch_size // 2)
            logging.info(f"Reducing batch size from {self.current_batch_size} to {new_batch_size} "
                        f"(memory: {memory_fraction:.2f})")
        elif memory_fraction < self.memory_threshold * 0.7:
            # Increase batch size if we have headroom
            new_batch_size = min(self.max_batch_size, self.current_batch_size * 2)
            if new_batch_size != self.current_batch_size:
                logging.info(f"Increasing batch size from {self.current_batch_size} to {new_batch_size}")
        else:
            new_batch_size = self.current_batch_size

        self.current_batch_size = new_batch_size
        return new_batch_size

    def get_optimal_batch_size(self, tensor_shape: Tuple[int, ...]) -> int:
        """
        Get optimal batch size for given tensor shape.

        Args:
            tensor_shape: Shape of tensors to process

        Returns:
            int: Optimal batch size
        """
        # Estimate memory usage per sample
        elements_per_sample = np.prod(tensor_shape[1:])  # Exclude batch dimension
        bytes_per_sample = elements_per_sample * 4  # Assume float32
        gb_per_sample = bytes_per_sample / (1024**3)

        # Calculate safe batch size
        available_memory = self.total_gpu_memory * self.memory_threshold
        safe_batch_size = max(1, int(available_memory / (gb_per_sample * 3)))  # Factor of 3 for safety

        return min(safe_batch_size, self.max_batch_size)


class MixedPrecisionManager:
    """Manager for mixed precision operations."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        self.scaler = GradScaler() if self.enabled else None

    @contextmanager
    def autocast_context(self):
        """Context manager for mixed precision."""
        if self.enabled:
            with autocast():
                yield
        else:
            yield

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision training."""
        if self.enabled and self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def step_optimizer(self, optimizer: torch.optim.Optimizer):
        """Step optimizer with mixed precision."""
        if self.enabled and self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()


class BatchedRolloutProcessor:
    """Processor for efficient batched rollout operations."""

    def __init__(self, model, device: torch.device, optimization_level: str = "balanced"):
        self.model = model
        self.device = device
        self.optimization_level = optimization_level
        self.memory_monitor = MemoryMonitor()
        self.batch_sizer = AdaptiveBatchSizer()
        self.mixed_precision = MixedPrecisionManager(enabled=True)

        # Optimization configurations
        self.optimization_configs = {
            "conservative": {
                "max_batch_size": 2,
                "memory_threshold": 0.7,
                "enable_checkpointing": True,
                "enable_mixed_precision": False,
            },
            "balanced": {
                "max_batch_size": 8,
                "memory_threshold": 0.85,
                "enable_checkpointing": True,
                "enable_mixed_precision": True,
            },
            "aggressive": {
                "max_batch_size": 16,
                "memory_threshold": 0.95,
                "enable_checkpointing": False,
                "enable_mixed_precision": True,
            }
        }

        self._configure_optimization()

    def _configure_optimization(self):
        """Configure optimization based on selected level."""
        config = self.optimization_configs[self.optimization_level]

        self.batch_sizer.max_batch_size = config["max_batch_size"]
        self.batch_sizer.memory_threshold = config["memory_threshold"]
        self.mixed_precision.enabled = config["enable_mixed_precision"]

        if config["enable_checkpointing"]:
            self._enable_gradient_checkpointing()

        logging.info(f"Configured {self.optimization_level} optimization level")

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        elif hasattr(self.model, 'enable_gradient_checkpointing'):
            self.model.enable_gradient_checkpointing()

    def process_batch_rollout(self, initial_conditions: torch.Tensor,
                            num_steps: int,
                            chunk_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Process rollout in batches with optimization.

        Args:
            initial_conditions: Initial conditions [B, T, C, H, W]
            num_steps: Number of rollout steps
            chunk_size: Optional chunk size for processing

        Returns:
            dict: Processing results with metrics
        """
        start_time = time.time()
        self.memory_monitor.log_memory_usage()

        batch_size = initial_conditions.shape[0]

        # Determine optimal chunk size
        if chunk_size is None:
            chunk_size = self.batch_sizer.get_optimal_batch_size(initial_conditions.shape)

        chunk_size = min(chunk_size, batch_size)

        logging.info(f"Processing batch rollout: {batch_size} samples, {num_steps} steps, "
                    f"chunk size: {chunk_size}")

        # Process in chunks
        all_predictions = []
        processing_times = []

        for i in range(0, batch_size, chunk_size):
            chunk_start = time.time()

            end_idx = min(i + chunk_size, batch_size)
            chunk_data = initial_conditions[i:end_idx]

            # Process chunk with mixed precision
            with self.mixed_precision.autocast_context():
                chunk_predictions = self._process_chunk_rollout(chunk_data, num_steps)

            all_predictions.append(chunk_predictions.cpu())

            chunk_time = time.time() - chunk_start
            processing_times.append(chunk_time)

            # Monitor memory and adjust if needed
            current_memory = self.memory_monitor.get_gpu_memory_usage()
            new_chunk_size = self.batch_sizer.adjust_batch_size(current_memory)

            if new_chunk_size != chunk_size:
                chunk_size = new_chunk_size

            # Clear cache periodically
            if i % (chunk_size * 4) == 0:
                self.memory_monitor.clear_gpu_cache()

            logging.debug(f"Processed chunk {i//chunk_size + 1}, time: {chunk_time:.2f}s")

        # Combine results
        final_predictions = torch.cat(all_predictions, dim=0)
        total_time = time.time() - start_time

        # Generate metrics
        metrics = self._generate_performance_metrics(
            batch_size, num_steps, total_time, processing_times
        )

        return {
            "predictions": final_predictions,
            "metrics": metrics,
            "processing_info": {
                "total_time": total_time,
                "chunk_times": processing_times,
                "final_chunk_size": chunk_size,
                "memory_usage": self.memory_monitor.get_gpu_memory_usage()
            }
        }

    def _process_chunk_rollout(self, chunk_data: torch.Tensor, num_steps: int) -> torch.Tensor:
        """
        Process a single chunk of data through rollout.

        Args:
            chunk_data: Chunk data [chunk_size, T, C, H, W]
            num_steps: Number of rollout steps

        Returns:
            torch.Tensor: Rollout predictions
        """
        chunk_data = chunk_data.to(self.device)

        # Use the UCAR rollout from the model
        if hasattr(self.model, 'ucar_rollout'):
            results = self.model.ucar_rollout.rollout_sequence(
                initial_conditions=chunk_data,
                num_steps=num_steps,
                correction_interval=1,
                correction_strength=1.0
            )
            return results['predictions']
        else:
            # Fallback to standard model prediction
            with torch.no_grad():
                predictions = self.model(chunk_data)
            return predictions

    def _generate_performance_metrics(self, batch_size: int, num_steps: int,
                                    total_time: float, processing_times: List[float]) -> PerformanceMetrics:
        """Generate performance metrics."""
        throughput = (batch_size * num_steps) / total_time
        avg_processing_time = np.mean(processing_times)
        memory_usage = self.memory_monitor.get_gpu_memory_usage()

        return PerformanceMetrics(
            throughput_steps_per_second=throughput,
            memory_usage_gb=memory_usage,
            gpu_utilization_percent=0.0,  # Would need additional monitoring
            batch_processing_time_seconds=avg_processing_time,
            total_processing_time_seconds=total_time,
            effective_batch_size=self.batch_sizer.current_batch_size,
            mixed_precision_enabled=self.mixed_precision.enabled,
            memory_optimization_level=self.optimization_level
        )


class PerformanceOptimizer:
    """Main performance optimizer for generative operator sampling."""

    def __init__(self, device: torch.device = None, optimization_level: str = "balanced"):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.optimization_level = optimization_level
        self.memory_monitor = MemoryMonitor()

        # Performance tracking
        self.performance_history = []

    def optimize_model(self, model: nn.Module) -> nn.Module:
        """
        Apply performance optimizations to model.

        Args:
            model: Model to optimize

        Returns:
            nn.Module: Optimized model
        """
        model = model.to(self.device)

        # Apply torch optimizations
        if hasattr(torch, 'compile') and self.optimization_level == "aggressive":
            try:
                model = torch.compile(model)
                logging.info("Applied torch.compile optimization")
            except Exception as e:
                logging.warning(f"Could not apply torch.compile: {e}")

        # Enable mixed precision if supported
        if self.device.type == "cuda" and self.optimization_level in ["balanced", "aggressive"]:
            # Model is already set up for mixed precision in MixedPrecisionManager
            pass

        # Apply gradient checkpointing for memory efficiency
        if self.optimization_level in ["conservative", "balanced"]:
            self._apply_gradient_checkpointing(model)

        return model

    def _apply_gradient_checkpointing(self, model: nn.Module):
        """Apply gradient checkpointing to reduce memory usage."""
        for module in model.modules():
            if hasattr(module, 'gradient_checkpointing_enable'):
                module.gradient_checkpointing_enable()

    def create_batched_processor(self, model: nn.Module) -> BatchedRolloutProcessor:
        """
        Create a batched rollout processor for the model.

        Args:
            model: Model to create processor for

        Returns:
            BatchedRolloutProcessor: Configured processor
        """
        return BatchedRolloutProcessor(model, self.device, self.optimization_level)

    def benchmark_configuration(self, model: nn.Module, test_data: torch.Tensor,
                              num_steps: int = 10) -> Dict[str, Any]:
        """
        Benchmark different optimization configurations.

        Args:
            model: Model to benchmark
            test_data: Test data for benchmarking
            num_steps: Number of rollout steps

        Returns:
            dict: Benchmark results
        """
        results = {}

        for level in ["conservative", "balanced", "aggressive"]:
            logging.info(f"Benchmarking {level} optimization level")

            # Create processor with this level
            processor = BatchedRolloutProcessor(model, self.device, level)

            try:
                # Run benchmark
                start_time = time.time()
                result = processor.process_batch_rollout(test_data, num_steps)
                benchmark_time = time.time() - start_time

                results[level] = {
                    "metrics": result["metrics"],
                    "benchmark_time": benchmark_time,
                    "success": True
                }

            except Exception as e:
                results[level] = {
                    "error": str(e),
                    "success": False
                }

            # Clear cache between benchmarks
            self.memory_monitor.clear_gpu_cache()

        return results

    def get_recommended_configuration(self, available_memory_gb: float) -> str:
        """
        Get recommended optimization configuration based on available memory.

        Args:
            available_memory_gb: Available GPU memory in GB

        Returns:
            str: Recommended optimization level
        """
        if available_memory_gb < 6:
            return "conservative"
        elif available_memory_gb < 12:
            return "balanced"
        else:
            return "aggressive"

    def monitor_performance(self, metrics: PerformanceMetrics):
        """Monitor and log performance metrics."""
        self.performance_history.append(metrics)

        logging.info(f"Performance - Throughput: {metrics.throughput_steps_per_second:.1f} steps/s, "
                    f"Memory: {metrics.memory_usage_gb:.2f}GB, "
                    f"Batch size: {metrics.effective_batch_size}")

        # Log warnings for performance issues
        if metrics.memory_usage_gb > 10:
            logging.warning(f"High memory usage: {metrics.memory_usage_gb:.2f}GB")

        if metrics.throughput_steps_per_second < 1:
            logging.warning(f"Low throughput: {metrics.throughput_steps_per_second:.2f} steps/s")


# Convenience functions
def optimize_generative_operator(model: nn.Module, device: torch.device = None,
                               optimization_level: str = "balanced") -> Tuple[nn.Module, BatchedRolloutProcessor]:
    """
    Optimize a generative operator model for performance.

    Args:
        model: Generative operator model
        device: Target device
        optimization_level: Optimization level

    Returns:
        tuple: (optimized_model, batched_processor)
    """
    optimizer = PerformanceOptimizer(device, optimization_level)
    optimized_model = optimizer.optimize_model(model)
    processor = optimizer.create_batched_processor(optimized_model)

    return optimized_model, processor


def run_optimized_rollout(model: nn.Module, initial_conditions: torch.Tensor,
                         num_steps: int, optimization_level: str = "balanced") -> Dict[str, Any]:
    """
    Run optimized rollout with automatic performance optimization.

    Args:
        model: Generative operator model
        initial_conditions: Initial conditions
        num_steps: Number of rollout steps
        optimization_level: Optimization level

    Returns:
        dict: Rollout results with performance metrics
    """
    optimized_model, processor = optimize_generative_operator(model, optimization_level=optimization_level)
    return processor.process_batch_rollout(initial_conditions, num_steps)


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    # Test with dummy data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data = torch.randn(8, 10, 3, 64, 64).to(device)

    # Create dummy model for testing
    class DummyModel(nn.Module):
        def forward(self, x):
            return x

    model = DummyModel()

    # Test optimization
    optimizer = PerformanceOptimizer(device, "balanced")
    optimized_model = optimizer.optimize_model(model)
    processor = optimizer.create_batched_processor(optimized_model)

    # Test processing
    result = processor.process_batch_rollout(test_data, num_steps=5)
    print(f"Processing completed. Metrics: {result['metrics']}")