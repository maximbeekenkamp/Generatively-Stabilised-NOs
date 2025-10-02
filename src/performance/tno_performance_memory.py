# TNO Performance Optimization and Memory Management Tools
# Phase 4: Performance and Memory Management

import torch
import numpy as np
import time
import psutil
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import gc

# Handle optional GPU dependencies
try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False
    GPUtil = None

@dataclass
class MemoryProfile:
    """Memory usage profile for TNO models."""
    cpu_ram_mb: float
    gpu_vram_mb: float
    peak_cpu_mb: float
    peak_gpu_mb: float
    batch_size: int
    sequence_length: int
    K: int
    L: int
    spatial_dims: Tuple[int, int]

class TNOMemoryOptimizer:
    """
    Memory optimization utilities for TNO sampling and training.
    """

    def __init__(self, device='cuda'):
        """
        Initialize memory optimizer.

        Args:
            device: Computing device (cuda/cpu)
        """
        self.device = device
        self.gpu_available = torch.cuda.is_available() and device == 'cuda'
        self.profiles = []

    def profile_memory_usage(self, model, data_shape: Tuple, tno_config: Dict) -> MemoryProfile:
        """
        Profile memory usage for TNO model with given configuration.

        Args:
            model: TNO model instance
            data_shape: Input data shape (B, T, C, H, W)
            tno_config: TNO configuration

        Returns:
            MemoryProfile: Memory usage statistics
        """
        B, T, C, H, W = data_shape
        K = tno_config.get('temporal_bundling_K', 1)
        L = tno_config.get('history_length_L', 1)

        # Clear cache
        if self.gpu_available:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

        # Initial memory
        cpu_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        gpu_before = 0
        if self.gpu_available:
            gpu_before = torch.cuda.memory_allocated() / 1024 / 1024

        # Create dummy input
        dummy_input = torch.randn(B, T, C, H, W, device=self.device)

        # Forward pass with memory tracking
        peak_cpu = cpu_before
        peak_gpu = gpu_before

        with torch.no_grad():
            # TNO forward pass
            if hasattr(model, 'forwardTNO'):
                _ = model.forwardTNO(dummy_input)
            else:
                _ = model(dummy_input)

            # Check memory after forward
            cpu_after = psutil.Process().memory_info().rss / 1024 / 1024
            gpu_after = 0
            if self.gpu_available:
                torch.cuda.synchronize()
                gpu_after = torch.cuda.memory_allocated() / 1024 / 1024
                peak_gpu = torch.cuda.max_memory_allocated() / 1024 / 1024

            peak_cpu = max(peak_cpu, cpu_after)

        # Clean up
        del dummy_input
        if self.gpu_available:
            torch.cuda.empty_cache()
        gc.collect()

        profile = MemoryProfile(
            cpu_ram_mb=cpu_after - cpu_before,
            gpu_vram_mb=gpu_after - gpu_before if self.gpu_available else 0,
            peak_cpu_mb=peak_cpu - cpu_before,
            peak_gpu_mb=peak_gpu - gpu_before if self.gpu_available else 0,
            batch_size=B,
            sequence_length=T,
            K=K,
            L=L,
            spatial_dims=(H, W)
        )

        self.profiles.append(profile)
        return profile

    def find_optimal_batch_size(self, model, base_shape: Tuple, tno_config: Dict,
                               max_memory_mb: float = 8000) -> int:
        """
        Find optimal batch size for TNO given memory constraints.

        Args:
            model: TNO model
            base_shape: Base data shape (1, T, C, H, W)
            tno_config: TNO configuration
            max_memory_mb: Maximum memory budget in MB

        Returns:
            int: Optimal batch size
        """
        _, T, C, H, W = base_shape

        # Binary search for optimal batch size
        min_batch = 1
        max_batch = 64
        optimal_batch = 1

        while min_batch <= max_batch:
            test_batch = (min_batch + max_batch) // 2
            test_shape = (test_batch, T, C, H, W)

            try:
                profile = self.profile_memory_usage(model, test_shape, tno_config)

                total_memory = profile.gpu_vram_mb if self.gpu_available else profile.cpu_ram_mb
                peak_memory = profile.peak_gpu_mb if self.gpu_available else profile.peak_cpu_mb

                if peak_memory < max_memory_mb * 0.9:  # 90% safety margin
                    optimal_batch = test_batch
                    min_batch = test_batch + 1
                else:
                    max_batch = test_batch - 1

            except (RuntimeError, torch.cuda.OutOfMemoryError):
                max_batch = test_batch - 1
                if self.gpu_available:
                    torch.cuda.empty_cache()

        return optimal_batch

    def optimize_tno_sampling_memory(self, model, data_loader, tno_config: Dict) -> Dict:
        """
        Optimize memory usage during TNO sampling with various strategies.

        Args:
            model: TNO model
            data_loader: Data loader
            tno_config: TNO configuration

        Returns:
            dict: Optimization recommendations
        """
        recommendations = {}

        # 1. Gradient checkpointing for long sequences
        seq_length = next(iter(data_loader))['data'].shape[1]
        if seq_length > 100:
            recommendations['gradient_checkpointing'] = True
            recommendations['checkpoint_interval'] = 50

        # 2. Mixed precision recommendations
        if self.gpu_available and torch.cuda.get_device_capability()[0] >= 7:
            recommendations['use_mixed_precision'] = True
            recommendations['autocast_enabled'] = True

        # 3. Temporal bundling optimization
        K = tno_config.get('temporal_bundling_K', 1)
        if seq_length > 200 and K > 2:
            # Reduce K for very long sequences to save memory
            recommendations['adaptive_K'] = min(K, 2)
            recommendations['reason_K'] = "Reduced K for long sequences to save memory"

        # 4. Sequential processing for large batches
        batch_size = next(iter(data_loader))['data'].shape[0]
        if batch_size > 16:
            recommendations['sequential_processing'] = True
            recommendations['mini_batch_size'] = 8

        # 5. Memory-efficient data loading
        recommendations['pin_memory'] = self.gpu_available
        recommendations['num_workers'] = min(4, psutil.cpu_count())
        recommendations['prefetch_factor'] = 2

        return recommendations

    def create_memory_efficient_sampler(self, model, test_dataset, tno_config: Dict,
                                       memory_budget_mb: float = 8000):
        """
        Create memory-efficient sampling strategy for TNO.

        Args:
            model: TNO model
            test_dataset: Test dataset
            tno_config: TNO configuration
            memory_budget_mb: Memory budget in MB

        Returns:
            callable: Memory-efficient sampling function
        """

        def efficient_sampling(batch_data, checkpoint_interval=50):
            """
            Memory-efficient TNO sampling with checkpointing.

            Args:
                batch_data: Input batch
                checkpoint_interval: Interval for memory checkpointing

            Returns:
                Predictions tensor
            """
            B, T, C, H, W = batch_data.shape
            K = tno_config.get('temporal_bundling_K', 1)
            L = tno_config.get('history_length_L', 1)

            predictions = []

            # Process in chunks to manage memory
            for t in range(0, T, checkpoint_interval):
                chunk_end = min(t + checkpoint_interval, T)
                chunk_data = batch_data[:, t:chunk_end]

                with torch.no_grad():
                    if hasattr(model, 'forwardTNO'):
                        chunk_pred = model.forwardTNO(chunk_data)
                    else:
                        chunk_pred = model(chunk_data)

                # Move to CPU to free GPU memory
                predictions.append(chunk_pred.cpu())

                # Clear GPU cache periodically
                if self.gpu_available and t % (checkpoint_interval * 2) == 0:
                    torch.cuda.empty_cache()

            # Concatenate predictions
            full_predictions = torch.cat(predictions, dim=1)

            return full_predictions

        return efficient_sampling


class TNOPerformanceProfiler:
    """
    Performance profiling tools for TNO models.
    """

    def __init__(self):
        """Initialize performance profiler."""
        self.timing_records = []
        self.throughput_records = []

    def profile_forward_pass(self, model, input_shape: Tuple,
                            tno_config: Dict, num_runs: int = 10) -> Dict:
        """
        Profile TNO forward pass performance.

        Args:
            model: TNO model
            input_shape: Input data shape
            tno_config: TNO configuration
            num_runs: Number of runs for averaging

        Returns:
            dict: Performance metrics
        """
        B, T, C, H, W = input_shape
        device = next(model.parameters()).device

        # Warm-up
        dummy = torch.randn(1, min(10, T), C, H, W, device=device)
        with torch.no_grad():
            _ = model(dummy)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        # Timing runs
        times = []

        for _ in range(num_runs):
            data = torch.randn(B, T, C, H, W, device=device)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            start = time.perf_counter()

            with torch.no_grad():
                if hasattr(model, 'forwardTNO'):
                    _ = model.forwardTNO(data)
                else:
                    _ = model(data)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append(end - start)

        # Calculate metrics
        metrics = {
            'mean_time_s': np.mean(times),
            'std_time_s': np.std(times),
            'min_time_s': np.min(times),
            'max_time_s': np.max(times),
            'throughput_fps': B * T / np.mean(times),  # Frames per second
            'time_per_frame_ms': np.mean(times) * 1000 / (B * T),
            'time_per_K_bundle_ms': np.mean(times) * 1000 * tno_config.get('K', 1) / (B * T),
        }

        self.timing_records.append(metrics)
        return metrics

    def profile_memory_bandwidth(self, model, input_shape: Tuple, tno_config: Dict) -> Dict:
        """
        Profile memory bandwidth utilization.

        Args:
            model: TNO model
            input_shape: Input shape
            tno_config: TNO configuration

        Returns:
            dict: Bandwidth metrics
        """
        B, T, C, H, W = input_shape

        # Calculate data sizes
        input_size_mb = B * T * C * H * W * 4 / 1024 / 1024  # float32

        # Estimate output size (may differ due to K bundling)
        K = tno_config.get('temporal_bundling_K', 1)
        output_size_mb = B * (T // K) * K * C * H * W * 4 / 1024 / 1024

        # Profile timing
        timing = self.profile_forward_pass(model, input_shape, tno_config, num_runs=5)

        # Calculate bandwidth
        total_data_mb = input_size_mb + output_size_mb
        bandwidth_gbps = total_data_mb / 1024 / timing['mean_time_s']

        metrics = {
            'input_size_mb': input_size_mb,
            'output_size_mb': output_size_mb,
            'total_data_mb': total_data_mb,
            'bandwidth_gbps': bandwidth_gbps,
            'efficiency_percent': min(100, bandwidth_gbps / 500 * 100)  # Assume 500 GB/s max
        }

        return metrics

    def compare_tno_configurations(self, model_factory, base_shape: Tuple,
                                  config_variations: List[Dict]) -> Dict:
        """
        Compare performance of different TNO configurations.

        Args:
            model_factory: Function to create model with config
            base_shape: Base input shape
            config_variations: List of TNO configurations to test

        Returns:
            dict: Comparison results
        """
        results = {}

        for i, config in enumerate(config_variations):
            config_name = f"K{config['K']}_L{config['L']}"

            # Create model with configuration
            model = model_factory(config)

            # Profile performance
            perf_metrics = self.profile_forward_pass(model, base_shape, config)
            mem_optimizer = TNOMemoryOptimizer()
            mem_profile = mem_optimizer.profile_memory_usage(model, base_shape, config)

            results[config_name] = {
                'config': config,
                'performance': perf_metrics,
                'memory': {
                    'gpu_mb': mem_profile.gpu_vram_mb,
                    'peak_gpu_mb': mem_profile.peak_gpu_mb,
                },
                'efficiency_score': perf_metrics['throughput_fps'] / (mem_profile.peak_gpu_mb + 1)
            }

            # Clean up
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Find best configuration
        best_perf = max(results.keys(), key=lambda k: results[k]['performance']['throughput_fps'])
        best_mem = min(results.keys(), key=lambda k: results[k]['memory']['peak_gpu_mb'])
        best_efficiency = max(results.keys(), key=lambda k: results[k]['efficiency_score'])

        results['summary'] = {
            'best_performance': best_perf,
            'best_memory': best_mem,
            'best_efficiency': best_efficiency,
        }

        return results

    def generate_performance_report(self) -> str:
        """
        Generate comprehensive performance report.

        Returns:
            str: Formatted performance report
        """
        report = ["=" * 60]
        report.append("TNO PERFORMANCE ANALYSIS REPORT")
        report.append("=" * 60)

        if self.timing_records:
            report.append("\n## Forward Pass Performance")
            report.append("-" * 40)

            for i, record in enumerate(self.timing_records):
                report.append(f"\nRun {i+1}:")
                report.append(f"  Mean time: {record['mean_time_s']:.3f}s")
                report.append(f"  Throughput: {record['throughput_fps']:.1f} fps")
                report.append(f"  Time per frame: {record['time_per_frame_ms']:.2f}ms")
                if 'time_per_K_bundle_ms' in record:
                    report.append(f"  Time per K-bundle: {record['time_per_K_bundle_ms']:.2f}ms")

        if self.throughput_records:
            report.append("\n## Throughput Analysis")
            report.append("-" * 40)

            avg_throughput = np.mean([r['throughput_fps'] for r in self.timing_records if 'throughput_fps' in r])
            report.append(f"  Average throughput: {avg_throughput:.1f} fps")

        report.append("\n" + "=" * 60)
        return "\n".join(report)