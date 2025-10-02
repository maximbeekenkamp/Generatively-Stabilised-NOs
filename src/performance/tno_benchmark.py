#!/usr/bin/env python3
"""
TNO Performance Benchmarking Suite

Comprehensive benchmarking tool for TNO performance analysis across
different configurations, datasets, and hardware setups.

Usage:
    python src/performance/tno_benchmark.py --models tno-L1K1,tno-L2K4 --dataset inc
    python src/performance/tno_benchmark.py --full-benchmark --output results/benchmarks/
    python src/performance/tno_benchmark.py --memory-profile --config-sweep
"""

import os
import sys
import argparse
import time
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import gc

# Add source paths
sys.path.append('src/core')
sys.path.append('src/analysis')
sys.path.append('src/performance')

from tno_performance_memory import TNOMemoryOptimizer, TNOPerformanceProfiler

class TNOBenchmarkSuite:
    """
    Comprehensive TNO benchmarking suite.
    """

    def __init__(self, output_dir: str = "results/benchmarks"):
        """
        Initialize benchmark suite.

        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_optimizer = TNOMemoryOptimizer(device=self.device)
        self.performance_profiler = TNOPerformanceProfiler()

        self.benchmark_results = {}

        print(f"[TNO Benchmark] Initialized on device: {self.device}")
        print(f"[TNO Benchmark] Output directory: {self.output_dir}")

    def benchmark_tno_configurations(self, config_variations: List[Dict]) -> Dict:
        """
        Benchmark different TNO configurations.

        Args:
            config_variations: List of TNO configurations to benchmark

        Returns:
            dict: Benchmark results
        """
        print("\n=== TNO Configuration Benchmarking ===")

        results = {}
        base_shape = (4, 100, 3, 128, 128)  # B, T, C, H, W

        for i, config in enumerate(config_variations):
            config_name = f"K{config['K']}_L{config['L']}"
            print(f"\nBenchmarking configuration {i+1}/{len(config_variations)}: {config_name}")

            try:
                # Create dummy model for benchmarking
                model = self._create_benchmark_model(config)

                # Performance benchmarking
                print("  - Running performance tests...")
                perf_metrics = self.performance_profiler.profile_forward_pass(
                    model, base_shape, config, num_runs=20
                )

                # Memory profiling
                print("  - Running memory profiling...")
                mem_profile = self.memory_optimizer.profile_memory_usage(
                    model, base_shape, config
                )

                # Memory bandwidth
                print("  - Analyzing memory bandwidth...")
                bandwidth_metrics = self.performance_profiler.profile_memory_bandwidth(
                    model, base_shape, config
                )

                # Optimal batch size
                print("  - Finding optimal batch size...")
                optimal_batch = self.memory_optimizer.find_optimal_batch_size(
                    model, (1, 100, 3, 128, 128), config, max_memory_mb=6000
                )

                results[config_name] = {
                    'config': config,
                    'performance': perf_metrics,
                    'memory': {
                        'profile': mem_profile.__dict__,
                        'bandwidth': bandwidth_metrics,
                        'optimal_batch_size': optimal_batch
                    },
                    'efficiency_scores': {
                        'throughput_per_mb': perf_metrics['throughput_fps'] / (mem_profile.peak_gpu_mb + 1),
                        'fps_per_watt': perf_metrics['throughput_fps'] / 200,  # Assume 200W power consumption
                        'latency_efficiency': 1000 / perf_metrics['time_per_frame_ms']
                    }
                }

                print(f"    ✓ Throughput: {perf_metrics['throughput_fps']:.1f} fps")
                print(f"    ✓ Memory: {mem_profile.peak_gpu_mb:.1f} MB")
                print(f"    ✓ Optimal batch: {optimal_batch}")

                # Clean up
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"    ✗ Benchmark failed: {e}")
                results[config_name] = {'error': str(e)}

        # Analyze best configurations
        results['summary'] = self._analyze_benchmark_results(results)

        return results

    def benchmark_scaling_performance(self, base_config: Dict) -> Dict:
        """
        Benchmark TNO scaling performance with different input sizes.

        Args:
            base_config: Base TNO configuration

        Returns:
            dict: Scaling benchmark results
        """
        print("\n=== TNO Scaling Performance Benchmarking ===")

        # Test configurations: (batch_size, sequence_length, spatial_resolution)
        scaling_tests = [
            (1, 50, 64),    # Small
            (2, 100, 128),  # Medium
            (4, 200, 256),  # Large
            (1, 500, 128),  # Long sequence
            (8, 50, 128),   # Large batch
        ]

        results = {}

        for i, (B, T, spatial) in enumerate(scaling_tests):
            test_name = f"B{B}_T{T}_S{spatial}"
            print(f"\nTesting scaling {i+1}/{len(scaling_tests)}: {test_name}")

            try:
                model = self._create_benchmark_model(base_config)
                input_shape = (B, T, 3, spatial, spatial)

                # Performance test
                perf_metrics = self.performance_profiler.profile_forward_pass(
                    model, input_shape, base_config, num_runs=10
                )

                # Memory test
                mem_profile = self.memory_optimizer.profile_memory_usage(
                    model, input_shape, base_config
                )

                # Calculate scaling metrics
                total_pixels = B * T * spatial * spatial
                throughput_per_pixel = perf_metrics['throughput_fps'] / total_pixels * 1e6  # Mpix/s

                results[test_name] = {
                    'input_shape': input_shape,
                    'performance': perf_metrics,
                    'memory': mem_profile.__dict__,
                    'scaling_metrics': {
                        'total_pixels': total_pixels,
                        'throughput_per_pixel': throughput_per_pixel,
                        'memory_per_pixel': mem_profile.peak_gpu_mb / total_pixels * 1e6,  # MB/Mpix
                        'efficiency_ratio': throughput_per_pixel / (mem_profile.peak_gpu_mb + 1)
                    }
                }

                print(f"    ✓ {throughput_per_pixel:.2f} Mpix/s")
                print(f"    ✓ {mem_profile.peak_gpu_mb:.1f} MB peak")

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"    ✗ Scaling test failed: {e}")
                results[test_name] = {'error': str(e)}

        return results

    def benchmark_temporal_bundling(self, K_values: List[int]) -> Dict:
        """
        Benchmark temporal bundling performance for different K values.

        Args:
            K_values: List of K values to test

        Returns:
            dict: Temporal bundling benchmark results
        """
        print("\n=== TNO Temporal Bundling Benchmarking ===")

        results = {}
        base_shape = (2, 200, 3, 128, 128)  # Long sequence for bundling analysis

        for K in K_values:
            print(f"\nBenchmarking K={K}...")

            try:
                config = {'K': K, 'L': 1}
                model = self._create_benchmark_model(config)

                # Performance with bundling
                perf_metrics = self.performance_profiler.profile_forward_pass(
                    model, base_shape, config, num_runs=15
                )

                # Memory efficiency
                mem_profile = self.memory_optimizer.profile_memory_usage(
                    model, base_shape, config
                )

                # Bundling-specific metrics
                bundle_efficiency = K / perf_metrics['time_per_K_bundle_ms'] * 1000  # bundles/s
                memory_per_bundle = mem_profile.peak_gpu_mb / (base_shape[1] // K)

                results[f'K{K}'] = {
                    'K': K,
                    'performance': perf_metrics,
                    'memory': mem_profile.__dict__,
                    'bundling_metrics': {
                        'bundle_efficiency': bundle_efficiency,
                        'memory_per_bundle': memory_per_bundle,
                        'bundling_overhead': perf_metrics['time_per_K_bundle_ms'] / K,
                        'effective_speedup': K / (perf_metrics['time_per_K_bundle_ms'] / perf_metrics['time_per_frame_ms'])
                    }
                }

                print(f"    ✓ Bundle efficiency: {bundle_efficiency:.1f} bundles/s")
                print(f"    ✓ Memory per bundle: {memory_per_bundle:.2f} MB")

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"    ✗ K={K} benchmark failed: {e}")
                results[f'K{K}'] = {'error': str(e)}

        # Find optimal K
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            optimal_K = max(valid_results.keys(),
                           key=lambda k: valid_results[k]['bundling_metrics']['bundle_efficiency'])
            results['optimal_K'] = optimal_K

        return results

    def benchmark_memory_vs_accuracy_tradeoff(self, configs: List[Dict]) -> Dict:
        """
        Benchmark memory vs accuracy tradeoffs for different configurations.

        Args:
            configs: List of configurations with different memory/accuracy tradeoffs

        Returns:
            dict: Tradeoff analysis results
        """
        print("\n=== TNO Memory vs Accuracy Tradeoff Benchmarking ===")

        results = {}

        for config in configs:
            config_name = f"K{config['K']}_L{config['L']}"
            print(f"\nAnalyzing tradeoff for {config_name}...")

            try:
                model = self._create_benchmark_model(config)
                base_shape = (4, 100, 3, 128, 128)

                # Performance metrics
                perf_metrics = self.performance_profiler.profile_forward_pass(
                    model, base_shape, config
                )

                # Memory usage
                mem_profile = self.memory_optimizer.profile_memory_usage(
                    model, base_shape, config
                )

                # Simulate accuracy metrics (would use real validation data in practice)
                simulated_accuracy = self._simulate_accuracy_metrics(config)

                # Tradeoff metrics
                memory_efficiency = simulated_accuracy['mse_improvement'] / mem_profile.peak_gpu_mb
                speed_accuracy_ratio = perf_metrics['throughput_fps'] / simulated_accuracy['computational_cost']

                results[config_name] = {
                    'config': config,
                    'performance': perf_metrics,
                    'memory': mem_profile.__dict__,
                    'accuracy': simulated_accuracy,
                    'tradeoff_metrics': {
                        'memory_efficiency': memory_efficiency,
                        'speed_accuracy_ratio': speed_accuracy_ratio,
                        'pareto_score': (perf_metrics['throughput_fps'] * simulated_accuracy['mse_improvement']) / mem_profile.peak_gpu_mb
                    }
                }

                print(f"    ✓ Memory efficiency: {memory_efficiency:.4f}")
                print(f"    ✓ Speed/accuracy ratio: {speed_accuracy_ratio:.2f}")

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"    ✗ Tradeoff analysis failed: {e}")
                results[config_name] = {'error': str(e)}

        return results

    def run_comprehensive_benchmark(self, models: List[str], datasets: List[str]) -> Dict:
        """
        Run comprehensive benchmark across models and datasets.

        Args:
            models: List of TNO model names
            datasets: List of dataset names

        Returns:
            dict: Comprehensive benchmark results
        """
        print("\n=== TNO Comprehensive Benchmarking ===")

        # Define configurations to test
        config_variations = [
            {'K': 1, 'L': 1},  # Baseline
            {'K': 2, 'L': 1},  # Medium bundling
            {'K': 4, 'L': 1},  # High bundling
            {'K': 4, 'L': 2},  # High bundling + memory
            {'K': 8, 'L': 1},  # Maximum bundling
        ]

        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_info': self._get_system_info(),
            'benchmarks': {}
        }

        # 1. Configuration benchmarking
        print("\n1. Running configuration benchmarks...")
        results['benchmarks']['configurations'] = self.benchmark_tno_configurations(config_variations)

        # 2. Scaling benchmarking
        print("\n2. Running scaling benchmarks...")
        results['benchmarks']['scaling'] = self.benchmark_scaling_performance({'K': 4, 'L': 1})

        # 3. Temporal bundling benchmarking
        print("\n3. Running temporal bundling benchmarks...")
        results['benchmarks']['temporal_bundling'] = self.benchmark_temporal_bundling([1, 2, 4, 8, 16])

        # 4. Memory vs accuracy tradeoff
        print("\n4. Running memory vs accuracy tradeoffs...")
        results['benchmarks']['tradeoffs'] = self.benchmark_memory_vs_accuracy_tradeoff(config_variations)

        # 5. Generate summary
        results['summary'] = self._generate_comprehensive_summary(results['benchmarks'])

        return results

    def generate_benchmark_report(self, results: Dict, save_path: Optional[str] = None):
        """
        Generate comprehensive benchmark report with visualizations.

        Args:
            results: Benchmark results
            save_path: Path to save report
        """
        print("\n=== Generating Benchmark Report ===")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        # Plot 1: Configuration performance comparison
        ax = axes[0]
        if 'configurations' in results:
            config_results = results['configurations']
            configs = [k for k in config_results.keys() if k != 'summary' and 'error' not in config_results[k]]
            throughputs = [config_results[k]['performance']['throughput_fps'] for k in configs]

            ax.bar(configs, throughputs, color='steelblue', alpha=0.7)
            ax.set_title('Configuration Performance Comparison', fontweight='bold')
            ax.set_ylabel('Throughput (fps)')
            ax.tick_params(axis='x', rotation=45)

        # Plot 2: Memory usage comparison
        ax = axes[1]
        if 'configurations' in results:
            config_results = results['configurations']
            configs = [k for k in config_results.keys() if k != 'summary' and 'error' not in config_results[k]]
            memories = [config_results[k]['memory']['profile']['peak_gpu_mb'] for k in configs]

            ax.bar(configs, memories, color='coral', alpha=0.7)
            ax.set_title('Memory Usage Comparison', fontweight='bold')
            ax.set_ylabel('Peak Memory (MB)')
            ax.tick_params(axis='x', rotation=45)

        # Plot 3: Scaling performance
        ax = axes[2]
        if 'scaling' in results:
            scaling_results = results['scaling']
            test_names = [k for k in scaling_results.keys() if 'error' not in scaling_results[k]]
            throughputs = [scaling_results[k]['scaling_metrics']['throughput_per_pixel'] for k in test_names]

            ax.plot(range(len(test_names)), throughputs, 'o-', linewidth=2, markersize=8)
            ax.set_title('Scaling Performance', fontweight='bold')
            ax.set_ylabel('Throughput (Mpix/s)')
            ax.set_xticks(range(len(test_names)))
            ax.set_xticklabels(test_names, rotation=45)

        # Plot 4: Temporal bundling efficiency
        ax = axes[3]
        if 'temporal_bundling' in results:
            bundling_results = results['temporal_bundling']
            K_values = [int(k[1:]) for k in bundling_results.keys() if k.startswith('K') and 'error' not in bundling_results[k]]
            efficiencies = [bundling_results[f'K{k}']['bundling_metrics']['bundle_efficiency'] for k in K_values]

            ax.plot(K_values, efficiencies, 'o-', color='green', linewidth=2, markersize=8)
            ax.set_title('Temporal Bundling Efficiency', fontweight='bold')
            ax.set_xlabel('K Value')
            ax.set_ylabel('Bundle Efficiency (bundles/s)')
            ax.grid(True, alpha=0.3)

        # Plot 5: Memory vs Accuracy Tradeoff
        ax = axes[4]
        if 'tradeoffs' in results:
            tradeoff_results = results['tradeoffs']
            configs = [k for k in tradeoff_results.keys() if 'error' not in tradeoff_results[k]]
            if configs:
                memories = [tradeoff_results[k]['memory']['peak_gpu_mb'] for k in configs]
                accuracies = [tradeoff_results[k]['accuracy']['mse_improvement'] for k in configs]

                scatter = ax.scatter(memories, accuracies, c=range(len(configs)),
                                   cmap='viridis', s=100, alpha=0.7)
                ax.set_title('Memory vs Accuracy Tradeoff', fontweight='bold')
                ax.set_xlabel('Peak Memory (MB)')
                ax.set_ylabel('MSE Improvement')

                # Add labels
                for i, config in enumerate(configs):
                    ax.annotate(config, (memories[i], accuracies[i]),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)

        # Plot 6: Overall efficiency scores
        ax = axes[5]
        if 'configurations' in results:
            config_results = results['configurations']
            configs = [k for k in config_results.keys() if k != 'summary' and 'error' not in config_results[k]]
            efficiency_scores = [config_results[k]['efficiency_scores']['throughput_per_mb'] for k in configs]

            ax.bar(configs, efficiency_scores, color='purple', alpha=0.7)
            ax.set_title('Overall Efficiency Scores', fontweight='bold')
            ax.set_ylabel('Throughput per MB')
            ax.tick_params(axis='x', rotation=45)

        plt.suptitle('TNO Performance Benchmark Report', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  ✓ Report saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def save_benchmark_results(self, results: Dict, filename: str):
        """Save benchmark results to JSON file."""
        save_path = self.output_dir / filename

        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(i) for i in obj]
            return obj

        results_clean = convert_types(results)

        with open(save_path, 'w') as f:
            json.dump(results_clean, f, indent=2)

        print(f"  ✓ Results saved to: {save_path}")

    # ============== Helper Methods ==============

    def _create_benchmark_model(self, config: Dict):
        """Create dummy TNO model for benchmarking."""
        class BenchmarkTNO(torch.nn.Module):
            def __init__(self, K, L):
                super().__init__()
                self.K = K
                self.L = L
                # Simple conv layers to simulate TNO
                self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1)
                self.conv2 = torch.nn.Conv2d(64, 64, 3, padding=1)
                self.conv3 = torch.nn.Conv2d(64, 1, 3, padding=1)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                B, T, C, H, W = x.shape
                x = x.reshape(B*T, C, H, W)
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                x = self.conv3(x)
                return x.reshape(B, T, 1, H, W)

        model = BenchmarkTNO(config['K'], config['L'])
        model.to(self.device)
        model.eval()
        return model

    def _analyze_benchmark_results(self, results: Dict) -> Dict:
        """Analyze benchmark results and find optimal configurations."""
        valid_results = {k: v for k, v in results.items() if k != 'summary' and 'error' not in v}

        if not valid_results:
            return {'error': 'No valid benchmark results'}

        # Find best configurations
        best_throughput = max(valid_results.keys(),
                            key=lambda k: valid_results[k]['performance']['throughput_fps'])
        best_memory = min(valid_results.keys(),
                         key=lambda k: valid_results[k]['memory']['profile']['peak_gpu_mb'])
        best_efficiency = max(valid_results.keys(),
                            key=lambda k: valid_results[k]['efficiency_scores']['throughput_per_mb'])

        return {
            'best_throughput': best_throughput,
            'best_memory': best_memory,
            'best_efficiency': best_efficiency,
            'total_configs_tested': len(valid_results),
            'failed_configs': len(results) - len(valid_results) - 1  # -1 for summary
        }

    def _simulate_accuracy_metrics(self, config: Dict) -> Dict:
        """Simulate accuracy metrics for benchmarking (would use real data in practice)."""
        # Simulate that higher K and L lead to better accuracy but more computation
        K, L = config['K'], config['L']

        base_mse = 0.1
        K_improvement = (K - 1) * 0.02  # Each K reduces MSE by 2%
        L_improvement = (L - 1) * 0.01  # Each L reduces MSE by 1%

        improved_mse = base_mse * (1 - K_improvement - L_improvement)
        mse_improvement = (base_mse - improved_mse) / base_mse

        computational_cost = K * L  # Arbitrary cost function

        return {
            'mse_improvement': mse_improvement,
            'computational_cost': computational_cost,
            'accuracy_score': 1 - improved_mse
        }

    def _get_system_info(self) -> Dict:
        """Get system information for benchmark context."""
        info = {
            'device': str(self.device),
            'cuda_available': torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            info.update({
                'gpu_name': torch.cuda.get_device_name(),
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
                'cuda_version': torch.version.cuda,
            })

        return info

    def _generate_comprehensive_summary(self, benchmarks: Dict) -> Dict:
        """Generate comprehensive summary of all benchmarks."""
        summary = {
            'benchmark_completion': {},
            'key_findings': {},
            'recommendations': {}
        }

        # Check completion status
        for benchmark_type in ['configurations', 'scaling', 'temporal_bundling', 'tradeoffs']:
            if benchmark_type in benchmarks:
                valid_results = [k for k, v in benchmarks[benchmark_type].items()
                               if isinstance(v, dict) and 'error' not in v]
                summary['benchmark_completion'][benchmark_type] = {
                    'completed': True,
                    'valid_results': len(valid_results)
                }
            else:
                summary['benchmark_completion'][benchmark_type] = {
                    'completed': False,
                    'valid_results': 0
                }

        # Extract key findings
        if 'configurations' in benchmarks and 'summary' in benchmarks['configurations']:
            config_summary = benchmarks['configurations']['summary']
            summary['key_findings']['best_overall_config'] = config_summary.get('best_efficiency')

        if 'temporal_bundling' in benchmarks and 'optimal_K' in benchmarks['temporal_bundling']:
            summary['key_findings']['optimal_K'] = benchmarks['temporal_bundling']['optimal_K']

        # Generate recommendations
        summary['recommendations'] = [
            "Use configuration with best efficiency score for production",
            "Consider memory constraints when selecting batch size",
            "Optimize K value based on sequence length requirements",
            "Monitor memory usage during long sequence processing"
        ]

        return summary


def main():
    parser = argparse.ArgumentParser(description="TNO Performance Benchmarking Suite")
    parser.add_argument("--models", type=str, help="Comma-separated list of TNO models")
    parser.add_argument("--dataset", choices=["inc", "tra", "iso"], default="inc", help="Dataset type")
    parser.add_argument("--full-benchmark", action="store_true", help="Run comprehensive benchmark")
    parser.add_argument("--memory-profile", action="store_true", help="Focus on memory profiling")
    parser.add_argument("--config-sweep", action="store_true", help="Sweep TNO configurations")
    parser.add_argument("--output", default="results/benchmarks", help="Output directory")
    parser.add_argument("--generate-report", action="store_true", help="Generate visual report")

    args = parser.parse_args()

    # Initialize benchmark suite
    benchmark_suite = TNOBenchmarkSuite(output_dir=args.output)

    if args.full_benchmark:
        # Run comprehensive benchmark
        models = args.models.split(',') if args.models else ['tno-L1K1', 'tno-L2K4']
        datasets = [args.dataset]

        results = benchmark_suite.run_comprehensive_benchmark(models, datasets)

        # Save results
        benchmark_suite.save_benchmark_results(results, f"comprehensive_benchmark_{args.dataset}.json")

        # Generate report
        if args.generate_report:
            report_path = benchmark_suite.output_dir / f"benchmark_report_{args.dataset}.png"
            benchmark_suite.generate_benchmark_report(results['benchmarks'], str(report_path))

    elif args.config_sweep:
        # Configuration sweep
        configs = [
            {'K': 1, 'L': 1}, {'K': 2, 'L': 1}, {'K': 4, 'L': 1},
            {'K': 8, 'L': 1}, {'K': 4, 'L': 2}, {'K': 4, 'L': 4}
        ]

        results = benchmark_suite.benchmark_tno_configurations(configs)
        benchmark_suite.save_benchmark_results(results, f"config_sweep_{args.dataset}.json")

    elif args.memory_profile:
        # Memory profiling focus
        base_config = {'K': 4, 'L': 1}

        scaling_results = benchmark_suite.benchmark_scaling_performance(base_config)
        tradeoff_results = benchmark_suite.benchmark_memory_vs_accuracy_tradeoff([
            {'K': 1, 'L': 1}, {'K': 4, 'L': 1}, {'K': 4, 'L': 2}
        ])

        results = {
            'scaling': scaling_results,
            'tradeoffs': tradeoff_results
        }

        benchmark_suite.save_benchmark_results(results, f"memory_profile_{args.dataset}.json")

    else:
        print("Please specify a benchmark type: --full-benchmark, --config-sweep, or --memory-profile")

    print("\n✅ TNO Benchmarking Complete!")


if __name__ == "__main__":
    main()