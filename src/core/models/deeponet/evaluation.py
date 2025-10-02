"""
DeepONet Evaluation and Inference Utilities

Provides comprehensive evaluation metrics, inference utilities, and model
assessment tools specifically designed for operator learning with DeepONet.
"""

from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import logging
from pathlib import Path
import time
from dataclasses import dataclass

from .deeponet_base import DeepONet
from .data_utils import OperatorDataLoader, DeepONetDataConfig


@dataclass
class EvaluationMetrics:
    """Container for DeepONet evaluation metrics."""
    l2_error: float
    relative_l2_error: float
    max_error: float
    rmse: float
    mae: float
    r2_score: float
    spectral_error: Optional[float] = None
    physics_error: Optional[float] = None


class DeepONetEvaluator:
    """
    Comprehensive evaluator for DeepONet models.

    Provides standardized evaluation metrics and visualization tools
    for assessing operator learning performance.
    """

    def __init__(self, model: DeepONet, device: str = 'cuda'):
        """
        Initialize DeepONet evaluator.

        Args:
            model: Trained DeepONet model
            device: Device for evaluation
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()

        self.evaluation_cache = {}
        self.last_evaluation = None

        logging.info(f"DeepONetEvaluator initialized for {type(model).__name__}")

    def evaluate_dataset(self,
                        dataloader: OperatorDataLoader,
                        return_predictions: bool = False,
                        compute_spectral: bool = True,
                        compute_physics: bool = True) -> Dict[str, Any]:
        """
        Evaluate model on a complete dataset.

        Args:
            dataloader: DataLoader with evaluation data
            return_predictions: Whether to return all predictions
            compute_spectral: Whether to compute spectral metrics
            compute_physics: Whether to compute physics-based metrics

        Returns:
            results: Dictionary with evaluation metrics and optional predictions
        """
        self.model.eval()

        all_predictions = []
        all_targets = []
        all_query_coords = []
        all_sensor_values = []

        batch_metrics = []
        total_time = 0.0

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                batch = self._move_batch_to_device(batch)

                # Time inference
                start_time = time.time()
                predictions = self._forward_pass(batch)
                inference_time = time.time() - start_time
                total_time += inference_time

                targets = batch['target_values']

                # Compute batch-level metrics
                batch_metric = self._compute_metrics(
                    predictions, targets,
                    compute_spectral=compute_spectral,
                    compute_physics=compute_physics,
                    query_coords=batch.get('query_coords'),
                    sensor_values=batch.get('input_sensors')
                )
                batch_metrics.append(batch_metric)

                if return_predictions:
                    all_predictions.append(predictions.cpu())
                    all_targets.append(targets.cpu())
                    all_query_coords.append(batch['query_coords'].cpu())
                    all_sensor_values.append(batch['input_sensors'].cpu())

        # Aggregate metrics across batches
        aggregated_metrics = self._aggregate_metrics(batch_metrics)

        # Compute dataset-level statistics
        results = {
            'metrics': aggregated_metrics,
            'num_samples': len(dataloader.dataset),
            'num_batches': len(dataloader),
            'total_inference_time': total_time,
            'avg_inference_time_per_batch': total_time / len(dataloader),
            'avg_inference_time_per_sample': total_time / len(dataloader.dataset)
        }

        if return_predictions:
            results.update({
                'predictions': torch.cat(all_predictions, dim=0),
                'targets': torch.cat(all_targets, dim=0),
                'query_coords': torch.cat(all_query_coords, dim=0),
                'sensor_values': torch.cat(all_sensor_values, dim=0)
            })

        self.last_evaluation = results
        return results

    def evaluate_operator_approximation(self,
                                      input_function: torch.Tensor,
                                      target_operator: Callable,
                                      query_points: torch.Tensor,
                                      sensor_locations: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Evaluate operator approximation against analytical operator.

        Args:
            input_function: Input function [T, C, H, W]
            target_operator: Analytical operator function
            query_points: Query coordinates [n_query, 2]
            sensor_locations: Optional sensor locations [n_sensors, 2]

        Returns:
            metrics: Dictionary of approximation quality metrics
        """
        self.model.eval()

        with torch.no_grad():
            # Add batch dimension
            input_batch = input_function.unsqueeze(0)  # [1, T, C, H, W]
            query_batch = query_points.unsqueeze(0)    # [1, n_query, 2]

            # Model prediction
            pred_values = self.model(input_batch, query_batch)  # [1, T, n_query, C]
            pred_values = pred_values.squeeze(0)  # [T, n_query, C]

            # Analytical solution
            analytical_values = []
            for t in range(input_function.shape[0]):
                for c in range(input_function.shape[1]):
                    # Apply analytical operator to each timestep and channel
                    func_t_c = input_function[t, c]  # [H, W]
                    analytical_t_c = target_operator(func_t_c, query_points)
                    analytical_values.append(analytical_t_c)

            analytical_values = torch.stack(analytical_values)
            analytical_values = analytical_values.view(
                input_function.shape[0], input_function.shape[1], -1
            ).transpose(1, 2)  # [T, n_query, C]

        # Compute approximation metrics
        metrics = self._compute_metrics(
            pred_values.unsqueeze(0),
            analytical_values.unsqueeze(0),
            compute_spectral=True,
            compute_physics=False
        )

        return {
            'l2_error': metrics.l2_error,
            'relative_l2_error': metrics.relative_l2_error,
            'max_error': metrics.max_error,
            'rmse': metrics.rmse,
            'r2_score': metrics.r2_score,
            'spectral_error': metrics.spectral_error
        }

    def compute_convergence_analysis(self,
                                   input_functions: torch.Tensor,
                                   target_functions: torch.Tensor,
                                   query_counts: List[int]) -> Dict[int, EvaluationMetrics]:
        """
        Analyze convergence behavior with varying numbers of query points.

        Args:
            input_functions: Input functions [B, T, C, H, W]
            target_functions: Target functions [B, T, C, H, W]
            query_counts: List of query point counts to test

        Returns:
            convergence_data: Dictionary mapping query counts to metrics
        """
        self.model.eval()
        convergence_results = {}

        H, W = input_functions.shape[-2:]

        with torch.no_grad():
            for n_query in query_counts:
                logging.info(f"Computing convergence for {n_query} query points")

                # Generate random query points
                query_coords = torch.rand(input_functions.shape[0], n_query, 2)
                query_coords = query_coords.to(self.device)

                # Model predictions
                predictions = self.model(input_functions, query_coords)

                # Interpolate target functions at query points
                targets = self._interpolate_targets_at_queries(
                    target_functions, query_coords
                )

                # Compute metrics
                metrics = self._compute_metrics(
                    predictions, targets,
                    compute_spectral=False,
                    compute_physics=False
                )

                convergence_results[n_query] = metrics

        return convergence_results

    def benchmark_inference_speed(self,
                                 input_shapes: List[Tuple[int, int, int, int, int]],
                                 query_counts: List[int],
                                 num_runs: int = 10) -> Dict[str, Any]:
        """
        Benchmark inference speed for different input sizes and query counts.

        Args:
            input_shapes: List of input shapes [B, T, C, H, W]
            query_counts: List of query point counts
            num_runs: Number of runs for averaging

        Returns:
            benchmark_results: Dictionary with timing results
        """
        self.model.eval()
        results = {}

        for shape in input_shapes:
            B, T, C, H, W = shape
            shape_key = f"B{B}_T{T}_C{C}_H{H}_W{W}"
            results[shape_key] = {}

            # Generate dummy input
            dummy_input = torch.randn(B, T, C, H, W, device=self.device)

            for n_query in query_counts:
                query_coords = torch.rand(B, n_query, 2, device=self.device)

                # Warmup
                with torch.no_grad():
                    for _ in range(3):
                        _ = self.model(dummy_input, query_coords)

                # Benchmark
                times = []
                with torch.no_grad():
                    for _ in range(num_runs):
                        torch.cuda.synchronize() if self.device == 'cuda' else None
                        start_time = time.time()
                        _ = self.model(dummy_input, query_coords)
                        torch.cuda.synchronize() if self.device == 'cuda' else None
                        end_time = time.time()
                        times.append(end_time - start_time)

                results[shape_key][n_query] = {
                    'mean_time': np.mean(times),
                    'std_time': np.std(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times)
                }

                logging.info(f"Shape {shape_key}, Queries {n_query}: "
                           f"{np.mean(times)*1000:.2f}±{np.std(times)*1000:.2f}ms")

        return results

    def _forward_pass(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Execute forward pass through model."""
        # For now, use the manual forward implementation
        # In a complete implementation, this would use the model's forward method directly
        input_sensors = batch['input_sensors']
        query_coords = batch['query_coords']

        B, T, n_sensors, C = input_sensors.shape
        n_query = query_coords.shape[1]

        outputs = []
        for t in range(T):
            u_t = input_sensors[:, t]  # [B, n_sensors, C]

            # Branch network processing
            branch_features = []
            for c in range(C):
                u_c = u_t[:, :, c]  # [B, n_sensors]
                if self.model.input_norm is not None:
                    u_c = self.model.input_norm(u_c)
                branch_out = self.model.branch_network(u_c)  # [B, latent_dim]
                branch_features.append(branch_out)

            branch_output = torch.stack(branch_features, dim=1).mean(dim=1)  # [B, latent_dim]

            # Trunk network processing
            coords_flat = query_coords.view(B * n_query, 2)
            if self.model.coord_norm is not None:
                coords_flat = self.model.coord_norm(coords_flat)
            trunk_output = self.model.trunk_network(coords_flat)
            trunk_output = trunk_output.view(B, n_query, self.model.config.latent_dim)

            # Dot product
            branch_expanded = branch_output.unsqueeze(1)
            dot_product = (branch_expanded * trunk_output).sum(dim=2)

            if self.model.bias is not None:
                dot_product = dot_product + self.model.bias

            outputs.append(dot_product)

        output = torch.stack(outputs, dim=1).unsqueeze(3)  # [B, T, n_query, 1]

        if C > 1:
            output = output.expand(-1, -1, -1, C)

        return output

    def _compute_metrics(self,
                        predictions: torch.Tensor,
                        targets: torch.Tensor,
                        compute_spectral: bool = True,
                        compute_physics: bool = True,
                        query_coords: Optional[torch.Tensor] = None,
                        sensor_values: Optional[torch.Tensor] = None) -> EvaluationMetrics:
        """Compute evaluation metrics between predictions and targets."""
        # Flatten for metric computation
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()

        # L2 error
        l2_error = torch.norm(pred_flat - target_flat).item()

        # Relative L2 error
        target_norm = torch.norm(target_flat)
        relative_l2_error = l2_error / (target_norm.item() + 1e-8)

        # Max error
        max_error = torch.max(torch.abs(pred_flat - target_flat)).item()

        # RMSE
        mse = F.mse_loss(pred_flat, target_flat)
        rmse = torch.sqrt(mse).item()

        # MAE
        mae = F.l1_loss(pred_flat, target_flat).item()

        # R² score
        target_mean = torch.mean(target_flat)
        ss_tot = torch.sum((target_flat - target_mean) ** 2)
        ss_res = torch.sum((target_flat - pred_flat) ** 2)
        r2_score = (1 - ss_res / (ss_tot + 1e-8)).item()

        # Optional spectral error
        spectral_error = None
        if compute_spectral and predictions.shape[2] > 1:  # Need multiple query points
            try:
                pred_fft = torch.fft.fft(predictions.flatten())
                target_fft = torch.fft.fft(targets.flatten())
                spectral_error = torch.norm(pred_fft - target_fft).item()
            except Exception:
                spectral_error = None

        # Optional physics error (placeholder)
        physics_error = None
        if compute_physics and query_coords is not None:
            # This would require physics-specific computations
            physics_error = 0.0

        return EvaluationMetrics(
            l2_error=l2_error,
            relative_l2_error=relative_l2_error,
            max_error=max_error,
            rmse=rmse,
            mae=mae,
            r2_score=r2_score,
            spectral_error=spectral_error,
            physics_error=physics_error
        )

    def _aggregate_metrics(self, batch_metrics: List[EvaluationMetrics]) -> EvaluationMetrics:
        """Aggregate metrics across batches."""
        if not batch_metrics:
            raise ValueError("No batch metrics to aggregate")

        # Simple averaging for now
        metrics_dict = {}
        for field in EvaluationMetrics.__dataclass_fields__:
            values = [getattr(m, field) for m in batch_metrics if getattr(m, field) is not None]
            if values:
                metrics_dict[field] = np.mean(values)
            else:
                metrics_dict[field] = None

        return EvaluationMetrics(**metrics_dict)

    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch tensors to evaluation device."""
        moved_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(self.device)
            else:
                moved_batch[key] = value
        return moved_batch

    def _interpolate_targets_at_queries(self,
                                      target_functions: torch.Tensor,
                                      query_coords: torch.Tensor) -> torch.Tensor:
        """Interpolate target functions at query coordinates."""
        B, T, C, H, W = target_functions.shape
        n_query = query_coords.shape[1]

        # Convert normalized coordinates to grid coordinates for interpolation
        grid_coords = query_coords * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        grid = grid_coords.view(B, n_query, 1, 2)

        interpolated_values = []
        for t in range(T):
            func_t = target_functions[:, t]  # [B, C, H, W]
            interp_t = F.grid_sample(func_t, grid,
                                   mode='bilinear',
                                   padding_mode='border',
                                   align_corners=True)
            interp_t = interp_t.squeeze(-1).transpose(1, 2)  # [B, n_query, C]
            interpolated_values.append(interp_t)

        return torch.stack(interpolated_values, dim=1)  # [B, T, n_query, C]

    def save_evaluation_report(self, filepath: str, include_plots: bool = True):
        """
        Save comprehensive evaluation report.

        Args:
            filepath: Path to save report
            include_plots: Whether to include visualization plots
        """
        if self.last_evaluation is None:
            raise ValueError("No evaluation results to save. Run evaluation first.")

        report_data = {
            'model_info': self.model.get_model_info(),
            'evaluation_results': self.last_evaluation,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        # Save as torch file for easy loading
        torch.save(report_data, filepath)
        logging.info(f"Evaluation report saved to {filepath}")

    def load_evaluation_report(self, filepath: str) -> Dict[str, Any]:
        """Load evaluation report from file."""
        report_data = torch.load(filepath, map_location='cpu')
        self.last_evaluation = report_data['evaluation_results']
        return report_data


class DeepONetInference:
    """
    High-level inference interface for trained DeepONet models.

    Provides convenient methods for operator evaluation and prediction.
    """

    def __init__(self, model: DeepONet, device: str = 'cuda'):
        """
        Initialize inference interface.

        Args:
            model: Trained DeepONet model
            device: Device for inference
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()

        logging.info("DeepONetInference interface initialized")

    def predict_operator(self,
                        input_functions: torch.Tensor,
                        query_points: torch.Tensor,
                        return_confidence: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict operator output at specified query points.

        Args:
            input_functions: Input functions [B, T, C, H, W]
            query_points: Query coordinates [B, n_query, 2]
            return_confidence: Whether to return prediction confidence

        Returns:
            predictions: Operator outputs [B, T, n_query, C]
            confidence: Optional confidence scores [B, T, n_query, C]
        """
        self.model.eval()
        input_functions = input_functions.to(self.device)
        query_points = query_points.to(self.device)

        with torch.no_grad():
            predictions = self.model(input_functions, query_points)

        if return_confidence:
            # Simple confidence based on prediction magnitude
            # In practice, this could use ensemble methods or Bayesian approaches
            confidence = torch.ones_like(predictions) * 0.95  # Placeholder
            return predictions.cpu(), confidence.cpu()

        return predictions.cpu()

    def predict_on_grid(self,
                       input_functions: torch.Tensor,
                       grid_resolution: Tuple[int, int] = (64, 64)) -> torch.Tensor:
        """
        Predict operator output on regular grid.

        Args:
            input_functions: Input functions [B, T, C, H, W]
            grid_resolution: Output grid resolution (H, W)

        Returns:
            grid_predictions: Predictions on grid [B, T, C, H, W]
        """
        H, W = grid_resolution
        B = input_functions.shape[0]

        # Generate grid query points
        h_coords = torch.linspace(0, 1, H)
        w_coords = torch.linspace(0, 1, W)
        hh, ww = torch.meshgrid(h_coords, w_coords, indexing='ij')
        query_grid = torch.stack([hh.flatten(), ww.flatten()], dim=1)  # [H*W, 2]
        query_grid = query_grid.unsqueeze(0).expand(B, -1, -1)  # [B, H*W, 2]

        # Predict
        predictions = self.predict_operator(input_functions, query_grid)  # [B, T, H*W, C]

        # Reshape to grid
        B, T, _, C = predictions.shape
        grid_predictions = predictions.view(B, T, H, W, C)
        grid_predictions = grid_predictions.permute(0, 1, 4, 2, 3)  # [B, T, C, H, W]

        return grid_predictions

    def batch_predict(self,
                     input_functions_list: List[torch.Tensor],
                     query_points_list: List[torch.Tensor],
                     batch_size: int = 8) -> List[torch.Tensor]:
        """
        Predict on list of inputs with batching.

        Args:
            input_functions_list: List of input function tensors
            query_points_list: List of query point tensors
            batch_size: Batch size for processing

        Returns:
            predictions_list: List of prediction tensors
        """
        predictions_list = []

        for i in range(0, len(input_functions_list), batch_size):
            batch_inputs = input_functions_list[i:i+batch_size]
            batch_queries = query_points_list[i:i+batch_size]

            # Stack into batches
            batch_input = torch.stack(batch_inputs)
            batch_query = torch.stack(batch_queries)

            # Predict
            batch_pred = self.predict_operator(batch_input, batch_query)

            # Split back into individual predictions
            for j in range(batch_pred.shape[0]):
                predictions_list.append(batch_pred[j])

        return predictions_list