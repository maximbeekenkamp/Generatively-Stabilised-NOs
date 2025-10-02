"""
Hyperparameter Optimization for DeepONet Models

Provides automated hyperparameter tuning using various optimization strategies
including grid search, random search, Bayesian optimization, and population-based methods.
"""

from typing import Dict, List, Tuple, Any, Optional, Callable, Union
import json
import logging
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from torch.utils.data import random_split

# Optional dependencies for advanced optimization
try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available. Bayesian optimization disabled.")

try:
    from sklearn.model_selection import ParameterGrid
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available. Grid search may be limited.")

from .model_factory import DeepONetFactory
from .data_utils import create_operator_dataloaders, DeepONetDataConfig
from ...training.training_deeponet import DeepONetTrainingConfig, create_deeponet_training_setup
from .evaluation import DeepONetEvaluator


@dataclass
class HyperparameterSpace:
    """Definition of hyperparameter search space."""

    # Model architecture parameters
    latent_dim: List[int] = None
    branch_layers: List[List[int]] = None
    trunk_layers: List[List[int]] = None
    activation: List[str] = None
    dropout: List[float] = None

    # Sensor configuration
    n_sensors: List[int] = None
    sensor_strategy: List[str] = None

    # Training parameters
    learning_rate: List[float] = None
    batch_size: List[int] = None
    optimizer_type: List[str] = None
    scheduler_type: List[str] = None

    # Loss function parameters
    operator_loss_type: List[str] = None
    physics_weight: List[float] = None
    gradient_weight: List[float] = None

    # Variant-specific parameters
    fourier_modes: List[int] = None
    physics_type: List[str] = None
    n_scales: List[int] = None

    def __post_init__(self):
        """Set default search spaces if not provided."""
        if self.latent_dim is None:
            self.latent_dim = [128, 256, 512]
        if self.branch_layers is None:
            self.branch_layers = [[64, 128], [128, 256], [256, 512]]
        if self.trunk_layers is None:
            self.trunk_layers = [[32, 64], [64, 128], [128, 256]]
        if self.activation is None:
            self.activation = ['gelu', 'relu', 'silu']
        if self.dropout is None:
            self.dropout = [0.0, 0.1, 0.2]
        if self.n_sensors is None:
            self.n_sensors = [50, 100, 200]
        if self.sensor_strategy is None:
            self.sensor_strategy = ['uniform', 'random', 'adaptive']
        if self.learning_rate is None:
            self.learning_rate = [1e-4, 1e-3, 1e-2]
        if self.batch_size is None:
            self.batch_size = [16, 32, 64]
        if self.optimizer_type is None:
            self.optimizer_type = ['adam', 'adamw']
        if self.scheduler_type is None:
            self.scheduler_type = ['step', 'exponential', 'cosine']
        if self.operator_loss_type is None:
            self.operator_loss_type = ['l2', 'relative_l2']
        if self.physics_weight is None:
            self.physics_weight = [0.0, 0.1, 0.5]
        if self.gradient_weight is None:
            self.gradient_weight = [0.0, 0.01, 0.1]
        if self.fourier_modes is None:
            self.fourier_modes = [16, 32, 64]
        if self.physics_type is None:
            self.physics_type = ['general', 'fluid', 'heat']
        if self.n_scales is None:
            self.n_scales = [2, 3, 4]


@dataclass
class OptimizationResult:
    """Result from hyperparameter optimization."""
    best_params: Dict[str, Any]
    best_score: float
    best_model_path: Optional[str]
    optimization_history: List[Dict[str, Any]]
    total_time: float
    n_trials: int


class DeepONetHyperparameterOptimizer:
    """
    Hyperparameter optimizer for DeepONet models.

    Supports multiple optimization strategies:
    - Grid search
    - Random search
    - Bayesian optimization (via Optuna)
    - Population-based training
    """

    def __init__(self,
                 train_data: Tuple[torch.Tensor, torch.Tensor],
                 val_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                 search_space: Optional[HyperparameterSpace] = None,
                 objective_metric: str = 'relative_l2_error',
                 direction: str = 'minimize',
                 device: str = 'cuda',
                 save_dir: str = './deeponet_hpo_results'):
        """
        Initialize hyperparameter optimizer.

        Args:
            train_data: Training data tuple (inputs, outputs)
            val_data: Validation data tuple (inputs, outputs)
            search_space: Hyperparameter search space definition
            objective_metric: Metric to optimize ('relative_l2_error', 'rmse', etc.)
            direction: Optimization direction ('minimize' or 'maximize')
            device: Device for training
            save_dir: Directory to save optimization results
        """
        self.train_data = train_data
        self.val_data = val_data
        self.search_space = search_space or HyperparameterSpace()
        self.objective_metric = objective_metric
        self.direction = direction
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Split validation data if not provided
        if self.val_data is None:
            train_input, train_output = train_data
            dataset_size = len(train_input)
            val_size = int(0.2 * dataset_size)
            train_size = dataset_size - val_size

            # Create indices for splitting
            indices = torch.randperm(dataset_size)
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

            self.train_data = (train_input[train_indices], train_output[train_indices])
            self.val_data = (train_input[val_indices], train_output[val_indices])

            logging.info(f"Split data: {train_size} train, {val_size} validation samples")

        self.optimization_history = []
        self.best_result = None

        logging.info(f"DeepONet hyperparameter optimizer initialized")
        logging.info(f"Objective: {direction} {objective_metric}")

    def grid_search(self,
                   max_trials: Optional[int] = None,
                   n_jobs: int = 1,
                   early_stopping_rounds: int = 10) -> OptimizationResult:
        """
        Perform grid search over hyperparameter space.

        Args:
            max_trials: Maximum number of trials (None for full grid)
            n_jobs: Number of parallel jobs
            early_stopping_rounds: Early stopping patience

        Returns:
            optimization_result: Best configuration and results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn required for grid search")

        logging.info("Starting grid search optimization")
        start_time = time.time()

        # Convert search space to parameter grid format
        param_grid = self._create_parameter_grid()
        grid = list(ParameterGrid(param_grid))

        if max_trials is not None:
            # Randomly sample from grid if too large
            if len(grid) > max_trials:
                indices = np.random.choice(len(grid), max_trials, replace=False)
                grid = [grid[i] for i in indices]

        logging.info(f"Grid search will evaluate {len(grid)} configurations")

        best_score = float('inf') if self.direction == 'minimize' else float('-inf')
        best_params = None
        best_model_path = None
        no_improve_count = 0

        # Sequential or parallel execution
        if n_jobs == 1:
            for i, params in enumerate(grid):
                result = self._evaluate_configuration(params, trial_id=i)
                self.optimization_history.append(result)

                score = result['score']
                is_better = (score < best_score if self.direction == 'minimize'
                           else score > best_score)

                if is_better:
                    best_score = score
                    best_params = params
                    best_model_path = result.get('model_path')
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                logging.info(f"Trial {i+1}/{len(grid)}: {self.objective_metric}={score:.6f}")

                # Early stopping
                if no_improve_count >= early_stopping_rounds:
                    logging.info(f"Early stopping after {i+1} trials")
                    break
        else:
            # Parallel execution (simplified)
            logging.warning("Parallel execution not fully implemented, using sequential")
            return self.grid_search(max_trials, n_jobs=1, early_stopping_rounds=early_stopping_rounds)

        total_time = time.time() - start_time

        result = OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_model_path=best_model_path,
            optimization_history=self.optimization_history,
            total_time=total_time,
            n_trials=len(self.optimization_history)
        )

        self.best_result = result
        self._save_optimization_result(result, 'grid_search')

        logging.info(f"Grid search completed in {total_time:.2f}s")
        logging.info(f"Best {self.objective_metric}: {best_score:.6f}")

        return result

    def random_search(self,
                     n_trials: int = 100,
                     early_stopping_rounds: int = 20,
                     seed: Optional[int] = None) -> OptimizationResult:
        """
        Perform random search over hyperparameter space.

        Args:
            n_trials: Number of random trials
            early_stopping_rounds: Early stopping patience
            seed: Random seed for reproducibility

        Returns:
            optimization_result: Best configuration and results
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        logging.info(f"Starting random search with {n_trials} trials")
        start_time = time.time()

        best_score = float('inf') if self.direction == 'minimize' else float('-inf')
        best_params = None
        best_model_path = None
        no_improve_count = 0

        for i in range(n_trials):
            # Sample random configuration
            params = self._sample_random_configuration()

            # Evaluate configuration
            result = self._evaluate_configuration(params, trial_id=i)
            self.optimization_history.append(result)

            score = result['score']
            is_better = (score < best_score if self.direction == 'minimize'
                       else score > best_score)

            if is_better:
                best_score = score
                best_params = params
                best_model_path = result.get('model_path')
                no_improve_count = 0
            else:
                no_improve_count += 1

            logging.info(f"Trial {i+1}/{n_trials}: {self.objective_metric}={score:.6f}")

            # Early stopping
            if no_improve_count >= early_stopping_rounds:
                logging.info(f"Early stopping after {i+1} trials")
                break

        total_time = time.time() - start_time

        result = OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_model_path=best_model_path,
            optimization_history=self.optimization_history,
            total_time=total_time,
            n_trials=len(self.optimization_history)
        )

        self.best_result = result
        self._save_optimization_result(result, 'random_search')

        logging.info(f"Random search completed in {total_time:.2f}s")
        logging.info(f"Best {self.objective_metric}: {best_score:.6f}")

        return result

    def bayesian_optimization(self,
                            n_trials: int = 100,
                            n_startup_trials: int = 10,
                            sampler: str = 'tpe',
                            early_stopping_rounds: int = 30) -> OptimizationResult:
        """
        Perform Bayesian optimization using Optuna.

        Args:
            n_trials: Number of optimization trials
            n_startup_trials: Number of random startup trials
            sampler: Sampling strategy ('tpe', 'random')
            early_stopping_rounds: Early stopping patience

        Returns:
            optimization_result: Best configuration and results
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna required for Bayesian optimization")

        logging.info(f"Starting Bayesian optimization with {n_trials} trials")
        start_time = time.time()

        # Create sampler
        if sampler == 'tpe':
            sampler_obj = TPESampler(n_startup_trials=n_startup_trials)
        elif sampler == 'random':
            sampler_obj = RandomSampler()
        else:
            raise ValueError(f"Unknown sampler: {sampler}")

        # Create study
        direction_str = 'minimize' if self.direction == 'minimize' else 'maximize'
        study = optuna.create_study(direction=direction_str, sampler=sampler_obj)

        # Define objective function for Optuna
        def objective(trial):
            params = self._suggest_parameters_optuna(trial)
            result = self._evaluate_configuration(params, trial_id=trial.number)
            self.optimization_history.append(result)

            # Log progress
            score = result['score']
            logging.info(f"Trial {trial.number+1}: {self.objective_metric}={score:.6f}")

            return score

        # Run optimization with early stopping
        study.optimize(objective, n_trials=n_trials,
                      callbacks=[self._early_stopping_callback(early_stopping_rounds)])

        total_time = time.time() - start_time

        # Get best results
        best_params = study.best_params
        best_score = study.best_value
        best_trial = study.best_trial

        # Find corresponding model path
        best_model_path = None
        if self.optimization_history:
            for result in self.optimization_history:
                if result.get('trial_id') == best_trial.number:
                    best_model_path = result.get('model_path')
                    break

        result = OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_model_path=best_model_path,
            optimization_history=self.optimization_history,
            total_time=total_time,
            n_trials=len(self.optimization_history)
        )

        self.best_result = result
        self._save_optimization_result(result, 'bayesian_optimization')

        logging.info(f"Bayesian optimization completed in {total_time:.2f}s")
        logging.info(f"Best {self.objective_metric}: {best_score:.6f}")

        return result

    def _evaluate_configuration(self, params: Dict[str, Any], trial_id: int = 0) -> Dict[str, Any]:
        """
        Evaluate a single hyperparameter configuration.

        Args:
            params: Hyperparameter configuration
            trial_id: Trial identifier

        Returns:
            result: Evaluation results including score and metadata
        """
        try:
            # Create training configuration
            config = self._params_to_config(params)

            # Setup training
            setup = create_deeponet_training_setup(
                train_data=self.train_data,
                val_data=self.val_data,
                config=config,
                device=self.device,
                log_dir=None  # Disable tensorboard for HPO
            )

            model = setup['model']
            trainer = setup['trainer']

            # Quick training (reduced epochs for HPO)
            hpo_epochs = min(config.num_epochs, 50)  # Limit epochs for speed

            best_val_score = float('inf') if self.direction == 'minimize' else float('-inf')
            patience = 10
            no_improve = 0

            for epoch in range(hpo_epochs):
                trainer.trainingStep(epoch)

                if epoch % 5 == 0:  # Validate every 5 epochs
                    trainer.validationStep(epoch)

                    # Get validation score
                    val_history = trainer.trainHistory.get_latest_validation_loss()
                    if val_history is not None:
                        is_better = (val_history < best_val_score if self.direction == 'minimize'
                                   else val_history > best_val_score)

                        if is_better:
                            best_val_score = val_history
                            no_improve = 0
                        else:
                            no_improve += 1

                        if no_improve >= patience:
                            logging.debug(f"Early stopping at epoch {epoch}")
                            break

            # Final evaluation
            evaluator = DeepONetEvaluator(model, self.device)
            val_results = evaluator.evaluate_dataset(
                setup['dataloaders']['val'],
                return_predictions=False,
                compute_spectral=False,
                compute_physics=False
            )

            # Get objective metric value
            metrics = val_results['metrics']
            if hasattr(metrics, self.objective_metric):
                score = getattr(metrics, self.objective_metric)
            else:
                # Fallback to loss if metric not available
                score = best_val_score

            # Save model if it's the best so far
            model_path = None
            if (self.best_result is None or
                (self.direction == 'minimize' and score < self.best_result.best_score) or
                (self.direction == 'maximize' and score > self.best_result.best_score)):
                model_path = self.save_dir / f"best_model_trial_{trial_id}.pth"
                trainer.save_checkpoint(str(model_path), hpo_epochs)

            result = {
                'trial_id': trial_id,
                'params': params,
                'score': score,
                'metrics': asdict(metrics),
                'model_path': str(model_path) if model_path else None,
                'epochs_trained': hpo_epochs,
                'success': True
            }

            # Clean up
            del model, trainer, setup
            torch.cuda.empty_cache() if self.device == 'cuda' else None

            return result

        except Exception as e:
            logging.error(f"Error evaluating configuration: {str(e)}")
            return {
                'trial_id': trial_id,
                'params': params,
                'score': float('inf') if self.direction == 'minimize' else float('-inf'),
                'error': str(e),
                'success': False
            }

    def _params_to_config(self, params: Dict[str, Any]) -> DeepONetTrainingConfig:
        """Convert hyperparameter dict to training configuration."""
        config = DeepONetTrainingConfig()

        # Model parameters
        if 'latent_dim' in params:
            config.model_config['latent_dim'] = params['latent_dim']
        if 'branch_layers' in params:
            config.model_config['branch_layers'] = params['branch_layers']
        if 'trunk_layers' in params:
            config.model_config['trunk_layers'] = params['trunk_layers']
        if 'activation' in params:
            config.model_config['activation'] = params['activation']
        if 'dropout' in params:
            config.model_config['dropout'] = params['dropout']

        # Sensor parameters
        if 'n_sensors' in params:
            config.model_config['n_sensors'] = params['n_sensors']
            config.data_config.n_sensors = params['n_sensors']
        if 'sensor_strategy' in params:
            config.data_config.sensor_strategy = params['sensor_strategy']

        # Training parameters
        if 'learning_rate' in params:
            config.optimizer_config['lr'] = params['learning_rate']
        if 'batch_size' in params:
            config.batch_size = params['batch_size']
        if 'optimizer_type' in params:
            config.optimizer_type = params['optimizer_type']
        if 'scheduler_type' in params:
            config.scheduler_type = params['scheduler_type']

        # Loss parameters
        if 'operator_loss_type' in params:
            config.loss_config['operator_loss_type'] = params['operator_loss_type']
        if 'physics_weight' in params:
            config.loss_config['physics_weight'] = params['physics_weight']
        if 'gradient_weight' in params:
            config.loss_config['gradient_weight'] = params['gradient_weight']

        # Variant-specific parameters
        if 'fourier_modes' in params:
            config.fourier_config['branch_fourier_modes'] = params['fourier_modes']
            config.fourier_config['trunk_fourier_modes'] = params['fourier_modes']
        if 'physics_type' in params:
            config.physics_config['physics_type'] = params['physics_type']
        if 'n_scales' in params:
            config.multiscale_config['n_scales'] = params['n_scales']

        return config

    def _create_parameter_grid(self) -> Dict[str, List]:
        """Create parameter grid from search space."""
        return {
            'latent_dim': self.search_space.latent_dim,
            'branch_layers': self.search_space.branch_layers,
            'trunk_layers': self.search_space.trunk_layers,
            'activation': self.search_space.activation,
            'dropout': self.search_space.dropout,
            'n_sensors': self.search_space.n_sensors,
            'learning_rate': self.search_space.learning_rate,
            'batch_size': self.search_space.batch_size,
            'operator_loss_type': self.search_space.operator_loss_type
        }

    def _sample_random_configuration(self) -> Dict[str, Any]:
        """Sample random configuration from search space."""
        return {
            'latent_dim': np.random.choice(self.search_space.latent_dim),
            'branch_layers': np.random.choice(self.search_space.branch_layers),
            'trunk_layers': np.random.choice(self.search_space.trunk_layers),
            'activation': np.random.choice(self.search_space.activation),
            'dropout': np.random.choice(self.search_space.dropout),
            'n_sensors': np.random.choice(self.search_space.n_sensors),
            'learning_rate': np.random.choice(self.search_space.learning_rate),
            'batch_size': np.random.choice(self.search_space.batch_size),
            'operator_loss_type': np.random.choice(self.search_space.operator_loss_type),
            'physics_weight': np.random.choice(self.search_space.physics_weight),
            'gradient_weight': np.random.choice(self.search_space.gradient_weight)
        }

    def _suggest_parameters_optuna(self, trial) -> Dict[str, Any]:
        """Suggest parameters using Optuna trial object."""
        params = {}

        params['latent_dim'] = trial.suggest_categorical('latent_dim', self.search_space.latent_dim)
        params['branch_layers'] = trial.suggest_categorical('branch_layers', self.search_space.branch_layers)
        params['trunk_layers'] = trial.suggest_categorical('trunk_layers', self.search_space.trunk_layers)
        params['activation'] = trial.suggest_categorical('activation', self.search_space.activation)
        params['dropout'] = trial.suggest_categorical('dropout', self.search_space.dropout)
        params['n_sensors'] = trial.suggest_categorical('n_sensors', self.search_space.n_sensors)
        params['learning_rate'] = trial.suggest_categorical('learning_rate', self.search_space.learning_rate)
        params['batch_size'] = trial.suggest_categorical('batch_size', self.search_space.batch_size)
        params['operator_loss_type'] = trial.suggest_categorical('operator_loss_type', self.search_space.operator_loss_type)
        params['physics_weight'] = trial.suggest_categorical('physics_weight', self.search_space.physics_weight)
        params['gradient_weight'] = trial.suggest_categorical('gradient_weight', self.search_space.gradient_weight)

        return params

    def _early_stopping_callback(self, patience: int):
        """Create early stopping callback for Optuna."""
        def callback(study, trial):
            if len(study.trials) < patience:
                return

            recent_trials = study.trials[-patience:]
            best_score = study.best_value

            # Check if no improvement in recent trials
            no_improve = all(
                (trial.value >= best_score if self.direction == 'minimize'
                 else trial.value <= best_score)
                for trial in recent_trials
                if trial.value is not None
            )

            if no_improve:
                logging.info(f"Early stopping: no improvement in {patience} trials")
                study.stop()

        return callback

    def _save_optimization_result(self, result: OptimizationResult, method: str):
        """Save optimization results to file."""
        result_file = self.save_dir / f"{method}_results.json"

        # Convert result to JSON-serializable format
        result_dict = asdict(result)

        with open(result_file, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)

        logging.info(f"Optimization results saved to {result_file}")

    def load_best_model(self) -> torch.nn.Module:
        """Load the best model from optimization."""
        if self.best_result is None or self.best_result.best_model_path is None:
            raise ValueError("No best model available. Run optimization first.")

        # This would require implementing model loading from checkpoint
        # For now, return None and log warning
        logging.warning("Model loading not implemented. Use model path: " +
                       str(self.best_result.best_model_path))
        return None


def optimize_deeponet_hyperparameters(
    train_data: Tuple[torch.Tensor, torch.Tensor],
    val_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    method: str = 'random',
    n_trials: int = 50,
    search_space: Optional[HyperparameterSpace] = None,
    objective_metric: str = 'relative_l2_error',
    device: str = 'cuda',
    save_dir: str = './deeponet_hpo'
) -> OptimizationResult:
    """
    High-level function for DeepONet hyperparameter optimization.

    Args:
        train_data: Training data tuple
        val_data: Optional validation data tuple
        method: Optimization method ('grid', 'random', 'bayesian')
        n_trials: Number of optimization trials
        search_space: Hyperparameter search space
        objective_metric: Metric to optimize
        device: Device for training
        save_dir: Directory to save results

    Returns:
        optimization_result: Best configuration and results
    """
    optimizer = DeepONetHyperparameterOptimizer(
        train_data=train_data,
        val_data=val_data,
        search_space=search_space,
        objective_metric=objective_metric,
        device=device,
        save_dir=save_dir
    )

    if method == 'grid':
        return optimizer.grid_search(max_trials=n_trials)
    elif method == 'random':
        return optimizer.random_search(n_trials=n_trials)
    elif method == 'bayesian':
        return optimizer.bayesian_optimization(n_trials=n_trials)
    else:
        raise ValueError(f"Unknown optimization method: {method}")