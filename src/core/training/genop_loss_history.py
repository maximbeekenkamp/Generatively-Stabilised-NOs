#!/usr/bin/env python3
"""
Generative Operator Loss History System

This module extends the existing LossHistory system to support dual metrics tracking
for generative operator training (prior vs corrected evaluation) and enhanced
TensorBoard integration with physics-aware metrics.

Author: Phase 4 Implementation
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import logging
from typing import List, Dict, Tuple, Optional, Any
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from torch.utils.tensorboard import SummaryWriter

from src.core.training.loss_history import LossHistory
from src.analysis.genop_metrics import GenerativeOperatorMetrics


class GenerativeOperatorLossHistory(LossHistory):
    """
    Extended LossHistory for generative operator training with dual metrics tracking.

    Features:
    - Stage-aware logging (prior training vs corrector training)
    - Dual evaluation metrics (prior vs corrected predictions)
    - Physics-aware metrics integration
    - Enhanced TensorBoard visualizations
    - Improvement tracking over time
    """

    def __init__(self, mode: str, modeLong: str, writer: SummaryWriter,
                 dataLoaderLength: int, epoch: int, epochStep: int,
                 printInterval: int = 0, logInterval: int = 1,
                 simFields: List[str] = [], dataset_type: str = "inc",
                 training_stage: int = 1):
        """
        Initialize GenerativeOperatorLossHistory.

        Args:
            mode: Training mode ('train', 'val')
            modeLong: Long mode description
            writer: TensorBoard SummaryWriter
            dataLoaderLength: Length of data loader
            epoch: Current epoch
            epochStep: Epoch step size
            printInterval: Print interval for batch updates
            logInterval: Log interval for batch updates
            simFields: Simulation field names
            dataset_type: Dataset type ("inc", "tra", "iso")
            training_stage: Current training stage (1=prior, 2=corrector)
        """
        super().__init__(mode, modeLong, writer, dataLoaderLength, epoch,
                        epochStep, printInterval, logInterval, simFields)

        self.dataset_type = dataset_type
        self.training_stage = training_stage

        # Initialize generative operator specific tracking
        self.genop_metrics_calculator = GenerativeOperatorMetrics(dataset_type=dataset_type)
        self.stage_history = {"stage_1": [], "stage_2": []}
        self.improvement_history = []

        # Clear to initialize genop-specific fields
        self.clear()

    def clear(self):
        """Clear batch tracking with generative operator extensions."""
        # Call parent clear
        super().clear()

        # Add generative operator specific loss tracking
        if self.training_stage == 1:
            # Stage 1: Prior training
            self.batchLoss.update({
                "lossPrior": [],
                "lossPriorMSE": [],
                "lossPriorL1": [],
                "lossPriorPhysics": []
            })
        elif self.training_stage == 2:
            # Stage 2: Corrector training
            self.batchLoss.update({
                "lossCorrector": [],
                "lossCorrectorMSE": [],
                "lossCorrectorConsistency": [],
                "lossCorrectorPerceptual": [],
                "lossCorrectorPhysics": [],
                "lossImprovement": []
            })

        # Dual evaluation tracking (for validation)
        self.batchEvaluation = {
            "evalPriorMSE": [],
            "evalCorrectedMSE": [],
            "evalImprovementRatio": [],
            "evalPhysicsCompliance": [],
            "evalSpectralImprovement": []
        }

    def updateBatch(self, lossParts: dict, lossSeq: dict, sample: int,
                   timeMin: float, eval_metrics: Optional[dict] = None):
        """
        Update batch tracking with generative operator extensions.

        Args:
            lossParts: Dictionary of loss components
            lossSeq: Dictionary of sequence losses
            sample: Current sample index
            timeMin: Time elapsed in minutes
            eval_metrics: Optional evaluation metrics for current batch
        """
        # Call parent update
        super().updateBatch(lossParts, lossSeq, sample, timeMin)

        step = self.dataLoaderLength * self.epoch + sample

        # Track generative operator specific losses
        for name in self.batchLoss:
            if name in lossParts:
                part = lossParts[name].detach().cpu().item()
                self.batchLoss[name].append(part)

        # Track dual evaluation metrics if provided
        if eval_metrics:
            for name in self.batchEvaluation:
                if name in eval_metrics:
                    value = eval_metrics[name]
                    if isinstance(value, torch.Tensor):
                        value = value.detach().cpu().item()
                    self.batchEvaluation[name].append(value)

        # Enhanced logging for generative operators
        toPrint = self.printInterval > 0 and sample % self.printInterval == self.printInterval - 1
        toLog = self.logInterval > 0 and sample % self.logInterval == self.logInterval - 1

        if toPrint or toLog:
            stage_info = f"Stage {self.training_stage}"
            loss = lossParts.get("lossFull", lossParts.get("lossPrior", lossParts.get("lossCorrector", 0)))
            if isinstance(loss, torch.Tensor):
                loss = loss.detach().cpu().item()

            message = f'[{self.epoch+1:2d}, {sample+1:4d}] ({timeMin:2.2f} min) {stage_info}: {loss:1.4f}'

            # Add improvement info if available
            if eval_metrics and "evalImprovementRatio" in eval_metrics:
                improvement = eval_metrics["evalImprovementRatio"]
                if isinstance(improvement, torch.Tensor):
                    improvement = improvement.detach().cpu().item()
                message += f" (Improvement: {improvement:+.2%})"

            if toPrint:
                print(message)
            if toLog:
                logging.info(message)

        # TensorBoard logging for detailed tracking
        if toLog:
            # Log individual loss components
            for name, value_list in self.batchLoss.items():
                if value_list:  # Only log if we have values
                    self.writer.add_scalar(f"{self.mode}/batch_{name}", value_list[-1], step)

            # Log evaluation metrics
            for name, value_list in self.batchEvaluation.items():
                if value_list:
                    self.writer.add_scalar(f"{self.mode}/batch_{name}", value_list[-1], step)

    def updateEpoch(self, timeMin: float, physics_metrics: Optional[dict] = None):
        """
        Update epoch tracking with generative operator extensions.

        Args:
            timeMin: Time elapsed in minutes
            physics_metrics: Optional physics-aware metrics for the epoch
        """
        # Calculate epoch averages
        epoch_averages = {}
        loss = 0
        partStr = ""

        # Process standard losses
        for name, lossList in self.batchLoss.items():
            if lossList:  # Only process if we have values
                part = np.mean(np.array(lossList))
                epoch_averages[name] = part
                self.writer.add_scalar(f"{self.mode}/epoch_{name}", part, self.epoch)

                if name in ["lossFull", "lossPrior", "lossCorrector"]:
                    loss = part
                else:
                    partStr += f"{name.replace('loss', '')} {part:1.3f} "

                # Update accuracy metrics
                accName = name.replace("lossRec", "r").replace("lossPred", "p").replace("lossFull", "Loss")
                accName = accName.replace("lossPrior", "Prior").replace("lossCorrector", "Corrector")
                self.accuracy[f"l_{accName}"] = part

                best_key = f"b_{accName}"
                if best_key not in self.accuracy:
                    self.accuracy[best_key] = float("inf")
                if self.accuracy[best_key] > part:
                    self.accuracy[best_key] = part

        # Process evaluation metrics
        eval_str = ""
        for name, valueList in self.batchEvaluation.items():
            if valueList:
                part = np.mean(np.array(valueList))
                epoch_averages[name] = part
                self.writer.add_scalar(f"{self.mode}/epoch_{name}", part, self.epoch)

                if "Improvement" in name:
                    eval_str += f"{name.replace('eval', '').replace('Ratio', '')} {part:+.2%} "
                else:
                    eval_str += f"{name.replace('eval', '')} {part:1.3f} "

        # Add physics metrics if provided
        if physics_metrics:
            for name, value in physics_metrics.items():
                epoch_averages[f"physics_{name}"] = value
                self.writer.add_scalar(f"{self.mode}/epoch_physics_{name}", value, self.epoch)

        # Enhanced epoch logging
        stage_str = f"Stage {self.training_stage} "
        print(f"{self.modeLong} {stage_str}Epoch {self.epoch+1} ({timeMin:2.2f} min): {loss:1.4f}    {partStr}")
        if eval_str:
            print(f"    Evaluation: {eval_str}")
        print("")

        logging.info(f"{self.modeLong} {stage_str}Epoch {self.epoch+1} ({timeMin:2.2f} min): {loss:1.4f}    {partStr}")
        if eval_str:
            logging.info(f"    Evaluation: {eval_str}")
        logging.info("")

        # Store stage history
        stage_key = f"stage_{self.training_stage}"
        self.stage_history[stage_key].append({
            'epoch': self.epoch,
            'loss': loss,
            'metrics': epoch_averages.copy()
        })

        # Track improvement over time
        if "evalImprovementRatio" in epoch_averages:
            self.improvement_history.append({
                'epoch': self.epoch,
                'improvement': epoch_averages["evalImprovementRatio"],
                'stage': self.training_stage
            })

        # Process batch comparisons (from parent)
        for name, lossList in self.batchComparison.items():
            if lossList:
                part = np.mean(np.array(lossList))
                self.writer.add_scalar(f"{self.mode}/epoch_{name}", part, self.epoch)

    def updateBatchWithDualEvaluation(self, lossParts: dict, lossSeq: dict,
                                    sample: int, timeMin: float,
                                    prior_pred: torch.Tensor,
                                    corrected_pred: torch.Tensor,
                                    ground_truth: torch.Tensor):
        """
        Update batch with dual evaluation (prior vs corrected).

        Args:
            lossParts: Dictionary of loss components
            lossSeq: Dictionary of sequence losses
            sample: Current sample index
            timeMin: Time elapsed in minutes
            prior_pred: Prior predictions [B, T, C, H, W]
            corrected_pred: Corrected predictions [B, T, C, H, W]
            ground_truth: Ground truth [B, T, C, H, W]
        """
        # Compute evaluation metrics
        eval_metrics = self._compute_dual_evaluation_metrics(
            prior_pred, corrected_pred, ground_truth
        )

        # Update with evaluation metrics
        self.updateBatch(lossParts, lossSeq, sample, timeMin, eval_metrics)

    def _compute_dual_evaluation_metrics(self, prior_pred: torch.Tensor,
                                       corrected_pred: torch.Tensor,
                                       ground_truth: torch.Tensor) -> dict:
        """Compute dual evaluation metrics for current batch."""
        with torch.no_grad():
            # Convert to numpy for metrics computation
            prior_np = prior_pred.detach().cpu().numpy()
            corrected_np = corrected_pred.detach().cpu().numpy()
            gt_np = ground_truth.detach().cpu().numpy()

            # Compute basic metrics
            prior_mse = np.mean((prior_np - gt_np) ** 2)
            corrected_mse = np.mean((corrected_np - gt_np) ** 2)

            improvement_ratio = (prior_mse - corrected_mse) / prior_mse if prior_mse > 0 else 0

            metrics = {
                "evalPriorMSE": prior_mse,
                "evalCorrectedMSE": corrected_mse,
                "evalImprovementRatio": improvement_ratio
            }

            # Add physics compliance if applicable
            if self.dataset_type == "inc" and prior_np.shape[2] >= 2:  # Velocity components available
                physics_compliance = self._compute_physics_compliance(corrected_np, gt_np)
                metrics["evalPhysicsCompliance"] = physics_compliance

            # Add spectral improvement if applicable
            spectral_improvement = self._compute_spectral_improvement(
                prior_np, corrected_np, gt_np
            )
            metrics["evalSpectralImprovement"] = spectral_improvement

            return metrics

    def _compute_physics_compliance(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Compute physics compliance score (simplified)."""
        # Simplified divergence check for incompressible flow
        if self.dataset_type == "inc" and pred.shape[2] >= 2:
            u_pred = pred[:, :, 0]  # x-velocity
            v_pred = pred[:, :, 1]  # y-velocity

            # Compute divergence using finite differences
            du_dx = np.diff(u_pred, axis=-1)
            dv_dy = np.diff(v_pred, axis=-2)

            # Align shapes
            min_size = min(du_dx.shape[-1], dv_dy.shape[-1])
            divergence = du_dx[..., :min_size] + dv_dy[..., :min_size]

            # Return inverse of divergence magnitude (higher is better)
            div_magnitude = np.mean(np.abs(divergence))
            return 1.0 / (1.0 + div_magnitude)

        return 1.0  # Default compliance

    def _compute_spectral_improvement(self, prior: np.ndarray,
                                    corrected: np.ndarray,
                                    gt: np.ndarray) -> float:
        """Compute spectral improvement score (simplified)."""
        try:
            # Take first sample and channel for spectral analysis
            prior_sample = prior[0, 0, 0]
            corrected_sample = corrected[0, 0, 0]
            gt_sample = gt[0, 0, 0]

            # Compute FFT
            prior_fft = np.abs(np.fft.fft2(prior_sample))
            corrected_fft = np.abs(np.fft.fft2(corrected_sample))
            gt_fft = np.abs(np.fft.fft2(gt_sample))

            # High-frequency error
            h, w = prior_fft.shape
            hf_mask = np.zeros((h, w))
            hf_mask[h//4:3*h//4, w//4:3*w//4] = 1

            prior_hf_error = np.mean((prior_fft * hf_mask - gt_fft * hf_mask) ** 2)
            corrected_hf_error = np.mean((corrected_fft * hf_mask - gt_fft * hf_mask) ** 2)

            # Improvement ratio
            if prior_hf_error > 0:
                return (prior_hf_error - corrected_hf_error) / prior_hf_error
            else:
                return 0.0

        except Exception:
            return 0.0  # Return 0 if spectral analysis fails

    def setTrainingStage(self, stage: int):
        """Set current training stage and clear metrics."""
        self.training_stage = stage
        self.clear()
        logging.info(f"Switched to training stage {stage}")

    def writeDualPredictionExample(self, prior_pred: torch.Tensor,
                                 corrected_pred: torch.Tensor,
                                 ground_truth: torch.Tensor):
        """
        Write dual prediction examples to TensorBoard.

        Args:
            prior_pred: Prior predictions [B, T, C, H, W]
            corrected_pred: Corrected predictions [B, T, C, H, W]
            ground_truth: Ground truth [B, T, C, H, W]
        """
        numExamples = min(prior_pred.shape[0], 1)

        # Transpose for visualization [B, T, C, H, W] -> [B, T, C, W, H]
        p_prior = torch.transpose(prior_pred[0:numExamples], 3, 4)
        p_corrected = torch.transpose(corrected_pred[0:numExamples], 3, 4)
        g = torch.transpose(ground_truth[0:numExamples], 3, 4)

        # Create side-by-side comparisons
        if ground_truth.ndim == 5:
            for c in range(min(3, ground_truth.shape[2])):  # Limit to 3 channels
                field_name = self.simFields[c] if c < len(self.simFields) else f"field_{c}"

                # Create comparison: [Ground Truth | Prior | Corrected]
                comparison = torch.concat([
                    g[:, :, c:c+1],
                    p_prior[:, :, c:c+1],
                    p_corrected[:, :, c:c+1]
                ], dim=3)

                # Normalize for visualization
                comp_min = torch.amin(comparison, dim=(1, 3, 4), keepdim=True)
                comp_max = torch.amax(comparison, dim=(1, 3, 4), keepdim=True)
                comparison = (comparison - comp_min) / (comp_max - comp_min + 1e-8)

                # Select time steps for visualization
                step = max(1, int(comparison.shape[1] / 4))
                img = comparison[0, [0, step, 2*step, -1]] if comparison.shape[1] > 3 else comparison[0, [0, -1]]

                # Add to TensorBoard
                comparison_3ch = comparison.expand(-1, -1, 3, -1, -1)
                self.writer.add_images(f"{self.mode}_GenOp_PredictionImg/{field_name}_comparison",
                                     img, self.epoch)
                self.writer.add_video(f"{self.mode}_GenOp_PredictionVid/{field_name}_comparison",
                                    comparison_3ch, self.epoch, fps=5)

                # Also create difference visualization (Correction Applied)
                correction = p_corrected[:, :, c:c+1] - p_prior[:, :, c:c+1]
                correction_norm = correction / (torch.std(correction) + 1e-8)  # Normalize by std dev

                correction_img = correction_norm[0, [0, step, 2*step, -1]] if correction_norm.shape[1] > 3 else correction_norm[0, [0, -1]]
                self.writer.add_images(f"{self.mode}_GenOp_PredictionImg/{field_name}_correction",
                                     correction_img, self.epoch)

    def writeImprovementTimeseries(self):
        """Write improvement timeseries plot to TensorBoard."""
        if not self.improvement_history:
            return

        epochs = [entry['epoch'] for entry in self.improvement_history]
        improvements = [entry['improvement'] for entry in self.improvement_history]
        stages = [entry['stage'] for entry in self.improvement_history]

        fig, ax = plt.subplots(1, figsize=(8, 4), tight_layout=True)
        ax.set_ylabel("Improvement Ratio")
        ax.set_xlabel('Epoch')
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)

        # Plot improvement over time with different colors for stages
        stage1_mask = np.array(stages) == 1
        stage2_mask = np.array(stages) == 2

        if np.any(stage1_mask):
            ax.plot(np.array(epochs)[stage1_mask], np.array(improvements)[stage1_mask],
                   'o-', color='blue', label='Stage 1 (Prior)', linewidth=1.5, markersize=3)

        if np.any(stage2_mask):
            ax.plot(np.array(epochs)[stage2_mask], np.array(improvements)[stage2_mask],
                   's-', color='red', label='Stage 2 (Corrector)', linewidth=1.5, markersize=3)

        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.legend()
        ax.set_title(f"Model Improvement Over Time ({self.dataset_type.upper()} Dataset)")

        self.writer.add_figure(f"{self.mode}_GenOp_Analytics/improvement_timeseries", fig, self.epoch)
        plt.close(fig)

    def generateStageComparison(self) -> dict:
        """Generate comparison between training stages."""
        comparison = {}

        for stage_key, history in self.stage_history.items():
            if history:
                final_metrics = history[-1]['metrics']
                comparison[stage_key] = {
                    'final_loss': final_metrics.get('lossFull', 0),
                    'final_improvement': final_metrics.get('evalImprovementRatio', 0),
                    'epochs_trained': len(history)
                }

        return comparison

    def updateAccuracy(self, params: List, otherHistories: List["LossHistory"], finalPrint: bool):
        """Extended accuracy update with generative operator metrics."""
        # Call parent method
        super().updateAccuracy(params, otherHistories, finalPrint)

        # Add generative operator specific metrics
        if finalPrint and self.improvement_history:
            print("")
            print("Generative Operator Performance Summary:")
            logging.info("")
            logging.info("Generative Operator Performance Summary:")

            # Overall improvement
            final_improvement = self.improvement_history[-1]['improvement']
            print(f"  Final improvement ratio: {final_improvement:+.2%}")
            logging.info(f"  Final improvement ratio: {final_improvement:+.2%}")

            # Stage comparison
            stage_comparison = self.generateStageComparison()
            for stage, metrics in stage_comparison.items():
                print(f"  {stage}: {metrics['epochs_trained']} epochs, "
                     f"final loss: {metrics['final_loss']:.4f}")
                logging.info(f"  {stage}: {metrics['epochs_trained']} epochs, "
                           f"final loss: {metrics['final_loss']:.4f}")

            # Add improvement metrics to TensorBoard hparams
            improvement_metrics = {
                'genop/final_improvement_ratio': final_improvement,
                'genop/total_stages': max(self.training_stage, 2)
            }

            # Extract parameters
            par = {}
            for p in params:
                if p:
                    par.update(p.asDict())

            # Combine all metrics
            all_metrics = {}
            for hist in [self] + otherHistories:
                for stat, value in hist.accuracy.items():
                    all_metrics[f"m/{hist.mode}_{stat}"] = value

            all_metrics.update(improvement_metrics)

            # Write to TensorBoard
            self.writer.add_hparams(par, all_metrics)


# Convenience function for creating generative operator loss history
def create_genop_loss_history(mode: str, modeLong: str, writer: SummaryWriter,
                            dataLoaderLength: int, epoch: int, epochStep: int,
                            dataset_type: str = "inc", training_stage: int = 1,
                            **kwargs) -> GenerativeOperatorLossHistory:
    """
    Create GenerativeOperatorLossHistory with sensible defaults.

    Args:
        mode: Training mode
        modeLong: Long mode description
        writer: TensorBoard writer
        dataLoaderLength: Data loader length
        epoch: Current epoch
        epochStep: Epoch step
        dataset_type: Dataset type
        training_stage: Training stage
        **kwargs: Additional arguments

    Returns:
        GenerativeOperatorLossHistory: Configured loss history
    """
    return GenerativeOperatorLossHistory(
        mode=mode,
        modeLong=modeLong,
        writer=writer,
        dataLoaderLength=dataLoaderLength,
        epoch=epoch,
        epochStep=epochStep,
        dataset_type=dataset_type,
        training_stage=training_stage,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage and testing
    from torch.utils.tensorboard import SummaryWriter
    import tempfile

    # Create temporary directory for TensorBoard logs
    with tempfile.TemporaryDirectory() as temp_dir:
        writer = SummaryWriter(temp_dir)

        # Create loss history
        loss_history = GenerativeOperatorLossHistory(
            mode="train",
            modeLong="Training",
            writer=writer,
            dataLoaderLength=100,
            epoch=0,
            epochStep=1,
            dataset_type="inc",
            training_stage=1
        )

        # Test batch update
        test_loss = {"lossPrior": torch.tensor(0.5)}
        test_eval = {"evalImprovementRatio": 0.1}

        loss_history.updateBatch(test_loss, {}, 0, 1.0, test_eval)
        loss_history.updateEpoch(1.0)

        print("GenerativeOperatorLossHistory test completed successfully!")

        writer.close()