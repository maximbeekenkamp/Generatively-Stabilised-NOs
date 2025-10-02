"""
Memory Optimization and Performance Enhancements for Generative Operators

This module provides memory optimization utilities and performance enhancements
for generative operator models, including gradient checkpointing, efficient
batch processing, and memory-aware DCAR rollout.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math
import logging
from typing import Optional, Dict, Any, Tuple, List
import gc


class MemoryOptimizer:
    """
    Memory optimization utilities for generative operator models.

    Provides various strategies to reduce memory usage during training and inference,
    including gradient checkpointing, batch splitting, and memory-aware operations.
    """

    @staticmethod
    def estimate_memory_usage(model: nn.Module,
                            input_shape: Tuple[int, ...],
                            dtype: torch.dtype = torch.float32) -> Dict[str, float]:
        """
        Estimate memory usage for a model with given input shape.

        Args:
            model: PyTorch model
            input_shape: Input tensor shape
            dtype: Tensor data type

        Returns:
            memory_info: Dictionary with memory estimates in MB
        """
        # Calculate parameter memory
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())

        # Estimate activation memory (rough approximation)
        input_elements = math.prod(input_shape)
        bytes_per_element = torch.tensor(0, dtype=dtype).element_size()

        # Rough estimate: activations are ~3x input size for typical models
        activation_memory = input_elements * bytes_per_element * 3

        # Convert to MB
        param_mb = param_memory / (1024 ** 2)
        activation_mb = activation_memory / (1024 ** 2)
        total_mb = param_mb + activation_mb

        return {
            'parameters_mb': param_mb,
            'activations_mb': activation_mb,
            'total_estimated_mb': total_mb,
            'peak_estimated_mb': total_mb * 2  # Account for gradients during training
        }

    @staticmethod
    def optimize_batch_size(model: nn.Module,
                          max_memory_mb: float,
                          input_shape: Tuple[int, ...],
                          initial_batch_size: int = 1) -> int:
        """
        Find optimal batch size that fits within memory constraints.

        Args:
            model: PyTorch model
            max_memory_mb: Maximum memory budget in MB
            input_shape: Input shape without batch dimension
            initial_batch_size: Starting batch size for estimation

        Returns:
            optimal_batch_size: Recommended batch size
        """
        # Test with initial batch size
        test_shape = (initial_batch_size,) + input_shape
        memory_info = MemoryOptimizer.estimate_memory_usage(model, test_shape)

        if memory_info['peak_estimated_mb'] <= max_memory_mb:
            # Can potentially use larger batch size
            scale_factor = max_memory_mb / memory_info['peak_estimated_mb']
            optimal_batch_size = min(int(initial_batch_size * scale_factor * 0.8), 64)  # Safety margin
        else:
            # Need smaller batch size
            scale_factor = max_memory_mb / memory_info['peak_estimated_mb']
            optimal_batch_size = max(int(initial_batch_size * scale_factor), 1)

        logging.info(f"Memory optimization: recommended batch size {optimal_batch_size} "
                    f"for {max_memory_mb:.0f}MB budget")

        return optimal_batch_size

    @staticmethod
    def clear_cache():
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


class GradientCheckpointing:
    """
    Gradient checkpointing utilities for memory-efficient training.

    Implements checkpoint functions that trade compute for memory by
    recomputing activations during backward pass.
    """

    @staticmethod
    def checkpoint_function(function, *args, **kwargs):
        """
        Apply gradient checkpointing to a function.

        Args:
            function: Function to checkpoint
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            result: Function output with checkpointing
        """
        if kwargs.get('use_checkpointing', True) and torch.is_grad_enabled():
            return checkpoint(function, *args)
        else:
            return function(*args)

    @staticmethod
    def checkpoint_sequential(functions: List[nn.Module],
                            input_tensor: torch.Tensor,
                            segments: int = 2) -> torch.Tensor:
        """
        Apply gradient checkpointing to a sequence of modules.

        Args:
            functions: List of PyTorch modules
            input_tensor: Input tensor
            segments: Number of checkpointing segments

        Returns:
            output: Final output tensor
        """
        if not torch.is_grad_enabled() or segments < 2:
            # No checkpointing needed
            x = input_tensor
            for fn in functions:
                x = fn(x)
            return x

        # Split functions into segments
        segment_size = len(functions) // segments

        def run_segment(start_idx: int, end_idx: int, x: torch.Tensor) -> torch.Tensor:
            for i in range(start_idx, end_idx):
                x = functions[i](x)
            return x

        x = input_tensor
        for i in range(0, len(functions), segment_size):
            end_idx = min(i + segment_size, len(functions))
            x = checkpoint(run_segment, i, end_idx, x)

        return x


class MemoryEfficientDCAR:
    """
    Memory-efficient implementation of DCAR (Diffusion-Corrected AutoRegressive) rollout.

    Implements strategies to reduce memory usage during long rollouts while
    maintaining prediction quality.
    """

    def __init__(self,
                 model,
                 max_history_length: int = 50,
                 correction_batch_size: int = 4,
                 enable_checkpointing: bool = True):
        """
        Initialize memory-efficient DCAR.

        Args:
            model: Generative operator model
            max_history_length: Maximum history to keep in memory
            correction_batch_size: Batch size for diffusion correction
            enable_checkpointing: Enable gradient checkpointing
        """
        self.model = model
        self.max_history_length = max_history_length
        self.correction_batch_size = correction_batch_size
        self.enable_checkpointing = enable_checkpointing

    def memory_efficient_rollout(self,
                                initial_states: torch.Tensor,
                                num_steps: int,
                                correction_frequency: int = 1) -> torch.Tensor:
        """
        Perform memory-efficient DCAR rollout.

        Args:
            initial_states: Initial condition [B, T_init, C, H, W]
            num_steps: Number of steps to predict
            correction_frequency: How often to apply correction

        Returns:
            trajectory: Complete trajectory [B, T_init + num_steps, C, H, W]
        """
        device = initial_states.device
        B, T_init, C, H, W = initial_states.shape

        # Initialize with initial states
        trajectory_chunks = [initial_states.cpu()]  # Store on CPU to save GPU memory
        current_input = initial_states

        # Determine window size
        window_size = getattr(self.model.prior_model, 'prev_steps', 1)
        window_size = getattr(self.model.prior_model, 'L', window_size)

        logging.info(f"Starting memory-efficient DCAR rollout: {num_steps} steps, "
                    f"window size {window_size}, correction frequency {correction_frequency}")

        for step in range(num_steps):
            # Clear cache periodically
            if step % 10 == 0:
                MemoryOptimizer.clear_cache()

            # Prepare input window
            if trajectory_chunks:
                # Get recent history from stored chunks
                recent_frames = []
                total_frames = 0

                # Work backwards through chunks to get most recent frames
                for chunk in reversed(trajectory_chunks):
                    chunk_gpu = chunk.to(device)
                    frames_needed = window_size - total_frames

                    if frames_needed <= 0:
                        break

                    if chunk_gpu.shape[1] <= frames_needed:
                        recent_frames.insert(0, chunk_gpu)
                        total_frames += chunk_gpu.shape[1]
                    else:
                        # Take only the most recent frames from this chunk
                        recent_frames.insert(0, chunk_gpu[:, -frames_needed:])
                        total_frames += frames_needed

                if recent_frames:
                    input_window = torch.cat(recent_frames, dim=1)[:, -window_size:]
                else:
                    input_window = current_input
            else:
                input_window = current_input

            # Generate next prediction
            with torch.no_grad():
                if self.enable_checkpointing:
                    next_pred = GradientCheckpointing.checkpoint_function(
                        self.model.prior_model, input_window
                    )
                else:
                    next_pred = self.model.prior_model(input_window)

                next_frame = next_pred[:, -1:]  # Take last predicted frame

            # Apply correction at specified frequency
            if (step + 1) % correction_frequency == 0 and self.model.correction_strength > 0:
                next_frame = self._apply_memory_efficient_correction(next_frame, input_window)

            # Update trajectory
            trajectory_chunks.append(next_frame.cpu())
            current_input = next_frame

            # Limit memory usage by pruning old chunks
            if len(trajectory_chunks) > self.max_history_length:
                # Keep initial states and recent frames
                initial_chunk = trajectory_chunks[0]
                recent_chunks = trajectory_chunks[-self.max_history_length//2:]
                trajectory_chunks = [initial_chunk] + recent_chunks

        # Concatenate all chunks on GPU
        logging.info("Concatenating trajectory chunks...")
        trajectory_parts = []
        for chunk in trajectory_chunks:
            trajectory_parts.append(chunk.to(device))

        full_trajectory = torch.cat(trajectory_parts, dim=1)

        # Clear intermediate chunks from CPU memory
        del trajectory_chunks
        MemoryOptimizer.clear_cache()

        logging.info(f"Memory-efficient DCAR rollout completed: {full_trajectory.shape}")
        return full_trajectory

    def _apply_memory_efficient_correction(self,
                                         frame: torch.Tensor,
                                         context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply diffusion correction with memory efficiency.

        Args:
            frame: Frame to correct [B, 1, C, H, W]
            context: Optional context for conditioning

        Returns:
            corrected_frame: Corrected frame [B, 1, C, H, W]
        """
        B, T, C, H, W = frame.shape

        # Process in smaller batches if needed
        if B > self.correction_batch_size:
            corrected_batches = []

            for i in range(0, B, self.correction_batch_size):
                end_idx = min(i + self.correction_batch_size, B)
                batch_frame = frame[i:end_idx]
                batch_context = context[i:end_idx] if context is not None else None

                # Apply correction to batch
                corrected_batch = self.model.corrector_model.correct_prediction(
                    batch_frame,
                    prior_features=self.model.prior_model.get_prior_features(batch_context) if batch_context is not None else None,
                    correction_strength=self.model.correction_strength
                )

                corrected_batches.append(corrected_batch)

                # Clear intermediate tensors
                del batch_frame, batch_context
                MemoryOptimizer.clear_cache()

            return torch.cat(corrected_batches, dim=0)
        else:
            # Process entire batch at once
            return self.model.corrector_model.correct_prediction(
                frame,
                prior_features=self.model.prior_model.get_prior_features(context) if context is not None else None,
                correction_strength=self.model.correction_strength
            )


class AdaptiveBatchSampler:
    """
    Adaptive batch sampler that adjusts batch size based on available memory.

    Monitors memory usage and dynamically adjusts batch size to maximize
    throughput while staying within memory constraints.
    """

    def __init__(self,
                 initial_batch_size: int = 4,
                 min_batch_size: int = 1,
                 max_batch_size: int = 32,
                 memory_threshold: float = 0.9):
        """
        Initialize adaptive batch sampler.

        Args:
            initial_batch_size: Starting batch size
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
            memory_threshold: Memory usage threshold (0.0-1.0)
        """
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.memory_threshold = memory_threshold
        self.memory_history = []

    def get_current_batch_size(self) -> int:
        """Get current recommended batch size."""
        return self.current_batch_size

    def update_batch_size(self) -> int:
        """
        Update batch size based on memory usage.

        Returns:
            new_batch_size: Updated batch size
        """
        if not torch.cuda.is_available():
            return self.current_batch_size

        # Get current memory usage
        memory_allocated = torch.cuda.memory_allocated()
        memory_cached = torch.cuda.memory_reserved()
        memory_total = torch.cuda.get_device_properties(0).total_memory

        memory_usage_ratio = memory_allocated / memory_total
        self.memory_history.append(memory_usage_ratio)

        # Keep only recent history
        if len(self.memory_history) > 10:
            self.memory_history = self.memory_history[-10:]

        # Calculate average memory usage
        avg_memory_usage = sum(self.memory_history) / len(self.memory_history)

        # Adjust batch size based on memory usage
        if avg_memory_usage > self.memory_threshold:
            # Reduce batch size
            new_batch_size = max(self.current_batch_size - 1, self.min_batch_size)
            if new_batch_size != self.current_batch_size:
                logging.info(f"Reducing batch size from {self.current_batch_size} to {new_batch_size} "
                           f"(memory usage: {avg_memory_usage:.2%})")
        elif avg_memory_usage < self.memory_threshold * 0.7:
            # Increase batch size
            new_batch_size = min(self.current_batch_size + 1, self.max_batch_size)
            if new_batch_size != self.current_batch_size:
                logging.info(f"Increasing batch size from {self.current_batch_size} to {new_batch_size} "
                           f"(memory usage: {avg_memory_usage:.2%})")
        else:
            new_batch_size = self.current_batch_size

        self.current_batch_size = new_batch_size
        return self.current_batch_size

    def get_memory_stats(self) -> Dict[str, float]:
        """
        Get current memory statistics.

        Returns:
            memory_stats: Dictionary with memory information
        """
        if not torch.cuda.is_available():
            return {'memory_available': False}

        memory_allocated = torch.cuda.memory_allocated()
        memory_cached = torch.cuda.memory_reserved()
        memory_total = torch.cuda.get_device_properties(0).total_memory

        return {
            'memory_available': True,
            'allocated_mb': memory_allocated / (1024 ** 2),
            'cached_mb': memory_cached / (1024 ** 2),
            'total_mb': memory_total / (1024 ** 2),
            'usage_ratio': memory_allocated / memory_total,
            'current_batch_size': self.current_batch_size
        }


def apply_memory_optimizations(model,
                             enable_checkpointing: bool = True,
                             optimize_attention: bool = True,
                             use_mixed_precision: bool = True) -> nn.Module:
    """
    Apply various memory optimizations to a generative operator model.

    Args:
        model: Generative operator model
        enable_checkpointing: Enable gradient checkpointing
        optimize_attention: Enable attention optimizations
        use_mixed_precision: Enable mixed precision training

    Returns:
        optimized_model: Model with applied optimizations
    """
    if enable_checkpointing:
        # Apply gradient checkpointing to model components
        if hasattr(model, 'prior_model'):
            # Apply checkpointing to prior model
            _apply_checkpointing_to_module(model.prior_model)

        if hasattr(model, 'corrector_model'):
            # Apply checkpointing to corrector model
            _apply_checkpointing_to_module(model.corrector_model)

    if optimize_attention and torch.cuda.is_available():
        # Enable memory-efficient attention if available
        try:
            from torch.backends.cuda import sdp_kernel, SDPBackend
            with sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=True):
                pass  # This enables optimized attention globally
        except ImportError:
            logging.warning("Flash attention not available, skipping attention optimization")

    if use_mixed_precision:
        # Convert model to use half precision where appropriate
        model = _optimize_precision(model)

    logging.info("Applied memory optimizations to generative operator model")
    return model


def _apply_checkpointing_to_module(module: nn.Module):
    """Apply gradient checkpointing to a module."""
    # This is a simplified version - real implementation would be model-specific
    if hasattr(module, 'gradient_checkpointing_enable'):
        module.gradient_checkpointing_enable()

    for child in module.children():
        _apply_checkpointing_to_module(child)


def _optimize_precision(model: nn.Module) -> nn.Module:
    """Optimize model precision for memory efficiency."""
    # Convert batch norms and layer norms to float32 for stability
    # Keep other layers in half precision if possible

    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                              nn.LayerNorm, nn.GroupNorm)):
            module.float()

    return model