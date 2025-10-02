#!/usr/bin/env python3
"""
Data Format Handler for Generative Operator Integration

This module handles data format compatibility between the generative operator
framework and the existing Gen Stabilised analysis pipeline. It provides
utilities for converting between different tensor formats, validating data
shapes, and ensuring NPZ output compatibility.

Author: Phase 3 Implementation
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from pathlib import Path


class DataFormatHandler:
    """Handles data format conversion and validation for generative operators."""

    def __init__(self):
        # Standard Gen Stabilised format: [B, T, C, H, W]
        self.gen_stabilised_format = "BTCHW"

        # Standard evaluation format: [models, evaluations, sequences, time, channels, H, W]
        self.evaluation_format = "MESTCHW"

        # Supported tensor formats
        self.supported_formats = {
            "BTCHW": "Batch, Time, Channels, Height, Width",
            "BCHW": "Batch, Channels, Height, Width",
            "TCHW": "Time, Channels, Height, Width",
            "MESTCHW": "Models, Evaluations, Sequences, Time, Channels, Height, Width"
        }

    def validate_gen_stabilised_format(self, tensor: torch.Tensor) -> bool:
        """
        Validate that tensor is in Gen Stabilised format [B, T, C, H, W].

        Args:
            tensor: Input tensor to validate

        Returns:
            bool: True if tensor is in correct format
        """
        if not isinstance(tensor, torch.Tensor):
            return False

        if len(tensor.shape) != 5:
            return False

        B, T, C, H, W = tensor.shape

        # Basic sanity checks
        if B <= 0 or T <= 0 or C <= 0 or H <= 0 or W <= 0:
            return False

        # Check for reasonable dimensions
        if C > 10:  # More than 10 channels seems unusual
            logging.warning(f"Unusual number of channels: {C}")

        if H != W and abs(H - W) > 32:  # Allow some aspect ratio variation
            logging.warning(f"Unusual aspect ratio: {H}x{W}")

        return True

    def convert_to_gen_stabilised_format(self, tensor: Union[torch.Tensor, np.ndarray],
                                       source_format: str) -> torch.Tensor:
        """
        Convert tensor from source format to Gen Stabilised format [B, T, C, H, W].

        Args:
            tensor: Input tensor
            source_format: Format of input tensor

        Returns:
            torch.Tensor: Tensor in Gen Stabilised format
        """
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)

        if source_format == "BTCHW":
            return tensor  # Already in correct format

        elif source_format == "BCHW":
            # Add time dimension: [B, C, H, W] -> [B, 1, C, H, W]
            return tensor.unsqueeze(1)

        elif source_format == "TCHW":
            # Add batch dimension: [T, C, H, W] -> [1, T, C, H, W]
            return tensor.unsqueeze(0)

        elif source_format == "MESTCHW":
            # Evaluation format: [M, E, S, T, C, H, W] -> [B, T, C, H, W]
            # Flatten first three dimensions into batch
            M, E, S, T, C, H, W = tensor.shape
            return tensor.reshape(M * E * S, T, C, H, W)

        else:
            raise ValueError(f"Unsupported source format: {source_format}")

    def convert_from_gen_stabilised_format(self, tensor: torch.Tensor,
                                         target_format: str) -> torch.Tensor:
        """
        Convert tensor from Gen Stabilised format to target format.

        Args:
            tensor: Input tensor in Gen Stabilised format [B, T, C, H, W]
            target_format: Target format

        Returns:
            torch.Tensor: Tensor in target format
        """
        if not self.validate_gen_stabilised_format(tensor):
            raise ValueError("Input tensor is not in Gen Stabilised format")

        B, T, C, H, W = tensor.shape

        if target_format == "BTCHW":
            return tensor  # Already in correct format

        elif target_format == "BCHW":
            if T != 1:
                raise ValueError(f"Cannot convert temporal tensor (T={T}) to BCHW format")
            return tensor.squeeze(1)

        elif target_format == "TCHW":
            if B != 1:
                raise ValueError(f"Cannot convert batched tensor (B={B}) to TCHW format")
            return tensor.squeeze(0)

        elif target_format == "MESTCHW":
            # This requires additional metadata about M, E, S dimensions
            # For now, assume single model, single evaluation
            return tensor.unsqueeze(0).unsqueeze(1)  # [B, T, C, H, W] -> [1, 1, B, T, C, H, W]

        else:
            raise ValueError(f"Unsupported target format: {target_format}")

    def prepare_for_npz_output(self, tensor: torch.Tensor,
                              num_models: int = 1,
                              num_evaluations: int = 1) -> np.ndarray:
        """
        Prepare tensor for NPZ output in evaluation format.

        Args:
            tensor: Input tensor in Gen Stabilised format [B, T, C, H, W]
            num_models: Number of models
            num_evaluations: Number of evaluations

        Returns:
            np.ndarray: Array in evaluation format [M, E, S, T, C, H, W]
        """
        if not self.validate_gen_stabilised_format(tensor):
            raise ValueError("Input tensor is not in Gen Stabilised format")

        B, T, C, H, W = tensor.shape

        # Convert to numpy if needed
        if isinstance(tensor, torch.Tensor):
            tensor_np = tensor.detach().cpu().numpy()
        else:
            tensor_np = tensor

        # Calculate number of sequences
        num_sequences = B // (num_models * num_evaluations)

        if num_sequences * num_models * num_evaluations != B:
            raise ValueError(f"Batch size {B} is not divisible by models ({num_models}) × evaluations ({num_evaluations})")

        # Reshape to evaluation format
        reshaped = tensor_np.reshape(num_models, num_evaluations, num_sequences, T, C, H, W)

        return reshaped

    def load_from_npz(self, npz_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load data from NPZ file and extract metadata.

        Args:
            npz_path: Path to NPZ file

        Returns:
            tuple: (data_array, metadata)
        """
        npz_data = np.load(npz_path)

        # Get the main data array (assuming first key is the data)
        data_key = list(npz_data.keys())[0]
        data_array = npz_data[data_key]

        # Extract metadata from shape
        if len(data_array.shape) == 7:  # MESTCHW format
            M, E, S, T, C, H, W = data_array.shape
            metadata = {
                "format": "MESTCHW",
                "num_models": M,
                "num_evaluations": E,
                "num_sequences": S,
                "sequence_length": T,
                "num_channels": C,
                "spatial_shape": (H, W),
                "total_batch_size": M * E * S
            }
        else:
            metadata = {
                "format": "unknown",
                "shape": data_array.shape
            }

        return data_array, metadata

    def save_to_npz(self, data: Union[torch.Tensor, np.ndarray],
                   output_path: str,
                   metadata: Optional[Dict[str, Any]] = None):
        """
        Save data to NPZ file with proper formatting.

        Args:
            data: Data to save
            output_path: Output NPZ file path
            metadata: Optional metadata to include
        """
        if isinstance(data, torch.Tensor):
            data_np = data.detach().cpu().numpy()
        else:
            data_np = data

        # Prepare save dictionary
        save_dict = {"predictions": data_np}

        if metadata:
            save_dict.update(metadata)

        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save with compression
        np.savez_compressed(output_path, **save_dict)

        logging.info(f"Saved data to {output_path} with shape {data_np.shape}")

    def validate_evaluation_output(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Validate evaluation output format and return diagnostics.

        Args:
            data: Evaluation output array

        Returns:
            dict: Validation results and diagnostics
        """
        diagnostics = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "info": {}
        }

        # Check dimensions
        if len(data.shape) != 7:
            diagnostics["valid"] = False
            diagnostics["issues"].append(f"Expected 7 dimensions, got {len(data.shape)}")
            return diagnostics

        M, E, S, T, C, H, W = data.shape

        # Store basic info
        diagnostics["info"] = {
            "num_models": M,
            "num_evaluations": E,
            "num_sequences": S,
            "sequence_length": T,
            "num_channels": C,
            "spatial_shape": (H, W),
            "data_type": str(data.dtype),
            "memory_size_mb": data.nbytes / (1024 * 1024)
        }

        # Validation checks
        if M <= 0 or E <= 0 or S <= 0 or T <= 0 or C <= 0 or H <= 0 or W <= 0:
            diagnostics["valid"] = False
            diagnostics["issues"].append("All dimensions must be positive")

        # Check for reasonable values
        if np.any(np.isnan(data)):
            diagnostics["valid"] = False
            diagnostics["issues"].append("Data contains NaN values")

        if np.any(np.isinf(data)):
            diagnostics["valid"] = False
            diagnostics["issues"].append("Data contains infinite values")

        # Warnings for unusual configurations
        if C > 10:
            diagnostics["warnings"].append(f"Unusual number of channels: {C}")

        if T > 1000:
            diagnostics["warnings"].append(f"Very long sequence: {T} timesteps")

        if data.nbytes > 1e9:  # > 1GB
            diagnostics["warnings"].append(f"Large memory usage: {data.nbytes / (1024**3):.2f} GB")

        return diagnostics

    def convert_legacy_format(self, legacy_data: np.ndarray,
                            legacy_format: str) -> torch.Tensor:
        """
        Convert legacy data formats to Gen Stabilised format.

        Args:
            legacy_data: Data in legacy format
            legacy_format: Description of legacy format

        Returns:
            torch.Tensor: Data in Gen Stabilised format
        """
        if legacy_format == "old_evaluation":
            # Old format might be [B, T, H, W, C] instead of [B, T, C, H, W]
            if len(legacy_data.shape) == 5:
                # Transpose to move channels to correct position
                legacy_data = np.transpose(legacy_data, (0, 1, 4, 2, 3))

        elif legacy_format == "raw_simulation":
            # Raw simulation data might need additional processing
            # This would be specific to the simulation output format
            pass

        # Convert to tensor and validate
        tensor = torch.from_numpy(legacy_data.astype(np.float32))

        if not self.validate_gen_stabilised_format(tensor):
            raise ValueError(f"Converted legacy data is not in valid Gen Stabilised format: {tensor.shape}")

        return tensor


class GenerativeOperatorDataProcessor:
    """Processor for generative operator data with format handling."""

    def __init__(self):
        self.format_handler = DataFormatHandler()

    def process_prediction_batch(self, predictions: torch.Tensor,
                                targets: Optional[torch.Tensor] = None,
                                model_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a batch of predictions with format validation and conversion.

        Args:
            predictions: Model predictions
            targets: Ground truth targets (optional)
            model_info: Model metadata (optional)

        Returns:
            dict: Processed data and metadata
        """
        # Validate input format
        if not self.format_handler.validate_gen_stabilised_format(predictions):
            raise ValueError("Predictions are not in Gen Stabilised format")

        # Process predictions
        processed = {
            "predictions": predictions.detach().cpu(),
            "shape": predictions.shape,
            "format": "BTCHW",
            "processing_metadata": {
                "timestamp": torch.tensor([0.0]),  # Placeholder
                "batch_size": predictions.shape[0],
                "sequence_length": predictions.shape[1],
                "num_channels": predictions.shape[2],
                "spatial_resolution": predictions.shape[3:5]
            }
        }

        # Add targets if provided
        if targets is not None:
            if not self.format_handler.validate_gen_stabilised_format(targets):
                raise ValueError("Targets are not in Gen Stabilised format")
            processed["targets"] = targets.detach().cpu()

        # Add model info if provided
        if model_info:
            processed["model_info"] = model_info

        return processed

    def prepare_for_analysis_pipeline(self, data: torch.Tensor,
                                    num_models: int = 1,
                                    num_evaluations: int = 1) -> np.ndarray:
        """
        Prepare data for integration with existing analysis pipeline.

        Args:
            data: Data in Gen Stabilised format
            num_models: Number of models
            num_evaluations: Number of evaluations

        Returns:
            np.ndarray: Data in evaluation format for analysis pipeline
        """
        return self.format_handler.prepare_for_npz_output(data, num_models, num_evaluations)


# Convenience functions for common operations
def convert_to_npz_format(tensor: torch.Tensor,
                         num_models: int = 1,
                         num_evaluations: int = 1) -> np.ndarray:
    """Convert tensor to NPZ evaluation format."""
    handler = DataFormatHandler()
    return handler.prepare_for_npz_output(tensor, num_models, num_evaluations)


def validate_genop_output(tensor: torch.Tensor) -> bool:
    """Validate generative operator output format."""
    handler = DataFormatHandler()
    return handler.validate_gen_stabilised_format(tensor)


def save_genop_predictions(predictions: torch.Tensor,
                          output_path: str,
                          num_models: int = 1,
                          num_evaluations: int = 1,
                          metadata: Optional[Dict[str, Any]] = None):
    """Save generative operator predictions in NPZ format."""
    handler = DataFormatHandler()

    # Convert to evaluation format
    eval_data = handler.prepare_for_npz_output(predictions, num_models, num_evaluations)

    # Save with metadata
    handler.save_to_npz(eval_data, output_path, metadata)


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    # Create test data
    test_tensor = torch.randn(4, 10, 3, 64, 64)  # [B=4, T=10, C=3, H=64, W=64]

    # Test format validation
    handler = DataFormatHandler()
    is_valid = handler.validate_gen_stabilised_format(test_tensor)
    print(f"Format validation: {is_valid}")

    # Test NPZ conversion
    npz_data = handler.prepare_for_npz_output(test_tensor, num_models=2, num_evaluations=2)
    print(f"NPZ format shape: {npz_data.shape}")

    # Test validation
    diagnostics = handler.validate_evaluation_output(npz_data)
    print(f"Validation results: {diagnostics['valid']}")
    print(f"Info: {diagnostics['info']}")