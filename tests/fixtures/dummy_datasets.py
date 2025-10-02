"""
Dummy datasets for testing all model variants

This module now imports from the enhanced dummy datasets implementation
while maintaining backwards compatibility with existing code.
"""

# Import everything from the enhanced version for backwards compatibility
from .dummy_datasets_enhanced import (
    DummyTurbulenceDataset,
    DummyDatasetFactory,
    DatasetConfig,
    PhysicalPatternGenerator,
    FieldType,
    get_dummy_batch,
    get_dummy_batch_for_model,
    create_test_dataloader,
    validate_dummy_data_quality
)

# Legacy compatibility - maintain old interface
def get_dummy_batch_legacy(dataset_name: str = "inc_low", batch_size: int = 2):
    """Legacy version of get_dummy_batch for backwards compatibility"""
    return get_dummy_batch(dataset_name, batch_size)


# Quick test to ensure compatibility
if __name__ == "__main__":
    print("Testing backwards compatibility...")

    # Test legacy interface
    input_batch, target_batch = get_dummy_batch_legacy("inc_low", batch_size=2)
    print(f"Legacy interface: Input {input_batch.shape}, Target {target_batch.shape}")

    # Test new interface
    input_batch, target_batch = get_dummy_batch("inc_low", batch_size=2)
    print(f"Enhanced interface: Input {input_batch.shape}, Target {target_batch.shape}")

    # Test model-specific interface
    input_batch, target_batch = get_dummy_batch_for_model("fno", "inc_low", batch_size=2)
    print(f"Model-specific interface: Input {input_batch.shape}, Target {target_batch.shape}")

    print("✅ Backwards compatibility maintained!")