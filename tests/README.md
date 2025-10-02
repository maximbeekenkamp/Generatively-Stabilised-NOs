# Neural Operator Testing Framework

A comprehensive testing framework for neural operator models with sophisticated test coverage, analysis integration, and utility functions.

## Overview

This testing framework provides extensive test coverage for all neural operator model variants including:

- **FNO** (Fourier Neural Operator) - FNO16, FNO32, FNO16+DM
- **TNO** (Transformer Neural Operator) - TNO, TNO+DM
- **UNet** - UNet, UNet ut, UNet tn, UNet+DM
- **ResNet** - ResNet, ResNet dilated
- **Transformer** - TF MGN, TF Enc, TF VAE
- **Refiner** - PDERefiner for iterative refinement
- **ACDM** - Adaptive Conditional Diffusion Model (ACDM, ACDM ncn)
- **Diffusion Variants** - All +DM (Diffusion Model) combinations

## Quick Start

### Basic Model Testing

```python
from tests.utils import BaseModelTestCase
from tests.fixtures.dummy_datasets import get_dummy_batch

class TestMyModel(BaseModelTestCase):
    def setUp(self):
        super().setUp()
        self.model = MyModel()  # Your model here
        self.input_batch, _ = get_dummy_batch("inc_low", batch_size=2)

    def test_basic_functionality(self):
        """Test basic model functionality"""
        self.run_standard_model_tests(self.model, self.input_batch, "MyModel")

    def test_output_validity(self):
        """Test output validity"""
        output = self.assert_valid_model_output(self.model, self.input_batch)
        # Additional assertions...
```

### Model-Specific Testing

```python
from tests.fixtures.dummy_datasets import get_dummy_batch_for_model

# Get data optimized for specific model types
fno_input, fno_target = get_dummy_batch_for_model("fno", "inc_low", batch_size=2)
tno_input, tno_target = get_dummy_batch_for_model("tno", "inc_low", batch_size=2)
diffusion_input, diffusion_target = get_dummy_batch_for_model("diffusion", "inc_low", batch_size=2)
```

### Multi-Model Comparison

```python
from tests.utils import ModelComparator

comparator = ModelComparator()
comparator.add_model("FNO16", fno_model)
comparator.add_model("TNO", tno_model)
comparator.add_model("UNet", unet_model)

# Compare performance
report = comparator.generate_comparison_report(input_tensor, ground_truth)
print(report)
```

## Framework Structure

```
tests/
├── README.md                          # This file
├── fixtures/
│   ├── dummy_datasets.py              # Backwards compatible interface
│   └── dummy_datasets_enhanced.py     # Enhanced dataset generation
├── unit/
│   ├── models/                        # Model-specific tests
│   │   ├── test_fno_variants.py      # FNO model tests
│   │   ├── test_tno_variants.py      # TNO model tests
│   │   ├── test_unet_variants.py     # UNet model tests
│   │   ├── test_resnet_variants.py   # ResNet model tests
│   │   ├── test_tf_variants.py       # Transformer model tests
│   │   ├── test_refiner_variants.py  # Refiner model tests
│   │   ├── test_acdm_variants.py     # ACDM model tests
│   │   ├── test_diffusion_variants.py # +DM integration tests
│   │   └── test_physical_validation.py # Physical property tests
│   ├── analysis/                      # Analysis integration tests
│   │   └── test_analysis_integration.py
│   ├── training/                      # Training workflow tests
│   │   └── test_training_workflows.py
│   └── visualization/                 # Visualization tests
│       └── test_visual_output_generation.py
└── utils/                             # Testing utilities
    ├── __init__.py
    ├── test_utilities.py             # Core utilities
    └── common_patterns.py            # Common testing patterns
```

## Key Features

### 1. Enhanced Dummy Dataset Factory

The enhanced dataset factory generates physically realistic test data:

```python
from tests.fixtures.dummy_datasets import DummyDatasetFactory, DatasetConfig

# Create custom dataset configuration
config = DatasetConfig(
    dataset_type='inc_low',
    num_samples=10,
    spatial_size=(32, 32),
    sequence_length=20,
    num_channels=3,
    field_types=["velocity_u", "velocity_v", "pressure"],
    reynolds_number=1000.0,
    add_noise=True,
    noise_level=0.05,
    temporal_evolution=True,
    physical_constraints=True
)

dataset = DummyDatasetFactory.create_custom_dataset(config)
```

### 2. Comprehensive Test Utilities

The framework includes sophisticated testing utilities:

```python
from tests.utils import (
    ModelTestSuite,
    PerformanceProfiler,
    ErrorInjector,
    TestDataValidator
)

# Performance profiling
profiler = PerformanceProfiler()
with profiler.profile("forward_pass"):
    output = model(input_tensor)

# Error injection for robustness testing
corrupted_input = ErrorInjector.corrupt_input(
    input_tensor,
    corruption_type="gaussian_noise",
    scale=0.1
)

# Data quality validation
validation_results = TestDataValidator.validate_tensor_properties(tensor)
```

### 3. Physical Validation

Tests ensure models produce physically reasonable outputs:

```python
from tests.utils import PhysicalValidationTestPattern

validation_pattern = PhysicalValidationTestPattern(self)
validation_pattern.test_physical_properties(model, input_tensor)
```

### 4. Analysis Integration

Tests verify integration with analysis tools from the reference folder:

```python
# Tests TKE analysis, frequency domain analysis, loss computation, etc.
python -m pytest tests/unit/analysis/test_analysis_integration.py -v
```

### 5. Training Workflow Testing

Comprehensive training pipeline testing:

```python
# Tests basic training, two-stage generative operator training, loss functions, etc.
python -m pytest tests/unit/training/test_training_workflows.py -v
```

### 6. Visual Output Generation

Tests for visualization capabilities:

```python
# Tests field visualization, temporal sequences, error fields, etc.
python -m pytest tests/unit/visualization/test_visual_output_generation.py -v
```

## Running Tests

### Run All Tests

```bash
# Run all tests (pytest.ini handles path configuration automatically)
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run with quiet mode for summary
pytest tests/ -q
```

### Run Specific Test Categories

```bash
# Model tests
python -m pytest tests/unit/models/ -v

# Analysis integration tests
python -m pytest tests/unit/analysis/ -v

# Training workflow tests
python -m pytest tests/unit/training/ -v

# Visualization tests
python -m pytest tests/unit/visualization/ -v
```

### Run Tests for Specific Models

```bash
# FNO tests
python -m pytest tests/unit/models/test_fno_variants.py -v

# UNet tests
python -m pytest tests/unit/models/test_unet_variants.py -v

# Diffusion integration tests
python -m pytest tests/unit/models/test_diffusion_variants.py -v
```

### Run with Specific Datasets

```python
# Test with specific dataset
from tests.fixtures.dummy_datasets import get_dummy_batch

for dataset_name in ['inc_low', 'inc_high', 'tra_ext', 'tra_inc', 'iso']:
    input_batch, target_batch = get_dummy_batch(dataset_name, batch_size=4)
    # Run your tests...
```

## Advanced Usage

### Custom Test Patterns

```python
from tests.utils import ModelVariantTestPattern, DatasetCompatibilityTestPattern

# Test all model variants
variant_pattern = ModelVariantTestPattern(self)
variant_configs = {
    "small": {"hidden_dim": 32, "num_layers": 2},
    "large": {"hidden_dim": 128, "num_layers": 4}
}
variant_pattern.test_all_variants(ModelFactory, variant_configs, input_tensor)

# Test dataset compatibility
dataset_pattern = DatasetCompatibilityTestPattern(self)
dataset_pattern.test_dataset_compatibility(
    model,
    ['inc_low', 'tra_ext', 'iso'],
    lambda name: get_dummy_batch(name, batch_size=2)
)
```

### Performance Benchmarking

```python
from tests.utils import BaseModelTestCase

class BenchmarkTest(BaseModelTestCase):
    def test_performance_benchmark(self):
        results = self.benchmark_model_performance(
            self.model,
            self.input_tensor,
            num_trials=20
        )
        print(f"Mean time: {results['mean_time']:.4f}s")
        print(f"Std time: {results['std_time']:.4f}s")
```

### Memory Usage Testing

```python
from tests.utils import PerformanceTestPattern

performance_pattern = PerformanceTestPattern(self)
results = performance_pattern.test_performance_requirements(
    model,
    input_tensor,
    max_time=2.0,      # Maximum 2 seconds
    max_memory_mb=500.0 # Maximum 500MB
)
```

### Robustness Testing

```python
from tests.utils import ErrorInjector

# Test with various corruptions
corruption_types = ["gaussian_noise", "salt_pepper", "dropout", "blur"]

for corruption in corruption_types:
    corrupted_input = ErrorInjector.corrupt_input(
        input_tensor,
        corruption_type=corruption
    )

    output = model(corrupted_input)
    # Verify model handles corruption gracefully...
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Neural Operator Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]

    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run model tests
      run: pytest tests/unit/models/ -v --cov=src

    - name: Run integration tests
      run: pytest tests/unit/analysis/ tests/unit/training/ -v

    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

## Contributing

### Adding New Model Tests

1. Create a new test file in `tests/unit/models/`
2. Inherit from `BaseModelTestCase`
3. Use `get_dummy_batch_for_model()` for model-specific data
4. Include all required test patterns:
   - Basic functionality
   - Physical validation
   - Dataset compatibility
   - Performance requirements

### Example New Model Test

```python
from tests.utils import BaseModelTestCase
from tests.fixtures.dummy_datasets import get_dummy_batch_for_model

class TestMyNewModel(BaseModelTestCase):
    def setUp(self):
        super().setUp()
        self.model = MyNewModel()  # Your new model

    def test_all_datasets(self):
        """Test compatibility with all datasets"""
        for dataset_name in ['inc_low', 'inc_high', 'tra_ext', 'tra_inc', 'iso']:
            with self.subTest(dataset=dataset_name):
                input_batch, target_batch = get_dummy_batch_for_model(
                    "my_model", dataset_name, batch_size=2
                )

                output = self.assert_valid_model_output(self.model, input_batch)
                # Additional tests...
```

## Best Practices

1. **Always use dummy datasets** - Don't load real data in unit tests
2. **Test all model variants** - Include all architectural variations
3. **Validate physical properties** - Ensure outputs are physically reasonable
4. **Test dataset compatibility** - Verify models work with all 5 datasets
5. **Include performance tests** - Set reasonable time/memory limits
6. **Use descriptive test names** - Make failures easy to diagnose
7. **Test error conditions** - Include robustness testing
8. **Document test requirements** - Explain what each test validates

## Troubleshooting

### Common Issues

1. **Import Errors**: Tests now use `conftest.py` for automatic path configuration. If issues persist, ensure `pytest.ini` is present in project root.
2. **CUDA Issues**: Tests automatically skip CUDA tests if unavailable
3. **Memory Issues**: Use smaller batch sizes in tests. The `conftest.py` automatically clears CUDA cache between tests.
4. **Timeout Issues**: Increase timeout limits for slow models
5. **Matplotlib Issues**: Tests use non-interactive backend (`Agg`) configured in `conftest.py`

### Debug Mode

```python
# Enable verbose debugging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use smaller test data
input_batch, _ = get_dummy_batch("inc_low", batch_size=1, spatial_size=(8, 8))

# Run specific test with verbose output
pytest tests/unit/models/test_fno_variants.py::TestFNOVariants::test_fno16_forward_pass -vv
```

## Performance Expectations

The testing framework is designed to be fast and efficient:

- **Individual model tests**: < 10 seconds
- **Full test suite**: < 5 minutes
- **Memory usage**: < 2GB RAM
- **Dummy data generation**: < 1 second

## Contact

For questions or issues with the testing framework, please check the test documentation or create an issue in the repository.