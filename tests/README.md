# Building Damage Assessment Pipeline Tests

This directory contains unit tests for the Building Damage Assessment Pipeline. The tests cover key functionality to ensure reliability and correctness.

## Test Coverage

The tests cover:

- **Dataset Utilities**: Tests for the dataset loading, mask conversions, and data batching
- **Logging Utilities**: Tests for logging setup, error handling, and performance measurement
- (Future) **Inference Pipeline**: Tests for the model prediction workflow
- (Future) **Visualization Utilities**: Tests for the annotation visualization tools

## Running Tests

To run all tests:

```bash
# From the dataset_pipeline directory
python -m unittest discover tests
```

To run a specific test file:

```bash
python -m unittest tests.test_dataset
python -m unittest tests.test_logging_utils
```

To run a specific test case:

```bash
python -m unittest tests.test_dataset.TestDatasetUtils.test_polygon_to_mask
```

## Debugging Tests

Add the `-v` flag for more verbose test output:

```bash
python -m unittest discover tests -v
```

## Creating New Tests

1. Create a new test file in the `tests` directory with the prefix `test_`
2. Import the `unittest` module and the components you want to test
3. Create a test class that inherits from `unittest.TestCase`
4. Add test methods with names starting with `test_`

Example:

```python
import unittest
from utils.my_module import my_function

class TestMyModule(unittest.TestCase):
    def test_my_function(self):
        result = my_function(10)
        self.assertEqual(result, 20)
``` 