# Building Damage Assessment Pipeline Examples

This directory contains example notebooks and scripts demonstrating how to use the Building Damage Assessment Pipeline. These examples provide step-by-step guidance for common workflows.

## Available Examples

### Jupyter Notebooks

- **example_workflow.ipynb**: Complete end-to-end pipeline demonstration, from inference through visualization and analysis
- (Future) **custom_training.ipynb**: How to train the model on custom datasets
- (Future) **visualization_techniques.ipynb**: Advanced visualization techniques for damage assessment

### Sample Scripts

- (Future) **quick_inference.py**: Minimal script for running inference on a single image
- (Future) **batch_process.py**: Script for batch processing multiple videos

## Running the Notebooks

### Prerequisites

Make sure you have Jupyter installed:

```bash
pip install jupyter
```

### Starting Jupyter

From the dataset_pipeline directory:

```bash
jupyter notebook examples/
```

Or, to run a specific notebook:

```bash
jupyter notebook examples/example_workflow.ipynb
```

## Notes on Example Data

The examples use simulated data by default to avoid dependencies on specific datasets. To use your own data:

1. Update the `DATASET_ROOT` in `config.py` to point to your dataset
2. Make sure your images follow the expected directory structure
3. Check that your annotations use the expected format (COCO by default)

## Customizing Examples

You can modify the examples to suit your specific needs:

- Change model parameters in `config.py`
- Adjust visualization settings in the notebook cells
- Add additional analysis steps after the basic workflow 