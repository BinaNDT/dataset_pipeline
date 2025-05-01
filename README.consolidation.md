# Building Damage Assessment Pipeline (Consolidated)

This is a streamlined version of the Building Damage Assessment Pipeline, with redundant files removed and the codebase structure organized for better maintainability.

## Directory Structure

```
dataset_pipeline/
├── config.py                   # Configuration parameters
├── train.py                    # Training script
├── inference.py                # Inference script
├── export_to_coco.py           # Convert predictions to COCO format
├── labelbox_importer.py        # Unified Labelbox import functionality
├── visualize_annotations.py    # Enhanced visualization with scaling
├── analyze_predictions.py      # Analysis of prediction statistics
├── upload_images_to_labelbox.py # For uploading images to Labelbox
├── utils/                      # Helper functions
│   ├── dataset.py              # Dataset utilities
│   └── logging_utils.py        # Logging utilities
├── scripts/                    # Utility scripts
│   └── cleanup.py              # Utility to clean up old model checkpoints
├── examples/                   # Example files and documentation
├── tests/                      # Unit tests
│   ├── test_dataset.py         # Tests for dataset utilities
│   └── test_logging_utils.py   # Tests for logging utilities
├── cloud_setup/                # Cloud storage configuration
├── outputs/                    # Generated during runtime
│   ├── models/                 # Saved model checkpoints
│   ├── predictions/            # Model predictions
│   └── logs/                   # Training and inference logs
├── archived/                   # Archived old files (can be deleted if not needed)
└── requirements.txt            # Python dependencies
```

## Core Functionality

### 1. Configuration
The `config.py` file defines all settings for the pipeline, including paths, model parameters, and credentials.

### 2. Training
Training is handled by `train.py`, which fine-tunes Mask2Former on the building damage dataset.

### 3. Inference
Running model inference is done with `inference.py`, generating predictions for images.

### 4. Export to COCO
The `export_to_coco.py` script converts model predictions to the COCO annotation format.

### 5. Visualization
The `visualize_annotations.py` script provides enhanced visualization of annotations, with scaling and highlighting options.

### 6. Analysis
The `analyze_predictions.py` script analyzes prediction statistics and quality.

### 7. Labelbox Integration
Two scripts handle Labelbox integration:
- `upload_images_to_labelbox.py`: Uploads images to Labelbox
- `labelbox_importer.py`: Imports annotations to Labelbox

## Usage Examples

Follow these steps for a complete workflow:

1. Configure settings in `config.py`
2. Train the model (optional): `python train.py`
3. Run inference: `python inference.py`
4. Export to COCO format: `python export_to_coco.py`
5. Visualize results: `python visualize_annotations.py --only-with-annotations --scale 10 --highlight-border`
6. Analyze predictions: `python analyze_predictions.py`
7. Upload images to Labelbox: `python upload_images_to_labelbox.py`
8. Upload annotations to Labelbox: `python labelbox_importer.py`

## Notes

- The `archived/` directory contains old versions and redundant files that were part of the original codebase.
- The `scripts/` directory contains utility scripts that can be useful for maintenance.
- Tests are located in the `tests/` directory and can be run with pytest.
