# Building Damage Assessment Pipeline

A comprehensive pipeline for detecting and classifying building damage in aerial imagery using Mask2Former. This pipeline handles the complete workflow from training through inference to annotation upload.

## Overview

This pipeline automates the process of:
1. Fine-tuning Mask2Former on custom building damage datasets
2. Running inference on large-scale video frame collections
3. Converting predictions to COCO and Labelbox formats
4. Visualizing and analyzing prediction quality
5. Uploading results to Labelbox for human review

### Damage Classification Categories

The model classifies buildings into four damage levels:
- **No Damage**: Buildings with no visible structural damage
- **Minor Damage**: Superficial damage, intact structural elements
- **Major Damage**: Significant structural damage but partially standing
- **Total Destruction**: Complete collapse or destruction

## Prerequisites

### Hardware Requirements
- CUDA-capable GPU(s)
- Minimum 16GB GPU memory per device
- Sufficient storage for dataset and model checkpoints

### Software Requirements
- Python 3.8+
- CUDA 11.3+
- PyTorch 2.0+
- OpenCV 4.5+
- Labelbox SDK 3.0+

### Dataset Requirements
- Images in PNG format
- COCO format annotations (for training)
- Organized in `/HurricaneVidNet_Dataset` structure

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd dataset_pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:

Copy the example environment file:
```bash
cp .env.example .env
```

Edit the `.env` file with your credentials:
```
LABELBOX_API_KEY=your_actual_api_key
LABELBOX_PROJECT_ID=your_actual_project_id
```

Alternatively, use the provided setup script for an interactive setup:
```bash
./setup_env.sh
```

You can also set these environment variables directly in your shell:
```bash
export LABELBOX_API_KEY="your_api_key"
export LABELBOX_PROJECT_ID="your_project_id"
```

For detailed security guidelines on handling credentials, see [Security Documentation](docs/security.md).

## Directory Structure

```
dataset_pipeline/
├── config.py                   # Configuration parameters
├── train.py                    # Training script
├── inference.py                # Inference script
├── export_to_coco.py           # Convert predictions to COCO format
├── labelbox_importer.py        # Unified Labelbox import functionality
├── visualize_annotations.py    # Basic visualization of annotations
├── fix_visualize_annotations.py # Enhanced visualization with scaling
├── analyze_predictions.py      # Analysis of prediction statistics
├── cleanup.py                  # Utility to clean up old model checkpoints
├── utils/                      # Helper functions
│   └── dataset.py              # Dataset utilities
├── outputs/                    # Generated during runtime
│   ├── models/                 # Saved model checkpoints
│   ├── predictions/            # Model predictions
│   └── logs/                   # Training and inference logs
├── docs/                       # Documentation
│   └── security.md             # Security guidelines
├── tests/                      # Unit tests
├── examples/                   # Example workflows and notebooks
├── .github/workflows/          # CI/CD configurations
│   └── security-checks.yml     # Security scan workflows
├── .secrets.baseline           # Baseline for secret detection
├── setup_env.sh                # Environment setup script
├── .env.example                # Template for environment variables
└── requirements.txt            # Python dependencies
```

## Complete Workflow

### Step 1: Configure Settings
Edit `config.py` to set up:
- Dataset paths
- Model parameters
- Labelbox credentials
- GPU configurations

### Step 2: Training (Optional)
Train the model on annotated dataset:

```bash
python train.py
```

### Step 3: Inference
Run inference on video frames:

```bash
python inference.py
```

This generates `outputs/predictions/predictions.json` containing segmentation masks and classifications.

### Step 4: Export to COCO Format
Convert predictions to COCO format:

```bash
python export_to_coco.py
```

This creates `outputs/predictions/predictions_coco.json` with standardized annotations.

### Step 5: Visualize Results
View the predictions with annotations:

```bash
# Basic visualization
python visualize_annotations.py --only-with-annotations

# Enhanced visualization with scaled annotations
python fix_visualize_annotations.py --only-with-annotations --scale 10 --highlight-border
```

### Step 6: Analyze Prediction Quality
Check prediction statistics:

```bash
python analyze_predictions.py
```

### Step 7: Upload to Labelbox
Import predictions to Labelbox for human review:

```bash
# From COCO format (default)
python labelbox_importer.py

# From raw predictions
python labelbox_importer.py --source predictions

# Debug mode with limited uploads
python labelbox_importer.py --debug --limit 5
```

### Step 8: Clean Up (Optional)
Manage disk space by removing old checkpoints:

```bash
# Keep only the latest checkpoint
python cleanup.py

# Keep the 3 most recent checkpoints
python cleanup.py --keep 3

# Preview what would be deleted without deleting
python cleanup.py --dry-run
```

## Script Details

### `config.py`
Configuration file with all parameters for the pipeline:
- Dataset paths and structure
- Model parameters and checkpoint paths
- Training hyperparameters
- Labelbox credentials
- Class definitions and color mappings

### `train.py`
Model training script with:
- Multi-GPU support
- Automatic mixed precision
- Gradient accumulation
- Checkpoint saving
- WandB integration
- Learning rate scheduling

### `inference.py`
Runs the model on video frames:
- Automatic GPU distribution
- Parallel video processing
- Confidence thresholding
- Structured JSON output

### `export_to_coco.py`
Converts model predictions to COCO format:
- Mask to polygon conversion
- COCO-compliant JSON structure
- Error handling for invalid masks

### `visualize_annotations.py`
Basic visualization of annotations:
- Renders predictions on original images
- Supports filtering by video or image ID
- Option to show only annotated images

### `fix_visualize_annotations.py`
Enhanced visualization with additional features:
- Scaling option for small annotations
- Border highlighting around annotation regions
- Annotation-only view option
- Better debug information

### `analyze_predictions.py`
Analyzes prediction data for quality assessment:
- Statistics on coverage and confidence
- Class distribution analysis
- Size and location analysis
- Warning for potential issues

### `labelbox_importer.py`
Unified Labelbox import with:
- Multiple source format support (COCO, raw predictions)
- Model-assisted labeling (MAL) import
- Debug mode with upload limits
- Error handling and recovery
- Progress monitoring

### `cleanup.py`
Utility for managing disk space:
- Removes old model checkpoint files
- Keeps specified number of recent files
- Provides disk space usage statistics
- Supports dry-run mode for preview

## Security Features

### Environment Variable Management
- API keys and credentials are stored in `.env` files (not committed to version control)
- The `setup_env.sh` script provides interactive setup with proper file permissions
- Automatic validation ensures credentials are properly configured

### CI/CD Security Checks
- GitHub Actions workflow scans for accidentally committed credentials
- Prevents pushing `.env` files to the repository
- Scans dependencies for known vulnerabilities
- Ensures code follows security best practices

## Debug Mode

All scripts support debug mode to limit processing and provide additional logging:

### Debug Flags

1. **Training Debug**:
   ```bash
   python train.py --debug
   ```
   Limits to 10 training steps per epoch with extra logging.

2. **Inference Debug**:
   ```bash
   python inference.py --debug
   ```
   Processes only 10 frames per video.

3. **Labelbox Import Debug**:
   ```bash
   python labelbox_importer.py --debug --limit 5
   ```
   Limits uploads to 5 annotated images.

4. **Visualization Debug**:
   ```bash
   python fix_visualize_annotations.py --debug --limit 3
   ```
   Processes only 3 images with extra annotation information.

### Debug Output
With debug mode enabled:
- More verbose console output
- Additional validation checks
- Detailed error messages
- Performance statistics

## Troubleshooting

### Common Issues

1. **Model Training Issues**:
   - **Symptom**: Out of memory (OOM) errors
   - **Fix**: Reduce `BATCH_SIZE_PER_GPU` or increase `GRADIENT_ACCUMULATION_STEPS`

2. **Inference Issues**:
   - **Symptom**: Empty predictions
   - **Fix**: Lower `CONFIDENCE_THRESHOLD` in config.py
   - **Symptom**: Process hangs
   - **Fix**: Try reducing `BATCH_SIZE_INFERENCE`

3. **Visualization Issues**:
   - **Symptom**: No annotations appear
   - **Fix**: Use `--only-with-annotations` and check file paths
   - **Symptom**: Annotations too small
   - **Fix**: Use `--scale 50` to increase visibility

4. **Labelbox Upload Issues**:
   - **Symptom**: Authentication errors
   - **Fix**: Verify API key in .env file or environment
   - **Symptom**: Upload times out
   - **Fix**: Use `--debug` with smaller batch to test connectivity
   - **Symptom**: Upload appears stuck
   - **Fix**: The script now has a 30-minute timeout. Check the logs for network issues. You can also check if the upload completed directly on Labelbox.
   - **Symptom**: "No internet connection available" error
   - **Fix**: Check your network connection and firewall settings. The script now checks connectivity before attempting uploads.

5. **Environment Setup Issues**:
   - **Symptom**: "API key not set" errors
   - **Fix**: Run `setup_env.sh` or manually create `.env` file
   - **Symptom**: Permission errors
   - **Fix**: Check `.env` file permissions with `chmod 600 .env`

### Log Files
Check these log files for detailed error information:
- `outputs/logs/training.log`: Training issues
- `outputs/logs/inference.log`: Prediction errors
- `outputs/logs/coco_export.log`: Format conversion problems
- `outputs/logs/labelbox_import.log`: Upload issues

## Performance Optimization

### Memory Usage
Control memory usage through these parameters in `config.py`:
- `BATCH_SIZE_PER_GPU`: Directly affects GPU memory usage
- `GRADIENT_ACCUMULATION_STEPS`: Higher values use less memory
- `IMAGE_SIZE`: Lower values reduce memory requirements

### Speed Optimization
For faster processing:
- Increase `NUM_GPUS` for parallel processing
- Adjust `BATCH_SIZE_INFERENCE` based on available memory
- Set `BATCH_SIZE_PER_GPU` higher for more efficient training

### Disk Space Management
- Use `cleanup.py` to manage model checkpoint files
- Set a reasonable `--keep` value based on your storage constraints

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Submit pull request

When contributing, ensure:
- Consistent debug mode implementation
- Comprehensive error handling
- Documentation of parameters
- Test coverage for new features

## License

[Specify License]

## Contact

[Add Contact Information] 