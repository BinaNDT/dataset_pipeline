# Building Damage Assessment Pipeline (Consolidated)

A comprehensive pipeline for detecting and classifying building damage in aerial imagery using Mask2Former. This pipeline handles the complete workflow from training through inference to annotation upload.

> **Note:** This codebase has been consolidated to remove redundant files and improve organization. Unused files have been moved to the `archived/` directory.

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
./scripts/setup_env.sh
```

You can also set these environment variables directly in your shell:
```bash
export LABELBOX_API_KEY="your_api_key"
export LABELBOX_PROJECT_ID="your_project_id"
```

For detailed security guidelines on handling credentials, see [Security Documentation](docs/security.md).

## Directory Structure (Consolidated)

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
# Enhanced visualization with scaled annotations
python visualize_annotations.py --only-with-annotations --scale 10 --highlight-border
```

### Step 6: Analyze Prediction Quality
Check prediction statistics:

```bash
python analyze_predictions.py
```

### Step 7: Upload Images to Labelbox
Before uploading annotations, you must first upload the images to Labelbox:

```bash
# Upload images with debug mode (limited number)
python upload_images_to_labelbox.py --debug --limit 5

# Upload all images
python upload_images_to_labelbox.py
```

This step is required because Labelbox needs to have the images in its system before you can add annotations to them.

### Step 8: Upload Annotations to Labelbox
Import predictions to Labelbox for human review:

```bash
# From COCO format using polygon annotations (default)
python labelbox_importer.py --debug --limit 5

# From COCO format using mask annotations
python labelbox_importer.py --format mask --debug --limit 5

# From raw predictions
python labelbox_importer.py --source predictions

# Production run with all annotations
python labelbox_importer.py
```

**Chunked Uploading Options**:

For more reliable uploading, especially with large datasets:

```bash
# Use smaller chunks (50 annotations per chunk)
python labelbox_importer.py --chunk-size 50

# Resume upload from a specific chunk (e.g., if it failed at chunk 3)
python labelbox_importer.py --start-chunk 3

# Use URLs for mask images instead of binary masks
python labelbox_importer.py --format mask --use-mask-urls

# Use simplified upload mode for better performance
python labelbox_importer.py --simplified --chunk-size 10
```

**Using Simplified Mode**:
If you encounter issues with the standard upload process, try the simplified mode:

```bash
python labelbox_importer.py --simplified --chunk-size 10
```

The simplified mode uploads all chunks in parallel and monitors them together, rather than sequentially. This can be more efficient and less likely to get stuck, though it provides less detailed progress information during upload.

**Note**: The annotation format (`--format mask` or `--format polygon`) must match what's configured in your Labelbox project. If you get "AnnotationFormatMismatch" errors, try switching to the other format or changing your project's ontology settings in Labelbox.

### Step 9: Clean Up (Optional)
Manage disk space by removing old checkpoints:

```bash
# Keep only the latest checkpoint
python scripts/cleanup.py

# Keep the 3 most recent checkpoints
python scripts/cleanup.py --keep 3

# Preview what would be deleted without deleting
python scripts/cleanup.py --dry-run
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
   python visualize_annotations.py --debug --limit 3
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
   - **Symptom**: Import gets stuck at 81% or another percentage
   - **Fix**: This is a common Labelbox issue. Try these options:
     - Use smaller chunks: `--chunk-size 10` 
     - Use simplified mode: `--simplified --chunk-size 10`
     - Check Labelbox UI to see if annotations are appearing despite the stuck progress
     - If a particular chunk is failing, you can resume from that point with `--start-chunk N`
   - **Symptom**: MAL import is too slow
   - **Fix**: Try using `--simplified` mode and/or `--use-mask-urls` with mask URLs hosted on a cloud storage service like Google Cloud Storage for faster processing.
   - **Symptom**: No progress information displayed
   - **Fix**: The improved script now shows detailed progress information. If it's still not showing, check the logs in `outputs/logs/labelbox_import.log`.

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
- Use `scripts/cleanup.py` to manage model checkpoint files
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

## Notes on Code Consolidation

- The `archived/` directory contains old versions and redundant files that were part of the original codebase.
- Shell scripts have been moved to the `scripts/` directory.
- Test and example files have been organized into appropriate directories.
- The visualization functionality was consolidated from multiple files into a single enhanced version.
- Redundant Labelbox importers were consolidated into a single `labelbox_importer.py` file with all functionality.
