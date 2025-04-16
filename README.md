# Building Damage Assessment Pipeline

A comprehensive pipeline for detecting and classifying building damage in aerial imagery using Mask2Former. This pipeline handles the complete workflow from training through inference to annotation upload.

## Overview

This pipeline automates the process of:
1. Fine-tuning Mask2Former on custom building damage datasets
2. Running inference on large-scale video frame collections
3. Converting predictions to Labelbox format for human review
4. Managing multi-GPU training and inference

### Damage Classification Categories

The model classifies buildings into four damage levels:
- No Damage: Buildings with no visible structural damage
- Minor Damage: Superficial damage, intact structural elements
- Major Damage: Significant structural damage but partially standing
- Total Destruction: Complete collapse or destruction

## Prerequisites

### Hardware Requirements
- CUDA-capable GPU(s)
- Minimum 16GB GPU memory per device
- Sufficient storage for dataset and model checkpoints

### Software Requirements
- Python 3.8+
- CUDA 11.3+
- PyTorch 2.0+

### Dataset Requirements
- Images in PNG format
- COCO format annotations
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
```bash
export LABELBOX_API_KEY="your_api_key"
export LABELBOX_PROJECT_ID="your_project_id"
```

## Directory Structure

```
dataset_pipeline/
├── config.py                 # Configuration parameters
├── train.py                 # Training script
├── inference.py             # Inference script
├── labelbox_uploader.py     # Labelbox conversion and upload
├── utils/
│   └── dataset.py          # Dataset utilities
├── outputs/                 # Generated during runtime
│   ├── models/             # Saved model checkpoints
│   ├── predictions/        # Model predictions
│   └── logs/              # Training and inference logs
└── requirements.txt        # Python dependencies
```

## Configuration

### Key Configuration Parameters (config.py)

```python
# GPU Configuration
NUM_GPUS = 2                  # Number of GPUs to use
BATCH_SIZE_PER_GPU = 2        # Batch size per GPU
GRADIENT_ACCUMULATION_STEPS = 4  # Steps before gradient update

# Training Parameters
NUM_EPOCHS = 50
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
IMAGE_SIZE = 1024

# Model Configuration
MODEL_CHECKPOINT = "facebook/mask2former-swin-large-coco-instance"
NUM_QUERIES = 100
```

### Modifying Configuration

1. GPU Usage:
   - Adjust `NUM_GPUS` based on availability
   - Modify `BATCH_SIZE_PER_GPU` based on GPU memory
   - Adjust `GRADIENT_ACCUMULATION_STEPS` to maintain effective batch size

2. Training Parameters:
   - Modify `NUM_EPOCHS` for longer/shorter training
   - Adjust `LEARNING_RATE` for training stability
   - Change `IMAGE_SIZE` based on input resolution

## Usage

### 1. Training

Train the model on your annotated dataset:

```bash
python train.py
```

Training features:
- Multi-GPU training using DistributedDataParallel
- Automatic mixed precision training
- Gradient accumulation for larger effective batch sizes
- Checkpoint saving based on validation loss
- WandB integration for monitoring
- Cosine learning rate scheduling with warmup

Monitor training:
- Check `outputs/logs/training.log` for progress
- View metrics in WandB dashboard
- Monitor GPU usage with `nvidia-smi`

### 2. Inference

Run inference on video frames:

```bash
python inference.py
```

Inference features:
- Automatic GPU distribution of workload
- Parallel processing of multiple videos
- Confidence thresholding for predictions
- Structured JSON output format

### 3. Labelbox Upload

Upload predictions for review:

```bash
python labelbox_uploader.py
```

Features:
- Automatic mask to polygon conversion
- Confidence score preservation
- Batch uploading with error recovery
- Local backup of annotations

## Monitoring and Logging

### Log Files
- `training.log`: Training progress, losses, metrics
- `inference.log`: Prediction progress, errors
- `labelbox_upload.log`: Upload status, errors

### GPU Monitoring
Check GPU usage:
```bash
nvidia-smi
```

### WandB Integration
- Real-time loss tracking
- Learning rate monitoring
- GPU utilization metrics
- Model parameter logging

## Error Handling and Recovery

### Training Recovery
- Checkpoints saved at best validation loss
- Resume training from latest checkpoint
- Automatic multi-GPU error handling

### Inference Recovery
- Per-frame error logging
- Continued processing on errors
- Failed frame tracking

### Upload Recovery
- Batch-wise upload verification
- Local backup before upload
- Error state logging

## Performance Optimization

### GPU Memory Usage
- Adjust `BATCH_SIZE_PER_GPU` based on available memory
- Monitor with `nvidia-smi`
- Use gradient accumulation for memory efficiency

### Processing Speed
- Distribute workload across available GPUs
- Parallel video processing
- Optimized data loading

## Troubleshooting

### Common Issues

1. Out of Memory (OOM):
   - Reduce `BATCH_SIZE_PER_GPU`
   - Increase `GRADIENT_ACCUMULATION_STEPS`
   - Reduce `IMAGE_SIZE`

2. GPU Issues:
   - Check GPU availability with `nvidia-smi`
   - Verify CUDA installation
   - Monitor GPU temperature

3. Training Issues:
   - Check learning rate
   - Verify dataset loading
   - Monitor validation loss

4. Labelbox Upload Issues:
   - Verify API credentials
   - Check network connection
   - Review error logs

### Debug Mode

Add debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Submit pull request

## License

[Specify License]

## Contact

[Specify Contact Information]

## Acknowledgments

- Facebook AI Research for Mask2Former
- Labelbox for annotation platform
- [Other acknowledgments] 