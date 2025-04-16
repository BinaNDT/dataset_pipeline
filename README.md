# Building Damage Assessment Pipeline

This pipeline automates the process of detecting and classifying building damage in video frames using Mask2Former. It includes training, inference, and uploading predictions to Labelbox for review.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
export LABELBOX_API_KEY="your_api_key"
export LABELBOX_PROJECT_ID="your_project_id"
```

## Directory Structure

```
dataset_pipeline/
├── config.py                 # Configuration parameters
├── train.py                 # Training script
├── inference.py            # Inference script
├── labelbox_uploader.py    # Labelbox conversion and upload
├── utils/
│   └── dataset.py         # Dataset utilities
└── outputs/               # Generated during runtime
    ├── models/           # Saved model checkpoints
    ├── predictions/      # Model predictions
    └── logs/            # Training and inference logs
```

## Usage

### 1. Training

Train the model on your annotated dataset:

```bash
python train.py
```

This will:
- Load your COCO format annotations
- Fine-tune Mask2Former using multi-GPU training
- Save checkpoints to `outputs/models/`
- Log metrics to WandB (optional)

### 2. Inference

Run inference on all video frames:

```bash
python inference.py
```

This will:
- Load the best checkpoint
- Run inference across all frames using multiple GPUs
- Save predictions to `outputs/predictions/predictions.json`

### 3. Upload to Labelbox

Convert and upload predictions to Labelbox:

```bash
python labelbox_uploader.py
```

This will:
- Convert predictions to Labelbox format
- Create/update project ontology
- Upload predictions as prelabels
- Save a backup to `outputs/predictions/labelbox_annotations.json`

## Configuration

Edit `config.py` to modify:
- Dataset paths
- Model parameters
- Training settings
- Inference thresholds
- GPU settings

## Damage Categories

The model classifies buildings into four damage levels:
1. No Damage
2. Minor Damage
3. Major Damage
4. Total Destruction

## Multi-GPU Support

The pipeline automatically utilizes all available GPUs for both training and inference:
- Training uses DistributedDataParallel (DDP)
- Inference distributes videos across GPUs
- Default batch sizes and learning rates are optimized for multi-GPU setup

## Monitoring

- Training progress is logged to `outputs/logs/training.log`
- Inference results are logged to `outputs/logs/inference.log`
- Labelbox upload status is logged to `outputs/logs/labelbox_upload.log`
- (Optional) Real-time metrics are tracked in WandB

## Error Handling

The pipeline includes robust error handling:
- Checkpointing for training recovery
- Batch-wise error logging during inference
- Failed frame tracking
- Labelbox upload verification

## Contributing

Feel free to submit issues and enhancement requests! 