import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Debug configuration
DEBUG = {
    "ENABLED": os.environ.get("DEBUG_MODE", "0") == "1",  # Enable via env var or code
    "UPLOAD_LIMIT": 10,            # Maximum items to process in debug mode
    "LOG_LEVEL": "DEBUG",          # Logging level for debug mode
    "VERBOSE_OUTPUT": True,        # Whether to print additional information
    "SAVE_INTERMEDIATES": True     # Whether to save intermediate results
}

# Dataset paths
DATASET_ROOT = Path("/data/datasets/HurricaneVidNet_Dataset")
IMAGES_DIR = DATASET_ROOT / "images"
ANNOTATIONS_FILE = DATASET_ROOT / "output.json"
CLASS_MAPPING_FILE = DATASET_ROOT / "class_mapping.csv"

# Pipeline paths
PIPELINE_ROOT = Path("/data/datasets/dataset_pipeline")
OUTPUT_DIR = PIPELINE_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
LOGS_DIR = OUTPUT_DIR / "logs"

# Model configuration
NUM_CLASSES = 7  # Updated to match model initialization
DEVICE = "cuda"
NUM_GPUS = 1  # Using just 1 GPU to avoid memory conflicts
GPU_ID = 7  # Use GPU 7 which has most available memory
BATCH_SIZE_PER_GPU = 1  # Reduced to accommodate large images
GRADIENT_ACCUMULATION_STEPS = 8  # Increased to maintain effective batch size
NUM_EPOCHS = 50
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01

# Training configuration
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
RANDOM_SEED = 42
IMAGE_SIZE = 512  # Reduced image size to save memory

# Mask2Former specific config
MODEL_CHECKPOINT = "facebook/mask2former-swin-large-coco-instance"
NUM_QUERIES = 100
MATCHER_TOPK = 5

# Inference configuration
CONFIDENCE_THRESHOLD = 0.5
OVERLAP_THRESHOLD = 0.5
BATCH_SIZE_INFERENCE = 4

# Labelbox configuration
LABELBOX_API_KEY = os.getenv("LABELBOX_API_KEY")
LABELBOX_PROJECT_ID = os.getenv("LABELBOX_PROJECT_ID")
LABELBOX_BATCH_SIZE = 50

# Visualization configuration
VISUALIZATION = {
    "SAVE_DIR": PREDICTIONS_DIR / "visualized_annotations",
    "HIGHLIGHT_COLOR": (0, 0, 255),  # Red highlight for annotations
    "DEFAULT_SCALE": 1.0,            # Default scaling for annotations
    "DEFAULT_LIMIT": 20              # Default number of images to visualize
}

# Class mapping
CLASS_NAMES = [
    "__background__",
    "Building-No-Damage",
    "Building-Minor-Damage",
    "Building-Major-Damage",
    "Building-Total-Destruction"
]

CLASS_COLORS = {
    "__background__": (0, 0, 0),
    "Building-No-Damage": (50, 255, 132),
    "Building-Minor-Damage": (214, 255, 50),
    "Building-Major-Damage": (255, 50, 50),
    "Building-Total-Destruction": (50, 132, 255)
}

# Error handling configuration
ERRORS = {
    "MAX_RETRIES": 3,            # Maximum retry attempts for operations
    "RETRY_DELAY": 5,            # Seconds to wait between retries
    "SKIP_FAILED_FRAMES": True,  # Whether to skip frames that fail processing
    "LOG_EXCEPTIONS": True       # Whether to log full exception tracebacks
}

# Create necessary directories
for directory in [OUTPUT_DIR, MODEL_DIR, PREDICTIONS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True) 