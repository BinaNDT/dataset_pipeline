import torch
import torch.nn.functional as F
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import json
from concurrent.futures import ThreadPoolExecutor
import sys
import gc
import traceback

sys.path.append(str(Path(__file__).parent))
from config import *
from utils.dataset import get_transform

# Debug mode will process fewer videos and use only one GPU
DEBUG_MODE = True

def setup_logging():
    """Set up logging configuration"""
    # Create logs directory if it doesn't exist
    logs_dir = Path(LOGS_DIR)
    logs_dir.mkdir(exist_ok=True, parents=True)
    
    # Set log level
    log_level = logging.DEBUG if DEBUG_MODE else logging.INFO
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logs_dir / 'inference.log'),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Inference process started (DEBUG_MODE: {DEBUG_MODE})")

class Predictor:
    def __init__(self, checkpoint_path: Path, device_id: int = 0):
        self.device = torch.device(f'cuda:{device_id}')
        self.transform = get_transform('test')
        
        # Load model
        logging.info(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create model with correct config
        config = Mask2FormerConfig.from_pretrained(
            MODEL_CHECKPOINT,
            num_labels=NUM_CLASSES,
            num_queries=NUM_QUERIES
        )
        
        # Initialize model with config then load weights
        self.model = Mask2FormerForUniversalSegmentation(config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logging.info(f"Successfully loaded model from {checkpoint_path}")
    
    @torch.no_grad()
    def predict_image(self, image_path: Path) -> dict:
        """Predict masks for a single image"""
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Run inference
            outputs = self.model(image_tensor)
            
            # Process predictions with better error handling
            try:
                masks_logits = outputs.masks_queries_logits[0]  # (num_queries, H, W)
                class_logits = outputs.class_queries_logits[0]  # (num_queries, num_classes)
                
                # Log tensor shapes for debugging
                logging.debug(f"masks_logits shape: {masks_logits.shape}, class_logits shape: {class_logits.shape}")
                
                # Get class predictions and probabilities
                class_probs = F.softmax(class_logits, dim=-1)
                class_ids = torch.argmax(class_probs, dim=-1)
                class_scores = torch.max(class_probs, dim=-1)[0]
                
                # Threshold predictions
                mask_probs = masks_logits.sigmoid()
                valid_mask = class_scores > CONFIDENCE_THRESHOLD
                
                predictions = []
                for query_idx in range(len(class_ids)):
                    if valid_mask[query_idx]:
                        mask = mask_probs[query_idx] > OVERLAP_THRESHOLD
                        if mask.sum() > 0:  # Only keep masks with non-zero area
                            # Convert mask to numpy boolean array
                            mask_np = mask.cpu().numpy().astype(bool)
                            class_id = int(class_ids[query_idx])
                            
                            # Verify class_id is valid
                            if class_id < len(CLASS_NAMES):
                                predictions.append({
                                    'class_id': class_id,
                                    'class_name': CLASS_NAMES[class_id],
                                    'confidence': float(class_scores[query_idx]),
                                    'mask': mask_np
                                })
                            else:
                                logging.warning(f"Invalid class ID {class_id} for {image_path.name}, max valid ID is {len(CLASS_NAMES)-1}")
            
            except IndexError as e:
                logging.error(f"Index error processing output for {image_path.name}: {e}")
                logging.debug(f"Output keys: {outputs.keys()}")
                logging.debug(f"Output shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in outputs.items()]}")
                return {
                    'image_path': str(image_path),
                    'predictions': [],
                    'error': f"Index error: {str(e)}"
                }
            
            logging.info(f"Processed {image_path.name}: Found {len(predictions)} predictions")
            return {
                'image_path': str(image_path),
                'predictions': predictions
            }
        except Exception as e:
            logging.error(f"Error processing {image_path}: {e}")
            return {
                'image_path': str(image_path),
                'predictions': [],
                'error': str(e)
            }

def process_video_frames(predictor: Predictor, video_name: str) -> list:
    """Process all frames from a video"""
    video_dir = IMAGES_DIR / video_name
    
    if not video_dir.exists():
        logging.error(f"Video directory not found: {video_dir}")
        return []
    
    frame_paths = sorted(video_dir.glob('*.png'))
    
    if not frame_paths:
        logging.warning(f"No frames found in {video_dir}")
        return []
    
    logging.info(f"Processing {video_name}: Found {len(frame_paths)} frames")
    
    # In debug mode, limit the number of frames
    if DEBUG_MODE:
        max_frames = 50  # Process only 50 frames per video in debug mode
        if len(frame_paths) > max_frames:
            logging.info(f"DEBUG MODE: Limiting {video_name} to {max_frames} frames (out of {len(frame_paths)})")
            frame_paths = frame_paths[:max_frames]
    
    predictions = []
    errors = 0
    
    for frame_path in tqdm(frame_paths, desc=f"Processing {video_name}"):
        try:
            pred = predictor.predict_image(frame_path)
            predictions.append(pred)
            
            # Clear GPU cache periodically
            if len(predictions) % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()
                
        except Exception as e:
            errors += 1
            logging.error(f"Error processing {frame_path}: {str(e)}")
            predictions.append({
                'image_path': str(frame_path),
                'predictions': [],
                'error': str(e)
            })
            
            # If too many errors, stop processing this video
            if errors > 10 and errors > len(frame_paths) * 0.1:  # More than 10% errors
                logging.error(f"Too many errors processing {video_name}, stopping after {len(predictions)} frames")
                break
    
    logging.info(f"Completed {video_name}: Processed {len(predictions)} frames, encountered {errors} errors")
    return predictions

def convert_numpy_for_json(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # Update for NumPy 2.0+ compatibility
    elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64, 
                          np.uint8, np.uint16, np.uint32, np.uint64, 
                          np.integer)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64, 
                         np.floating)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [convert_numpy_for_json(item) for item in obj]
    return obj

def main():
    setup_logging()
    
    # Find latest checkpoint
    checkpoints = sorted(MODEL_DIR.glob('checkpoint_epoch_*.pt'))
    if not checkpoints:
        raise ValueError("No checkpoints found in model directory")
    latest_checkpoint = checkpoints[-1]
    logging.info(f"Using checkpoint: {latest_checkpoint}")
    
    # Create output directory
    PREDICTIONS_DIR.mkdir(exist_ok=True, parents=True)
    
    # Create predictors for each GPU
    num_gpus = 1 if DEBUG_MODE else torch.cuda.device_count()
    logging.info(f"Using {num_gpus} GPU(s) for inference")
    
    try:
        # Create predictor(s)
        predictors = [Predictor(latest_checkpoint, GPU_ID if 'GPU_ID' in globals() else i) 
                     for i in range(num_gpus)]
        
        # Get list of videos
        video_dirs = [d for d in IMAGES_DIR.iterdir() if d.is_dir()]
        logging.info(f"Found {len(video_dirs)} video directories")
        
        # In debug mode, only process the first few directories
        if DEBUG_MODE:
            video_dirs = video_dirs[:2]  # Process only 2 videos in debug mode
            logging.info(f"DEBUG MODE: Processing only {len(video_dirs)} videos")
        
        # Distribute videos across GPUs
        predictions_by_video = {}
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for i, video_dir in enumerate(video_dirs):
                predictor = predictors[i % num_gpus]
                future = executor.submit(process_video_frames, predictor, video_dir.name)
                futures.append((video_dir.name, future))
            
            # Collect results
            for video_name, future in tqdm(futures, desc="Processing videos"):
                try:
                    predictions_by_video[video_name] = future.result()
                    logging.info(f"Completed processing video {video_name}")
                except Exception as e:
                    logging.error(f"Error processing video {video_name}: {str(e)}")
                    predictions_by_video[video_name] = []
        
        # Save predictions
        output_file = PREDICTIONS_DIR / 'predictions.json'
        logging.info(f"Saving predictions to {output_file}")
        try:
            # Convert all NumPy arrays to Python lists before serializing
            serializable_predictions = convert_numpy_for_json(predictions_by_video)
            with open(output_file, 'w') as f:
                json.dump(serializable_predictions, f)
            logging.info(f"Successfully saved predictions to {output_file}")
        except Exception as e:
            logging.error(f"Error during inference process: {str(e)}")
            traceback.print_exc()
            return False
        
        # Report prediction stats
        total_frames = sum(len(preds) for preds in predictions_by_video.values())
        total_predictions = sum(
            sum(1 for frame in video_preds if 'error' not in frame 
                for _ in frame['predictions']) 
            for video_preds in predictions_by_video.values()
        )
        logging.info(f"Processed {total_frames} frames with {total_predictions} total predictions")
        
    except Exception as e:
        logging.error(f"Error during inference process: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main() 