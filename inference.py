import torch
import torch.nn.functional as F
from transformers import Mask2FormerForUniversalSegmentation
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import json
from concurrent.futures import ThreadPoolExecutor
import sys

sys.path.append(str(Path(__file__).parent))
from config import *
from utils.dataset import get_transform

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOGS_DIR / 'inference.log'),
            logging.StreamHandler()
        ]
    )

class Predictor:
    def __init__(self, checkpoint_path: Path, device_id: int = 0):
        self.device = torch.device(f'cuda:{device_id}')
        self.transform = get_transform('test')
        
        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            MODEL_CHECKPOINT,
            state_dict=checkpoint['model_state_dict']
        ).to(self.device)
        self.model.eval()
        
        logging.info(f"Loaded model from {checkpoint_path}")
    
    @torch.no_grad()
    def predict_image(self, image_path: Path) -> dict:
        """Predict masks for a single image"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Run inference
        outputs = self.model(image_tensor)
        
        # Process predictions
        masks_logits = outputs.masks_queries_logits[0]  # (num_queries, H, W)
        class_logits = outputs.class_queries_logits[0]  # (num_queries, num_classes)
        
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
                    predictions.append({
                        'class_id': int(class_ids[query_idx]),
                        'class_name': CLASS_NAMES[int(class_ids[query_idx])],
                        'confidence': float(class_scores[query_idx]),
                        'mask': mask.cpu().numpy().astype(bool)
                    })
        
        return {
            'image_path': str(image_path),
            'predictions': predictions
        }

def process_video_frames(predictor: Predictor, video_name: str) -> list:
    """Process all frames from a video"""
    video_dir = IMAGES_DIR / video_name
    frame_paths = sorted(video_dir.glob('*.png'))
    
    predictions = []
    for frame_path in tqdm(frame_paths, desc=f"Processing {video_name}"):
        try:
            pred = predictor.predict_image(frame_path)
            predictions.append(pred)
        except Exception as e:
            logging.error(f"Error processing {frame_path}: {str(e)}")
    
    return predictions

def main():
    setup_logging()
    
    # Find latest checkpoint
    checkpoints = sorted(MODEL_DIR.glob('checkpoint_epoch_*.pt'))
    if not checkpoints:
        raise ValueError("No checkpoints found in model directory")
    latest_checkpoint = checkpoints[-1]
    
    # Create predictors for each GPU
    num_gpus = torch.cuda.device_count()
    predictors = [Predictor(latest_checkpoint, i) for i in range(num_gpus)]
    
    # Get list of videos
    video_dirs = [d for d in IMAGES_DIR.iterdir() if d.is_dir()]
    
    # Distribute videos across GPUs
    predictions_by_video = {}
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for i, video_dir in enumerate(video_dirs):
            predictor = predictors[i % num_gpus]
            future = executor.submit(process_video_frames, predictor, video_dir.name)
            futures.append((video_dir.name, future))
        
        # Collect results
        for video_name, future in futures:
            try:
                predictions_by_video[video_name] = future.result()
            except Exception as e:
                logging.error(f"Error processing video {video_name}: {str(e)}")
    
    # Save predictions
    output_file = PREDICTIONS_DIR / 'predictions.json'
    with open(output_file, 'w') as f:
        json.dump(predictions_by_video, f)
    
    logging.info(f"Saved predictions to {output_file}")

if __name__ == '__main__':
    main() 