import json
import numpy as np
import datetime
import logging
from pathlib import Path
import sys
from tqdm import tqdm
import cv2

# Setup path to import from config
sys.path.append(str(Path(__file__).parent))
from config import *

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOGS_DIR / 'coco_export.log'),
            logging.StreamHandler()
        ]
    )

def binary_mask_to_polygon(binary_mask):
    """Convert a binary mask to polygon format"""
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    polygons = []
    for contour in contours:
        # Convert contour to polygon
        segmentation = contour.flatten().tolist()
        
        # Skip invalid polygons
        if len(segmentation) < 6:  # Need at least 3 points (x,y)
            continue
            
        polygons.append(segmentation)
        
    return polygons

def create_coco_annotation(
    annotation_id, image_id, category_id, segmentation, 
    area, bbox, iscrowd=0, confidence=None
):
    """Create a COCO annotation dictionary"""
    annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": segmentation,
        "area": float(area),
        "bbox": [float(x) for x in bbox],
        "iscrowd": iscrowd
    }
    
    # Add confidence if available
    if confidence is not None:
        annotation["score"] = float(confidence)
        
    return annotation

def main():
    setup_logging()
    
    # Check for the predictions file
    predictions_file = PREDICTIONS_DIR / 'predictions.json'
    if not predictions_file.exists():
        logging.error(f"Predictions file not found at {predictions_file}")
        return
    
    # Load predictions
    logging.info(f"Loading predictions from {predictions_file}")
    with open(predictions_file, 'r') as f:
        predictions_by_video = json.load(f)
    
    # Define COCO categories (based on your class names)
    categories = []
    for i, class_name in enumerate(CLASS_NAMES[1:], 1):  # Skip background class
        categories.append({
            "id": i,
            "name": class_name,
            "supercategory": "Building"
        })
    
    # Initialize COCO format
    coco_data = {
        "info": {
            "description": "Building Damage Detection Dataset",
            "url": "",
            "version": "1.0",
            "year": datetime.datetime.now().year,
            "contributor": "Dataset Pipeline",
            "date_created": datetime.datetime.now().strftime("%Y/%m/%d")
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "categories": categories,
        "images": [],
        "annotations": []
    }
    
    # Track annotation and image IDs
    annotation_id = 1
    image_id = 1
    
    # Process each video and frame
    total_frames = sum(len(frames) for frames in predictions_by_video.values())
    processed_frames = 0
    annotation_count = 0
    
    logging.info(f"Processing {len(predictions_by_video)} videos with {total_frames} frames")
    
    for video_name, frame_predictions in tqdm(predictions_by_video.items(), desc="Processing videos"):
        for frame_pred in frame_predictions:
            image_path = frame_pred['image_path']
            
            # Skip frames with errors
            if 'error' in frame_pred:
                logging.warning(f"Skipping frame with error: {image_path}")
                processed_frames += 1
                continue
            
            # Get image width and height
            image = cv2.imread(image_path)
            if image is None:
                logging.warning(f"Could not read image: {image_path}")
                processed_frames += 1
                continue
                
            height, width = image.shape[:2]
            
            # Add image to COCO
            coco_data["images"].append({
                "id": image_id,
                "file_name": Path(image_path).name,
                "width": width,
                "height": height,
                "date_captured": "",
                "license": 1,
                "coco_url": "",
                "flickr_url": "",
                "video_name": video_name
            })
            
            # Process predictions for this frame
            for pred in frame_pred['predictions']:
                # Get class ID (add 1 to skip background class)
                class_id = pred['class_id'] if pred['class_id'] > 0 else pred['class_id'] + 1
                
                # Convert mask to numpy array if it's not already
                mask = np.array(pred['mask']).astype(np.uint8)
                
                # Calculate area
                area = float(mask.sum())
                if area < 1:
                    continue
                
                # Convert mask to polygons
                polygons = binary_mask_to_polygon(mask)
                
                if not polygons:
                    continue
                    
                # Find bounding box
                y_indices, x_indices = np.where(mask > 0)
                if len(y_indices) == 0 or len(x_indices) == 0:
                    continue
                    
                # COCO format is [x, y, width, height]
                x_min, x_max = x_indices.min(), x_indices.max()
                y_min, y_max = y_indices.min(), y_indices.max()
                bbox = [
                    float(x_min), 
                    float(y_min), 
                    float(x_max - x_min), 
                    float(y_max - y_min)
                ]
                
                # Add annotation to COCO
                coco_annotation = create_coco_annotation(
                    annotation_id=annotation_id,
                    image_id=image_id,
                    category_id=class_id,
                    segmentation=polygons,
                    area=area,
                    bbox=bbox,
                    confidence=pred.get('confidence', 1.0)
                )
                
                coco_data["annotations"].append(coco_annotation)
                annotation_id += 1
                annotation_count += 1
            
            # Update IDs for next frame
            image_id += 1
            processed_frames += 1
            
            # Log progress periodically
            if processed_frames % 100 == 0:
                logging.info(f"Processed {processed_frames}/{total_frames} frames with {annotation_count} annotations")
    
    # Save COCO file
    coco_file = PREDICTIONS_DIR / 'predictions_coco.json'
    with open(coco_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    logging.info(f"Successfully exported {annotation_count} annotations for {processed_frames} frames")
    logging.info(f"COCO file saved to {coco_file}")

if __name__ == "__main__":
    main() 