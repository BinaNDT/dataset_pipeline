import json
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse

# Setup path to import from config
sys.path.append(str(Path(__file__).parent))
from config import *
from utils import setup_logging, timer, debug_info, validate_inputs

def main():
    """
    Analyze the predictions to identify issues with annotation coordinates.
    
    This script analyzes the prediction data from the model to identify any
    potential issues with annotations, masks, or scaling. It performs:
    1. Frame-level analysis of predictions
    2. Class distribution analysis
    3. Confidence score analysis
    4. Mask and polygon size analysis
    5. Potential issue detection
    """
    parser = argparse.ArgumentParser(description="Analyze model predictions and identify potential issues")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with more detailed analysis")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of frames to analyze per video")
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging("analyze_predictions", debug=args.debug)
    
    logger.info("Analyzing prediction data...")
    
    # Check for the predictions file
    predictions_file = PREDICTIONS_DIR / 'predictions.json'
    if not predictions_file.exists():
        logger.error(f"Predictions file not found at {predictions_file}")
        return
    
    # Load predictions
    with timer("Loading predictions", logger):
        logger.info(f"Loading predictions from {predictions_file}")
        with open(predictions_file, 'r') as f:
            predictions_by_video = json.load(f)
    
    # Also load COCO file for comparison
    coco_file = PREDICTIONS_DIR / 'predictions_coco.json'
    if coco_file.exists():
        with timer("Loading COCO data", logger):
            logger.info(f"Loading COCO file from {coco_file}")
            with open(coco_file, 'r') as f:
                coco_data = json.load(f)
    else:
        coco_data = None
    
    # Statistics counters
    total_frames = 0
    frames_with_predictions = 0
    total_predictions = 0
    class_distribution = defaultdict(int)
    confidence_values = []
    mask_sizes = []
    
    # For COCO comparison
    polygon_sizes = []
    bbox_sizes = []
    
    # Analyze each video and frame
    for video_name, frame_predictions in predictions_by_video.items():
        logger.info(f"Analyzing video: {video_name} ({len(frame_predictions)} frames)")
        
        # Apply limit in debug mode
        if args.limit and args.debug:
            frame_predictions = frame_predictions[:args.limit]
            logger.debug(f"Debug mode: Limited to first {args.limit} frames")
        
        for frame_idx, frame_pred in enumerate(frame_predictions):
            total_frames += 1
            
            # Skip frames with errors
            if 'error' in frame_pred:
                logger.debug(f"Skipping frame {frame_idx} with error: {frame_pred['error']}")
                continue
            
            predictions = frame_pred.get('predictions', [])
            if predictions:
                frames_with_predictions += 1
                total_predictions += len(predictions)
                
                # Analyze each prediction
                for pred_idx, pred in enumerate(predictions):
                    # Class distribution
                    class_id = pred['class_id']
                    class_name = CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else f"Unknown ({class_id})"
                    class_distribution[class_name] += 1
                    
                    # Confidence values
                    if 'confidence' in pred:
                        confidence_values.append(pred['confidence'])
                    
                    # Mask dimensions
                    mask = np.array(pred['mask'])
                    mask_sizes.append(mask.shape)
                    
                    # Mask statistics - what proportion of the image is covered by the mask?
                    mask_area = mask.sum()
                    mask_percentage = mask_area / (mask.shape[0] * mask.shape[1]) * 100
                    
                    # Print detailed info for the first few predictions of each video
                    if (frame_idx < 5 and pred_idx < 3) or args.debug:
                        logger.debug(f"Frame {frame_idx}, Pred {pred_idx}:")
                        logger.debug(f"  Class: {class_name} (ID: {class_id}), Confidence: {pred.get('confidence', 'N/A')}")
                        logger.debug(f"  Mask shape: {mask.shape}, Area: {mask_area} pixels ({mask_percentage:.2f}% of image)")
                        
                        # Find the bounding coordinates of the mask
                        if mask_area > 0:
                            y_indices, x_indices = np.where(mask > 0)
                            x_min, x_max = x_indices.min(), x_indices.max()
                            y_min, y_max = y_indices.min(), y_indices.max()
                            logger.debug(f"  Mask bounds: x=[{x_min}-{x_max}], y=[{y_min}-{y_max}], size={x_max-x_min+1}x{y_max-y_min+1}")
    
    # COCO file analysis
    if coco_data:
        logger.info("COCO Data Analysis:")
        coco_info = {
            "Images": len(coco_data['images']),
            "Annotations": len(coco_data['annotations']),
            "Categories": [cat['name'] for cat in coco_data['categories']]
        }
        debug_info(coco_info, logger, "COCO Dataset Summary")
        
        # Analyze segmentation points and bounding boxes
        sample_size = 10 if not args.debug else min(50, len(coco_data['annotations']))
        for anno_idx, anno in enumerate(coco_data['annotations'][:sample_size]):  # Sample the first few
            if args.debug:
                logger.debug(f"Annotation {anno_idx+1}:")
                logger.debug(f"  Image ID: {anno['image_id']}, Category: {anno['category_id']}, Score: {anno.get('score', 'N/A')}")
                logger.debug(f"  Bounding Box (x,y,w,h): {anno['bbox']}")
                logger.debug(f"  Segmentation points: {anno['segmentation'][0][:8]}... (total: {len(anno['segmentation'][0])} points)")
            
            # Calculate polygon size
            points = np.array(anno['segmentation'][0]).reshape(-1, 2)
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            if args.debug:
                logger.debug(f"  Polygon bounds: x=[{x_min}-{x_max}], y=[{y_min}-{y_max}], size={x_max-x_min}x{y_max-y_min}")
            
            polygon_sizes.append((x_max-x_min, y_max-y_min))
            bbox_sizes.append((anno['bbox'][2], anno['bbox'][3]))  # width, height
    
    # Summary statistics
    logger.info("Summary Statistics:")
    stats = {
        "Total frames analyzed": total_frames,
        "Frames with predictions": f"{frames_with_predictions} ({frames_with_predictions/total_frames*100:.1f}%)",
        "Total predictions": total_predictions,
        "Average predictions per frame": f"{total_predictions/frames_with_predictions:.2f}" if frames_with_predictions else "N/A"
    }
    debug_info(stats, logger, "Prediction Statistics")
    
    # Class distribution
    class_stats = {
        class_name: f"{count} ({count/total_predictions*100:.1f}%)" 
        for class_name, count in class_distribution.items()
    }
    debug_info(class_stats, logger, "Class Distribution")
    
    if confidence_values:
        confidence_stats = {
            "Min": f"{min(confidence_values):.4f}",
            "Max": f"{max(confidence_values):.4f}",
            "Mean": f"{np.mean(confidence_values):.4f}",
            "Median": f"{np.median(confidence_values):.4f}"
        }
        debug_info(confidence_stats, logger, "Confidence Values")
    
    logger.info("Analysis complete!")
    
    # Recommendations based on findings
    issues_found = False
    
    if coco_data and polygon_sizes:
        avg_width, avg_height = np.mean(polygon_sizes, axis=0)
        logger.info(f"Average polygon size: {avg_width:.1f}x{avg_height:.1f} pixels")
        
        if avg_width < 100 or avg_height < 100:
            issues_found = True
            logger.warning("Polygon sizes are very small. This might indicate a scaling issue during mask-to-polygon conversion.")
            logger.warning("Suggestion: Check if the binary_mask_to_polygon function is correctly handling mask dimensions.")
    
    if coco_data and all(x[0] < 100 and x[1] < 100 for x in polygon_sizes[:5]):
        issues_found = True
        logger.warning("All polygons appear to be in the top-left corner of the image.")
        logger.warning("This could indicate:")
        logger.warning("  1. A problem with the mask generation during prediction")
        logger.warning("  2. A coordinate system mismatch during polygon creation")
        logger.warning("  3. Issues with the original data or model training")
        logger.warning("Try inspecting the original image masks in the predictions.json file")
    
    if not issues_found:
        logger.info("No major issues detected in predictions")

if __name__ == "__main__":
    main() 