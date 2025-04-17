import json
import cv2
import numpy as np
from pathlib import Path
import sys
import argparse
import random
import os
from datetime import datetime

# Setup path to import from config
sys.path.append(str(Path(__file__).parent))
from config import *

def draw_polygon(image, points, color, label=None, thickness=2):
    """Draw a polygon on an image"""
    # Convert points to numpy array
    pts = np.array(points).reshape((-1, 2)).astype(np.int32)
    
    # Draw polygon
    cv2.polylines(image, [pts], True, color, thickness)
    
    # Add label if provided
    if label:
        # Calculate centroid
        centroid_x = int(np.mean(pts[:, 0]))
        centroid_y = int(np.mean(pts[:, 1]))
        
        # Get text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_size = cv2.getTextSize(label, font, font_scale, 1)[0]
        
        # Draw text background
        cv2.rectangle(
            image, 
            (centroid_x - 5, centroid_y - text_size[1] - 5), 
            (centroid_x + text_size[0] + 5, centroid_y + 5), 
            color, 
            -1
        )
        
        # Draw text
        cv2.putText(
            image, 
            label, 
            (centroid_x, centroid_y), 
            font, 
            font_scale, 
            (0, 0, 0), 
            1, 
            cv2.LINE_AA
        )

def main(args):
    # Create output directory for saving images
    output_dir = PREDICTIONS_DIR / f"visualized_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(exist_ok=True)
    print(f"Will save annotated images to: {output_dir}")
    
    # Load COCO annotations
    coco_file = PREDICTIONS_DIR / 'predictions_coco.json'
    if not coco_file.exists():
        print(f"COCO file not found at {coco_file}")
        print("Please run 'python dataset_pipeline/export_to_coco.py' first")
        return
    
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create lookups
    image_lookup = {img['id']: img for img in coco_data['images']}
    category_lookup = {cat['id']: cat for cat in coco_data['categories']}
    
    # Group annotations by image ID
    annotations_by_image = {}
    for anno in coco_data['annotations']:
        image_id = anno['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(anno)
    
    # Print some statistics
    print(f"Found {len(annotations_by_image)} images with annotations out of {len(image_lookup)} total images")
    print(f"Categories: {[cat['name'] for cat in coco_data['categories']]}")
    
    # Select only certain image IDs if specified
    if args.image_id:
        image_ids = [int(args.image_id)]
    elif args.video:
        # Filter images by video name
        image_ids = [img['id'] for img in coco_data['images'] 
                    if 'video_name' in img and img['video_name'] == args.video]
    else:
        # Get all image IDs
        image_ids = list(image_lookup.keys())
    
    # Filter to only images with annotations if requested
    if args.only_with_annotations:
        original_count = len(image_ids)
        image_ids = [img_id for img_id in image_ids if img_id in annotations_by_image]
        print(f"Filtered from {original_count} to {len(image_ids)} images with annotations")
    # Otherwise, prioritize images with annotations
    elif args.prioritize_annotations:
        # Sort image IDs to put those with annotations first
        image_ids = sorted(image_ids, key=lambda img_id: img_id not in annotations_by_image)
    
    # Shuffle if random flag is set
    if args.random:
        random.shuffle(image_ids)
    
    # Only take a subset if limit is specified
    if args.limit > 0:
        image_ids = image_ids[:args.limit]
    
    # Assign colors to categories
    colors = {
        "Building-No-Damage": (0, 255, 0),       # Green
        "Building-Minor-Damage": (0, 255, 255),  # Yellow
        "Building-Major-Damage": (0, 0, 255),    # Red
        "Building-Total-Destruction": (255, 0, 0) # Blue
    }
    
    # Count statistics for processed images
    images_with_annotations = 0
    images_without_annotations = 0
    
    # Visualization loop
    for img_idx, image_id in enumerate(image_ids):
        if image_id not in image_lookup:
            print(f"Image ID {image_id} not found in dataset")
            continue
        
        image_data = image_lookup[image_id]
        image_path = Path(image_data['file_name'])
        
        # Handle images from base paths
        if not image_path.is_absolute():
            # Try different potential paths
            potential_paths = [
                Path(IMAGES_DIR) / image_data.get('video_name', '') / image_path.name,
                Path(image_data.get('file_name', ''))
            ]
            
            for p in potential_paths:
                if p.exists():
                    image_path = p
                    break
        
        # Skip if image not found
        if not image_path.exists():
            print(f"Image not found: {image_path}")
            continue
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue
        
        # Draw all annotations
        annotations = annotations_by_image.get(image_id, [])
        
        # Check if this image has annotations
        has_annotations = len(annotations) > 0
        
        if has_annotations:
            images_with_annotations += 1
            for anno in annotations:
                category_id = anno['category_id']
                category = category_lookup.get(category_id, {})
                category_name = category.get('name', f"Unknown ({category_id})")
                
                # Get color
                color = colors.get(category_name, (255, 255, 255))
                
                # Draw segmentation polygons
                for segmentation in anno['segmentation']:
                    # Convert flat array to points
                    points = [(segmentation[i], segmentation[i+1]) 
                             for i in range(0, len(segmentation), 2)]
                    
                    draw_polygon(image, points, color, 
                               f"{category_name} ({anno.get('score', 1.0):.2f})",
                               thickness=2)
        else:
            images_without_annotations += 1
            # Add text to indicate no annotations
            cv2.putText(
                image,
                "NO ANNOTATIONS",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )
            print(f"Image {image_id} ({image_path.name}) has no annotations")
        
        # Save image with annotations
        annotation_status = "with_annotations" if has_annotations else "no_annotations"
        image_name = f"{img_idx+1:04d}_{Path(image_path).stem}_{annotation_status}.jpg"
        output_path = output_dir / image_name
        cv2.imwrite(str(output_path), image)
        print(f"Saved image {img_idx+1}/{len(image_ids)}: {output_path}")
    
    print(f"Finished visualizing {len(image_ids)} images")
    print(f"Images with annotations: {images_with_annotations}")
    print(f"Images without annotations: {images_without_annotations}")
    print(f"All images saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize COCO annotations")
    parser.add_argument('--image-id', type=str, help='Specific image ID to visualize')
    parser.add_argument('--video', type=str, help='Visualize images from a specific video')
    parser.add_argument('--random', action='store_true', help='Randomize image order')
    parser.add_argument('--limit', type=int, default=10, help='Limit number of images to visualize (default: 10)')
    parser.add_argument('--output-dir', type=str, help='Custom directory to save annotated images')
    parser.add_argument('--only-with-annotations', action='store_true', help='Only process images that have annotations')
    parser.add_argument('--prioritize-annotations', action='store_true', help='Process images with annotations first')
    args = parser.parse_args()
    
    # Use custom output directory if provided
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
    
    main(args) 