#!/usr/bin/env python3
"""
Visualize the test annotations being sent to Labelbox

This script loads and visualizes the visible test annotations that are
being sent to Labelbox to help diagnose issues with displaying them.
"""

import json
import cv2
import numpy as np
from pathlib import Path
import os
import argparse
from config import IMAGES_DIR, PREDICTIONS_DIR, DATASET_ROOT

def draw_polygon(image, points, color, label=None, thickness=2):
    """Draw a polygon on the image"""
    # Convert points to numpy array
    points = np.array(points, np.int32)
    
    # Draw the polygon
    cv2.polylines(image, [points], True, color, thickness)
    
    # Fill with transparent color
    overlay = image.copy()
    cv2.fillPoly(overlay, [points], color)
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # Add label if provided
    if label:
        # Calculate centroid
        M = cv2.moments(points)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # Draw label background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(
                image,
                (cx - 5, cy - text_size[1] - 5),
                (cx + text_size[0] + 5, cy + 5),
                (0, 0, 0),
                -1
            )
            # Draw text
            cv2.putText(
                image,
                label,
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

def main():
    parser = argparse.ArgumentParser(description="Visualize test annotations for Labelbox")
    parser.add_argument('--annotations-file', type=str, default="outputs/predictions/visible_fixed_annotations.ndjson",
                        help='Path to the test annotations file')
    args = parser.parse_args()
    
    # Load the test annotations
    annotations_file = Path(args.annotations_file)
    print(f"Loading test annotations from: {annotations_file}")
    
    if not annotations_file.exists():
        print(f"Annotations file not found: {annotations_file}")
        return
    
    # Load annotations
    annotations_list = []
    with open(annotations_file, "r") as f:
        for line in f:
            annotations_list.append(json.loads(line.strip()))
    
    print(f"Loaded {len(annotations_list)} annotation entries")
    
    # Create output directory
    output_dir = PREDICTIONS_DIR / "visualized_test_annotations"
    output_dir.mkdir(exist_ok=True)
    print(f"Will save images to: {output_dir}")
    
    # Define colors for different categories
    colors = {
        "Building_No_Damage": (0, 255, 0),       # Green
        "Building_Minor_Damage": (0, 255, 255),  # Yellow
        "Building_Major_Damage": (0, 0, 255),    # Red
        "Building_Total_Destruction": (255, 0, 0) # Blue
    }
    
    # Process each annotation entry
    for i, entry in enumerate(annotations_list):
        filename = entry["uuid"].replace("test_", "")
        print(f"Processing annotations for {filename}")
        
        # Look for the image file in multiple possible locations
        image_path = None
        potential_paths = [
            Path(IMAGES_DIR) / filename,
            Path(DATASET_ROOT) / "images" / filename,
            Path(DATASET_ROOT) / "frames" / filename,
            Path("/data/datasets") / filename,
            Path(filename)
        ]
        
        # Also try to find by pattern (searching for numbered frames)
        frame_number = int(filename.split('.')[0])
        for subdir in ["Hurricane_Ian_10012022_Palmeto_Palms-1", "Hurricane_Ian_Coastal"]:
            potential_paths.append(Path(DATASET_ROOT) / "frames" / subdir / f"{frame_number:07d}.png")
        
        for p in potential_paths:
            if p.exists():
                image_path = p
                break
        
        if not image_path or not image_path.exists():
            print(f"Image not found for {filename}, creating blank image")
            # Create a blank image (1920x1080) if original not found
            image = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
            
            # Add information about missing image
            cv2.putText(
                image,
                f"Image {filename} not found",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
        else:
            print(f"Found image at: {image_path}")
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue
        
        # Create a copy of the image for annotations
        annotated_image = image.copy()
        
        # Draw all annotations
        for anno in entry["annotations"]:
            category_name = anno["name"]
            points = anno["value"]["points"]
            
            # Get color
            color = colors.get(category_name, (128, 128, 128))
            
            # Draw polygon
            draw_polygon(
                annotated_image,
                points,
                color,
                f"{category_name}",
                thickness=3
            )
        
        # Add information about Labelbox ID
        labelbox_id = entry["dataRow"]["id"]
        cv2.putText(
            annotated_image,
            f"Labelbox ID: {labelbox_id}",
            (50, image.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Save the annotated image
        output_path = output_dir / f"{i+1:02d}_{filename}"
        cv2.imwrite(str(output_path), annotated_image)
        print(f"Saved annotated image: {output_path}")
        
        # Also generate a visualization of just the annotations on a blank image
        # to clearly see the shape without any image background
        blank_image = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
        
        # Draw title
        cv2.putText(
            blank_image,
            f"Annotations for {filename} (Labelbox ID: {labelbox_id})",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2
        )
        
        # Draw all annotations
        for anno in entry["annotations"]:
            category_name = anno["name"]
            points = anno["value"]["points"]
            
            # Get color
            color = colors.get(category_name, (128, 128, 128))
            
            # Draw polygon
            draw_polygon(
                blank_image,
                points,
                color,
                f"{category_name}",
                thickness=3
            )
        
        # Save the blank image with annotations
        blank_output_path = output_dir / f"{i+1:02d}_{filename.replace('.png', '_annotations_only.png')}"
        cv2.imwrite(str(blank_output_path), blank_image)
        print(f"Saved annotations-only image: {blank_output_path}")
    
    print(f"Finished visualizing {len(annotations_list)} images")
    print(f"All images saved to: {output_dir}")
    print(f"You can check these visualizations to confirm the annotations being sent to Labelbox")

if __name__ == "__main__":
    main() 