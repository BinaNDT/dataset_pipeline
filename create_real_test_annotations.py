#!/usr/bin/env python3
"""
Create test annotations using real coordinates from COCO dataset
"""

import json
import os
from pathlib import Path

# Load COCO file
coco_file = Path("outputs/predictions/predictions_coco.json")
print(f"Loading COCO file: {coco_file}")

with open(coco_file, "r") as f:
    data = json.load(f)

# Get first 5 images
first_5_images = data["images"][:5]
first_5_image_ids = [img["id"] for img in first_5_images]

# Hardcoded Labelbox IDs based on check_labelbox.py output
labelbox_ids = {
    "0000000.png": "cma2wb55mw6vh0773ggg252nr",
    "0000001.png": "cma2wb55qw6vi0773y07qh8c0",
    "0000002.png": "cma2wb55gw6vg0773ma3441wh",
    "0000003.png": "cma2wb55ww6vk0773tiwu8vp7",
    "0000004.png": "cma2wb55tw6vj07736oxg9mxa"
}

# Create NDJSON import file with real annotations
print("\nCreating real test import file")
annotations_list = []

for img in first_5_images:
    filename = img["file_name"]
    image_id = img["id"]
    
    if filename in labelbox_ids:
        labelbox_id = labelbox_ids[filename]
        image_annotations = []
        
        # Get real annotations for this image
        for ann in data["annotations"]:
            if ann["image_id"] == image_id:
                # Get category
                cat_id = ann["category_id"]
                cat_name = next((c["name"] for c in data["categories"] if c["id"] == cat_id), None)
                
                if not cat_name:
                    continue
                
                # Get segmentation
                if "segmentation" not in ann or not ann["segmentation"] or not ann["segmentation"][0]:
                    continue
                
                # Convert segmentation to points
                points = []
                for i in range(0, len(ann["segmentation"][0]), 2):
                    if i + 1 < len(ann["segmentation"][0]):
                        x = ann["segmentation"][0][i]
                        y = ann["segmentation"][0][i+1]
                        points.append([x, y])
                
                if points:
                    # Add annotation
                    image_annotations.append({
                        "uuid": str(ann["id"]),
                        "name": cat_name,
                        "value": {
                            "format": "polygon2d",
                            "points": points
                        }
                    })
        
        if image_annotations:
            # Add entry for this image
            annotations_list.append({
                "uuid": f"real_test_{filename}",
                "dataRow": {
                    "id": labelbox_id
                },
                "annotations": image_annotations
            })
            print(f"Added {len(image_annotations)} real annotations for {filename}")
        else:
            print(f"No real annotations found for {filename}")

# Save test file
test_file = Path("outputs/predictions/real_test_annotations.ndjson")
with open(test_file, "w") as f:
    for ann in annotations_list:
        f.write(json.dumps(ann) + "\n")

print(f"Saved real test file with {len(annotations_list)} images to {test_file}")
print(f"You can now import this file using the import_test_annotations.sh script") 