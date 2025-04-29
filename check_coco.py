#!/usr/bin/env python3
"""
Check COCO annotations file
"""

import json
import os
from pathlib import Path

# Load COCO file
coco_file = Path("outputs/predictions/predictions_coco.json")
print(f"Loading COCO file: {coco_file}")

with open(coco_file, "r") as f:
    data = json.load(f)

# Print summary
print(f"Number of images: {len(data['images'])}")
print(f"Number of annotations: {len(data['annotations'])}")
print(f"Number of categories: {len(data['categories'])}")

# Print first 5 images
print("\nFirst 5 images:")
for img in data["images"][:5]:
    print(f"ID: {img['id']}, Filename: {img['file_name']}")

# Print first 5 annotations
print("\nFirst 5 annotations:")
for ann in data["annotations"][:5]:
    print(f"Annotation ID: {ann['id']}, Image ID: {ann['image_id']}, Category: {ann['category_id']}")

# Print categories
print("\nCategories:")
for cat in data["categories"]:
    print(f"ID: {cat['id']}, Name: {cat['name']}")

# Check annotations for first 5 files
first_5_filenames = ["0000000.png", "0000001.png", "0000002.png", "0000003.png", "0000004.png"]
print("\nChecking for annotations for first 5 files:")

# Build file_name to ID mapping
filename_to_id = {img["file_name"]: img["id"] for img in data["images"]}

for filename in first_5_filenames:
    if filename in filename_to_id:
        image_id = filename_to_id[filename]
        annotations = [ann for ann in data["annotations"] if ann["image_id"] == image_id]
        print(f"File {filename} (ID: {image_id}) has {len(annotations)} annotations")
    else:
        print(f"File {filename} not found in COCO file")

# Hardcoded Labelbox IDs based on check_labelbox.py output
labelbox_ids = {
    "0000000.png": "cma2wb55mw6vh0773ggg252nr",
    "0000001.png": "cma2wb55qw6vi0773y07qh8c0",
    "0000002.png": "cma2wb55gw6vg0773ma3441wh",
    "0000003.png": "cma2wb55ww6vk0773tiwu8vp7",
    "0000004.png": "cma2wb55tw6vj07736oxg9mxa"
}

# Create NDJSON import file for testing
print("\nCreating test import file")
annotations_list = []

for filename, labelbox_id in labelbox_ids.items():
    if filename in filename_to_id:
        image_id = filename_to_id[filename]
        image_annotations = []
        
        # Get annotations for this image
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
                "uuid": f"test_{filename}",
                "dataRow": {
                    "id": labelbox_id
                },
                "annotations": image_annotations
            })
            print(f"Added {len(image_annotations)} annotations for {filename}")

# Save test file
test_file = Path("outputs/predictions/test_annotations.ndjson")
with open(test_file, "w") as f:
    for ann in annotations_list:
        f.write(json.dumps(ann) + "\n")

print(f"Saved test file with {len(annotations_list)} images to {test_file}")
print(f"You can now import this file using the labelbox_importer.py script") 