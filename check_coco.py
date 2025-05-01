#!/usr/bin/env python3
"""
Check COCO annotations file
"""

import json
import os
import uuid
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

# Find images that have annotations
print("\nFinding images with annotations...")
image_to_annotations = {}
id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

for ann in data["annotations"]:
    image_id = ann["image_id"]
    if image_id in id_to_filename:
        filename = id_to_filename[image_id]
        if filename not in image_to_annotations:
            image_to_annotations[filename] = []
        image_to_annotations[filename].append(ann)

print(f"Found {len(image_to_annotations)} images with annotations")

# Select up to 5 images with annotations for testing
test_filenames = list(image_to_annotations.keys())[:5]
print("\nTesting with these images:")
for filename in test_filenames:
    image_id = next(img["id"] for img in data["images"] if img["file_name"] == filename)
    print(f"File {filename} (ID: {image_id}) has {len(image_to_annotations[filename])} annotations")

# Create NDJSON import file for testing
print("\nCreating test import file")
annotations_list = []

for filename in test_filenames:
    # Use the filename as the globalKey
    global_key = filename
    print(f"Using global key: {global_key}")
    
    # Get annotations for this image
    for ann in image_to_annotations[filename]:
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
                points.append({"x": x, "y": y})
        
        # Ensure polygon is closed (first point == last point)
        if points and (points[0]["x"] != points[-1]["x"] or points[0]["y"] != points[-1]["y"]):
            points.append({"x": points[0]["x"], "y": points[0]["y"]})
        
        if points:
            # Create a valid UUID for each annotation
            annotation_uuid = str(uuid.uuid4())
            
            # Add annotation directly (don't use 'annotations' array)
            annotations_list.append({
                "uuid": annotation_uuid,
                "name": cat_name,
                "polygon": points,
                "classifications": [],
                "dataRow": {
                    "globalKey": global_key
                }
            })
            print(f"Added annotation for {filename} (Category: {cat_name})")

# Save test file
test_file = Path("outputs/predictions/test_annotations.ndjson")
with open(test_file, "w") as f:
    for ann in annotations_list:
        f.write(json.dumps(ann) + "\n")

print(f"Saved test file with {len(annotations_list)} annotations to {test_file}")
print(f"You can now import this file using the labelbox_importer.py script") 