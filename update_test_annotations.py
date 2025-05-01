#!/usr/bin/env python3
"""
Update test annotations with more visible coordinates

This script modifies the existing test annotations to use coordinates 
that would be unmistakable if they appear in the Labelbox UI.
"""

import json
from pathlib import Path

# Load the existing test annotations
test_file = Path("outputs/predictions/test_annotations.ndjson")
print(f"Loading test annotations from: {test_file}")

annotations_list = []
with open(test_file, "r") as f:
    for line in f:
        annotations_list.append(json.loads(line.strip()))

print(f"Loaded {len(annotations_list)} annotation entries")

# Create highly visible shapes for each image
for entry in annotations_list:
    filename = entry["uuid"].replace("test_", "")
    print(f"Updating annotations for {filename}")
    
    # Clear existing annotations
    entry["annotations"] = []
    
    # Add a large X shape across the image
    entry["annotations"].append({
        "uuid": f"visible_x_{filename}",
        "name": "Building-Total-Destruction",  # Using this category for visibility
        "value": {
            "format": "polygon2d",
            "points": [
                [200, 200],
                [400, 400],
                [600, 600],
                [800, 800],
                [1000, 1000],
                [1200, 800],
                [1400, 600],
                [1600, 400],
                [1800, 200],
                [1600, 200],
                [1400, 400],
                [1200, 600],
                [1000, 800],
                [800, 600],
                [600, 400],
                [400, 200],
                [200, 200]
            ]
        }
    })
    
    # Add a large rectangle in the center
    entry["annotations"].append({
        "uuid": f"visible_rect_{filename}",
        "name": "Building-No-Damage",
        "value": {
            "format": "polygon2d",
            "points": [
                [600, 300],
                [1300, 300],
                [1300, 700],
                [600, 700],
                [600, 300]
            ]
        }
    })
    
    # Add a triangle on the left side
    entry["annotations"].append({
        "uuid": f"visible_tri_{filename}",
        "name": "Building-Minor-Damage",
        "value": {
            "format": "polygon2d",
            "points": [
                [100, 800],
                [300, 500],
                [500, 800],
                [100, 800]
            ]
        }
    })

# Save updated test file
updated_file = Path("outputs/predictions/visible_test_annotations.ndjson")
with open(updated_file, "w") as f:
    for entry in annotations_list:
        f.write(json.dumps(entry) + "\n")

print(f"Saved updated test file with visible annotations to {updated_file}")
print("Try importing these annotations - they should be clearly visible if they appear at all in Labelbox")
print("To import: Update the file path in import_test_annotations.sh and run it") 