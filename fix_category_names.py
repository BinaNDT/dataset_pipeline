#!/usr/bin/env python3
"""
Fix category names in test annotations

This script updates the test annotations to use the correct category names
that match the Labelbox schema (replacing dashes with underscores).
"""

import json
from pathlib import Path

# Category name mapping (from COCO to Labelbox)
category_mapping = {
    "Building-No-Damage": "Building_No_Damage",
    "Building-Minor-Damage": "Building_Minor_Damage",
    "Building-Major-Damage": "Building_Major_Damage",
    "Building-Total-Destruction": "Building_Total_Destruction"
}

# Load the test annotations
test_file = Path("outputs/predictions/test_annotations.ndjson")
print(f"Loading test annotations from: {test_file}")

annotations_list = []
with open(test_file, "r") as f:
    for line in f:
        annotations_list.append(json.loads(line.strip()))

print(f"Loaded {len(annotations_list)} annotation entries")

# Update category names
for entry in annotations_list:
    for ann in entry["annotations"]:
        if ann["name"] in category_mapping:
            old_name = ann["name"]
            ann["name"] = category_mapping[old_name]
            print(f"Updated category name: {old_name} -> {ann['name']}")

# Save updated annotations
updated_file = Path("outputs/predictions/fixed_name_annotations.ndjson")
with open(updated_file, "w") as f:
    for entry in annotations_list:
        f.write(json.dumps(entry) + "\n")

print(f"Saved updated annotations to: {updated_file}")

# Now also create a version with highly visible shapes
visible_file = Path("outputs/predictions/visible_fixed_annotations.ndjson")
print(f"Creating visible annotation shapes in: {visible_file}")

for entry in annotations_list:
    filename = entry["uuid"].replace("test_", "")
    print(f"Updating annotations for {filename}")
    
    # Clear existing annotations
    entry["annotations"] = []
    
    # Add a large X shape across the image
    entry["annotations"].append({
        "uuid": f"visible_x_{filename}",
        "name": "Building_Total_Destruction",  # Using updated name
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
        "name": "Building_No_Damage",  # Using updated name
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
        "name": "Building_Minor_Damage",  # Using updated name
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

# Save the visible annotations
with open(visible_file, "w") as f:
    for entry in annotations_list:
        f.write(json.dumps(entry) + "\n")

print(f"Saved visible annotations with fixed names to: {visible_file}")
print("Use these files with the import script to test if annotations display correctly") 