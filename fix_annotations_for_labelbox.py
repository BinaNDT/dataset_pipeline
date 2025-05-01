#!/usr/bin/env python3
"""
Fix COCO annotations for Labelbox compatibility

This script updates the COCO annotation file to use category names
compatible with Labelbox (replacing dashes with underscores).
"""

import json
import os
from pathlib import Path

# Category name mapping (from COCO to Labelbox)
category_mapping = {
    "Building-No-Damage": "Building_No_Damage",
    "Building-Minor-Damage": "Building_Minor_Damage",
    "Building-Major-Damage": "Building_Major_Damage",
    "Building-Total-Destruction": "Building_Total_Destruction"
}

def main():
    # Load COCO file
    coco_file = Path("outputs/predictions/predictions_coco.json")
    print(f"Loading COCO file: {coco_file}")

    if not coco_file.exists():
        print(f"Error: COCO file not found at {coco_file}")
        return

    with open(coco_file, "r") as f:
        data = json.load(f)

    # Print original summary
    print("Original COCO file:")
    print(f"Number of images: {len(data['images'])}")
    print(f"Number of annotations: {len(data['annotations'])}")
    print(f"Number of categories: {len(data['categories'])}")

    # Update category names
    for cat in data["categories"]:
        if cat["name"] in category_mapping:
            old_name = cat["name"]
            cat["name"] = category_mapping[old_name]
            print(f"Updated category: {old_name} -> {cat['name']}")

    # Save updated COCO file
    output_file = Path("outputs/predictions/labelbox_coco.json")
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved updated COCO file to: {output_file}")
    print("You can now use this file for importing to Labelbox")

    # Create a small script to use this file with labelbox_importer.py
    script_file = Path("import_fixed_coco.sh")
    script_content = f"""#!/bin/bash
# Import fixed COCO file to Labelbox

# Run the importer with the fixed COCO file
python labelbox_importer.py \\
  --source coco \\
  --coco-file outputs/predictions/labelbox_coco.json \\
  --dataset-name Fixed_Names_Import

echo "Import process complete!"
"""

    with open(script_file, "w") as f:
        f.write(script_content)
    
    os.chmod(script_file, 0o755)
    print(f"Created import script: {script_file}")
    print(f"Run it with: ./{script_file}")

if __name__ == "__main__":
    main() 