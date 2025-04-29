#!/usr/bin/env python3
"""
Create test annotations for Labelbox
"""

import json
import time
from pathlib import Path

# Hardcoded Labelbox IDs based on check_labelbox.py output
labelbox_ids = {
    "0000000.png": "cma2wb55mw6vh0773ggg252nr",
    "0000001.png": "cma2wb55qw6vi0773y07qh8c0",
    "0000002.png": "cma2wb55gw6vg0773ma3441wh",
    "0000003.png": "cma2wb55ww6vk0773tiwu8vp7",
    "0000004.png": "cma2wb55tw6vj07736oxg9mxa"
}

# Create dummy annotations for each image
print("Creating test annotations...")
annotations_list = []

# Categories from COCO file
categories = [
    "Building-No-Damage",
    "Building-Minor-Damage",
    "Building-Major-Damage",
    "Building-Total-Destruction"
]

# Image dimensions from checking the COCO file
width = 1920
height = 1080

for filename, labelbox_id in labelbox_ids.items():
    # Create 2 dummy annotations for each image
    image_annotations = []
    
    # Annotation 1: Building-No-Damage in the top-left corner
    points1 = [
        [100, 100],
        [200, 100],
        [200, 200],
        [100, 200],
        [100, 100]
    ]
    
    image_annotations.append({
        "uuid": f"test_ann1_{filename}",
        "name": "Building-No-Damage",
        "value": {
            "format": "polygon2d",
            "points": points1
        }
    })
    
    # Annotation 2: Building-Total-Destruction in the bottom-right corner
    points2 = [
        [width - 200, height - 200],
        [width - 100, height - 200],
        [width - 100, height - 100],
        [width - 200, height - 100],
        [width - 200, height - 200]
    ]
    
    image_annotations.append({
        "uuid": f"test_ann2_{filename}",
        "name": "Building-Total-Destruction",
        "value": {
            "format": "polygon2d",
            "points": points2
        }
    })
    
    # Create entry for this image
    annotations_list.append({
        "uuid": f"test_{filename}",
        "dataRow": {
            "id": labelbox_id
        },
        "annotations": image_annotations
    })
    print(f"Added 2 annotations for {filename}")

# Save test file
test_file = Path("outputs/predictions/test_annotations.ndjson")
with open(test_file, "w") as f:
    for ann in annotations_list:
        f.write(json.dumps(ann) + "\n")

print(f"Saved test file with {len(annotations_list)} images to {test_file}")

# Create a Labelbox MAL import script
import_script = Path("import_test_annotations.sh")
with open(import_script, "w") as f:
    f.write("""#!/bin/bash
# Import test annotations to Labelbox

# Set timestamp for import name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Importing test annotations to Labelbox..."
python -c "
import os
import json
import labelbox as lb
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()
API_KEY = os.getenv('LABELBOX_API_KEY')
PROJECT_ID = os.getenv('LABELBOX_PROJECT_ID')

print(f'Using project ID: {PROJECT_ID}')

# Connect to Labelbox
client = lb.Client(api_key=API_KEY)

# Prepare annotations
annotations_file = Path('outputs/predictions/test_annotations.ndjson')
print(f'Using annotations file: {annotations_file}')

# Try to create DataRow objects for each annotation
predictions = []

with open(annotations_file, 'r') as f:
    for line in f:
        ann_entry = json.loads(line.strip())
        data_row_id = ann_entry['dataRow']['id']
        
        # Create DataRow object
        data_row = lb.DataRow(client, data_row_id)
        
        # Create annotations
        annotations = []
        for ann in ann_entry['annotations']:
            # Convert points to Labelbox format
            points = [{'x': p[0], 'y': p[1]} for p in ann['value']['points']]
            
            # Create Polygon annotation
            annotation = lb.Polygon(
                name=ann['name'],
                value=lb.Polygon.ValueData(points=points)
            )
            annotations.append(annotation)
        
        # Create prediction
        if annotations:
            prediction = lb.MALPrediction(
                data_row=data_row,
                annotations=annotations
            )
            predictions.append(prediction)

# Create import
if predictions:
    try:
        upload_job = lb.MALPredictionImport.create_from_objects(
            client=client,
            project_id=PROJECT_ID,
            name=f'Test_Import_{os.getenv(\\"TIMESTAMP\\")}',
            predictions=predictions
        )
        print(f'Import started with ID: {upload_job.uid}')
        
        # Monitor progress
        import time
        start_time = time.time()
        max_wait_time = 2 * 60  # 2 minutes
        
        while time.time() - start_time < max_wait_time:
            upload_job.refresh()
            
            if hasattr(upload_job, 'state'):
                state = upload_job.state
                progress = upload_job.progress if hasattr(upload_job, 'progress') else 'unknown'
                
                print(f'Status: {state}, Progress: {progress}')
                
                if state in ('COMPLETE', 'FINISHED'):
                    print('Import completed successfully!')
                    break
                elif state in ('FAILED', 'ERROR'):
                    print('Import failed!')
                    break
            
            time.sleep(10)
    except Exception as e:
        print(f'Error: {str(e)}')
else:
    print('No predictions to upload')
"

echo "Process complete!"
""")

# Make the script executable
import os
os.chmod(import_script, 0o755)

print(f"Created import script: {import_script}")
print("To import the test annotations, run: ./import_test_annotations.sh") 