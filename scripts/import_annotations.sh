#!/bin/bash
# Script to import annotations to Labelbox
# This uses the existing labelbox_importer.py script with a modified approach

# Generate timestamp for unique import name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
IMPORT_NAME="Fixed_Import_$TIMESTAMP" 

# Load environment variables (API keys, etc.)
source .env

echo "Starting manual annotation import..."
echo "Using project ID: $LABELBOX_PROJECT_ID"

# Set the fixed row IDs to match with 0000000.png through 0000004.png files
# Based on the Labelbox IDs we discovered in the check_labelbox.py script
# Note: These values are hardcoded based on inspection of your specific Labelbox project

# Run the import with hard-coded datarow IDs
echo "Running import with datarow mapping..."
python -c "
import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Import labelbox
import labelbox as lb
from dotenv import load_dotenv
load_dotenv()

# Import config
from config import LABELBOX_API_KEY, LABELBOX_PROJECT_ID, PREDICTIONS_DIR

# Connect to Labelbox
client = lb.Client(api_key=LABELBOX_API_KEY)

# Load COCO annotations
coco_file = PREDICTIONS_DIR / 'predictions_coco.json'
with open(coco_file, 'r') as f:
    coco_data = json.load(f)

# Hard-coded mapping from filenames to Labelbox Data Row IDs (based on the check_labelbox.py output)
filename_to_datarow = {
    '0000000.png': 'cma2wb55mw6vh0773ggg252nr', # External ID 1
    '0000001.png': 'cma2wb55qw6vi0773y07qh8c0', # External ID 2 
    '0000002.png': 'cma2wb55gw6vg0773ma3441wh', # External ID 3
    '0000003.png': 'cma2wb55ww6vk0773tiwu8vp7', # External ID 4
    '0000004.png': 'cma2wb55tw6vj07736oxg9mxa'  # External ID 5
}

# Build image ID to filename mapping
image_id_to_filename = {}
for image in coco_data['images']:
    image_id_to_filename[image['id']] = image['file_name']

# Create ndjson format annotations
annotations_list = []

# Process each file we have in Labelbox
for filename, data_row_id in filename_to_datarow.items():
    # Get image ID from filename
    image_id = None
    for img_id, img_filename in image_id_to_filename.items():
        if img_filename == filename:
            image_id = img_id
            break
    
    if image_id is None:
        continue
    
    # Find annotations for this image
    image_annotations = []
    
    for ann in coco_data['annotations']:
        if ann['image_id'] == image_id:
            # Get category info
            cat_id = ann['category_id']
            cat_name = next((c['name'] for c in coco_data['categories'] 
                           if c['id'] == cat_id), None)
            
            if not cat_name:
                continue
            
            # Create polygon
            segmentation = ann.get('segmentation', [[]])
            if not segmentation or not segmentation[0]:
                continue
            
            polygon_points = []
            for i in range(0, len(segmentation[0]), 2):
                x = segmentation[0][i]
                y = segmentation[0][i+1]
                polygon_points.append([x, y])
            
            # Add annotation
            image_annotations.append({
                'uuid': str(ann['id']),
                'name': cat_name,
                'value': {
                    'format': 'polygon2d',
                    'points': polygon_points
                }
            })
    
    # Create entry for this image
    if image_annotations:
        annotations_list.append({
            'uuid': f'{time.time()}_{filename}',
            'dataRow': {
                'id': data_row_id
            },
            'annotations': image_annotations
        })

# Save to ndjson file
ndjson_file = PREDICTIONS_DIR / f'mapped_annotations_{datetime.now().strftime(\"%Y%m%d_%H%M%S\")}.ndjson'
with open(ndjson_file, 'w') as f:
    for ann in annotations_list:
        f.write(json.dumps(ann) + '\\n')

print(f'Created {len(annotations_list)} annotations')
print(f'Saved to {ndjson_file}')

# Try to import
try:
    # Try method 1 - create_from_file
    try:
        mal_import = lb.MALPredictionImport.create_from_file(
            client=client,
            project_id=LABELBOX_PROJECT_ID,
            name='$IMPORT_NAME',
            predictions_file=str(ndjson_file)  # Try this parameter name
        )
        print(f'Import started with ID: {mal_import.uid}')
    except TypeError:
        print('First method failed, trying alternate parameter name')
        # Try method 2 - alternate parameter name
        try:
            mal_import = lb.MALPredictionImport.create_from_file(
                client=client,
                project_id=LABELBOX_PROJECT_ID,
                name='$IMPORT_NAME',
                file=str(ndjson_file)  # Try this parameter name
            )
            print(f'Import started with ID: {mal_import.uid}')
        except TypeError:
            # Try method 3 - create from objects
            print('Trying to create from objects directly')
            predictions = []
            for annotation_entry in annotations_list:
                data_row_id = annotation_entry['dataRow']['id']
                
                # Create a DataRow object
                data_row = lb.DataRow(client, data_row_id)
                
                # Create annotations
                annotations = []
                for ann in annotation_entry['annotations']:
                    annotation = lb.Polygon(
                        name=ann['name'],
                        value=lb.Polygon.ValueData(
                            points=[
                                {'x': point[0], 'y': point[1]} 
                                for point in ann['value']['points']
                            ]
                        )
                    )
                    annotations.append(annotation)
                
                # Create prediction
                prediction = lb.MALPrediction(
                    data_row=data_row,
                    annotations=annotations
                )
                predictions.append(prediction)
            
            # Create import
            upload_job = lb.MALPredictionImport.create_from_objects(
                client=client,
                project_id=LABELBOX_PROJECT_ID,
                name='$IMPORT_NAME',
                predictions=predictions
            )
            print(f'Import started with ID: {upload_job.uid}')
except Exception as e:
    print(f'Error: {str(e)}')
"

echo "Process complete!" 