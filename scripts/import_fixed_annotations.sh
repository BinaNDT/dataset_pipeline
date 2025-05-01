#!/bin/bash
# Import fixed annotations to Labelbox

# Set timestamp for import name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export TIMESTAMP  # Export so it's available in the Python subprocess

echo "Importing fixed annotations to Labelbox..."
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
TIMESTAMP = os.environ.get('TIMESTAMP')  # Get TIMESTAMP from environment

print(f'Using project ID: {PROJECT_ID}')
print(f'Using timestamp: {TIMESTAMP}')

# Connect to Labelbox
client = lb.Client(api_key=API_KEY)

# Prepare annotations - using the FIXED category names
annotations_file = Path('outputs/predictions/visible_fixed_annotations.ndjson')
print(f'Using annotations file: {annotations_file}')

# Try to create DataRow objects for each annotation
predictions = []

with open(annotations_file, 'r') as f:
    for line in f:
        ann_entry = json.loads(line.strip())
        data_row_id = ann_entry['dataRow']['id']
        
        # Create annotations
        annotations = []
        for ann in ann_entry['annotations']:
            # Convert points to Labelbox format
            points = [{'x': p[0], 'y': p[1]} for p in ann['value']['points']]
            
            # Create annotation using the correct format for Labelbox SDK
            annotation = {
                'schemaId': ann['name'],  # Use the name directly as schemaId (with underscores)
                'annotationName': ann['name'],  # Keep the corrected name
                'polygon': {
                    'vertices': points,
                    'locationData': {}
                }
            }
            annotations.append(annotation)
        
        # Create prediction
        if annotations:
            prediction = {
                'dataRow': {'id': data_row_id},
                'annotations': annotations
            }
            predictions.append(prediction)

# Create import
if predictions:
    try:
        # Create import using the correct format
        upload_job = lb.MALPredictionImport.create_from_objects(
            client=client,
            project_id=PROJECT_ID,
            name=f'Fixed_Visible_Import_{TIMESTAMP}',
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
                    if hasattr(upload_job, 'error'):
                        print(f'Error: {upload_job.error}')
                    break
            
            time.sleep(10)
    except Exception as e:
        print(f'Error: {str(e)}')
else:
    print('No predictions to upload')
"

echo "Process complete!" 