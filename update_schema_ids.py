#!/usr/bin/env python3
"""
Update test annotations with correct schema IDs

This script updates the test annotations to use the correct schema IDs
instead of the display names for Labelbox polygon tools.
"""

import json
from pathlib import Path

# Schema ID mapping (from display name to internal ID)
schema_id_mapping = {
    "Building_No_Damage": "building_no_damage",
    "Building_Minor_Damage": "building_minor_damage",
    "Building_Major_Damage": "building_major_damage",
    "Building_Total_Destruction": "building_total_destruction"
}

def main():
    # Load the test annotations
    test_file = Path("outputs/predictions/visible_fixed_annotations.ndjson")
    print(f"Loading test annotations from: {test_file}")

    annotations_list = []
    with open(test_file, "r") as f:
        for line in f:
            annotations_list.append(json.loads(line.strip()))

    print(f"Loaded {len(annotations_list)} annotation entries")

    # Update schema IDs
    for entry in annotations_list:
        for ann in entry["annotations"]:
            if ann["name"] in schema_id_mapping:
                display_name = ann["name"]
                schema_id = schema_id_mapping[display_name]
                
                # Keep the original name for display/reference
                ann["annotationName"] = display_name
                
                # Set the schema ID to the lowercase version
                ann["schemaId"] = schema_id
                
                print(f"Updated schema ID: {display_name} -> {schema_id}")

    # Save updated annotations
    updated_file = Path("outputs/predictions/schema_id_annotations.ndjson")
    with open(updated_file, "w") as f:
        for entry in annotations_list:
            f.write(json.dumps(entry) + "\n")

    print(f"Saved updated annotations to: {updated_file}")
    print("Use this file with the import script to test if annotations display correctly")
    
    # Also update the import script
    import_script = Path("import_schema_id_test.sh")
    script_content = """#!/bin/bash
# Import schema ID test annotations to Labelbox

# Set timestamp for import name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export TIMESTAMP  # Export so it's available in the Python subprocess

echo "Importing schema ID test annotations to Labelbox..."
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

# Prepare annotations with schema IDs
annotations_file = Path('outputs/predictions/schema_id_annotations.ndjson')
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
            # Use the schema ID from our updated file
            annotation = {
                'schemaId': ann.get('schemaId', ann['name']),  # Use schemaId if available
                'annotationName': ann.get('annotationName', ann['name']),  # Use proper display name
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
            name=f'SchemaID_Test_Import_{TIMESTAMP}',
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
"""

    with open(import_script, "w") as f:
        f.write(script_content)
    
    import os
    os.chmod(import_script, 0o755)
    print(f"Created import script: {import_script}")
    print(f"Run it with: ./{import_script}")

if __name__ == "__main__":
    main() 