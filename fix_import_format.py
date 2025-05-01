#!/usr/bin/env python3
"""
Fix import format for Labelbox

This script corrects the import format to match Labelbox's expected structure.
"""

import json
from pathlib import Path
import os

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

    # Create correctly formatted predictions
    predictions = []
    
    for entry in annotations_list:
        data_row_id = entry["dataRow"]["id"]
        
        # Process each annotation as a separate prediction
        for ann in entry["annotations"]:
            if ann["name"] in schema_id_mapping:
                # Get the correct schema ID
                schema_id = schema_id_mapping[ann["name"]]
                
                # Get the points
                points = ann["value"]["points"]
                
                # Format the points as Labelbox expects
                vertices = [{"x": p[0], "y": p[1]} for p in points]
                
                # Create prediction in the correct format
                prediction = {
                    "dataRow": {
                        "id": data_row_id
                    },
                    "schemaId": schema_id,
                    "polygon": {
                        "vertices": vertices,
                        "locationData": {}
                    }
                }
                
                predictions.append(prediction)
                print(f"Created prediction for {ann['name']} ({schema_id}) on data row {data_row_id}")
    
    # Save correctly formatted predictions
    predictions_file = Path("outputs/predictions/fixed_format_predictions.json")
    with open(predictions_file, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Saved {len(predictions)} predictions to: {predictions_file}")
    
    # Create the import script
    import_script = Path("import_correct_format.sh")
    script_content = """#!/bin/bash
# Import correctly formatted annotations to Labelbox

# Set timestamp for import name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export TIMESTAMP  # Export so it's available in the Python subprocess

echo "Importing correctly formatted annotations to Labelbox..."
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

print(f'Using project ID: {{PROJECT_ID}}')
print(f'Using timestamp: {{TIMESTAMP}}')

# Connect to Labelbox
client = lb.Client(api_key=API_KEY)

# Load the correctly formatted predictions
predictions_file = Path('outputs/predictions/fixed_format_predictions.json')
print(f'Using predictions file: {{predictions_file}}')

with open(predictions_file, 'r') as f:
    predictions = json.load(f)

print(f'Loaded {{len(predictions)}} predictions')

# Create import
try:
    # Create import using the correct format
    upload_job = lb.MALPredictionImport.create_from_objects(
        client=client,
        project_id=PROJECT_ID,
        name=f'CorrectFormat_Import_{{TIMESTAMP}}',
        predictions=predictions
    )
    print(f'Import started with ID: {{upload_job.uid}}')
    
    # Monitor progress
    import time
    start_time = time.time()
    max_wait_time = 2 * 60  # 2 minutes
    
    while time.time() - start_time < max_wait_time:
        upload_job.refresh()
        
        if hasattr(upload_job, 'state'):
            state = upload_job.state
            progress = upload_job.progress if hasattr(upload_job, 'progress') else 'unknown'
            
            print(f'Status: {{state}}, Progress: {{progress}}')
            
            if state in ('COMPLETE', 'FINISHED'):
                print('Import completed successfully!')
                break
            elif state in ('FAILED', 'ERROR'):
                print('Import failed!')
                if hasattr(upload_job, 'errors'):
                    for i, error in enumerate(upload_job.errors):
                        print(f'Error {{i+1}}: {{error}}')
                elif hasattr(upload_job, 'error'):
                    print(f'Error: {{upload_job.error}}')
                break
        
        time.sleep(10)
except Exception as e:
    print(f'Error: {{str(e)}}')
"

echo "Process complete!"
"""

    with open(import_script, "w") as f:
        f.write(script_content)
    
    os.chmod(import_script, 0o755)
    print(f"Created import script: {import_script}")
    print(f"Run it with: ./{import_script}")

if __name__ == "__main__":
    main() 