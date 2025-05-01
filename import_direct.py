#!/usr/bin/env python3
"""
Direct import of fixed format predictions to Labelbox
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import traceback
import numpy as np
import labelbox as lb
from labelbox.types import DataRow, MaskData, ObjectAnnotation

# Load environment variables
load_dotenv()

# Load necessary variables
API_KEY = os.getenv('LABELBOX_API_KEY')
PROJECT_ID = os.getenv('LABELBOX_PROJECT_ID')
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# Get actual schema IDs (CUIDs) from Labelbox
def get_ontology_schema_ids(client, project_id):
    print("Getting ontology schema IDs from Labelbox...")
    project = client.get_project(project_id)
    ontology = project.ontology()
    
    schema_id_map = {}
    # Get the list of tools
    tools = ontology.tools()
    if tools:
        # Update the mapping using the correct attribute for schema ID
        for tool in tools:
            schema_id_map[tool.name] = tool.feature_schema_id
    else:
        print("No tools found in the ontology")
    
    print(f"Found schema IDs: {schema_id_map}")
    return schema_id_map

# Fix mapping from our internal naming to actual schema CUIDs
name_mapping = {
    "building_no_damage": "Building_No_Damage",
    "building_minor_damage": "Building_Minor_Damage",
    "building_major_damage": "Building_Major_Damage",
    "building_total_destruction": "Building_Total_Destruction"
}

def main():
    try:
        # Import Labelbox SDK
        print(f'Using project ID: {PROJECT_ID}')
        print(f'Using timestamp: {TIMESTAMP}')
        
        # Connect to Labelbox
        client = lb.Client(api_key=API_KEY)
        
        # Get actual schema IDs from the ontology
        schema_id_map = get_ontology_schema_ids(client, PROJECT_ID)
        
        # Load the predictions
        predictions_file = Path('outputs/predictions/fixed_format_predictions.json')
        print(f'Using predictions file: {predictions_file}')
        
        with open(predictions_file, 'r') as f:
            predictions = json.load(f)
        
        print(f'Loaded {len(predictions)} predictions')
        
        # Show sample prediction format for debugging
        if predictions:
            print(f'Sample prediction format:')
            print(json.dumps(predictions[0], indent=2))
        
        # Prepare predictions using SDK classes
        predictions_sdk = []
        for entry in predictions:
            # Map internal to actual schema ID
            display_name = name_mapping.get(entry.get('schemaId'), entry.get('schemaId'))
            if display_name not in schema_id_map:
                print(f"Warning: schema '{display_name}' not found, skipping entry {entry['dataRow']['id']}")
                continue
            actual_schema_id = schema_id_map[display_name]

            # Create DataRow object
            data_row = lb.DataRow(client, entry['dataRow']['id'])

            # Use mask URL for mask annotation
            mask_url = f"https://storage.googleapis.com/example-bucket/masks/{entry['dataRow']['id']}.png"
            mask_data = MaskData(url=mask_url)
            object_annotation = ObjectAnnotation(
                name=display_name,
                value=Mask(mask=mask_data)
            )

            # Build MALPrediction object and add to list
            mal_prediction = lb.MALPrediction(
                data_row=data_row,
                annotations=[object_annotation]
            )
            predictions_sdk.append(mal_prediction)

        print(f"Built {len(predictions_sdk)} MALPrediction objects.")
        # Create import job with SDK objects
        upload_job = lb.MALPredictionImport.create_from_objects(
            client=client,
            project_id=PROJECT_ID,
            name=f"DirectImport_{TIMESTAMP}",
            predictions=predictions_sdk
        )
        print(f'Import started with ID: {upload_job.uid}')
        
        # Monitor progress
        start_time = time.time()
        max_wait_time = 5 * 60  # 5 minutes
        
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
                    if hasattr(upload_job, 'errors'):
                        for i, error in enumerate(upload_job.errors):
                            print(f'Error {i+1}: {error}')
                    elif hasattr(upload_job, 'error'):
                        print(f'Error: {upload_job.error}')
                    break
            
            time.sleep(10)
    except Exception as e:
        print(f'Error: {str(e)}')
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 