#!/bin/bash
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
