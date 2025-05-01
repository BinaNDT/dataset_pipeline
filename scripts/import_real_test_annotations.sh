#!/bin/bash
# Import REAL test annotations to Labelbox with enhanced debugging

# Set timestamp for import name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export TIMESTAMP  # Export so it's available in the Python subprocess

echo "Importing REAL test annotations to Labelbox..."
python -c "
import os
import json
import labelbox as lb
from dotenv import load_dotenv
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()
API_KEY = os.getenv('LABELBOX_API_KEY')
PROJECT_ID = os.getenv('LABELBOX_PROJECT_ID')
TIMESTAMP = os.environ.get('TIMESTAMP')  # Get TIMESTAMP from environment

logging.info(f'Using project ID: {PROJECT_ID}')
logging.info(f'Using timestamp: {TIMESTAMP}')

# Connect to Labelbox
client = lb.Client(api_key=API_KEY)

try:
    # Test authentication by getting user info
    user = client.get_user()
    logging.info(f'Successfully authenticated as: {user.email}')
    
    # Get project info
    project = client.get_project(PROJECT_ID)
    logging.info(f'Connected to project: {project.name}')
    
    # Get ontology (schema)
    ontology = project.ontology()
    logging.info(f'Project ontology ID: {ontology.uid}')
    
    # Print ontology details for debugging
    logging.debug('Project ontology schema:')
    schema_json = ontology.asdict()
    for tool in schema_json.get('tools', []):
        logging.debug(f'Tool: {tool.get(\"name\")} - {tool.get(\"tool\")}')
        for feature in tool.get('classifications', []):
            logging.debug(f'  - Feature: {feature.get(\"name\")}')
    
    # Print available classes
    classes = []
    for tool in schema_json.get('tools', []):
        if tool.get('tool') == 'polygon':
            for feature in tool.get('classifications', []):
                if feature.get('type') == 'radio':
                    for option in feature.get('options', []):
                        classes.append(option.get('value'))
            
    logging.info(f'Available classes: {classes}')
    
    # Prepare annotations
    annotations_file = Path('outputs/predictions/real_test_annotations.ndjson')
    logging.info(f'Using annotations file: {annotations_file}')
    
    # Try to create DataRow objects for each annotation
    predictions = []
    
    with open(annotations_file, 'r') as f:
        for line in f:
            ann_entry = json.loads(line.strip())
            data_row_id = ann_entry['dataRow']['id']
            
            # Debug data row ID
            logging.debug(f'Processing data row ID: {data_row_id}')
            
            # Create annotations
            annotations = []
            for ann in ann_entry['annotations']:
                # Log annotation details for debugging
                logging.debug(f'Processing annotation: {ann.get(\"name\")}')
                logging.debug(f'Points count: {len(ann[\"value\"][\"points\"])}')
                
                # Convert points to Labelbox format
                points = [{'x': p[0], 'y': p[1]} for p in ann['value']['points']]
                
                # Create annotation using the correct format for Labelbox SDK
                annotation = {
                    'schemaId': ann.get('schemaId', ann['name']),  # Use name as schemaId if not provided
                    'annotationName': ann['name'],  # Keep the original name
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
                logging.debug(f'Added prediction with {len(annotations)} annotations')
    
    # Create import
    if predictions:
        try:
            logging.info(f'Starting import with {len(predictions)} predictions')
            
            # Print first prediction for debugging
            if predictions:
                logging.debug(f'First prediction sample: {json.dumps(predictions[0], indent=2)}')
            
            # Create import using the correct format
            upload_job = lb.MALPredictionImport.create_from_objects(
                client=client,
                project_id=PROJECT_ID,
                name=f'Real_Test_Import_{TIMESTAMP}',
                predictions=predictions
            )
            logging.info(f'Import started with ID: {upload_job.uid}')
            
            # Monitor progress
            import time
            start_time = time.time()
            max_wait_time = 5 * 60  # 5 minutes
            
            while time.time() - start_time < max_wait_time:
                upload_job.refresh()
                
                if hasattr(upload_job, 'state'):
                    state = upload_job.state
                    progress = upload_job.progress if hasattr(upload_job, 'progress') else 'unknown'
                    
                    logging.info(f'Status: {state}, Progress: {progress}')
                    
                    if state in ('COMPLETE', 'FINISHED'):
                        logging.info('Import completed successfully!')
                        break
                    elif state in ('FAILED', 'ERROR'):
                        logging.error('Import failed!')
                        # Check for error details
                        if hasattr(upload_job, 'error'):
                            logging.error(f'Error details: {upload_job.error}')
                        break
                
                time.sleep(10)
        except Exception as e:
            logging.error(f'Error during import: {str(e)}')
            import traceback
            logging.error(traceback.format_exc())
    else:
        logging.warning('No predictions to upload')
except Exception as e:
    logging.error(f'Error: {str(e)}')
    import traceback
    logging.error(traceback.format_exc())
"

echo "Process complete!" 