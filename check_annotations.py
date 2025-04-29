#!/usr/bin/env python3
"""
Check if annotations exist for specific data rows in Labelbox
"""

import os
import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv('LABELBOX_API_KEY')
PROJECT_ID = os.getenv('LABELBOX_PROJECT_ID')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

try:
    import labelbox as lb
except ImportError:
    logging.error("Labelbox SDK not installed. Please run: pip install labelbox")
    sys.exit(1)

def main():
    # Connect to Labelbox
    logging.info("Connecting to Labelbox...")
    client = lb.Client(api_key=API_KEY)
    
    # Read the test annotations file to get the data row IDs
    annotations_file = Path('outputs/predictions/test_annotations.ndjson')
    logging.info(f"Reading test annotations file: {annotations_file}")
    
    data_row_ids = []
    with open(annotations_file, 'r') as f:
        for line in f:
            ann_entry = json.loads(line.strip())
            data_row_id = ann_entry['dataRow']['id']
            data_row_ids.append(data_row_id)
    
    logging.info(f"Found {len(data_row_ids)} data row IDs in test annotations")
    
    # Check if a MAL import exists for the project
    logging.info(f"Checking for MAL imports in project {PROJECT_ID}...")
    
    # Get project
    project = client.get_project(PROJECT_ID)
    logging.info(f"Found project: {project.name}")
    
    # Get MAL imports for the project
    imports_query = f"""
    {{
      project(where: {{ id: "{PROJECT_ID}" }}) {{
        malPredictionImports {{
          id
          name
          createdAt
          state
          progress
        }}
      }}
    }}
    """
    
    try:
        imports_result = client.execute(imports_query)
        mal_imports = imports_result.get("project", {}).get("malPredictionImports", [])
        
        logging.info(f"Found {len(mal_imports)} MAL imports in the project")
        
        if mal_imports:
            # Sort by creation date, newest first
            mal_imports.sort(key=lambda x: x.get("createdAt", ""), reverse=True)
            
            for i, imp in enumerate(mal_imports[:5]):  # Show the 5 most recent
                imp_id = imp.get("id")
                imp_name = imp.get("name")
                imp_state = imp.get("state")
                imp_progress = imp.get("progress", "N/A")
                
                logging.info(f"Import {i+1}: {imp_name} (ID: {imp_id})")
                logging.info(f"  State: {imp_state}, Progress: {imp_progress}")
        
        # For each data row ID, check if it has annotations
        logging.info("\nChecking for annotations on specific data rows...")
        
        for i, data_row_id in enumerate(data_row_ids):
            try:
                # Query to get annotations for this data row
                annotations_query = f"""
                {{
                  dataRow(id: "{data_row_id}") {{
                    id
                    externalId
                    labeledState
                    predictions {{
                      id
                      createdAt
                    }}
                  }}
                }}
                """
                
                annotations_result = client.execute(annotations_query)
                data_row = annotations_result.get("dataRow", {})
                
                if not data_row:
                    logging.error(f"Data row {data_row_id} not found in Labelbox")
                    continue
                
                labeled_state = data_row.get("labeledState", "UNLABELED")
                predictions = data_row.get("predictions", [])
                
                logging.info(f"Data row {i+1}: {data_row_id}")
                logging.info(f"  External ID: {data_row.get('externalId', 'None')}")
                logging.info(f"  Labeled state: {labeled_state}")
                logging.info(f"  Has predictions: {len(predictions) > 0} (count: {len(predictions)})")
                
                if predictions:
                    # Show the most recent prediction
                    latest_prediction = sorted(predictions, key=lambda x: x.get("createdAt", ""), reverse=True)[0]
                    logging.info(f"  Latest prediction ID: {latest_prediction.get('id')}")
                    logging.info(f"  Created at: {latest_prediction.get('createdAt')}")
                    
                    # Query detailed information about this prediction
                    pred_query = f"""
                    {{
                      prediction(id: "{latest_prediction.get('id')}") {{
                        id
                        annotations {{
                          schemaId
                          value
                        }}
                      }}
                    }}
                    """
                    
                    try:
                        pred_result = client.execute(pred_query)
                        prediction = pred_result.get("prediction", {})
                        
                        if prediction:
                            annotations = prediction.get("annotations", [])
                            logging.info(f"  Prediction has {len(annotations)} annotations")
                            
                            # Show sample annotation details
                            if annotations and len(annotations) > 0:
                                sample = annotations[0]
                                logging.info(f"  Sample annotation schema ID: {sample.get('schemaId')}")
                    except Exception as e:
                        logging.error(f"Error fetching prediction details: {str(e)}")
            
            except Exception as e:
                logging.error(f"Error checking annotations for data row {data_row_id}: {str(e)}")
    
    except Exception as e:
        logging.error(f"Error checking MAL imports: {str(e)}")

if __name__ == "__main__":
    main() 