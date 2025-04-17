import json
import logging
import uuid
from pathlib import Path
import sys
from datetime import datetime

# Setup path to import from config
sys.path.append(str(Path(__file__).parent))
from config import *

# Import Labelbox SDK
from labelbox import Client

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOGS_DIR / 'labelbox_import.log'),
            logging.StreamHandler()
        ]
    )

def main():
    setup_logging()
    
    # Check for the NDJSON file
    # Find the most recent NDJSON file
    ndjson_files = list(PREDICTIONS_DIR.glob("labelbox_ndjson_*.ndjson"))
    if not ndjson_files:
        logging.error("No NDJSON files found in predictions directory")
        return
    
    # Sort by modification time and get the latest
    ndjson_file = sorted(ndjson_files, key=lambda f: f.stat().st_mtime)[-1]
    logging.info(f"Using NDJSON file: {ndjson_file}")
    
    # Initialize Labelbox client
    client = Client(api_key=LABELBOX_API_KEY)
    logging.info("Connected to Labelbox")
    
    try:
        # Get the project
        project = client.get_project(LABELBOX_PROJECT_ID)
        logging.info(f"Connected to project: {project.name}")
        
        # Get dataset
        logging.info("Looking for the most recent Building_Damage dataset...")
        datasets = [ds for ds in client.get_datasets() if "Building_Damage_DEBUG" in ds.name]
        if not datasets:
            logging.error("No Building_Damage datasets found. Please run the uploader first.")
            return
        
        # Sort by creation date (newest first)
        datasets.sort(key=lambda x: x.created_at, reverse=True)
        dataset = datasets[0]
        logging.info(f"Using dataset: {dataset.name} (created at {dataset.created_at})")
        
        # Try more methods to connect dataset to project
        try:
            # Try method 1: querying for batch and connecting data
            logging.info("Trying to connect dataset to project...")
            
            # Create a batch to connect the dataset
            batch_name = f"Building_Damage_Batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Get the data rows from the dataset
            data_rows = list(dataset.data_rows())
            if data_rows:
                # Get the ID or global keys of data rows
                data_row_ids = [dr.uid for dr in data_rows]
                
                # Create a batch with these data rows
                try:
                    batch = project.create_batch(
                        name=batch_name,
                        data_rows=data_row_ids[:10],  # Use the first 10 rows to test
                        priority=1  # High priority
                    )
                    logging.info(f"Created batch '{batch_name}' with {len(data_row_ids[:10])} data rows")
                except Exception as e:
                    logging.warning(f"Error creating batch: {str(e)}")
            else:
                logging.warning("No data rows found in dataset")
        
        except Exception as e:
            logging.warning(f"Error connecting dataset to project: {str(e)}")
        
        # Load annotations from NDJSON
        annotations = []
        with open(ndjson_file, 'r') as f:
            for line in f:
                annotations.append(json.loads(line))
        
        logging.info(f"Loaded {len(annotations)} annotations from {ndjson_file}")
        
        # Print manual import instructions
        logging.info("\n" + "="*80)
        logging.info("MANUAL IMPORT INSTRUCTIONS")
        logging.info("="*80)
        logging.info("Step 1: Go to https://app.labelbox.com/projects/" + LABELBOX_PROJECT_ID)
        logging.info("Step 2: Click on 'Data' tab, and look for the dataset: " + dataset.name)
        logging.info("Step 3: If your dataset is not visible, click 'Add data' > 'From dataset' and select it")
        logging.info("Step 4: Click on 'Import labels' tab")
        logging.info("Step 5: Click on 'Pre-labeled data'")
        logging.info("Step 6: Upload the NDJSON file: " + str(ndjson_file))
        logging.info("Step 7: After import completes, go to 'Label' tab to verify the annotations")
        logging.info("="*80)
        
        # Note for future implementation
        logging.info("\nNOTE: For programmatic import in future SDK updates:")
        logging.info("According to Labelbox docs, you would use code similar to:")
        logging.info("""
        upload_job = MALPredictionImport.create_from_objects(
            client=client,
            project_id=project.uid,
            name=job_name,
            predictions=labels
        )
        """)
        
    except Exception as e:
        logging.error(f"Error during setup: {str(e)}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    main() 