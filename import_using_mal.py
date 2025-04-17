import json
import sys
import logging
import uuid
from pathlib import Path
from datetime import datetime

# Setup path to import from config
sys.path.append(str(Path(__file__).parent))
from config import *

# Import Labelbox SDK - matching their example
import labelbox as lb
import labelbox.data.annotation_types as lb_types

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOGS_DIR / 'labelbox_mal_import.log'),
            logging.StreamHandler()
        ]
    )

def main():
    setup_logging()
    
    # Check for the NDJSON file
    ndjson_files = list(PREDICTIONS_DIR.glob("labelbox_ndjson_*.ndjson"))
    if not ndjson_files:
        logging.error("No NDJSON files found in predictions directory")
        return
    
    # Sort by modification time and get the latest
    ndjson_file = sorted(ndjson_files, key=lambda f: f.stat().st_mtime)[-1]
    logging.info(f"Using NDJSON file: {ndjson_file}")
    
    # Initialize Labelbox client
    client = lb.Client(api_key=LABELBOX_API_KEY)
    logging.info("Connected to Labelbox")
    
    try:
        # Get the project
        project = client.get_project(LABELBOX_PROJECT_ID)
        logging.info(f"Connected to project: {project.name}")
        
        # Get dataset
        datasets = [ds for ds in client.get_datasets() if "Building_Damage_DEBUG" in ds.name]
        if not datasets:
            logging.error("No Building_Damage datasets found. Please run the uploader first.")
            return
        
        # Sort by creation date (newest first)
        datasets.sort(key=lambda x: x.created_at, reverse=True)
        dataset = datasets[0]
        logging.info(f"Found dataset: {dataset.name}")
        
        # Get data rows from the dataset
        data_rows = list(dataset.data_rows())
        logging.info(f"Found {len(data_rows)} data rows in dataset")
        
        if not data_rows:
            logging.error("No data rows found in dataset. Cannot create annotations.")
            return
        
        # Load the NDJSON file to extract annotation data
        with open(ndjson_file, 'r') as f:
            ndjson_annotations = [json.loads(line) for line in f]
        
        logging.info(f"Loaded {len(ndjson_annotations)} annotations from NDJSON file")
        
        # Create a data row ID to global key mapping
        data_row_map = {dr.external_id: dr.uid for dr in data_rows}
        logging.info(f"Created mapping for {len(data_row_map)} data rows")
        
        # Group annotations by data row ID
        annotation_by_data_row = {}
        for anno in ndjson_annotations:
            data_row_id = anno.get("dataRow", {}).get("id")
            if data_row_id:
                if data_row_id not in annotation_by_data_row:
                    annotation_by_data_row[data_row_id] = []
                annotation_by_data_row[data_row_id].append(anno)
        
        logging.info(f"Grouped annotations for {len(annotation_by_data_row)} data rows")
        
        # Convert the annotations to Labelbox annotation types
        all_labels = []
        for data_row in data_rows:
            # Get the external ID (from uploader)
            external_id = data_row.external_id
            
            # Path from external ID
            image_path = data_row.row_data
            
            # Skip if no annotations for this data row
            if data_row.uid not in annotation_by_data_row:
                continue
            
            # Get annotations for this data row
            row_annotations = annotation_by_data_row[data_row.uid]
            
            # Convert annotation points to Polygon objects
            lb_annotations = []
            for anno in row_annotations:
                # Get annotation details
                anno_data = anno.get("annotations", [])[0] if anno.get("annotations", []) else {}
                if not anno_data:
                    continue
                
                # Get class name
                class_name = anno_data.get("name")
                
                if not class_name:
                    continue
                
                # Get polygon points
                polygon_data = anno_data.get("value", {})
                if polygon_data.get("format") != "polygon2d":
                    continue
                
                points = polygon_data.get("points", [])
                if not points:
                    continue
                
                # Create a polygon annotation
                try:
                    polygon = lb_types.Polygon(points=points)
                    obj_annotation = lb_types.ObjectAnnotation(
                        name=class_name,
                        value=polygon
                    )
                    lb_annotations.append(obj_annotation)
                except Exception as e:
                    logging.error(f"Error creating annotation for {external_id}: {str(e)}")
            
            if lb_annotations:
                # Create a label
                label = lb_types.Label(
                    data={"global_key": external_id},
                    annotations=lb_annotations
                )
                all_labels.append(label)
        
        logging.info(f"Created {len(all_labels)} labels")
        
        if all_labels:
            # Import as MAL predictions
            try:
                import_name = f"Building_Damage_Import_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                logging.info(f"Starting MAL import with name: {import_name}")
                
                # Upload MAL predictions
                upload_job = lb.MALPredictionImport.create_from_objects(
                    client=client,
                    project_id=project.uid,
                    name=import_name,
                    predictions=all_labels
                )
                
                logging.info("Waiting for import to complete...")
                upload_job.wait_till_done()
                
                logging.info(f"Import completed with status: {upload_job.status}")
                
                if hasattr(upload_job, 'errors') and upload_job.errors:
                    logging.error(f"Import had errors: {upload_job.errors}")
                else:
                    logging.info("Import completed successfully!")
                    logging.info(f"You can view your annotations at: https://app.labelbox.com/projects/{LABELBOX_PROJECT_ID}")
            except Exception as e:
                logging.error(f"Error during MAL import: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            logging.warning("No valid labels to import")
        
    except Exception as e:
        logging.error(f"Error during import preparation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 