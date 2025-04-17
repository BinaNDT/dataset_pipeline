import json
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
from labelbox import Client
from labelbox.data.annotation_types import ObjectAnnotation, Label, Polygon
import sys
import datetime
import time
import uuid

sys.path.append(str(Path(__file__).parent))
from config import *

# Debug mode will limit uploads
DEBUG_MODE = True
DEBUG_UPLOAD_LIMIT = 10  # Max images to upload in debug mode

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOGS_DIR / 'labelbox_upload.log'),
            logging.StreamHandler()
        ]
    )

def mask_to_polygon(mask: np.ndarray) -> list:
    """Convert binary mask to polygon format"""
    from skimage import measure
    
    # Find contours
    contours = measure.find_contours(mask.astype(np.uint8), 0.5)
    
    polygons = []
    for contour in contours:
        # Simplify contour to reduce points
        if len(contour) > 3:  # Need at least 3 points for a polygon
            # Swap x,y coordinates to match Labelbox format
            polygon = [[float(x), float(y)] for y, x in contour]
            polygons.append(polygon)
    
    return polygons

def create_labelbox_annotations(predictions_file: Path) -> list:
    """Convert model predictions to Labelbox format"""
    with open(predictions_file, 'r') as f:
        predictions_by_video = json.load(f)
    
    labelbox_annotations = []
    
    for video_name, frame_predictions in tqdm(predictions_by_video.items(), desc="Converting predictions"):
        for frame_pred in frame_predictions:
            image_path = frame_pred['image_path']
            
            # Skip frames with errors
            if 'error' in frame_pred:
                logging.warning(f"Skipping frame with error: {image_path}")
                continue
            
            # Create annotations array for this image
            annotations = []
            
            # Create annotations for each prediction
            for pred in frame_pred['predictions']:
                mask = np.array(pred['mask'])
                polygons = mask_to_polygon(mask)
                
                if polygons:  # Only add if valid polygons were found
                    # For each polygon in the mask, create an annotation
                    for poly_points in polygons:
                        annotations.append({
                            'name': pred['class_name'],
                            'value': pred['class_name'],
                            'confidence': pred['confidence'],
                            'polygon': poly_points
                        })
            
            if annotations:  # Only add frames with predictions
                labelbox_annotations.append({
                    'data_row_id': Path(image_path).stem,
                    'external_id': image_path,
                    'annotations': annotations
                })
    
    return labelbox_annotations

def convert_to_labelbox_annotations(annotations):
    """Convert our annotation format to Labelbox's NDJSON format"""
    ndjson_annotations = []
    
    for item in annotations:
        image_path = item['external_id']
        external_id = Path(image_path).stem
        
        # Create a unique annotation ID
        annotation_id = str(uuid.uuid4())
        
        # Create base annotation object
        annotation = {
            "uuid": annotation_id,
            "dataRow": {
                "id": external_id
            },
            "annotations": []
        }
        
        # Add each polygon annotation
        for anno in item['annotations']:
            # Create a unique object ID
            object_id = str(uuid.uuid4())
            
            # Get the polygon points
            polygon_points = anno['polygon']
            
            # Convert to Labelbox format
            polygon_annotation = {
                "uuid": object_id,
                "name": anno['name'],
                "confidence": anno['confidence'],
                "value": {
                    "format": "polygon2d",
                    "points": polygon_points
                }
            }
            
            annotation["annotations"].append(polygon_annotation)
        
        ndjson_annotations.append(annotation)
    
    return ndjson_annotations

def setup_labelbox_project(client) -> str:
    """Setup Labelbox project and return project ID"""
    try:
        project = client.get_project(LABELBOX_PROJECT_ID)
        logging.info(f"Successfully connected to project: {project.name}")
        return project.uid
    except Exception as e:
        logging.error(f"Error connecting to project: {str(e)}")
        raise

def list_available_projects(client):
    """List all available projects in the Labelbox account"""
    try:
        projects = client.get_projects()
        logging.info("Available projects in your Labelbox account:")
        for project in projects:
            logging.info(f"Project Name: {project.name} | Project ID: {project.uid}")
        return True
    except Exception as e:
        logging.error(f"Error listing projects: {str(e)}")
        return False

def upload_to_labelbox(annotations: list, dataset_name: str):
    """Upload annotations to Labelbox"""
    if not LABELBOX_API_KEY:
        logging.error("Labelbox API key not set")
        return
        
    logging.info("Initializing Labelbox client at 'https://api.labelbox.com/graphql'")
    client = Client(api_key=LABELBOX_API_KEY)
    
    # Try to list available projects to help the user
    logging.info("Checking available projects...")
    projects_found = list_available_projects(client)
    
    if not projects_found:
        logging.error("Could not retrieve project list. Check your API key.")
        return
    
    # Verify API key works and project exists
    try:
        # Check that we can access the project
        project_id = setup_labelbox_project(client)
        logging.info(f"Successfully connected to Labelbox project {project_id}")
    except Exception as e:
        logging.error(f"Error connecting to Labelbox: {str(e)}")
        logging.error(f"The project ID '{LABELBOX_PROJECT_ID}' may not exist or you may not have access to it.")
        logging.error("Please check the project ID in config.py against the available projects listed above.")
        return
    
    # Create dataset
    try:
        dataset = client.create_dataset(name=dataset_name)
        logging.info(f"Created dataset: {dataset_name} (ID: {dataset.uid})")
    except Exception as e:
        logging.error(f"Error creating dataset: {str(e)}")
        return
    
    # Upload data rows with annotations
    try:
        logging.info(f"Uploading {len(annotations)} data rows to dataset...")
        
        # Prepare list of data rows to create
        data_row_items = []
        data_row_mapping = {}  # Map external_id to data_row_id
        
        for item in annotations:
            external_id = Path(item['external_id']).stem
            data_row_items.append({
                'row_data': item['external_id'],  # URL or filepath 
                'external_id': external_id
            })
        
        # Create data rows in batch
        task = dataset.create_data_rows(data_row_items)
        task.wait_till_done()  # Wait for upload to complete
        logging.info(f"Successfully uploaded data rows to dataset")
        
        # Get the data rows to build the mapping
        data_rows = list(dataset.data_rows())
        for data_row in data_rows:
            data_row_mapping[data_row.external_id] = data_row.uid
        
        # Save NDJSON format for manual import
        ndjson_file = PREDICTIONS_DIR / f"labelbox_ndjson_{dataset_name}.ndjson"
        with open(ndjson_file, 'w') as f:
            for item in annotations:
                external_id = Path(item['external_id']).stem
                data_row_id = data_row_mapping.get(external_id)
                
                if not data_row_id:
                    logging.warning(f"Could not find data row ID for {external_id}")
                    continue
                
                for anno in item['annotations']:
                    # Create annotation entry
                    annotation = {
                        "uuid": str(uuid.uuid4()),
                        "dataRow": {
                            "id": data_row_id
                        },
                        "annotations": [
                            {
                                "uuid": str(uuid.uuid4()),
                                "name": anno['name'],
                                "value": {
                                    "format": "polygon2d",
                                    "points": anno['polygon']
                                }
                            }
                        ]
                    }
                    f.write(json.dumps(annotation) + '\n')
        
        logging.info(f"Saved annotations in NDJSON format to {ndjson_file}")
        
        # Link dataset to project
        try:
            dataset.add_to_project(LABELBOX_PROJECT_ID)
            logging.info(f"Dataset has been linked to project {LABELBOX_PROJECT_ID}")
        except Exception as e:
            logging.warning(f"Error connecting dataset to project: {str(e)}")
            logging.warning("You may need to manually add the dataset to your project:")
            logging.warning(f"1. Go to https://app.labelbox.com/projects/{LABELBOX_PROJECT_ID}")
            logging.warning(f"2. Click 'Data' > 'Add data' > 'From dataset'")
            logging.warning(f"3. Select the dataset: {dataset_name} (ID: {dataset.uid})")
        
        logging.info("=" * 80)
        logging.info("IMPORTANT: To import your annotations:")
        logging.info("1. Go to https://app.labelbox.com/projects/{LABELBOX_PROJECT_ID}")
        logging.info("2. Click 'Label' > 'Import'")
        logging.info("3. Select 'Pre-labeled data'")
        logging.info("4. Upload the file: " + str(ndjson_file))
        logging.info("=" * 80)
        
        logging.info("Dataset and data rows have been created successfully!")
        logging.info(f"You can view your dataset at: https://app.labelbox.com/data/{dataset.uid}")
        
    except Exception as e:
        logging.error(f"Error during upload: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    setup_logging()
    
    # Check API credentials
    if LABELBOX_API_KEY == "YOUR_LABELBOX_API_KEY_HERE" or LABELBOX_PROJECT_ID == "YOUR_LABELBOX_PROJECT_ID_HERE":
        logging.error("Labelbox API key or project ID not set in config.py.")
        logging.error("Please edit config.py and replace the placeholder values with your actual Labelbox credentials.")
        logging.error("API Key: Should look like 'lb_xxx...'")
        logging.error("Project ID: Should look like 'cls_xxx...'")
        return
    
    # Check for predictions file
    predictions_file = PREDICTIONS_DIR / 'predictions.json'
    if not predictions_file.exists():
        raise FileNotFoundError(f"Predictions file not found at {predictions_file}")
    
    # Convert predictions to Labelbox format
    logging.info("Converting predictions to Labelbox format...")
    labelbox_annotations = create_labelbox_annotations(predictions_file)
    
    # Apply debug limits if needed
    if DEBUG_MODE and len(labelbox_annotations) > DEBUG_UPLOAD_LIMIT:
        original_count = len(labelbox_annotations)
        labelbox_annotations = labelbox_annotations[:DEBUG_UPLOAD_LIMIT]
        logging.warning(f"DEBUG MODE: Limiting upload to {DEBUG_UPLOAD_LIMIT} images (out of {original_count})")
    
    logging.info(f"Prepared {len(labelbox_annotations)} images for upload")
    
    # Save annotations locally as backup
    backup_file = PREDICTIONS_DIR / 'labelbox_annotations.json'
    with open(backup_file, 'w') as f:
        json.dump(labelbox_annotations, f)
    logging.info(f"Saved backup of Labelbox annotations to {backup_file}")
    
    # Upload to Labelbox
    if labelbox_annotations:
        if LABELBOX_API_KEY and LABELBOX_PROJECT_ID:
            # Create a unique dataset name with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            mode_tag = "DEBUG" if DEBUG_MODE else "FULL"
            dataset_name = f"Building_Damage_{mode_tag}_{timestamp}"
            
            logging.info(f"Uploading {len(labelbox_annotations)} annotations to Labelbox dataset '{dataset_name}'...")
            upload_to_labelbox(labelbox_annotations, dataset_name)
        else:
            logging.warning("Labelbox API key or project ID not set. Skipping upload.")
    else:
        logging.warning("No annotations to upload. The model likely didn't detect any objects.")

if __name__ == '__main__':
    main() 