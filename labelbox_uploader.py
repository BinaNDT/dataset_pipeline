import json
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
from labelbox import Client, OntologyBuilder
from labelbox.schema.bulk_import_request import BulkImportRequest
from labelbox.schema.enums import DataRowState
import sys
import datetime

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
                
            # Create annotations for each prediction
            annotations = []
            for pred in frame_pred['predictions']:
                mask = np.array(pred['mask'])
                polygons = mask_to_polygon(mask)
                
                if polygons:  # Only add if valid polygons were found
                    annotations.append({
                        'name': pred['class_name'],
                        'value': pred['class_name'],
                        'confidence': pred['confidence'],
                        'polygon': polygons[0] if len(polygons) == 1 else polygons
                    })
            
            if annotations:  # Only add frames with predictions
                labelbox_annotations.append({
                    'data_row_id': Path(image_path).stem,
                    'external_id': image_path,
                    'annotations': annotations
                })
    
    return labelbox_annotations

def setup_labelbox_project() -> tuple:
    """Setup Labelbox project and ontology"""
    client = Client(api_key=LABELBOX_API_KEY)
    project = client.get_project(LABELBOX_PROJECT_ID)
    
    # Create ontology
    ontology_builder = OntologyBuilder()
    
    # Add damage level classifications
    for class_name in CLASS_NAMES[1:]:  # Skip background class
        ontology_builder.add_tool(
            tool='polygon',
            name=class_name,
            color=CLASS_COLORS[class_name]
        )
    
    # Update project ontology
    project.setup_editor({
        "tools": ontology_builder.asdict()["tools"],
        "classifications": [],
        "relationships": []
    })
    
    return client, project

def upload_to_labelbox(annotations: list, dataset_name: str):
    """Upload annotations to Labelbox"""
    if not LABELBOX_API_KEY or not LABELBOX_PROJECT_ID:
        logging.error("Labelbox API key or project ID not set")
        return
        
    client, project = setup_labelbox_project()
    
    # Create upload job
    upload_job = BulkImportRequest.create(
        client=client,
        project_id=LABELBOX_PROJECT_ID,
        name=dataset_name,
        attachments=annotations
    )
    
    # Start upload
    upload_job.start()
    logging.info(f"Started upload job {upload_job.uid}")
    
    # Wait for completion
    upload_job.wait_until_done()
    results = upload_job.get_status()
    
    # Log results
    logging.info(f"Upload completed with status: {results.state}")
    if results.state == DataRowState.ERROR:
        logging.error(f"Errors during upload: {results.errors}")
    else:
        logging.info(f"Successfully uploaded {len(annotations)} annotations")

def main():
    setup_logging()
    
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