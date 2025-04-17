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
import labelbox as lb
import labelbox.data.annotation_types as lb_types

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
    
    # Check for the COCO file
    coco_file = PREDICTIONS_DIR / 'predictions_coco.json'
    if not coco_file.exists():
        logging.error(f"COCO file not found at {coco_file}")
        logging.error("Please run 'python dataset_pipeline/export_to_coco.py' first")
        return
    
    # Initialize Labelbox client
    client = lb.Client(api_key=LABELBOX_API_KEY)
    logging.info("Connected to Labelbox")
    
    # Get project
    try:
        project = client.get_project(LABELBOX_PROJECT_ID)
        logging.info(f"Connected to project: {project.name}")
    except Exception as e:
        logging.error(f"Failed to connect to project: {str(e)}")
        return
        
    # Load COCO data
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create a lookup from image ID to image metadata
    image_lookup = {img['id']: img for img in coco_data['images']}
    
    # Create a lookup from category ID to category name
    category_lookup = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Group annotations by image ID
    annotations_by_image = {}
    for anno in coco_data['annotations']:
        image_id = anno['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(anno)
    
    # Create Labelbox predictions
    all_labels = []
    
    for image_id, annotations in annotations_by_image.items():
        if image_id not in image_lookup:
            logging.warning(f"Image ID {image_id} not found in image lookup")
            continue
            
        image_data = image_lookup[image_id]
        global_key = Path(image_data['file_name']).stem
        
        # Process all annotations for this image
        lb_annotations = []
        
        for anno in annotations:
            category_id = anno['category_id']
            if category_id not in category_lookup:
                logging.warning(f"Category ID {category_id} not found in category lookup")
                continue
                
            category_name = category_lookup[category_id]
            
            # Get segmentation points
            if not anno['segmentation']:
                continue
                
            # Use the first polygon if there are multiple
            polygon_points = anno['segmentation'][0]
            
            # Convert flat list to list of Point objects
            points = []
            for i in range(0, len(polygon_points), 2):
                if i + 1 < len(polygon_points):
                    # Create a Point object with x, y coordinates
                    points.append(lb_types.Point(x=polygon_points[i], y=polygon_points[i+1]))
            
            try:
                # Create polygon
                if len(points) >= 3:  # Need at least 3 points for a polygon
                    polygon = lb_types.Polygon(points=points)
                    
                    # Create object annotation
                    obj_annotation = lb_types.ObjectAnnotation(
                        name=category_name,
                        value=polygon,
                        confidence=anno.get('score', 1.0)
                    )
                    
                    lb_annotations.append(obj_annotation)
            except Exception as e:
                logging.error(f"Error creating polygon for {global_key}: {str(e)}")
        
        if lb_annotations:
            # Create label
            label = lb_types.Label(
                data={"global_key": global_key},
                annotations=lb_annotations
            )
            all_labels.append(label)
    
    logging.info(f"Created {len(all_labels)} labels from COCO annotations")
    
    # Import labels to Labelbox
    if all_labels:
        try:
            # Create a unique name for the import
            import_name = f"Building_Damage_Import_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logging.info(f"Starting MAL import with name: {import_name}")
            
            # Import as model-assisted labeling predictions
            upload_job = lb.MALPredictionImport.create_from_objects(
                client=client,
                project_id=project.uid,
                name=import_name,
                predictions=all_labels
            )
            
            logging.info("Waiting for import to complete...")
            upload_job.wait_till_done()
            
            if hasattr(upload_job, 'errors') and upload_job.errors:
                logging.error(f"Import had errors: {upload_job.errors}")
            else:
                logging.info(f"Successfully imported {len(all_labels)} annotations to Labelbox")
                logging.info(f"You can view the annotations at: https://app.labelbox.com/projects/{LABELBOX_PROJECT_ID}")
        except Exception as e:
            logging.error(f"Error during Labelbox import: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        logging.warning("No valid labels to import to Labelbox")

if __name__ == "__main__":
    main() 