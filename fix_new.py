#!/usr/bin/env python3
"""
Fix Annotations script

This script ensures annotations are correctly linked to images in Labelbox
by creating a custom import that uses the image filenames to match annotations.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime

# Load environment
import labelbox as lb
import labelbox.data.annotation_types as lb_types
from dotenv import load_dotenv
load_dotenv()

# Import configuration
from config import LABELBOX_API_KEY, LABELBOX_PROJECT_ID, PREDICTIONS_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def main():
    """Main function to fix annotations"""
    
    # 1. Connect to Labelbox
    logging.info("Connecting to Labelbox...")
    client = lb.Client(api_key=LABELBOX_API_KEY)
    project = client.get_project(LABELBOX_PROJECT_ID)
    
    # 2. Get existing images in the project
    logging.info("Fetching existing images in the project...")
    data_rows = []
    
    # Use GraphQL to get datasets from project
    datasets_query = f"""
    {{
        project(where: {{ id: "{LABELBOX_PROJECT_ID}" }}) {{
            datasets {{
                id
                name
            }}
        }}
    }}
    """
    result = client.execute(datasets_query)
    
    try:
        for dataset_info in result["project"]["datasets"]:
            dataset_id = dataset_info["id"]
            dataset_name = dataset_info["name"]
            logging.info(f"Processing dataset: {dataset_name}")
            
            dataset = client.get_dataset(dataset_id)
            try:
                # Get all data rows in this dataset
                for batch in dataset.data_rows().iter_pages():
                    data_rows.extend(batch)
                    logging.info(f"Retrieved {len(batch)} data rows")
            except Exception as e:
                logging.error(f"Error retrieving data rows from dataset {dataset_name}: {e}")
    except (KeyError, Exception) as e:
        logging.error(f"Error processing datasets: {e}")
        if 'result' in locals():
            logging.error(f"API response: {result}")
        return
    
    logging.info(f"Found total of {len(data_rows)} data rows in the project")
    
    if len(data_rows) == 0:
        logging.error("No images found in the project. Please upload images first.")
        return
    
    # 3. Load COCO annotations
    coco_file = PREDICTIONS_DIR / 'predictions_coco.json'
    logging.info(f"Loading COCO annotations from {coco_file}")
    
    try:
        with open(coco_file, 'r') as f:
            coco_data = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load COCO file: {e}")
        return
    
    # Log COCO file structure for debugging
    logging.info(f"COCO categories: {[c['name'] for c in coco_data['categories']]}")
    logging.info(f"Number of COCO images: {len(coco_data['images'])}")
    logging.info(f"Number of COCO annotations: {len(coco_data['annotations'])}")
    logging.info(f"Sample COCO filenames: {[img['file_name'] for img in coco_data['images'][:5]]}")
    
    # 4. Build filename to data_row mapping - with better filename handling
    filename_to_row = {}
    external_ids = []
    global_keys = []
    
    for row in data_rows:
        # Capture all external IDs and global keys for debugging
        if hasattr(row, 'external_id') and row.external_id:
            external_ids.append(row.external_id)
        if hasattr(row, 'global_key') and row.global_key:
            global_keys.append(row.global_key)
        
        # Try multiple filename extracting approaches
        if hasattr(row, 'external_id') and row.external_id:
            # Direct match with external_id
            filename_to_row[row.external_id] = {
                "id": row.uid,
                "global_key": getattr(row, 'global_key', '') or row.external_id
            }
            
            # Also try without extension
            base_filename = os.path.splitext(row.external_id)[0]
            filename_to_row[base_filename] = {
                "id": row.uid,
                "global_key": getattr(row, 'global_key', '') or row.external_id
            }
            
            # Also try with leading zeros stripped
            if base_filename.startswith('0'):
                stripped_filename = base_filename.lstrip('0')
                if stripped_filename:  # Ensure we don't create an empty key
                    filename_to_row[stripped_filename] = {
                        "id": row.uid,
                        "global_key": getattr(row, 'global_key', '') or row.external_id
                    }
    
    logging.info(f"Mapped {len(filename_to_row)} variations of filenames to Labelbox data rows")
    logging.info(f"Sample external IDs from Labelbox: {external_ids[:5]}")
    logging.info(f"Sample global keys from Labelbox: {global_keys[:5]}")
    
    # 5. Build image ID to filename mapping from COCO
    image_id_to_filename = {}
    image_id_to_info = {}
    
    for image in coco_data['images']:
        # Store with and without extension
        image_id_to_filename[image['id']] = image['file_name']
        base_name = os.path.splitext(image['file_name'])[0]
        image_id_to_info[image['id']] = {
            "filename": image['file_name'],
            "base_name": base_name,
            "stripped_base": base_name.lstrip('0') if base_name.startswith('0') else base_name
        }
    
    # 6. Build filename to annotations mapping with multiple filename variations
    filename_to_annotations = {}
    
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id in image_id_to_info:
            info = image_id_to_info[image_id]
            
            # Add annotation to multiple filename variations for better matching
            for fname in [info['filename'], info['base_name'], info['stripped_base']]:
                if fname not in filename_to_annotations:
                    filename_to_annotations[fname] = []
                filename_to_annotations[fname].append(ann)
    
    # Debug the matching between COCO and Labelbox filenames
    coco_filenames = set(filename_to_annotations.keys())
    labelbox_filenames = set(filename_to_row.keys())
    
    common_filenames = coco_filenames.intersection(labelbox_filenames)
    logging.info(f"Found {len(common_filenames)} filename matches between COCO and Labelbox")
    logging.info(f"Sample matching filenames: {list(common_filenames)[:5]}")
    
    if not common_filenames:
        logging.error("No matching filenames found between COCO and Labelbox.")
        logging.error(f"COCO filenames: {list(coco_filenames)[:10]}")
        logging.error(f"Labelbox filenames: {list(labelbox_filenames)[:10]}")
        
        # Try direct mapping by position as fallback
        logging.info("Trying fallback direct mapping by position...")
        coco_images = coco_data['images']
        if len(coco_images) <= len(data_rows):
            direct_mapping = {}
            for i, (coco_img, lb_row) in enumerate(zip(coco_images, data_rows)):
                coco_filename = coco_img['file_name']
                logging.info(f"Mapping COCO {coco_filename} to Labelbox row {getattr(lb_row, 'external_id', 'unknown')}")
                direct_mapping[coco_filename] = {
                    "id": lb_row.uid,
                    "global_key": getattr(lb_row, 'global_key', '') or getattr(lb_row, 'external_id', '') or coco_filename
                }
            
            filename_to_row = direct_mapping
            logging.info(f"Created direct position-based mapping for {len(filename_to_row)} files")
    
    # 7. Create annotations using Labelbox types
    labels = []
    matched_count = 0
    
    # Try to match using our filename mappings
    for filename, row_info in filename_to_row.items():
        matched_annotations = None
        
        # Try to find annotations using any of our filename variations
        if filename in filename_to_annotations:
            matched_annotations = filename_to_annotations[filename]
        
        if not matched_annotations:
            continue
            
        matched_count += 1
        
        # Create annotations for this image using Labelbox types
        image_annotations = []
        
        for ann in matched_annotations:
            # Get category name
            cat_id = ann['category_id']
            cat_name = next((c['name'] for c in coco_data['categories'] if c['id'] == cat_id), None)
            
            if not cat_name:
                continue
            
            if 'segmentation' in ann and ann['segmentation']:
                # Handle segmentation mask annotation
                # For now, we'll create polygon annotations from the segmentation
                if isinstance(ann['segmentation'], list) and len(ann['segmentation']) > 0:
                    # Create points for polygon
                    points = []
                    for i in range(0, len(ann['segmentation'][0]), 2):
                        if i+1 < len(ann['segmentation'][0]):  # Ensure we have both x and y
                            x = ann['segmentation'][0][i]
                            y = ann['segmentation'][0][i+1]
                            points.append(lb_types.Point(x=x, y=y))
                    
                    if points:
                        polygon_annotation = lb_types.ObjectAnnotation(
                            name=cat_name,
                            value=lb_types.Polygon(points=points)
                        )
                        image_annotations.append(polygon_annotation)
        
        # Create a label with global key for this image
        if image_annotations:
            global_key = row_info["global_key"]
            if not global_key:  # If no global key, use filename
                global_key = filename
                
            # Add the label
            labels.append(
                lb_types.Label(
                    data={"global_key": global_key},
                    annotations=image_annotations,
                )
            )
    
    logging.info(f"Created annotations for {matched_count} images using Labelbox types")
    logging.info(f"Total labels created: {len(labels)}")
    
    if not labels:
        logging.error("No annotations to upload. Check that filenames match between Labelbox and COCO.")
        return
    
    # 8. Save annotations for debugging (optional)
    debug_file = PREDICTIONS_DIR / f"debug_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(debug_file, 'w') as f:
        # Convert to JSON-serializable structure for debugging
        debug_labels = []
        for label in labels:
            debug_labels.append({
                "global_key": label.data.get("global_key"),
                "annotation_count": len(label.annotations),
                "annotation_names": [a.name for a in label.annotations if hasattr(a, 'name')]
            })
        json.dump(debug_labels, f, indent=2)
    
    logging.info(f"Saved debug info to {debug_file}")
    
    # 9. Upload annotations
    logging.info("Uploading annotations to Labelbox...")
    
    # Create a unique import name
    import_name = f"Fixed_Import_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Upload annotations using the SDK
        mal_import = lb.MALPredictionImport.create_from_objects(
            client=client,
            project_id=LABELBOX_PROJECT_ID,
            name=import_name,
            predictions=labels
        )
        
        logging.info(f"Upload started with job ID: {mal_import.uid}")
        logging.info("Monitoring upload progress...")
        
        # Monitor progress
        start_time = time.time()
        max_wait_time = 5 * 60  # 5 minutes
        
        while time.time() - start_time < max_wait_time:
            try:
                mal_import.refresh()
                
                if hasattr(mal_import, 'state'):
                    state = mal_import.state
                    progress = mal_import.progress if hasattr(mal_import, 'progress') else "unknown"
                    
                    logging.info(f"Upload status: {state}, Progress: {progress}")
                    
                    if state == "COMPLETE" or state == "FINISHED":
                        logging.info("Upload completed successfully!")
                        break
                    elif state == "FAILED" or state == "ERROR":
                        logging.error("Upload failed!")
                        if hasattr(mal_import, 'errors') and mal_import.errors:
                            logging.error(f"Errors: {mal_import.errors}")
                        break
                
                time.sleep(10)
            except Exception as e:
                logging.error(f"Error monitoring upload: {e}")
                time.sleep(10)
        
        logging.info("Process completed!")
        logging.info("Check the Labelbox UI to confirm annotations are visible.")
        
    except Exception as e:
        logging.error(f"Error uploading annotations: {e}")
        logging.error(f"Exception details: {str(e)}")

if __name__ == "__main__":
    main() 