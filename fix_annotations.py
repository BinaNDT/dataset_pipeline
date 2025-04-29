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
    # Use dataset query to get data rows from project
    dataset_query = f"""
    {{
      project(where: {{ id: "{LABELBOX_PROJECT_ID}" }}) {{
        datasets {{
          id
          name
          dataRows(skip: 0, first: 100) {{
            id
            externalId
            rowData
          }}
        }}
      }}
    }}
    """
    
    result = client.execute(dataset_query)
    data_rows = []
    
    # Extract data rows from all datasets in the project
    try:
        for dataset in result["project"]["datasets"]:
            logging.info(f"Found dataset: {dataset['name']}")
            dataset_id = dataset['id']
            
            # Fetch all data rows in batches of 100
            skip = 0
            while True:
                batch_query = f"""
                {{
                  dataset(where: {{ id: "{dataset_id}" }}) {{
                    dataRows(skip: {skip}, first: 100) {{
                      id
                      externalId
                    }}
                  }}
                }}
                """
                
                batch_result = client.execute(batch_query)
                batch_rows = batch_result.get("dataset", {}).get("dataRows", [])
                
                if not batch_rows:
                    break
                    
                data_rows.extend(batch_rows)
                skip += 100
                
                if len(batch_rows) < 100:
                    break
                    
    except (KeyError, TypeError) as e:
        logging.error(f"Error processing datasets: {e}")
        logging.error(f"API response: {result}")
        return
    
    logging.info(f"Found {len(data_rows)} images in the project")
    
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
    
    # 4. Build filename to data_row mapping
    filename_to_row = {}
    
    for row in data_rows:
        # Try to extract filename from externalId
        if row.get("externalId") and '.png' in row["externalId"].lower():
            filename = row["externalId"]
            filename_to_row[filename] = row["id"]
        else:
            # Skip rows we can't identify
            continue
    
    logging.info(f"Mapped {len(filename_to_row)} images to Labelbox data rows")
    logging.info(f"Filenames found: {list(filename_to_row.keys())}")
    
    # 5. Build image ID to filename mapping from COCO
    image_id_to_filename = {}
    for image in coco_data['images']:
        image_id_to_filename[image['id']] = image['file_name']
    
    # Print first few filenames from COCO to help debug
    logging.info(f"COCO filenames sample: {list(image_id_to_filename.values())[:10]}")
    
    # 6. Build filename to annotations mapping
    filename_to_annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id in image_id_to_filename:
            filename = image_id_to_filename[image_id]
            if filename not in filename_to_annotations:
                filename_to_annotations[filename] = []
            filename_to_annotations[filename].append(ann)
    
    # 7. Create annotations in Labelbox ndjson format
    annotations_list = []
    matched_count = 0
    
    for filename, data_row_id in filename_to_row.items():
        if filename not in filename_to_annotations:
            logging.info(f"No annotations found for image: {filename}")
            continue
            
        matched_count += 1
        annotations = filename_to_annotations[filename]
        
        # Create annotation entries for this image
        image_annotations = []
        
        for ann in annotations:
            # Get category name
            cat_id = ann['category_id']
            cat_name = next((c['name'] for c in coco_data['categories'] if c['id'] == cat_id), None)
            
            if not cat_name:
                continue
                
            # Create polygon annotation
            polygon_points = []
            for i in range(0, len(ann['segmentation'][0]), 2):
                x = ann['segmentation'][0][i]
                y = ann['segmentation'][0][i+1]
                polygon_points.append([x, y])
            
            # Add to image annotations
            image_annotations.append({
                "uuid": str(ann['id']),  # Use annotation ID as UUID
                "name": cat_name,
                "value": {
                    "format": "polygon2d",
                    "points": polygon_points
                }
            })
        
        # Create entry for this image
        if image_annotations:
            annotations_list.append({
                "uuid": str(time.time()) + "_" + filename,
                "dataRow": {
                    "id": data_row_id
                },
                "annotations": image_annotations
            })
    
    logging.info(f"Created annotations for {matched_count} images")
    
    if not annotations_list:
        logging.error("No annotations to upload. Check that filenames match between Labelbox and COCO.")
        return
    
    # 8. Save annotations to ndjson file
    ndjson_file = PREDICTIONS_DIR / f"fixed_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ndjson"
    
    with open(ndjson_file, 'w') as f:
        for ann in annotations_list:
            f.write(json.dumps(ann) + '\n')
    
    logging.info(f"Saved annotations to {ndjson_file}")
    
    # 9. Upload annotations
    logging.info("Uploading annotations to Labelbox...")
    
    # Create a unique import name
    import_name = f"Fixed_Import_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Upload the annotations file
        mal_import = lb.MALPredictionImport.create_from_file(
            client=client,
            project_id=LABELBOX_PROJECT_ID,
            name=import_name,
            predictions_filepath=str(ndjson_file)
        )
        
        logging.info(f"Upload started with job ID: {mal_import.uid}")
        logging.info("Monitoring upload progress...")
        
        # Monitor progress
        start_time = time.time()
        max_wait_time = 5 * 60  # 5 minutes
        
        while time.time() - start_time < max_wait_time:
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
                    break
            
            time.sleep(10)
        
        logging.info("Process completed!")
        logging.info("Check the Labelbox UI to confirm annotations are visible.")
        
    except Exception as e:
        logging.error(f"Error uploading annotations: {e}")
        logging.error(f"Exception details: {str(e)}")

if __name__ == "__main__":
    main() 