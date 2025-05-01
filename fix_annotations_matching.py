#!/usr/bin/env python3
"""
Fix Annotations by Matching URLs

This script uses the image URL path to match Labelbox data rows to COCO annotations.
"""

import os
import sys
import json
import time
import logging
import re
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

def get_data_rows_from_project():
    """Get all data rows from the project using pagination and better error handling"""
    client = lb.Client(api_key=LABELBOX_API_KEY)
    
    # Get all datasets in the project using GraphQL
    project_with_datasets_query = f"""
    {{
      project(where: {{ id: "{LABELBOX_PROJECT_ID}" }}) {{
        datasets {{
          id
          name
        }}
      }}
    }}
    """
    
    result = client.execute(project_with_datasets_query)
    datasets = result.get("project", {}).get("datasets", [])
    logging.info(f"Found {len(datasets)} datasets in the project")
    
    # Initialize to store all data rows
    all_data_rows = []
    
    # Get data rows from each dataset
    for dataset in datasets:
        dataset_name = dataset.get("name")
        dataset_id = dataset.get("id")
        
        logging.info(f"Fetching rows from dataset: {dataset_name} (ID: {dataset_id})")
        
        # Get all data rows for this dataset using pagination
        skip = 0
        page_size = 100
        total_rows = 0
        
        while True:
            try:
                # Use GraphQL to get data rows
                data_row_query = f"""
                {{
                  dataset(where: {{ id: "{dataset_id}" }}) {{
                    dataRows(skip: {skip}, first: {page_size}) {{
                      id
                      externalId
                      rowData
                    }}
                  }}
                }}
                """
                
                data_row_result = client.execute(data_row_query)
                data_rows = data_row_result.get("dataset", {}).get("dataRows", [])
                
                if not data_rows:
                    logging.info(f"No more rows in dataset {dataset_name}")
                    break
                    
                total_rows += len(data_rows)
                all_data_rows.extend(data_rows)
                
                if len(data_rows) < page_size:
                    logging.info(f"Retrieved all {total_rows} rows from dataset {dataset_name}")
                    break
                    
                skip += page_size
                logging.info(f"Retrieved {total_rows} rows so far from dataset {dataset_name}")
                
            except Exception as e:
                logging.error(f"Error fetching data rows from dataset {dataset_name}: {e}")
                break
    
    logging.info(f"Found {len(all_data_rows)} total data rows across all datasets")
    return all_data_rows

def main():
    """Main function to fix annotations"""
    
    # 1. Connect to Labelbox
    logging.info("Connecting to Labelbox...")
    client = lb.Client(api_key=LABELBOX_API_KEY)
    
    # 2. Get existing images in the project using the improved function
    logging.info("Fetching existing images in the project...")
    all_data_rows = get_data_rows_from_project()
    
    if not all_data_rows:
        logging.error("No data rows found in the project. Cannot proceed.")
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
    
    # 4. Create a more comprehensive mapping from filenames to data row IDs
    url_to_row_id = {}
    
    # A. Extract filename from URLs, handling various formats
    filename_pattern = re.compile(r'(\d{6,7}\.(?:png|jpg|jpeg))', re.IGNORECASE)
    
    for row in all_data_rows:
        row_id = row.get("id")  # Use get() method since we're working with dict objects now
        
        # Handle row_data from GraphQL response
        row_data = row.get("rowData")
        
        if row_data:
            # Convert to string if not already
            url = str(row_data)
            
            # Try regex pattern first
            match = filename_pattern.search(url)
            if match:
                filename = match.group(1)
                url_to_row_id[filename] = row_id
                # Also store without extension
                name_without_ext = Path(filename).stem
                url_to_row_id[name_without_ext] = row_id
                continue
            
            # Extract filename from the end of the URL
            url_parts = url.split('?')[0].split('/')
            if url_parts:
                last_part = url_parts[-1]
                # If it contains an image extension, use it
                if any(ext in last_part.lower() for ext in ['.png', '.jpg', '.jpeg']):
                    url_to_row_id[last_part] = row_id
                    # Also store without extension
                    name_without_ext = Path(last_part).stem
                    url_to_row_id[name_without_ext] = row_id
    
    # B. Extract from external IDs when available
    for row in all_data_rows:
        row_id = row.get("id")  # Use get() method since we're working with dict objects now
        
        # Get external ID if available
        external_id = row.get("externalId")
        
        if external_id:
            # Store the external ID directly
            url_to_row_id[external_id] = row_id
            
            # If it looks like a number, try formatted versions
            if external_id.isdigit():
                # Try different zero-padded formats
                for padding in [6, 7]:
                    padded_id = external_id.zfill(padding)
                    url_to_row_id[padded_id] = row_id
                    url_to_row_id[f"{padded_id}.png"] = row_id
    
    # C. Check if the URL contains any of the COCO image filenames
    for row in all_data_rows:
        row_id = row.get("id")  # Use get() method since we're working with dict objects now
        
        # Handle row_data from GraphQL response
        row_data = row.get("rowData")
        
        if row_data:
            url = str(row_data)
            
            # Check if URL contains image filename
            for image in coco_data.get("images", []):
                image_filename = image.get("file_name")
                if image_filename and image_filename in url:
                    url_to_row_id[image_filename] = row_id
                    # Also try without extension
                    name_without_ext = Path(image_filename).stem
                    url_to_row_id[name_without_ext] = row_id
    
    logging.info(f"Mapped {len(url_to_row_id)} image filenames to data rows")
    
    # 5. Build image ID to filename mapping from COCO
    image_id_to_filename = {}
    for image in coco_data.get("images", []):
        image_id_to_filename[image.get("id")] = image.get("file_name")
    
    # 6. Match annotations to data rows
    annotations_list = []
    matched_count = 0
    unmatched_count = 0
    
    for image_id, filename in image_id_to_filename.items():
        # Try to find data row ID for this image
        data_row_id = None
        
        # Try different variations of the filename
        for key in [filename, Path(filename).stem, f"{Path(filename).stem}.png"]:
            if key in url_to_row_id:
                data_row_id = url_to_row_id[key]
                break
        
        if not data_row_id:
            unmatched_count += 1
            logging.warning(f"No matching data row found for image: {filename}")
            continue
            
        # Find annotations for this image
        image_annotations = []
        
        # Find annotations for this image ID
        for ann in coco_data.get("annotations", []):
            if ann.get("image_id") == image_id:
                # Get category name
                cat_id = ann.get("category_id")
                cat_name = next((c.get("name") for c in coco_data.get("categories", []) 
                               if c.get("id") == cat_id), None)
                
                if not cat_name:
                    continue
                    
                # Create polygon annotation
                try:
                    segmentation = ann.get("segmentation", [[]])
                    if not segmentation or not segmentation[0]:
                        continue
                        
                    polygon_points = []
                    for i in range(0, len(segmentation[0]), 2):
                        if i+1 < len(segmentation[0]):  # Ensure we have both x and y
                            x = segmentation[0][i]
                            y = segmentation[0][i+1]
                            polygon_points.append([x, y])
                    
                    # Skip polygons with too few points
                    if len(polygon_points) < 3:
                        logging.warning(f"Skipping polygon with fewer than 3 points for {filename}")
                        continue
                    
                    # Add to image annotations
                    image_annotations.append({
                        "uuid": str(ann.get("id", time.time())),
                        "name": cat_name,
                        "value": {
                            "format": "polygon2d",
                            "points": polygon_points
                        }
                    })
                except Exception as e:
                    logging.error(f"Error processing annotation for {filename}: {e}")
                    continue
        
        # Create entry for this image
        if image_annotations:
            matched_count += 1
            annotations_list.append({
                "uuid": str(time.time()) + "_" + filename,
                "dataRow": {
                    "id": data_row_id
                },
                "annotations": image_annotations
            })
    
    logging.info(f"Created annotations for {matched_count} images")
    logging.info(f"Could not match {unmatched_count} images to data rows")
    
    if not annotations_list:
        logging.error("No annotations matched to data rows")
        return
    
    # 7. Save annotations to ndjson file
    ndjson_file = PREDICTIONS_DIR / f"fixed_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ndjson"
    
    with open(ndjson_file, 'w') as f:
        for ann in annotations_list:
            f.write(json.dumps(ann) + '\n')
    
    logging.info(f"Saved annotations to {ndjson_file}")
    
    # 8. Upload annotations
    logging.info("Uploading annotations to Labelbox...")
    
    # Create a unique import name
    import_name = f"Fixed_Import_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Upload the annotations file using MALPredictionImport
        mal_import = lb.MALPredictionImport.create_from_file(
            client=client,
            project_id=LABELBOX_PROJECT_ID,
            name=import_name,
            file_path=str(ndjson_file)
        )
        
        logging.info(f"Upload started with job ID: {mal_import.uid}")
        logging.info("Monitoring upload progress...")
        
        # Monitor progress
        start_time = time.time()
        max_wait_time = 5 * 60  # 5 minutes
        
        while time.time() - start_time < max_wait_time:
            try:
                mal_import.refresh()
                
                state = None
                progress = None
                
                if hasattr(mal_import, 'state'):
                    state = mal_import.state
                if hasattr(mal_import, 'progress'):
                    progress = mal_import.progress
                
                logging.info(f"Upload status: {state}, Progress: {progress}")
                
                if state in ["COMPLETE", "FINISHED"]:
                    logging.info("Upload completed successfully!")
                    break
                elif state in ["FAILED", "ERROR"]:
                    logging.error("Upload failed!")
                    break
                
                # Wait before checking again
                time.sleep(10)
                
            except Exception as e:
                logging.error(f"Error checking import status: {e}")
                time.sleep(10)
        
        logging.info("Process completed!")
        logging.info("Check the Labelbox UI to confirm annotations are visible.")
        
    except Exception as e:
        logging.error(f"Error uploading annotations: {e}")
        logging.error(f"Exception details: {str(e)}")
        
        # Try a standard LabelImport approach as fallback
        try:
            logging.info("Attempting to use LabelImport as fallback...")
            
            # For fallback, we'll try to manually construct the import using GraphQL
            # This avoids using lb.data.annotation_types which doesn't exist in this SDK version
            
            import_query = f"""
            mutation CreateLabelImport($projectId: ID!, $input: CreateLabelImportInput!) {{
                createLabelImport(projectId: $projectId, input: $input) {{
                    id
                    name
                    status
                    createdAt
                }}
            }}
            """
            
            variables = {
                "projectId": LABELBOX_PROJECT_ID,
                "input": {
                    "name": import_name,
                    "sourceType": "NDJSON_URL",
                    "source": str(ndjson_file)
                }
            }
            
            try:
                result = client.execute(import_query, variables=variables)
                logging.info(f"Label import created: {result}")
                return True
            except Exception as e:
                logging.error(f"Error creating label import: {e}")
                return False
            
        except Exception as inner_e:
            logging.error(f"Error in LabelImport attempt: {inner_e}")
            return False

if __name__ == "__main__":
    main() 