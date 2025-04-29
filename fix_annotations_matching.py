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

def main():
    """Main function to fix annotations"""
    
    # 1. Connect to Labelbox
    logging.info("Connecting to Labelbox...")
    client = lb.Client(api_key=LABELBOX_API_KEY)
    
    # 2. Get existing images in the project
    logging.info("Fetching existing images in the project...")
    
    # Get all datasets in the project
    project_query = f"""
    {{
      project(where: {{ id: "{LABELBOX_PROJECT_ID}" }}) {{
        datasets {{
          id
          name
          rowCount
        }}
      }}
    }}
    """
    
    result = client.execute(project_query)
    datasets = result.get("project", {}).get("datasets", [])
    
    logging.info(f"Found {len(datasets)} datasets in the project")
    
    # Initialize to store all data rows
    all_data_rows = []
    
    # Get data rows from each dataset
    for dataset in datasets:
        dataset_id = dataset.get("id")
        dataset_name = dataset.get("name")
        
        logging.info(f"Fetching rows from dataset: {dataset_name}")
        
        # Get all data rows for this dataset
        skip = 0
        batch_size = 100
        
        while True:
            data_row_query = f"""
            {{
              dataset(where: {{ id: "{dataset_id}" }}) {{
                dataRows(skip: {skip}, first: {batch_size}) {{
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
                break
                
            all_data_rows.extend(data_rows)
            
            if len(data_rows) < batch_size:
                break
                
            skip += batch_size
    
    logging.info(f"Found {len(all_data_rows)} total data rows")
    
    if not all_data_rows:
        logging.error("No data rows found in the project")
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
    
    # 4. Extract filename pattern from row data URLs
    filename_pattern = re.compile(r'(\d{7}\.png)', re.IGNORECASE)
    
    # Build URL to data row ID mapping
    url_to_row_id = {}
    for row in all_data_rows:
        row_id = row.get("id")
        row_data = row.get("rowData")
        
        if not isinstance(row_data, str) or not row_data.startswith("http"):
            continue
            
        # Extract filename from URL
        match = filename_pattern.search(row_data)
        if not match:
            # Try another pattern - look for the last segment before query parameters
            url_parts = row_data.split('?')[0].split('/')
            if url_parts:
                last_part = url_parts[-1]
                # If it contains the .png extension, use it
                if '.png' in last_part.lower():
                    url_to_row_id[last_part] = row_id
            continue
            
        filename = match.group(1)
        url_to_row_id[filename] = row_id
    
    # Now try a more exact approach - extract the image ID from URL
    for row in all_data_rows:
        row_id = row.get("id")
        row_data = row.get("rowData")
        
        if not isinstance(row_data, str) or not row_data.startswith("http"):
            continue
        
        # Check if the URL contains any of the COCO image filenames
        for image in coco_data.get("images", []):
            if image.get("file_name") in row_data:
                url_to_row_id[image.get("file_name")] = row_id
    
    logging.info(f"Mapped {len(url_to_row_id)} image URLs to data rows")
    
    # If still empty, try another approach
    if not url_to_row_id:
        # For ian_pipeline dataset, we know external IDs are numbers
        # and images are named as 000000X.png
        id_to_filename = {
            "1": "0000000.png",
            "2": "0000001.png",
            "3": "0000002.png",
            "4": "0000003.png",
            "5": "0000004.png"
        }
        
        for row in all_data_rows:
            row_id = row.get("id")
            external_id = row.get("externalId")
            
            if external_id in id_to_filename:
                url_to_row_id[id_to_filename[external_id]] = row_id
        
        logging.info(f"Using external ID mapping: found {len(url_to_row_id)} matches")
    
    # 5. Build image ID to filename mapping from COCO
    image_id_to_filename = {}
    for image in coco_data.get("images", []):
        image_id_to_filename[image.get("id")] = image.get("file_name")
    
    # 6. Match annotations to data rows
    annotations_list = []
    matched_count = 0
    
    for filename, data_row_id in url_to_row_id.items():
        # Find annotations for this image
        image_annotations = []
        
        # Get image ID from filename
        image_id = None
        for img_id, img_filename in image_id_to_filename.items():
            if img_filename == filename:
                image_id = img_id
                break
        
        if image_id is None:
            continue
            
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
                        x = segmentation[0][i]
                        y = segmentation[0][i+1]
                        polygon_points.append([x, y])
                    
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
                    logging.error(f"Error processing annotation: {e}")
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
        # Check what method signatures are available
        logging.info(f"Available methods: {dir(lb.MALPredictionImport)}")
        
        # Try alternate method to upload
        # Upload the annotations file - use file_path instead of predictions_filepath
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
        
        # Try a direct, manual approach through the API
        try:
            logging.info("Attempting direct API upload...")
            
            # Upload the predictions file to Labelbox cloud storage
            with open(ndjson_file, 'rb') as f:
                file_content = f.read()
                
            # Create a mutation to upload file
            upload_query = """
            mutation UploadFile($file: Upload!, $contentLength: Int!, $sign: Boolean) {
                uploadFile(file: $file, contentLength: $contentLength, sign: $sign) {
                    url
                    filename
                }
            }
            """
            
            # Execute the upload mutation
            file_size = os.path.getsize(ndjson_file)
            upload_result = client.execute(
                upload_query,
                variables={
                    "file": file_content,
                    "contentLength": file_size,
                    "sign": False
                }
            )
            
            logging.info(f"File upload result: {upload_result}")
            
            # Now import the predictions using the API
            if upload_result.get("uploadFile", {}).get("url"):
                file_url = upload_result["uploadFile"]["url"]
                
                import_query = f"""
                mutation {{
                    createModelAssistedLabelingPredictionImport(
                        data: {{
                            projectId: "{LABELBOX_PROJECT_ID}",
                            name: "{import_name}",
                            inputUriPrefix: "{file_url}"
                        }}
                    ) {{
                        id
                    }}
                }}
                """
                
                import_result = client.execute(import_query)
                logging.info(f"Import result: {import_result}")
            
        except Exception as inner_e:
            logging.error(f"Error in direct API upload: {inner_e}")

if __name__ == "__main__":
    main() 