#!/usr/bin/env python3
"""
Simple Labelbox Importer for Labelbox SDK 6.x

This script is compatible with Labelbox SDK 6.x which doesn't have the data module.
It imports annotations in NDJSON format directly.

Usage:
    python simple_labelbox_import_6x.py [--source {predictions,coco}] [--debug] [--limit N]
"""

import os
import json
import logging
import uuid
import sys
import traceback
import argparse
import numpy as np
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple

# Setup path to import from config
sys.path.append(str(Path(__file__).parent))
from config import *

# Import Labelbox SDK
try:
    import labelbox as lb
except ImportError:
    print("Labelbox SDK not installed. Please install it with:")
    print("pip install --user labelbox")
    sys.exit(1)


class SimpleLabelboxImporter:
    """Handles Labelbox importing using NDJSON format for SDK 6.x"""
    
    def __init__(self, args):
        """Initialize the importer with configuration settings"""
        self.args = args
        self.debug_mode = args.debug
        self.upload_limit = args.limit if args.debug else None
        self.source = args.source
        self.client = None
        self.project = None
        
        # Setup logging
        self.setup_logging()
        
        # Validate environment and inputs
        if not self.validate_environment():
            logging.error("Environment validation failed. Please fix the issues above.")
            sys.exit(1)
    
    def setup_logging(self):
        """Configure logging with file and console output"""
        log_level = logging.DEBUG if self.debug_mode else logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(LOGS_DIR / 'labelbox_import.log'),
                logging.StreamHandler()
            ]
        )
        
        # Additional debug information
        if self.debug_mode:
            logging.debug(f"Debug mode enabled. Upload limit: {self.upload_limit}")
            logging.debug(f"Using source format: {self.source}")
    
    def validate_environment(self) -> bool:
        """Validate environment variables and required files"""
        validation_passed = True
        
        # Check API key and project ID
        if not LABELBOX_API_KEY:
            logging.error("Labelbox API key not set. Please set the LABELBOX_API_KEY environment variable.")
            logging.error("You can use the setup_env.sh script or create a .env file based on .env.example")
            validation_passed = False
        
        if not LABELBOX_PROJECT_ID:
            logging.error("Labelbox project ID not set. Please set the LABELBOX_PROJECT_ID environment variable.")
            logging.error("You can use the setup_env.sh script or create a .env file based on .env.example")
            validation_passed = False
        
        # Check required files based on source format
        if self.source == 'coco':
            coco_file = PREDICTIONS_DIR / 'predictions_coco.json'
            if not coco_file.exists():
                logging.error(f"COCO file not found at {coco_file}")
                logging.error("Please run 'python dataset_pipeline/export_to_coco.py' first")
                validation_passed = False
        elif self.source == 'predictions':
            predictions_file = PREDICTIONS_DIR / 'predictions.json'
            if not predictions_file.exists():
                logging.error(f"Predictions file not found at {predictions_file}")
                logging.error("Please run 'python dataset_pipeline/inference.py' first")
                validation_passed = False
        
        return validation_passed
    
    def connect_to_labelbox(self) -> bool:
        """Initialize connection to Labelbox and verify project access"""
        try:
            logging.info("Initializing Labelbox client")
            self.client = lb.Client(api_key=LABELBOX_API_KEY)
            
            logging.info("Connecting to Labelbox project")
            self.project = self.client.get_project(LABELBOX_PROJECT_ID)
            logging.info(f"Connected to project: {self.project.name}")
            
            return True
        except Exception as e:
            logging.error(f"Failed to connect to Labelbox: {str(e)}")
            if "Resource not found" in str(e):
                logging.error(f"Project with ID '{LABELBOX_PROJECT_ID}' not found")
                self.list_available_projects()
            return False
    
    def list_available_projects(self):
        """List all available projects in the user's Labelbox account"""
        if not self.client:
            logging.error("Client not initialized")
            return
        
        try:
            projects = self.client.get_projects()
            logging.info("Available projects in your Labelbox account:")
            for project in projects:
                logging.info(f"  • {project.name} (ID: {project.uid})")
        except Exception as e:
            logging.error(f"Error listing projects: {str(e)}")
    
    def get_data_row_ids(self) -> Dict[str, str]:
        """Get mapping of filenames to data row IDs from Labelbox"""
        logging.info("Fetching data rows from Labelbox project...")
        
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
        
        result = self.client.execute(project_with_datasets_query)
        datasets = result.get("project", {}).get("datasets", [])
        
        if not datasets:
            logging.error("No datasets found in the project")
            return {}
        
        # Initialize empty dictionary to store mapping
        filename_to_row_id = {}
        
        # Loop through all datasets in the project
        for dataset in datasets:
            dataset_id = dataset.get("id")
            dataset_name = dataset.get("name")
            logging.info(f"Processing dataset: {dataset_name}")
            
            # Get data rows from this dataset with pagination
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
                    
                    data_row_result = self.client.execute(data_row_query)
                    data_rows = data_row_result.get("dataset", {}).get("dataRows", [])
                    
                    if not data_rows:
                        break
                    
                    count = len(data_rows)
                    total_rows += count
                    
                    for row in data_rows:
                        row_id = row.get("id")
                        external_id = row.get("externalId")
                        row_data = row.get("rowData")
                        
                        # Extract filename from row data URL or external ID
                        if external_id:
                            # Map both with and without extension
                            filename_to_row_id[external_id] = row_id
                            if not external_id.lower().endswith('.png'):
                                filename_to_row_id[f"{external_id}.png"] = row_id
                        
                        # Extract filename from row data (URL path)
                        if isinstance(row_data, str) and row_data.startswith("http"):
                            # Extract filename from URL
                            filename = row_data.split('/')[-1].split('?')[0]
                            if filename:
                                filename_to_row_id[filename] = row_id
                                # Also store without extension
                                name_without_ext = Path(filename).stem
                                if name_without_ext:
                                    filename_to_row_id[name_without_ext] = row_id
                    
                    # If we got fewer rows than page_size, we've reached the end
                    if count < page_size:
                        break
                    
                    # Update skip for next page
                    skip += count
                    logging.info(f"Processed {total_rows} data rows from dataset {dataset_name}")
                    
                except Exception as e:
                    logging.error(f"Error fetching data rows: {str(e)}")
                    logging.debug(traceback.format_exc())
                    break
        
        logging.info(f"Found {len(filename_to_row_id)} data row mappings")
        return filename_to_row_id
    
    def import_from_coco(self) -> bool:
        """Import annotations from COCO format to Labelbox"""
        coco_file = PREDICTIONS_DIR / 'predictions_coco.json'
        
        try:
            # Load COCO data
            with open(coco_file, 'r') as f:
                coco_data = json.load(f)
            
            # Get data row IDs from Labelbox
            filename_to_row_id = self.get_data_row_ids()
            
            if not filename_to_row_id:
                logging.error("Failed to retrieve data row IDs from Labelbox")
                return False
            
            # Create a lookup from image ID to image external_id/filename
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
            
            # Create Labelbox annotations in NDJSON format
            all_annotations = []
            processed_count = 0
            skipped_count = 0
            
            # If debug mode, limit the number of images processed
            image_ids = list(annotations_by_image.keys())
            if self.debug_mode and self.upload_limit:
                logging.debug(f"Debug mode: limiting to {self.upload_limit} images")
                image_ids = image_ids[:self.upload_limit]
            
            logging.info(f"Processing {len(image_ids)} images with annotations")
            
            for image_id in image_ids:
                if image_id not in image_lookup:
                    logging.warning(f"Image ID {image_id} not found in image lookup")
                    skipped_count += 1
                    continue
                    
                image_data = image_lookup[image_id]
                image_filename = image_data['file_name']
                filename_base = Path(image_filename).stem
                
                # Find data row ID from filename
                data_row_id = None
                
                # Try different variations of the filename
                for key in [image_filename, filename_base, f"{filename_base}.png"]:
                    if key in filename_to_row_id:
                        data_row_id = filename_to_row_id[key]
                        break
                
                if not data_row_id:
                    logging.warning(f"No data row ID found for image {image_filename}")
                    skipped_count += 1
                    continue
                
                # Process all annotations for this image
                for anno in annotations_by_image[image_id]:
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
                    
                    try:
                        # Create points for polygon in the format Labelbox expects
                        points = []
                        for i in range(0, len(polygon_points), 2):
                            if i+1 < len(polygon_points):
                                x = polygon_points[i]
                                y = polygon_points[i+1]
                                points.append([x, y])
                        
                        # Skip polygons with too few points
                        if len(points) < 3:
                            logging.warning(f"Skipping polygon with fewer than 3 points for {image_filename}")
                            continue
                        
                        # Create the annotation in NDJSON format
                        annotation = {
                            "uuid": str(anno.get("id", uuid.uuid4())),
                            "name": category_name,
                            "value": {
                                "format": "polygon2d",
                                "points": points
                            },
                            "dataRow": {
                                "id": data_row_id
                            }
                        }
                        
                        all_annotations.append(annotation)
                        
                    except Exception as e:
                        logging.error(f"Error creating polygon for {image_filename}: {str(e)}")
                
                processed_count += 1
                
                # Log progress periodically
                if processed_count % 10 == 0 and processed_count > 0:
                    logging.info(f"Processed {processed_count}/{len(image_ids)} images")
            
            logging.info(f"Created {len(all_annotations)} annotations from COCO annotations")
            logging.info(f"Skipped {skipped_count} images due to missing data row IDs")
            
            # Import annotations to Labelbox
            if all_annotations:
                return self.upload_annotations_to_labelbox(all_annotations)
            else:
                logging.warning("No valid annotations to import to Labelbox")
                return False
                
        except Exception as e:
            logging.error(f"Error during COCO import: {str(e)}")
            logging.debug(traceback.format_exc())
            return False
    
    def import_from_predictions(self) -> bool:
        """Import annotations directly from predictions.json"""
        predictions_file = PREDICTIONS_DIR / 'predictions.json'
        
        try:
            # Load predictions
            with open(predictions_file, 'r') as f:
                predictions_by_video = json.load(f)
            
            # Get data row IDs from Labelbox
            filename_to_row_id = self.get_data_row_ids()
            
            if not filename_to_row_id:
                logging.error("Failed to retrieve data row IDs from Labelbox")
                return False
            
            all_annotations = []
            processed_count = 0
            skipped_count = 0
            
            # If debug mode, limit the number of frames processed
            if self.debug_mode and self.upload_limit:
                logging.debug(f"Debug mode: limiting to {self.upload_limit} frames")
            
            for video_name, frames in predictions_by_video.items():
                logging.info(f"Processing video: {video_name} ({len(frames)} frames)")
                
                # In debug mode, limit frames per video
                video_frames = frames
                if self.debug_mode and self.upload_limit:
                    video_frames = frames[:min(self.upload_limit, len(frames))]
                
                for frame in video_frames:
                    # Skip frames with errors
                    if 'error' in frame:
                        logging.warning(f"Skipping frame with error: {frame.get('image_path', 'unknown')}")
                        continue
                    
                    image_path = frame['image_path']
                    filename = Path(image_path).name
                    filename_base = Path(image_path).stem
                    
                    # Find data row ID from filename
                    data_row_id = None
                    
                    # Try different variations of the filename
                    for key in [filename, filename_base, f"{filename_base}.png"]:
                        if key in filename_to_row_id:
                            data_row_id = filename_to_row_id[key]
                            break
                    
                    if not data_row_id:
                        logging.warning(f"No data row ID found for image {filename}")
                        skipped_count += 1
                        continue
                    
                    # Process predictions for this frame
                    for pred in frame.get('predictions', []):
                        class_id = pred.get('class_id')
                        if not 0 <= class_id < len(CLASS_NAMES):
                            logging.warning(f"Invalid class ID: {class_id}")
                            continue
                        
                        class_name = CLASS_NAMES[class_id] if class_id > 0 else CLASS_NAMES[1]  # Skip background
                        mask = np.array(pred['mask']).astype(np.uint8)
                        
                        # Skip empty masks
                        if mask.sum() == 0:
                            continue
                        
                        # Convert mask to polygons
                        try:
                            import cv2
                            contours, _ = cv2.findContours(
                                mask, 
                                cv2.RETR_EXTERNAL, 
                                cv2.CHAIN_APPROX_SIMPLE
                            )
                            
                            for contour in contours:
                                # Skip small contours
                                if len(contour) < 3:
                                    continue
                                
                                # Convert contour to list of points
                                points = []
                                for point in contour:
                                    # Extract the point as a normal list [x, y]
                                    points.append([float(point[0][0]), float(point[0][1])])
                                
                                # Create polygon in NDJSON format
                                if len(points) >= 3:
                                    annotation = {
                                        "uuid": str(uuid.uuid4()),
                                        "name": class_name,
                                        "value": {
                                            "format": "polygon2d",
                                            "points": points
                                        },
                                        "dataRow": {
                                            "id": data_row_id
                                        }
                                    }
                                    all_annotations.append(annotation)
                        except Exception as e:
                            logging.error(f"Error creating polygon for {filename}: {str(e)}")
                    
                    processed_count += 1
                    
                    # Log progress periodically
                    if processed_count % 10 == 0:
                        logging.info(f"Processed {processed_count} frames")
                    
                    # In debug mode, check if we've hit the overall limit
                    if self.debug_mode and len(all_annotations) >= self.upload_limit:
                        logging.debug(f"Debug mode: reached limit of {self.upload_limit} frames with annotations")
                        break
                
                # In debug mode, check if we've hit the overall limit
                if self.debug_mode and len(all_annotations) >= self.upload_limit:
                    break
            
            logging.info(f"Created {len(all_annotations)} annotations from predictions")
            logging.info(f"Skipped {skipped_count} frames due to missing data row IDs")
            
            # Import annotations to Labelbox
            if all_annotations:
                return self.upload_annotations_to_labelbox(all_annotations)
            else:
                logging.warning("No valid annotations to import to Labelbox")
                return False
                
        except Exception as e:
            logging.error(f"Error during predictions import: {str(e)}")
            logging.debug(traceback.format_exc())
            return False
    
    def upload_annotations_to_labelbox(self, annotations: List) -> bool:
        """Upload prepared annotations to Labelbox using MALPredictionImport"""
        try:
            # Create a unique name for the import
            import_name = f"Building_Damage_Import_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logging.info(f"Starting annotation import with name: {import_name}")
            
            logging.info(f"Uploading {len(annotations)} annotations to Labelbox")
            
            # First save the annotations to a temporary NDJSON file
            temp_file = PREDICTIONS_DIR / f"temp_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ndjson"
            
            with open(temp_file, 'w') as f:
                for annotation in annotations:
                    f.write(json.dumps(annotation) + '\n')
            
            logging.info(f"Saved annotations to temporary file: {temp_file}")
            
            # Import using MALPredictionImport
            try:
                upload_job = lb.MALPredictionImport.create_from_file(
                    client=self.client,
                    project_id=self.project.uid,
                    name=import_name,
                    file_path=str(temp_file)
                )
                
                logging.info(f"Upload started with job ID: {upload_job.uid}")
                logging.info("Monitoring upload progress...")
                
                # Monitor progress with timeout
                max_wait_time = 15 * 60  # 15 minutes
                start_time = time.time()
                
                while time.time() - start_time < max_wait_time:
                    try:
                        upload_job.refresh()
                        
                        state = None
                        if hasattr(upload_job, 'state'):
                            state = upload_job.state
                        
                        progress = None
                        if hasattr(upload_job, 'progress'):
                            progress = upload_job.progress
                            
                        logging.info(f"Upload status: {state}, Progress: {progress}")
                        
                        if state in ["COMPLETE", "FINISHED"]:
                            logging.info("Upload completed successfully!")
                            break
                        elif state in ["FAILED", "ERROR"]:
                            logging.error("Upload failed!")
                            break
                        
                        # Wait between status checks
                        time.sleep(15)
                    except Exception as e:
                        logging.warning(f"Couldn't refresh job status: {str(e)}")
                        time.sleep(15)
                
                # Check for errors
                if hasattr(upload_job, 'errors') and upload_job.errors:
                    logging.error(f"Import had errors: {upload_job.errors}")
                    return False
                
                logging.info(f"Successfully imported annotations to Labelbox")
                logging.info(f"Total import time: {int(time.time() - start_time)} seconds")
                logging.info(f"You can view the annotations at: https://app.labelbox.com/projects/{LABELBOX_PROJECT_ID}")
                return True
                    
            except Exception as e:
                logging.error(f"Error during MALPredictionImport: {str(e)}")
                logging.debug(traceback.format_exc())
                
                # Try alternate method if the first one fails
                try:
                    logging.info("Trying alternate import method...")
                    
                    # Execute a direct GraphQL mutation to upload the annotations
                    project_query = f"""
                    mutation CreateImport($projectId: ID!, $input: CreateModelRunImportInput!) {{
                        createModelRunImport(projectId: $projectId, input: $input) {{
                            modelRunImport {{
                                id
                                name
                                status
                                createdAt
                            }}
                        }}
                    }}
                    """
                    
                    variables = {
                        "projectId": self.project.uid,
                        "input": {
                            "name": import_name,
                            "sourceType": "NDJSON_URL",
                            "source": str(temp_file),
                            "includeInPlatformMetrics": True
                        }
                    }
                    
                    result = self.client.execute(project_query, variables=variables)
                    logging.info(f"Import result: {result}")
                    
                    return True
                except Exception as inner_e:
                    logging.error(f"Error during alternate import: {str(inner_e)}")
                    logging.debug(traceback.format_exc())
                    return False
                
        except Exception as e:
            logging.error(f"Error during Labelbox import: {str(e)}")
            logging.debug(traceback.format_exc())
            return False
    
    def run(self) -> bool:
        """Main execution method"""
        # Connect to Labelbox
        if not self.connect_to_labelbox():
            return False
        
        # Run appropriate import method based on source
        if self.source == 'coco':
            return self.import_from_coco()
        elif self.source == 'predictions':
            return self.import_from_predictions()
        else:
            logging.error(f"Unknown source format: {self.source}")
            return False


def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Simple import of annotations to Labelbox (compatible with SDK 6.x)')
    parser.add_argument('--source', type=str, choices=['predictions', 'coco'], default='coco',
                        help='Source format for predictions (default: coco)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with limited uploads')
    parser.add_argument('--limit', type=int, default=10, help='Max number of images to upload in debug mode')
    
    args = parser.parse_args()
    
    # Start the import process
    importer = SimpleLabelboxImporter(args)
    success = importer.run()
    
    if success:
        logging.info("Import completed successfully")
        return 0
    else:
        logging.error("Import failed. Check the logs for details.")
        return 1


if __name__ == "__main__":
    # Execute main function
    sys.exit(main()) 