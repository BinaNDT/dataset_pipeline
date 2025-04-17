#!/usr/bin/env python3
"""
Labelbox Importer

This script handles all Labelbox-related uploading functionality, including:
1. Converting model predictions to Labelbox format
2. Creating datasets in Labelbox
3. Uploading annotations as model-assisted labeling (MAL) import
4. Handling error recovery and validation

It consolidates functionality from previous scripts:
- fixed_labelbox_import.py
- import_using_mal.py
- labelbox_uploader.py

Usage:
    python labelbox_importer.py [--source {predictions,coco}] [--debug] [--limit N] [--dataset-name NAME]

Arguments:
    --source        Source format for predictions (default: coco)
    --debug         Enable debug mode with limited uploads
    --limit         Max number of images to upload in debug mode (default: 10)
    --dataset-name  Custom name for the Labelbox dataset (default: auto-generated with timestamp)
"""

import os
import json
import logging
import uuid
import sys
import traceback
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple

# Setup path to import from config
sys.path.append(str(Path(__file__).parent))
from config import *

# Import Labelbox SDK
try:
    import labelbox as lb
    import labelbox.data.annotation_types as lb_types
except ImportError:
    print("Error: Labelbox SDK not installed. Please run: pip install labelbox>=3.0.0")
    sys.exit(1)


class LabelboxImporter:
    """Handles all operations related to Labelbox importing"""
    
    def __init__(self, args):
        """Initialize the importer with configuration settings"""
        self.args = args
        self.debug_mode = args.debug
        self.upload_limit = args.limit if args.debug else None
        self.source = args.source
        self.dataset_name = self._generate_dataset_name(args.dataset_name)
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
            logging.debug(f"Dataset name: {self.dataset_name}")
    
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
    
    def _generate_dataset_name(self, custom_name: Optional[str] = None) -> str:
        """Generate a unique dataset name using timestamp"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if custom_name:
            dataset_name = f"{custom_name}_{timestamp}"
        else:
            prefix = "Building_Damage_DEBUG" if self.debug_mode else "Building_Damage"
            dataset_name = f"{prefix}_{timestamp}"
        
        return dataset_name
    
    def connect_to_labelbox(self) -> bool:
        """Initialize connection to Labelbox and verify project access"""
        try:
            logging.info("Initializing Labelbox client")
            self.client = lb.Client(api_key=LABELBOX_API_KEY)
            
            # Check basic connectivity
            if not self._check_connectivity():
                return False
            
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
    
    def _check_connectivity(self) -> bool:
        """Check basic connectivity to Labelbox API"""
        try:
            import requests
            import socket
            
            # Check internet connectivity
            logging.info("Checking internet connectivity...")
            try:
                socket.create_connection(("8.8.8.8", 53), timeout=5)
                logging.info("Internet connection is available")
            except OSError:
                logging.error("No internet connection available")
                return False
            
            # Check Labelbox API access
            logging.info("Checking Labelbox API connectivity...")
            response = requests.get(
                "https://api.labelbox.com/api/health", 
                timeout=10
            )
            if response.status_code == 200:
                logging.info("Labelbox API is accessible")
                return True
            else:
                logging.error(f"Labelbox API returned unexpected status: {response.status_code}")
                return False
        except Exception as e:
            logging.error(f"Connectivity check failed: {str(e)}")
            logging.debug(traceback.format_exc())
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
    
    def binary_mask_to_polygon(self, binary_mask: np.ndarray) -> List[List[float]]:
        """Convert a binary mask to polygon format"""
        try:
            import cv2
            contours, _ = cv2.findContours(
                binary_mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            polygons = []
            for contour in contours:
                # Convert contour to polygon
                segmentation = contour.flatten().tolist()
                
                # Skip invalid polygons
                if len(segmentation) < 6:  # Need at least 3 points (x,y)
                    continue
                    
                polygons.append(segmentation)
            
            return polygons
        except Exception as e:
            logging.error(f"Error converting mask to polygon: {str(e)}")
            return []
    
    def import_from_coco(self) -> bool:
        """Import annotations from COCO format to Labelbox"""
        coco_file = PREDICTIONS_DIR / 'predictions_coco.json'
        
        try:
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
            processed_count = 0
            
            # If debug mode, limit the number of images processed
            image_ids = list(annotations_by_image.keys())
            if self.debug_mode and self.upload_limit:
                logging.debug(f"Debug mode: limiting to {self.upload_limit} images")
                image_ids = image_ids[:self.upload_limit]
            
            logging.info(f"Processing {len(image_ids)} images with annotations")
            
            for image_id in image_ids:
                if image_id not in image_lookup:
                    logging.warning(f"Image ID {image_id} not found in image lookup")
                    continue
                    
                image_data = image_lookup[image_id]
                global_key = Path(image_data['file_name']).stem
                
                # Process all annotations for this image
                lb_annotations = []
                
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
                    processed_count += 1
                
                # Log progress periodically
                if processed_count % 10 == 0 and processed_count > 0:
                    logging.info(f"Processed {processed_count}/{len(image_ids)} images")
            
            logging.info(f"Created {len(all_labels)} labels from COCO annotations")
            
            # Import labels to Labelbox
            if all_labels:
                return self.upload_labels_to_labelbox(all_labels)
            else:
                logging.warning("No valid labels to import to Labelbox")
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
            
            all_labels = []
            processed_count = 0
            total_frames = sum(len(frames) for frames in predictions_by_video.values())
            
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
                    global_key = Path(image_path).stem
                    
                    # Process predictions for this frame
                    lb_annotations = []
                    
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
                        polygons = self.binary_mask_to_polygon(mask)
                        
                        if not polygons:
                            continue
                        
                        # Create an annotation for each polygon
                        for polygon_points in polygons:
                            points = []
                            for i in range(0, len(polygon_points), 2):
                                if i + 1 < len(polygon_points):
                                    points.append(lb_types.Point(x=polygon_points[i], y=polygon_points[i+1]))
                            
                            try:
                                if len(points) >= 3:
                                    polygon = lb_types.Polygon(points=points)
                                    
                                    obj_annotation = lb_types.ObjectAnnotation(
                                        name=class_name,
                                        value=polygon,
                                        confidence=pred.get('confidence', 1.0)
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
                    
                    processed_count += 1
                    
                    # Log progress periodically
                    if processed_count % 10 == 0:
                        logging.info(f"Processed {processed_count} frames")
                    
                    # In debug mode, check if we've hit the overall limit
                    if self.debug_mode and len(all_labels) >= self.upload_limit:
                        logging.debug(f"Debug mode: reached limit of {self.upload_limit} frames with annotations")
                        break
                
                # In debug mode, check if we've hit the overall limit
                if self.debug_mode and len(all_labels) >= self.upload_limit:
                    break
            
            logging.info(f"Created {len(all_labels)} labels from predictions")
            
            # Import labels to Labelbox
            if all_labels:
                return self.upload_labels_to_labelbox(all_labels)
            else:
                logging.warning("No valid labels to import to Labelbox")
                return False
                
        except Exception as e:
            logging.error(f"Error during predictions import: {str(e)}")
            logging.debug(traceback.format_exc())
            return False
    
    def upload_labels_to_labelbox(self, labels: List[lb_types.Label]) -> bool:
        """Upload prepared labels to Labelbox as MAL import"""
        try:
            # Create a unique name for the import
            import_name = f"Building_Damage_Import_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logging.info(f"Starting MAL import with name: {import_name}")
            
            # Log information about the upload
            logging.info(f"Uploading {len(labels)} annotations to Labelbox")
            logging.debug(f"Project ID: {LABELBOX_PROJECT_ID}")
            
            # Import as model-assisted labeling predictions
            try:
                upload_job = lb.MALPredictionImport.create_from_objects(
                    client=self.client,
                    project_id=self.project.uid,
                    name=import_name,
                    predictions=labels
                )
                logging.info(f"Upload job created with ID: {upload_job.uid if hasattr(upload_job, 'uid') else 'unknown'}")
            except Exception as e:
                logging.error(f"Failed to create upload job: {str(e)}")
                logging.debug(traceback.format_exc())
                if "connection" in str(e).lower() or "timeout" in str(e).lower():
                    logging.error("Network connection issue detected. Check your internet connection.")
                if "unauthorized" in str(e).lower() or "403" in str(e).lower():
                    logging.error("Authorization issue. Check that your API key is valid and has proper permissions.")
                return False
            
            logging.info("Waiting for import to complete...")
            
            # Set a timeout limit (30 minutes)
            max_wait_time = 30 * 60  # seconds
            start_time = time.time()
            wait_count = 0
            
            # Monitor progress with timeout
            while not upload_job.done:
                elapsed_time = time.time() - start_time
                wait_count += 1
                
                # Check if we've exceeded the timeout
                if elapsed_time > max_wait_time:
                    logging.error(f"Import timed out after {int(elapsed_time)} seconds")
                    logging.info("The import may still be processing on Labelbox servers.")
                    logging.info(f"Check manually at: https://app.labelbox.com/projects/{LABELBOX_PROJECT_ID}")
                    return False
                
                # Get more detailed status every few iterations
                if wait_count % 3 == 0:
                    try:
                        upload_job.refresh()
                        status_details = f"{upload_job.status}"
                        if hasattr(upload_job, 'progress') and upload_job.progress:
                            status_details += f" - Progress: {upload_job.progress:.1f}%"
                        logging.info(f"Import status: {status_details}. Elapsed time: {int(elapsed_time)}s. Waiting...")
                    except Exception as status_e:
                        logging.warning(f"Couldn't refresh job status: {str(status_e)}")
                else:
                    logging.info(f"Waiting for import... Elapsed time: {int(elapsed_time)}s")
                
                # Wait between status checks
                time.sleep(10)
            
            # Check for errors
            if hasattr(upload_job, 'errors') and upload_job.errors:
                logging.error(f"Import had errors: {upload_job.errors}")
                return False
            else:
                logging.info(f"Successfully imported {len(labels)} annotations to Labelbox")
                logging.info(f"Total import time: {int(time.time() - start_time)} seconds")
                logging.info(f"You can view the annotations at: https://app.labelbox.com/projects/{LABELBOX_PROJECT_ID}")
                return True
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
    parser = argparse.ArgumentParser(description='Import annotations to Labelbox')
    parser.add_argument('--source', type=str, choices=['predictions', 'coco'], default='coco',
                        help='Source format for predictions (default: coco)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with limited uploads')
    parser.add_argument('--limit', type=int, default=10, help='Max number of images to upload in debug mode')
    parser.add_argument('--dataset-name', type=str, help='Custom name for the Labelbox dataset')
    
    args = parser.parse_args()
    
    # Start the import process
    importer = LabelboxImporter(args)
    success = importer.run()
    
    if success:
        logging.info("Import completed successfully")
        return 0
    else:
        logging.error("Import failed. Check the logs for details.")
        return 1


if __name__ == "__main__":
    # Add module-level import for time
    import time
    
    # Execute main function
    sys.exit(main()) 