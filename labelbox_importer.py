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
    python labelbox_importer.py [--source {predictions,coco,test}] [--debug] [--limit N] [--format {mask,polygon}] [--dataset-name NAME]

Arguments:
    --source        Source format for predictions (default: coco)
    --debug         Enable debug mode with limited uploads
    --limit         Max number of images to upload in debug mode (default: 10)
    --format        Annotation format to use (default: polygon)
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
        self.annotation_format = args.format  # 'mask' or 'polygon'
        self.dataset_name = self._generate_dataset_name(args.dataset_name)
        self.chunk_size = args.chunk_size  # Add chunk size parameter
        self.start_chunk = args.start_chunk  # Add start chunk parameter
        self.use_mask_urls = args.use_mask_urls  # Option to use mask URLs
        self.simplified = args.simplified  # Simplified upload mode
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
            logging.debug(f"Using annotation format: {self.annotation_format}")
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
            # Even if connectivity check fails, try authentication anyway
            connectivity_ok = self._check_connectivity()
            if not connectivity_ok:
                logging.warning("Connectivity check failed, but continuing with authentication attempt...")
            
            # Test authentication by getting user information
            try:
                logging.info("Verifying API key by getting user information...")
                user = self.client.get_user()
                logging.info(f"Successfully authenticated as: {user.email}")
            except Exception as auth_error:
                logging.error(f"Authentication failed: {str(auth_error)}")
                logging.error("Please check your LABELBOX_API_KEY in the .env file")
                return False
            
            # Try to access the project
            logging.info("Connecting to Labelbox project...")
            try:
                self.project = self.client.get_project(LABELBOX_PROJECT_ID)
                logging.info(f"Connected to project: {self.project.name}")
            except Exception as project_error:
                logging.error(f"Failed to access project: {str(project_error)}")
                if "Resource not found" in str(project_error):
                    logging.error(f"Project with ID '{LABELBOX_PROJECT_ID}' not found")
                    self.list_available_projects()
                return False
            
            return True
        except Exception as e:
            logging.error(f"Failed to connect to Labelbox: {str(e)}")
            logging.debug(traceback.format_exc())
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
            
            # Check Labelbox API access - use main endpoint instead of /api/health
            logging.info("Checking Labelbox API connectivity...")
            try:
                # Just test that we can connect to api.labelbox.com - don't worry about the status code
                # We'll get our authentication checked in the next step, this is just for connectivity
                response = requests.get("https://api.labelbox.com", timeout=10)
                logging.info(f"Labelbox API is reachable (status: {response.status_code})")
                return True
            except requests.RequestException as e:
                logging.error(f"Cannot reach Labelbox API: {str(e)}")
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
            
            # Create Labelbox labels
            all_labels = []
            processed_count = 0
            
            # If debug mode, limit the number of images processed
            image_ids = list(annotations_by_image.keys())
            if self.debug_mode and self.upload_limit:
                logging.debug(f"Debug mode: limiting to {self.upload_limit} images")
                image_ids = image_ids[:self.upload_limit]
            
            logging.info(f"Processing {len(image_ids)} images with annotations")
            logging.info(f"Using annotation format: {self.annotation_format}")
            
            # If using mask URLs, add a warning about needing to upload mask images
            if self.use_mask_urls:
                logging.warning("Using mask URLs option - this requires you to have uploaded mask images to a cloud storage")
                logging.warning("and have valid URLs for each mask. The script will use a placeholder URL which you must replace.")
            
            for image_id in image_ids:
                if image_id not in image_lookup:
                    logging.warning(f"Image ID {image_id} not found in image lookup")
                    continue
                    
                image_data = image_lookup[image_id]
                file_name = image_data['file_name']
                base_name = Path(file_name).stem
                
                # Find the actual file to get the video directory name
                found_video_dir = None
                for video_dir in IMAGES_DIR.iterdir():
                    if not video_dir.is_dir():
                        continue
                    if (video_dir / file_name).exists():
                        found_video_dir = video_dir.name
                        break
                
                if not found_video_dir:
                    logging.warning(f"Could not find image file: {file_name} in any video directory")
                    continue
                
                # Create unique global key using the same format as upload_images_to_labelbox.py
                global_key = f"{found_video_dir}_{base_name}"
                logging.debug(f"Using global key: {global_key} for image: {file_name}")
                
                # Process all annotations for this image
                lb_annotations = []
                
                for anno in annotations_by_image[image_id]:
                    category_id = anno['category_id']
                    if category_id not in category_lookup:
                        logging.warning(f"Category ID {category_id} not found in category lookup")
                        continue
                        
                    category_name = category_lookup[category_id]
                    
                    # Get segmentation info
                    if not anno['segmentation']:
                        continue
                    
                    try:
                        if self.annotation_format == 'mask':
                            img_width = image_data.get('width', 0)
                            img_height = image_data.get('height', 0)
                            
                            if img_width <= 0 or img_height <= 0:
                                logging.warning(f"Invalid image dimensions for {global_key}: {img_width}x{img_height}")
                                continue
                            
                            if self.use_mask_urls:
                                # Create a mask annotation using MaskData URL approach
                                # Note: This is a placeholder URL that users need to replace with actual mask image URLs
                                mask_url = f"https://storage.googleapis.com/your-bucket/masks/{found_video_dir}/{base_name}_{category_id}.png"
                                logging.debug(f"Using placeholder mask URL: {mask_url}")
                                
                                # Create a MaskData object with the URL - following the example provided
                                mask_data = lb_types.MaskData(url=mask_url)
                                
                                # Create a mask annotation with the mask data
                                obj_annotation = lb_types.ObjectAnnotation(
                                    name=category_name,
                                    value=lb_types.Mask(mask=mask_data, color=(255, 255, 255))
                                )
                            else:
                                # Convert segmentation polygon to mask
                                mask = self.segmentation_to_mask(
                                    anno['segmentation'], 
                                    img_height, 
                                    img_width
                                )
                                
                                # Create a mask annotation
                                obj_annotation = lb_types.ObjectAnnotation(
                                    name=category_name,
                                    value=lb_types.Mask(mask=mask)
                                )
                        
                        else:  # polygon format
                            # Use the first polygon if there are multiple
                            polygon_points = anno['segmentation'][0]
                            
                            # Convert flat list to list of Point objects
                            points = []
                            for i in range(0, len(polygon_points), 2):
                                if i + 1 < len(polygon_points):
                                    # Create a Point object with x, y coordinates
                                    points.append(lb_types.Point(x=polygon_points[i], y=polygon_points[i+1]))
                            
                            # Create polygon
                            if len(points) >= 3:  # Need at least 3 points for a polygon
                                polygon = lb_types.Polygon(points=points)
                                
                                # Create object annotation
                                obj_annotation = lb_types.ObjectAnnotation(
                                    name=category_name,
                                    value=polygon
                                )
                            else:
                                logging.warning(f"Not enough points for polygon: {len(points)}")
                                continue
                        
                        lb_annotations.append(obj_annotation)
                    except Exception as e:
                        logging.error(f"Error creating annotation for {global_key}: {str(e)}")
                
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
    
    def segmentation_to_mask(self, segmentation, height, width):
        """Convert COCO segmentation format to a binary mask"""
        try:
            import numpy as np
            import cv2
            
            # Create an empty mask
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # For each polygon in the segmentation
            for polygon in segmentation:
                # Convert to numpy array and reshape
                poly = np.array(polygon, dtype=np.int32).reshape((-1, 2))
                
                # Fill the polygon with 1s
                cv2.fillPoly(mask, [poly], 1)
            
            # Return the mask as a 2D list for Labelbox
            return mask.tolist()
        except Exception as e:
            logging.error(f"Error converting segmentation to mask: {str(e)}")
            return [[]]
    
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
    
    def _get_detailed_job_status(self, upload_job) -> dict:
        """Get detailed status information for an upload job"""
        status_info = {
            "state": "UNKNOWN",
            "progress": None,
            "errors": [],
            "details": {}
        }
        
        try:
            # Basic status from the job object
            if hasattr(upload_job, 'state'):
                status_info["state"] = upload_job.state
            
            if hasattr(upload_job, 'progress'):
                status_info["progress"] = upload_job.progress
            
            if hasattr(upload_job, 'errors') and upload_job.errors:
                status_info["errors"] = upload_job.errors
            
            # Try to get more detailed information using GraphQL
            try:
                # Only attempt this if we have a client
                if self.client:
                    # Query the import details if needed
                    query = """
                    query getImportDetails($importId: ID!) {
                        importDetails(importId: $importId) {
                            id
                            name
                            status
                            progress
                            errors {
                                message
                                code
                            }
                            createdAt
                            completedAt
                        }
                    }
                    """
                    
                    if hasattr(upload_job, 'uid'):
                        variables = {'importId': upload_job.uid}
                        result = self.client.execute(query, variables)
                        
                        if result and 'importDetails' in result:
                            status_info["details"] = result['importDetails']
            except Exception as detail_error:
                # Failure to get details shouldn't block the main process
                logging.debug(f"Could not get detailed job status: {str(detail_error)}")
        
        except Exception as e:
            logging.debug(f"Error getting job status: {str(e)}")
        
        return status_info
    
    def upload_labels_to_labelbox(self, labels: List[lb_types.Label]) -> bool:
        """Upload prepared labels to Labelbox as MAL import"""
        try:
            # Create a unique name for the import
            import_name = f"Building_Damage_Import_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logging.info(f"Starting MAL import with name: {import_name}")
            
            # Additional debug logs
            logging.info(f"Uploading {len(labels)} annotations to Labelbox via MAL")
            logging.debug(f"Project ID: {LABELBOX_PROJECT_ID}")
            
            # Split labels into smaller chunks for more reliable uploading
            # This helps prevent timeouts on large uploads
            chunk_size = self.chunk_size  # Use the chunk size from arguments
            
            # Force chunking by ensuring chunk_size is respected
            if len(labels) > chunk_size:
                logging.info(f"Enforcing chunk size of {chunk_size} labels per chunk")
            else:
                logging.info(f"Using chunk size of {chunk_size}, but only have {len(labels)} labels total")
            
            total_chunks = (len(labels) + chunk_size - 1) // chunk_size
            
            if total_chunks > 1:
                logging.info(f"Splitting {len(labels)} annotations into {total_chunks} chunks of ~{chunk_size} each")
            
            successful_chunks = 0
            failed_chunks = 0
            
            # If start_chunk is specified, skip to that chunk
            starting_chunk = self.start_chunk if self.start_chunk > 0 else 0
            if starting_chunk > 0:
                logging.info(f"Starting from chunk {starting_chunk} as requested")
            
            # Use simplified mode if requested - this uploads all chunks without waiting for each to complete
            if self.simplified:
                logging.info("Using simplified upload mode - uploading all chunks without waiting for each to complete")
                return self._upload_simplified(labels, starting_chunk, total_chunks, chunk_size, import_name)
            
            # Standard upload mode - process each chunk and wait for it to complete before moving to the next
            for chunk_idx in range(starting_chunk, total_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, len(labels))
                
                # Ensure we're actually chunking
                chunk_labels = labels[start_idx:end_idx]
                
                # Skip empty chunks
                if not chunk_labels:
                    continue
                    
                chunk_name = f"{import_name}_chunk_{chunk_idx+1}of{total_chunks}"
                logging.info(f"Processing chunk {chunk_idx+1}/{total_chunks} with {len(chunk_labels)} annotations")
                
                # Upload this chunk
                try:
                    # Import as model-assisted labeling predictions
                    upload_job = lb.MALPredictionImport.create_from_objects(
                        client=self.client,
                        project_id=self.project.uid,
                        name=chunk_name,
                        predictions=chunk_labels
                    )
                    
                    logging.info(f"Chunk {chunk_idx+1} upload job created with name '{chunk_name}', checking status...")
                    
                    # Set a longer timeout limit (30 minutes per chunk)
                    max_wait_time = 30 * 60  # seconds
                    start_time = time.time()
                    last_progress = None
                    progress_stuck_time = 0
                    last_progress_change_time = time.time()
                    recurring_log_time = time.time()
                    warn_for_81_percent = False
                    
                    # Wait for the import to process
                    while True:
                        # Check if we've exceeded the timeout
                        elapsed_time = time.time() - start_time
                        if elapsed_time > max_wait_time:
                            logging.error(f"Chunk {chunk_idx+1} import timed out after {int(elapsed_time)} seconds")
                            logging.info("The import may still be processing on Labelbox servers.")
                            logging.info(f"Check manually at: https://app.labelbox.com/projects/{LABELBOX_PROJECT_ID}")
                            failed_chunks += 1
                            break
                        
                        # Refresh job state
                        try:
                            # Update the job status
                            upload_job.refresh()
                            
                            # Get detailed status including any available error information
                            status_info = self._get_detailed_job_status(upload_job)
                            
                            # Check for errors
                            if status_info["errors"]:
                                logging.error(f"Chunk {chunk_idx+1} import had errors: {status_info['errors']}")
                                failed_chunks += 1
                                break
                            
                            # Get current state and progress
                            job_state = status_info["state"]
                            current_progress = status_info["progress"]
                            
                            # Log status regardless of changes every 60 seconds
                            if time.time() - recurring_log_time >= 60:
                                recurring_log_time = time.time()
                                if current_progress is not None:
                                    progress_percent = int(current_progress * 100)
                                    logging.info(f"Periodic status check - Chunk {chunk_idx+1}: {progress_percent}% ({job_state}). Elapsed: {int(elapsed_time)}s")
                                else:
                                    logging.info(f"Periodic status check - Chunk {chunk_idx+1}: {job_state}. Elapsed: {int(elapsed_time)}s")
                            
                            # More detailed progress reporting
                            if current_progress is not None:
                                progress_percent = int(current_progress * 100)
                                
                                # Special handling for 81% progress stuck issue
                                if progress_percent == 81 and not warn_for_81_percent:
                                    warn_for_81_percent = True
                                    logging.warning("Progress at 81% - this is a common sticking point. If it stays here, consider:")
                                    logging.warning("1. Using an even smaller chunk size (--chunk-size 10)")
                                    logging.warning("2. Using mask URLs if possible (--use-mask-urls)")
                                    logging.warning("3. Checking the Labelbox UI to see if the import is actually working")
                                
                                logging.info(f"Chunk {chunk_idx+1} progress: {progress_percent}% ({job_state}). Elapsed time: {int(elapsed_time)}s")
                            else:
                                logging.info(f"Chunk {chunk_idx+1} state: {job_state}. Elapsed time: {int(elapsed_time)}s")
                            
                            # Check if progress has changed
                            if current_progress != last_progress:
                                last_progress = current_progress
                                last_progress_change_time = time.time()
                                progress_stuck_time = 0
                                warn_for_81_percent = False  # Reset the warning if progress changes
                            else:
                                # Calculate how long progress has been stuck
                                progress_stuck_time = time.time() - last_progress_change_time
                                
                                # Only report if stuck for more than 30 seconds
                                if progress_stuck_time > 30:
                                    if current_progress is not None:
                                        progress_percent = int(current_progress * 100)
                                        logging.warning(f"Progress stuck at {progress_percent}% for {int(progress_stuck_time)} seconds")
                                        
                                        # If stuck at 81% for a while, provide more specific guidance
                                        if progress_percent == 81 and progress_stuck_time > 120:
                                            logging.warning("This is a known issue at 81%. The import may still complete eventually.")
                                            logging.warning("You can try viewing the import in the Labelbox UI to check its actual status.")
                                    else:
                                        logging.warning(f"Progress unknown, no update for {int(progress_stuck_time)} seconds")
                                
                                # If progress is stuck for more than 5 minutes, consider it a failure
                                if progress_stuck_time >= 300:
                                    progress_percent = int(current_progress * 100) if current_progress else 0
                                    logging.error(f"Import appears to be stuck at {progress_percent}% for over 5 minutes")
                                    
                                    # Special handling for 81% stuck
                                    if progress_percent == 81:
                                        logging.error("The import is stuck at 81% which is a common issue.")
                                        logging.error("You can:")
                                        logging.error("1. Wait longer - sometimes imports complete after being stuck")
                                        logging.error("2. Check the Labelbox UI to see if annotations are appearing")
                                        logging.error("3. Try again with fewer annotations per chunk (--chunk-size 5)")
                                    
                                    logging.error("Aborting this chunk. You can check the import status manually in Labelbox.")
                                    failed_chunks += 1
                                    break
                        
                            # Check if complete
                            if job_state in ("COMPLETE", "FINISHED"):
                                logging.info(f"Chunk {chunk_idx+1} completed successfully!")
                                successful_chunks += 1
                                break
                            elif job_state in ("FAILED", "ERROR"):
                                logging.error(f"Chunk {chunk_idx+1} import failed with state: {job_state}")
                                failed_chunks += 1
                                break
                        except Exception as e:
                            logging.warning(f"Couldn't refresh job status: {str(e)}")
                        
                        # Wait between checks - include more info in the sleep message
                        progress_info = f"{int(current_progress * 100)}%" if current_progress is not None else "unknown"
                        logging.info(f"Progress: {progress_info}, waiting 15 seconds before next check... (elapsed: {int(elapsed_time)}s)")
                        time.sleep(15)  # Longer interval to reduce API calls
                    
                except Exception as e:
                    logging.error(f"Error during chunk {chunk_idx+1} import: {str(e)}")
                    logging.debug(traceback.format_exc())
                    failed_chunks += 1
                    
                # Add a small delay between chunks to avoid overwhelming the API
                if chunk_idx < total_chunks - 1:
                    logging.info("Waiting 10 seconds before starting next chunk...")
                    time.sleep(10)
            
            # Report overall status
            total_processed = successful_chunks + failed_chunks
            if successful_chunks == total_chunks:
                logging.info(f"All {total_chunks} chunks were successfully imported to Labelbox")
                logging.info(f"You can view the annotations at: https://app.labelbox.com/projects/{LABELBOX_PROJECT_ID}")
                return True
            elif successful_chunks > 0:
                logging.warning(f"Partial success: {successful_chunks}/{total_chunks} chunks were imported")
                logging.info(f"You can view the completed annotations at: https://app.labelbox.com/projects/{LABELBOX_PROJECT_ID}")
                return True  # Return True for partial success to avoid re-uploads
            else:
                logging.error(f"All {total_chunks} chunks failed to import")
                return False
                
        except Exception as e:
            logging.error(f"Error during Labelbox import: {str(e)}")
            logging.debug(traceback.format_exc())
            return False
    
    def _upload_simplified(self, labels, starting_chunk, total_chunks, chunk_size, import_name):
        """Simplified upload approach that doesn't wait for each chunk to complete
        before starting the next one. This can be more efficient but offers less
        detailed progress reporting."""
        try:
            # Track all jobs
            all_jobs = []
            
            # Start all uploads first
            for chunk_idx in range(starting_chunk, total_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, len(labels))
                chunk_labels = labels[start_idx:end_idx]
                
                # Skip empty chunks
                if not chunk_labels:
                    continue
                    
                chunk_name = f"{import_name}_chunk_{chunk_idx+1}of{total_chunks}"
                logging.info(f"Uploading chunk {chunk_idx+1}/{total_chunks} with {len(chunk_labels)} annotations")
                
                try:
                    # Import as model-assisted labeling predictions
                    upload_job = lb.MALPredictionImport.create_from_objects(
                        client=self.client,
                        project_id=self.project.uid,
                        name=chunk_name,
                        predictions=chunk_labels
                    )
                    
                    all_jobs.append({
                        "job": upload_job,
                        "chunk_idx": chunk_idx,
                        "name": chunk_name,
                        "status": "UPLOADING"
                    })
                    
                    logging.info(f"Chunk {chunk_idx+1} upload job started with name '{chunk_name}'")
                    
                    # Add a small delay between chunk uploads to avoid rate limiting
                    time.sleep(2)
                    
                except Exception as e:
                    logging.error(f"Error starting chunk {chunk_idx+1} upload: {str(e)}")
                    logging.debug(traceback.format_exc())
            
            # If no jobs started successfully, return failure
            if not all_jobs:
                logging.error("No upload jobs started successfully")
                return False
            
            logging.info(f"Started {len(all_jobs)} upload jobs, now monitoring their progress")
            
            # Monitor all jobs
            max_wait_time = 45 * 60  # 45 minutes total wait time
            start_time = time.time()
            
            completed_jobs = 0
            failed_jobs = 0
            
            # Loop until all jobs are complete or timeout
            while completed_jobs + failed_jobs < len(all_jobs):
                # Check if we've exceeded the timeout
                elapsed_time = time.time() - start_time
                if elapsed_time > max_wait_time:
                    logging.error(f"Import timed out after {int(elapsed_time)} seconds")
                    logging.info("Some jobs may still be processing on Labelbox servers.")
                    logging.info(f"Check manually at: https://app.labelbox.com/projects/{LABELBOX_PROJECT_ID}")
                    break
                
                # Check each job's status
                for job_info in all_jobs:
                    if job_info["status"] in ["COMPLETE", "FAILED"]:
                        continue
                        
                    try:
                        job_info["job"].refresh()
                        
                        if hasattr(job_info["job"], 'state'):
                            current_state = job_info["job"].state
                            
                            # Only log if status changed
                            if current_state != job_info["status"] and current_state != "UNKNOWN":
                                chunk_idx = job_info["chunk_idx"]
                                
                                # Check progress if available
                                progress_str = ""
                                if hasattr(job_info["job"], 'progress') and job_info["job"].progress is not None:
                                    try:
                                        # Convert progress to float first, then to percentage
                                        progress = float(job_info["job"].progress)
                                        progress = int(progress * 100)
                                        progress_str = f" ({progress}%)"
                                        
                                        # Special warning for 81% progress
                                        if progress == 81:
                                            logging.warning(f"Chunk {chunk_idx+1} is at 81% - this is a common sticking point")
                                    except (ValueError, TypeError) as e:
                                        logging.debug(f"Error parsing progress value: {str(e)}")
                                        progress_str = ""
                                
                                logging.info(f"Chunk {chunk_idx+1} status: {current_state}{progress_str}")
                                job_info["status"] = current_state
                                
                                # Update counters for completed/failed jobs
                                if current_state in ("COMPLETE", "FINISHED"):
                                    completed_jobs += 1
                                    logging.info(f"Chunk {chunk_idx+1} completed successfully!")
                                elif current_state in ("FAILED", "ERROR"):
                                    failed_jobs += 1
                                    logging.error(f"Chunk {chunk_idx+1} failed with state: {current_state}")
                                    
                                    # Check for errors
                                    if hasattr(job_info["job"], 'errors') and job_info["job"].errors:
                                        logging.error(f"Errors: {job_info['job'].errors}")
                    except Exception as e:
                        # Don't fail on status check errors
                        logging.debug(f"Error checking job status: {str(e)}")
                
                # Log overall status
                total_jobs = len(all_jobs)
                remaining = total_jobs - (completed_jobs + failed_jobs)
                logging.info(f"Status: {completed_jobs} complete, {failed_jobs} failed, {remaining} in progress. Elapsed: {int(elapsed_time)}s")
                
                # Sleep before checking again
                time.sleep(30)
            
            # Final status report
            logging.info(f"Final upload status: {completed_jobs}/{len(all_jobs)} chunks completed successfully")
            
            # Return success if at least some uploads succeeded
            if completed_jobs > 0:
                success_percent = (completed_jobs / len(all_jobs)) * 100
                logging.info(f"Import partially successful ({success_percent:.1f}% of chunks completed)")
                logging.info(f"You can view the annotations at: https://app.labelbox.com/projects/{LABELBOX_PROJECT_ID}")
                return True
            else:
                logging.error("All upload chunks failed")
                return False
                
        except Exception as e:
            logging.error(f"Error in simplified upload: {str(e)}")
            logging.debug(traceback.format_exc())
            return False
    
    def import_from_test(self) -> bool:
        """Import directly from the test_annotations.ndjson file"""
        try:
            test_file = PREDICTIONS_DIR / 'test_annotations.ndjson'
            if not test_file.exists():
                logging.error(f"Test annotations file not found at {test_file}")
                return False

            logging.info(f"Loading test annotations from {test_file}")
            
            # Load NDJSON file line by line (each line is a separate JSON object)
            annotations = []
            with open(test_file, 'r') as f:
                for line in f:
                    try:
                        annotation = json.loads(line)
                        annotations.append(annotation)
                    except json.JSONDecodeError as e:
                        logging.error(f"Error parsing JSON: {str(e)}")
                        continue
            
            logging.info(f"Loaded {len(annotations)} annotations from test file")
            
            # Upload annotations directly (they're already in Labelbox format)
            if not self.project:
                if not self.connect_to_labelbox():
                    return False
                
            # Import annotations directly
            import_name = f"test_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logging.info(f"Creating MAL prediction import with name: {import_name}")
            
            try:
                upload_job = lb.MALPredictionImport.create_from_objects(
                    client=self.client,
                    project_id=self.project.uid,
                    name=import_name,
                    predictions=annotations
                )
                
                # Check for errors
                if upload_job.errors:
                    for error in upload_job.errors:
                        logging.error(f"Upload error: {error}")
                    return False
                
                logging.info("MAL prediction import created successfully")
                logging.info(f"Status: {upload_job.statuses}")
                return True
                
            except Exception as e:
                logging.error(f"Error creating MAL prediction import: {str(e)}")
                logging.debug(traceback.format_exc())
                return False
            
        except Exception as e:
            logging.error(f"Error importing from test file: {str(e)}")
            logging.debug(traceback.format_exc())
            return False
    
    def run(self) -> bool:
        """Run the import process end-to-end"""
        try:
            # Connect to Labelbox
            if not self.connect_to_labelbox():
                return False
            
            # Import data based on source format
            if self.source == 'coco':
                return self.import_from_coco()
            elif self.source == 'predictions':
                return self.import_from_predictions()
            elif self.source == 'test':
                return self.import_from_test()
            else:
                logging.error(f"Unsupported source format: {self.source}")
                return False
        except Exception as e:
            logging.error(f"Error running import: {str(e)}")
            logging.debug(traceback.format_exc())
            return False


def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description='Import annotations into Labelbox')
    parser.add_argument('--source', choices=['predictions', 'coco', 'test'], default='coco',
                        help='Source format for predictions (default: coco)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with limited uploads')
    parser.add_argument('--limit', type=int, default=10, help='Max number of images to upload in debug mode')
    parser.add_argument('--format', type=str, choices=['mask', 'polygon'], default='polygon',
                        help='Annotation format to use (default: polygon)')
    parser.add_argument('--dataset-name', type=str, help='Custom name for the Labelbox dataset')
    parser.add_argument('--chunk-size', type=int, default=25, 
                        help='Number of annotations to upload in each chunk (default: 25)')
    parser.add_argument('--start-chunk', type=int, default=0,
                        help='Start uploading from this chunk index (0-based, for resuming failed uploads)')
    parser.add_argument('--use-mask-urls', action='store_true',
                        help='Use Labelbox MaskData URLs instead of binary masks')
    parser.add_argument('--simplified', action='store_true',
                        help='Use simplified upload mode (uploads all chunks at once, less detailed progress)')
    
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