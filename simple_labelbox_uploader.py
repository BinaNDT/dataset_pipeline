#!/usr/bin/env python3
"""
Simple Labelbox Uploader

This script demonstrates the recommended approach for uploading 
mask annotations to Labelbox using mask URLs instead of binary mask data.

Usage:
    python simple_labelbox_uploader.py --api-key YOUR_API_KEY [--mask-url MASK_URL] [--image-key IMAGE_KEY]
"""

import os
import argparse
import logging
import uuid
import sys
import time
from pathlib import Path

# Import Labelbox SDK
try:
    import labelbox as lb
    import labelbox.data.annotation_types as lb_types
except ImportError:
    print("Error: Labelbox SDK not installed. Please run: pip install labelbox>=3.0.0")
    sys.exit(1)

def setup_logging():
    """Configure logging with file and console output"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('simple_labelbox_upload.log'),
            logging.StreamHandler()
        ]
    )

def upload_to_labelbox(api_key, project_id, mask_url=None, image_key=None):
    """Upload annotations using mask URLs to Labelbox"""
    # Check for required parameters
    if not api_key:
        logging.error("API key must be provided")
        return False
    
    if not project_id:
        logging.error("Project ID must be provided")
        return False
    
    if not mask_url:
        # Use a placeholder URL if none provided
        logging.warning("No mask URL provided, using placeholder URL.")
        logging.warning("For production use, replace with actual URLs to mask images in cloud storage")
        mask_url = "https://storage.googleapis.com/your-bucket/mask_placeholder.png"
    
    if not image_key:
        # Use a placeholder key if none provided
        image_key = f"test_image_{uuid.uuid4()}"
        logging.warning(f"No image key provided, using generated key: {image_key}")
        logging.warning("The image with this key must already exist in Labelbox")
    
    try:
        # Initialize client
        logging.info(f"Connecting to Labelbox with project ID: {project_id}")
        client = lb.Client(api_key)
        
        # Check connectivity by getting user info
        user = client.get_user()
        logging.info(f"Connected as user: {user.email}")
        
        # Create the annotations exactly as in the example
        logging.info("Creating annotations")
        color = (255, 255, 255)
        
        # Create different annotations for each damage class
        mask_data_1 = lb_types.MaskData(url=mask_url)
        mask_annotation_1 = lb_types.ObjectAnnotation(
            name="Building-No-Damage",
            value=lb_types.Mask(mask=mask_data_1, color=color)
        )
        
        mask_data_2 = lb_types.MaskData(url=mask_url)
        mask_annotation_2 = lb_types.ObjectAnnotation(
            name="Building-Minor-Damage",
            value=lb_types.Mask(mask=mask_data_2, color=color)
        )
        
        mask_data_3 = lb_types.MaskData(url=mask_url)
        mask_annotation_3 = lb_types.ObjectAnnotation(
            name="Building-Major-Damage",
            value=lb_types.Mask(mask=mask_data_3, color=color)
        )
        
        mask_data_4 = lb_types.MaskData(url=mask_url)
        mask_annotation_4 = lb_types.ObjectAnnotation(
            name="Building-Total-Destruction",
            value=lb_types.Mask(mask=mask_data_4, color=color)
        )
        
        # Combine all annotations
        annotations = [
            mask_annotation_1,
            mask_annotation_2,
            mask_annotation_3,
            mask_annotation_4
        ]
        
        # Create label with the global key
        labels = [
            lb_types.Label(
                data={"global_key": image_key},
                annotations=annotations,
            )
        ]
        
        # Create a unique job name
        job_name = f"mal_job_{uuid.uuid4()}"
        logging.info(f"Starting upload job: {job_name}")
        
        # Upload using MALPredictionImport
        upload_job = lb.MALPredictionImport.create_from_objects(
            client=client,
            project_id=project_id,
            name=job_name,
            predictions=labels
        )
        
        # Wait for the job to complete
        logging.info("Waiting for job to complete...")
        start_time = time.time()
        
        # Monitor job status
        while True:
            # Refresh job status
            upload_job.refresh()
            
            # Get current state and progress
            current_state = upload_job.state if hasattr(upload_job, 'state') else "UNKNOWN"
            current_progress = upload_job.progress if hasattr(upload_job, 'progress') else None
            
            # Log status
            if current_progress is not None:
                progress_percent = int(current_progress * 100)
                logging.info(f"Progress: {progress_percent}% ({current_state})")
            else:
                logging.info(f"Status: {current_state}")
            
            # Check for completion
            if current_state == "COMPLETE":
                logging.info("Job completed successfully!")
                break
            elif current_state in ["FAILED", "ERROR"]:
                if hasattr(upload_job, 'errors') and upload_job.errors:
                    logging.error(f"Job failed with errors: {upload_job.errors}")
                else:
                    logging.error(f"Job failed with status: {current_state}")
                return False
            
            # Check for timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > 300:  # 5 minutes
                logging.error("Job timed out after 5 minutes")
                return False
            
            # Wait before checking again
            time.sleep(10)
        
        # Log completion
        elapsed_time = time.time() - start_time
        logging.info(f"Upload completed in {int(elapsed_time)} seconds")
        logging.info("You can view the annotations in Labelbox")
        
        return True
        
    except Exception as e:
        logging.error(f"Error during upload: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Simple Labelbox MAL annotation uploader')
    parser.add_argument('--api-key', type=str, help='Labelbox API key')
    parser.add_argument('--project-id', type=str, default="cm9kigul10fk807wfeh6sccc5", 
                        help='Labelbox project ID (default from example)')
    parser.add_argument('--mask-url', type=str, 
                        help='URL to a mask image in cloud storage')
    parser.add_argument('--image-key', type=str, 
                        help='Global key of the image in Labelbox')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    # Use environment variable for API key if not provided as argument
    api_key = args.api_key or os.environ.get('LABELBOX_API_KEY')
    if not api_key:
        logging.error("API key must be provided via --api-key or LABELBOX_API_KEY environment variable")
        return 1
    
    # Run the upload
    success = upload_to_labelbox(
        api_key=api_key,
        project_id=args.project_id,
        mask_url=args.mask_url,
        image_key=args.image_key
    )
    
    if success:
        logging.info("Upload completed successfully")
        return 0
    else:
        logging.error("Upload failed. Check logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 