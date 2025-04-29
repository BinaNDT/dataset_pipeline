#!/usr/bin/env python3
"""
Upload Images to Labelbox

This script uploads images to Labelbox before attempting to add annotations.
This is a required step - you must upload images first, then add annotations.

Usage:
    python upload_images_to_labelbox.py [--debug] [--limit N] [--dataset DATASET_NAME]
"""

import os
import json
import logging
import sys
import traceback
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm

# Add project root to the path so we can import modules
sys.path.append(str(Path(__file__).parent.parent))
from dataset_pipeline.config import PREDICTIONS_DIR, LOGS_DIR, IMAGES_DIR

# Load environment variables from .env file
load_dotenv()

# Import Labelbox SDK
try:
    import labelbox as lb
except ImportError:
    print("Error: Labelbox SDK not installed. Please run: pip install labelbox>=3.0.0")
    sys.exit(1)


def setup_logging(debug_mode=False):
    """Configure logging with file and console output"""
    log_level = logging.DEBUG if debug_mode else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOGS_DIR / 'labelbox_upload.log'),
            logging.StreamHandler()
        ]
    )


def validate_environment():
    """
    Validate that all required environment variables are set.
    Returns: True if all required environment variables are set, False otherwise.
    """
    valid = True
    
    # Check for required environment variables
    if not os.environ.get("LABELBOX_API_KEY"):
        logging.error("LABELBOX_API_KEY environment variable is not set. Please add it to your .env file.")
        valid = False
    
    if not os.environ.get("LABELBOX_PROJECT_ID"):
        logging.error("LABELBOX_PROJECT_ID environment variable is not set. Please add it to your .env file.")
        valid = False
    
    if not valid:
        logging.error("Please set the required environment variables in your .env file.")
        logging.error("You can copy .env.example to .env and update the values.")
        logging.error("Run: cp .env.example .env && nano .env")
    
    return valid


def get_image_paths_from_coco(coco_file, limit=None):
    """
    Get image paths and global keys from COCO file
    
    Args:
        coco_file (str): Path to COCO file
        limit (int): Limit number of images to return (for debug mode)
        
    Returns:
        list: List of dictionaries containing image paths and global keys
    """
    try:
        # Verify the file exists
        if not Path(coco_file).exists():
            logging.error(f"COCO file not found: {coco_file}")
            return []
            
        # Open and parse the COCO file
        with open(coco_file, 'r') as f:
            try:
                coco_data = json.load(f)
            except json.JSONDecodeError:
                logging.error(f"Invalid JSON in COCO file: {coco_file}")
                return []
                
        # Validate COCO format
        if 'images' not in coco_data:
            logging.error(f"Invalid COCO format - missing 'images' section: {coco_file}")
            return []
            
        # Get all image filenames from COCO file
        images = []
        for img in coco_data['images']:
            if 'file_name' not in img:
                logging.warning(f"Image entry missing file_name: {img}")
                continue
                
            image_name = img['file_name']
            image_id = Path(image_name).stem
            
            # Find the actual file - we need to search through video directories
            # since the COCO file doesn't contain the full path
            found = False
            
            # Verify images directory exists
            if not IMAGES_DIR.exists():
                logging.error(f"Images directory not found: {IMAGES_DIR}")
                return []
                
            # List all video directories
            try:
                video_dirs = [d for d in IMAGES_DIR.iterdir() if d.is_dir()]
                logging.debug(f"Searching for {image_name} in {len(video_dirs)} video directories")
            except Exception as e:
                logging.error(f"Error reading video directories: {e}")
                return []
            
            # Search each video directory for the image
            for video_dir in video_dirs:
                image_path = video_dir / image_name
                if image_path.exists():
                    # Create a unique global key by combining video directory name and image ID
                    video_name = video_dir.name
                    global_key = f"{video_name}_{image_id}"
                    
                    # Detailed logging for global key creation
                    logging.debug(f"Image found: {image_name}")
                    logging.debug(f"  - Video directory: {video_name}")
                    logging.debug(f"  - Image ID: {image_id}")
                    logging.debug(f"  - Generated global key: {global_key}")
                    
                    images.append({
                        'file_path': str(image_path),
                        'global_key': global_key,
                        'width': img.get('width', 0),
                        'height': img.get('height', 0)
                    })
                    found = True
                    break
            
            if not found:
                logging.warning(f"Could not find image file: {image_name} in any video directory")
        
        logging.info(f"Found {len(images)} images out of {len(coco_data['images'])} in COCO file")
        
        if len(images) == 0:
            logging.error("No images could be found matching the COCO file entries")
            return []
            
        if limit and limit > 0:
            logging.info(f"Limiting to {limit} images for debug mode")
            images = images[:limit]
            
        return images
        
    except Exception as e:
        logging.error(f"Error processing COCO file: {e}")
        logging.error(traceback.format_exc())
        return []


def upload_images_to_labelbox(dataset_name="ian_pipeline", debug=False, target_count=10):
    """
    Upload images to Labelbox
    
    Args:
        dataset_name (str): Name of the dataset (defaults to 'ian_pipeline')
        debug (bool): Debug mode - limit uploads to target_count
        target_count (int): Number of images to upload in debug mode
    
    Returns:
        int: Number of successful uploads
    """
    # Validate environment before proceeding
    if not validate_environment():
        logging.error("Environment validation failed. Cannot proceed with upload.")
        return 0
    
    # Get API key and project ID from environment
    api_key = os.environ.get('LABELBOX_API_KEY')
    project_id = os.environ.get('LABELBOX_PROJECT_ID')
    
    # Create Labelbox client
    try:
        client = lb.Client(api_key=api_key)
        logging.info("Successfully authenticated with Labelbox API")
    except Exception as e:
        logging.error(f"Failed to authenticate with Labelbox API: {e}")
        return 0
    
    # Get project
    try:
        project = client.get_project(project_id)
        logging.info(f"Connected to Labelbox project: {project.name}")
    except Exception as e:
        logging.error(f"Failed to access Labelbox project: {e}")
        logging.error("Please verify your LABELBOX_PROJECT_ID is correct in .env file")
        return 0
    
    # Find existing dataset by name
    try:
        datasets = client.get_datasets(where=(lb.Dataset.name == dataset_name))
        dataset = datasets.get_one()
        if dataset:
            logging.info(f"Found existing dataset: {dataset.name} ({dataset.uid})")
        else:
            # If dataset doesn't exist, create it
            logging.info(f"Dataset '{dataset_name}' not found, creating it now")
            dataset = client.create_dataset(name=dataset_name)
            logging.info(f"Created dataset: {dataset.name} ({dataset.uid})")
    except Exception as e:
        logging.error(f"Failed to find or create dataset: {e}")
        return 0
    
    # Get images from COCO file
    coco_file = PREDICTIONS_DIR / 'predictions_coco.json'
    try:
        with open(coco_file) as f:
            coco_data = json.load(f)
    except FileNotFoundError:
        logging.error(f"COCO file not found: {coco_file}")
        return 0
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in COCO file: {coco_file}")
        return 0
    
    # Get images
    images = coco_data["images"]
    logging.info(f"Found {len(images)} images in COCO file")
    
    if debug:
        logging.info(f"Debug mode: limiting to {target_count} images")
        images = images[:target_count]
    
    # Upload images
    data_rows = []
    
    # Find all video directories once to improve performance
    video_dirs = [d for d in IMAGES_DIR.iterdir() if d.is_dir()]
    logging.info(f"Found {len(video_dirs)} video directories to search for images")
    
    for image in tqdm(images, desc="Preparing images"):
        image_id = image["id"]
        file_name = image["file_name"]
        
        # The file_name in COCO may include the video subdirectory
        image_path = IMAGES_DIR / file_name
        
        # If the direct path doesn't exist, try to split and find in subdirectories
        if not image_path.exists():
            # Try to extract the filename without subdirectory
            parts = Path(file_name).parts
            if len(parts) > 1:
                # The file_name already contains a directory, try that
                image_path = IMAGES_DIR / file_name
            else:
                # Search for the image in all video directories
                found = False
                for video_dir in video_dirs:
                    potential_path = video_dir / file_name
                    if potential_path.exists():
                        image_path = potential_path
                        found = True
                        break
                if not found:
                    logging.warning(f"Image not found: {file_name}")
                    continue
        
        # Log every 100 images to show progress
        if len(data_rows) % 100 == 0 and len(data_rows) > 0:
            logging.info(f"Prepared {len(data_rows)} images so far")
        
        # Create data row - upload local file to Labelbox
        try:
            # Upload the image to Labelbox storage and get a URL
            file_url = client.upload_file(str(image_path.absolute()))
            logging.debug(f"Uploaded {image_path} to {file_url}")
            
            # Create a unique global key from image ID and file path
            global_key = f"{image_id}_{Path(file_name).stem}"
            
            # Add timestamp to global key to avoid duplicates
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            global_key = f"{global_key}_{timestamp}"
            
            data_row = {
                "row_data": file_url,
                "external_id": str(image_id),
                "global_key": global_key
            }
            data_rows.append(data_row)
        except Exception as e:
            logging.error(f"Error uploading file {file_name}: {e}")
    
    if not data_rows:
        logging.error("No valid images found to upload")
        return 0
        
    logging.info(f"Uploading {len(data_rows)} images to Labelbox")
    
    # Upload images in batches
    batch_size = 1000
    total_images = len(data_rows)
    if total_images == 0:
        logging.error("No valid images found to upload.")
        return 0

    logging.info(f"Uploading {total_images} images in batches of {batch_size}")
    
    success_count = 0
    failure_count = 0
    
    # Process in batches
    for i in range(0, total_images, batch_size):
        batch = data_rows[i:i+batch_size]
        logging.info(f"Processing batch {i//batch_size + 1}/{(total_images + batch_size - 1)//batch_size} ({len(batch)} images)")
        
        try:
            task = dataset.create_data_rows(batch)
            task.wait_till_done()
            
            # Check if task.errors is None before trying to get its length
            if task.errors is not None and len(task.errors) > 0:
                logging.error(f"Errors during batch upload: {task.errors}")
                failure_count += len(batch)
            else:
                success_count += len(batch)
                logging.info(f"Successfully uploaded batch {i//batch_size + 1}")
        except Exception as e:
            logging.error(f"Error during batch upload: {str(e)}")
            failure_count += len(batch)
    
    logging.info(f"Uploaded {success_count} images successfully, {failure_count} failed")
    
    # Connect dataset to project if not already connected
    try:
        # Instead of checking if the dataset is connected to the project,
        # we'll just attempt to connect it and catch any errors if it's already connected
        try:
            project.add_dataset(dataset)
            logging.info(f"Successfully attached dataset to project")
        except Exception as e:
            if "already attached" in str(e).lower():
                logging.info(f"Dataset already attached to project")
            else:
                raise e
    except Exception as e:
        logging.error(f"Failed to attach dataset to project: {str(e)}")
    
    return success_count


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Upload images to Labelbox")
    parser.add_argument("--debug", action="store_true", help="Debug mode - limit uploads")
    parser.add_argument("--limit", type=int, default=10, help="Number of images to upload in debug mode")
    parser.add_argument("--dataset", type=str, default="ian_pipeline", help="Name of the dataset to use")
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Set up logging with our standardized function
    setup_logging(args.debug)
    logging.info("Starting Labelbox image upload script")
    
    # Verify .env file is loaded
    if not os.path.exists('.env'):
        logging.error("No .env file found in the current directory.")
        logging.error("Please create a .env file with your Labelbox credentials.")
        logging.error("You can copy .env.example to .env and update the values.")
        logging.error("Run: cp .env.example .env && nano .env")
        return 1
    
    # Validate environment
    if not validate_environment():
        logging.error("Environment validation failed. Exiting.")
        return 1
    
    try:
        # Upload images to Labelbox
        logging.info(f"Using dataset named: {args.dataset}")
        success_count = upload_images_to_labelbox(args.dataset, args.debug, args.limit)
        
        if success_count > 0:
            logging.info(f"Successfully uploaded {success_count} images to Labelbox")
            return 0
        else:
            logging.error("Failed to upload any images to Labelbox")
            return 1
    except Exception as e:
        logging.error(f"Unexpected error in main execution: {e}")
        logging.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main()) 