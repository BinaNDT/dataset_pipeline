#!/usr/bin/env python3
"""
Import mask predictions to Labelbox using the Python SDK
This script demonstrates creating model-assisted labeling predictions
with binary masks for a given data row (image) and uploading them.
"""
import os
import time
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import Labelbox SDK
import labelbox as lb
from labelbox.data.annotation_types import ObjectAnnotation, MaskData, Mask
from labelbox.data.media_types import ImageData

# Configuration - update these variables as needed
API_KEY = os.getenv('LABELBOX_API_KEY')
PROJECT_ID = os.getenv('LABELBOX_PROJECT_ID')
# Use the global_key you assigned when uploading images to Labelbox
GLOBAL_KEY = "video1_frame0001_20250429_123456"

# Schema ID for your ontology class (e.g., from get_ontology_schema_ids output)
# Replace with the correct CUID for "Building-No-Damage" or other classes
SCHEMA_ID = "cm6rebjoj041807zb4irbemc2"  # Example schema ID

# Binary mask shape - adjust to your image dimensions
MASK_HEIGHT = 512
MASK_WIDTH = 512

# Create a sample binary mask (square in center)
def create_sample_mask():
    mask = np.zeros((MASK_HEIGHT, MASK_WIDTH), dtype=np.uint8)
    # Draw a centered square
    h0, h1 = MASK_HEIGHT//4, 3*MASK_HEIGHT//4
    w0, w1 = MASK_WIDTH//4, 3*MASK_WIDTH//4
    mask[h0:h1, w0:w1] = 1
    return mask

# Upload function
def upload_mask_prediction():
    # Initialize client
    client = lb.Client(api_key=API_KEY)
    print(f"Connected to Labelbox project: {PROJECT_ID}")

    # Create mask array
    mask_array = create_sample_mask()

    # Wrap mask in MaskData
    mask_data = MaskData(mask=mask_array)

    # Create Mask annotation
    annotation = ObjectAnnotation(
        name="Building-No-Damage",  # Must match your Labelbox class name
        value=Mask(mask=mask_data)
    )

    # Create Label with ImageData referencing the global_key
    label = lb.data.Label(
        data=ImageData(global_key=GLOBAL_KEY),
        annotations=[annotation]
    )

    # Start import job
    job_name = f"MaskImport_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"Starting import job: {job_name}")
    upload_job = lb.MALPredictionImport.create_from_objects(
        client=client,
        project_id=PROJECT_ID,
        name=job_name,
        predictions=[label]
    )

    # Monitor progress
    start = time.time()
    timeout = 5 * 60  # 5 minutes
    while True:
        upload_job.refresh()
        state = upload_job.state if hasattr(upload_job, 'state') else upload_job.status
        progress = upload_job.progress if hasattr(upload_job, 'progress') else None
        print(f"Status: {state}, Progress: {progress}")
        if state in ["COMPLETE", "FINISHED"]:
            print("Import completed successfully.")
            break
        if state in ["FAILED", "ERROR"]:
            print(f"Import failed: {upload_job.errors}")
            break
        if time.time() - start > timeout:
            print("Import timed out.")
            break
        time.sleep(10)

if __name__ == "__main__":
    upload_mask_prediction() 