#!/usr/bin/env python3
"""
Test importing to Labelbox using SDK classes
"""

import os
import numpy as np
import time
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Import Labelbox SDK classes
import labelbox as lb
# Try different import paths for the SDK
try:
    # Try the current path structure
    from labelbox.schema.annotation import ObjectAnnotation
    from labelbox.schema.annotation import Mask
    from labelbox.schema.data import ImageData
    from labelbox.schema.label_import import Label
except ImportError:
    try:
        # Try alternate path structure
        from labelbox.schema.annotation_types import ObjectAnnotation, Mask
        from labelbox.schema.media_type import ImageData
        from labelbox.schema.label_import import Label
    except ImportError:
        # Fallback to older version
        print("Warning: Using older/alternate Labelbox SDK import structure")
        import labelbox.types as lb_types
        ObjectAnnotation = lb_types.ObjectAnnotation
        Mask = lb_types.Mask
        ImageData = lb_types.ImageData
        Label = lb_types.Label

# Get API key and project ID
API_KEY = os.getenv('LABELBOX_API_KEY')
PROJECT_ID = os.getenv('LABELBOX_PROJECT_ID')
DATA_ROW_ID = "cma2wb55mw6vh0773ggg252nr"  # Test with one data row ID

def main():
    print("Testing Labelbox import using SDK classes...")
    
    # Create Labelbox client
    client = lb.Client(api_key=API_KEY)
    
    # Get project
    project = client.get_project(PROJECT_ID)
    print(f"Connected to project: {project.name}")
    
    # Create a test mask (20x20 with a square in the middle)
    mask_data = np.zeros((20, 20), dtype=np.uint8)
    mask_data[5:15, 5:15] = 1  # Square in the middle
    
    # Create a Mask object
    mask = Mask(mask=mask_data)
    
    # Create object annotation
    annotation = ObjectAnnotation(
        name="Building-No-Damage",
        value=mask
    )
    
    # Create label
    label = Label(
        # Use external_id to reference the data row
        data=ImageData(external_id=DATA_ROW_ID),
        annotations=[annotation]
    )
    
    # Import label
    print("Uploading annotation...")
    import_name = f"SDK_Test_{datetime.now().strftime('%H%M%S')}"
    
    upload_job = lb.LabelImport.create_from_objects(
        client=client,
        project_id=PROJECT_ID,
        name=import_name,
        labels=[label]
    )
    
    print(f"Upload started with ID: {upload_job.uid}")
    
    # Monitor progress
    max_wait = 60  # 1 minute timeout
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        upload_job.refresh()
        
        print(f"Status: {upload_job.status}, Progress: {upload_job.progress if hasattr(upload_job, 'progress') else 'unknown'}")
        
        if upload_job.finished:
            if upload_job.errors:
                print(f"Upload finished with errors: {upload_job.errors}")
            else:
                print("Upload completed successfully!")
            break
        
        time.sleep(5)
    
    print("Testing completed.")

if __name__ == "__main__":
    main() 