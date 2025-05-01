#!/usr/bin/env python3
"""
Basic Labelbox import example following documentation
"""
import os
import json
import numpy as np
import time
from dotenv import load_dotenv
from datetime import datetime
import cv2
import base64
from io import BytesIO

# Load environment variables
load_dotenv()

# Import Labelbox
import labelbox as lb

# Configuration
API_KEY = os.getenv('LABELBOX_API_KEY')
PROJECT_ID = os.getenv('LABELBOX_PROJECT_ID')
DATA_ROW_ID = "cma2wb55mw6vh0773ggg252nr"  # Test with one data row

def main():
    print(f"Running basic Labelbox import test for data row: {DATA_ROW_ID}")
    
    # Create client
    client = lb.Client(api_key=API_KEY)
    
    # Get project info
    project = client.get_project(PROJECT_ID)
    print(f"Connected to project: {project.name}")
    
    # Simple binary mask: small box in center
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:40, 20:40] = 1  # 20x20 box in center
    
    # Properly encode mask to PNG and then to base64
    # Convert to PNG using OpenCV
    _, buffer = cv2.imencode('.png', mask * 255)  # Scale to 0-255 range
    
    # Convert to base64
    encoded_mask = base64.b64encode(buffer).decode('utf-8')
    
    # Create predictions
    prediction = {
        "dataRow": {
            "id": DATA_ROW_ID
        },
        "schemaId": "cm6rebjoj041807zb4irbemc2",  # Building-No-Damage schema ID
        "mask": {
            "png": f"data:image/png;base64,{encoded_mask}"
        }
    }
    
    # Save for reference
    with open("example_prediction.json", "w") as f:
        json.dump(prediction, f, indent=2)
    
    # Create import job
    import_name = f"Example_{datetime.now().strftime('%H%M%S')}"
    
    try:
        print(f"Creating import job: {import_name}")
        upload_job = lb.MALPredictionImport.create_from_objects(
            client=client,
            project_id=PROJECT_ID,
            name=import_name,
            predictions=[prediction]
        )
        
        print(f"Upload started with ID: {upload_job.uid}")
        
        # Monitor progress
        max_wait = 60  # 1 minute max wait
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            upload_job.refresh()
            
            if hasattr(upload_job, 'state'):
                state = upload_job.state
            else:
                state = "Unknown"
                
            if hasattr(upload_job, 'progress'):
                progress = upload_job.progress
            else:
                progress = "Unknown"
                
            print(f"Status: {state}, Progress: {progress}")
            
            if state in ['COMPLETE', 'FINISHED']:
                print("Upload completed successfully!")
                break
            elif state in ['FAILED', 'ERROR']:
                print("Upload failed.")
                if hasattr(upload_job, 'errors') and upload_job.errors:
                    print(f"Errors: {upload_job.errors}")
                break
                
            time.sleep(5)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
    print("Test completed.")

if __name__ == "__main__":
    main() 