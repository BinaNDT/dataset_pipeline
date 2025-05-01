#!/usr/bin/env python3
"""
Simple Labelbox Test Import

This script attempts to import a single test annotation to verify the Labelbox connection
and that the data row exists in the project.
"""

import os
import json
import sys
import uuid
import time
from pathlib import Path
from datetime import datetime
import traceback

# Import config
sys.path.append(str(Path(__file__).parent))
from config import LABELBOX_API_KEY, LABELBOX_PROJECT_ID

# Import Labelbox
try:
    import labelbox as lb
except ImportError:
    print("Error: Labelbox SDK not installed. Please run: pip install labelbox>=3.0.0")
    sys.exit(1)

def main():
    print("=== Simple Labelbox Test Import ===")
    print(f"Project ID: {LABELBOX_PROJECT_ID}")
    
    # Connect to Labelbox
    print("Connecting to Labelbox...")
    client = lb.Client(api_key=LABELBOX_API_KEY)
    
    try:
        # Verify project exists
        project = client.get_project(LABELBOX_PROJECT_ID)
        print(f"Connected to project: {project.name}")
        
        # Get the available dataset in the project
        print("Checking for existing data rows...")
        datasets = project.datasets()
        
        # Find a data row to test with
        data_row = None
        
        for dataset in datasets:
            print(f"Checking dataset: {dataset.name} (ID: {dataset.uid})")
            data_rows = dataset.data_rows().items()
            
            # Get the first data row
            for dr in data_rows:
                data_row = dr
                print(f"Found data row with ID: {data_row.uid}, Global Key: {data_row.global_key}")
                break
                
            if data_row:
                break
        
        if not data_row:
            print("Error: No data rows found in the project.")
            sys.exit(1)
        
        # Create a simple polygon annotation
        points = [
            {"x": 100, "y": 100},
            {"x": 100, "y": 200},
            {"x": 200, "y": 200},
            {"x": 200, "y": 100},
            {"x": 100, "y": 100}  # Close the polygon
        ]
        
        # Get project's ontology to find feature schema IDs
        print("Getting project ontology...")
        ontology = project.ontology()
        tools = ontology.tools()
        
        # Find a valid feature name
        feature_name = None
        for tool in tools:
            print(f"Found tool: {tool.name} (Schema ID: {tool.feature_schema_id})")
            if tool.tool == "polygon":
                feature_name = tool.name
                break
        
        if not feature_name:
            print("Error: No polygon tool found in ontology")
            feature_name = "Building-Total-Destruction"  # Fallback
            print(f"Using fallback feature name: {feature_name}")
        
        # Create test annotation
        print(f"Creating test annotation with feature name: {feature_name}")
        
        # Method 1: Direct API approach
        print("\nMethod 1: Using direct format")
        annotation = {
            "uuid": str(uuid.uuid4()),
            "name": feature_name,
            "polygon": points,
            "dataRow": {
                "globalKey": data_row.global_key
            }
        }
        
        # Save test annotation
        test_file = Path("outputs/predictions/direct_test.ndjson")
        with open(test_file, "w") as f:
            f.write(json.dumps(annotation))
        
        print(f"Created test annotation file: {test_file}")
        
        # Import using direct approach
        print("Importing using direct approach...")
        import_name = f"Direct_Test_{datetime.now().strftime('%H%M%S')}"
        
        try:
            upload_job = lb.MALPredictionImport.create_from_objects(
                client=client,
                project_id=LABELBOX_PROJECT_ID,
                name=import_name,
                predictions=[annotation]
            )
            
            print(f"Upload started with ID: {upload_job.uid}")
            
            # Monitor progress
            max_wait = 60  # 1 minute max wait
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                upload_job.refresh()
                
                state = getattr(upload_job, 'state', "Unknown")
                progress = getattr(upload_job, 'progress', "Unknown")
                    
                print(f"Status: {state}, Progress: {progress}")
                
                if state in ['COMPLETE', 'FINISHED']:
                    print("Direct upload completed successfully!")
                    break
                elif state in ['FAILED', 'ERROR']:
                    print("Direct upload failed.")
                    errors = getattr(upload_job, 'errors', [])
                    if errors:
                        print(f"Errors: {errors}")
                    break
                    
                time.sleep(5)
                
        except Exception as e:
            print(f"Error in direct upload: {e}")
            traceback.print_exc()
        
        # Method 2: SDK approach
        print("\nMethod 2: Using SDK classes")
        try:
            # Create annotation using SDK classes
            obj_annotation = lb.ObjectAnnotation(
                name=feature_name,
                value=lb.Polygon(points=[lb.Point(x=p["x"], y=p["y"]) for p in points])
            )
            
            # Create label
            label = lb.Label(
                data={"global_key": data_row.global_key},
                annotations=[obj_annotation]
            )
            
            # Import
            sdk_import_name = f"SDK_Test_{datetime.now().strftime('%H%M%S')}"
            sdk_upload_job = lb.MALPredictionImport.create_from_objects(
                client=client,
                project_id=LABELBOX_PROJECT_ID,
                name=sdk_import_name,
                predictions=[label]
            )
            
            print(f"SDK upload started with ID: {sdk_upload_job.uid}")
            
            # Monitor progress
            max_wait = 60  # 1 minute max wait
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                sdk_upload_job.refresh()
                
                state = getattr(sdk_upload_job, 'state', "Unknown")
                progress = getattr(sdk_upload_job, 'progress', "Unknown")
                    
                print(f"SDK Status: {state}, Progress: {progress}")
                
                if state in ['COMPLETE', 'FINISHED']:
                    print("SDK upload completed successfully!")
                    break
                elif state in ['FAILED', 'ERROR']:
                    print("SDK upload failed.")
                    errors = getattr(sdk_upload_job, 'errors', [])
                    if errors:
                        print(f"SDK Errors: {errors}")
                    break
                    
                time.sleep(5)
        except Exception as e:
            print(f"Error in SDK upload: {e}")
            traceback.print_exc()
            
        print("Test completed.")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 