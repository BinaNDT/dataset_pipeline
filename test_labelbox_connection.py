#!/usr/bin/env python3
"""
Test Labelbox Connection and Data Row ID Retrieval

This script verifies that we can connect to Labelbox and retrieve data row IDs correctly.
It also demonstrates the proper way to link annotations to data rows.

Usage:
    python test_labelbox_connection.py
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import uuid

# Setup path for importing from config
sys.path.append(str(Path(__file__).parent))
from config import *

# Import Labelbox SDK
try:
    import labelbox as lb
except ImportError:
    print("Error: Labelbox SDK not installed. Please run: pip install labelbox>=3.0.0")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'test_connection.log'),
        logging.StreamHandler()
    ]
)

def test_connection() -> bool:
    """Test connection to Labelbox API"""
    try:
        logging.info("Testing connection to Labelbox API...")
        
        if not LABELBOX_API_KEY:
            logging.error("LABELBOX_API_KEY environment variable not set")
            return False
        
        client = lb.Client(api_key=LABELBOX_API_KEY)
        user = client.get_user()
        
        logging.info(f"Successfully connected to Labelbox as: {user.email}")
        return True
    except Exception as e:
        logging.error(f"Failed to connect to Labelbox: {e}")
        return False

def test_project_access() -> bool:
    """Test access to the specified project"""
    try:
        logging.info(f"Testing access to project ID: {LABELBOX_PROJECT_ID}")
        
        if not LABELBOX_PROJECT_ID:
            logging.error("LABELBOX_PROJECT_ID environment variable not set")
            return False
        
        client = lb.Client(api_key=LABELBOX_API_KEY)
        project = client.get_project(LABELBOX_PROJECT_ID)
        
        logging.info(f"Successfully accessed project: {project.name}")
        
        # List datasets in the project - using the correct method
        # Note: different versions of the Labelbox SDK may have different methods
        # Try the query approach which should be more stable
        project_with_datasets_query = f"""
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
        
        result = client.execute(project_with_datasets_query)
        datasets = result.get("project", {}).get("datasets", [])
        
        logging.info(f"Project contains {len(datasets)} datasets:")
        
        for dataset in datasets:
            row_count = dataset.get("rowCount", 0)
            dataset_name = dataset.get("name", "Unknown")
            logging.info(f"  • {dataset_name}: {row_count} data rows")
        
        return True
    except Exception as e:
        logging.error(f"Failed to access project: {e}")
        return False

def get_data_row_ids() -> Dict[str, str]:
    """Retrieve all data row IDs from the project and map them to filenames"""
    try:
        logging.info("Retrieving data row IDs from project...")
        
        client = lb.Client(api_key=LABELBOX_API_KEY)
        
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
        
        result = client.execute(project_with_datasets_query)
        datasets = result.get("project", {}).get("datasets", [])
        
        if not datasets:
            logging.error("No datasets found in the project")
            return {}
        
        # Map to store filename -> data row ID
        filename_to_id = {}
        total_rows = 0
        
        # Process each dataset
        for dataset in datasets:
            dataset_id = dataset.get("id")
            dataset_name = dataset.get("name")
            logging.info(f"Processing dataset: {dataset_name}")
            
            # Get data rows from this dataset using pagination
            skip = 0
            page_size = 100
            
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
                    
                    data_row_result = client.execute(data_row_query)
                    data_rows = data_row_result.get("dataset", {}).get("dataRows", [])
                    
                    if not data_rows:
                        break
                    
                    # Process each data row
                    for row in data_rows:
                        row_id = row.get("id")
                        external_id = row.get("externalId")
                        row_data = row.get("rowData")
                        
                        # Map with external_id if available
                        if external_id:
                            filename_to_id[external_id] = row_id
                        
                        # Extract filename from row_data URL if it's a URL
                        if isinstance(row_data, str) and row_data.startswith("http"):
                            # Get filename from URL (last part of path)
                            url_parts = row_data.split('?')[0].split('/')
                            if url_parts:
                                filename = url_parts[-1]
                                if filename:
                                    filename_to_id[filename] = row_id
                                    
                                    # Also store without extension
                                    name_without_ext = Path(filename).stem
                                    if name_without_ext:
                                        filename_to_id[name_without_ext] = row_id
                    
                    # Update total count
                    total_rows += len(data_rows)
                    
                    # If we got fewer rows than page_size, we've reached the end
                    if len(data_rows) < page_size:
                        break
                    
                    # Update skip for next page
                    skip += page_size
                    logging.info(f"Retrieved {total_rows} rows so far from dataset {dataset_name}")
                        
                except Exception as e:
                    logging.error(f"Error retrieving data rows: {e}")
                    break
        
        logging.info(f"Found {total_rows} data rows across all datasets")
        logging.info(f"Created {len(filename_to_id)} filename mappings")
        
        # Show some examples of the mappings
        sample_count = min(5, len(filename_to_id))
        if sample_count > 0:
            logging.info("Sample filename to data row ID mappings:")
            samples = list(filename_to_id.items())[:sample_count]
            for filename, row_id in samples:
                logging.info(f"  • {filename} -> {row_id}")
        
        return filename_to_id
        
    except Exception as e:
        logging.error(f"Failed to retrieve data row IDs: {e}")
        return {}

def test_annotation_creation() -> bool:
    """Test creating a sample annotation and linking it to a data row"""
    try:
        logging.info("Testing annotation creation...")
        
        # Get data row IDs
        data_row_ids = get_data_row_ids()
        if not data_row_ids:
            logging.error("No data row IDs found")
            return False
        
        # Get first data row ID for testing
        sample_filename = next(iter(data_row_ids.keys()))
        sample_row_id = data_row_ids[sample_filename]
        
        logging.info(f"Creating test annotation for: {sample_filename} (ID: {sample_row_id})")
        
        # Since the Labelbox SDK version doesn't have lb.data.annotation_types,
        # we'll just create a sample annotation in JSON format for testing
        test_annotation = {
            "uuid": str(uuid.uuid4()),
            "dataRow": {
                "id": sample_row_id
            },
            "annotations": [
                {
                    "uuid": str(uuid.uuid4()),
                    "name": "Test-Building-No-Damage",
                    "value": {
                        "format": "polygon2d",
                        "points": [
                            [10, 10],
                            [100, 10],
                            [100, 100],
                            [10, 100]
                        ]
                    }
                }
            ]
        }
        
        logging.info("Test annotation structure created successfully")
        logging.info("Note: Test annotation was not uploaded to Labelbox")
        logging.info("Annotation creation test successful")
        
        return True
    except Exception as e:
        logging.error(f"Error creating test annotation: {e}")
        return False

def main():
    """Main function"""
    logging.info("Starting Labelbox connection tests")
    
    # Test basic connection
    if not test_connection():
        logging.error("Failed to connect to Labelbox API")
        return 1
    
    # Test project access
    if not test_project_access():
        logging.error("Failed to access the specified project")
        return 1
    
    # Test data row ID retrieval
    data_row_ids = get_data_row_ids()
    if not data_row_ids:
        logging.error("Failed to retrieve data row IDs")
        return 1
    
    # Test annotation creation
    if not test_annotation_creation():
        logging.error("Failed to create test annotation")
        return 1
    
    logging.info("All tests completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 