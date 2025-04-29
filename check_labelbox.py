#!/usr/bin/env python3
"""
Check Labelbox Data

This script prints detailed information about data rows in Labelbox
to help diagnose issues with annotations.
"""

import os
import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Import configuration
from config import LABELBOX_API_KEY, LABELBOX_PROJECT_ID

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

try:
    import labelbox as lb
except ImportError:
    logging.error("Labelbox SDK not installed. Please run: pip install labelbox")
    sys.exit(1)

def main():
    """Main function to check Labelbox data"""
    
    # 1. Connect to Labelbox
    logging.info("Connecting to Labelbox...")
    client = lb.Client(api_key=LABELBOX_API_KEY)
    
    # 2. Get project details
    logging.info(f"Fetching project details for ID: {LABELBOX_PROJECT_ID}")
    
    project_query = f"""
    {{
      project(where: {{ id: "{LABELBOX_PROJECT_ID}" }}) {{
        name
        description
        datasets {{
          id
          name
          rowCount
        }}
      }}
    }}
    """
    
    try:
        project_result = client.execute(project_query)
        project_info = project_result.get("project", {})
        
        if not project_info:
            logging.error(f"Project not found with ID: {LABELBOX_PROJECT_ID}")
            return
            
        logging.info(f"Project name: {project_info.get('name')}")
        logging.info(f"Project description: {project_info.get('description')}")
        
        # 3. Get datasets in the project
        datasets = project_info.get("datasets", [])
        logging.info(f"Found {len(datasets)} datasets in the project")
        
        for dataset in datasets:
            dataset_id = dataset.get("id")
            dataset_name = dataset.get("name")
            row_count = dataset.get("rowCount", 0)
            
            logging.info(f"Dataset: {dataset_name} (ID: {dataset_id}) - {row_count} rows")
            
            # 4. Get data rows for each dataset
            data_row_query = f"""
            {{
              dataset(where: {{ id: "{dataset_id}" }}) {{
                dataRows(skip: 0, first: 100) {{
                  id
                  externalId
                  rowData
                }}
              }}
            }}
            """
            
            data_row_result = client.execute(data_row_query)
            data_rows = data_row_result.get("dataset", {}).get("dataRows", [])
            
            logging.info(f"Fetched {len(data_rows)} data rows from dataset {dataset_name}")
            
            # 5. Print data row details
            for i, row in enumerate(data_rows):
                row_id = row.get("id")
                external_id = row.get("externalId", "None")
                row_data = row.get("rowData", {})
                
                logging.info(f"Row {i+1}:")
                logging.info(f"  ID: {row_id}")
                logging.info(f"  External ID: {external_id}")
                logging.info(f"  Row Data: {json.dumps(row_data, indent=2)}")
    
    except Exception as e:
        logging.error(f"Error checking Labelbox data: {e}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main() 