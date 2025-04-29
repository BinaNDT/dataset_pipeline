#!/usr/bin/env python3
"""
Check Labelbox Schema and Category Mappings

This script retrieves and displays the Labelbox project schema/ontology
to help diagnose issues with annotation importing.
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

def check_coco_categories():
    """Check COCO dataset categories"""
    try:
        coco_file = Path("outputs/predictions/predictions_coco.json")
        if not coco_file.exists():
            logging.warning(f"COCO file not found: {coco_file}")
            return None
        
        with open(coco_file, "r") as f:
            coco_data = json.load(f)
        
        categories = coco_data.get("categories", [])
        logging.info(f"Found {len(categories)} categories in COCO file:")
        
        for cat in categories:
            logging.info(f"  - ID: {cat['id']}, Name: {cat['name']}")
        
        return categories
    except Exception as e:
        logging.error(f"Error reading COCO categories: {e}")
        return None

def main():
    """Main function to check Labelbox schema"""
    
    # 1. Connect to Labelbox
    logging.info("Connecting to Labelbox...")
    client = lb.Client(api_key=LABELBOX_API_KEY)
    
    # 2. Get project schema
    logging.info(f"Fetching project schema for ID: {LABELBOX_PROJECT_ID}")
    
    try:
        # Get project
        project = client.get_project(LABELBOX_PROJECT_ID)
        logging.info(f"Connected to project: {project.name}")
        
        # Get ontology
        ontology = project.ontology()
        logging.info(f"Project ontology ID: {ontology.uid}")
        
        # Convert to dict for easier handling
        schema_json = ontology.asdict()
        
        # 3. Print tools and features
        logging.info("Project tools and features:")
        print("\nPROJECT TOOLS AND FEATURES:")
        print("-" * 50)
        
        for tool in schema_json.get('tools', []):
            tool_name = tool.get('name')
            tool_type = tool.get('tool')
            print(f"Tool: {tool_name} (Type: {tool_type})")
            
            # Print classifications (features) for this tool
            for feature in tool.get('classifications', []):
                feature_name = feature.get('name')
                feature_type = feature.get('type')
                print(f"  - Feature: {feature_name} (Type: {feature_type})")
                
                # Print options for radio/checklist types
                if feature_type in ['radio', 'checklist']:
                    print(f"    Options:")
                    for option in feature.get('options', []):
                        option_value = option.get('value')
                        option_label = option.get('label', option_value)
                        print(f"      * {option_value} (Label: {option_label})")
        
        # 4. Get available category classes for polygons
        polygon_classes = []
        for tool in schema_json.get('tools', []):
            if tool.get('tool') == 'polygon':
                for feature in tool.get('classifications', []):
                    if feature.get('type') == 'radio':
                        for option in feature.get('options', []):
                            polygon_classes.append(option.get('value'))
        
        print("\nAVAILABLE POLYGON CLASSES:")
        print("-" * 50)
        for cls in polygon_classes:
            print(f"  - {cls}")
        
        # 5. Compare with COCO categories
        print("\nCOMPARING WITH COCO CATEGORIES:")
        print("-" * 50)
        
        coco_categories = check_coco_categories()
        if coco_categories:
            # Check for mismatches
            coco_names = [cat['name'] for cat in coco_categories]
            
            # Show matches and mismatches
            print("Category matching:")
            for coco_name in coco_names:
                if coco_name in polygon_classes:
                    print(f"  ✓ {coco_name} - Found in both COCO and Labelbox")
                else:
                    print(f"  ✗ {coco_name} - In COCO but NOT in Labelbox")
            
            for lb_class in polygon_classes:
                if lb_class not in coco_names:
                    print(f"  ⚠ {lb_class} - In Labelbox but NOT in COCO")
        
        # 6. Output schema to file for reference
        schema_file = Path("outputs/labelbox_schema.json")
        os.makedirs(schema_file.parent, exist_ok=True)
        
        with open(schema_file, "w") as f:
            json.dump(schema_json, f, indent=2)
        
        logging.info(f"Saved schema to {schema_file}")
        
    except Exception as e:
        logging.error(f"Error checking Labelbox schema: {e}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main() 