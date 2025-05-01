#!/usr/bin/env python3
"""
Simple Labelbox Schema Checker
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
    
    # 2. Get project info
    logging.info(f"Fetching project info for ID: {LABELBOX_PROJECT_ID}")
    
    try:
        # Get project
        project = client.get_project(LABELBOX_PROJECT_ID)
        logging.info(f"Connected to project: {project.name}")
        
        # Get project information using a simpler query
        project_query = f"""
        {{
          project(where: {{ id: "{LABELBOX_PROJECT_ID}" }}) {{
            name
            description
            ontology {{
              id
              name
              normalized
            }}
          }}
        }}
        """
        
        project_result = client.execute(project_query)
        project_info = project_result.get('project', {})
        
        print("\nPROJECT INFO:")
        print("-" * 50)
        print(f"Name: {project_info.get('name')}")
        print(f"Description: {project_info.get('description')}")
        
        # Get ontology info
        ontology_info = project_info.get('ontology', {})
        print(f"Ontology ID: {ontology_info.get('id')}")
        print(f"Ontology Name: {ontology_info.get('name')}")
        print(f"Normalized: {ontology_info.get('normalized')}")
        
        # Try querying for ontology schema via specific schema fields
        ontology_id = ontology_info.get('id')
        if ontology_id:
            try:
                # Query for the actual ontology schema details
                schema_query = f"""
                {{
                  ontology(where: {{ id: "{ontology_id}" }}) {{
                    name
                    jsonInterface
                  }}
                }}
                """
                
                schema_result = client.execute(schema_query)
                schema_info = schema_result.get('ontology', {})
                
                # Try to get the schema from jsonInterface
                schema_json = None
                if 'jsonInterface' in schema_info:
                    try:
                        schema_json = json.loads(schema_info['jsonInterface'])
                        print("\nSchema loaded successfully from jsonInterface")
                    except json.JSONDecodeError:
                        logging.error("Could not parse jsonInterface as JSON")
                
                if schema_json:
                    # Extract available classes based on the schema format
                    available_classes = []
                    
                    # Look for classifications in the schema
                    if 'classifications' in schema_json:
                        for classification in schema_json['classifications']:
                            if classification.get('type') == 'radio':
                                for option in classification.get('options', []):
                                    available_classes.append(option.get('value'))
                    
                    # Also look in tools for polygon
                    if 'tools' in schema_json:
                        for tool in schema_json['tools']:
                            if tool.get('tool') == 'polygon':
                                for feature in tool.get('classifications', []):
                                    if feature.get('type') == 'radio':
                                        for option in feature.get('options', []):
                                            available_classes.append(option.get('value'))
                            
                    # List all found classes
                    print("\nAVAILABLE CLASSES:")
                    print("-" * 50)
                    for cls in available_classes:
                        print(f"  - {cls}")
                    
                    # Compare with COCO
                    coco_categories = check_coco_categories()
                    if coco_categories:
                        coco_names = [cat['name'] for cat in coco_categories]
                        
                        print("\nCATEGORY MATCHING:")
                        print("-" * 50)
                        for coco_name in coco_names:
                            if coco_name in available_classes:
                                print(f"  ✓ {coco_name} - Found in both COCO and Labelbox")
                            else:
                                print(f"  ✗ {coco_name} - In COCO but NOT in Labelbox")
                        
                        for lb_class in available_classes:
                            if lb_class not in coco_names:
                                print(f"  ⚠ {lb_class} - In Labelbox but NOT in COCO")
                    
                    # Save schema to file
                    schema_file = Path("outputs/labelbox_schema.json")
                    os.makedirs(schema_file.parent, exist_ok=True)
                    
                    with open(schema_file, "w") as f:
                        json.dump(schema_json, f, indent=2)
                    
                    logging.info(f"Saved schema to {schema_file}")
                else:
                    print("\nCould not extract schema JSON interface.")
                    print("Let's try a direct approach to get class names:")
                    
                    # Direct query for ontology classes from the project's feature schema
                    feature_query = f"""
                    {{
                      project(where: {{ id: "{LABELBOX_PROJECT_ID}" }}) {{
                        ontology {{
                          featureSchema {{
                            specifications {{
                              ... on Classification {{
                                options {{
                                  value
                                  label
                                }}
                              }}
                            }}
                          }}
                        }}
                      }}
                    }}
                    """
                    
                    try:
                        feature_result = client.execute(feature_query)
                        feature_data = feature_result.get('project', {}).get('ontology', {}).get('featureSchema', {})
                        
                        # Extract options from specifications
                        class_values = []
                        for spec in feature_data.get('specifications', []):
                            for option in spec.get('options', []):
                                class_values.append(option.get('value'))
                        
                        print("\nEXTRACTED CLASS VALUES:")
                        print("-" * 50)
                        for value in class_values:
                            print(f"  - {value}")
                            
                        # Compare with COCO
                        coco_categories = check_coco_categories()
                        if coco_categories:
                            coco_names = [cat['name'] for cat in coco_categories]
                            
                            print("\nCATEGORY MATCHING:")
                            print("-" * 50)
                            for coco_name in coco_names:
                                if coco_name in class_values:
                                    print(f"  ✓ {coco_name} - Found in both COCO and Labelbox")
                                else:
                                    print(f"  ✗ {coco_name} - In COCO but NOT in Labelbox")
                            
                            for lb_class in class_values:
                                if lb_class not in coco_names:
                                    print(f"  ⚠ {lb_class} - In Labelbox but NOT in COCO")
                    except Exception as feature_error:
                        logging.error(f"Error querying features: {feature_error}")
            except Exception as schema_error:
                logging.error(f"Error querying ontology schema: {schema_error}")
                
    except Exception as e:
        logging.error(f"Error checking Labelbox schema: {e}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main() 