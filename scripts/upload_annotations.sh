#!/bin/bash
# This script helps run the Labelbox annotation upload process with the fixed scripts

echo "====== Labelbox Annotation Upload Process ======"
echo "This script will help you upload annotations to Labelbox using the fixed pipeline"
echo

# Check if API key and project ID are set
if [ -z "$LABELBOX_API_KEY" ] || [ -z "$LABELBOX_PROJECT_ID" ]; then
    echo "LABELBOX_API_KEY or LABELBOX_PROJECT_ID environment variables are not set."
    echo "Please set them first using:"
    echo "  export LABELBOX_API_KEY=your_api_key"
    echo "  export LABELBOX_PROJECT_ID=your_project_id"
    echo
    echo "You can also source the setup_env.sh script if available."
    exit 1
fi

# Step 1: Check connection
echo "Step 1: Testing connection to Labelbox..."
python test_labelbox_connection.py
if [ $? -ne 0 ]; then
    echo "Connection test failed. Please fix the issues before continuing."
    exit 1
fi
echo

# Step 2: Choose the annotation source
echo "Step 2: Choose the annotation source:"
echo "1. COCO format (predictions_coco.json)"
echo "2. Predictions format (predictions.json)"
echo "3. Use the matching script for complex cases"
echo
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo "Running the importer with COCO source..."
        python simple_labelbox_import.py --source coco
        ;;
    2)
        echo "Running the importer with predictions source..."
        python simple_labelbox_import.py --source predictions
        ;;
    3)
        echo "Running the matching script for complex cases..."
        python fix_annotations_matching.py
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo
echo "Annotation upload process completed."
echo "Check the logs for details on the upload status."
echo
echo "If you encounter any issues, try using debug mode:"
echo "  python simple_labelbox_import.py --source coco --debug"
echo "Or check the README.update.md file for more information." 