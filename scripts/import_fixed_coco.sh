#!/bin/bash
# Import fixed COCO file to Labelbox

# Run the importer with the fixed COCO file
python labelbox_importer.py \
  --source coco \
  --coco-file outputs/predictions/labelbox_coco.json \
  --dataset-name Fixed_Names_Import

echo "Import process complete!"
