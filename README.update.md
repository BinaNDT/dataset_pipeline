# Labelbox Annotation Pipeline Fixes

## Overview of Changes

The Labelbox annotation pipeline had issues with properly linking annotations to data rows in Labelbox, resulting in validation errors and missing annotations. The following fixes have been implemented:

### 1. Improved Data Row ID Retrieval

- Added a comprehensive `get_data_row_ids()` method that properly retrieves all data rows from the Labelbox project
- Implemented proper pagination handling to ensure all data rows are retrieved
- Created robust mappings between filenames and data row IDs, including:
  - Mapping with and without file extensions
  - Handling various URL formats
  - Utilizing external IDs when available

### 2. Fixed Annotation Creation and Linking

- Modified `simple_labelbox_import.py` to use data row IDs (UIDs) instead of global keys
- Changed the `ImageData` constructor to use `uid` parameter instead of `global_key`
- Added better error handling for missing data row IDs
- Improved polygon point validation to prevent invalid polygons

### 3. Enhanced Error Handling

- Added proper exception handling throughout the pipeline
- Implemented detailed logging for debugging
- Added fallback mechanisms when primary upload methods fail

### 4. Testing and Verification

- Created a new `test_labelbox_connection.py` script to verify:
  - API connectivity
  - Project access
  - Data row ID retrieval
  - Annotation creation and linking

## Using the Fixed Pipeline

To use the fixed annotation pipeline:

1. First run the test script to verify connectivity and data row retrieval:
   ```
   python dataset_pipeline/test_labelbox_connection.py
   ```

2. Import annotations using the improved importer:
   ```
   python dataset_pipeline/simple_labelbox_import.py --source coco
   ```

3. For difficult cases, use the matching script:
   ```
   python dataset_pipeline/fix_annotations_matching.py
   ```

## Technical Details

The core issue was that annotations were being created with `global_key` references instead of direct data row `uid` references. Labelbox requires annotations to be linked directly to data row IDs for proper validation. The key fix was changing:

```python
# Old, problematic approach
label = Label(
    data=ImageData(global_key=global_key),  # This relies on Labelbox to match by filename
    annotations=lb_annotations
)

# New, fixed approach
label = Label(
    data=ImageData(uid=data_row_id),  # Direct reference to data row ID
    annotations=lb_annotations
)
```

This ensures that each annotation is correctly associated with its corresponding data row in Labelbox.
