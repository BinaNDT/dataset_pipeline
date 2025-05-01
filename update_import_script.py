#!/usr/bin/env python3
"""
Update import_test_annotations.sh to use visible test annotations
"""

import re
from pathlib import Path

# Original and new annotation paths
original_path = "outputs/predictions/test_annotations.ndjson"
updated_path = "outputs/predictions/visible_test_annotations.ndjson"

# Load the import script
script_path = Path("import_test_annotations.sh")
print(f"Updating import script: {script_path}")

with open(script_path, "r") as f:
    script_content = f.read()

# Replace the file path
updated_content = script_content.replace(
    f"annotations_file = Path('{original_path}')", 
    f"annotations_file = Path('{updated_path}')"
)

# Also update the import name to show it's using visible annotations
updated_content = re.sub(
    r"name=f'Test_Import_\{TIMESTAMP\}'",
    r"name=f'Visible_Test_Import_{TIMESTAMP}'",
    updated_content
)

# Save the updated script
updated_script_path = Path("import_visible_test.sh")
with open(updated_script_path, "w") as f:
    f.write(updated_content)

print(f"Created updated import script: {updated_script_path}")
print(f"Run it with: bash {updated_script_path}")

# Make the new script executable
import os
os.chmod(updated_script_path, 0o755)
print("Made the script executable.") 