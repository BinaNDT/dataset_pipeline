#!/usr/bin/env python3
"""
Check if the labelbox module is installed correctly
"""

import sys
import subprocess

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

try:
    import labelbox
    print(f"\nLabelbox is installed!")
    print(f"Labelbox version: {labelbox.__version__ if hasattr(labelbox, '__version__') else 'unknown'}")
    print(f"Labelbox path: {labelbox.__file__}")
    
    print("\nLabelbox modules available:")
    for item in dir(labelbox):
        if not item.startswith('__'):
            print(f"  - {item}")
    
    # Check if specific components are available
    if hasattr(labelbox, 'Client'):
        print("\nLabelbox Client is available")
    else:
        print("\nLabelbox Client is NOT available")
    
    if hasattr(labelbox, 'LabelImport'):
        print("LabelImport is available")
    else:
        print("LabelImport is NOT available")
    
    if hasattr(labelbox, 'MALPredictionImport'):
        print("MALPredictionImport is available")
    else:
        print("MALPredictionImport is NOT available")
    
    if hasattr(labelbox, 'data'):
        print("The data module IS available")
    else:
        print("The data module is NOT available")
    
except ImportError:
    print("\nLabelbox is NOT installed in this Python environment!")
    
    # Try installing it
    print("\nAttempting to install labelbox...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "labelbox", "--user"])
        print("Installation successful. Please restart your script.")
    except subprocess.CalledProcessError:
        print("Installation failed. Please try installing manually with:")
        print("pip install --user labelbox")
        
except Exception as e:
    print(f"\nError while checking labelbox installation: {e}")

if __name__ == "__main__":
    pass 