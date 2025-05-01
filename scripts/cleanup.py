#!/usr/bin/env python3
"""
Cleanup script to remove old/excess model checkpoint files
"""

import os
import glob
from pathlib import Path
import shutil
import argparse

def get_size_str(size_bytes):
    """Convert bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"

def cleanup_checkpoints(models_dir, keep_latest=1, dry_run=False):
    """
    Clean up checkpoint files, keeping only the specified number of latest ones
    
    Args:
        models_dir: Directory containing model checkpoints
        keep_latest: Number of latest checkpoints to keep
        dry_run: If True, just print what would be done without actually deleting
    """
    models_dir = Path(models_dir)
    if not models_dir.exists():
        print(f"Directory not found: {models_dir}")
        return
        
    # Get all checkpoint files
    checkpoint_files = sorted(
        models_dir.glob("checkpoint_epoch_*.pt"), 
        key=lambda x: int(x.stem.split('_')[-1])
    )
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {models_dir}")
        return
        
    # Calculate total size
    total_size = sum(f.stat().st_size for f in checkpoint_files)
    print(f"Found {len(checkpoint_files)} checkpoint files, " 
          f"total size: {get_size_str(total_size)}")
    
    # Keep the latest N files
    files_to_keep = checkpoint_files[-keep_latest:] if keep_latest > 0 else []
    files_to_delete = [f for f in checkpoint_files if f not in files_to_keep]
    
    size_to_delete = sum(f.stat().st_size for f in files_to_delete)
    size_to_keep = sum(f.stat().st_size for f in files_to_keep)
    
    print(f"Keeping {len(files_to_keep)} files ({get_size_str(size_to_keep)})")
    for f in files_to_keep:
        print(f"  - {f.name} ({get_size_str(f.stat().st_size)})")
        
    print(f"Will delete {len(files_to_delete)} files ({get_size_str(size_to_delete)})")
    
    if not dry_run:
        for f in files_to_delete:
            print(f"  - Deleting {f.name} ({get_size_str(f.stat().st_size)})")
            f.unlink()
        print(f"Deleted {len(files_to_delete)} files, freed {get_size_str(size_to_delete)}")
    else:
        print("Dry run - no files were actually deleted")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up model checkpoint files")
    parser.add_argument("--models-dir", type=str, default="outputs/models",
                        help="Directory containing model checkpoints")
    parser.add_argument("--keep", type=int, default=1,
                        help="Number of latest checkpoints to keep")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without actually deleting files")
    
    args = parser.parse_args()
    cleanup_checkpoints(args.models_dir, args.keep, args.dry_run) 