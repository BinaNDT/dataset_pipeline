import os
import json
import subprocess
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('upload.log'),
        logging.StreamHandler()
    ]
)

def load_config():
    """Load configuration from config.json"""
    with open('config.json', 'r') as f:
        return json.load(f)

def create_bucket(bucket_name, region):
    """Create a Google Cloud Storage bucket"""
    try:
        subprocess.run([
            'gsutil', 'mb',
            '-l', region,
            f'gs://{bucket_name}'
        ], check=True)
        logging.info(f"Created bucket: gs://{bucket_name}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to create bucket: {e}")
        return False

def upload_directory(local_path, bucket_name, remote_path):
    """Upload a directory to Google Cloud Storage"""
    try:
        subprocess.run([
            'gsutil', '-m', 'cp', '-r',
            local_path,
            f'gs://{bucket_name}/{remote_path}'
        ], check=True)
        logging.info(f"Uploaded {local_path} to gs://{bucket_name}/{remote_path}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to upload {local_path}: {e}")
        return False

def make_bucket_public(bucket_name):
    """Make the bucket publicly readable"""
    try:
        subprocess.run([
            'gsutil', 'iam', 'ch',
            'allUsers:objectViewer',
            f'gs://{bucket_name}'
        ], check=True)
        logging.info(f"Made bucket gs://{bucket_name} publicly readable")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to make bucket public: {e}")
        return False

def generate_urls_file(bucket_name):
    """Generate a file with all image URLs"""
    try:
        subprocess.run([
            'gsutil', 'ls', '-r',
            f'gs://{bucket_name}/**'
        ], stdout=open('image_urls.txt', 'w'))
        logging.info("Generated image_urls.txt")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to generate URLs file: {e}")
        return False

def main():
    config = load_config()
    bucket_name = config['bucket_name']
    region = config['region']
    
    # Create bucket
    if not create_bucket(bucket_name, region):
        return
    
    # Upload data
    for data_type, local_path in config['dataset_paths'].items():
        if os.path.exists(local_path):
            upload_directory(local_path, bucket_name, data_type)
    
    # Make bucket public
    make_bucket_public(bucket_name)
    
    # Generate URLs file
    generate_urls_file(bucket_name)
    
    logging.info("Upload process completed successfully")

if __name__ == "__main__":
    main() 