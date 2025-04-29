#!/usr/bin/env python3
"""
Test Labelbox Connection

A simple utility script to test connectivity to Labelbox API.
This helps diagnose connection issues before attempting full uploads.

Usage:
    python test_labelbox_connection.py
"""

import os
import sys
import time
import logging
import requests
import socket
import traceback
from pathlib import Path

# Setup path to import from config
sys.path.append(str(Path(__file__).parent))
from config import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def check_internet_connectivity():
    """Check basic internet connectivity"""
    try:
        logging.info("Testing general internet connectivity...")
        socket.create_connection(("8.8.8.8", 53), timeout=5)
        logging.info("✅ Internet connection is available")
        return True
    except OSError as e:
        logging.error(f"❌ No internet connection available: {str(e)}")
        return False

def check_labelbox_api():
    """Check if Labelbox API is reachable"""
    try:
        logging.info("Testing connection to Labelbox API...")
        start_time = time.time()
        
        # Try a different endpoint - main API URL
        response = requests.get("https://api.labelbox.com", timeout=10)
        elapsed = time.time() - start_time
        
        if response.status_code < 400:  # Any non-error response is fine
            logging.info(f"✅ Labelbox API is accessible (response time: {elapsed:.2f}s)")
            return True
        else:
            logging.error(f"❌ Labelbox API returned status code {response.status_code}")
            # Still proceed even if we get an error response
            logging.info("Continuing with authentication test anyway...")
            return True
    except Exception as e:
        logging.error(f"❌ Failed to connect to Labelbox API: {str(e)}")
        # Handle connection timeout
        if "timeout" in str(e).lower():
            logging.error("The connection timed out. This might be due to network restrictions.")
            logging.info("Check if your network allows connections to api.labelbox.com")
        return False

def check_labelbox_auth():
    """Check if Labelbox API key is valid"""
    if not LABELBOX_API_KEY:
        logging.error("❌ LABELBOX_API_KEY is not set")
        logging.info("Please set up your environment variables using setup_env.sh")
        return False
    
    try:
        import labelbox as lb
        
        logging.info("Testing Labelbox authentication...")
        start_time = time.time()
        client = lb.Client(api_key=LABELBOX_API_KEY)
        user = client.get_user()
        elapsed = time.time() - start_time
        
        logging.info(f"✅ Successfully authenticated with Labelbox (response time: {elapsed:.2f}s)")
        logging.info(f"✅ Connected as user: {user.email}")
        return True
    except Exception as e:
        logging.error(f"❌ Authentication with Labelbox failed: {str(e)}")
        if "Invalid API key" in str(e):
            logging.info("Please check your API key in the .env file")
        return False

def check_project_access():
    """Check if the configured project ID is accessible"""
    if not LABELBOX_PROJECT_ID:
        logging.error("❌ LABELBOX_PROJECT_ID is not set")
        logging.info("Please set up your environment variables using setup_env.sh")
        return False
    
    try:
        import labelbox as lb
        
        logging.info(f"Testing access to project ID: {LABELBOX_PROJECT_ID}")
        client = lb.Client(api_key=LABELBOX_API_KEY)
        project = client.get_project(LABELBOX_PROJECT_ID)
        
        logging.info(f"✅ Successfully accessed project: {project.name}")
        return True
    except Exception as e:
        logging.error(f"❌ Failed to access project: {str(e)}")
        if "Resource not found" in str(e):
            # List available projects
            logging.info("Available projects in your Labelbox account:")
            try:
                client = lb.Client(api_key=LABELBOX_API_KEY)
                projects = client.get_projects()
                for project in projects:
                    logging.info(f"  • {project.name} (ID: {project.uid})")
            except Exception:
                logging.error("Failed to list available projects")
        return False

def main():
    """Run all connectivity tests"""
    print("\n=== Labelbox Connectivity Test ===\n")
    
    internet_ok = check_internet_connectivity()
    if not internet_ok:
        logging.error("❌ No internet connection. Please check your network.")
        return 1
    
    # Try API connectivity, but continue anyway
    api_ok = check_labelbox_api()
    if not api_ok:
        logging.warning("⚠️ API check failed, but continuing with authentication test...")
        # We'll continue anyway, as the auth test is more important
    
    auth_ok = check_labelbox_auth()
    if not auth_ok:
        return 1
    
    project_ok = check_project_access()
    if not project_ok:
        return 1
    
    print("\n=== All Tests Passed ===\n")
    logging.info("✅ Your Labelbox connection is working correctly")
    logging.info("You should be able to use labelbox_importer.py successfully")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logging.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        logging.debug(traceback.format_exc())
        sys.exit(1) 