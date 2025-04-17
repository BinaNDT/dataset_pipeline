#!/bin/bash

# Building Damage Assessment Pipeline Environment Setup Script
# This script helps to set up the environment variables needed for the pipeline

echo "Building Damage Assessment Pipeline Environment Setup"
echo "===================================================="
echo

# Check if .env file already exists
if [ -f .env ]; then
    echo "A .env file already exists. Do you want to:"
    echo "1. Keep existing file"
    echo "2. Create a new file (will backup existing as .env.bak)"
    read -p "Enter choice [1-2]: " choice
    
    if [ "$choice" == "2" ]; then
        echo "Backing up existing .env to .env.bak"
        cp .env .env.bak
    else
        echo "Keeping existing .env file. Exiting."
        exit 0
    fi
fi

# Create new .env file
if [ "$choice" == "2" ] || [ ! -f .env ]; then
    echo "Creating new .env file..."
    
    # Get Labelbox credentials
    read -p "Enter your Labelbox API Key: " api_key
    read -p "Enter your Labelbox Project ID: " project_id
    
    # Debug mode
    read -p "Enable debug mode? [y/N]: " debug_choice
    if [[ $debug_choice =~ ^[Yy]$ ]]; then
        debug_mode="1"
    else
        debug_mode="0"
    fi
    
    # Write to .env file
    cat > .env << EOF
# Building Damage Assessment Pipeline Environment Variables
# Generated on $(date)

# Labelbox credentials
LABELBOX_API_KEY=${api_key}
LABELBOX_PROJECT_ID=${project_id}

# Debug mode (0 = disabled, 1 = enabled)
DEBUG_MODE=${debug_mode}
EOF
    
    # Set restrictive permissions on the .env file
    echo "Setting secure file permissions..."
    chmod 600 .env
    
    echo "Environment file created successfully!"
    echo "File permissions set to owner read/write only (600)"
    echo "To use these variables in your shell session, run:"
    echo "  source .env"
fi

# Remind about security
echo
echo "SECURITY REMINDER:"
echo "- Never commit your .env file to version control"
echo "- Keep your API keys secure and rotate them periodically"
echo "- The .env file is added to .gitignore to prevent accidental commits"
echo
echo "Environment setup complete!" 