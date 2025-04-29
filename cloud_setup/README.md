# Cloud Storage Setup for Labelbox Integration

This directory contains scripts and configuration for setting up Google Cloud Storage to work with Labelbox.

## Prerequisites

1. Install Google Cloud SDK:
```bash
curl https://sdk.cloud.google.com | bash
source ~/.bashrc
gcloud init
```

2. Configure your Labelbox credentials in `config.json`:
- Replace `YOUR_LABELBOX_PROJECT_ID` with your actual project ID
- Replace `YOUR_LABELBOX_API_KEY` with your actual API key

## Usage

1. Configure the settings in `config.json`:
   - Set your desired bucket name
   - Verify the dataset paths
   - Update Labelbox credentials

2. Run the upload script:
```bash
python upload_to_gcs.py
```

3. The script will:
   - Create a Google Cloud Storage bucket
   - Upload your dataset
   - Make the bucket publicly readable
   - Generate a file with all image URLs

4. After upload, you can use the generated `image_urls.txt` with your Labelbox import script.

## Cost Estimation

For a 20GB dataset:
- Storage: ~$0.40/month (at $0.02/GB/month)
- Network egress: ~$2.40/month (at $0.12/GB)
- Total: ~$2.80/month

## Troubleshooting

1. If upload fails:
   - Check the `upload.log` file for detailed error messages
   - Verify your Google Cloud credentials
   - Ensure you have sufficient permissions

2. If bucket creation fails:
   - Try a different bucket name (must be globally unique)
   - Verify your Google Cloud project is properly set up

3. If URLs are not accessible:
   - Verify the bucket is made public
   - Check the bucket's IAM permissions

## Security Notes

- The bucket is made publicly readable to work with Labelbox
- Consider using object lifecycle management for cost optimization
- Regularly monitor your Google Cloud usage and costs 