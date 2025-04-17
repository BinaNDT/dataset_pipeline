# Security Best Practices

This document outlines security best practices for the Building Damage Assessment Pipeline.

## Handling Credentials

### API Keys and Sensitive Data

The pipeline requires access to Labelbox services through API keys. Follow these guidelines to handle credentials securely:

1. **Never hardcode credentials in code files**
   - Do not embed API keys, tokens, or passwords directly in code
   - Do not commit credentials to version control

2. **Use environment variables**
   - Store sensitive data in environment variables
   - Use the `.env` file for local development (never commit this file)
   - For production, use secure environment variable management provided by your platform

3. **Use the provided tools**
   - The `setup_env.sh` script helps create a secure `.env` file
   - The project automatically loads variables from `.env` via python-dotenv

### Setting Up Environment Variables

#### Option 1: Using the setup script
```bash
./setup_env.sh
```
This interactive script will:
- Guide you through setting up your environment variables
- Create a `.env` file with proper permissions
- Backup any existing file for safety

#### Option 2: Manual configuration
1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file with your actual credentials:
   ```
   LABELBOX_API_KEY=your_actual_api_key
   LABELBOX_PROJECT_ID=your_actual_project_id
   ```

#### Option 3: Direct shell export
For temporary use or in CI/CD environments:
```bash
export LABELBOX_API_KEY="your_api_key"
export LABELBOX_PROJECT_ID="your_project_id"
```

## Configuration Security

1. **File Permissions**
   - Ensure `.env` file has restrictive permissions:
     ```bash
     chmod 600 .env
     ```
   - This makes the file readable only by the owner

2. **API Key Rotation**
   - Rotate API keys periodically
   - Immediately rotate if you suspect a key has been exposed

3. **Minimal Privilege**
   - Use API keys with the minimum required privileges
   - Create separate keys for development and production

## Secure Coding Practices

1. **Input Validation**
   - All user inputs should be validated before use
   - Use the `validate_inputs()` utility for consistent validation

2. **Error Handling**
   - Use the `log_error()` utility to log errors without exposing sensitive details
   - Never expose stack traces in production logs

3. **Dependency Security**
   - Regularly update dependencies to patch security vulnerabilities
   - Consider running dependency security scans

## Troubleshooting

### Common Security Issues

1. **API Authentication Failures**
   - Check that the `.env` file exists and contains valid credentials
   - Verify that the API key has the necessary permissions
   - Check if the API key has expired or been revoked

2. **File Permission Issues**
   - Ensure the `.env` file is readable by the application
   - Check ownership and permissions of config files

### Security Contacts

If you discover a security vulnerability, please report it responsibly by:

1. **Not** disclosing the issue publicly until it has been addressed
2. Contacting the maintainers directly via email
3. Providing clear steps to reproduce the issue 