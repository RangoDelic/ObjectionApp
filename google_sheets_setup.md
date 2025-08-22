# Google Sheets API Setup Guide

Follow these steps to enable Google Sheets integration for your Objection Handler app.

## Step 1: Create a Google Cloud Project

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Note your project ID

## Step 2: Enable Google Sheets API

1. In the Google Cloud Console, go to **APIs & Services** > **Library**
2. Search for "Google Sheets API"
3. Click on it and press **Enable**
4. Also enable "Google Drive API" (needed for accessing sheets)

## Step 3: Create a Service Account

1. Go to **APIs & Services** > **Credentials**
2. Click **Create Credentials** > **Service Account**
3. Fill in the service account details:
   - Name: `goolets-objection-handler`
   - Description: `Service account for Objection Handler app`
4. Click **Create and Continue**
5. Skip role assignment for now (click **Continue**)
6. Click **Done**

## Step 4: Generate Service Account Key

1. Click on the service account you just created
2. Go to the **Keys** tab
3. Click **Add Key** > **Create new key**
4. Select **JSON** format
5. Download the JSON file (keep it secure!)

## Step 5: Share Your Google Sheet

1. Open your Google Sheet with the objection data
2. Click the **Share** button
3. Add the service account email (from the JSON file) as an editor
   - The email looks like: `goolets-objection-handler@your-project.iam.gserviceaccount.com`
4. Copy your Google Sheet ID from the URL
   - URL format: `https://docs.google.com/spreadsheets/d/SHEET_ID/edit`

## Step 6: Configure Streamlit Secrets

### For Local Development:
Create or edit `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "your-openai-api-key"

GOOGLE_SHEETS_CREDENTIALS = '''
{
  "type": "service_account",
  "project_id": "your-project-id",
  "private_key_id": "your-private-key-id",
  "private_key": "-----BEGIN PRIVATE KEY-----\\nYour-private-key-content\\n-----END PRIVATE KEY-----\\n",
  "client_email": "your-service-account@your-project.iam.gserviceaccount.com",
  "client_id": "your-client-id",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "your-cert-url"
}
'''

GOOGLE_SHEET_ID = "your-google-sheet-id"
SHEET_NAME = "Sheet1"
```

### For Streamlit Cloud Deployment:
1. Go to your Streamlit Cloud app settings
2. Go to **Secrets** section
3. Add the same content as above

## Step 7: Test the Integration

1. Run your app locally: `streamlit run app.py`
2. Check the sidebar - it should show "Data loaded from Google Sheets" if successful
3. Use the "Refresh Data" button to reload data from Google Sheets

## Troubleshooting

**Common Issues:**

1. **Permission denied**: Make sure you shared the sheet with the service account email
2. **Invalid credentials**: Double-check the JSON format in secrets.toml
3. **Sheet not found**: Verify the GOOGLE_SHEET_ID and SHEET_NAME
4. **API not enabled**: Ensure both Google Sheets API and Google Drive API are enabled

**Fallback Behavior:**
- If Google Sheets fails, the app will automatically use the local TSV file
- Check the sidebar for status messages about data source