# Goolets Objection Handler

A Streamlit chat application that helps yacht charter sales teams handle customer objections by providing AI-powered solution matching.

## Features

- AI-Powered Objection Matching: Uses OpenAI embeddings for semantic similarity matching
- Chat Interface: Interactive chat interface for natural conversations
- Stage-Based Filtering: Filter objections by customer journey stage
- 190+ Yacht Charter Objections: Comprehensive database of real objections and solutions
- Multiple Solutions: Up to 8 solutions per objection

## Setup

1. **Clone the repository**
   ```bash
   git clone git@github.com:RangoDelic/ObjectionApp.git
   cd ObjectionApp
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Google Sheets API**
   
   Follow the detailed setup guide in `google_sheets_setup.md` to:
   - Create a Google Cloud Project
   - Enable Google Sheets API
   - Create a Service Account
   - Share your Google Sheet with the service account
   - Get your credentials

4. **Add your API keys and configuration**
   
   Edit `.streamlit/secrets.toml`:
   ```toml
   OPENAI_API_KEY = "your-openai-api-key-here"
   
   GOOGLE_SHEETS_CREDENTIALS = '''
   {
     "type": "service_account",
     "project_id": "your-project-id",
     "private_key_id": "your-private-key-id",
     "private_key": "-----BEGIN PRIVATE KEY-----\\nYour-private-key\\n-----END PRIVATE KEY-----\\n",
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

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## Deployment on Streamlit Cloud

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://share.streamlit.io)
3. Connect your GitHub repository
4. Add your `OPENAI_API_KEY` in the Streamlit Cloud secrets section
5. Deploy!

## Data Structure

The app uses a TSV file with the following structure:
- **Stage**: Customer journey stage (Research, Knows Goolets, Pre-booking)
- **Objection**: The customer objection/concern
- **Solution 1-8**: Up to 8 solutions for each objection

## Usage

1. Type any yacht charter objection or concern in the chat
2. The AI will find the most similar objection in the database
3. Get multiple proven solutions instantly
4. Use stage filtering to narrow down results

## Customer Journey Stages

- **Stage 1**: Customer is researching yacht charters
- **Stage 2**: Customer already knows Goolets and received a proposal
- **Stage 3**: Customer is close to booking

## Technology Stack

- **Streamlit**: Web application framework
- **OpenAI API**: Text embeddings for semantic matching
- **Pandas**: Data processing
- **Scikit-learn**: Cosine similarity calculations