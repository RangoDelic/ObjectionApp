import streamlit as st
import pandas as pd
import openai
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
import gspread
from google.oauth2.service_account import Credentials
from typing import List, Dict, Tuple

# Configure Streamlit page
st.set_page_config(
    page_title="Goolets Objection Handler",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize OpenAI client
@st.cache_resource
def init_openai():
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found. Please add it to secrets or environment variables.")
        st.stop()
    return OpenAI(api_key=api_key)

client = init_openai()

# Google Sheets integration
@st.cache_resource
def init_google_sheets():
    """Initialize Google Sheets connection"""
    try:
        # Try to get credentials from Streamlit secrets
        if "GOOGLE_SHEETS_CREDENTIALS" in st.secrets:
            credentials_dict = json.loads(st.secrets["GOOGLE_SHEETS_CREDENTIALS"])
            credentials = Credentials.from_service_account_info(
                credentials_dict,
                scopes=['https://spreadsheets.google.com/feeds',
                       'https://www.googleapis.com/auth/drive']
            )
            return gspread.authorize(credentials)
        else:
            return None
    except Exception as e:
        st.warning(f"Google Sheets integration not configured: {e}")
        return None


def find_best_objection_match_with_openai(user_query: str, gc) -> Tuple[str, str, str, List[str]]:
    """Use OpenAI embeddings to find the most semantically similar objection"""
    try:
        # Get sheet data
        sheet_id = st.secrets["GOOGLE_SHEET_ID"]
        sheet_name = st.secrets.get("SHEET_NAME", "Sheet1")
        sheet = gc.open_by_key(sheet_id).worksheet(sheet_name)
        all_data = sheet.get_all_records()

        # Filter valid objections
        valid_objections = []
        for row in all_data:
            objection = row.get('Objection', '').strip()
            if objection:
                valid_objections.append(row)

        if not valid_objections:
            return "", "", "0.0", []

        # Get embedding for user query
        query_response = client.embeddings.create(
            model="text-embedding-3-small",
            input=user_query
        )
        query_embedding = np.array(query_response.data[0].embedding)

        # Get embeddings for all objections
        objection_texts = [row['Objection'] for row in valid_objections]

        # Process in batches to avoid token limits
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(objection_texts), batch_size):
            batch = objection_texts[i:i+batch_size]
            batch_response = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            batch_embeddings = [data.embedding for data in batch_response.data]
            all_embeddings.extend(batch_embeddings)

        # Convert to numpy array
        objection_embeddings = np.array(all_embeddings)

        # Calculate cosine similarities
        similarities = cosine_similarity([query_embedding], objection_embeddings)[0]

        # Find best match
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        best_match = valid_objections[best_match_idx]

        # Extract data from the matched row
        objection = best_match.get('Objection', '')
        stage = best_match.get('Stage', '')

        # Extract solutions
        solutions = []
        for i in range(1, 9):
            solution = best_match.get(f'Solution {i}', '')
            if solution and str(solution).strip():
                solutions.append(str(solution).strip())

        return objection, stage, f"{best_similarity:.3f}", solutions

    except Exception as e:
        st.error(f"Error finding objection match with OpenAI embeddings: {e}")
        return "", "", "0.0", []

def find_best_objection_match_simple(user_query: str, gc) -> Tuple[str, str, str, List[str]]:
    """Simple text matching fallback without API calls"""
    try:
        # Get sheet data
        sheet_id = st.secrets["GOOGLE_SHEET_ID"]
        sheet_name = st.secrets.get("SHEET_NAME", "Sheet1")
        sheet = gc.open_by_key(sheet_id).worksheet(sheet_name)
        all_data = sheet.get_all_records()

        query_lower = user_query.lower()
        best_match = None
        best_similarity = 0.0

        for row in all_data:
            objection = row.get('Objection', '')
            if not objection or not objection.strip():
                continue

            objection_lower = objection.lower()

            # Simple keyword overlap scoring
            query_words = set(query_lower.split())
            objection_words = set(objection_lower.split())

            if query_words and objection_words:
                overlap = len(query_words & objection_words)
                similarity = overlap / max(len(query_words), len(objection_words))

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = row

        if best_match:
            # Extract solutions
            solutions = []
            for i in range(1, 9):
                solution = best_match.get(f'Solution {i}', '')
                if solution and str(solution).strip():
                    solutions.append(str(solution).strip())

            return best_match.get('Objection', ''), best_match.get('Stage', ''), f"{best_similarity:.2f}", solutions
        else:
            return "", "", "0.0", []

    except Exception as e:
        st.error(f"Error in fallback matching: {e}")
        return "", "", "0.0", []


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "google_client" not in st.session_state:
    st.session_state.google_client = init_google_sheets()

# Sidebar
st.sidebar.title("Goolets Objection Handler")
st.sidebar.markdown("Ask about any yacht charter objection and get tailored solutions!")

# Check Google Sheets connection
if not st.session_state.google_client:
    st.sidebar.error("Google Sheets not configured")
else:
    st.sidebar.success("Connected to Google Sheets")

# Stage filter (static options since we search on-demand)
selected_stage = st.sidebar.selectbox(
    "Filter by Stage (Optional)",
    ["All Stages", "Research", "Knows Goolets", "Pre-booking"]
)

# Clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Refresh connection button
if st.sidebar.button("Refresh Connection"):
    st.session_state.google_client = init_google_sheets()
    st.rerun()

# Main chat interface
st.title("Goolets Objection Handler")
st.markdown("Ask me about any yacht charter objection, and I'll provide you with proven solutions!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What objection are you facing?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Finding the best solution..."):
            gc = st.session_state.google_client

            if not gc:
                response = "Google Sheets connection not available. Please check your configuration."
            else:
                # Try OpenAI matching first, fallback to simple matching
                try:
                    objection, stage, similarity, solutions = find_best_objection_match_with_openai(prompt, gc)
                except:
                    # If OpenAI fails, use simple text matching
                    objection, stage, similarity, solutions = find_best_objection_match_simple(prompt, gc)

                # Filter by stage if selected
                if selected_stage != "All Stages" and stage != selected_stage:
                    response = f"No objections found for the selected stage: {selected_stage}"
                elif objection:
                    # Format response
                    response = f"**Best Match:** {objection}\n\n"
                    response += f"**Stage:** {stage}\n\n"
                    response += f"**Similarity:** {similarity}\n\n"

                    if solutions:
                        response += "**Solutions:**\n\n"
                        for i, solution in enumerate(solutions, 1):
                            response += f"{i}. {solution}\n\n"
                    else:
                        response += "No solutions available for this objection."
                else:
                    response = "No matching objection found. Please try rephrasing your question."
        
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown("*Built for Goolets yacht charter objection handling*")