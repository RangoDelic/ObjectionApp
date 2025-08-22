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

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_objection_data():
    """Load objection data from Google Sheets"""
    gc = init_google_sheets()
    
    # Check if Google Sheets is properly configured
    if not gc:
        st.error("Google Sheets integration not configured. Please add GOOGLE_SHEETS_CREDENTIALS to your secrets.")
        st.stop()
    
    if "GOOGLE_SHEET_ID" not in st.secrets:
        st.error("Google Sheet ID not found. Please add GOOGLE_SHEET_ID to your secrets.")
        st.stop()
    
    try:
        sheet_id = st.secrets["GOOGLE_SHEET_ID"]
        sheet_name = st.secrets.get("SHEET_NAME", "Sheet1")
        
        sheet = gc.open_by_key(sheet_id).worksheet(sheet_name)
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        
        # Clean the data
        df = df.dropna(subset=['Objection'])
        
        if df.empty:
            st.error("No valid objection data found in Google Sheets. Please check your sheet structure.")
            st.stop()
        
        st.sidebar.success("Data loaded from Google Sheets")
        return df
        
    except Exception as e:
        st.error(f"Failed to load data from Google Sheets: {e}")
        st.info("Please check your Google Sheets configuration and ensure the sheet is shared with your service account.")
        st.stop()

def find_best_objection_match_with_openai(user_query: str, df: pd.DataFrame) -> Tuple[int, float]:
    """Use OpenAI to directly find the best matching objection without pre-computed embeddings"""
    try:
        # Create a prompt with all objections for OpenAI to match against
        objections_text = "\n".join([f"{i}: {obj}" for i, obj in enumerate(df['Objection'].tolist())])
        
        prompt = f"""You are helping match a customer objection to the most similar objection from a database.

User's objection: "{user_query}"

Database objections:
{objections_text}

Please respond with only the number (index) of the most similar objection from the database. Consider semantic meaning, not just exact word matches.

Response format: Just the number (e.g., "5")"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0
        )
        
        match_idx = int(response.choices[0].message.content.strip())
        
        # Calculate a rough similarity score (since we're using GPT matching, we'll estimate)
        similarity_score = 0.85  # Assume good match since GPT chose it
        
        return match_idx, similarity_score
        
    except Exception as e:
        st.error(f"Error finding objection match with OpenAI: {e}")
        # Fallback to simple text matching
        return find_best_objection_match_simple(user_query, df)

def find_best_objection_match_simple(user_query: str, df: pd.DataFrame) -> Tuple[int, float]:
    """Simple text matching fallback without API calls"""
    query_lower = user_query.lower()
    best_match_idx = 0
    best_similarity = 0.0
    
    for i, objection in enumerate(df['Objection'].tolist()):
        objection_lower = objection.lower()
        
        # Simple keyword overlap scoring
        query_words = set(query_lower.split())
        objection_words = set(objection_lower.split())
        
        if query_words and objection_words:
            overlap = len(query_words & objection_words)
            similarity = overlap / max(len(query_words), len(objection_words))
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = i
    
    return best_match_idx, best_similarity

def format_solutions(row: pd.Series) -> List[str]:
    """Extract and format solutions from a DataFrame row"""
    solutions = []
    for i in range(1, 9):  # Solution 1 to Solution 8
        solution_col = f"Solution {i}"
        if solution_col in row.index and pd.notna(row[solution_col]) and str(row[solution_col]).strip():
            solutions.append(str(row[solution_col]).strip())
    return solutions

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "objection_data" not in st.session_state:
    st.session_state.objection_data = load_objection_data()

# Sidebar
st.sidebar.title("Goolets Objection Handler")
st.sidebar.markdown("Ask about any yacht charter objection and get tailored solutions!")

# Display data statistics
df = st.session_state.objection_data
st.sidebar.markdown("### Data Overview")
st.sidebar.metric("Total Objections", len(df))
st.sidebar.metric("Customer Journey Stages", df['Stage'].nunique())

# Stage filter
stages = df['Stage'].unique()
selected_stage = st.sidebar.selectbox(
    "Filter by Stage (Optional)",
    ["All Stages"] + list(stages)
)

# Clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Refresh data button (for Google Sheets)
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
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
            # Filter data by stage if selected
            filtered_df = df if selected_stage == "All Stages" else df[df['Stage'] == selected_stage]
            
            if filtered_df.empty:
                response = "No objections found for the selected stage."
            else:
                # Find best matching objection using OpenAI or simple matching
                working_df = filtered_df if selected_stage != "All Stages" else df
                
                # Try OpenAI matching first, fallback to simple matching
                try:
                    best_idx, similarity = find_best_objection_match_with_openai(prompt, working_df)
                    matched_row = working_df.iloc[best_idx]
                except:
                    # If OpenAI fails, use simple text matching
                    best_idx, similarity = find_best_objection_match_simple(prompt, working_df)
                    matched_row = working_df.iloc[best_idx]
                
                # Format response
                objection = matched_row['Objection']
                stage = matched_row['Stage']
                solutions = format_solutions(matched_row)
                
                response = f"**Best Match:** {objection}\n\n"
                response += f"**Stage:** {stage}\n\n"
                response += f"**Similarity:** {similarity:.2%}\n\n"
                
                if solutions:
                    response += "**Solutions:**\n\n"
                    for i, solution in enumerate(solutions, 1):
                        response += f"{i}. {solution}\n\n"
                else:
                    response += "No solutions available for this objection."
        
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown("*Built for Goolets yacht charter objection handling*")