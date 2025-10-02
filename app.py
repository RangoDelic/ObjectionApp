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
import datetime
import uuid

# Analytics tracking via Google Sheets
def track_analytics_event(event_type: str, event_data: dict = None):
    """Track analytics events by logging to Analytics tab in the same Google Sheet"""
    if st.session_state.get("google_client") is None:
        return  # Skip analytics if no Google Sheets connection

    try:
        # Generate session ID if not exists
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())[:8]

        # Use the SAME Google Sheet as objections data
        gc = st.session_state.google_client
        sheet_id = st.secrets["GOOGLE_SHEET_ID"]  # Same sheet as your objections
        spreadsheet = gc.open_by_key(sheet_id)

        # Get or create Analytics worksheet in the same spreadsheet
        try:
            analytics_sheet = spreadsheet.worksheet("Analytics")
        except:
            # Create Analytics tab with proper formatting for hundreds of entries
            analytics_sheet = spreadsheet.add_worksheet(
                title="Analytics",
                rows=2000,  # Space for many entries
                cols=8      # Well-organized columns
            )

            # Set up clear headers
            headers = [
                "Date", "Time", "Session_ID", "Event",
                "Details", "Query_Preview", "Stage", "Result"
            ]
            analytics_sheet.append_row(headers)

            # Format header row - bold, background color, freeze
            analytics_sheet.format("A1:H1", {
                "backgroundColor": {"red": 0.2, "green": 0.6, "blue": 0.8},
                "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
                "horizontalAlignment": "CENTER"
            })

            # Freeze header row
            analytics_sheet.freeze(rows=1)

            # Set column widths for readability
            analytics_sheet.update_dimension_range("A:A", "COLUMNS", {"pixelSize": 100})  # Date
            analytics_sheet.update_dimension_range("B:B", "COLUMNS", {"pixelSize": 80})   # Time
            analytics_sheet.update_dimension_range("C:C", "COLUMNS", {"pixelSize": 90})   # Session
            analytics_sheet.update_dimension_range("D:D", "COLUMNS", {"pixelSize": 150})  # Event
            analytics_sheet.update_dimension_range("E:E", "COLUMNS", {"pixelSize": 200})  # Details
            analytics_sheet.update_dimension_range("F:F", "COLUMNS", {"pixelSize": 250})  # Query
            analytics_sheet.update_dimension_range("G:G", "COLUMNS", {"pixelSize": 100})  # Stage
            analytics_sheet.update_dimension_range("H:H", "COLUMNS", {"pixelSize": 120})  # Result

        # Prepare clean, readable analytics data
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        # Clean event type for display
        event_display = event_type.replace("_", " ").title()

        # Extract key details for easy reading
        details = ""
        query_preview = ""
        stage = ""
        result = ""

        if event_data:
            # Query details
            if "query_length" in event_data:
                details = f"Length: {event_data['query_length']} chars"
            if "query_preview" in event_data:
                query_preview = event_data["query_preview"]

            # Stage info
            stage = event_data.get("stage", event_data.get("selected_stage", ""))

            # Result indicators
            if event_type == "objection_match_found":
                result = f"✅ Found ({event_data.get('similarity', 'N/A')})"
            elif event_type == "no_objection_match":
                result = "❌ No Match"
            elif event_type == "openai_matching_used":
                result = "✅ OpenAI"
            elif event_type == "openai_fallback_to_simple":
                result = "⚠️ Fallback"
            elif event_type == "client_response_generated":
                result = "✅ Generated"
            elif event_type in ["clear_chat_clicked", "refresh_connection_clicked", "stage_filter_used"]:
                result = "✅ Action"

        analytics_row = [
            date_str,
            time_str,
            st.session_state.session_id,
            event_display,
            details,
            query_preview,
            stage,
            result
        ]

        # Add the row to Analytics tab
        analytics_sheet.append_row(analytics_row)

    except Exception as e:
        # Silently fail if analytics logging doesn't work
        pass

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

def generate_client_response(user_objection: str, matched_objection: str, solutions: List[str], output_format: str = "Email") -> str:
    """Generate a client-ready response based on the guidance solutions and desired output format"""
    if not solutions:
        return ""

    try:
        solutions_text = "\n".join([f"- {solution}" for solution in solutions])

        # Format-specific instructions
        format_instructions = {
            "Email": """- Write as a professional email response
- Use proper email etiquette with greeting and closing
- Be detailed and comprehensive
- Maintain formal yet friendly tone
- Include all relevant details and benefits""",
            "Instagram Post": """- Write as an engaging Instagram post or caption
- Use short paragraphs and line breaks for readability
- Include a compelling hook at the start
- Can use 1-2 relevant emojis if appropriate
- Keep it conversational and engaging
- Limit to 150-200 words
- Add a call-to-action at the end""",
            "Facebook Post": """- Write as an engaging Facebook post
- Use a conversational, friendly tone
- Break into readable paragraphs
- Can use emojis if appropriate
- Include engaging opening and call-to-action
- Limit to 200-250 words
- Make it shareable and relatable""",
            "Blog Post": """- Write as a section of a blog post
- Use a professional yet conversational tone
- Include clear structure with potential subheadings
- Be more detailed and informative
- Provide context and background where helpful
- Can be 300-400 words
- Focus on educating and informing the reader"""
        }

        prompt = f"""You are a professional yacht charter sales expert. A client has raised this objection: "{user_objection}"

The closest matching objection from our database is: "{matched_objection}"

Here are the internal guidance solutions for handling this objection:
{solutions_text}

Based on this guidance, write a response formatted for {output_format}. The response should:
- Address their specific concern professionally
- Be ready to copy-paste directly
- Sound natural and conversational
- Maintain Goolets' professional but friendly tone
- IMPORTANT: Write the response in English, regardless of the language of the client's objection

FORMAT-SPECIFIC REQUIREMENTS for {output_format}:
{format_instructions.get(output_format, format_instructions["Email"])}

Write the response as if you're the yacht charter expert speaking directly to the client:"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        st.error(f"Error generating client response: {e}")
        return ""


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "google_client" not in st.session_state:
    st.session_state.google_client = init_google_sheets()
if "output_format" not in st.session_state:
    st.session_state.output_format = None

# Track app initialization
if "analytics_initialized" not in st.session_state:
    track_analytics_event("app_initialized", {
        "google_sheets_connected": st.session_state.google_client is not None
    })
    st.session_state.analytics_initialized = True

# Sidebar
st.sidebar.image("1758712112554blob.jpg", width=200)
st.sidebar.title("Goolets Objection Handler")

# Check Google Sheets connection
if not st.session_state.google_client:
    st.sidebar.error("Google Sheets not configured")
else:
    st.sidebar.success("Connected to Google Sheets")

# Output format selector (REQUIRED)
st.sidebar.markdown("### Select Output Format")
output_format = st.sidebar.radio(
    "Response Format",
    ["Email", "Instagram Post", "Facebook Post", "Blog Post"],
    index=None,
    key="output_format_selector",
    help="Select the format for your response before asking a question"
)

# Update session state with selected format
if output_format:
    st.session_state.output_format = output_format

# Stage filter (static options since we search on-demand)
selected_stage = st.sidebar.selectbox(
    "Filter by Stage (Optional)",
    ["All Stages", "Research", "Knows Goolets", "Pre-booking"]
)

# Track stage filter usage
if selected_stage != "All Stages":
    track_analytics_event("stage_filter_used", {"stage": selected_stage})

# Clear chat button
if st.sidebar.button("Clear Chat"):
    track_analytics_event("clear_chat_clicked")
    st.session_state.messages = []
    st.rerun()

# Refresh connection button
if st.sidebar.button("Refresh Connection"):
    track_analytics_event("refresh_connection_clicked")
    st.session_state.google_client = init_google_sheets()
    st.rerun()

# Main chat interface
st.title("Goolets Objection Handler")
st.markdown("Ask me about any yacht charter objection, and I'll provide you with proven solutions!")

# Check if format is selected
if not st.session_state.output_format:
    st.warning("⚠️ Please select an output format in the sidebar before asking a question.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input - disabled if no format selected
chat_disabled = not st.session_state.output_format
if prompt := st.chat_input("What objection are you facing?", disabled=chat_disabled):
    # Track user query
    track_analytics_event("user_query_submitted", {
        "query_length": len(prompt),
        "query_word_count": len(prompt.split()),
        "selected_stage": selected_stage,
        "output_format": st.session_state.output_format,
        "query_preview": prompt[:50] + "..." if len(prompt) > 50 else prompt
    })

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
                track_analytics_event("google_sheets_error")
            else:
                # Try OpenAI matching first, fallback to simple matching
                try:
                    objection, stage, similarity, solutions = find_best_objection_match_with_openai(prompt, gc)
                    track_analytics_event("openai_matching_used")
                except Exception as e:
                    # If OpenAI fails, use simple text matching
                    track_analytics_event("openai_fallback_to_simple", {"error_type": str(type(e).__name__)})
                    objection, stage, similarity, solutions = find_best_objection_match_simple(prompt, gc)

                # Filter by stage if selected
                if selected_stage != "All Stages" and stage != selected_stage:
                    response = f"No objections found for the selected stage: {selected_stage}"
                    track_analytics_event("no_match_stage_filter", {"selected_stage": selected_stage})
                elif objection:
                    # Track successful match
                    track_analytics_event("objection_match_found", {
                        "matched_objection": objection[:50] + "..." if len(objection) > 50 else objection,
                        "stage": stage,
                        "similarity": similarity,
                        "num_solutions": len(solutions),
                        "output_format": st.session_state.output_format
                    })

                    # Generate client-ready response with selected format
                    client_response = generate_client_response(prompt, objection, solutions, st.session_state.output_format)

                    # Format response
                    response = f"**Best Match:** {objection}\n\n"
                    response += f"**Stage:** {stage}\n\n"

                    if client_response:
                        response += f"**{st.session_state.output_format} Response (Ready to Copy/Paste):**\n\n"
                        response += f"```\n{client_response}\n```\n\n"
                        track_analytics_event("client_response_generated", {
                            "has_client_response": True,
                            "output_format": st.session_state.output_format
                        })

                    if solutions:
                        response += "**Internal Guidance:**\n\n"
                        for i, solution in enumerate(solutions, 1):
                            response += f"{i}. {solution}\n\n"
                    else:
                        response += "No solutions available for this objection."
                else:
                    response = "No matching objection found. Please try rephrasing your question."
                    track_analytics_event("no_objection_match", {"query_length": len(prompt)})
        
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown("*Built for Goolets yacht charter objection handling*")