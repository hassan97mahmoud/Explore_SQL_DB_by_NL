# App2f.py (Frontend - Professional Chat UI Version)

# --- Imports ---
import streamlit as st
import requests
import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv
import ast  # ast is used to safely evaluate the string representation of the database result

# --- Page Configuration ---
st.set_page_config(
    layout="centered",
    page_title="NL to SQL Query Engine",
    page_icon="🤖"
)

# --- Custom CSS for Professional UI ---
st.markdown("""
    <style>
        .main .block-container {
            max-width: 800px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .fixed-title {
            position: fixed;
            top: 15px;
            left: 0;
            width: 100%;
            background: #0e1117;
            padding: 1rem 0;
            text-align: center;
            z-index: 999;
            border-bottom: 1px solid #262730;
        }
        .stApp {
            padding-top: 7rem;
        }
        .user-question-bubble {
            background-color: #2b313e;
            padding: 10px 15px;
            border-radius: 18px;
            margin-bottom: 5px;
            display: inline-block;
            max-width: 90%;
            color: #ffffff;
            font-size: 1.05em;
        }
        .assistant-answer-bubble {
            background-color: #1e1f24;
            padding: 10px 15px;
            border-radius: 18px;
            margin-bottom: 5px;
            display: inline-block;
            max-width: 90%;
            color: #d1d5db;
            border: 1px solid #3b3f4a;
        }
        .st-expander {
            border: 1px solid #3b3f4a !important;
            border-radius: 10px !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- Configuration Loading ---
load_dotenv(find_dotenv())
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")
FASTAPI_BASE_URL = "http://localhost:8000"

# --- Helper Functions to Interact with FastAPI ---
def process_user_query(user_question: str):
    try:
        response = requests.post(f"{FASTAPI_BASE_URL}/process-query", json={"user_question": user_question})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error_message": f"Connection error: Could not connect to the backend. Is it running? Details: {e}"}

# (Other helper functions remain the same)
def get_health_check():
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/health")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return {"status": "error", "detail": "Backend not reachable"}

def get_all_examples():
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/get-all-examples")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching examples: {e}")
        return None

# --- UI Rendering Functions ---

def render_assistant_response(result):
    """Renders the assistant's response in a professional, structured format."""
    
    # 1. Handle errors first
    if not result:
        st.error("Sorry, I encountered a connection error and could not process your request.")
        return "Sorry, a connection error occurred."

    if result.get("error_message"):
        st.error(f"An unexpected error occurred: {result['error_message']}")
        return f"Error: {result['error_message']}"
    
    # 2. Display the natural language response
    nl_response = result.get("nl_response")
    if nl_response:
        st.markdown(f"**{nl_response}**")
    else:
        st.warning("I couldn't generate a natural language answer, but here is the raw data.")
        nl_response = "Here is the data I found." # Fallback for chat history

    # 3. Display the data in a table if it exists and is complex enough
    query_result_str = result.get("query_result")
    if query_result_str:
        try:
            # Safely evaluate the string to turn it into a Python list of tuples
            data = ast.literal_eval(query_result_str)
            
            # Only show a table if there's data and it's not a single simple value
            if isinstance(data, list) and data and (len(data) > 1 or len(data[0]) > 1):
                # We don't have column headers from the backend, so we let pandas create default ones
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
        except (ValueError, SyntaxError):
            # If the result is not a list (e.g., just a string), show it as text
            st.code(query_result_str, language=None)
        except Exception as e:
            st.warning(f"Could not display data as a table. Raw result: {query_result_str}")

    # 4. Show the generated SQL in a collapsible expander
    generated_sql = result.get("generated_sql")
    if generated_sql:
        with st.expander("View Generated SQL Query"):
            st.code(generated_sql, language="sql")

    return nl_response

def render_login_ui():
    """Displays the initial UI for selecting User or Admin mode."""
    st.header("Welcome!")
    st.write("Please select your role to continue.")

    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        if st.button("Continue as User", use_container_width=True, type="primary"):
            st.session_state.app_mode = "user_chat"
            st.rerun()

        with st.expander("Login as Admin"):
            with st.form("admin_login_form"):
                password = st.text_input("Enter Admin Password:", type="password")
                submitted = st.form_submit_button("Login")
                if submitted:
                    if password == ADMIN_PASSWORD:
                        st.session_state.app_mode = "admin_dashboard"
                        st.rerun()
                    else:
                        st.error("Incorrect password.")

def render_chat_ui():
    """Displays the main chat interface for the user."""
    st.header("Chat with your Database")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you explore the database today?"}]

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # For assistant messages, we might have more complex data stored
            if isinstance(message["content"], dict) and message["role"] == "assistant":
                 render_assistant_response(message["content"])
            else:
                # Regular user messages or simple assistant startup message
                bubble_class = "user-question-bubble" if message["role"] == "user" else "assistant-answer-bubble"
                st.markdown(f'<div class="{bubble_class}">{message["content"]}</div>', unsafe_allow_html=True)

    # Accept new user input
    if prompt := st.chat_input("Ask a question about your data..."):
        # Add and display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(f'<div class="user-question-bubble">{prompt}</div>', unsafe_allow_html=True)

        # Process and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = process_user_query(prompt)
                # The render function will display the rich output
                render_assistant_response(result)
        
        # Add the full result dictionary to history for re-rendering
        st.session_state.messages.append({"role": "assistant", "content": result})

def render_admin_ui():
    # (Admin UI remains the same)
    st.header("Admin Dashboard")
    admin_action = st.selectbox("Choose an action:", ["Test NL to SQL", "View All Examples", "Backend Health"])
    if admin_action == "Test NL to SQL":
        st.subheader("Test NL to SQL")
        nl_query_admin = st.text_area("Enter your query:", height=100)
        if st.button("Process Query"):
            if nl_query_admin:
                with st.spinner("Processing..."):
                    result = process_user_query(nl_query_admin)
                if result: st.json(result)
    elif admin_action == "View All Examples":
        st.subheader("View All Examples")
        if st.button("Fetch Examples"):
            with st.spinner("Fetching..."): data = get_all_examples()
            if data and "points" in data:
                df_data = [p['payload'] for p in data['points'] if p.get('payload')]
                st.dataframe(pd.DataFrame(df_data))
            else: st.error("Failed to fetch examples or no examples found.")
    elif admin_action == "Backend Health":
        st.subheader("Backend Health")
        if st.button("Check Health"):
            health_status = get_health_check()
            st.json(health_status)

# --- Main Application Logic ---
st.markdown('<div class="fixed-title"><h1>Natural Language to SQL Query Engine</h1></div>', unsafe_allow_html=True)

if "app_mode" not in st.session_state:
    st.session_state.app_mode = "login"

if st.session_state.app_mode != "login":
    if st.sidebar.button("Back to Main Menu"):
        for key in list(st.session_state.keys()):
            if key != 'app_mode':
                del st.session_state[key]
        st.session_state.app_mode = "login"
        st.rerun()

if st.session_state.app_mode == "login":
    render_login_ui()
elif st.session_state.app_mode == "user_chat":
    render_chat_ui()
elif st.session_state.app_mode == "admin_dashboard":
    render_admin_ui()