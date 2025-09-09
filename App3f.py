# --- Imports ---
import streamlit as st
import requests
import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv
import ast

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
    if not result:
        st.error("Sorry, I encountered a connection error and could not process your request.")
        return "Sorry, a connection error occurred."
    if result.get("error_message"):
        st.error(f"An unexpected error occurred: {result['error_message']}")
        return f"Error: {result['error_message']}"
    
    nl_response = result.get("nl_response")
    if nl_response:
        st.markdown(f"**{nl_response}**")
    else:
        st.warning("I couldn't generate a natural language answer, but here is the raw data.")
        nl_response = "Here is the data I found."
        
    query_result_str = result.get("query_result")
    if query_result_str:
        try:
            data = ast.literal_eval(query_result_str)
            if isinstance(data, list) and data and (len(data) > 1 or len(data[0]) > 1):
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
        except:
            st.code(query_result_str, language=None)
            
    generated_sql = result.get("generated_sql")
    if generated_sql:
        with st.expander("View Generated SQL Query"):
            st.code(generated_sql, language="sql")
    return nl_response

def render_login_ui():
    """Displays the initial UI for selecting User or Admin mode."""
    st.header("Welcome!")

    # MODIFICATION: Use local file paths from the 'assets' folder
    sql_icon_path = "assets/SQL.png"
    nl_icon_path = "assets/NL.png"

    # Add a check to ensure the files exist before trying to display them
    if os.path.exists(sql_icon_path) and os.path.exists(nl_icon_path):
        col_main_1, col_main_2, col_main_3 = st.columns([1, 2, 1])
        with col_main_2:
            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.image(sql_icon_path, width=150)
            with img_col2:
                st.image(nl_icon_path, width=150)
    else:
        st.warning("Could not find image files in the 'assets' folder. Please ensure 'NL.png' and 'SQL.png' are present.")

    st.write("Please select your role to continue.")

    col_btn_1, col_btn_2, col_btn_3 = st.columns([1, 1.5, 1])
    with col_btn_2:
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

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], dict) and message["role"] == "assistant":
                 render_assistant_response(message["content"])
            else:
                bubble_class = "user-question-bubble" if message["role"] == "user" else "assistant-answer-bubble"
                st.markdown(f'<div class="{bubble_class}">{message["content"]}</div>', unsafe_allow_html=True)

    if prompt := st.chat_input("Ask a question about your data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(f'<div class="user-question-bubble">{prompt}</div>', unsafe_allow_html=True)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = process_user_query(prompt)
                render_assistant_response(result)
        
        st.session_state.messages.append({"role": "assistant", "content": result})

def render_admin_ui():
    """Displays the admin dashboard."""
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
st.markdown('<div class="fixed-title"><h1>Explore and Chat with Your SQL DB by Natural Language</h1></div>', unsafe_allow_html=True)

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