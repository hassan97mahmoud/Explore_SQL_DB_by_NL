# App1f.py (Frontend)

# --- Imports ---
import streamlit as st
import requests
import pandas as pd
import os
from dotenv import load_dotenv

# --- Configuration Loading ---
load_dotenv()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")

# The URL of our running FastAPI backend
FASTAPI_BASE_URL = "http://localhost:8000"


# --- Helper Functions to Interact with FastAPI ---
def process_user_query(user_question: str):
    """Sends a question to the backend and gets the response."""
    try:
        response = requests.post(f"{FASTAPI_BASE_URL}/process-query", json={"user_question": user_question})
        # Raise an exception if the response is an error (e.g., 404, 500)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: Could not connect to the backend. Is it running? Details: {e}")
        return None

def get_health_check():
    """Gets the health status from the backend."""
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/health")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return {"status": "error", "detail": "Backend not reachable"}

def get_all_examples():
    """Gets all examples from the backend."""
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/get-all-examples")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching examples: {e}")
        return None


# --- Streamlit User Interface ---

st.set_page_config(layout="wide", page_title="NL to SQL Query Engine")
st.title("Natural Language to SQL Query Engine")

# Initialize session state variables
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "mode" not in st.session_state:
    st.session_state.mode = "User"

# Login UI
if not st.session_state.logged_in:
    st.sidebar.header("Login")
    password = st.sidebar.text_input("Enter Admin Password:", type="password")
    if st.sidebar.button("Login as Admin"):
        if password == ADMIN_PASSWORD:
            st.session_state.logged_in = True
            st.session_state.mode = "Admin"
            st.rerun()
        else:
            st.sidebar.error("Incorrect password.")
    if st.sidebar.button("Continue as User"):
        st.session_state.logged_in = True
        st.session_state.mode = "User"
        st.rerun()
else:
    st.sidebar.header(f"Mode: {st.session_state.mode}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.mode = "User"
        st.rerun()

    # User Mode UI
    if st.session_state.mode == "User":
        st.header("Ask Your Question")
        user_question = st.text_area("Enter your query:", height=100)
        if st.button("Get Answer"):
            if user_question:
                with st.spinner("Processing..."):
                    result = process_user_query(user_question)
                if result:
                    if result.get("nl_response"):
                        st.success("Answer:")
                        st.markdown(result["nl_response"])
                    else:
                        st.error(f"Error: {result.get('error_message', 'Unknown error')}")
            else:
                st.warning("Please enter a question.")

    # Admin Mode UI
    elif st.session_state.mode == "Admin":
        st.header("Admin Dashboard")
        admin_action = st.selectbox("Choose an action:", ["Test NL to SQL", "View All Examples", "Backend Health"])

        if admin_action == "Test NL to SQL":
            st.subheader("Test NL to SQL")
            nl_query_admin = st.text_area("NL Query:", height=100)
            if st.button("Process Query"):
                if nl_query_admin:
                    with st.spinner("Processing..."):
                        result = process_user_query(nl_query_admin)
                    if result:
                        st.json(result)

        elif admin_action == "View All Examples":
            st.subheader("View All Examples")
            if st.button("Fetch Examples"):
                with st.spinner("Fetching..."):
                    data = get_all_examples()
                if data and "points" in data:
                    df_data = [p['payload'] for p in data['points'] if p.get('payload')]
                    st.dataframe(pd.DataFrame(df_data))
                else:
                    st.error("Failed to fetch examples or no examples found.")

        elif admin_action == "Backend Health":
            st.subheader("Backend Health")
            if st.button("Check Health"):
                health_status = get_health_check()
                st.json(health_status)