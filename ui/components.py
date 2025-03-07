import os
import streamlit as st

def display_chat_history():
    """Displays chat messages from history"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

def chat_input(disabled=False):
    """Displays chat input box"""
    return st.chat_input("Your question:", disabled=disabled, key="user_input")

def display_sidebar():
    with st.sidebar:
        st.header("Configuration")
        # API key input with env fallback (value not shown by default)
        api_key = st.text_input(
            "Groq API Key (Optional)", 
            value='',  # Do NOT show default value 
            type="password",
            help="Leave blank to use environment variable. Hidden for security"
        )
        if not api_key:
            api_key = os.getenv('GROQ_API_KEY', '')
        session_id = st.text_input("Session ID", value="default_session")
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        return api_key, session_id, uploaded_file