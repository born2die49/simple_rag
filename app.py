import streamlit as st
from dotenv import load_dotenv
from handlers.interaction_handlers import handle_user_input
from utils.session_manager import initialize_session_store
from ui.components import display_chat_history, chat_input, display_sidebar
from workflows.document_processing import process_document  # New UI components

load_dotenv()

def main():
    st.title("Conversational RAG with PDF Uploads")
    
    # Initialize session state
    initialize_session_store()
    
    # Sidebar components
    api_key, session_id, uploaded_file = display_sidebar()
    
    # Main chat interface
    display_chat_history()
    
    # Handle file upload and processing
    if uploaded_file and api_key:
        process_document(uploaded_file, api_key, session_id)
    
    # Always show chat input (disabled until ready)
    user_input = chat_input(disabled=not (uploaded_file and api_key))
    if user_input:
        handle_user_input(user_input, session_id)

if __name__ == "__main__":
    main()