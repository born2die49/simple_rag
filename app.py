import streamlit as st
from dotenv import load_dotenv
from config import settings
from handlers.interaction_handlers import handle_user_input
from utils.session_manager import initialize_session_store
from ui.components import display_chat_history, chat_input, display_sidebar
from utils.sidebar_utils import display_vector_store_selection
from workflows.document_processing import process_document

load_dotenv()

def main():
    st.title("Chatbot with PDF Uploads")
    
    # Initialize session state
    initialize_session_store()
    
    # Sidebar components
    api_key, session_id, uploaded_file = display_sidebar()
    
    # Main chat interface
    display_chat_history()
    
    vector_store_type = display_vector_store_selection()
    
    # Handle file upload and processing
    if uploaded_file and api_key:
        if uploaded_file.size > settings.MAX_UPLOAD_SIZE_MB * 1024**2:
            st.error(f"File too large! Maximum allowed: {settings.MAX_UPLOAD_SIZE_MB}MB")
            uploaded_file = None  # Prevent processing
    current_file_name = uploaded_file.name if uploaded_file else None
    
    # Check if a new file is uploaded
    if current_file_name != st.session_state.previous_file_name:
        # Reset processing flag for new file
        st.session_state.file_processed = False
        st.session_state.previous_file_name = current_file_name  # Update stored filename
    # Process the file only if not already processed
    if not st.session_state.file_processed:
        process_document(uploaded_file, api_key, session_id, vector_store_type)
        st.session_state.file_processed = True  # Mark as processed
            
    # Always show chat input (disabled until ready)
    user_input = chat_input(disabled=not (uploaded_file and api_key))
    if user_input:
        handle_user_input(user_input, session_id)

if __name__ == "__main__":
    main()