import streamlit as st
from config.settings import get_embedding_model
from services.pinecone_vector_store import get_pinecone_retriever
from utils.conversation_utils import handle_processing_error, initialize_conversation_chain
from utils.document_utils import is_file_processed, process_new_file, update_processed_files
from utils.session_utils import reset_document_state


def process_document(uploaded_file, api_key, session_id):
    """Handle document processing with reduced complexity"""
    try:
        with st.spinner("Processing PDF... This may take a few moments"):
            if not uploaded_file:
                return
            
            # Check processing status
            is_processed, file_hash, processed_hashes = is_file_processed(uploaded_file)
            
            if is_processed:
                # Reuse existing embeddings
                st.session_state.file_processed = True
                embeddings = get_embedding_model()
                retriever = get_pinecone_retriever(embeddings)
                st.success("Using cached embeddings for this file ")
                
            else:
                # Process new file
                current_file = st.session_state.get('current_file')
                if current_file != uploaded_file.name:
                    reset_document_state()
                    st.session_state.current_file = uploaded_file.name
                    st.session_state.chat_history = []
                
                # Core processing steps
                embeddings = get_embedding_model()
                retriever = process_new_file(uploaded_file, embeddings)
                update_processed_files(file_hash, processed_hashes)
            
            # Initialize conversational chain
            initialize_conversation_chain(api_key, retriever, session_id)
            
    except Exception as e:
        handle_processing_error(e)