import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from config.settings import get_embedding_model
from handlers.document_handlers import handle_pdf_upload
from services.llm_service import initialize_answer_llm
from services.vector_store import create_vector_store
from services.retriever_service import initialize_retriever
from chains.rag_chains import create_conversational_rag_chain
from utils.text_processing import get_text_splitter
from utils.session_manager import initialize_session_store, get_session_history
from ui.components import display_chat_history, chat_input  # New UI components

load_dotenv()

def main():
    st.title("Conversational RAG with PDF Uploads")
    
    # Initialize session state
    initialize_session_store()
    
    # Sidebar components
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Groq API Key", type="password")
        session_id = st.text_input("Session ID", value="default_session")
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    # Main chat interface
    display_chat_history()
    
    # Handle file upload and processing
    if uploaded_file and api_key:
        process_document(uploaded_file, api_key, session_id)
    
    # Always show chat input (disabled until ready)
    user_input = chat_input(disabled=not (uploaded_file and api_key))
    if user_input:
        handle_user_input(user_input, session_id)

def process_document(uploaded_file, api_key, session_id):
    """Handle document processing with comprehensive validation"""
    try:
        if not uploaded_file:
            return

        # Reset state for new documents
        current_file = st.session_state.get('current_file')
        if current_file != uploaded_file.name:
            reset_document_state()
            st.session_state.current_file = uploaded_file.name
            st.session_state.chat_history = []

        # Validate document processing
        documents = handle_pdf_upload(uploaded_file)
        if not documents:
            raise ValueError("Failed to process PDF document")

        # Split with metadata preservation
        text_splitter = get_text_splitter()
        splits = text_splitter.split_documents(documents)
        if not splits or len(splits) == 0:
            raise ValueError("Document splitting failed - no text chunks created")

        # Create vector store
        embedding_model = get_embedding_model()
        retriever = create_vector_store(splits, embedding_model)

        # Initialize LLM and chains
        answer_llm = initialize_answer_llm(api_key)
        rag_chain = initialize_retriever(answer_llm, retriever)
        
        st.session_state.rag_chain = create_conversational_rag_chain(
            rag_chain,
            lambda session_id: get_session_history(st.session_state.store, session_id)
        )

    except Exception as e:
        st.error(f"Document processing failed: {str(e)}")
        reset_document_state()
        st.session_state.chat_history = []
        raise
        
def reset_document_state():
    """Clears document-related session state while preserving chat history"""
    keys_to_clear = ['rag_chain', 'current_file', 'store', 'chat_history']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

def handle_user_input(user_input, session_id):
    """Process user input and update chat history"""
    with st.spinner("Thinking..."):
        response = st.session_state.rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        
    # Update chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": response['answer']})
    st.rerun()

if __name__ == "__main__":
    main()