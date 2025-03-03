import streamlit as st
from dotenv import load_dotenv

from config.settings import get_embedding_model, get_answer_model
from handlers.document_handlers import handle_pdf_upload
from services.vector_store import create_vector_store
from services.retriever_service import initialize_retriever
from chains.rag_chains import create_conversational_rag_chain
from utils.text_processing import get_text_splitter
from utils.session_manager import get_session_history, initialize_session_store

load_dotenv()

# Streamlit UI setup
st.title("Conversational RAG with pdf uploads and chat history")
st.write("Upload Pdf's and chat with their content")

def main():
    """Main entry point for the Streamlit application. 
       Handles UI setup, session management, and coordinates workflow between components.
       Orchestrates document processing and chat interactions."""
    # Input for Groq API key
    api_key = st.text_input("Enter your Groq API key", type="password")
    
    if api_key:        
        # Session management
        session_id = st.text_input("Session id", value="default_session")
        initialize_session_store()
        
        # File upload handling
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=False)
        
        if uploaded_file:
            # Process document
            documents = handle_pdf_upload(uploaded_file)
            text_splitter = get_text_splitter()
            splits = text_splitter.split_documents(documents)
            
            # Create vector store
            embedding_model = get_embedding_model()
            retriever = create_vector_store(splits, embedding_model)
            
            # Initialize RAG chain
            answer_model = get_answer_model(api_key)
            rag_chain = initialize_retriever(answer_model, retriever)            
            conversational_rag_chain = create_conversational_rag_chain(
                rag_chain,
                lambda session_id: get_session_history(st.session_state.store, session_id)
            )
            
            # Chat interface
            user_input = st.text_input("Your question:")
            if user_input:
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}}
                )
                st.write("Assistant:", response['answer'])

if __name__ == "__main__":
    main()