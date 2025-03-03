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
    """Process uploaded document and initialize RAG chain"""
    if 'rag_chain' not in st.session_state:
        documents = handle_pdf_upload(uploaded_file)
        text_splitter = get_text_splitter()
        splits = text_splitter.split_documents(documents)
        
        embedding_model = get_embedding_model()
        retriever = create_vector_store(splits, embedding_model)
        
        answer_llm = initialize_answer_llm(api_key)
        rag_chain = initialize_retriever(answer_llm, retriever)
        
        st.session_state.rag_chain = create_conversational_rag_chain(
            rag_chain,
            lambda session_id: get_session_history(st.session_state.store, session_id)
        )

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