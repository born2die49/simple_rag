import streamlit as st
from chains.rag_chains import create_conversational_rag_chain
from services.llm_service import initialize_answer_llm
from services.retriever_service import initialize_retriever
from utils.session_manager import get_session_history
from utils.session_utils import reset_document_state


def initialize_conversation_chain(api_key, retriever, session_id):
    """Setup RAG chain with dependency injection"""
    answer_llm = initialize_answer_llm(api_key)
    rag_chain = initialize_retriever(answer_llm, retriever)
    st.session_state.rag_chain = create_conversational_rag_chain(
        rag_chain,
        lambda session_id: get_session_history(st.session_state.store, session_id)
    )

def handle_processing_error(e):
    """Centralized error handling"""
    st.error(f"Document processing failed: {str(e)}")
    reset_document_state()
    st.session_state.chat_history = []
    raise e