import streamlit as st


def reset_document_state():
    """Clears document-related session state while preserving chat history"""
    keys_to_clear = ['rag_chain', 'current_file', 'store', 'chat_history']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]