import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

def initialize_session_store():
    """Initializes the session state storage for chat histories if not present.
       Side Effect: Modifies Streamlit session state with empty store dictionary"""
    if 'store' not in st.session_state:
        st.session_state.store = {}
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def get_session_history(store, session_id: str) -> BaseChatMessageHistory:
    """Retrieves or creates chat history for a given session ID.
       Args:
           store (dict): Session state storage dictionary
           session_id (str): Unique session identifier
       Returns: BaseChatMessageHistory - Persistent chat history for the session"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]