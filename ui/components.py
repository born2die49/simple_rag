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