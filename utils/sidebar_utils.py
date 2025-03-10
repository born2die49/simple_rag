import streamlit as st


def display_vector_store_selection():
    """Add vector store selection to sidebar"""
    return st.sidebar.selectbox(
        "Vector Store",
        ["Chroma (Local)", "Pinecone (Remote)"],
        index=0,
        help="Chroma uses in-memory storage; Pinecone persists data across sessions."
    )