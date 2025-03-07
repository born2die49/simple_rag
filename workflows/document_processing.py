import streamlit as st

from chains.rag_chains import create_conversational_rag_chain
from config.settings import get_embedding_model
from handlers.document_handlers import handle_pdf_upload
from services.llm_service import initialize_answer_llm
from services.retriever_service import initialize_retriever
from services.pinecone_vector_store import create_pinecone_vector_store, get_pinecone_retriever
from utils.file_utils import compute_file_hash, load_processed_files, save_processed_files
from utils.session_manager import get_session_history
from utils.text_processing import determine_chunk_params, get_text_splitter


def process_document(uploaded_file, api_key, session_id):
    """Handle document processing with comprehensive validation"""
    try:
        if not uploaded_file:
            return
        
        file_hash = compute_file_hash(uploaded_file)
        processed_hashes = load_processed_files()
        
        if file_hash in processed_hashes:
            # File already processed, use existing Pinecone data
            st.session_state.file_processed = True
            embeddings = get_embedding_model()
            retriever = get_pinecone_retriever(embeddings)
            
        else:
            # New file: process and store hash, Reset state for new documents
            current_file = st.session_state.get('current_file')
            if current_file != uploaded_file.name:
                reset_document_state()
                st.session_state.current_file = uploaded_file.name
                st.session_state.chat_history = []

            # Validate document processing
            documents = handle_pdf_upload(uploaded_file)
            if not documents:
                raise ValueError("Failed to process PDF document")
            
            # Calculate average text length per page (heuristic for density)
            total_text_length = sum(len(document.page_content) for document in documents)
            avg_text_length = total_text_length / len(documents) if documents else 0

            # Determine chunk parameters based on density
            chunk_size, chunk_overlap = determine_chunk_params(avg_text_length)

            # Split with metadata preservation
            text_splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            splits = text_splitter.split_documents(documents)
            if not splits or len(splits) == 0:
                raise ValueError("Document splitting failed - no text chunks created")

            # Create Pinecone vector store (upserts new data)
            embedding_model = get_embedding_model()
            retriever = create_pinecone_vector_store(splits, embedding_model)
            processed_hashes.append(file_hash)
            save_processed_files(processed_hashes)

        # Initialize the conversational chain regardless of new/existing file
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