from handlers.document_handlers import handle_pdf_upload
from services.chroma_vector_store import create_chroma_vector_store
from services.pinecone_vector_store import create_pinecone_vector_store
from utils.file_utils import compute_file_hash, load_processed_files, save_processed_files
from utils.text_processing import determine_chunk_params, get_text_splitter


def is_file_processed(uploaded_file):
    """Check if file has been processed before using hash"""
    file_hash = compute_file_hash(uploaded_file)
    processed_hashes = load_processed_files()
    return file_hash in processed_hashes, file_hash, processed_hashes

def process_new_file(uploaded_file, embeddings,vector_store_type):
    """Handle new file processing steps"""
    documents = handle_pdf_upload(uploaded_file)
    if not documents:
        raise ValueError("Failed to process PDF document")
    
    # Calculate chunk parameters
    total_text = sum(len(d.page_content) for d in documents)
    avg_length = total_text / len(documents) if documents else 0
    chunk_size, chunk_overlap = determine_chunk_params(avg_length)
    
    # Split and validate
    text_splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(documents)
    if not splits:
        raise ValueError("Document splitting failed")
    
    # Create vector store
    if vector_store_type == "Chroma (Local)":
        return create_chroma_vector_store(splits, embeddings)
    else:
        return create_pinecone_vector_store(splits, embeddings)

def update_processed_files(file_hash, processed_hashes):
    """Save new file hash to persistent storage"""
    processed_hashes.append(file_hash)
    save_processed_files(processed_hashes)