import os
import uuid
from pinecone import Pinecone as pcc
from langchain_pinecone import PineconeVectorStore

def create_pinecone_vector_store(splits, embeddings):
    # Initialize Pinecone client (configure API key and environment)
    pc = pcc(
        api_key=os.getenv("PINECONE_API_KEY"),  # Required
    )
    
    index_name = "simple-rag"  # Must already exist in Pinecone dashboard
    
    # Generate custom IDs (prioritizing existing metadata fields)
    ids = []
    for i, split in enumerate(splits):
        doc_id = (
            split.metadata.get("doc_uuid") or 
            split.metadata.get("source") or 
            str(uuid.uuid4())
        )
        ids.append(f"{doc_id}_chunk{i}")
        
        # Add the text to metadata
        split.metadata["text"] = split.page_content  # Critical fix
    
    # Extract texts and metadata from splits
    texts = [split.page_content for split in splits]
    metadatas = [split.metadata for split in splits]
    
    # Get embeddings (ensure they match Pinecone index's dimension)
    embeddings_list = embeddings.embed_documents(texts)
    
    # Upsert vectors with custom IDs using Pinecone client
    pinecone_index = pc.Index(index_name)
    pinecone_index.upsert(
        vectors=[
            (
                ids[i],
                embeddings_list[i],
                metadatas[i]  # Optional metadata
            ) for i in range(len(texts))
        ]
    )
    
    # Initialize Pinecone vector store wrapper
    vector_store = PineconeVectorStore(
        pinecone_index,  # Pinecone Index instance
        embedding=embeddings,
        text_key="text"  # Explicitly set text key
    )
    
    return vector_store.as_retriever()