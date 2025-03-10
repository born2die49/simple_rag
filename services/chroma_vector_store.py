from langchain_chroma import Chroma
from chromadb import EphemeralClient
import uuid

def get_chroma_retriever(embeddings):
    """Connect to existing Chroma vector store (ephemeral)"""
    return Chroma(
        client=EphemeralClient(),
        embedding_function=embeddings,
    ).as_retriever()

def create_chroma_vector_store(splits, embeddings):
    """Create vector store with guaranteed valid IDs"""
    # Generate robust IDs with multiple fallbacks
    ids = []
    for i, split in enumerate(splits):
        doc_id = (
            split.metadata.get("doc_uuid") or 
            split.metadata.get("source") or 
            str(uuid.uuid4())
        )
        ids.append(f"{doc_id}_chunk{i}")
    
    # Create collection with explicit validation
    return Chroma.from_documents(
        client=EphemeralClient(),
        documents=splits,
        embedding=embeddings,
        ids=ids,
        collection_name=f"coll_{uuid.uuid4().hex}",
    ).as_retriever()