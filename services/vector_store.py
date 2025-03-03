from langchain_chroma import Chroma

def create_vector_store(splits, embeddings):
    """Creates and returns a vector store retriever from document splits.
       Args:
           splits (List[Document]): Text chunks for vectorization
           embeddings: Embeddings model for vector creation
       Returns: Retriever - Vector store interface for similarity searches"""
    vectorestore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="rag_collection",
        persist_directory="./chroma_db"
    )
    return vectorestore.as_retriever()