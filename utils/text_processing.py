from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_text_splitter():
    """Provides configured text splitting utility for document processing.
       Returns: RecursiveCharacterTextSplitter - Text splitter with fixed chunk parameters
       Note: Adjust chunk_size/chunk_overlap here for different document types"""
    return RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=500
    )