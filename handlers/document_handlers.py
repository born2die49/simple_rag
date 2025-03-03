import os
from langchain.document_loaders import PyPDFLoader

def handle_pdf_upload(uploaded_file):
    """Processes uploaded PDF files: saves temporarily, loads documents, cleans up temp files.
       Args: uploaded_file (UploadedFile): Streamlit file object
       Returns: List[Document] - List of processed document objects
       Note: Add similar handlers here for other file types (DOCX, TXT, etc.)"""
    documents = []
    temppdf = "./temp.pdf"
    with open(temppdf, "wb") as file:
        file.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temppdf)
    docs = loader.load()
    documents.extend(docs)
    os.remove(temppdf)
    return documents