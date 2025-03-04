import os
import tempfile
import uuid
from langchain.document_loaders import PyPDFLoader

def handle_pdf_upload(uploaded_file):
    """Process PDF with guaranteed metadata and content validation"""
    try:
        # Existing file handling
        documents = []
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name
            
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        
        # Validate loaded content
        if not docs or len(docs) == 0:
            raise ValueError("PDF contains no readable content")
            
        # Add robust metadata
        doc_uuid = str(uuid.uuid4())
        for doc in docs:
            doc.metadata.update({
                "doc_uuid": doc_uuid,
                "source": uploaded_file.name,
                "page": doc.metadata.get("page", 0)
            })
            
        return docs
    finally:
        if temp_path:
            os.unlink(temp_path)