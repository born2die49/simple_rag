import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_text_splitter(chunk_size=1500, chunk_overlap=200):
    """
    Create a text splitter with dynamic parameters.
    Args:
        chunk_size (int): Maximum chunk size (default 1500 tokens).
        chunk_overlap (int): Overlap between chunks (default 200 tokens).
    Returns:
        RecursiveCharacterTextSplitter: Configured text splitter.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],  # Prioritize paragraphs, sentences, words
        add_start_index=True,
        strip_whitespace=False,
        keep_separator=True,
        length_function=len,
        is_separator_regex=False
    )
    
def clean_text(text):
    """
    Clean text by replacing newlines, removing extra spaces, and stripping special characters.
    """
    # 1. Replace newlines with spaces
    text = re.sub(r'[\n\r]+', ' ', text)  # Replace all newlines with a single space 
    # 2. Remove extra spaces (including tabs, multiple spaces)
    text = re.sub(r'\s+', ' ', text)      # Replace sequences of whitespace with a single space 
    # 3. Remove special characters (keep letters, numbers, spaces, and common punctuation)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)  # Remove non-alphanumeric characters except basic punctuation 
    # Convert to lowercase
    text = text.lower()  
    # Remove page numbers
    text = re.sub(r'page \d+ of \d+', '', text)  
    
    # Strip leading/trailing whitespace
    return text.strip()

def determine_chunk_params(avg_text_length_per_page):
    """
    Determine chunk size and overlap based on document density.
    Args:
        avg_text_length_per_page (int): Average text length (characters) per page.
    Returns:
        chunk_size (int): Dynamic chunk size.
        chunk_overlap (int): Dynamic overlap.
    """
    if avg_text_length_per_page > 1000:  # Heuristic: Denser documents have more text per page
        chunk_size = 500  # Smaller chunks for dense documents
        chunk_overlap = 50  # Less overlap for smaller chunks
    else:
        chunk_size = 1500  # Larger chunks for less dense documents
        chunk_overlap = 200  # Moderate overlap for readability
    return chunk_size, chunk_overlap