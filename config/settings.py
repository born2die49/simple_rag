from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq

def get_embedding_model():
    """Configuration for document embedding model (Ollama gemma2:2b)
       Returns: 
           OllamaEmbeddings - Lightweight model for vector transformations"""
    return OllamaEmbeddings(model="gemma2:2b")

def get_answer_model(api_key: str):
    """Configuration for answer generation model (Groq gemma2-9b-it)
       Args:
           api_key: Groq API credentials
       Returns:
           ChatGroq - Powerful model for final response generation"""
    return ChatGroq(groq_api_key=api_key, model="gemma2-9b-it")