from config.settings import get_answer_model

def initialize_answer_llm(api_key: str):
    """Initialize and return the Groq LLM for answer generation
       Args: 
           api_key: Groq cloud API credentials
       Returns: 
           Configured ChatGroq instance"""
    return get_answer_model(api_key)