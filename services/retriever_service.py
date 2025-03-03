from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from chains.rag_chains import create_contextualize_q_prompt, create_qa_prompt
from langchain.chains.combine_documents import create_stuff_documents_chain

def initialize_retriever(answer_llm, retriever):
    """Create retrieval chain using answer generation LLM
       Args:
           answer_llm: Language model for query rewriting
           retriever: Vector store interface
       Returns:
           Context-aware retrieval chain"""
    contextualize_q_prompt = create_contextualize_q_prompt()
    history_aware_retriever = create_history_aware_retriever(
        answer_llm, retriever, contextualize_q_prompt
    )
    
    qa_prompt = create_qa_prompt()
    question_answer_chain = create_stuff_documents_chain(answer_llm, qa_prompt)
    
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)