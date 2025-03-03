from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory

def create_contextualize_q_prompt():
    """Constructs the prompt template for historical context-aware query rewriting.
       Returns: ChatPromptTemplate - Prompt template for question contextualization"""
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question, "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do not answer the question, "
        "just reformulate it if needed otherwise return as it is."
    )
    return ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

def create_qa_prompt():
    """Builds the answer generation prompt template with system instructions.
       Returns: ChatPromptTemplate - Prompt template for final answer generation"""
    system_prompt = (
        "You are an assistant for question answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n{context}"
    )
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

def create_question_answer_chain(llm, qa_prompt):
    """Creates the document-based QA processing chain.
       Args:
           llm: Language model for answer generation
           qa_prompt: Configured prompt template
       Returns: Chain - Document processing chain for final answers"""
    return create_stuff_documents_chain(llm, qa_prompt)

def create_conversational_rag_chain(rag_chain, get_session_history_func):
    """Wraps the RAG chain with session history management capabilities.
       Args:
           rag_chain: Configured retrieval chain
           get_session_history_func: Session history provider function
       Returns: RunnableWithMessageHistory - Full conversational chain with memory"""
    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history_func,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )