import streamlit as st

def handle_user_input(user_input, session_id):
    """Process user input and update chat history"""
    with st.spinner("Thinking..."):
        response = st.session_state.rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        
    # Update chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": response['answer']})
    st.rerun()