import streamlit as st
from rag_pipeline import build_rag_chain
import os

# Page config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG Chatbot")
st.subheader("Chat with your PDF documents!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize RAG chain
if "rag_chain" not in st.session_state:
    with st.spinner("Loading AI model... please wait ⏳"):
        st.session_state.rag_chain = build_rag_chain()
    st.success("✅ Model loaded! Ask me anything about your document.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your document..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.rag_chain.invoke({"query": prompt})
            answer = result["result"]
            st.markdown(answer)

    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})