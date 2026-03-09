import streamlit as st
import tempfile
import os
from document_loader import load_and_split_pdf
from vector_store import create_vector_store
from rag_pipeline import build_rag_chain

# Page config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG Chatbot")
st.subheader("Upload a PDF and chat with it!")

# Sidebar for PDF upload
with st.sidebar:
    st.header("📄 Upload Your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        if st.button("🚀 Process PDF"):
            with st.spinner("Processing PDF... please wait ⏳"):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                # Create vector store from uploaded PDF
                create_vector_store(tmp_path)

                # Reset chat history and reload chain
                st.session_state.messages = []
                st.session_state.rag_chain = build_rag_chain()

                os.unlink(tmp_path)
            st.success("✅ PDF processed! Ask me anything!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize RAG chain with default PDF
if "rag_chain" not in st.session_state:
    if os.path.exists("faiss_index"):
        with st.spinner("Loading AI model... ⏳"):
            st.session_state.rag_chain = build_rag_chain()
        st.success("✅ Model loaded! Ask me anything.")
    else:
        st.warning("👈 Please upload a PDF from the sidebar to get started!")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if "rag_chain" in st.session_state:
    if prompt := st.chat_input("Ask a question about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.rag_chain.invoke({"query": prompt})
                answer = result["result"]
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})