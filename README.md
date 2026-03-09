# 🤖 RAG Chatbot - Chat with Your PDFs

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload any PDF and ask questions about it using AI.

## 🚀 Features
- 📄 Upload any PDF document
- 💬 Chat with your document using natural language
- 🔍 Semantic search using FAISS vector store
- 🤖 Powered by HuggingFace models (100% free, no API key needed)
- 🖥️ Clean Streamlit web interface

## 🛠️ Tech Stack
- **LangChain** - RAG pipeline framework
- **HuggingFace** - Embeddings & LLM (flan-t5-base)
- **FAISS** - Vector database for semantic search
- **Streamlit** - Web interface
- **Python 3.11**

## ⚙️ Installation

1. Clone the repository
```bash
   git clone https://github.com/vijayasundari937-glitch/rag-chatbot.git
   cd rag-chatbot
```

2. Create virtual environment
```bash
   py -3.11 -m venv venv
   venv\Scripts\activate
```

3. Install dependencies
```bash
   pip install streamlit==1.32.0 pypdf python-dotenv faiss-cpu==1.7.4 langchain==0.2.16 langchain-community==0.2.16 langchain-core==0.2.38 langchain-huggingface==0.0.3 langchain-text-splitters==0.2.4 sentence-transformers==2.7.0 transformers==4.40.2 torch==2.2.2
```

4. Run the app
```bash
   streamlit run app.py
```

## 📖 How It Works
1. Upload a PDF using the sidebar
2. The app splits it into chunks and creates embeddings
3. Embeddings are stored in a FAISS vector database
4. When you ask a question, it finds the most relevant chunks
5. The LLM generates an answer based on those chunks

## 🧠 Concepts Covered
- Retrieval-Augmented Generation (RAG)
- Text chunking and embeddings
- Vector databases and similarity search
- LLM integration
- Building AI-powered web apps