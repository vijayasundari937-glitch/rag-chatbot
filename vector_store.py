from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from document_loader import load_and_split_pdf

def create_vector_store(pdf_path):
    print("Loading and splitting PDF...")
    chunks = load_and_split_pdf(pdf_path)
    
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    print("Creating FAISS vector store...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Save it locally
    vector_store.save_local("faiss_index")
    print("Vector store saved!")
    
    return vector_store

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vector_store = FAISS.load_local(
        "faiss_index", 
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store

# Test it
if __name__ == "__main__":
    vector_store = create_vector_store("sample_docs/sample.pdf")
    
    # Test a search
    query = "what is this document about?"
    results = vector_store.similarity_search(query, k=3)
    
    print("\n--- Top 3 Relevant Chunks ---")
    for i, doc in enumerate(results):
        print(f"\nChunk {i+1}:")
        print(doc.page_content)