from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_pdf(pdf_path):
    print(f"Loading PDF: {pdf_path}")
    
    # Load the PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print(f"Total pages loaded: {len(documents)}")
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # each chunk = 500 characters
        chunk_overlap=50      # overlap between chunks = 50 characters
    )
    
    chunks = text_splitter.split_documents(documents)
    
    print(f"Total chunks created: {len(chunks)}")
    return chunks

# Test it
if __name__ == "__main__":
    chunks = load_and_split_pdf("sample_docs/sample.pdf")
    
    print("\n--- Sample Chunk ---")
    print(chunks[0].page_content)