from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

# Load vector store
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

# Load LLM
def load_llm():
    print("Loading LLM model...")
    model_name = "google/flan-t5-base"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        device="cpu"
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

# Build RAG chain
def build_rag_chain():
    vector_store = load_vector_store()
    llm = load_llm()
    
    prompt_template = """Use the following context to answer the question.
If you don't know the answer, say "I don't know".

Context: {context}

Question: {question}

Answer:"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt}
    )
    
    return rag_chain

# Test it
if __name__ == "__main__":
    print("Building RAG chain...")
    rag_chain = build_rag_chain()
    
    question = "What is this document about?"
    print(f"\nQuestion: {question}")
    
    result = rag_chain.invoke({"query": question})
    print(f"Answer: {result['result']}")