import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def build_vector_db():
    print("1. Loading Insurance PDF...")

    loader = PyPDFLoader("data/A_PLUS_HEALTH_INSURANCE.pdf")
    documents = loader.load()

    print("2. Chunking Document...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("3. Generating Embeddings & Building Vector DB...")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    

    Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    print("Vector DB successfully built at ./chroma_db")

if __name__ == "__main__":
    build_vector_db()