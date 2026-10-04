import os
import warnings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


warnings.filterwarnings("ignore", category=DeprecationWarning)

def build_vector_db():

    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(current_dir, "data", "A_PLUS_HEALTH_INSURANCE.pdf")
    db_path = os.path.join(current_dir, "chroma_db")

    print(f"1. Looking for Insurance PDF at: {pdf_path}")
    

    if not os.path.exists(pdf_path):
        print(f"❌ FATAL ERROR: Cannot find the PDF.")
        print(f"Fix this: Make sure the file exists exactly at {pdf_path}")
        return


    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print("✅ PDF Loaded Successfully.")

    print("2. Chunking Document...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks.")

    print("3. Generating Embeddings & Building Vector DB... (This might take a minute)")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    

    Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=db_path
    )
    print(f"✅ Vector DB successfully built at {db_path}")

if __name__ == "__main__":
    build_vector_db()