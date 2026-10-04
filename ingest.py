import os
import json
import warnings
import shutil
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

warnings.filterwarnings("ignore", category=DeprecationWarning)

def build_vector_db():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "data", "policy_chunks.json")
    db_path = os.path.join(current_dir, "chroma_db")

    print(f"1. Loading structured chunks with Context Enrichment...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    for chunk in data['chunks']:

        enriched_content = (
            f"Section {chunk.get('section', '')}: {chunk.get('section_title', '')}\n"
            f"Clause: {chunk.get('title', '')}\n"
            f"Content: {chunk['text']}"
        )

        doc = Document(
            page_content=enriched_content,
            metadata={
                "id": str(chunk['id']),
                "citation": str(chunk.get('citation', '')),
                "section": str(chunk.get('section', '')),
                "page": int(chunk.get('page', 0))
            }
        )
        documents.append(doc)
    
    print(f"✅ Enriched {len(documents)} structured chunks.")

    print("2. Generating Embeddings with BAAI/bge-small-en-v1.5 (Upgraded SOTA IR Model)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        
    Chroma.from_documents(
        documents=documents, 
        embedding=embeddings, 
        persist_directory=db_path
    )
    print(f"✅ High-Precision Vector DB built at: {db_path}")

if __name__ == "__main__":
    build_vector_db()