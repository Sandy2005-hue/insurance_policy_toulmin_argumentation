import json
import os
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

def get_hybrid_retriever(k=4):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "chroma_db")
    chunks_file = os.path.join(current_dir, "data", "policy_chunks.json")

    # 1. Dense Retriever
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=db_path, embedding_function=embeddings)
    chroma_retriever = vector_store.as_retriever(search_kwargs={"k": k})

    # 2. Sparse Retriever
    with open(chunks_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    bm25_docs = [
        Document(
            page_content=f"{c.get('section_title', '')} {c.get('title', '')} {c['text']}", 
            metadata={"id": str(c['id']), "citation": str(c.get('citation', ''))}
        ) for c in data['chunks']
    ]
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = k

    def hybrid_search(query):
        sparse_hits = bm25_retriever.invoke(query)
        dense_hits = chroma_retriever.invoke(query)

        merged_docs = []
        seen_ids = set()
        for pair in zip(sparse_hits, dense_hits):
            for doc in pair:
                doc_id = doc.metadata.get('id', '').split('#')[0]
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    merged_docs.append(doc)
        return merged_docs[:k]

    return hybrid_search