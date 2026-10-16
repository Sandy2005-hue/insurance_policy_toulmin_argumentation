import json
import os
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever


def _build(k):
    """Build the dense, BM25 and hybrid search functions (all return up to k unique clauses)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "chroma_db")
    chunks_file = os.path.join(current_dir, "data", "policy_chunks.json")

    # 1. Dense retriever
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=db_path, embedding_function=embeddings)
    chroma_retriever = vector_store.as_retriever(search_kwargs={"k": k})

    # 2. Sparse retriever
    with open(chunks_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    bm25_docs = [
        Document(
            page_content=f"{c.get('section_title', '')} {c.get('title', '')} {c['text']}",
            metadata={"id": str(c["id"]), "citation": str(c.get("citation", ""))},
        )
        for c in data["chunks"]
    ]
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = k

    def _unique(docs):
        out, seen = [], set()
        for doc in docs:
            doc_id = doc.metadata.get("id", "").split("#")[0]
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                out.append(doc)
        return out

    def dense_search(query):
        return _unique(chroma_retriever.invoke(query))[:k]

    def bm25_search(query):
        return _unique(bm25_retriever.invoke(query))[:k]

    def hybrid_search(query):
        # BM25 and dense results alternate (rank interleaving), duplicates removed by clause ID
        sparse_hits = bm25_retriever.invoke(query)
        dense_hits = chroma_retriever.invoke(query)

        merged_docs = []
        seen_ids = set()
        for pair in zip(sparse_hits, dense_hits):
            for doc in pair:
                doc_id = doc.metadata.get("id", "").split("#")[0]
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    merged_docs.append(doc)
        return merged_docs[:k]

    return dense_search, bm25_search, hybrid_search


def get_hybrid_retriever(k=4):
    """Used by the agent: returns the hybrid search function."""
    return _build(k)[2]


def get_retrievers(k=4):
    """Used by the benchmark: all three search functions built from the same indexes."""
    dense, bm25, hybrid = _build(k)
    return {"dense": dense, "bm25": bm25, "hybrid": hybrid}