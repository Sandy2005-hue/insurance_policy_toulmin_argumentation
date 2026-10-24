import json
import os
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever


def _build(k, pool=10, rrf_const=60):
    """Build the dense, BM25, hybrid (interleave) and hybrid RRF search functions.

    Every search function returns up to k unique clauses. Each underlying retriever fetches a larger
    candidate pool so that fusion can choose among more than k results.
    """
    pool = max(pool, k)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "chroma_db")
    chunks_file = os.path.join(current_dir, "data", "policy_chunks.json")

    # 1. Dense retriever
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=db_path, embedding_function=embeddings)
    chroma_retriever = vector_store.as_retriever(search_kwargs={"k": pool})

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
    bm25_retriever.k = pool

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

    def hybrid_rrf_search(query):
        # Reciprocal Rank Fusion: score = sum over retrievers of 1 / (rrf_const + rank)
        scores, docs = {}, {}
        for hits in (bm25_retriever.invoke(query), chroma_retriever.invoke(query)):
            seen_in_list = set()
            rank = 0
            for doc in hits:
                doc_id = doc.metadata.get("id", "").split("#")[0]
                if not doc_id or doc_id in seen_in_list:
                    continue
                seen_in_list.add(doc_id)
                rank += 1
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_const + rank)
                docs.setdefault(doc_id, doc)
        best = sorted(scores, key=scores.get, reverse=True)[:k]
        return [docs[i] for i in best]

    return dense_search, bm25_search, hybrid_search, hybrid_rrf_search


def get_hybrid_retriever(k=4):
    """Used by the agent: returns the hybrid (interleave) search function."""
    return _build(k)[2]


def get_retrievers(k=4):
    """Used by the benchmark: all three search functions built from the same indexes."""
    dense, bm25, hybrid, hybrid_rrf = _build(k)
    return {"dense": dense, "bm25": bm25, "hybrid": hybrid, "hybrid_rrf": hybrid_rrf}