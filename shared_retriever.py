"""Shared policy retrievers for the health-insurance claim assistant.

Retrieval strategies available to the live agent:
    interleave (default): alternate BM25 and dense results
    rrf:                  reciprocal-rank fusion of BM25 and dense results
    dense:                dense vector retrieval only
    bm25:                 BM25 keyword retrieval only

Set RETRIEVAL_STRATEGY in the process environment to choose the live strategy.
For example, in Bash:
    RETRIEVAL_STRATEGY=rrf USE_REWRITER=0 streamlit run app.py

The default remains `interleave` until the new strategy is validated end-to-end.
"""
import json
import os

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever


def _build(k, pool=10, rrf_const=60):
    """Build dense, BM25, interleaved hybrid, and RRF search functions.

    Each underlying retriever fetches a wider candidate pool so the hybrid
    strategies can fuse more candidates before returning at most `k` chunks.
    Chunk IDs are kept intact: G.2#1 and G.2#2 are distinct policy chunks and
    must not be collapsed into the same parent section.
    """
    pool = max(pool, k)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "chroma_db")
    chunks_file = os.path.join(current_dir, "data", "policy_chunks.json")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
    )
    chroma_retriever = vector_store.as_retriever(search_kwargs={"k": pool})

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
        """Deduplicate repeated full chunk IDs, preserving sibling chunks."""
        out, seen = [], set()
        for doc in docs:
            doc_id = str(doc.metadata.get("id", "")).strip()
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                out.append(doc)
        return out

    def dense_search(query):
        return _unique(chroma_retriever.invoke(query))[:k]

    def bm25_search(query):
        return _unique(bm25_retriever.invoke(query))[:k]

    def hybrid_search(query):
        """Legacy hybrid strategy: alternate BM25 and dense results."""
        sparse_hits = bm25_retriever.invoke(query)
        dense_hits = chroma_retriever.invoke(query)

        merged_docs, seen_ids = [], set()
        for pair in zip(sparse_hits, dense_hits):
            for doc in pair:
                doc_id = str(doc.metadata.get("id", "")).strip()
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    merged_docs.append(doc)
        return merged_docs[:k]

    def hybrid_rrf_search(query):
        """Fuse dense and BM25 ranks using reciprocal-rank fusion."""
        scores, docs = {}, {}
        for hits in (bm25_retriever.invoke(query), chroma_retriever.invoke(query)):
            seen_in_list = set()
            rank = 0
            for doc in hits:
                doc_id = str(doc.metadata.get("id", "")).strip()
                if not doc_id or doc_id in seen_in_list:
                    continue
                seen_in_list.add(doc_id)
                rank += 1
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_const + rank)
                docs.setdefault(doc_id, doc)
        best = sorted(scores, key=scores.get, reverse=True)[:k]
        return [docs[doc_id] for doc_id in best]

    return dense_search, bm25_search, hybrid_search, hybrid_rrf_search


def get_hybrid_retriever(k=4, strategy=None):
    """Return the selected live retriever.

    `strategy` overrides RETRIEVAL_STRATEGY. The default stays `interleave`
    so simply replacing this file does not silently change existing behavior.
    """
    selected = (strategy or os.getenv("RETRIEVAL_STRATEGY", "interleave")).strip().lower()
    dense, bm25, hybrid, hybrid_rrf = _build(k)
    retrievers = {
        "interleave": hybrid,
        "hybrid": hybrid,
        "rrf": hybrid_rrf,
        "hybrid_rrf": hybrid_rrf,
        "dense": dense,
        "bm25": bm25,
    }
    if selected not in retrievers:
        choices = ", ".join(("interleave", "rrf", "dense", "bm25"))
        raise ValueError(
            f"Unknown RETRIEVAL_STRATEGY={selected!r}. Choose one of: {choices}."
        )
    print(f"[retrieval] strategy={selected}")
    return retrievers[selected]


def get_retrievers(k=4):
    """Return all strategies for benchmarks; does not use RETRIEVAL_STRATEGY."""
    dense, bm25, hybrid, hybrid_rrf = _build(k)
    return {
        "dense": dense,
        "bm25": bm25,
        "hybrid": hybrid,
        "hybrid_rrf": hybrid_rrf,
    }
