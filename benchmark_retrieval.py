import json
import os
import time
import warnings
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
warnings.filterwarnings("ignore")

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash-latest", temperature=0.0)

rewriter_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a health insurance retrieval query expansion engine.
Convert the colloquial user question into formal Indian health insurance contract terms, clause titles, synonyms, and numerical conversions.
Rules:
- If years are mentioned, include months (e.g., 5 years -> 60 continuous months moratorium period).
- If overseas/countries are mentioned, include "outside geographical limits of India" or "global cover".
- If no admission is mentioned, include "Out-Patient OPD treatment".
- If falling sick early is mentioned, include "first thirty days waiting period".
- If pre-existing disease is mentioned, include "Pre-existing Diseases PED waiting period".
- Output ONLY 4-8 dense search keywords separated by spaces. No explanations."""),
    ("user", "{question}")
])

query_rewriter = rewriter_prompt | llm | StrOutputParser()

def evaluate_retrieval(k=4):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    questions_file = os.path.join(current_dir, "data", "retrieval_questions.json")
    chunks_file = os.path.join(current_dir, "data", "policy_chunks.json")
    db_path = os.path.join(current_dir, "chroma_db")

    print(f"📊 Running ADVANCED RAG (Query Rewriting + Hybrid Search) @ Top-K = {k}...")

    # 1. Dense Semantic Retriever (MiniLM)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=db_path, embedding_function=embeddings)
    chroma_retriever = vector_store.as_retriever(search_kwargs={"k": k})


    with open(chunks_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    bm25_docs = [
        Document(
            page_content=f"{c.get('section_title', '')} {c.get('title', '')} {c['text']}", 
            metadata={"id": str(c['id'])}
        ) for c in data['chunks']
    ]
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = k

    with open(questions_file, 'r', encoding='utf-8') as f:
        benchmarks = json.load(f)

    total_queries = len(benchmarks)
    hits = 0
    reciprocal_ranks = []
    precision_scores = []
    recall_scores = []

    print("\n" + "="*75)
    print(f"{'Query ID':<8} | {'Status':<6} | {'Rank':<5} | {'Expanded Keywords'}")
    print("="*75)

    for i, item in enumerate(benchmarks):
        query = item['q']
        gold_ids = set(item['gold'])


        try:
            expansion = query_rewriter.invoke({"question": query})
            expanded_query = f"{query} {expansion}"
        except Exception:
            expanded_query = query
            expansion = "N/A"

  
        sparse_hits = bm25_retriever.invoke(expanded_query)
        dense_hits = chroma_retriever.invoke(expanded_query)

  
        merged_docs = []
        seen_ids = set()
        for pair in zip(sparse_hits, dense_hits):
            for doc in pair:
                doc_id = doc.metadata.get('id', '').split('#')[0]
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    merged_docs.append(doc)
        
        retrieved_docs = merged_docs[:k]
        retrieved_ids = [doc.metadata.get('id', '').split('#')[0] for doc in retrieved_docs]

        hit_rank = None
        for rank, r_id in enumerate(retrieved_ids, 1):
            if r_id in gold_ids:
                hit_rank = rank
                break

        if hit_rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / hit_rank)
            status = "HIT"
        else:
            reciprocal_ranks.append(0.0)
            status = "MISS"

        relevant_retrieved = len(set(retrieved_ids).intersection(gold_ids))
        p_at_k = relevant_retrieved / k
        r_at_k = relevant_retrieved / len(gold_ids) if len(gold_ids) > 0 else 0

        precision_scores.append(p_at_k)
        recall_scores.append(r_at_k)

  
        exp_preview = (expansion[:40] + '..') if len(expansion) > 40 else expansion
        print(f"Q{i+1:<7} | {status:<6} | {str(hit_rank):<5} | {exp_preview}")
        
        time.sleep(5) 

    hit_rate = (hits / total_queries) * 100
    mrr = sum(reciprocal_ranks) / total_queries
    avg_precision = (sum(precision_scores) / total_queries) * 100
    avg_recall = (sum(recall_scores) / total_queries) * 100

    print("\n" + "="*50)
    print("🎯 OFFICIAL FIELD 6: ADVANCED RAG PERFORMANCE METRICS")
    print("="*50)
    print(f"Total Benchmark Queries: {total_queries}")
    print(f"Hit Rate @ K={k}:         {hit_rate:.2f}%")
    print(f"Mean Reciprocal Rank:    {mrr:.4f}")
    print(f"Average Precision @ K:   {avg_precision:.2f}%")
    print(f"Average Recall @ K:      {avg_recall:.2f}%")
    print("="*50)

if __name__ == "__main__":
    evaluate_retrieval(k=4)