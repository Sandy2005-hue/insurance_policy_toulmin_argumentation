"""Retrieval benchmark: dense vs BM25 vs hybrid on the RAW questions.

Uses the same retriever code as the app (shared_retriever.py), so there is no duplicated logic.
Hand-written query expansions, if the file exists, are shown on ONE clearly labelled
upper-bound row. They are not a real system.

Usage:  python benchmark_retrieval.py
"""
import json
import os
import warnings

from shared_retriever import get_retrievers

warnings.filterwarnings("ignore")
K = 4


def base_ids(docs):
    return list(dict.fromkeys(
        d.metadata.get("id", "").split("#")[0] for d in docs if d.metadata.get("id")))


def evaluate(search, questions, expansions=None):
    rows, hits, rr_sum, recall_sum = [], 0, 0.0, 0.0
    for item in questions:
        q, gold = item["q"], set(item["gold"])
        query = f"{q} {expansions.get(q, '')}".strip() if expansions else q
        ids = base_ids(search(query))[:K]
        rank = next((r for r, i in enumerate(ids, 1) if i in gold), None)
        hits += rank is not None
        rr_sum += 1.0 / rank if rank else 0.0
        recall_sum += len(gold & set(ids)) / len(gold)
        rows.append({"q": q, "gold": sorted(gold), "retrieved": ids, "hit_rank": rank})
    n = len(questions)
    return {"hit": hits, "n": n, "mrr": rr_sum / n, "recall": recall_sum / n, "rows": rows}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "data", "retrieval_questions.json"), encoding="utf-8") as f:
        questions = json.load(f)
    expansions = None
    exp_path = os.path.join(here, "data", "query_expansions.json")
    if os.path.exists(exp_path):
        with open(exp_path, encoding="utf-8") as f:
            expansions = json.load(f)

    searchers = get_retrievers(k=K)
    results = {
        "Dense only (raw query)": evaluate(searchers["dense"], questions),
        "BM25 only (raw query)": evaluate(searchers["bm25"], questions),
        "Hybrid, rank interleaving (raw query)": evaluate(searchers["hybrid"], questions),
        "Hybrid, reciprocal rank fusion (raw query)": evaluate(searchers["hybrid_rrf"], questions),
    }
    raw_names = list(results)
    if expansions:
        results["Hybrid + hand-written expansion (UPPER BOUND, not a real system)"] = evaluate(
            searchers["hybrid"], questions, expansions)

    print(f"\nRetrieval benchmark: {len(questions)} questions, top {K} clauses\n")
    print(f"{'Configuration':<66} {'Hit@' + str(K):>14} {'MRR':>7} {'Recall@' + str(K):>10}")
    for name, r in results.items():
        print(f"{name:<66} {r['hit']:>3}/{r['n']} = {100 * r['hit'] / r['n']:>3.0f}% "
              f"{r['mrr']:>7.3f} {100 * r['recall']:>9.0f}%")

    best = max(raw_names, key=lambda n: (results[n]["hit"], results[n]["mrr"]))
    print(f"\nQuestions missed by the best raw-query configuration ({best}), no gold clause in the top {K}:")
    missed = [row for row in results[best]["rows"] if row["hit_rank"] is None]
    if not missed:
        print("  none")
    for row in missed:
        print(f"  - {row['q'][:70]}\n      gold {row['gold']} | retrieved {row['retrieved']}")

    out = os.path.join(here, "retrieval_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved per-question details to retrieval_results.json")
    print("Report Hit@4 and MRR. Precision@4 is capped by the 1-3 gold clauses per question, so it is not shown.")


if __name__ == "__main__":
    main()