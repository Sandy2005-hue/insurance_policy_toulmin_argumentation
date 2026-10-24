"""Summarise an end-to-end results file for the report.

Usage:  python analyze_results.py e2e_fallback_run.json [claims_file.json]
Shows accuracy by category and by model, how often ALL gold clauses were retrieved
(strict) versus at least one (loose), and which clauses were most often missing.
"""
import json
import os
import sys
from collections import Counter, defaultdict


def base(ids):
    return {str(i).split("#")[0] for i in (ids or [])}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    name = sys.argv[1] if len(sys.argv) > 1 else "e2e_results.json"
    with open(os.path.join(here, name), encoding="utf-8") as f:
        rows = json.load(f)["rows"]
    claims_name = sys.argv[2] if len(sys.argv) > 2 else os.path.join("data", "claims_gold.json")
    with open(os.path.join(here, claims_name), encoding="utf-8") as f:
        category = {c["id"]: c["category"] for c in json.load(f)}

    for r in rows:
        r["ok"] = r["expected"] == str(r["predicted"]).strip()
        gold, got = base(r["gold_clause_ids"]), base(r["retrieved_ids"])
        r["loose"] = bool(gold & got)
        r["strict"] = gold <= got
        r["missing"] = sorted(gold - got)

    def pct(a, b):
        return f"{a}/{b} = {100 * a / b:.0f}%" if b else "n/a"

    print(f"File: {name} | claims scored: {len(rows)} | correct: {pct(sum(r['ok'] for r in rows), len(rows))}\n")

    print("Accuracy by category")
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[category.get(r["id"], "?")].append(r["ok"])
    for c, v in sorted(by_cat.items()):
        print(f"  {c:<20} {pct(sum(v), len(v))}")

    print("\nAccuracy by model (different claims per model, so do NOT compare models with this)")
    by_model = defaultdict(list)
    for r in rows:
        by_model[r.get("model", "?")].append(r["ok"])
    for m, v in sorted(by_model.items()):
        print(f"  {m:<20} {pct(sum(v), len(v))}")

    print("\nRetrieval vs correctness")
    print(f"  at least one gold clause retrieved (loose): {pct(sum(r['loose'] for r in rows), len(rows))}")
    print(f"  ALL gold clauses retrieved (strict):        {pct(sum(r['strict'] for r in rows), len(rows))}")
    for label, group in (("all gold clauses retrieved", [r for r in rows if r["strict"]]),
                         ("some gold clause missing", [r for r in rows if not r["strict"]])):
        print(f"  accuracy when {label}: {pct(sum(r['ok'] for r in group), len(group))}")

    freq = Counter(c for r in rows for c in base(r["retrieved_ids"]))
    print("\nClauses retrieved most often across ALL claims (a clause that shows up for most claims is a 'hub')")
    print("  " + ", ".join(f"{c} ({n}/{len(rows)})" for c, n in freq.most_common(8)))

    missing = Counter(c for r in rows for c in r["missing"])
    if missing:
        print("\nGold clauses most often NOT retrieved:", ", ".join(f"{c} ({n})" for c, n in missing.most_common()))

    print("\nWrong predictions")
    for r in rows:
        if not r["ok"]:
            print(f"  {r['id']}: expected {r['expected']} -> got {r['predicted']} | missing gold clauses: {r['missing'] or 'none'}")


if __name__ == "__main__":
    main()