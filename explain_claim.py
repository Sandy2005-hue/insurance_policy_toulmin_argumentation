"""Show the model's full reasoning for chosen gold claims (for the error analysis).

Usage:
    python explain_claim.py GC-001 GC-022
    python explain_claim.py GC-001 --no-rewriter       # match the no-rewriter run
Add --no-fallback to stay on the first model in MODEL_CHAIN.
"""
import json
import os
import sys
import time

ARGS = sys.argv[1:]
if "--no-rewriter" in ARGS:
    os.environ["USE_REWRITER"] = "0"
if "--no-fallback" in ARGS:
    os.environ["MODEL_FALLBACK"] = "0"
IDS = [a for a in ARGS if not a.startswith("--")]

from toulmin_agent import run_toulmin_pipeline  # noqa: E402


def short(value, n=600):
    text = str(value)
    return text if len(text) <= n else text[:n] + " ...[cut]"


def main():
    if not IDS:
        print("Give at least one claim ID, e.g. python explain_claim.py GC-001")
        return
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "data", "claims_gold.json"), encoding="utf-8") as f:
        claims = {c["id"]: c for c in json.load(f)}
    for cid in IDS:
        c = claims.get(cid)
        if not c:
            print(f"{cid}: not found in claims_gold.json")
            continue
        print("=" * 78)
        print(f"{cid} | expected: {c['expected_verdict']} | gold clauses: {c['gold_clause_ids']}")
        print(f"Claim: {c['text']}\n")
        try:
            res = run_toulmin_pipeline(c["text"])
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {str(e)[:200]}\n")
            continue
        for key in ("Claim", "Grounds", "Warrant", "Backing", "Rebuttal", "Guardrail_Status", "Model_Used"):
            print(f"{key}: {short(res.get(key))}\n")
        print(f"Retrieved clauses: {res.get('Retrieved_IDs')}\n")
        time.sleep(5)


if __name__ == "__main__":
    main()