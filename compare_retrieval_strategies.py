"""Compare retrieval strategies for CLAIMS with no LLM calls (no API quota used).

For every claim in the file it checks whether the clauses listed in `gold_clause_ids`
were retrieved. Use it to pick a retrieval strategy on the DEVELOPMENT claims, then
confirm the choice on hold-out claims the strategy was never tuned on.

Usage:
    python compare_retrieval_strategies.py                      # data/claims_gold.json
    python compare_retrieval_strategies.py data/claims_holdout.json
"""
import json
import os
import sys
import warnings
from collections import Counter

from shared_retriever import get_retrievers

warnings.filterwarnings("ignore")


def ids(docs):
    return list(dict.fromkeys(
        d.metadata.get("id", "").split("#")[0] for d in docs if d.metadata.get("id")))


def union(*lists):
    return list(dict.fromkeys(i for lst in lists for i in lst))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "claims_gold.json")
    with open(os.path.join(here, path), encoding="utf-8") as f:
        claims = json.load(f)

    r4 = get_retrievers(k=4)
    h4, d4, rrf4 = r4["hybrid"], r4["dense"], r4["hybrid_rrf"]
    h6 = get_retrievers(k=6)["hybrid"]
    h8 = get_retrievers(k=8)["hybrid"]

    strategies = {
        "S0 current: 2 prefixed queries, k=4 each": lambda c: union(
            ids(h4(f"benefits coverage {c}")), ids(h4(f"waiting periods exclusions {c}"))),
        "S1 plain claim, 1 query, k=8": lambda c: ids(h8(c)),
        "S2 plain claim, 1 query, k=6": lambda c: ids(h6(c)),
        "S3 plain claim + exclusion-focused query, k=4 each": lambda c: union(
            ids(h4(c)), ids(h4(f"{c} exclusion waiting period"))),
        "D0 dense only, 2 prefixed queries, k=4 each": lambda c: union(
            ids(d4(f"benefits coverage {c}")), ids(d4(f"waiting periods exclusions {c}"))),
        "R0 RRF hybrid, 2 prefixed queries, k=4 each": lambda c: union(
            ids(rrf4(f"benefits coverage {c}")), ids(rrf4(f"waiting periods exclusions {c}"))),
        "D3 dense only, plain + exclusion-focused, k=4 each": lambda c: union(
            ids(d4(c)), ids(d4(f"{c} exclusion waiting period"))),
        "R3 RRF hybrid, plain + exclusion-focused, k=4 each": lambda c: union(
            ids(rrf4(c)), ids(rrf4(f"{c} exclusion waiting period"))),
    }

    print(f"\nClaims: {len(claims)} from {path}\n")
    print(f"{'Strategy':<54} {'all gold':>9} {'any gold':>9} {'recall':>7} {'avg clauses':>12}")
    details = {}
    for name, fn in strategies.items():
        strict = loose = 0
        recall = 0.0
        sizes = []
        hubs = Counter()
        misses = []
        for c in claims:
            gold = {g.split("#")[0] for g in c["gold_clause_ids"]}
            got = set(fn(c["text"]))
            strict += gold <= got
            loose += bool(gold & got)
            recall += len(gold & got) / len(gold)
            sizes.append(len(got))
            hubs.update(got)
            if not gold <= got:
                misses.append(f"{c['id']} ({', '.join(sorted(gold - got))})")
        n = len(claims)
        print(f"{name:<54} {strict:>4}/{n:<4} {loose:>4}/{n:<4} {100 * recall / n:>6.0f}% {sum(sizes) / n:>12.1f}")
        details[name] = (hubs.most_common(4), misses)

    print("\nStill missing gold clauses, and the clauses retrieved for most claims, per strategy:")
    for name, (hubs, misses) in details.items():
        print(f"\n{name}")
        print("  most frequent clauses:", ", ".join(f"{c} ({k})" for c, k in hubs))
        print("  missing:", "; ".join(misses) if misses else "none")
    print("\n'avg clauses' is the amount of policy text the model would see, so compare strategies at similar sizes.")


if __name__ == "__main__":
    main()