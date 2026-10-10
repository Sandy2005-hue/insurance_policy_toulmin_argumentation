"""End-to-end verdict benchmark.

Usage:
    python benchmark_e2e.py --no-fallback                 # recommended for reported results
    python benchmark_e2e.py --no-fallback --no-rewriter   # without LLM query expansion
    python benchmark_e2e.py --no-fallback --resume        # continue after rate-limit failures
    python benchmark_e2e.py --out e2e_norewriter.json     # choose the results file name

Results are saved after every claim, so a crash or rate limit never loses work.
API errors are never counted as wrong answers; unscored claims are listed at the end.
"""
import json
import os
import sys
import time
import warnings
from collections import Counter

ARGV = sys.argv[1:]
# Flags must be applied BEFORE the agent module is imported.
if "--no-rewriter" in ARGV:
    os.environ["USE_REWRITER"] = "0"
if "--no-fallback" in ARGV:
    os.environ["MODEL_FALLBACK"] = "0"

from toulmin_agent import run_toulmin_pipeline  # noqa: E402

warnings.filterwarnings("ignore")

LABELS = ["Claim Approved", "Claim Denied", "Needs Human Review"]
MAX_RETRIES = 3                                   # attempts per claim when rate-limited
WAIT_ON_LIMIT = int(os.getenv("E2E_WAIT", "30"))  # seconds before retrying
GAP = int(os.getenv("E2E_GAP", "8"))              # seconds between claims
COMPARABLE_FLAGS = ("--no-rewriter", "--no-fallback")


def is_rate_limit(e):
    s = str(e).lower()
    return any(k in s for k in ("429", "503", "exhausted", "quota"))


def run_one(text):
    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return run_toulmin_pipeline(text), None
        except Exception as e:
            last = e
            if is_rate_limit(e) and attempt < MAX_RETRIES:
                print(f"   rate limit, waiting {WAIT_ON_LIMIT}s (retry {attempt}/{MAX_RETRIES - 1})")
                time.sleep(WAIT_ON_LIMIT)
            else:
                break
    return None, last


def norm(v):
    v = str(v).strip()
    return v if v in LABELS else "OTHER"


def base_ids(ids):
    return {str(i).split("#")[0] for i in (ids or [])}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_name = ARGV[ARGV.index("--out") + 1] if "--out" in ARGV else "e2e_results.json"
    out_path = os.path.join(here, out_name)
    with open(os.path.join(here, "data", "claims_gold.json"), encoding="utf-8") as f:
        claims = json.load(f)

    flags = sorted(a for a in ARGV if a in COMPARABLE_FLAGS)
    rows = []
    if "--resume" in ARGV and os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            prev = json.load(f)
        if prev.get("flags") == flags:
            rows = prev.get("rows", [])
            print(f"Resuming: {len(rows)} claims already scored in {out_name}")
        else:
            print("Previous results used different flags; starting fresh.")

    def save():
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"flags": flags, "rows": rows}, f, indent=2)

    done = {r["id"] for r in rows}
    print(f"Claims: {len(claims)} | flags: {flags or 'none'} | results file: {out_name}\n")
    print(f"{'ID':<8} | {'Expected':<20} | {'Predicted':<20} | Cite | Gold? | Model")

    consecutive_fail = 0
    for item in claims:
        if item["id"] in done:
            continue
        res, err = run_one(item["text"])
        if res is None:
            consecutive_fail += 1
            print(f"{item['id']:<8} | NOT SCORED (API/other error): {str(err)[:70]}")
            if consecutive_fail >= 3:
                print("\nSTOPPING: 3 claims in a row failed, so the API quota is probably used up for this model.\n"
                      "Wait (or try again tomorrow) and re-run the same command with --resume.")
                break
        else:
            consecutive_fail = 0
            exp = item["expected_verdict"]
            pred = norm(res.get("Claim", ""))
            cited_ok = "Verified" in str(res.get("Guardrail_Status", ""))
            gold_in_ctx = bool(base_ids(item.get("gold_clause_ids")) & base_ids(res.get("Retrieved_IDs")))
            model = res.get("Model_Used") or "?"
            rows.append({
                "id": item["id"], "expected": exp, "predicted": res.get("Claim"),
                "citation_check": res.get("Guardrail_Status"), "gold_clause_retrieved": gold_in_ctx,
                "model": model, "rewriter_model": res.get("Rewriter_Model"),
                "retrieved_ids": res.get("Retrieved_IDs"), "gold_clause_ids": item.get("gold_clause_ids"),
            })
            print(f"{item['id']:<8} | {exp:<20} | {'OK ' if exp == pred else 'XX '}{pred[:17]:<17} | "
                  f"{'Y' if cited_ok else 'N':<4} | {'Y' if gold_in_ctx else 'N':<5} | {model}")
            save()
        time.sleep(GAP)

    # ---------------- metrics, always computed from the saved rows ----------------
    scored_ids = {r["id"] for r in rows}
    unscored = [c["id"] for c in claims if c["id"] not in scored_ids]
    n = len(rows)
    cols = LABELS + ["OTHER"]
    conf = Counter((r["expected"], norm(r["predicted"])) for r in rows)
    correct = sum(r["expected"] == norm(r["predicted"]) for r in rows)
    cited = sum("Verified" in str(r["citation_check"]) for r in rows)
    gold_ctx = sum(bool(r["gold_clause_retrieved"]) for r in rows)
    models = Counter(r["model"] for r in rows)
    gold_denied = sum(conf[("Claim Denied", p)] for p in cols)
    reviews = sum(conf[(e, "Needs Human Review")] for e in LABELS)

    print("\nConfusion matrix (rows = expected, columns = predicted)")
    print(f"{'':<20}" + "".join(f"{c[:12]:>14}" for c in cols))
    for e in LABELS:
        print(f"{e:<20}" + "".join(f"{conf[(e, p)]:>14}" for p in cols))

    print(f"\nClaims in set: {len(claims)} | scored: {n} | not scored: {len(unscored)} {unscored or ''}")
    print(f"Accuracy on scored claims: {correct}/{n} = {100 * correct / max(n, 1):.1f}%")
    print(f"False approvals (expected Denied, predicted Approved): {conf[('Claim Denied', 'Claim Approved')]}/{gold_denied}")
    print(f"Sent to human review: {reviews}/{n}")
    print(f"Citation check passed: {cited}/{n}")
    print(f"Gold clause present in retrieved context: {gold_ctx}/{n}")
    print(f"Models that answered: {dict(models)}")
    if len(models) > 1:
        print("WARNING: more than one model answered. For reported results re-run with --no-fallback "
              "(or report per-model results).")

    wrong = [r for r in rows if r["expected"] != norm(r["predicted"])]
    if wrong:
        print("\nWrong predictions (write one line of cause for each):")
        for r in wrong:
            print(f"  {r['id']}: expected {r['expected']}, got {r['predicted']} | "
                  f"gold clause retrieved: {'yes' if r['gold_clause_retrieved'] else 'NO'} | "
                  f"gold {r['gold_clause_ids']} | retrieved {r['retrieved_ids']}")
    if unscored:
        print(f"\nRe-run with --resume to score the remaining {len(unscored)} claim(s).")
    print(f"Saved to {out_name}")


if __name__ == "__main__":
    main()