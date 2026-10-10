import json, os, sys, time, warnings
from collections import Counter

if "--no-rewriter" in sys.argv:
    os.environ["USE_REWRITER"] = "0"          # must be set before importing the agent
from toulmin_agent import run_toulmin_pipeline

warnings.filterwarnings("ignore")
LABELS = ["Claim Approved", "Claim Denied", "Needs Human Review"]
MAX_RETRIES, WAIT_ON_LIMIT, GAP = 3, 30, 8

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
                print(f"   rate limit, waiting {WAIT_ON_LIMIT}s (retry {attempt}/{MAX_RETRIES-1})")
                time.sleep(WAIT_ON_LIMIT)
            else:
                break
    return None, last

def norm(v):
    v = str(v).strip()
    return v if v in LABELS else "OTHER"

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "data", "claims_gold.json"), encoding="utf-8") as f:
        claims = json.load(f)

    conf, errors, rows, correct, g_ok = Counter(), [], [], 0, 0
    print(f"{'ID':<8} | {'Expected':<20} | {'Predicted':<20} | Guardrail")
    for item in claims:
        res, err = run_one(item["text"])
        if res is None:
            errors.append((item["id"], str(err)[:80]))
            print(f"{item['id']:<8} | API/ERROR: {str(err)[:60]}")
        else:
            exp, pred = item["expected_verdict"], norm(res.get("Claim", ""))
            ok = pred == exp
            correct += ok
            g = "Verified" in str(res.get("Guardrail_Status", ""))
            g_ok += g
            conf[(exp, pred)] += 1
            rows.append({"id": item["id"], "expected": exp, "predicted": res.get("Claim"),
                         "guardrail": res.get("Guardrail_Status")})
            print(f"{item['id']:<8} | {exp:<20} | {'OK ' if ok else 'XX '}{pred[:17]:<17} | {'Y' if g else 'N'}")
        time.sleep(GAP)

    scored = len(claims) - len(errors)
    cols = LABELS + ["OTHER"]
    print("\nConfusion (rows = expected, cols = predicted)")
    print(f"{'':<20}" + "".join(f"{c[:12]:>14}" for c in cols))
    for e in LABELS:
        print(f"{e:<20}" + "".join(f"{conf[(e, p)]:>14}" for p in cols))

    gold_denied = sum(conf[("Claim Denied", p)] for p in cols)
    reviews = sum(conf[(e, "Needs Human Review")] for e in LABELS)
    print(f"\nClaims in set: {len(claims)} | scored: {scored} | API/other errors: {len(errors)}")
    print(f"Accuracy on scored claims: {correct}/{scored} = {100*correct/max(scored,1):.1f}%")
    print(f"False approvals (gold Denied -> Approved): {conf[('Claim Denied','Claim Approved')]}/{gold_denied}")
    print(f"Sent to human review: {reviews}/{scored}")
    print(f"Citations verified: {g_ok}/{scored}")
    if errors: print("Errors:", errors)

    with open(os.path.join(here, "e2e_results.json"), "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "errors": errors}, f, indent=2)

if __name__ == "__main__":
    main()