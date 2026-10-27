"""One-claim diagnostic for GC-022. Run from the project root.

This calls the configured Gemini model once through toulmin_agent.run_toulmin_pipeline
and prints only the fields needed to diagnose a citation/guardrail override.
It does not modify project files or print environment variables/API keys.
"""
import json
import os
import sys

# Make this a controlled, low-call diagnostic before importing the agent.
os.environ["USE_REWRITER"] = "0"
os.environ["MODEL_FALLBACK"] = "0"

from toulmin_agent import run_toulmin_pipeline  # noqa: E402

CLAIM = (
    "I had fracture surgery and was admitted for 3 days, 8 months after buying "
    "my Gold plan. I had no earlier medical conditions."
)


def main():
    print("Running one diagnostic claim (GC-022); this makes one adjudication call.\n")
    try:
        result = run_toulmin_pipeline(CLAIM)
    except Exception as exc:
        print(f"Diagnostic failed: {type(exc).__name__}: {exc}")
        sys.exit(1)

    print(f"Final verdict: {result.get('Claim', '(missing)')}")
    print(f"Model's original verdict: {result.get('Model_Claim', '(not overridden or not recorded)')}")
    print(f"Citation status: {result.get('Guardrail_Status', '(missing)')}")
    print(f"Override reason: {result.get('Decision_Override_Reason', '(none recorded)')}")
    print("\nBacking quote:")
    print(result.get("Backing", "(missing)"))
    print("\nRebuttal quote:")
    print(result.get("Rebuttal", "(missing)"))
    print("\nWarrant:")
    print(result.get("Warrant", "(missing)"))
    print("\nRetrieved clause IDs:")
    print(", ".join(str(x) for x in result.get("Retrieved_IDs", [])))
    print(f"\nModel used: {result.get('Model_Used', '(unknown)')}")


if __name__ == "__main__":
    main()
