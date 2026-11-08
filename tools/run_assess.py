"""Norm comparisons, faults and feedback on every manifest clip's pipeline output
(results/pipeline/*.json), as the API would add them. Writes results/assess_summary.csv.

    uv run python tools/run_assess.py          # template feedback, no API calls
    uv run python tools/run_assess.py --llm    # LLM feedback; responses cached in
                                               # results/feedback_llm/ (committed)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from kickcoach.compare import norms
from kickcoach.feedback import generate as gen
from kickcoach.llm import GeminiClient
from kickcoach.llm.gemini import DEV_MODEL
from kickcoach.session import assess

ROOT = Path(__file__).resolve().parents[1]
LLM_CACHE = ROOT / "results" / "feedback_llm"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    args = ap.parse_args()
    verified = norms.verified_norms()
    client = GeminiClient(DEV_MODEL, LLM_CACHE) if args.llm else None
    rows = []
    for p in sorted((ROOT / "results" / "pipeline").glob("c*.json")):
        res = json.loads(p.read_text())
        out = assess(res, use_llm=args.llm, client=client, verified=verified)
        fb = out["feedback"]
        attempts = fb.get("attempts", [])
        rows.append(
            {
                "clip_id": res["clip_id"],
                "accepted": res["quality"]["accepted"],
                "faults": " ".join(f["fault_id"] for f in out["faults"]),
                **{f"verdict_{c['feature_id']}": c["verdict"] for c in out["comparisons"]},
                "feedback_by": fb["generated_by"],
                "llm_attempts": len(attempts),
                "first_try_passed": (not attempts[0]["errors"]) if attempts else "",
                "first_try_errors": " | ".join(attempts[0]["errors"]) if attempts else "",
                "fallback_reason": fb.get("fallback_reason") or "",
            }
        )
        print(
            f"{res['clip_id']}: {rows[-1]['faults'] or '-'} -> {fb['generated_by']}"
            f"{' (' + str(len(attempts)) + ' attempts)' if attempts else ''}",
            flush=True,
        )
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(ROOT / "results" / "assess_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(
        f"prompts {gen.RAG_PROMPT_VERSION} (with passages) / {gen.PROMPT_VERSION}; wrote "
        "results/assess_summary.csv"
    )


if __name__ == "__main__":
    main()
