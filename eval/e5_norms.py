"""E5: how accurate is LLM-assisted norm extraction? (eval plan E5, handoff D12)

Compares the extractor's raw output (knowledge/norms_candidates.csv, never edited) with the
rows Aamir verified against the papers (knowledge/norms.csv):
1. **Field-level accuracy:** for every row not rejected, each scored field of the candidate
   vs the verified row. A "corrected" row differs in the fields he changed.
2. **Hallucinated rows:** rows rejected as unsupported by the source, plus any row from a
   control paper (its numbers are only in figures, so no row can be supported).
3. **Recall:** rows the extractor missed (knowledge/norms_missed.csv).
4. **The QA flags** (tools/extract_norms.py): how often a flagged row was actually wrong.

    uv run python -m eval.e5_norms
"""

from __future__ import annotations

import csv
from pathlib import Path

from eval.e1_pose import table, write_csv
from kickcoach.compare import norms

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MISSED_CSV = ROOT / "knowledge" / "norms_missed.csv"
# The fields a reviewer judges. norm_id/source/verified/review are bookkeeping; quote and
# notes are free text checked for support, not for equality.
SCORED_FIELDS = [
    c for c in norms.COLUMNS
    if c not in ("norm_id", "source_id", "source", "quote", "notes", "verified", "review",
                 "review_note")
]  # fmt: skip


def _controls() -> set[str]:
    from tools.extract_norms import SOURCES

    return {s["id"] for s in SOURCES if s["role"] == "control"}


def score(
    candidates: list[norms.NormRow],
    verified: list[norms.NormRow],
    missed: list[dict],
    controls: set[str],
) -> dict:
    by_id = {r.norm_id: r for r in verified}
    rows, field_hits, field_total = [], 0, 0
    for c in candidates:
        v = by_id.get(c.norm_id)
        cf = norms.to_flat(c)
        review = v.review if v else "pending"
        hallucinated = review == "rejected" or c.source_id in controls
        wrong = []
        if v is not None and not hallucinated and review != "pending":
            vf = norms.to_flat(v)
            wrong = [f for f in SCORED_FIELDS if str(cf[f]) != str(vf[f])]
            field_hits += len(SCORED_FIELDS) - len(wrong)
            field_total += len(SCORED_FIELDS)
        rows.append(
            {
                "norm_id": c.norm_id,
                "source_id": c.source_id,
                "variable": c.variable,
                "review": review,
                "hallucinated": hallucinated,
                "wrong_fields": " ".join(wrong),
                "qa_flagged": "QA:" in (c.notes or ""),
            }
        )
    pending = sum(r["review"] == "pending" for r in rows)
    supported = sum(not r["hallucinated"] and r["review"] != "pending" for r in rows)
    flagged = [r for r in rows if r["qa_flagged"]]
    return {
        "rows": rows,
        "n": len(rows),
        "pending": pending,
        "hallucinated": sum(r["hallucinated"] for r in rows),
        "from_controls": sum(r["source_id"] in controls for r in rows),
        "supported": supported,
        "rows_all_fields_right": sum(
            not r["hallucinated"] and r["review"] != "pending" and not r["wrong_fields"]
            for r in rows
        ),
        "field_hits": field_hits,
        "field_total": field_total,
        "missed": len(missed),
        "flagged": len(flagged),
        "flagged_wrong": sum(bool(r["wrong_fields"]) or r["hallucinated"] for r in flagged),
    }


def report(s: dict, missed: list[dict], controls: set[str]) -> str:
    by_src: dict[str, list[dict]] = {}
    for r in s["rows"]:
        by_src.setdefault(r["source_id"], []).append(r)
    src_body = [
        [
            sid,
            "control" if sid in controls else "source",
            str(len(rs)),
            str(sum(not r["wrong_fields"] and not r["hallucinated"] for r in rs)),
            str(sum(r["hallucinated"] for r in rs)),
        ]
        for sid, rs in by_src.items()
    ]
    for sid in sorted(controls - set(by_src)):
        src_body.append([sid, "control", "0", "–", "0"])
    recall_den = s["supported"] + s["missed"]
    L = [
        "# E5: LLM-assisted norm extraction vs human verification",
        "",
        "Extractor: `tools/extract_norms.py` (prompt norms-v1, `gemini-3.6-flash`, one call per "
        "paper, structured output against the D12 schema). Verification: Aamir, in-session, "
        "against the papers (`knowledge/norms.csv`, column `review`). "
        f"Fields scored per row: {len(SCORED_FIELDS)} ({', '.join(SCORED_FIELDS)}).",
        "",
    ]
    if s["pending"]:
        L += [f"**{s['pending']} rows are still pending review; they are not scored.**", ""]
    L += [
        "| metric | value |",
        "|---|---|",
        f"| rows extracted | {s['n']} (from {len(by_src)} papers; 2 control papers gave "
        f"{s['from_controls']}) |",
        f"| hallucinated rows (rejected, or from a control paper) | {s['hallucinated']} of "
        f"{s['n']} |",
        f"| rows with every field right | {s['rows_all_fields_right']} of {s['supported']} |",
        f"| field-level accuracy | {s['field_hits']} of {s['field_total']} fields "
        f"({100 * s['field_hits'] / max(s['field_total'], 1):.0f}%) |",
        f"| recall (supported / supported + missed) | {s['supported']} of {recall_den} "
        f"({100 * s['supported'] / max(recall_den, 1):.0f}%) |",
        f"| QA-flagged rows that were wrong | {s['flagged_wrong']} of {s['flagged']} |",
        "",
        "## By paper",
        "",
        *table(["paper", "role", "rows", "all fields right", "hallucinated"], src_body),
        "",
        "## Missed rows",
        "",
        *(
            table(
                ["paper", "where", "variable", "quote", "decision"],
                [
                    [m["source_id"], m["page_or_table"], m["variable"], m["quote"],
                     f"{m['decision']} ({m['decided_by']})"]
                    for m in missed
                ],
            )
            if missed
            else ["None recorded."]
        ),
        "",
        "## Caveats",
        "",
        "- **Small n:** 4 source papers, 2 controls, one extraction run with one model. The "
        "result says this extractor didn't invent rows here, not that it never does.",
        "- **Verification was answered per table group** (7 groups for 32 rows), each with "
        "every field shown. A group answer of \"all correct\" covers every field of every row "
        "in it.",
        (
            "- **The QA flags caught layout, not errors:** no flagged row was wrong; the flags "
            "came from quotes rebuilt out of table columns that pypdf extracted on separate "
            "lines. A flag means \"check the table\"."
            if s["flagged"] and not s["flagged_wrong"]
            else f"- **QA flags:** {s['flagged_wrong']} of {s['flagged']} flagged rows were wrong."
        ),
        "- **Correct ≠ usable:** some verified rows can't be compared with the pipeline (cm vs "
        "leg lengths, a suggestion rather than a measurement). That is the comparison "
        "engine's job (`kickcoach/compare/engine.py`), not E5's.",
        "",
    ]  # fmt: skip
    return "\n".join(L)


def load_missed(path: Path = MISSED_CSV) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    candidates = norms.load(norms.CANDIDATES_CSV)
    verified = norms.load(norms.NORMS_CSV)
    missed = load_missed()
    controls = _controls()
    s = score(candidates, verified, missed, controls)
    write_csv(RESULTS / "e5.csv", s["rows"])
    md = report(s, missed, controls)
    (RESULTS / "e5_summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
