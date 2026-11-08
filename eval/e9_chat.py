"""E9: does the coach chat pick the right tools, restate only numbers the tools returned, and
refuse to talk about features that weren't measured? (docs/PLAN.md Phase 9; DEC-043)

The scripted questions and the fixture players are in eval/e9_questions.yaml, written before
any run. Each question goes through the real agent (kickcoach.chat.agent.ask) against a
fresh SQLite database built from committed pipeline outputs. Model turns are cached in
results/e9/llm/ and free-text search queries in results/e9/query_embeddings.npz, so a fresh
clone rescores with no key.

    uv run python -m eval.e9_chat run     # ask every question (needs a key on a cache miss)
    uv run python -m eval.e9_chat         # score from the cache: results/e9.csv, e9_summary.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path

import yaml

from eval.e1_pose import table, write_csv
from kickcoach.chat.agent import MAX_TOOL_ROUNDS, NO_KEY, PROMPT_VERSION, ask
from kickcoach.chat.tools import Tools
from kickcoach.chat.validate import (
    _ID,
    NOT_MEASURED_PHRASES,
    check,
    grounding,
    number_errors,
)
from kickcoach.compare import norms
from kickcoach.feedback.validate import DEFAULT_TOLERANCE, TOLERANCE, number_spans, numbers_in
from kickcoach.llm import DailyQuotaExhausted, GeminiClient
from kickcoach.llm.gemini import DEV_MODEL
from kickcoach.retrieval.embed import MODEL as EMBED_MODEL
from kickcoach.retrieval.embed import Embedder
from kickcoach.session import assess
from kickcoach.storage import db

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
QUESTIONS = ROOT / "eval" / "e9_questions.yaml"
E9_DIR = RESULTS / "e9"
LLM_CACHE = E9_DIR / "llm"
QUERY_CACHE = E9_DIR / "query_embeddings.npz"
RUNS = E9_DIR / "runs.json"
# A number "for" a velocity: the refusal check looks for one next to its unit.
UNIT_PATTERN = {"peak_knee_ext_velocity": r"(?:deg/s|°/s|degrees? per second)"}


def load_spec() -> dict:
    return yaml.safe_load(QUESTIONS.read_text())


def pipeline(clip: str) -> dict:
    return json.loads((RESULTS / "pipeline" / f"{clip}.json").read_text())


def build_fixture(path: Path, spec: dict) -> None:
    """The fixture players' sessions, from committed pipeline outputs (template feedback,
    no API calls), with the spec's dates so tool results, and so cache keys, are stable."""
    con = db.connect(path)
    verified = norms.verified_norms()
    for player, sessions in spec["players"].items():
        for s in sessions:
            res = assess(pipeline(s["clip"]), use_llm=False, verified=verified)
            db.save_session(con, player=player, level=None, clip_name=s["clip"], result=res,
                            inputs={"fixture": True},
                            created_at=f"{s['date']}T10:00:00+00:00")  # fmt: skip
    con.close()


def run(only: list[str] | None = None, api_key: str | None = None) -> dict:
    spec = load_spec()
    client = GeminiClient(DEV_MODEL, LLM_CACHE, api_key=api_key)
    embed_client = GeminiClient(EMBED_MODEL, QUERY_CACHE.parent, api_key=api_key)
    embedder = Embedder(write_cache=QUERY_CACHE, client=embed_client)
    runs = json.loads(RUNS.read_text()) if RUNS.exists() and api_key != "" else {}
    with tempfile.TemporaryDirectory() as tmp:
        build_fixture(Path(tmp) / "fixture.db", spec)
        con = db.connect(Path(tmp) / "fixture.db")
        for q in spec["questions"]:
            if only and q["id"] not in only:
                continue
            tools = Tools(con, q["player"], embedder=embedder)
            try:
                out = ask(q["text"], tools, client, tag=q["id"])
            except DailyQuotaExhausted as e:
                print(f"{q['id']}: daily quota used up; stopping ({str(e)[:120]})", flush=True)
                break
            if out["fallback_reason"] == NO_KEY:  # a cache miss with no key: not run
                print(f"{q['id']}: not cached", flush=True)
                continue
            runs[q["id"]] = out
            calls = ", ".join(f"{c['name']}({json.dumps(c['args'])})" for c in out["tool_calls"])
            print(f"{q['id']}: {out['generated_by']} {out['model_calls']} calls [{calls}]\n"
                  f"    {out['answer'][:300]}", flush=True)  # fmt: skip
        con.close()
    return runs


# -- scoring --------------------------------------------------------------------------------
def expected_value(ref: dict) -> tuple[float, str]:
    def feat(clip, fid):
        f = next(f for f in pipeline(clip)["features"] if f["id"] == fid)
        return float(f["value"]), f["unit"]

    if "change" in ref:
        (a, unit), (b, _) = (feat(c, ref["change"]) for c in ref["clips"])
        return b - a, unit
    return feat(ref["clip"], ref["feature"])


def has_value(answer: str, refs: list[dict]) -> bool:
    nums = numbers_in(answer)
    for ref in refs:
        v, unit = expected_value(ref)
        tol = TOLERANCE.get(unit, DEFAULT_TOLERANCE)
        if any(abs(abs(x) - abs(v)) <= tol for x in nums):
            return True
    return False


def refused(answer: str, fid: str) -> bool:
    low = answer.lower()
    says = any(p in low for p in NOT_MEASURED_PHRASES)
    value = re.search(rf"\d\s*{UNIT_PATTERN.get(fid, '(?!)')}", low)
    return says and not value


def tool_choice(q: dict, calls: list[dict]) -> tuple[bool, bool]:
    """(the right tools were called, they got the expected arguments)."""
    names = {c["name"] for c in calls}
    if q.get("no_tools"):
        return not names, not names
    tools_ok = all(names & set(group) for group in q.get("required", []))
    args_ok = all(
        any(c["name"] == tool and all(c["args"].get(k) == v for k, v in want.items())
            for c in calls)
        for tool, want in (q.get("args") or {}).items()
    )  # fmt: skip
    return tools_ok, args_ok


def score_one(q: dict, out: dict) -> dict:
    final = out["answer"]
    first = out["attempts"][0]["answer"] if out["attempts"] else ""
    g = grounding([c["result"] for c in out["tool_calls"]], q["text"])
    tools_ok, args_ok = tool_choice(q, out["tool_calls"])
    row = {
        "id": q["id"] + (" (post hoc)" if q.get("post_hoc") else ""), "kind": q["kind"],
        "player": q["player"],
        "generated_by": out["generated_by"], "model_calls": out["model_calls"],
        "attempts": len(out["attempts"]),
        "first_try_passed": bool(out["attempts"]) and not out["attempts"][0]["errors"],
        "first_try_errors": " | ".join(out["attempts"][0]["errors"]) if out["attempts"] else "",
        "tools_called": " ".join(c["name"] for c in out["tool_calls"]),
        "tool_args": json.dumps([c["args"] for c in out["tool_calls"]]),
        "tools_ok": tools_ok, "args_ok": args_ok,
    }  # fmt: skip
    for name, text in (("first", first), ("final", final)):
        row[f"{name}_numbers"] = len(number_spans(_ID.sub("", text)))
        row[f"{name}_ungrounded"] = len(number_errors(text, g))
    row["final_check_errors"] = len(check(final, g)) if out["generated_by"] == "llm" else ""
    row["has_value"] = has_value(final, q["expect_values"]) if q.get("expect_values") else ""
    low = final.lower()
    row["mentions_ok"] = (all(any(p.lower() in low for p in grp) for grp in q["must_mention"])
                          if q.get("must_mention") else "")  # fmt: skip
    row["refused_ok"] = refused(final, q["refuse"]) if q.get("refuse") else ""
    row["answer"] = final
    return row


def _n(rows, key, val=True) -> str:
    scored = [r for r in rows if r[key] != ""]
    return f"{sum(r[key] == val for r in scored)} of {len(scored)}"


def report(rows: list[dict], spec_sha: str, missing: list[str]) -> str:
    llm = [r for r in rows if r["generated_by"] == "llm"]
    L = [
        "# E9: the coach chat's tool use and faithfulness",
        "",
        f"Generated by `uv run python -m eval.e9_chat`. Model `{DEV_MODEL}`, prompt "
        f"`{PROMPT_VERSION}`, at most {MAX_TOOL_ROUNDS} tool rounds then a forced answer, one "
        f"retry with the checker's errors. {len(rows)} scripted questions "
        f"(`eval/e9_questions.yaml`, sha256 `{spec_sha[:16]}`; written before any run, except "
        "those marked post hoc, added after a run from a failure seen outside the set) "
        "against two **fixture players**: real pipeline outputs from different people's clips "
        "stored as sessions with made-up dates. The trends are not anyone's progression.",
        "",
    ]
    if missing:
        L += [f"**Not run yet:** {', '.join(missing)} (no cached answer).", ""]
    if not rows:
        return "\n".join(L) + "\n"
    first_nums = sum(r["first_numbers"] for r in rows)
    first_bad = sum(r["first_ungrounded"] for r in rows)
    final_nums = sum(r["final_numbers"] for r in llm)
    final_bad = sum(r["final_ungrounded"] for r in llm)
    L += [
        "## Headline",
        "",
        *table(
            ["metric", "result"],
            [
                ["right tools called", _n(rows, "tools_ok")],
                ["right tools with the expected arguments", _n(rows, "args_ok")],
                [
                    "numbers in first drafts that a tool returned",
                    f"{first_nums - first_bad} of {first_nums}",
                ],
                ["first drafts passing the checker", _n(rows, "first_try_passed")],
                [
                    "final answers by the LLM (rest: deterministic fallback)",
                    f"{len(llm)} of {len(rows)}",
                ],
                [
                    "numbers in final LLM answers that a tool returned",
                    f"{final_nums - final_bad} of {final_nums}",
                ],
                ["expected value restated", _n(rows, "has_value")],
                ["not-measured feature refused (said so, no value)", _n(rows, "refused_ok")],
                ["wording checks (heuristic phrase lists)", _n(rows, "mentions_ok")],
                ["model calls", str(sum(r["model_calls"] for r in rows))],
            ],
        ),  # fmt: skip
        "",
        "## Per question",
        "",
        *table(
            [
                "q",
                "kind",
                "tools called",
                "tools",
                "args",
                "first try",
                "value",
                "refusal",
                "wording",
                "by",
            ],
            [
                [
                    r["id"],
                    r["kind"],
                    r["tools_called"] or "–",
                    _yn(r["tools_ok"]),
                    _yn(r["args_ok"]),
                    _yn(r["first_try_passed"]),
                    _yn(r["has_value"]),
                    _yn(r["refused_ok"]),
                    _yn(r["mentions_ok"]),
                    r["generated_by"],
                ]
                for r in rows
            ],
        ),  # fmt: skip
        "",
    ]
    rejected = [r for r in rows if r["first_try_errors"]]
    if rejected:
        L += ["## What the checker rejected on the first try", ""]
        L += [f"- **{r['id']}**: {r['first_try_errors']}" for r in rejected]
        L.append("")
    L += [
        "## Caveats",
        "",
        f"- **n = {len(rows)} questions, one run each.** The model isn't deterministic; a rerun "
        "from an empty cache would give different answers. A run reuses any turn whose request "
        "is unchanged (e.g. the first tool choice), so runs aren't independent samples.",
        "- **Run 1 is kept** (`results/e9/run1/`): its checker had no units for history values "
        'and fault thresholds, so it rejected correct roundings and missed "-0.32 m" for a '
        "leg-length offset (DEC-043).",
        "- **Claude wrote the questions, the expectations and the agent.** The expectations "
        "were fixed before any run (sha256 above), but nobody independent wrote them.",
        "- **The wording checks are phrase lists**, not judgements of meaning. A correct answer "
        "in other words fails them; a wrong one with the right words passes.",
        "- **Grounding is checked by value, not by meaning:** a number a tool returned, used "
        "for the wrong thing, still counts as grounded.",
        "- **Fixture players,** not real users or real progressions.",
    ]
    return "\n".join(L) + "\n"


def _yn(v) -> str:
    return "–" if v == "" else ("yes" if v else "**no**")


def score() -> tuple[list[dict], str]:
    spec = load_spec()
    runs = json.loads(RUNS.read_text()) if RUNS.exists() else {}
    rows = [score_one(q, runs[q["id"]]) for q in spec["questions"] if q["id"] in runs]
    missing = [q["id"] for q in spec["questions"] if q["id"] not in runs]
    sha = hashlib.sha256(QUESTIONS.read_bytes()).hexdigest()
    return rows, report(rows, sha, missing)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="score", choices=["run", "score"])
    ap.add_argument("--only", nargs="*", help="question ids (run only)")
    args = ap.parse_args()
    if args.cmd == "run":
        runs = run(args.only)
        E9_DIR.mkdir(parents=True, exist_ok=True)
        RUNS.write_text(json.dumps(runs, indent=1, ensure_ascii=False))
    else:
        # Replay every question from the cache, so the committed runs.json is regenerated
        # from the committed model turns, not trusted as written.
        runs = run(api_key="")
        if runs:
            E9_DIR.mkdir(parents=True, exist_ok=True)
            RUNS.write_text(json.dumps(runs, indent=1, ensure_ascii=False))
    rows, md = score()
    write_csv(RESULTS / "e9.csv", rows)
    (RESULTS / "e9_summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
