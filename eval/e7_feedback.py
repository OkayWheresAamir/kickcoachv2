"""E7: is the feedback correct, specific and actionable, and can an LLM judge stand in for
Aamir? (eval plan E7; docs/PLAN.md Phase 8; DEC-040)

Every fault item the corpus produces exists in three variants: the LLM with retrieved
coaching passages (prompt v3, the app's default), the LLM without them (v2) and the
template. Aamir rates a fixed 20-item sample (1-5 on correctness, specificity,
actionability) without seeing which variant is which; an LLM judge rates every item with
the same facts and rubric. The judge is trusted beyond the sample only on criteria where
its agreement with Aamir clears the bar fixed before either rated (DEC-040).

    uv run python -m eval.e7_feedback prepare   # results/e7/items.json (frozen once built)
    uv run python tools/rate_feedback.py         # Aamir rates the sample
    uv run python -m eval.e7_feedback judge     # the judge; answers cached in results/e7/judge_raw/
    uv run python -m eval.e7_feedback           # score: results/e7.csv, results/e7_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import spearmanr

from eval.e1_pose import table, write_csv
from kickcoach.compare import engine, faults, norms
from kickcoach.compare.faults import load_rules
from kickcoach.feedback.generate import DEFINITIONS, SOURCES, build_context
from kickcoach.feedback.validate import validate
from kickcoach.llm import GeminiClient, NoApiKey
from kickcoach.llm.gemini import DEV_MODEL
from kickcoach.retrieval import corpus as coaching
from kickcoach.session import assess

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
E7_DIR = RESULTS / "e7"
ITEMS = E7_DIR / "items.json"
RATINGS = E7_DIR / "aamir_ratings.json"
# LLM raters of the 20-item sample, kept apart from Aamir's file (DEC-041): an AI persona
# panel Aamir supplied (5 simulated players and a coach, one external tool, model not
# recorded) and Claude's rule-based ratings (not blind). Neither is a human rating.
PANEL = E7_DIR / "raters" / "persona_panel.csv"
CLAUDE = E7_DIR / "raters" / "claude.json"
LLM_CACHE = RESULTS / "feedback_llm"
VARIANTS = ("llm_rag", "llm", "template")
CRITERIA = ("correctness", "specificity", "actionability")
SEED = 20261103
# The sample: 8 (clip, fault) pairs rated in both LLM variants, plus 4 templates, stratified
# by fault in proportion to how often each fires (lean-back 10, shallow backswing 5, plant
# foot 3 of 18 items; the shank-before-thigh rule never fires on these clips).
PAIRS = {"F-LEAN-BACK": 3, "F-SHALLOW-BACKSWING": 3, "F-PLANT-FAR-BEHIND": 2}
TEMPLATES = {"F-LEAN-BACK": 2, "F-SHALLOW-BACKSWING": 1, "F-PLANT-FAR-BEHIND": 1}
# heuristic: the judge stands in for Aamir on a criterion only if the quadratic weighted
# kappa between them is at least 0.6 (Landis & Koch 1977 call 0.61-0.80 "substantial";
# the cut-off itself is a convention, fixed here before any rating existed).
KAPPA_BAR = 0.6
# Not the generator's model, so it isn't grading its own wording. gemini-3.6-flash was the
# planned judge; its free tier allows 20 requests a day (the 429 on 3 Nov). On 3 Nov the judge
# moved to Gemma 4, which then stalled (500s, 503s, hangs) at 30 of 54. On 7 Nov Flash's next
# free day covered the rest of the 20-item sample, so Flash is the judge again and Gemma the
# second (DEC-040). No agreement with a human existed at either switch.
JUDGE_MODEL = "gemini-3.6-flash"
SECOND_JUDGE = "gemma-4-31b-it"
JUDGE_ROOT = E7_DIR / "judge_raw"
JUDGE_PROMPT_VERSION = "judge-v1"

RUBRIC = {
    "correctness": {
        "question": "Is everything it says true to the facts and the sources it cites?",
        5: "Every statement matches the measurements and the cited sources; nothing invented.",
        3: "Mostly right, but one claim is unsupported, overstated or slightly off.",
        1: "Contradicts a measurement (e.g. says too bent when the number says too straight) "
        "or invents facts or sources.",
    },
    "specificity": {
        "question": "Is it about this kick, not football in general?",
        5: "Built on this kick's measurement and fault; it couldn't be pasted onto another kick.",
        3: "Names the fault, but the rest would fit any player.",
        1: "Generic advice that ignores what was measured.",
    },
    "actionability": {
        "question": "Could the player act on the cue and drill at their next session?",
        5: "A clear cue and a drill you could set up and do alone, without asking anyone.",
        3: "Understandable, but vague about what to do or how to practise it.",
        1: "No usable instruction.",
    },
}


# -- items ----------------------------------------------------------------------------------
def _no_api_client() -> GeminiClient:
    return GeminiClient(DEV_MODEL, LLM_CACHE, api_key="")  # cache hits only


def _source_text(cid: str, rule, rows: dict, chunks: dict) -> dict:
    if cid in chunks:
        c = chunks[cid]
        return {"id": cid, "label": f"{c.cite}, {c.where}", "text": c.text}
    if cid in rows:
        r = rows[cid]
        s = r.stat
        stat = f"{s.mean} ± {s.sd} {r.unit}" if s.sd is not None else f"{s.mean} {r.unit}"
        return {"id": cid, "label": f"{r.source}, {r.page_or_table}",
                "text": f"{r.variable}: {stat} ({r.population.level or ''}, n = {r.n}). "
                        f"Paper's words: \"{r.quote}\""}  # fmt: skip
    text = rule.text_source if rule and cid in rule.citations else ""
    return {"id": cid, "label": SOURCES.get(cid, cid), "text": text}


def build_items() -> list[dict]:
    rules = {r.id: r for r in load_rules()}
    verified = norms.verified_norms()
    rows = {r.norm_id: r for r in verified}
    chunks = {c.id: c for c in coaching.load()}
    client = _no_api_client()
    items = []
    for p in sorted((RESULTS / "pipeline").glob("c*.json")):
        res = json.loads(p.read_text())
        if not res["quality"]["accepted"]:
            continue
        outs = {
            "llm_rag": assess(res, client=client, verified=verified, retrieval=True),
            "llm": assess(res, client=client, verified=verified, retrieval=False),
            "template": assess(res, use_llm=False, verified=verified),
        }
        base = outs["template"]
        measured = [
            {"id": f["id"], "value": f["value"], "unit": f["unit"],
             "definition": DEFINITIONS.get(f["id"])}
            for f in base["features"] if f.get("status") == "measured"
        ]  # fmt: skip
        not_measured = [f["id"] for f in base["features"] if f.get("status") != "measured"]
        for fault in base["faults"]:
            rule = rules[fault["fault_id"]]
            comp = [c for c in base["comparisons"] if c["feature_id"] == rule.feature]
            for variant, out in outs.items():
                fb = out["feedback"]
                (item,) = [i for i in fb["items"] if i["fault_id"] == rule.id]
                attempts = fb.get("attempts", [])
                items.append(
                    {"item_id": f"{res['clip_id']}-{rule.id}-{variant}",
                     "clip_id": res["clip_id"], "fault_id": rule.id, "variant": variant,
                     "generated_by": fb["generated_by"],
                     "first_try_passed": (not attempts[0]["errors"]) if attempts else None,
                     "first_try_errors": attempts[0]["errors"] if attempts else [],
                     "n_attempts": len(attempts),
                     "passages_given": (fb.get("passages") or {}).get(rule.id, []),
                     "facts": {
                         "fault": {"name": rule.name, "feature": rule.feature,
                                   "value": fault["evidence"][rule.feature],
                                   "unit": fault.get("unit"), "fires_when": fault["fires_when"],
                                   "threshold": fault.get("threshold"),
                                   "threshold_source": rule.threshold_source},
                         "measured": measured, "not_measured": not_measured,
                         "norm_comparison": [
                             {k: c.get(k) for k in ("verdict", "norm_ref", "norm_mean",
                                                    "norm_sd", "population")} for c in comp],
                     },
                     "feedback": {k: item[k] for k in ("what", "why_it_matters", "cue",
                                                       "drill")},
                     "cited": [_source_text(c, rule, rows, chunks) for c in item["citations"]]}
                )  # fmt: skip
    return items


def sample(items: list[dict]) -> list[str]:
    """The 20 item ids Aamir rates, in a fixed shuffled order."""
    rng = random.Random(SEED)
    llm_ok = {
        (i["clip_id"], i["fault_id"])
        for i in items
        if i["variant"] == "llm_rag" and i["generated_by"] == "llm"
    } & {(i["clip_id"], i["fault_id"]) for i in items
         if i["variant"] == "llm" and i["generated_by"] == "llm"}  # fmt: skip
    chosen, used = [], set()
    for fid, n in PAIRS.items():
        pool = sorted(p for p in llm_ok if p[1] == fid)
        for clip, f in rng.sample(pool, n):
            chosen += [f"{clip}-{f}-llm_rag", f"{clip}-{f}-llm"]
            used.add(clip)
    for fid, n in TEMPLATES.items():
        pool = sorted({i["clip_id"] for i in items if i["fault_id"] == fid} - used) or sorted(
            {i["clip_id"] for i in items if i["fault_id"] == fid}
        )
        chosen += [f"{c}-{fid}-template" for c in rng.sample(pool, n)]
    rng.shuffle(chosen)
    return chosen


def prepare(rebuild: bool = False) -> None:
    # The items are frozen once built: they hold the exact texts every rater scored (prompts
    # v2/v3, Phase 8). The prompt and validator have changed since (DEC-042), so rebuilding
    # now would put new texts under old ratings. --rebuild exists for a new round of rating.
    if ITEMS.exists() and not rebuild:
        data = json.loads(ITEMS.read_text())
        missing = set(data["sample"]) - {i["item_id"] for i in data["items"]}
        if missing:
            raise SystemExit(f"sampled items missing from {ITEMS}: {sorted(missing)}")
        print(f"{ITEMS} is frozen: {len(data['items'])} items, the texts that were rated")
        return
    items = build_items()
    E7_DIR.mkdir(parents=True, exist_ok=True)
    # The sample is drawn once and then frozen: rebuilding the items (e.g. after a late LLM
    # answer for one clip) must not change what Aamir has already rated.
    frozen = json.loads(ITEMS.read_text())["sample"] if ITEMS.exists() else None
    ids = {i["item_id"] for i in items}
    if frozen and not set(frozen) <= ids:
        raise SystemExit(f"sampled items no longer exist: {sorted(set(frozen) - ids)}")
    data = {"seed": SEED, "rubric": RUBRIC, "sample": frozen or sample(items), "items": items}
    ITEMS.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    by = {v: sum(i["variant"] == v for i in items) for v in VARIANTS}
    print(f"wrote {ITEMS}: {len(items)} items {by}; sample of {len(data['sample'])}")


def recheck(items: list[dict]) -> dict[str, list[str]]:
    """Today's validator on the frozen LLM items: {item_id: errors}. Its checks 6 and 7 were
    written from these ratings (DEC-042), so this shows what they catch on the rated texts.
    One call wrote all of a clip's items, so they are validated together, with the passages
    each was given."""
    verified = norms.verified_norms()
    chunks = {c.id: c for c in coaching.load()}
    groups: dict[tuple[str, str], list[dict]] = {}
    for it in items:
        if it["variant"] != "template" and it["generated_by"] == "llm":
            groups.setdefault((it["clip_id"], it["variant"]), []).append(it)
    out = {}
    for (clip, variant), its in sorted(groups.items()):
        res = json.loads((RESULTS / "pipeline" / f"{clip}.json").read_text())
        comps = engine.compare(res["features"], verified, None)
        fired, _ = faults.evaluate(res["features"], verified)
        passages = {it["fault_id"]: [chunks[c] for c in it["passages_given"]] for it in its}
        ctx, _ = build_context(res["features"], fired, comps, verified,
                               passages if variant == "llm_rag" else None)  # fmt: skip
        raw = {"items": [{"fault_id": it["fault_id"], **it["feedback"],
                          "citations": [c["id"] for c in it["cited"]]} for it in its]}  # fmt: skip
        errors = validate(raw, ctx)[1]
        for it in its:
            out[it["item_id"]] = [e for e in errors if e.startswith(it["fault_id"])]
    return out


# -- judge ----------------------------------------------------------------------------------
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        **{c: {"type": "integer", "minimum": 1, "maximum": 5} for c in CRITERIA},
        "reason": {"type": "string"},
    },
    "required": [*CRITERIA, "reason"],
}


def judge_prompt(item: dict, rubric: dict = RUBRIC) -> str:
    view = {"facts": item["facts"], "feedback": item["feedback"], "cited_sources": item["cited"]}
    anchors = "\n".join(
        f"- {c}: {r['question']} 5 = {r[5]} 3 = {r[3]} 1 = {r[1]} (2 and 4 are in between.)"
        for c, r in rubric.items()
    )
    return f"""\
You are checking coaching feedback written about one football (soccer) instep kick. A
separate system measured the kick from video; "facts" is its output. "feedback" was written
from those facts, and "cited_sources" are the sources the feedback cites (empty if none).

Rate the feedback from 1 to 5 on each criterion:
{anchors}

Judge only what is shown. A claim that no fact or cited source supports counts against
correctness even if it is common coaching advice. Give a one-sentence reason.

{json.dumps(view, indent=1, ensure_ascii=False)}
"""


def parse_judge(text: str) -> dict | None:
    """The judge's scores, or None. Gemma 4 ignores the JSON mime type enough to add a stray
    code fence after the object, so the first {...} block is parsed, then checked."""
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        d = json.loads(m.group()) if m else None
    except json.JSONDecodeError:
        return None
    ok = isinstance(d, dict) and all(d.get(c) in (1, 2, 3, 4, 5) for c in CRITERIA)
    return d if ok else None


def run_judge(
    items: list[dict], client: GeminiClient | None = None, model: str = JUDGE_MODEL
) -> dict[str, dict]:
    client = client or GeminiClient(model, JUDGE_ROOT / model)
    out = {}
    for i, item in enumerate(items, 1):
        res = client.generate_json(
            prompt=judge_prompt(item), schema=JUDGE_SCHEMA, temperature=0.0,
            tag=f"{JUDGE_PROMPT_VERSION} {item['item_id']}",
        )  # fmt: skip
        out[item["item_id"]] = parse_judge(res["text"])
        print(f"{i}/{len(items)} {item['item_id']}: {out[item['item_id']]}", flush=True)
    return out


def cached_judge(items: list[dict], model: str = JUDGE_MODEL) -> dict[str, dict]:
    client = GeminiClient(model, JUDGE_ROOT / model, api_key="")
    out = {}
    for item in items:
        rec = client.cached(prompt=judge_prompt(item), schema=JUDGE_SCHEMA, temperature=0.0)
        if rec is not None and (parsed := parse_judge(rec["text"])):
            out[item["item_id"]] = parsed
    return out


# -- scoring --------------------------------------------------------------------------------
def weighted_kappa(a: list[int], b: list[int], k: int = 5) -> float:
    """Cohen's kappa with quadratic weights on a 1..k scale."""
    o = np.zeros((k, k))
    for x, y in zip(a, b, strict=True):
        o[x - 1, y - 1] += 1
    o /= o.sum()
    e = np.outer(o.sum(1), o.sum(0))
    i, j = np.indices((k, k))
    w = (i - j) ** 2 / (k - 1) ** 2
    den = (w * e).sum()
    return float("nan") if den == 0 else 1 - (w * o).sum() / den


def agreement(pairs: list[tuple[int, int]]) -> dict:
    a, b = [x for x, _ in pairs], [y for _, y in pairs]
    rho = spearmanr(a, b).statistic if len(set(a)) > 1 and len(set(b)) > 1 else float("nan")
    return {
        "n": len(pairs),
        "kappa_w": weighted_kappa(a, b),
        "spearman": float(rho),
        "exact": sum(x == y for x, y in pairs) / len(pairs),
        "within_1": sum(abs(x - y) <= 1 for x, y in pairs) / len(pairs),
        "mean_aamir": mean(a),
        "mean_judge": mean(b),
    }


def llm_raters(sample: list[str]) -> tuple[dict, dict, dict]:
    """Claude's ratings, the panel's per-item median, and every persona's, keyed by item id."""
    key_to_id = {f"R{n:02d}": iid for n, iid in enumerate(sample, 1)}
    claude = {}
    if CLAUDE.exists():
        claude = {key_to_id[k]: v for k, v in json.loads(CLAUDE.read_text())["ratings"].items()}
    personas: dict[str, dict] = {}
    if PANEL.exists():
        with open(PANEL, newline="") as f:
            for r in csv.DictReader(f):
                personas.setdefault(r["Persona"], {})[key_to_id[r["Item"]]] = {
                    c: int(r[c.capitalize()]) for c in CRITERIA
                }
    panel = {
        iid: {c: int(np.median([p[iid][c] for p in personas.values()])) for c in CRITERIA}
        for iid in sample
        if personas and all(iid in p for p in personas.values())
    }
    return claude, panel, personas


def score() -> tuple[list[dict], str]:
    data = json.loads(ITEMS.read_text())
    items = {i["item_id"]: i for i in data["items"]}
    claude, panel, _ = llm_raters(data["sample"])
    ratings = json.loads(RATINGS.read_text()) if RATINGS.exists() else {}
    rated = {k: v for k, v in ratings.items() if all(v.get(c) for c in CRITERIA)}
    judged = cached_judge(list(items.values()))
    second = cached_judge(list(items.values()), SECOND_JUDGE)
    now = recheck(list(items.values()))
    rows = []
    for iid, it in items.items():
        r, j, j2 = rated.get(iid, {}), judged.get(iid) or {}, second.get(iid) or {}
        cl, pa = claude.get(iid, {}), panel.get(iid, {})
        rows.append(
            {"item_id": iid, "clip_id": it["clip_id"], "fault_id": it["fault_id"],
             "variant": it["variant"], "generated_by": it["generated_by"],
             "in_sample": iid in data["sample"], "first_try_passed": it["first_try_passed"],
             "n_passages_given": len(it["passages_given"]),
             "n_passages_cited": sum(c["id"].startswith("K-") for c in it["cited"]),
             **{f"aamir_{c}": r.get(c) for c in CRITERIA},
             **{f"judge_{c}": j.get(c) for c in CRITERIA},
             **{f"judge2_{c}": j2.get(c) for c in CRITERIA},
             **{f"claude_{c}": cl.get(c) for c in CRITERIA},
             **{f"panel_{c}": pa.get(c) for c in CRITERIA},
             "validator_now": " | ".join(now.get(iid, [])),
             "judge_reason": j.get("reason", ""), "aamir_note": r.get("note", "")}
        )  # fmt: skip
    return rows, report(rows, data)


RATERS = {  # column prefix -> how the report names it
    "claude": "Claude (rule-based, not blind)",
    "panel": "AI persona panel (median of 5)",
    "judge": JUDGE_MODEL,
    "judge2": SECOND_JUDGE,
}


def llm_rater_section(rows: list[dict]) -> list[str]:
    s = [r for r in rows if r["in_sample"]]
    present = [k for k in RATERS if any(r[f"{k}_correctness"] for r in s)]
    if not present:
        return []
    L = [
        f"## LLM raters on the {len(s)}-item sample (DEC-041)",
        "",
        "None of these is a human. Claude's ratings follow written rules with a reason per "
        "item (`results/e7/raters/claude.json`); the persona panel came from one external AI "
        "tool playing five players and a coach (`results/e7/raters/persona_panel.csv`).",
        "",
        "Mean rating (correctness / specificity / actionability), by variant:",
        "",
    ]
    body = []
    for k in present:
        cells = []
        for v in VARIANTS:
            vr = [r for r in s if r["variant"] == v and r[f"{k}_correctness"]]
            cells.append(
                " / ".join(_f(mean(r[f"{k}_{c}"] for r in vr), 1) for c in CRITERIA) +
                f" (n {len(vr)})" if vr else "–"
            )  # fmt: skip
        body.append([RATERS[k], *cells])
    L += [*table(["rater", *VARIANTS], body), ""]
    # The 8 pairs: the same clip and fault, written with and without retrieved passages.
    pairs = {}
    for r in s:
        if r["variant"] in ("llm_rag", "llm"):
            pairs.setdefault((r["clip_id"], r["fault_id"]), {})[r["variant"]] = r
    pairs = {p: v for p, v in pairs.items() if len(v) == 2}
    body = []
    for k in present:
        both = [v for v in pairs.values() if v["llm_rag"][f"{k}_correctness"]
                and v["llm"][f"{k}_correctness"]]  # fmt: skip
        if both:
            diffs = {c: [v["llm_rag"][f"{k}_{c}"] - v["llm"][f"{k}_{c}"] for v in both]
                     for c in CRITERIA}  # fmt: skip
            body.append([RATERS[k], str(len(both)), *(
                f"{mean(d):+.2f} (better {sum(x > 0 for x in d)}, worse {sum(x < 0 for x in d)})"
                for d in diffs.values())])  # fmt: skip
    if body:
        L += [
            "With passages minus without, on the same clip and fault (positive = passages "
            "rated higher):",
            "",
            *table(["rater", "pairs", *CRITERIA], body),
            "",
        ]
    body = []
    for i, a in enumerate(present):
        for b in present[i + 1 :]:
            both = [r for r in s if r[f"{a}_correctness"] and r[f"{b}_correctness"]]
            if len(both) >= 5:
                body.append([f"{RATERS[a]} vs {RATERS[b]}", str(len(both)), *(
                    _f(agreement([(r[f"{a}_{c}"], r[f"{b}_{c}"]) for r in both])["kappa_w"])
                    for c in CRITERIA)])  # fmt: skip
    if body:
        L += [
            "Agreement between raters (weighted κ):",
            "",
            *table(["raters", "items", *CRITERIA], body),
            "",
        ]
    return L


def recheck_section(rows: list[dict], sample: list[str]) -> list[str]:
    llm = [r for r in rows if r["variant"] != "template" and r["generated_by"] == "llm"]
    bad = [r for r in llm if r["validator_now"]]
    key = {iid: f"R{n:02d}" for n, iid in enumerate(sample, 1)}
    L = [
        "## Today's validator on the rated texts",
        "",
        "The ratings found three slips the Phase 8 validator passed: a knee angle called "
        '"bend" (R02, R13) and a lean the kick didn\'t have (R11). Phase 9 added a check for '
        "each kind (`kickcoach/feedback/validate.py` checks 6 and 7, DEC-042). Run on the frozen "
        f"LLM items, today's validator rejects {len(bad)} of {len(llm)}; the app now retries "
        "such an answer with the error, then falls back to the template:",
        "",
    ]
    body = [[key.get(r["item_id"], "–"), r["item_id"], r["validator_now"]] for r in bad]
    return L + [*table(["sample", "item", "error"], body), ""] if bad else L


def _f(x, dp: int = 2) -> str:
    return "–" if x is None or x != x else f"{x:.{dp}f}"


def report(rows: list[dict], data: dict) -> str:
    sample_rows = [r for r in rows if r["in_sample"] and r["aamir_correctness"]]
    both = [r for r in sample_rows if r["judge_correctness"]]
    agree = {
        c: agreement([(r[f"aamir_{c}"], r[f"judge_{c}"]) for r in both]) for c in CRITERIA
    } if both else {}  # fmt: skip
    trusted = [c for c in agree if agree[c]["kappa_w"] >= KAPPA_BAR]
    L = [
        "# E7: feedback quality: human, LLM-rater and LLM-judge ratings",
        "",
        f"Items: every fault item from the {len({r['clip_id'] for r in rows})} accepted clips "
        "with a fault, in three variants: LLM with retrieved passages (prompt v3, the app's "
        "default), LLM without (v2), and the template. Generator: `gemini-3.5-flash-lite`; "
        f"judge: `{JUDGE_MODEL}` (prompt {JUDGE_PROMPT_VERSION}). Rubric 1–5 on correctness, "
        "specificity, actionability (`results/e7/items.json`). Aamir rates a fixed sample of "
        f"{len(data['sample'])} without seeing the variant.",
        "",
        "## Validator, first try",
        "",
        "Counted per item; one LLM call writes all of a clip's items, so one rejected call "
        "fails every item of that clip.",
        "",
        *table(
            ["variant", "items", "LLM answered", "first-try pass", "fell back to template"],
            [
                [
                    v,
                    str(len(vr := [r for r in rows if r["variant"] == v])),
                    str(sum(r["generated_by"] == "llm" for r in vr)),
                    (
                        f"{sum(r['first_try_passed'] is True for r in vr)} of "
                        f"{sum(r['first_try_passed'] is not None for r in vr)}"
                    )
                    if v != "template"
                    else "–",
                    str(sum(r["generated_by"] == "template" for r in vr))
                    if v != "template"
                    else "–",
                ]
                for v in VARIANTS
            ],
        ),  # fmt: skip
        "",
    ]
    rag = [r for r in rows if r["variant"] == "llm_rag" and r["generated_by"] == "llm"]
    if rag:
        L += [
            f"Retrieved passages: {mean(r['n_passages_given'] for r in rag):.1f} per item on "
            f"average; {sum(r['n_passages_cited'] > 0 for r in rag)} of {len(rag)} RAG items "
            "cite at least one.",
            "",
        ]
    L += [f"## Agreement between Aamir and the judge ({len(both)} items)", ""]
    if both:
        L += [
            *table(
                [
                    "criterion",
                    "weighted κ",
                    "Spearman ρ",
                    "exact",
                    "within 1",
                    "mean Aamir",
                    "mean judge",
                    f"judge used (κ ≥ {KAPPA_BAR})",
                ],
                [
                    [
                        c,
                        _f(a["kappa_w"]),
                        _f(a["spearman"]),
                        f"{a['exact']:.0%}",
                        f"{a['within_1']:.0%}",
                        _f(a["mean_aamir"]),
                        _f(a["mean_judge"]),
                        "yes" if c in trusted else "no",
                    ]
                    for c, a in agree.items()
                ],
            ),  # fmt: skip
            "",
        ]
    elif not sample_rows:
        L += [
            "**No human rating exists.** The sample wasn't rated by a human; an AI persona "
            "panel was supplied instead (7 Nov). So the bar above can't be applied, the judge is "
            "not validated, and it is not used beyond the sample (DEC-040, DEC-041). The "
            "section below compares LLM raters with each other, which shows consistency, not "
            "correctness.",
            "",
        ]
    else:
        L += ["Not computed yet: needs the judge's answers on the rated items.", ""]
    L += llm_rater_section(rows)
    if sample_rows:
        L += [
            "## Aamir's ratings by variant (the sample)",
            "",
            *table(
                ["variant", "n", *CRITERIA],
                [
                    [
                        v,
                        str(len(vr := [r for r in sample_rows if r["variant"] == v])),
                        *(_f(mean(r[f"aamir_{c}"] for r in vr)) if vr else "–" for c in CRITERIA),
                    ]
                    for v in VARIANTS
                ],
            ),  # fmt: skip
            "",
        ]
    judged = [r for r in rows if r["judge_correctness"]]
    if trusted and judged:
        L += [
            "## The judge on every item (trusted criteria only)",
            "",
            *table(
                ["variant", "n", *trusted],
                [
                    [
                        v,
                        str(len(vr := [r for r in judged if r["variant"] == v])),
                        *(_f(mean(r[f"judge_{c}"] for r in vr)) for c in trusted),
                    ]
                    for v in VARIANTS
                ],
            ),  # fmt: skip
            "",
        ]
    two = [r for r in rows if r["judge_correctness"] and r["judge2_correctness"]]
    if two:
        L += [
            f"## The two judges compared ({len(two)} items both rated)",
            "",
            f"`{JUDGE_MODEL}` is the judge; `{SECOND_JUDGE}` stood in while Flash's free "
            "quota (20 requests a day) was used up (DEC-040). Where both rated:",
            "",
            *table(
                [
                    "criterion",
                    "weighted κ",
                    "exact",
                    "within 1",
                    f"mean {JUDGE_MODEL}",
                    f"mean {SECOND_JUDGE}",
                ],
                [
                    [
                        c,
                        _f(
                            (a := agreement([(r[f"judge_{c}"], r[f"judge2_{c}"]) for r in two]))[
                                "kappa_w"
                            ]
                        ),
                        f"{a['exact']:.0%}",
                        f"{a['within_1']:.0%}",
                        _f(a["mean_aamir"]),
                        _f(a["mean_judge"]),
                    ]
                    for c in CRITERIA
                ],
            ),  # fmt: skip
            "",
        ]
    L += recheck_section(rows, data["sample"])
    L += [
        "## Caveats",
        "",
        "- **No human rated:** every score here comes from an LLM, including Claude's, and "
        "Claude also built the pipeline being rated. Agreement between LLM raters is "
        "consistency, not validity.",
        "- **Small n:** 20 items, 8 pairs.",
        "- **Blinding is partial:** the variant is hidden, but each fault's template has fixed "
        "wording that repeats across clips.",
        "- **No item for the shank-before-thigh rule:** it never fires on these clips.",
    ]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="score", choices=["prepare", "judge", "score"])
    ap.add_argument("--model", default=JUDGE_MODEL, help="judge model (judge only)")
    ap.add_argument(
        "--rebuild",
        action="store_true",
        help="prepare: rebuild the frozen items from today's code (new texts)",
    )
    ap.add_argument("--sample-only", action="store_true",
                    help="judge only the 20 sampled items (fits a Flash free day)")  # fmt: skip
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(args.rebuild)
        return
    if args.cmd == "judge":
        data = json.loads(ITEMS.read_text())
        by_id = {i["item_id"]: i for i in data["items"]}
        # Sampled items first: they are the ones every rater shares.
        order = [by_id[i] for i in data["sample"]]
        if not args.sample_only:
            order += [i for i in data["items"] if i["item_id"] not in set(data["sample"])]
        try:
            run_judge(order, model=args.model)
        except NoApiKey:
            raise SystemExit("the judge needs GEMINI_API_KEY") from None
    rows, md = score()
    write_csv(RESULTS / "e7.csv", rows)
    (RESULTS / "e7_summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
