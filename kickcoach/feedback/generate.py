"""Feedback: structured LLM output -> validator -> one retry -> template (handoff D13).

The LLM only words feedback about faults the deterministic rules already found, from facts
it is handed; it never measures (CLAUDE.md rule 3). Without an API key, or if both attempts
fail the validator, the template from knowledge/faults.yaml is used. Both attempts are kept
in ``attempts`` so a failure can be inspected later.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from kickcoach.compare.faults import FaultRule, load_rules
from kickcoach.compare.norms import NormRow
from kickcoach.feedback.schema import RESPONSE_SCHEMA, Feedback, FeedbackItem
from kickcoach.feedback.validate import Context, validate
from kickcoach.llm import GeminiClient, NoApiKey, RateLimited
from kickcoach.llm.gemini import DEV_MODEL

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache" / "feedback"
PROMPT_VERSION = "feedback-v2"  # v2: each measured feature comes with its definition
SOURCES = {  # knowledge/sources.md keys a fault rule may cite
    "kellis_katis_2007": "Kellis & Katis (2007), J Sports Sci Med 6(2):154-165, a review",
}
# What each number means, so the wording can't flip a convention (v1 called a 98 deg interior
# knee angle "98 degrees of knee flexion", i.e. 82 deg in flexion terms).
DEFINITIONS = {
    "knee_angle_contact": "interior angle between thigh and shin at ball contact; 180 = straight",
    "max_knee_flexion_backswing": (
        "the smallest interior knee angle in the backswing (most bent); 180 = straight, so a "
        "larger number means a less bent knee"
    ),
    "trunk_lean_contact": "trunk angle from vertical at contact; negative = leaning back",
    "plant_foot_offset": (
        "plant ankle's horizontal distance from the ball centre, in leg lengths; negative = "
        "behind the ball"
    ),
    "segment_sequence_order": "whether the thigh's speed peaked before the shank's",
    "peak_knee_ext_velocity": "fastest knee straightening speed before contact",
}
# Decimal places when a value is shown in text (templates and the facts handed to the LLM).
DISPLAY_DP = {"deg": 0, "deg/s": 0, "leg_lengths": 2, "ms": 0}


def display(value: float, unit: str | None) -> str:
    return f"{abs(value):.{DISPLAY_DP.get(unit or '', 2)}f}"


def template_feedback(fired: list[dict], rules: list[FaultRule]) -> Feedback:
    by_id = {r.id: r for r in rules}
    items = []
    for f in fired:
        rule = by_id[f["fault_id"]]
        (value,) = f["evidence"].values()
        shown = display(value, rule.unit) if isinstance(value, float | int) else str(value)
        t = {k: v.format(value=shown) for k, v in rule.template.items()}
        items.append(FeedbackItem(fault_id=rule.id, citations=list(rule.citations), **t))
    return Feedback(items=items)


def _years(text: str) -> list[int]:
    return [int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", text)]


def build_context(
    features: list[dict], fired: list[dict], comparisons: list[dict], norms: list[NormRow]
) -> tuple[Context, dict]:
    """The validator's context and the citable sources, from exactly what the LLM is shown."""
    norm_by_id = {r.norm_id: r for r in norms}
    cites: dict[str, str] = {}
    for f in fired:
        for c in f.get("citations", []):
            cites[c] = SOURCES.get(c) or _norm_line(norm_by_id.get(c))
    for c in comparisons:
        if c.get("norm_ref"):
            cites[c["norm_ref"]] = _norm_line(norm_by_id.get(c["norm_ref"]))
    ctx = Context(
        fault_ids=[f["fault_id"] for f in fired],
        citation_ids=set(cites),
        not_measured=[f["id"] for f in features if f.get("status") != "measured"],
    )
    for f in features:
        if f.get("status") == "measured" and isinstance(f.get("value"), float | int):
            ctx.allow(f["value"], f.get("unit"))
    for f in fired:
        if isinstance(f.get("threshold"), float | int):
            ctx.allow(f["threshold"], f.get("unit"))
    for c in comparisons:
        for k in ("norm_mean", "norm_sd"):
            if c.get(k) is not None:
                ctx.allow(c[k], c.get("unit"))
        for x in c.get("norm_band") or []:
            ctx.allow(x, c.get("unit"))
    for text in cites.values():
        for y in _years(text):
            ctx.allow(y, "year")
    return ctx, cites


def _norm_line(row: NormRow | None) -> str:
    if row is None:
        return "unknown"
    return f"{row.source}, {row.page_or_table}: {row.variable} ({row.population.level or ''})"


def build_prompt(facts: dict, previous: str | None = None, errors: list[str] | None = None) -> str:
    retry = ""
    if previous is not None:
        retry = (
            "\n\nYour previous answer was rejected by an automatic checker:\n"
            + "\n".join(f"- {e}" for e in errors or [])
            + f"\n\nPrevious answer:\n{previous}\n\nWrite a corrected answer."
        )
    return f"""\
You are a football (soccer) coach writing feedback on one instep kick, for a player whose
level is: {facts["player_level"] or "not stated"}. The kick was measured from video by a
separate system; the facts below are its output. You write words only.

Rules:
1. Write exactly one item per fault in "faults", with that fault_id.
2. Use only these facts. If you mention a number, copy it from the facts (you may round an
   angle to whole degrees). Write no other numbers at all: no rep counts, distances,
   percentages, times or list numbering.
3. Never mention a feature listed in "not_measured", or anything it describes.
4. "citations" may contain only IDs from "citable_sources", and only when the item uses that
   source. An empty list is fine.
5. "what": what happened in this kick, in plain words. "why_it_matters": why it matters for
   the kick. "cue": one short coaching cue. "drill": one practice drill, described in words.
6. Short sentences a teenager would understand. No medical advice.

Facts (JSON):
{json.dumps(facts, indent=1)}{retry}
"""


def generate_feedback(
    features: list[dict],
    fired: list[dict],
    comparisons: list[dict],
    norms: list[NormRow],
    *,
    level: str | None = None,
    client: GeminiClient | None = None,
    use_llm: bool = True,
    rules: list[FaultRule] | None = None,
) -> dict:
    """The MVP response's ``feedback`` block, plus ``attempts`` and ``fallback_reason``."""
    rules = load_rules() if rules is None else rules
    out = {"items": [], "generated_by": "template", "validator_passed": None,
           "attempts": [], "fallback_reason": None}  # fmt: skip
    if not fired:
        out["fallback_reason"] = "no fault rule fired, so there is nothing to explain"
        return out

    ctx, cites = build_context(features, fired, comparisons, norms)
    template = template_feedback(fired, rules)
    template_errors = validate(template.model_dump(), ctx)[1]

    def fallback(reason: str) -> dict:
        out.update(
            items=[i.model_dump() for i in template.items],
            generated_by="template",
            validator_passed=not template_errors,
            fallback_reason=reason,
        )
        if template_errors:
            out["template_errors"] = template_errors
        return out

    if not use_llm:
        return fallback("LLM feedback switched off")
    facts = {
        "player_level": level,
        "measured": [
            {"id": f["id"], "value": f["value"], "unit": f["unit"],
             "definition": DEFINITIONS.get(f["id"])}
            for f in features if f.get("status") == "measured"
        ],
        "not_measured": [{"id": f["id"], "reason": f.get("reason")} for f in features
                         if f.get("status") != "measured"],
        "faults": [
            {k: f[k] for k in ("fault_id", "name", "evidence", "unit", "fires_when",
                               "threshold", "threshold_source", "citations")}
            for f in fired
        ],
        "norm_comparisons": [
            {k: c.get(k) for k in ("feature_id", "verdict", "norm_ref", "norm_mean", "norm_sd",
                                   "norm_band", "population", "caveats")}
            for c in comparisons if c.get("norm_ref")
        ],
        "citable_sources": cites,
    }  # fmt: skip
    client = client or GeminiClient(DEV_MODEL, CACHE_DIR)
    previous, errors = None, None
    for attempt in (1, 2):
        try:
            res = client.generate_json(
                prompt=build_prompt(facts, previous, errors),
                schema=RESPONSE_SCHEMA,
                temperature=None,
                tag=f"{PROMPT_VERSION} attempt {attempt}",
            )
        except NoApiKey:
            return fallback("no GEMINI_API_KEY: template feedback (the app runs without one)")
        except RateLimited as e:
            return fallback(f"Gemini unavailable ({type(e).__name__}); template feedback")
        fb, errors = validate(res["parsed"], ctx)
        out["attempts"].append(
            {"attempt": attempt, "cache_key": res["key"], "cached": res["cached"],
             "text": res["text"], "errors": errors}
        )  # fmt: skip
        if fb is not None and not errors:
            out.update(
                items=[i.model_dump() for i in fb.items],
                generated_by="llm",
                validator_passed=True,
                fallback_reason=None,
            )
            return out
        previous = res["text"]
    return fallback("both LLM attempts failed the validator; template feedback")
