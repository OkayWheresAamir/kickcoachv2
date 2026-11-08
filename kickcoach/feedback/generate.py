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
from kickcoach.feedback.validate import (
    DIRECTION_CLAIMS,
    KNEE_ANGLE_FEATURES,
    Context,
    validate,
)
from kickcoach.llm import GeminiClient, NoApiKey, RateLimited
from kickcoach.llm.gemini import DEV_MODEL
from kickcoach.retrieval import corpus as coaching
from kickcoach.retrieval.embed import Embedder
from kickcoach.retrieval.retrieve import NEEDS, retrieve

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache" / "feedback"
PROMPT_VERSION = "feedback-v2"  # v2: each measured feature comes with its definition
# With retrieved coaching passages (DEC-039). v3 (Phase 8) added the passages; v4 (Phase 9)
# adds rules 6-8 after E7: v3 dropped the measured value from "what" in 7 of 8 rated pairs,
# and E7 found a knee angle called "bend" and a lean the kick didn't have (DEC-042). E7's
# items keep the v3 answers they were rated on (results/e7/items.json is frozen). Without
# passages the prompt is v2 word for word, so v2's cached answers stay valid.
RAG_PROMPT_VERSION = "feedback-v4"
# heuristic: passages per need (why / cue / drill) per fault, deduplicated; E6's MRR of 0.75
# says the first is usually relevant, the second gives the wording some choice.
PASSAGES_PER_NEED = 2
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
# Each feature's unit, as kickcoach.features.kick reports it (a test checks the committed
# pipeline outputs). The chat states units from here and checks them (DEC-043).
FEATURE_UNITS = {
    "knee_angle_contact": "deg",
    "max_knee_flexion_backswing": "deg",
    "trunk_lean_contact": "deg",
    "plant_foot_offset": "leg_lengths",
    "segment_sequence_order": "order",
    "peak_knee_ext_velocity": "deg/s",
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
    features: list[dict],
    fired: list[dict],
    comparisons: list[dict],
    norms: list[NormRow],
    passages: dict[str, list[coaching.Chunk]] | None = None,
) -> tuple[Context, dict]:
    """The validator's context and the citable sources, from exactly what the LLM is shown."""
    norm_by_id = {r.norm_id: r for r in norms}
    cites: dict[str, str] = {}
    for f in fired:
        for c in f.get("citations", []):
            cites[c] = SOURCES.get(c) or _norm_line(norm_by_id.get(c))
    for chunks in (passages or {}).values():
        for ch in chunks:
            cites[ch.id] = f"{ch.cite}, {ch.where}"
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
            if f["id"] in DIRECTION_CLAIMS:
                ctx.signed[f["id"]] = float(f["value"])
            if f["id"] in KNEE_ANGLE_FEATURES:
                ctx.knee_angles.append(float(f["value"]))
    for f in fired:
        if isinstance(f.get("threshold"), float | int):
            ctx.allow(f["threshold"], f.get("unit"))
            if set(f["evidence"]) & set(KNEE_ANGLE_FEATURES):
                ctx.knee_angles.append(float(f["threshold"]))
    for c in comparisons:
        for k in ("norm_mean", "norm_sd"):
            if c.get(k) is not None:
                ctx.allow(c[k], c.get("unit"))
        for x in c.get("norm_band") or []:
            ctx.allow(x, c.get("unit"))
        if c.get("feature_id") in KNEE_ANGLE_FEATURES:  # the SD is a spread, not an angle
            ctx.knee_angles += [x for x in (c.get("norm_mean"), *(c.get("norm_band") or []))
                                if x is not None]  # fmt: skip
    for text in cites.values():
        for y in _years(text):
            ctx.allow(y, "year")
    return ctx, cites


def _norm_line(row: NormRow | None) -> str:
    if row is None:
        return "unknown"
    return f"{row.source}, {row.page_or_table}: {row.variable} ({row.population.level or ''})"


_RULES_V2 = """
5. "what": what happened in this kick, in plain words. "why_it_matters": why it matters for
   the kick. "cue": one short coaching cue. "drill": one practice drill, described in words.
6. Short sentences a teenager would understand. No medical advice."""
_RULES_V4 = """
5. "coaching_passages" are short excerpts from coaching sources, found for each fault by a
   search; some may not fit. Use a passage only where it really supports what you write,
   say it in your own words, and add its ID to that item's citations. Copy no numbers or
   names from a passage. A field no passage supports is written without a citation.
6. "what": what happened in this kick, in plain words, with the fault's measured value from
   its "evidence" (rounded as rule 2 allows) and its unit, and which way it points, read
   from the feature's definition. "why_it_matters": why it matters for the kick. "cue": one
   short coaching cue. "drill": one practice drill, described in words.
7. Knee values are interior angles between thigh and shin (180 = straight). Call a knee
   number an angle, never an amount of "bend" or "flexion".
8. Describe this kick only as its measurements show it: don't say the player leaned back or
   planted behind the ball unless that feature's value says so.
9. Short sentences a teenager would understand. No medical advice."""


def build_prompt(facts: dict, previous: str | None = None, errors: list[str] | None = None) -> str:
    last_rules = _RULES_V4 if facts.get("coaching_passages") else _RULES_V2
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
   source. An empty list is fine.{last_rules}

Facts (JSON):
{json.dumps(facts, indent=1)}{retry}
"""


def retrieve_passages(
    fired: list[dict], rules: list[FaultRule], embedder: Embedder | None = None
) -> dict[str, list[coaching.Chunk]]:
    """For each fired fault, the top passages for each need, deduplicated in rank order. Every
    (rule, need) query is embedded in the committed cache (a test checks), so this needs no
    key; if a query were missing and there were no key, feedback goes on without passages."""
    by_id = {r.id: r for r in rules}
    chunks, embedder = coaching.load(), embedder or Embedder()
    out = {}
    try:
        for f in fired:
            rule = by_id[f["fault_id"]]
            found: dict[str, coaching.Chunk] = {}
            for need in NEEDS:
                for c, _ in retrieve(chunks, rule.id, rule.name, need, embedder,
                                     k=PASSAGES_PER_NEED):  # fmt: skip
                    found.setdefault(c.id, c)
            out[rule.id] = list(found.values())
    except NoApiKey:
        return {}
    return out


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
    retrieval: bool = False,
    embedder: Embedder | None = None,
) -> dict:
    """The MVP response's ``feedback`` block, plus ``attempts`` and ``fallback_reason``.
    With ``retrieval``, coaching passages for each fault go to the LLM (prompt v3) and the
    block gains ``passages`` (IDs per fault) and ``sources`` (what each passage ID cites)."""
    rules = load_rules() if rules is None else rules
    out = {"items": [], "generated_by": "template", "validator_passed": None,
           "attempts": [], "fallback_reason": None}  # fmt: skip
    if not fired:
        out["fallback_reason"] = "no fault rule fired, so there is nothing to explain"
        return out

    passages = retrieve_passages(fired, rules, embedder) if retrieval and use_llm else {}
    if passages:
        out["passages"] = {fid: [c.id for c in cs] for fid, cs in passages.items()}
        out["sources"] = {c.id: {"cite": c.cite, "where": c.where, "url": c.url}
                          for cs in passages.values() for c in cs}  # fmt: skip
    ctx, cites = build_context(features, fired, comparisons, norms, passages)
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
    if passages:
        facts["coaching_passages"] = [
            {"id": c.id, "for_fault": fid, "text": c.text}
            for fid, cs in passages.items() for c in cs
        ]  # fmt: skip
    version = RAG_PROMPT_VERSION if passages else PROMPT_VERSION
    client = client or GeminiClient(DEV_MODEL, CACHE_DIR)
    previous, errors = None, None
    for attempt in (1, 2):
        try:
            res = client.generate_json(
                prompt=build_prompt(facts, previous, errors),
                schema=RESPONSE_SCHEMA,
                temperature=None,
                tag=f"{version} attempt {attempt}",
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
