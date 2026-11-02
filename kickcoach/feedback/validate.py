"""The deterministic feedback validator (handoff D13). An LLM answer passes only if:

1. it parses against the schema (kickcoach.feedback.schema);
2. its items are exactly the detected faults, one each;
3. every citation is an ID it was given;
4. every number in its text matches a provided number within tolerance;
5. it says nothing about a feature that is ``not_measured``.

The validator never edits the answer. It returns the errors, which go back to the LLM once;
then the app falls back to the template (kickcoach.feedback.generate).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from kickcoach.feedback.schema import TEXT_FIELDS, Feedback

# heuristic: how far a number in the text may be from the value it restates. Whole-degree
# rounding of an angle must pass (145.58 -> "146"); for leg lengths, 2-decimal rounding.
TOLERANCE = {"deg": 1.0, "deg/s": 5.0, "leg_lengths": 0.01, "ms": 1.0, "year": 0.0}
DEFAULT_TOLERANCE = 0.01
_NUMBER = re.compile(r"(?<![\w.])[-−]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)?)")
_NORM_ID = re.compile(r"\bN-\d+\b")  # a citation, not a number

# heuristic: words that mean a feature is being discussed, used only to block claims about
# features the pipeline did not measure. Specific phrases, so a fault about another feature
# doesn't trip them.
FEATURE_WORDS = {
    "knee_angle_contact": ("knee angle at contact", "knee at contact", "knee at impact",
                           "leg at contact", "leg at impact"),
    "max_knee_flexion_backswing": ("backswing", "knee bend", "knee flexion", "bent your knee",
                                   "heel swing"),
    "trunk_lean_contact": ("trunk", "lean", "torso", "upper body", "chest"),
    "plant_foot_offset": ("plant foot", "standing foot", "support foot", "supporting foot",
                          "planting foot", "plant leg"),
    "segment_sequence_order": ("sequence", "proximal", "distal", "shank", "top speed"),
    "peak_knee_ext_velocity": ("velocity", "angular speed", "deg/s", "°/s",
                               "extension speed", "how fast your knee"),
}  # fmt: skip


@dataclass
class Context:
    """What the LLM was given, i.e. the only things it may restate."""

    fault_ids: list[str]
    citation_ids: set[str]
    numbers: list[tuple[float, float]] = field(default_factory=list)  # (value, tolerance)
    not_measured: list[str] = field(default_factory=list)

    def allow(self, value: float, unit: str | None) -> None:
        tol = TOLERANCE.get(unit or "", DEFAULT_TOLERANCE)
        self.numbers.append((float(value), tol))
        self.numbers.append((abs(float(value)), tol))  # "12 deg back" restates -12


def numbers_in(text: str) -> list[float]:
    out = []
    for m in _NUMBER.finditer(_NORM_ID.sub("", text)):
        s = m.group().replace("−", "-")
        s = (
            s.replace(",", "")
            if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s)
            else s.replace(",", ".")
        )
        out.append(float(s))
    return out


def validate(raw: dict | None, ctx: Context) -> tuple[Feedback | None, list[str]]:
    if raw is None:
        return None, ["the response was not valid JSON"]
    try:
        fb = Feedback.model_validate(raw)
    except ValidationError as e:
        return None, [f"schema: {err['loc']}: {err['msg']}" for err in e.errors()]

    errors = []
    ids = [it.fault_id for it in fb.items]
    for fid in sorted(set(ids) - set(ctx.fault_ids)):
        errors.append(f"item for fault {fid!r}, which was not detected")
    for fid in ctx.fault_ids:
        if ids.count(fid) != 1:
            errors.append(f"fault {fid!r} needs exactly one item, got {ids.count(fid)}")
    for it in fb.items:
        for c in it.citations:
            if c not in ctx.citation_ids:
                errors.append(f"{it.fault_id}: citation {c!r} is not one of the given IDs")
        for name in TEXT_FIELDS:
            text = getattr(it, name)
            for x in numbers_in(text):
                if not any(abs(x - v) <= tol for v, tol in ctx.numbers):
                    errors.append(
                        f"{it.fault_id}.{name}: the number {x:g} doesn't match any measured "
                        "or provided value"
                    )
            low = text.lower()
            for fid in ctx.not_measured:
                hits = [w for w in FEATURE_WORDS.get(fid, ()) if w in low]
                if hits:
                    errors.append(
                        f"{it.fault_id}.{name}: mentions {hits[0]!r}, but {fid} was not measured"
                    )
    return fb, errors
