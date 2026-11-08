"""The deterministic feedback validator (handoff D13). An LLM answer passes only if:

1. it parses against the schema (kickcoach.feedback.schema);
2. its items are exactly the detected faults, one each;
3. every citation is an ID it was given;
4. every number in its text matches a provided number within tolerance;
5. it says nothing about a feature that is ``not_measured``;
6. it claims no direction a measurement contradicts, e.g. leaning back when the trunk
   leaned forward (E7's R11);
7. it doesn't call a knee's interior angle an amount of bend (E7's R02 and R13: "knee bend
   was 101 degrees" for a 101 deg angle, which is 79 deg of bend).

Checks 6 and 7 came from E7's ratings (Phase 9). Like 5, they match phrases, so they check
wording, not meaning: a negated claim ("you didn't lean back") is rejected too, which costs
a retry, not a wrong answer.

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

# heuristic: phrases that claim a direction for a signed feature, the sign they claim, and the
# fields that are checked. The fault's own direction (back, behind) is checked in "what" and
# "why_it_matters", where R11 put "makes your body lean away" on a kick that leaned forward.
# The corrective direction (forward, ahead) is the normal wording of advice, so it is checked
# only in "what", which describes this kick. On the 31 cached Phase 8 answers this rejects
# R11 and nothing else.
_BACK = tuple(f"{verb} {way}" for verb in ("lean", "leaned", "leaning", "leans")
              for way in ("back", "away"))  # fmt: skip
_FORWARD = ("lean forward", "leaned forward", "leaning forward", "leans forward")
DIRECTION_CLAIMS = {
    "trunk_lean_contact": ((_BACK, -1, ("what", "why_it_matters")), (_FORWARD, +1, ("what",))),
    "plant_foot_offset": ((("behind the ball",), -1, ("what", "why_it_matters")),
                          (("ahead of the ball", "in front of the ball", "past the ball"), +1,
                           ("what",))),
}  # fmt: skip
SIGN_WORDS = {-1: "negative", +1: "positive"}
# Knee angles are interior angles (180 = straight, DEC-029). A sentence that restates one with
# a bend word and no "angle" reads as degrees of bend, the opposite scale.
KNEE_ANGLE_FEATURES = ("knee_angle_contact", "max_knee_flexion_backswing")
_BEND = re.compile(r"\b(?:bend|bends|bent|bending|flex\w*)\b")
_SENTENCE = re.compile(r"(?<=[.!?;])\s+")


@dataclass
class Context:
    """What the LLM was given, i.e. the only things it may restate."""

    fault_ids: list[str]
    citation_ids: set[str]
    numbers: list[tuple[float, float]] = field(default_factory=list)  # (value, tolerance)
    not_measured: list[str] = field(default_factory=list)
    signed: dict[str, float] = field(default_factory=dict)  # DIRECTION_CLAIMS features
    knee_angles: list[float] = field(default_factory=list)  # interior angles given, any source

    def allow(self, value: float, unit: str | None) -> None:
        tol = TOLERANCE.get(unit or "", DEFAULT_TOLERANCE)
        self.numbers.append((float(value), tol))
        self.numbers.append((abs(float(value)), tol))  # "12 deg back" restates -12


def parse_number(s: str) -> float:
    s = s.replace("−", "-")
    s = (
        s.replace(",", "")
        if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s)
        else s.replace(",", ".")
    )
    return float(s)


def number_spans(text: str) -> list[tuple[float, int]]:
    """(value, index just after it) for each number; the caller strips IDs first."""
    return [(parse_number(m.group()), m.end()) for m in _NUMBER.finditer(text)]


def numbers_in(text: str) -> list[float]:
    return [x for x, _ in number_spans(_NORM_ID.sub("", text))]


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
            errors += direction_errors(f"{it.fault_id}.{name}", name, low, ctx)
            errors += knee_bend_errors(f"{it.fault_id}.{name}", text, ctx)
    return fb, errors


def direction_errors(where: str, field_name: str, low: str, ctx: Context) -> list[str]:
    out = []
    for fid, value in ctx.signed.items():
        for phrases, sign, fields in DIRECTION_CLAIMS.get(fid, ()):
            if field_name not in fields or value * sign > 0:
                continue
            hit = next((p for p in phrases if p in low), None)
            if hit:
                out.append(
                    f"{where}: says {hit!r}, but {fid} is {value:g} (a {SIGN_WORDS[sign]} value "
                    "would mean that); this kick's measurement shows the opposite or neither"
                )
    return out


def knee_bend_errors(where: str, text: str, ctx: Context) -> list[str]:
    out = []
    for sentence in _SENTENCE.split(text):
        low = sentence.lower()
        if "angle" in low or not _BEND.search(low):
            continue
        for x in numbers_in(sentence):
            if any(abs(x - v) <= TOLERANCE["deg"] for v in ctx.knee_angles):
                out.append(
                    f"{where}: {x:g} is the knee's interior angle (180 = straight), not an "
                    "amount of bend; call it the knee angle"
                )
    return out
