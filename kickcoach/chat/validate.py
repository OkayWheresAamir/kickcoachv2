"""The chat answer validator: the feedback validator's checks (DEC-029, DEC-042), adapted to
an answer written from tool results instead of a fixed set of facts.

An answer passes only if:
1. it isn't empty;
2. every number in it is a number the tools returned in this conversation, or one from the
   question, within the feedback validator's per-unit tolerance; a number written with a
   unit must match a value in that unit (E9's run 1: "-0.32 m" for -0.32 leg lengths);
3. every passage, norm or fault ID it mentions (K-12, N-032, F-LEAN-BACK) was returned by a
   tool;
4. any sentence about a feature that no tool ever returned as measured says it wasn't
   measured (the phrase lists are heuristic);
5. it doesn't call a knee's interior angle an amount of bend.

Like the feedback validator it checks numbers, IDs and wording, not meaning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from kickcoach.feedback.generate import FEATURE_UNITS
from kickcoach.feedback.validate import (
    _SENTENCE,
    DEFAULT_TOLERANCE,
    FEATURE_WORDS,
    KNEE_ANGLE_FEATURES,
    TOLERANCE,
    Context,
    knee_bend_errors,
    number_spans,
    numbers_in,
)

_ID = re.compile(r"\b(?:[KN]-\d{2,3}|F-[A-Z]+(?:-[A-Z]+)*)\b")
# heuristic: unit words written right after a number, and the unit each means. Lengths, times
# and percentages are units no tool returns, so a number written with one never matches.
_UNIT_AFTER = re.compile(
    r"\s*(°/s|deg/s|degrees? per second|°|degrees?\b|deg\b|leg[ _]lengths?\b|cm\b|mm\b"
    r"|metres?\b|meters?\b|m\b|%|ms\b|seconds?\b)",
    re.I,
)
_UNIT_WORDS = {"°/s": "deg/s", "deg/s": "deg/s", "degree per second": "deg/s",
               "degrees per second": "deg/s", "°": "deg", "deg": "deg", "degree": "deg",
               "leg": "leg_lengths", "cm": "length", "mm": "length", "m": "length",
               "metre": "length", "meter": "length", "%": "percent", "ms": "time",
               "second": "time"}  # fmt: skip
ANY = "any"  # a number found inside a tool's text: its unit is unknown, so any is accepted
# heuristic: ways of saying "not measured". A sentence about a never-measured feature needs
# one of these.
NOT_MEASURED_PHRASES = (
    "not measured", "wasn't measured", "weren't measured", "isn't measured", "aren't measured",
    "couldn't be measured", "could not be measured", "can't be measured", "cannot be measured",
    "couldn't measure", "could not measure", "can't measure", "cannot measure",
    "didn't measure", "did not measure", "no measurement", "not available", "unavailable",
    "withheld", "not recorded", "no data", "don't have", "do not have", "wasn't recorded",
    "not trusted", "can't tell", "cannot tell", "no value",
)  # fmt: skip


@dataclass
class Grounding:
    """Everything the answer may restate, collected from the tool results."""

    numbers: list[tuple[float, float, str | None]] = field(default_factory=list)
    # (value, tolerance, unit)
    ids: set[str] = field(default_factory=set)
    measured: set[str] = field(default_factory=set)
    not_measured: set[str] = field(default_factory=set)
    knee_angles: list[float] = field(default_factory=list)

    def allow(self, value: float, tol: float, unit: str | None = ANY) -> None:
        self.numbers += [(float(value), tol, unit), (abs(float(value)), tol, unit)]

    @property
    def never_measured(self) -> set[str]:
        return self.not_measured - self.measured


def grounding(results: list[dict], question: str = "") -> Grounding:
    g = Grounding()
    for x in numbers_in(question):
        g.allow(x, 0.0)
    for r in results:
        _walk(r, None, None, g)
    return g


def _walk(node, unit: str | None, key: str | None, g: Grounding) -> None:
    if isinstance(node, dict):
        unit = node.get("unit") or unit
        fid = node.get("id") or node.get("feature_id")
        if fid in FEATURE_WORDS:
            (g.not_measured if node.get("status") == "not_measured" else g.measured).add(fid)
            if fid in KNEE_ANGLE_FEATURES and isinstance(node.get("value"), float | int):
                g.knee_angles.append(float(node["value"]))
        for k in ("id", "norm_ref", "fault_id"):
            if node.get(k):
                g.ids.add(str(node[k]))
        if isinstance(node.get("values"), dict):  # get_history: {feature_id: value or text}
            for k, v in node["values"].items():
                is_nm = isinstance(v, str) and v.startswith("not measured")
                (g.not_measured if is_nm else g.measured).add(k)
        for k, v in node.items():  # {feature_id: value} (history values, evidence)
            _walk(v, FEATURE_UNITS.get(k, unit), k, g)
    elif isinstance(node, list):
        for v in node:
            _walk(v, unit, key, g)
    elif isinstance(node, bool) or node is None:
        return
    elif isinstance(node, int | float):
        g.allow(node, TOLERANCE.get(unit or "", DEFAULT_TOLERANCE), unit)
    elif isinstance(node, str):
        for x in numbers_in(node):
            g.allow(x, 0.0)
        g.ids.update(_ID.findall(node))


def stated_unit(text: str, end: int) -> str | None:
    m = _UNIT_AFTER.match(text, end)
    if m is None:
        return None
    word = m.group(1).lower()
    return next(u for w, u in _UNIT_WORDS.items() if word.startswith(w))


def number_errors(answer: str, g: Grounding) -> list[tuple[float, str]]:
    """(number, why) for each number in the answer that no tool result supports."""
    text = _ID.sub("", answer)
    out = []
    for x, end in number_spans(text):
        near = [u for v, tol, u in g.numbers if abs(x - v) <= tol]
        if not near:
            out.append((x, f"the number {x:g} isn't in any tool result; copy numbers from them"))
            continue
        unit = stated_unit(text, end)
        if unit is not None and not any(u in (unit, ANY) for u in near):
            have = " / ".join(sorted({u for u in near if u not in (None, ANY)}) or ["no unit"])
            out.append((x, f"{x:g} {unit} isn't in any tool result (that number appears only "
                           f"in {have}); copy the value and its unit from a tool"))  # fmt: skip
    return out


def check(answer: str, g: Grounding) -> list[str]:
    if not answer.strip():
        return ["the answer is empty"]
    errors = [msg for _, msg in number_errors(answer, g)]
    for i in sorted(set(_ID.findall(answer)) - g.ids):
        errors.append(f"{i} wasn't returned by any tool; cite only IDs the tools returned")
    for sentence in _SENTENCE.split(answer):
        low = sentence.lower()
        for fid in sorted(g.never_measured):
            hit = next((w for w in FEATURE_WORDS[fid] if w in low), None)
            if hit and not any(p in low for p in NOT_MEASURED_PHRASES):
                errors.append(
                    f"mentions {hit!r}, but {fid} was not measured: say so and give the reason "
                    "instead of describing it"
                )
    ctx = Context(fault_ids=[], citation_ids=set(), knee_angles=g.knee_angles)
    errors += [e.removeprefix("answer: ") for e in knee_bend_errors("answer", answer, ctx)]
    return errors
