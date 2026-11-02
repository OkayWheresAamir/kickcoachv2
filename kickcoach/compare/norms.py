"""Norm rows (handoff D12): literature reference values, extracted offline with an LLM and
verified by a human before anything uses them.

``knowledge/norms_candidates.csv`` is the extractor's raw output and is never edited.
``knowledge/norms.csv`` starts as a copy, and only Aamir's verification changes it: each row
gets a ``review`` of correct / corrected / rejected, and ``verified`` is true only for the
first two. E5 scores the candidates against it; rows the extractor missed are listed in
``knowledge/norms_missed.csv`` (E5 recall). Runtime code reads ``verified_norms()`` only.

Fields beyond D12 exist because verification and comparison need them: ``quote`` (the exact
supporting text, so a reviewer can find it and a script can check it exists), ``condition``
(e.g. an approach angle), ``angle_convention`` (papers disagree on what 0 deg means), and
``primary_source`` (a review's value really comes from the study it cites).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
NORMS_CSV = ROOT / "knowledge" / "norms.csv"
CANDIDATES_CSV = ROOT / "knowledge" / "norms_candidates.csv"

# The variables the extractor looks for: the ones a pipeline feature can be compared with,
# plus the support-foot distances (reported in cm, so not comparable yet; kept for honesty).
VARIABLES: dict[str, str] = {
    "kicking_knee_angle_at_impact": ("the kicking leg's knee angle at ball impact / contact"),
    "kicking_knee_max_flexion": (
        "the kicking leg's deepest knee flexion (most bent) during the backswing or swing, "
        "before impact"
    ),
    "trunk_angle_at_impact": "the trunk's inclination (lean) at ball impact",
    "support_foot_posterior_distance": (
        "how far the support (plant, standing) foot is behind the ball, along the kick direction"
    ),
    "support_foot_lateral_distance": (
        "how far the support foot is to the side of the ball, across the kick direction"
    ),
    "support_foot_distance": (
        "the support-foot-to-ball distance when the paper doesn't say in which direction"
    ),
    "peak_knee_extension_velocity": (
        "the kicking leg's maximum knee extension angular velocity during the swing"
    ),
    "peak_thigh_angular_velocity": "the kicking thigh's maximum angular velocity",
    "peak_shank_angular_velocity": "the kicking shank's maximum angular velocity",
}

Convention = Literal[
    "interior_180_is_straight", "flexion_0_is_straight", "not_applicable", "unclear"
]
Review = Literal["pending", "correct", "corrected", "rejected"]


class Population(BaseModel):
    level: str | None = None  # as the paper words it, e.g. "recreational", "elite"
    age_group: str | None = None  # e.g. "adult", "youth (13 yr)"
    sex: Literal["male", "female", "mixed", "unspecified"] = "unspecified"


class Stat(BaseModel):
    mean: float | None = None
    sd: float | None = None
    min: float | None = None
    max: float | None = None


class NormRow(BaseModel):
    norm_id: str
    source_id: str  # key in knowledge/sources.md
    variable: str
    kick_type: str
    condition: str | None = None
    population: Population = Field(default_factory=Population)
    stat: Stat = Field(default_factory=Stat)
    unit: str
    n: int | None = None
    method: str
    angle_convention: Convention = "not_applicable"
    source: str  # human-readable citation
    page_or_table: str
    primary_source: str | None = None
    quote: str
    notes: str | None = None
    verified: bool = False
    review: Review = "pending"
    review_note: str | None = None


# Flat CSV columns, in order: nested fields become population_* / stat_*.
COLUMNS = [
    "norm_id", "source_id", "variable", "kick_type", "condition",
    "population_level", "population_age_group", "population_sex",
    "stat_mean", "stat_sd", "stat_min", "stat_max", "unit", "n", "method",
    "angle_convention", "source", "page_or_table", "primary_source", "quote", "notes",
    "verified", "review", "review_note",
]  # fmt: skip


def to_flat(row: NormRow) -> dict:
    d = row.model_dump()
    flat = {k: v for k, v in d.items() if k not in ("population", "stat")}
    flat |= {f"population_{k}": v for k, v in d["population"].items()}
    flat |= {f"stat_{k}": v for k, v in d["stat"].items()}
    return {c: "" if flat[c] is None else flat[c] for c in COLUMNS}


def from_flat(rec: dict) -> NormRow:
    def val(k: str):
        v = rec.get(k, "")
        return None if v == "" else v

    nested = {"population": {}, "stat": {}}
    plain = {}
    for c in COLUMNS:
        v = val(c)
        group, _, key = c.partition("_")
        if group in nested and key:
            if v is not None:
                nested[group][key] = v
        elif v is not None:
            plain[c] = v
    if "verified" in plain:
        plain["verified"] = str(plain["verified"]).strip().lower() == "true"
    return NormRow.model_validate({**plain, **nested})


def save(rows: list[NormRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(to_flat(r) for r in rows)


def load(path: Path = NORMS_CSV) -> list[NormRow]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [from_flat(r) for r in csv.DictReader(f)]


def verified_norms(path: Path = NORMS_CSV) -> list[NormRow]:
    """The only rows runtime code may use (D12)."""
    return [r for r in load(path) if r.verified and r.review in ("correct", "corrected")]
