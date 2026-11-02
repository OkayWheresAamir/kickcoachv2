"""Deterministic comparison of measured features with verified norm rows (handoff D11, D12).

Only verified rows are used. A row is comparable when its variable maps to a pipeline
feature, its unit matches, and its angle convention is known (flexion angles are converted:
interior = 180 - flexion). Of the comparable rows, one is chosen per feature by a fixed
ranking (``_rank``) and the others are listed, so the same input always gives the same
verdict. Every comparison carries caveats: the norms come from small lab samples measured
differently from a phone video.
"""

from __future__ import annotations

from kickcoach.compare.norms import NormRow, Stat

# Pipeline feature -> the norm variable it can be compared with.
FEATURE_VARIABLE = {
    "knee_angle_contact": "kicking_knee_angle_at_impact",
    "max_knee_flexion_backswing": "kicking_knee_max_flexion",
    "trunk_lean_contact": "trunk_angle_at_impact",
    "plant_foot_offset": "support_foot_posterior_distance",
    "peak_knee_ext_velocity": "peak_knee_extension_velocity",
}
# Unit spellings found in papers, per pipeline unit.
UNIT_ALIASES = {
    "deg": {"deg", "degrees", "degree", "°"},
    "deg/s": {"deg/s", "deg·s-1", "deg s-1", "°/s", "deg.s-1"},
    "leg_lengths": {"leg_lengths"},
}
BAND_RULE = (
    "heuristic: 'within' = mean ± 1 SD of the study sample, or inside a reported range; "
    "the band is a description of that sample, not a target"
)
# heuristic: which study populations count as each user level. The papers' own wording.
LEVEL_WORDS = {
    "beginner": ("recreational",),
    "intermediate": ("amateur", "trained", "college", "intercollegiate", "club"),
    "advanced": ("elite", "professional", "skilled"),
}
_CHILD_WORDS = ("child", "youth", "adolescent", "boy", "girl", "year")
_ACCURACY_WORDS = ("penalty", "accura")


def _unit_ok(row: NormRow, unit: str) -> bool:
    return row.unit.strip().lower() in {u.lower() for u in UNIT_ALIASES.get(unit, {unit})}


def pipeline_stat(row: NormRow, unit: str) -> tuple[Stat | None, str | None]:
    """The row's statistics in the pipeline's convention, or (None, why not comparable)."""
    if not _unit_ok(row, unit):
        return None, f"the norm is in {row.unit}, the measurement in {unit}"
    s = row.stat
    if unit == "deg" and row.variable.startswith("kicking_knee"):
        if row.angle_convention == "flexion_0_is_straight":
            flip = lambda x: None if x is None else round(180.0 - x, 2)  # noqa: E731
            return Stat(mean=flip(s.mean), sd=s.sd, min=flip(s.max), max=flip(s.min)), None
        if row.angle_convention != "interior_180_is_straight":
            return None, f"the paper's knee angle convention is {row.angle_convention}"
    return s, None


def band(s: Stat) -> tuple[float, float] | None:
    if s.mean is not None and s.sd is not None:
        return s.mean - s.sd, s.mean + s.sd
    if s.min is not None and s.max is not None:
        return s.min, s.max
    return None


def _is_child(row: NormRow) -> bool:
    text = f"{row.population.age_group or ''} {row.population.level or ''}".lower()
    return any(w in text for w in _CHILD_WORDS) and "adult" not in text


def _rank(row: NormRow, stat: Stat, level: str | None) -> tuple:
    pop = (row.population.level or "").lower()
    level_match = bool(level) and any(w in pop for w in LEVEL_WORDS.get(level, ()))
    accuracy_kick = any(w in f"{row.kick_type} {row.condition or ''}".lower()
                        for w in _ACCURACY_WORDS)  # fmt: skip
    return (
        not level_match,
        _is_child(row),
        accuracy_kick,
        band(stat) is None,
        -(row.n or 0),
        row.norm_id,
    )


def caveats(row: NormRow, feature_id: str, verified: list[NormRow]) -> list[str]:
    out = []
    method = row.method.lower()
    if any(w in method for w in ("3d", "imu", "motion capture", "dlt")):
        out.append(
            f"Measured in a lab ({row.method}); this app measures a 2D side-on angle from "
            "video, so some difference is expected."
        )
    elif row.method in ("unclear", ""):
        out.append("The review doesn't say how this value was measured.")
    if feature_id == "peak_knee_ext_velocity":
        out.append(
            "2D video distorts knee angular velocity in the last 50 ms before impact "
            "(Nunome & Ikegami 2006, pp. 2-3)."
        )
    kick = f"{row.kick_type} {row.condition or ''}".lower()
    if any(w in kick for w in _ACCURACY_WORDS):
        fast = next((r for r in verified if r.norm_id == "N-011"), None)
        slow = next((r for r in verified if r.norm_id == "N-012"), None)
        pair = (
            f" (e.g. {fast.stat.mean:g} vs {slow.stat.mean:g} deg/s knee extension, "
            f"{fast.norm_id}/{slow.norm_id})"
            if fast and slow and fast.stat.mean and slow.stat.mean
            else ""
        )
        out.append(
            f"These were kicks aimed for accuracy ({row.kick_type}); accurate kicks use slower "
            f"motion than maximum-power kicks{pair}."
        )
    if row.n is not None and row.n < 30:
        out.append(f"Small sample: n = {row.n}.")
    return out


def _population(row: NormRow) -> str:
    p = row.population
    bits = [b for b in (p.level, p.age_group, None if p.sex == "unspecified" else p.sex) if b]
    return ", ".join(bits) or "not stated"


def compare(features: list[dict], verified: list[NormRow], level: str | None = None) -> list[dict]:
    """One comparison per numeric measured feature with a mapped norm variable."""
    out = []
    for f in features:
        fid = f["id"]
        var = FEATURE_VARIABLE.get(fid)
        if var is None or f.get("status") != "measured":
            continue
        rows = [r for r in verified if r.variable == var]
        if not rows:
            out.append(
                {"feature_id": fid, "norm_ref": None, "verdict": "no_comparable_norm",
                 "reason": "no verified norm row for this feature"}
            )  # fmt: skip
            continue
        usable, why_not = [], []
        for r in rows:
            s, why = pipeline_stat(r, f["unit"])
            if s is None:
                why_not.append(f"{r.norm_id}: {why}")
            else:
                usable.append((r, s))
        if not usable:
            out.append(
                {"feature_id": fid, "norm_ref": None, "verdict": "no_comparable_norm",
                 "reason": "; ".join(why_not)}
            )  # fmt: skip
            continue
        usable.sort(key=lambda rs: _rank(rs[0], rs[1], level))
        row, stat = usable[0]
        b = band(stat)
        value = float(f["value"])
        if b is None:
            verdict = "reference_only"
        else:
            verdict = (
                "below_norm" if value < b[0] else "above_norm" if value > b[1] else "within_norm"
            )
        out.append(
            {
                "feature_id": fid,
                "value": value,
                "unit": f["unit"],
                "norm_ref": row.norm_id,
                "verdict": verdict,
                "norm_mean": stat.mean,
                "norm_sd": stat.sd,
                "norm_band": None if b is None else [round(b[0], 2), round(b[1], 2)],
                "band_rule": BAND_RULE,
                "source": f"{row.source}, {row.page_or_table}"
                + (f" (from {row.primary_source})" if row.primary_source else ""),
                "population": _population(row),
                "kick_type": row.kick_type + (f", {row.condition}" if row.condition else ""),
                "converted": row.angle_convention == "flexion_0_is_straight",
                "caveats": caveats(row, fid, verified),
                "also_matching": [r.norm_id for r, _ in usable[1:]],
            }
        )
    return out
