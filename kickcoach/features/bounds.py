"""Sanity bounds (handoff D10). A value outside its bound becomes ``not_measured`` with the
reason; it is never clamped. Every bound cites a source or says ``heuristic`` and why.

These are limits of what is physically possible, not norms: a value inside a bound can
still be a poor kick. Norms come later (knowledge/norms.csv, Phase 5).
"""

from __future__ import annotations

from dataclasses import dataclass

# heuristic: margin over the highest literature value, for 2D projection and for players
# outside the studied samples. Not from a source.
VELOCITY_MARGIN = 1.5
# Kellis & Katis 2007, p. 155: during the forward swing "knee extension velocity is
# maximized (860-1720 deg·s-1)" (citing Nunome et al. 2002).
KNEE_EXT_VEL_LIT_MAX = 1720.0
# Kellis & Katis 2007, p. 155: shank angular velocity increases "until ball impact reaching
# values of 1891 deg·s-1" (citing Dorge et al. 1999).
SHANK_VEL_LIT_MAX = 1891.0


@dataclass(frozen=True)
class Bound:
    lo: float
    hi: float
    source: str


BOUNDS: dict[str, Bound] = {
    "knee_angle_contact": Bound(
        30.0, 180.0,
        "heuristic: a 2D knee angle under 30 deg would mean over 150 deg of flexion, beyond "
        "a normal knee's range; 180 deg (straight) is the geometric maximum",
    ),
    "max_knee_flexion_backswing": Bound(
        30.0, 180.0, "heuristic: as knee_angle_contact"
    ),
    "trunk_lean_contact": Bound(
        -60.0, 60.0,
        "heuristic: a trunk more than 60 deg from vertical at contact would mean the player is "
        "falling, or the hips/shoulders are misplaced",
    ),
    "plant_foot_offset": Bound(
        -1.0, 1.0,
        "heuristic: both feet hang from the same hips, so with the kicking foot at the ball "
        "the plant ankle can't be much more than a leg length from it horizontally",
    ),
    "peak_knee_ext_velocity": Bound(
        0.0, VELOCITY_MARGIN * KNEE_EXT_VEL_LIT_MAX,
        f"Kellis & Katis 2007, p. 155: peak knee extension 860-1720 deg/s; bound = "
        f"{VELOCITY_MARGIN} x 1720 (margin heuristic). <= 0 means no extension before contact",
    ),
    "peak_shank_ang_velocity": Bound(
        0.0, VELOCITY_MARGIN * SHANK_VEL_LIT_MAX,
        f"Kellis & Katis 2007, p. 155: shank angular velocity up to 1891 deg/s at impact; "
        f"bound = {VELOCITY_MARGIN} x 1891 (margin heuristic)",
    ),
}  # fmt: skip


def violation(feature_id: str, value: float) -> str | None:
    """None if ``value`` is within the feature's bound, else the reason for ``not_measured``."""
    b = BOUNDS[feature_id]
    if b.lo <= value <= b.hi:
        return None
    return f"value {value:.1f} is outside the sanity bound [{b.lo:g}, {b.hi:g}] ({b.source})"
