"""2D joint and segment angles. Used both by the pipeline and to derive ground-truth angles
from human labels, so the two are always computed by the same code.

Conventions (image coordinates):
- points are (x, y) in pixels, x to the right, y DOWN;
- inputs may be single points (2,) or stacked (..., 2); outputs broadcast accordingly;
- degenerate inputs (coincident points) give NaN, never a made-up number.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-9


def joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Interior angle at ``b`` between segments b->a and b->c, in [0, 180] degrees.

    Knee angle = joint_angle_deg(hip, knee, ankle): 180 is a straight leg.
    """
    ba = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    bc = np.asarray(c, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n_ba = np.linalg.norm(ba, axis=-1)
    n_bc = np.linalg.norm(bc, axis=-1)
    degenerate = (n_ba < _EPS) | (n_bc < _EPS)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(ba * bc, axis=-1) / (n_ba * n_bc)
    angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return np.where(degenerate, np.nan, angle)


def lean_from_vertical_deg(lower: np.ndarray, upper: np.ndarray, facing: int) -> np.ndarray:
    """Signed angle of the segment lower->upper from image vertical, in (-180, 180] degrees.

    ``facing`` is +1 if the player kicks toward +x (image right), -1 toward -x. Positive
    means the upper point is ahead of the lower one in the kicking direction, so trunk lean
    = lean_from_vertical_deg(hip_mid, shoulder_mid, facing) is positive for a forward lean.
    Assumes a level camera (see the capture protocol).
    """
    if facing not in (1, -1):
        raise ValueError(f"facing must be +1 or -1, got {facing!r}")
    d = np.asarray(upper, dtype=np.float64) - np.asarray(lower, dtype=np.float64)
    degenerate = np.linalg.norm(d, axis=-1) < _EPS
    # y points down, so "up" is -dy.
    angle = np.degrees(np.arctan2(facing * d[..., 0], -d[..., 1]))
    return np.where(degenerate, np.nan, angle)
