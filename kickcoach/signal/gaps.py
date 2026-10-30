"""Low-confidence masking and short-gap filling, per keypoint track."""

from __future__ import annotations

import numpy as np

# heuristic: RTMPose's score on the 516 labelled body points never fell below 0.3 (1st
# percentile over all clips' leg points: 0.35), and median error only rose from 0.03 to 0.09
# thigh lengths between scores >= 0.7 and 0.4-0.5 (E1 data). So this mostly removes frames
# where no person was found, not merely blurred ones. Scores are model-specific (DEC-011).
MIN_CONF = 0.3
# heuristic: interpolate across at most this many missing frames. Longer gaps stay missing,
# because a straight line through a fast swing would invent the path.
MAX_GAP = 3


def mask_low_confidence(poses: np.ndarray, min_conf: float = MIN_CONF) -> np.ndarray:
    """[T, K, 3] -> [T, K, 2] positions with low-confidence or missing points set to NaN."""
    xy = poses[..., :2].astype(np.float64).copy()
    bad = ~(poses[..., 2] >= min_conf) | np.isnan(xy).any(-1)
    xy[bad] = np.nan
    return xy


def fill_gaps(xy: np.ndarray, max_gap: int = MAX_GAP) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation over interior gaps of <= max_gap frames, per track.

    ``xy`` is [T, ...] with NaN for missing. Returns (filled copy, bool mask of filled
    samples). Gaps at the start or end of the clip are never filled (nothing to bracket them).
    """
    out = np.array(xy, dtype=np.float64, copy=True)
    flat = out.reshape(out.shape[0], -1)
    filled = np.zeros(flat.shape, dtype=bool)
    t = np.arange(flat.shape[0])
    for j in range(flat.shape[1]):
        col = flat[:, j]
        ok = ~np.isnan(col)
        if ok.sum() < 2:
            continue
        idx = np.flatnonzero(ok)
        for a, b in zip(idx[:-1], idx[1:], strict=True):
            gap = b - a - 1
            if 0 < gap <= max_gap:
                seg = t[a + 1 : b]
                col[a + 1 : b] = np.interp(seg, [a, b], [col[a], col[b]])
                filled[a + 1 : b, j] = True
    return out, filled.reshape(xy.shape)


# heuristic: a thigh or shank more than this factor longer or shorter than its rolling median
# is a glitch (a point jumped), not foreshortening. On RTMPose's tracks the glitches seen
# (c07 partial swap, c12 ankle near the hip) were 0.42-0.47x and 1.57-2.07x; the rule masks
# 3.1% of all frames (JOURNAL, Phase 3).
SEGMENT_TOL = 1.5
# heuristic: half-width of the rolling median, in frames. Wide enough that a 1-5 frame glitch
# can't drag the median; narrow enough to follow a player growing as they approach.
SEGMENT_HALF_WINDOW = 7
LEG_CHAIN = ("knee", "ankle", "big_toe", "small_toe", "heel")


def rolling_nanmedian(x: np.ndarray, half: int) -> np.ndarray:
    pad = np.concatenate([np.full(half, np.nan), x, np.full(half, np.nan)])
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1)
    out = np.full(len(x), np.nan)
    ok = np.isfinite(win).any(1)
    out[ok] = np.nanmedian(win[ok], axis=1)
    return out


def mask_segment_glitches(
    xy: np.ndarray, tol: float = SEGMENT_TOL, half: int = SEGMENT_HALF_WINDOW
) -> tuple[np.ndarray, np.ndarray]:
    """Set a leg's chain (knee down) to NaN on frames where its thigh or shank length jumps
    away from the rolling median. Bones don't change length; a jump means a point is wrong.
    Returns (masked copy, [T, 2] bool: glitch frames for (left, right))."""
    from kickcoach.pose import skeleton

    out = xy.copy()
    flags = np.zeros((xy.shape[0], 2), dtype=bool)
    for i, side in enumerate(("left", "right")):
        for a, b in (("hip", "knee"), ("knee", "ankle")):
            L = np.linalg.norm(
                xy[:, skeleton.INDEX[f"{side}_{a}"]] - xy[:, skeleton.INDEX[f"{side}_{b}"]], axis=-1
            )
            r = L / rolling_nanmedian(L, half)
            flags[:, i] |= (r > tol) | (r < 1 / tol)
        chain = [skeleton.INDEX[f"{side}_{j}"] for j in LEG_CHAIN]
        out[np.ix_(flags[:, i], chain)] = np.nan
    return out, flags
