"""Ball track and ball-based contact (handoff D9) from per-frame detector candidates.

The ball rests until it is kicked, so it is found as the place where candidates keep
appearing before the kick. Contact is the last frame the ball is still at that place, if
by the next detection it has moved or vanished. That is the labelling rule (DEC-015):
contact is the touch frame, and "by the next frame the ball has moved".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# heuristic: candidates at least this confident are trusted to place the resting ball and to
# show it has moved. A lower-scoring candidate still counts as "at rest" at the rest position:
# the kicking foot covers the ball at contact (c03 frame 68 scored 0.1).
MIN_TRACK_SCORE = 0.3
# heuristic, set on tuning clips (c04, c07, c08, c10): within this many ball diameters of
# the rest position = still at rest. A resting ball's box centre jittered by 0.00-0.02 D; on
# the labelled contact frame it had shifted 0.04-0.05 D, and on the next frame 0.15-0.27 D.
REST_TOL = 0.1
# heuristic: the ball must be seen resting on at least this many frames.
MIN_REST_FRAMES = 3


@dataclass(frozen=True)
class Rest:
    x: float
    y: float
    diameter: float  # px
    frames: tuple[int, ...]


@dataclass(frozen=True)
class BallContact:
    frame: int
    confidence: str  # "high" | "medium" | "low": see contact_from_ball
    next_seen: int | None  # first frame after contact with the ball detected away from rest


def _centres(c: np.ndarray) -> np.ndarray:
    return np.stack([(c[:, 0] + c[:, 2]) / 2, (c[:, 1] + c[:, 3]) / 2], -1)


def _diam(c: np.ndarray) -> np.ndarray:
    return ((c[:, 2] - c[:, 0]) + (c[:, 3] - c[:, 1])) / 2


def rest_position(cands: list[np.ndarray], until: int | None = None) -> Rest | None:
    """The spot where confident candidates recur most often in frames [0, until)."""
    pts = []
    for t, c in enumerate(cands[:until]):
        c = c[c[:, 4] >= MIN_TRACK_SCORE] if len(c) else c
        if len(c):
            best = c[np.argmax(c[:, 4])][None]
            pts.append((t, *_centres(best)[0], _diam(best)[0]))
    if not pts:
        return None
    P = np.array(pts)
    xy, d = P[:, 1:3], P[:, 3]
    near = np.linalg.norm(xy[:, None] - xy[None], axis=-1) <= REST_TOL * d[:, None]
    counts = near.sum(1)
    i = int(np.argmax(counts))
    members = near[i]
    if members.sum() < MIN_REST_FRAMES:
        return None
    cx, cy = np.median(xy[members], 0)
    return Rest(
        float(cx), float(cy), float(np.median(d[members])), tuple(int(t) for t in P[members, 0])
    )


def _at_rest(c: np.ndarray, rest: Rest) -> bool:
    if not len(c):
        return False
    d = np.linalg.norm(_centres(c) - [rest.x, rest.y], axis=-1)
    return bool((d <= REST_TOL * rest.diameter).any())


def _moved(c: np.ndarray, rest: Rest) -> bool:
    if not len(c):
        return False
    c = c[c[:, 4] >= MIN_TRACK_SCORE]
    d = np.linalg.norm(_centres(c) - [rest.x, rest.y], axis=-1) if len(c) else np.array([])
    return bool((d > REST_TOL * rest.diameter).any()) and not _at_rest(c, rest)


def contact_from_ball(
    cands: list[np.ndarray], rest: Rest, search_end: int | None = None
) -> BallContact | None:
    """Last frame at rest (up to ``search_end``), with how clearly the ball then left.

    confidence: "high" = detected away from rest on the very next frame; "medium" = within 3
    frames; "low" = never seen again (it left the image, or blurred out of detection).
    """
    end = len(cands) if search_end is None else min(search_end, len(cands))
    rest_frames = [t for t in range(end) if _at_rest(cands[t], rest)]
    if not rest_frames:
        return None
    last = rest_frames[-1]
    nxt = next((t for t in range(last + 1, len(cands)) if _moved(cands[t], rest)), None)
    if nxt == last + 1:
        conf = "high"
    elif nxt is not None and nxt <= last + 3:
        conf = "medium"
    else:
        conf = "low"
    return BallContact(last, conf, nxt)
