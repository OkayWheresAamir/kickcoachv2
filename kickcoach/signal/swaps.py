"""Left/right leg-swap fix (handoff D8): in a side view, models often trade the near and far
leg on some frames.

Each frame is either kept or swapped (two states). The sequence of states is chosen by
dynamic programming (Viterbi) to minimise:
- motion cost: how far each leg lands from where constant velocity predicts it, from the two
  previous frames, in thigh lengths. A fast kicking foot is predicted well from its own last
  two positions, but not from its last position alone: in a kick it can move about as far
  in one frame as the gap between the legs. A first version matched each frame to the
  previous one and flipped 30-59-frame stretches of c06, c07, c09 and c10, swapping legs
  RTMPose had right (JOURNAL, Phase 3);
- a prior cost per swapped frame: overruling the model needs evidence, so when the legs
  cross and both assignments fit, the model's labels stand. E1 found RTMPose had no
  whole-leg swaps on 43 labelled frames, so the prior should usually win.
Each frame's motion evidence is capped, so the fix repairs transient swaps (a few frames)
and leaves long, consistent runs to the model: inside such a run the legs move smoothly, and
motion alone can't tell it from the legs passing each other in a stride. MoveNet's swaps in
E1 were mostly runs of 14-32 frames, which no motion rule can repair.

Only the leg chain (knee down) is fixed. E1's MoveNet swaps always crossed knees and ankles
but the hips only sometimes (c04 and c19 kept them). Hips sit 4-13 px apart in most clips,
about the model's jitter, so motion can't tell a hip swap from noise, and which hip is which
barely moves any feature in a side view. Shoulders are left alone too: only their midpoint
is used (trunk lean), which a swap can't move.
"""

from __future__ import annotations

import numpy as np

from kickcoach.pose import skeleton

LEG_CHAIN = ("knee", "ankle", "big_toe", "small_toe", "heel")
# heuristic: cost of overruling the model on one frame, in thigh lengths of motion error.
# For scale: RTMPose's median point error is 0.02 thigh lengths (E1); a one-frame swap of legs
# 0.5 apart costs about 4 x 0.5 to leave alone (the jump in, its extrapolation, the jump out).
PRIOR_COST = 0.5
# heuristic: the most motion evidence one frame can contribute (thigh lengths). Without it, a
# single garbled pose (c09 frame 2) outweighed everything and flipped 32 correct frames.
MAX_FRAME_COST = 1.0


def _pair_index(joints: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    left = np.array([skeleton.INDEX[f"left_{j}"] for j in joints])
    right = np.array([skeleton.INDEX[f"right_{j}"] for j in joints])
    return left, right


def thigh_scale(xy: np.ndarray) -> float:
    """Median hip-knee distance over the clip (both sides), in px; 1.0 if unknown."""
    d = [
        np.linalg.norm(
            xy[:, skeleton.INDEX[f"{s}_hip"]] - xy[:, skeleton.INDEX[f"{s}_knee"]], axis=-1
        )
        for s in ("left", "right")
    ]
    d = np.concatenate(d)
    d = d[~np.isnan(d)]
    return float(np.median(d)) if len(d) and np.median(d) > 0 else 1.0


def _motion_cost(cur: np.ndarray, prev: np.ndarray, prev2: np.ndarray, scale: float) -> float:
    """Median over joints of |cur - constant-velocity prediction|, in thigh lengths. Falls
    back to constant position where the frame before last is missing; 0 with no evidence."""
    pred = np.where(np.isnan(prev2), prev, 2 * prev - prev2)
    d = np.linalg.norm(cur - pred, axis=-1).ravel()
    d = d[~np.isnan(d)]
    return min(float(np.median(d)) / scale, MAX_FRAME_COST) if len(d) else 0.0


def swap_flags(
    xy: np.ndarray, joints: tuple[str, ...] = LEG_CHAIN, prior: float = PRIOR_COST
) -> np.ndarray:
    """[T] bool: frames whose left/right ``joints`` should be exchanged. ``xy`` is [T, K, 2]."""
    li, ri = _pair_index(joints)
    T = xy.shape[0]
    if T == 0:
        return np.zeros(0, dtype=bool)
    # chains[t, s]: [2 legs, J, 2] with state s (0 = as the model says, 1 = swapped).
    straight = np.stack([xy[:, li], xy[:, ri]], 1)
    chains = np.stack([straight, straight[:, ::-1]], 1)
    scale = thigh_scale(xy)
    nan = np.full_like(chains[0, 0], np.nan)
    # D[b, c]: best cost with state b at t-1 and c at t. Start with a virtual frame -1.
    D = np.array([[0.0, prior], [np.inf, np.inf]])
    back = np.zeros((T, 2, 2), dtype=np.int8)
    for t in range(1, T):
        new = np.full((2, 2), np.inf)
        for b in (0, 1):
            for c in (0, 1):
                best, arg = np.inf, 0
                for a in (0, 1):
                    if not np.isfinite(D[a, b]):
                        continue
                    prev2 = chains[t - 2, a] if t >= 2 else nan
                    cost = D[a, b] + _motion_cost(chains[t, c], chains[t - 1, b], prev2, scale)
                    if cost < best:
                        best, arg = cost, a
                new[b, c] = best + prior * c
                back[t, b, c] = arg
        D = new
    if T == 1:
        return np.array([False])
    b, c = np.unravel_index(np.argmin(D), D.shape)
    states = np.zeros(T, dtype=np.int8)
    states[T - 1], states[T - 2] = c, b
    for t in range(T - 1, 1, -1):
        states[t - 2] = back[t, states[t - 1], states[t]]
    return states.astype(bool)


def apply_swaps(arr: np.ndarray, flags: np.ndarray, joints: tuple[str, ...]) -> np.ndarray:
    """Exchange left/right ``joints`` on flagged frames. ``arr`` is [T, K, ...]."""
    li, ri = _pair_index(joints)
    out = arr.copy()
    f = np.flatnonzero(flags)
    out[np.ix_(f, li)], out[np.ix_(f, ri)] = arr[np.ix_(f, ri)], arr[np.ix_(f, li)]
    return out


def fix_leg_swaps(xy: np.ndarray, conf: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fix leg-chain swaps. Returns (xy, conf, [T] bool: frames that were swapped)."""
    f = swap_flags(xy, LEG_CHAIN)
    return apply_swaps(xy, f, LEG_CHAIN), apply_swaps(conf, f, LEG_CHAIN), f
