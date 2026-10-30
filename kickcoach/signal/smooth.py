"""Zero-lag low-pass filtering and derivatives (handoff D7).

Frequencies are in cycles per frame (cpf; Nyquist = 0.5), because for most clips the real
time between frames is unknown (DEC-010). With a trusted frame rate the cutoff is set in Hz.

Filtering through ball impact distorts the kinematics around it (Nunome et al. 2006, p. 12:
filtering swing and impact together "with a standard recursive Butterworth filtering
approach is likely to distort kinematic information before and after impact"). So filtering
accepts ``breaks``: each side of a break is filtered on its own, and the pipeline puts one
at the contact frame, which ends the swing.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, sosfiltfilt

ORDER = 2
# Nunome et al. 2006, p. 11: kicking studies "filtered the displacement data with a cut-off
# frequency of 6 - 18 Hz", which the authors call "appropriate for describing the swing
# phase kinematics". Our features sit at the end of the swing, so we take the top of it.
CUTOFF_HZ = 18.0
# heuristic: the cutoff when the real time per frame is unknown. 0.15 cpf is CUTOFF_HZ at
# 120 fps capture, the lowest rate the handoff trusts for velocities (D6). On the tuning split
# (eval/tune_cutoff.py, knee angle at the labelled frames, with the contact break) it scored
# 2.52 deg MAE vs 2.67 unfiltered; the best, 0.08-0.1 cpf, scored 2.47. Every cutoff landed
# within ~0.4 deg, and the unfiltered row is flattered by pre-fill anchoring, so the data
# can't rank them finely. E2 reports the sensitivity (D7).
CUTOFF_CPF_UNTRUSTED = 0.15
# heuristic: a valid run shorter than this can't be filtered meaningfully and is dropped.
MIN_RUN = 4


def cutoff_cpf(seconds_per_frame: float | None) -> float | None:
    """The target cutoff for a clip, or None when it is at or above Nyquist (no filtering
    possible without removing swing content, e.g. 18 Hz on real-time 30 fps video)."""
    cpf = CUTOFF_HZ * seconds_per_frame if seconds_per_frame else CUTOFF_CPF_UNTRUSTED
    return cpf if cpf < 0.5 else None


def design_cpf(target_cpf: float, order: int = ORDER) -> float:
    """Single-pass design cutoff so that the forward-backward (two-pass) filter is -3 dB at
    ``target_cpf``.

    A digital Butterworth has |H|^2 = 1 / (1 + (tan(pi f) / tan(pi fd))^(2N)). Two passes
    square it, so -3 dB at f needs tan(pi f) / tan(pi fd) = (sqrt(2) - 1)^(1 / 2N).
    """
    c = (math.sqrt(2) - 1) ** (1 / (2 * order))
    return math.atan(math.tan(math.pi * target_cpf) / c) / math.pi


def _sos(target_cpf: float, order: int = ORDER) -> np.ndarray:
    return butter(order, 2 * design_cpf(target_cpf, order), btype="low", output="sos")


def _segments(n: int, breaks: tuple[int, ...]) -> list[tuple[int, int]]:
    """[start, stop) pieces; a break at b ends a piece *at* b (b is the last frame of it)."""
    cuts = sorted({b + 1 for b in breaks if 0 <= b < n - 1})
    edges = [0, *cuts, n]
    return list(zip(edges[:-1], edges[1:], strict=True))


def _runs(ok: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) runs of True."""
    d = np.diff(np.concatenate([[0], ok.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True))


def lowpass(
    x: np.ndarray, target_cpf: float | None, breaks: tuple[int, ...] = (), order: int = ORDER
) -> np.ndarray:
    """Zero-lag Butterworth low-pass along axis 0 of [T, ...], NaN-aware.

    Each run of valid samples within each piece between breaks is filtered separately; runs
    shorter than MIN_RUN become NaN. ``target_cpf=None`` returns a copy unfiltered.
    """
    x = np.asarray(x, dtype=np.float64)
    if target_cpf is None:
        return x.copy()
    sos = _sos(target_cpf, order)
    default_pad = 3 * (2 * len(sos) + 1)  # sosfiltfilt's default padlen for this filter
    flat = x.reshape(x.shape[0], -1)
    out = np.full_like(flat, np.nan)
    for j in range(flat.shape[1]):
        col = flat[:, j]
        for a, b in _segments(len(col), breaks):
            seg = col[a:b]
            for s, e in _runs(~np.isnan(seg)):
                n = e - s
                if n < MIN_RUN:
                    continue
                pad = min(default_pad, n - 1)
                out[a + s : a + e, j] = sosfiltfilt(sos, seg[s:e], padlen=pad)
    return out.reshape(x.shape)


def derivative(x: np.ndarray, dt: float = 1.0, breaks: tuple[int, ...] = ()) -> np.ndarray:
    """Central differences along axis 0 (one-sided at the ends of each piece between breaks),
    so a velocity at the contact frame never uses the frame after impact. Units per ``dt``."""
    x = np.asarray(x, dtype=np.float64)
    out = np.full_like(x, np.nan)
    for a, b in _segments(x.shape[0], breaks):
        if b - a >= 2:
            out[a:b] = np.gradient(x[a:b], dt, axis=0)
    return out
