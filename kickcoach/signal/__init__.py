"""Signal stage: raw pose [T, K, 3] -> cleaned positions for events and features.

Order: mask low confidence -> fix left/right swaps -> mask segment-length glitches -> fill
short gaps -> zero-lag low-pass. The plan lists the gap fill before the swap fix. Fixing
swaps first means a gap is never bridged between two frames that label the legs
differently. The glitch mask was added after real clips showed single-frame jumps (a knee
"angle" of 3 deg on c12) that a low-pass filter only smears. Derivatives are taken by the
caller (``derivative``) on the filtered signal, never on raw points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kickcoach.signal.gaps import fill_gaps, mask_low_confidence, mask_segment_glitches
from kickcoach.signal.smooth import cutoff_cpf, derivative, lowpass
from kickcoach.signal.swaps import fix_leg_swaps

__all__ = ["Cleaned", "clean", "cutoff_cpf", "derivative", "lowpass"]


@dataclass(frozen=True)
class Cleaned:
    xy: np.ndarray  # [T, K, 2] filtered positions, NaN = unknown
    unfiltered: np.ndarray  # [T, K, 2] after masking, swap fix and gap fill
    conf: np.ndarray  # [T, K] model confidence, moved along with any swap
    filled: np.ndarray  # [T, K] bool: interpolated across a short gap
    swapped: np.ndarray  # [T] bool: the leg chains were exchanged on this frame
    glitches: np.ndarray  # [T, 2] bool: (left, right) leg chain masked as a length glitch
    cutoff_cpf: float | None  # None = not filtered
    breaks: tuple[int, ...]


def clean(
    poses: np.ndarray,
    *,
    seconds_per_frame: float | None = None,
    breaks: tuple[int, ...] = (),
    target_cpf: float | None | str = "auto",
) -> Cleaned:
    """``target_cpf="auto"`` picks the cutoff from the frame timing (smooth.cutoff_cpf)."""
    xy = mask_low_confidence(poses)
    conf = np.where(np.isnan(xy[..., 0]), np.nan, poses[..., 2].astype(np.float64))
    xy, conf, swapped = fix_leg_swaps(xy, conf)
    xy, glitches = mask_segment_glitches(xy)
    xy, filled = fill_gaps(xy)
    cpf = cutoff_cpf(seconds_per_frame) if target_cpf == "auto" else target_cpf
    return Cleaned(
        xy=lowpass(xy, cpf, breaks),
        unfiltered=xy,
        conf=conf,
        filled=filled.any(-1),
        swapped=swapped,
        glitches=glitches,
        cutoff_cpf=cpf,
        breaks=tuple(breaks),
    )
