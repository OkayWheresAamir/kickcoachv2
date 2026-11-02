"""Read stream metadata and the frames inside a time window, with PyAV (bundled FFmpeg)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

# heuristic: a pixel "changed" if its grey level moved by more than this between frames
# (tolerates re-encoding noise). A frame pair is "still" below _STILL changed pixels and
# "moving" above _MOVING. Not from a source; chosen by looking at the c21 frames.
_CHANGE_LEVEL = 8
_STILL = 1e-4
_MOVING = 1e-3
_MAX_STALL_RUN = 3  # repeating each frame up to 4x is the slow-down this looks for


@dataclass(frozen=True)
class StreamInfo:
    codec: str
    width: int
    height: int
    r_frame_rate: float | None  # ffprobe's r_frame_rate (PyAV: base_rate)
    avg_frame_rate: float | None  # ffprobe's avg_frame_rate (PyAV: average_rate)
    duration_s: float | None
    n_frames_header: int  # 0 when the container doesn't say

    def to_dict(self) -> dict:
        return asdict(self)


def _f(rate: Fraction | None) -> float | None:
    return float(rate) if rate else None


def probe(path: Path) -> StreamInfo:
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        dur = float(s.duration * s.time_base) if s.duration is not None else None
        return StreamInfo(
            codec=s.codec_context.name,
            width=s.codec_context.width,
            height=s.codec_context.height,
            r_frame_rate=_f(s.base_rate),
            avg_frame_rate=_f(s.average_rate),
            duration_s=dur,
            n_frames_header=s.frames,
        )


def iter_window(path: Path, start_s: float, end_s: float) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (presentation time in s, RGB uint8 [H, W, 3]) for frames with start <= t < end,
    turned upright by the stream's display rotation (PyAV's ``rotation`` is counterclockwise,
    as is ``np.rot90``). Mirrored display matrices are not handled."""
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        # Seek lands on the keyframe at or before start; decode forward from there.
        c.seek(int(start_s / s.time_base), stream=s, backward=True)
        for frame in c.decode(s):
            if frame.pts is None:
                continue
            t = float(frame.pts * s.time_base)
            if t < start_s:
                continue
            if t >= end_s:
                break
            img = frame.to_ndarray(format="rgb24")
            # Phones store portrait video as landscape pixels plus a display rotation; without
            # applying it the player lies on their side and the pose model fails.
            k = round(frame.rotation / 90) % 4
            yield t, (np.ascontiguousarray(np.rot90(img, k)) if k else img)


def changed_fraction(prev: np.ndarray, cur: np.ndarray) -> float:
    g0 = prev.astype(np.int16).mean(axis=2)
    g1 = cur.astype(np.int16).mean(axis=2)
    return float(np.mean(np.abs(g1 - g0) > _CHANGE_LEVEL))


def repeated_frame_ratio(changed: np.ndarray) -> float:
    """Share of frame steps during motion that look like a repeated frame.

    ``changed[i]`` is the changed-pixel fraction between frames i and i+1. A repeated frame
    shows up as a short run of still steps with moving steps on both sides. A player who
    simply stands still gives long still runs, which are not counted (c21 has 100 such
    frames before the run-up and no repeats).
    """
    changed = np.asarray(changed, dtype=np.float64)
    moving = np.flatnonzero(changed > _MOVING)
    if len(moving) < 2:
        return 0.0
    lo, hi = moving[0], moving[-1]
    stalls, i = 0, lo
    while i <= hi:
        if changed[i] < _STILL:
            j = i
            while j <= hi and changed[j] < _STILL:
                j += 1
            run = j - i
            if run <= _MAX_STALL_RUN and changed[i - 1] > _MOVING and changed[j] > _MOVING:
                stalls += run
            i = j
        else:
            i += 1
    return stalls / (hi - lo + 1)


def constant_frame_rate(times_s: list[float], rel_tol: float = 0.01) -> bool:
    """True if every frame step equals the median step (within rel_tol)."""
    if len(times_s) < 3:
        return True
    d = np.diff(np.asarray(times_s))
    med = float(np.median(d))
    return bool(med > 0 and np.all(np.abs(d - med) <= rel_tol * med))


class Mp4Writer:
    """Streams frames to near-lossless H.264 at a constant rate, so frame i of the file is
    frame i of the extracted JPEGs. Odd dimensions lose one pixel row/column (yuv420p)."""

    def __init__(self, path: Path, fps: Fraction) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._out = av.open(str(path), mode="w")
        self._fps = fps
        self._stream = None

    def write(self, img: np.ndarray) -> None:
        if self._stream is None:
            h, w = img.shape[:2]
            s = self._out.add_stream("libx264", rate=self._fps, options={"crf": "12"})
            s.width, s.height, s.pix_fmt = w - w % 2, h - h % 2, "yuv420p"
            self._stream = s
        s = self._stream
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(img[: s.height, : s.width]))
        self._out.mux(s.encode(vf))

    def close(self) -> None:
        if self._stream is not None:
            self._out.mux(self._stream.encode())
        self._out.close()

    def __enter__(self) -> Mp4Writer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
