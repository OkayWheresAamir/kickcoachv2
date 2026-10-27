"""Ingest: one source video + a time window -> numbered frames, a trimmed clip, metadata.

Frame i is the same image in ``<frames_dir>/<i:06d>.jpg``, in the trimmed MP4 and in every
pose array's first axis, so labels, model outputs and LLM inputs all share one index.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

from kickcoach.ingest.fps import FpsTrust, fps_trust
from kickcoach.ingest.video import (
    Mp4Writer,
    changed_fraction,
    constant_frame_rate,
    iter_window,
    probe,
    repeated_frame_ratio,
)

__all__ = ["FpsTrust", "fps_trust", "ingest_clip", "load_meta", "probe"]


def ingest_clip(
    src: Path,
    *,
    clip_id: str,
    start_s: float,
    end_s: float,
    capture_fps_known: str,
    capture_fps: float | None,
    slow_motion: str,
    frames_dir: Path,
    clip_path: Path,
) -> dict:
    info = probe(src)
    if not info.r_frame_rate:
        raise ValueError(f"{src}: stream has no frame rate")
    rate = Fraction(info.r_frame_rate).limit_denominator(1001)

    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.jpg"):
        old.unlink()

    times: list[float] = []
    changed: list[float] = []
    prev: np.ndarray | None = None
    with Mp4Writer(clip_path, rate) as mp4:
        for i, (t, img) in enumerate(iter_window(src, start_s, end_s)):
            Image.fromarray(img).save(frames_dir / f"{i:06d}.jpg", quality=95)
            mp4.write(img)
            if prev is not None:
                changed.append(changed_fraction(prev, img))
            prev = img
            times.append(t)
    if not times:
        raise ValueError(f"{clip_id}: no frames in [{start_s}, {end_s}) s of {src.name}")

    steps = np.diff(times)
    playback_fps = float(1.0 / np.median(steps)) if len(steps) else info.r_frame_rate
    cfr = constant_frame_rate(times)
    repeated = repeated_frame_ratio(np.asarray(changed))
    trust = fps_trust(
        capture_fps_known=capture_fps_known,
        capture_fps=capture_fps,
        slow_motion=slow_motion,
        playback_fps=playback_fps,
        constant_frame_rate=cfr,
        repeated_frame_ratio=repeated,
    )
    meta = {
        "clip_id": clip_id,
        "source": src.name,
        "window_s": [start_s, end_s],
        "stream": info.to_dict(),
        "n_frames": len(times),
        "frame_times_s": [round(t, 6) for t in times],
        "playback_fps": playback_fps,
        "constant_frame_rate": cfr,
        "repeated_frame_ratio": repeated,
        "changed_fraction": [round(c, 6) for c in changed],
        "fps_trusted": trust.trusted,
        "seconds_per_frame": trust.seconds_per_frame,
        "fps_reason": trust.reason,
    }
    (frames_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def load_meta(frames_dir: Path) -> dict:
    return json.loads((frames_dir / "meta.json").read_text())
