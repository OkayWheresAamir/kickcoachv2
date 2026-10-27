"""Decide whether a clip's frame timing can be trusted for velocities.

The container frame rate (``r_frame_rate``/``avg_frame_rate``) is the *playback* rate. A
slow-motion clip captured at 240 fps and played at 30 fps has 1/240 s of real time between
frames, not 1/30 s, so a velocity computed from the playback rate would be 8x too slow.
Only two cases give a real-time step per frame:
- the capture rate is stated by the source, and each stored frame is one captured frame;
- the clip is real time (not slowed), so capture rate = playback rate.
Everything else is untrusted: angles are still measured, velocities are withheld.
"""

from __future__ import annotations

from dataclasses import dataclass

# heuristic: share of frame steps during motion that look like repeated frames, above which
# we assume the editor slowed the clip by repeating frames, which breaks "one stored frame =
# one captured frame". Not from a source; revisit with the E8 results.
MAX_REPEATED_FRAME_RATIO = 0.10


@dataclass(frozen=True)
class FpsTrust:
    trusted: bool
    seconds_per_frame: float | None  # real time between consecutive frames, if trusted
    reason: str


def fps_trust(
    *,
    capture_fps_known: str,
    capture_fps: float | None,
    slow_motion: str,
    playback_fps: float,
    constant_frame_rate: bool,
    repeated_frame_ratio: float,
) -> FpsTrust:
    """``capture_fps_known`` and ``slow_motion`` are manifest values: "y", "n" or "" (unknown)."""
    if not constant_frame_rate:
        return FpsTrust(False, None, "frame timestamps are not evenly spaced in the window")
    if repeated_frame_ratio > MAX_REPEATED_FRAME_RATIO:
        return FpsTrust(
            False,
            None,
            f"{repeated_frame_ratio:.0%} of frame steps during motion look repeated "
            f"(> {MAX_REPEATED_FRAME_RATIO:.0%}, heuristic)",
        )
    if capture_fps_known == "y":
        if not capture_fps or capture_fps <= 0:
            raise ValueError("capture_fps_known is 'y' but capture_fps is missing")
        if capture_fps < playback_fps:
            return FpsTrust(False, None, "stated capture rate is below the playback rate")
        return FpsTrust(
            True, 1.0 / capture_fps, f"capture rate stated by source: {capture_fps:g} fps"
        )
    if slow_motion == "n":
        return FpsTrust(True, 1.0 / playback_fps, f"real-time clip: playback {playback_fps:g} fps")
    if slow_motion == "y":
        return FpsTrust(False, None, "slow motion with unknown capture rate")
    return FpsTrust(False, None, "unknown whether the clip is slowed down")
