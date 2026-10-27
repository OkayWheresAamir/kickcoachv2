from fractions import Fraction

import numpy as np
import pytest

from kickcoach.ingest import fps_trust, ingest_clip, probe
from kickcoach.ingest.video import Mp4Writer, constant_frame_rate, repeated_frame_ratio

TRUST_DEFAULTS = dict(
    capture_fps_known="n",
    capture_fps=None,
    slow_motion="",
    playback_fps=30.0,
    constant_frame_rate=True,
    repeated_frame_ratio=0.0,
)


def trust(**kw):
    return fps_trust(**{**TRUST_DEFAULTS, **kw})


def test_stated_capture_rate_sets_real_time_step():
    t = trust(capture_fps_known="y", capture_fps=240.0, slow_motion="y")
    assert t.trusted and t.seconds_per_frame == pytest.approx(1 / 240)


def test_real_time_clip_uses_playback_rate():
    t = trust(slow_motion="n", playback_fps=29.97)
    assert t.trusted and t.seconds_per_frame == pytest.approx(1 / 29.97)


@pytest.mark.parametrize("slow", ["y", ""])
def test_slow_or_unknown_without_capture_rate_is_untrusted(slow):
    t = trust(slow_motion=slow)
    assert not t.trusted and t.seconds_per_frame is None


def test_variable_frame_rate_overrides_a_stated_capture_rate():
    t = trust(capture_fps_known="y", capture_fps=240.0, constant_frame_rate=False)
    assert not t.trusted


def test_repeated_frames_override_a_stated_capture_rate():
    t = trust(capture_fps_known="y", capture_fps=240.0, repeated_frame_ratio=0.5)
    assert not t.trusted and "heuristic" in t.reason


def test_capture_below_playback_is_untrusted():
    assert not trust(capture_fps_known="y", capture_fps=15.0).trusted


def test_constant_frame_rate():
    assert constant_frame_rate([i / 30 for i in range(50)])
    jitter = [i / 30 for i in range(50)]
    jitter[20] += 0.01
    assert not constant_frame_rate(jitter)


def test_repeated_frame_ratio_ignores_a_player_standing_still():
    # 100 still steps, then motion: the c21 pattern that fooled the first heuristic.
    changed = np.r_[np.zeros(100), np.full(80, 0.01)]
    assert repeated_frame_ratio(changed) == 0.0


def test_repeated_frame_ratio_detects_each_frame_shown_twice():
    changed = np.tile([0.01, 0.0], 60)  # moving, repeat, moving, repeat ...
    assert repeated_frame_ratio(changed) == pytest.approx(0.5, abs=0.02)


def _moving_square_video(path, n=60, fps=30, w=160, h=90, repeat=1):
    """Synthetic clip: a white square moving right by 2 px per *distinct* frame."""
    with Mp4Writer(path, Fraction(fps)) as out:
        for i in range(n):
            img = np.zeros((h, w, 3), np.uint8)
            x = 10 + 2 * (i // repeat)
            img[30:50, x : x + 20] = 255
            out.write(img)


def test_ingest_trims_window_and_reads_rates(tmp_path):
    src = tmp_path / "src.mp4"
    _moving_square_video(src, n=60, fps=30)
    info = probe(src)
    assert (info.width, info.height) == (160, 90)
    assert info.r_frame_rate == pytest.approx(30)
    assert info.avg_frame_rate == pytest.approx(30)

    meta = ingest_clip(
        src,
        clip_id="t",
        start_s=0.5,
        end_s=1.5,
        capture_fps_known="n",
        capture_fps=None,
        slow_motion="n",
        frames_dir=tmp_path / "frames",
        clip_path=tmp_path / "clip.mp4",
    )
    assert meta["n_frames"] == 30
    assert meta["frame_times_s"][0] == pytest.approx(0.5)
    assert meta["playback_fps"] == pytest.approx(30)
    assert meta["constant_frame_rate"] and meta["fps_trusted"]
    assert len(list((tmp_path / "frames").glob("*.jpg"))) == 30
    assert probe(tmp_path / "clip.mp4").r_frame_rate == pytest.approx(30)


def test_ingest_flags_a_clip_slowed_by_repeating_frames(tmp_path):
    src = tmp_path / "src.mp4"
    _moving_square_video(src, n=60, fps=30, repeat=2)
    meta = ingest_clip(
        src,
        clip_id="t",
        start_s=0,
        end_s=2,
        capture_fps_known="y",
        capture_fps=240.0,
        slow_motion="y",
        frames_dir=tmp_path / "frames",
        clip_path=tmp_path / "clip.mp4",
    )
    assert meta["repeated_frame_ratio"] > 0.4
    assert not meta["fps_trusted"]
