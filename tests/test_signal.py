import numpy as np
import pytest
from scipy.signal import sosfreqz
from synthetic_kick import make_kick

from kickcoach.pose import skeleton
from kickcoach.signal import clean
from kickcoach.signal.gaps import fill_gaps, mask_low_confidence
from kickcoach.signal.smooth import (
    CUTOFF_CPF_UNTRUSTED,
    _sos,
    cutoff_cpf,
    derivative,
    lowpass,
)
from kickcoach.signal.swaps import LEG_CHAIN, apply_swaps, fix_leg_swaps, swap_flags


@pytest.mark.parametrize("target", [0.05, 0.15, 0.3, 0.45])
def test_two_pass_filter_is_minus_3db_at_the_target(target):
    w, h = sosfreqz(_sos(target), worN=[2 * np.pi * target])
    assert abs(h[0]) ** 2 == pytest.approx(1 / np.sqrt(2), rel=1e-6)  # |H|^2 twice = 1/2


def test_cutoff_from_frame_timing():
    assert cutoff_cpf(1 / 240) == pytest.approx(18 / 240)
    assert cutoff_cpf(None) == CUTOFF_CPF_UNTRUSTED == pytest.approx(18 / 120)
    assert cutoff_cpf(1 / 30) is None  # 18 Hz is above Nyquist at 30 fps: no filtering


def test_lowpass_keeps_slow_and_removes_fast_motion():
    t = np.arange(400)
    slow = np.sin(2 * np.pi * 0.02 * t)
    fast = np.sin(2 * np.pi * 0.4 * t)
    out = lowpass(slow + fast, 0.15)
    mid = slice(50, 350)
    np.testing.assert_allclose(out[mid], slow[mid], atol=0.03)


def test_lowpass_has_no_lag():
    t = np.arange(200)
    x = np.exp(-(((t - 100) / 15.0) ** 2))
    assert np.argmax(lowpass(x, 0.1)) == 100


def test_lowpass_handles_nan_runs():
    x = np.arange(40, dtype=float)
    x[10:13] = np.nan  # a gap
    x[13:15] = 5.0  # a 2-frame run: too short to filter
    x[15:18] = np.nan
    out = lowpass(x, 0.2)
    assert np.isnan(out[10:18]).all()
    np.testing.assert_allclose(out[20:35], x[20:35], atol=1e-3)  # a ramp passes through


def test_break_keeps_an_impact_from_smearing_backwards():
    x = np.where(np.arange(60) <= 30, 0.0, 10.0)  # a step right after frame 30
    smeared = lowpass(x, 0.1)
    kept = lowpass(x, 0.1, breaks=(30,))
    assert abs(smeared[30]) > 1.0  # without the break the step leaks into frame 30
    np.testing.assert_allclose(kept[:31], 0.0, atol=1e-9)
    np.testing.assert_allclose(kept[31:], 10.0, atol=1e-9)


def test_derivative_central_differences_and_breaks():
    x = 3.0 * np.arange(10.0)
    np.testing.assert_allclose(derivative(x, dt=0.5), 6.0)
    y = np.where(np.arange(10) <= 4, 0.0, 100.0)
    d = derivative(y, breaks=(4,))
    assert d[4] == 0.0  # one-sided at the break: never uses the frame after it


def test_mask_low_confidence_and_missing():
    poses = skeleton.empty(3)
    poses[:, 0] = [[1, 2, 0.9], [1, 2, 0.1], [np.nan, np.nan, 0.0]]
    xy = mask_low_confidence(poses)
    assert not np.isnan(xy[0, 0]).any()
    assert np.isnan(xy[1, 0]).all() and np.isnan(xy[2, 0]).all()


def test_fill_gaps_short_interior_only():
    x = np.array([np.nan, 0, 1, np.nan, np.nan, 4, np.nan, np.nan, np.nan, np.nan, 9, np.nan])
    out, filled = fill_gaps(x[:, None], max_gap=3)
    np.testing.assert_allclose(out[1:6, 0], [0, 1, 2, 3, 4])
    assert np.isnan(out[6:10, 0]).all()  # a 4-frame gap stays missing
    assert np.isnan(out[0, 0]) and np.isnan(out[-1, 0])  # never extrapolated
    assert filled[:, 0].tolist() == [False, False, False, True, True] + [False] * 7


def _swap_frames(poses, frames):
    flags = np.zeros(len(poses), dtype=bool)
    flags[list(frames)] = True
    return apply_swaps(poses, flags, LEG_CHAIN)


@pytest.mark.parametrize("fps", [30, 120])
def test_injected_leg_swaps_are_repaired(fps):
    k = make_kick(fps, noise_px=1.0, seed=1)
    T = len(k.poses)
    bad = {3, T // 2, *range(T // 3, T // 3 + 4)}  # single frames and a stretch
    xy, conf, flags = fix_leg_swaps(
        mask_low_confidence(_swap_frames(k.poses, bad)), k.poses[..., 2].astype(float)
    )
    assert set(np.flatnonzero(flags)) == bad
    np.testing.assert_allclose(xy, mask_low_confidence(k.poses), atol=1e-9)


def test_a_long_consistent_run_is_left_to_the_model():
    # Inside a long run the legs move smoothly; only its edge is evidence, which can't
    # outweigh overruling the model on every frame of it (by design: see swaps.py).
    k = make_kick(60, noise_px=0.5)
    T = len(k.poses)
    flags = swap_flags(mask_low_confidence(_swap_frames(k.poses, range(0, T // 3))), LEG_CHAIN)
    assert flags.sum() <= 2


@pytest.mark.parametrize("fps", [30, 60, 120, 240])
def test_no_swaps_invented_on_a_clean_kick(fps):
    k = make_kick(fps, noise_px=1.5, seed=fps)
    assert not swap_flags(mask_low_confidence(k.poses), LEG_CHAIN).any()


def test_clean_kick_has_no_swaps_and_noise_is_reduced():
    k = make_kick(240, noise_px=2.0, seed=2)
    truth = make_kick(240).poses[..., :2]
    out = clean(k.poses, seconds_per_frame=1 / 240, breaks=(k.contact_frame,))
    assert not out.swapped.any()
    raw_err = np.nanmean(np.linalg.norm(k.poses[..., :2] - truth, axis=-1))
    filt_err = np.nanmean(np.linalg.norm(out.xy - truth, axis=-1))
    assert filt_err < 0.6 * raw_err
