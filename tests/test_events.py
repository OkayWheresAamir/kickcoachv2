import numpy as np
import pytest
from synthetic_kick import ball_detections, make_kick

from kickcoach import events
from kickcoach.events.ball import Rest, contact_from_ball, rest_position
from kickcoach.events.kick import knee_angle, max_backswing
from kickcoach.signal import clean

FPS = [30, 60, 120, 240]


@pytest.mark.parametrize("facing", [1, -1])
def test_kicking_leg_and_direction_are_detected(facing):
    k = make_kick(60, noise_px=1.0, facing=facing)
    ev = events.detect(clean(k.poses).xy, None)
    assert ev.kicking_leg.leg == k.kicking_leg
    assert ev.facing == facing


@pytest.mark.parametrize("fps", FPS)
def test_ball_contact_is_exact_when_the_ball_is_seen_leaving(fps):
    k = make_kick(fps, noise_px=1.0)
    ev = events.detect(clean(k.poses).xy, ball_detections(k))
    assert ev.contact_method == "ball_motion" and ev.contact_confidence == "high"
    assert ev.contact_frame == k.contact_frame


@pytest.mark.parametrize("fps", [60, 120, 240])
def test_hidden_touch_is_bracketed_and_picked_by_foot_speed(fps):
    k = make_kick(fps, noise_px=0.5)
    cands = ball_detections(k)
    hidden = range(k.contact_frame - 2, k.contact_frame + 3)  # the foot covers the ball
    cands = [np.zeros((0, 5)) if t in hidden else c for t, c in enumerate(cands)]
    xy = clean(k.poses).xy
    raw = events.detect(xy, cands, bracket_lead_frames=0)
    assert raw.contact_method == "ball_bracket"
    # The foot peaks just before it meets the ball, so the raw pick runs up to 2 frames early
    # (on the real tuning clips: 0-4, median 2). The shipped pick is moved 2 frames later.
    assert -2 <= raw.contact_frame - k.contact_frame <= 0
    ev = events.detect(xy, cands)
    assert ev.contact_frame == raw.contact_frame + events.BRACKET_LEAD_FRAMES
    assert -1 <= ev.contact_frame - k.contact_frame <= 2


def test_bracket_shift_stays_inside_the_bracket():
    k = make_kick(60, noise_px=0.5)
    cands = ball_detections(k)
    # The ball reappears moved one frame after contact: the bracket is [c - 3, c].
    hidden = range(k.contact_frame - 2, k.contact_frame + 1)
    cands = [np.zeros((0, 5)) if t in hidden else c for t, c in enumerate(cands)]
    ev = events.detect(clean(k.poses).xy, cands, bracket_lead_frames=10)
    assert ev.contact_method == "ball_bracket"
    assert ev.contact_frame == ev.ball_contact.next_seen - 1


@pytest.mark.parametrize("fps", FPS)
def test_foot_speed_fallback_without_a_ball(fps):
    k = make_kick(fps, noise_px=1.0)
    ev = events.detect(clean(k.poses).xy, None)
    assert ev.contact_method == "foot_speed" and ev.contact_confidence == "low"
    assert -2 <= ev.contact_frame - k.contact_frame <= 0  # see the bracket test


def test_a_ball_far_from_the_foot_is_rejected():
    k = make_kick(60)
    far = [c + np.array([[0, -300, 0, -300, 0]]) if len(c) else c for c in ball_detections(k)]
    ev = events.detect(clean(k.poses).xy, far)
    assert ev.contact_method == "foot_speed"
    assert any("rejected" in n for n in ev.notes)


@pytest.mark.parametrize("fps", FPS)
def test_backswing_skips_the_deeper_bend_of_the_previous_stride(fps):
    k = make_kick(fps, noise_px=1.0)
    knee = knee_angle(clean(k.poses, breaks=(k.contact_frame,)).xy, "right")
    bs = max_backswing(knee, k.contact_frame)
    assert abs(bs.frame - k.max_flex_frame) <= max(1, fps // 60)
    assert bs.window_start > round(0.15 * fps)  # the stride's 90-deg bend is outside
    assert not bs.window_from_clip_start


def test_rest_position_ignores_a_one_off_candidate():
    ball = np.array([[90, 90, 110, 110, 0.9]])
    junk = np.array([[400, 50, 420, 70, 0.95]])
    cands = [ball] * 10 + [junk] + [ball] * 5
    r = rest_position(cands)
    assert (r.x, r.y, r.diameter) == (100, 100, 20)
    assert len(r.frames) == 15


def test_contact_confidence_reflects_how_soon_the_ball_is_seen_moving():
    rest = Rest(100.0, 100.0, 20.0, (0, 1, 2))
    at, gone = np.array([[90, 90, 110, 110, 0.9]]), np.array([[150, 80, 170, 100, 0.8]])
    none = np.zeros((0, 5))
    assert contact_from_ball([at, at, at, gone], rest).confidence == "high"
    assert contact_from_ball([at, at, at, none, none, gone], rest).confidence == "medium"
    low = contact_from_ball([at, at, at, none, none, none, none, none], rest)
    assert (low.frame, low.confidence, low.next_seen) == (2, "low", None)
