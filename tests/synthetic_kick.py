"""A synthetic side-view instep kick with known angles and a known contact frame.

Angle curves are monotone cubic (PCHIP) through keyframes, so every extreme sits exactly on a
keyframe and the true values are known:
- the previous running stride flexes the kicking knee to 90 deg at 0.15 s (a trap: deeper
  than the kick's backswing, and it must not be picked);
- max knee flexion of the kick: 95 deg at 0.6 s, after the thigh has started forward (as in
  Kellis & Katis 2007, p. 155);
- contact at 0.7 s: knee 150 deg, trunk lean -8 deg (leaning back);
- the thigh swings forward fastest around max knee flexion and slows into contact, while
  the shank is still speeding up (the proximal-to-distal order, Kellis & Katis 2007, p. 155);
- knee extension speeds up all the way to contact, peaking there at 1000 deg/s (inside the
  860-1720 deg/s that Kellis & Katis report), then the impact cuts it to ~200 deg/s at once.
  That jump is what filtering through impact smears (Nunome et al. 2006).
The ball rests until the contact frame and has moved by the next one (the DEC-015 contact
definition). Times are real seconds; 30, 60, 120 and 240 fps all put a frame on 0.6 s and
0.7 s. The right leg kicks, and is the near leg.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator

from kickcoach.pose import skeleton

T_KEYS = [0.0, 0.15, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
THIGH = [20, 10, -5, -10, -25, 10, 30, 40, 42]  # deg from downward vertical, + = forward
KNEE = [150, 90, 170, 140, 110, 95, 150, 170, 172]  # interior angle
LEAN_T, LEAN = [0.0, 0.35, 0.7, 0.9], [5, 10, -8, 0]
T_CONTACT, T_MAX_FLEX, DURATION = 0.7, 0.6, 0.9
KNEE_VEL_AT_CONTACT = 1000.0  # deg/s, the peak (see the docstring)
THIGH_PX = SHANK_PX = 100.0
TRUNK_PX, FOOT_PX, HEEL_PX = 160.0, 45.0, 15.0
HIP_Y, HIP_X0, HIP_VX = 250.0, 300.0, 80.0  # px, px, px/s
PLANT_DROP, T_PLANT = 175.0, 0.5  # plant ankle below the hip (px); plant-foot landing (s)
BALL_R, BALL_V = 22.0, 4000.0  # px, px/s after contact


@dataclass
class Kick:
    poses: np.ndarray  # [T, K, 3]
    ball: np.ndarray  # [T, 2] true centre
    fps: float
    contact_frame: int
    max_flex_frame: int
    knee_contact: float
    knee_min: float
    trunk_contact: float
    peak_knee_ext_vel: float  # deg/s, true peak between max flexion and contact
    plant_offset: float  # facing * (plant ankle x - ball x) / plant leg length, at contact
    facing: int
    kicking_leg: str = "right"
    width: int = 900


def _dir(deg: np.ndarray) -> np.ndarray:
    """Unit vector at ``deg`` from downward vertical, + toward +x (y is down)."""
    r = np.radians(deg)
    return np.stack([np.sin(r), np.cos(r)], -1)


def knee_curve():
    """Knee angle vs time: smooth up to contact, with a kink in velocity at contact."""
    i = T_KEYS.index(T_CONTACT)
    pre_t, pre_v = T_KEYS[: i + 1], KNEE[: i + 1]
    slopes = PchipInterpolator(pre_t, pre_v).derivative()(pre_t)
    slopes[-1] = KNEE_VEL_AT_CONTACT
    pre = CubicHermiteSpline(pre_t, pre_v, slopes)
    post = PchipInterpolator(T_KEYS[i:], KNEE[i:])
    return lambda t: np.where(np.asarray(t) <= T_CONTACT, pre(t), post(t)), pre


def make_kick(fps: float, noise_px: float = 0.0, seed: int = 0, facing: int = 1) -> Kick:
    t = np.arange(round(DURATION * fps) + 1) / fps
    thigh = PchipInterpolator(T_KEYS, THIGH)
    knee, knee_pre = knee_curve()
    lean = PchipInterpolator(LEAN_T, LEAN)

    hip = np.stack([HIP_X0 + HIP_VX * t, np.full_like(t, HIP_Y)], -1)
    r_hip, l_hip = hip + [2.0, 0], hip - [2.0, 0]  # each leg hangs from its own hip point
    th = thigh(t)
    shank = th - (180.0 - knee(t))
    k_knee = r_hip + THIGH_PX * _dir(th)
    k_ankle = k_knee + SHANK_PX * _dir(shank)
    k_toe = k_ankle + FOOT_PX * _dir(shank + 65)
    k_heel = k_ankle + HEEL_PX * _dir(shank - 30)

    c = round(T_CONTACT * fps)
    rest = k_toe[c] + [BALL_R * 0.8, -BALL_R * 0.4]
    # Plant leg: the foot lands 40 px behind the ball at T_PLANT and stays; the knee comes
    # from two-link inverse kinematics (equal segments, knee bending forward).
    land_x = rest[0] - 40.0
    p_ankle = np.stack(
        [land_x - 2 * HIP_VX * np.clip(T_PLANT - t, 0, None), np.full_like(t, HIP_Y + PLANT_DROP)],
        -1,
    )
    d = p_ankle - l_hip
    dist = np.linalg.norm(d, axis=-1)
    assert (dist < 2 * THIGH_PX).all(), "plant leg can't reach its foot"
    a = np.degrees(np.arccos(dist / (2 * THIGH_PX)))
    base = np.degrees(np.arctan2(d[:, 0], d[:, 1]))
    p_knee = l_hip + THIGH_PX * _dir(base + a)
    p_shank = np.degrees(np.arctan2(*(p_ankle - p_knee).T))
    p_toe = p_ankle + FOOT_PX * _dir(p_shank + 85)
    p_heel = p_ankle + HEEL_PX * _dir(p_shank - 40)

    lr = np.radians(lean(t))
    sh_mid = hip + TRUNK_PX * np.stack([np.sin(lr), -np.cos(lr)], -1)

    pts = {
        "right_hip": r_hip, "left_hip": l_hip,
        "right_shoulder": sh_mid + [3.0, 0], "left_shoulder": sh_mid - [3.0, 0],
        "right_knee": k_knee, "right_ankle": k_ankle,
        "right_big_toe": k_toe, "right_small_toe": k_toe - [2.0, 0], "right_heel": k_heel,
        "left_knee": p_knee, "left_ankle": p_ankle,
        "left_big_toe": p_toe, "left_small_toe": p_toe - [2.0, 0], "left_heel": p_heel,
        "nose": sh_mid + [10.0, -30.0],
    }  # fmt: skip
    poses = skeleton.empty(len(t))
    rng = np.random.default_rng(seed)
    for name, xy in pts.items():
        noisy = xy + rng.normal(0, noise_px, xy.shape) if noise_px else xy
        poses[:, skeleton.INDEX[name], :2] = noisy
        poses[:, skeleton.INDEX[name], 2] = 0.7

    after = np.clip(t - T_CONTACT, 0, None)
    ball = rest + np.stack([BALL_V * after, -0.3 * BALL_V * after], -1)

    fine = np.linspace(T_MAX_FLEX, T_CONTACT, 2001)
    peak_vel = float(knee_pre.derivative()(fine).max())
    plant_len = np.linalg.norm(l_hip[c] - p_knee[c]) + np.linalg.norm(p_knee[c] - p_ankle[c])
    kick = Kick(
        poses=poses,
        ball=ball,
        fps=fps,
        contact_frame=c,
        max_flex_frame=round(T_MAX_FLEX * fps),
        knee_contact=float(knee(T_CONTACT)),
        knee_min=float(min(KNEE[4:7])),
        trunk_contact=float(lean(T_CONTACT)),
        peak_knee_ext_vel=peak_vel,
        plant_offset=float((p_ankle[c, 0] - rest[0]) / plant_len),
        facing=1,
    )
    return mirror(kick) if facing == -1 else kick


def mirror(k: Kick) -> Kick:
    """The same kick toward -x (image left), with the left leg as the near, kicking leg."""
    poses = k.poses.copy()
    poses[..., 0] = k.width - poses[..., 0]
    for j in ("shoulder", "hip", "knee", "ankle", "big_toe", "small_toe", "heel"):
        li, ri = skeleton.INDEX[f"left_{j}"], skeleton.INDEX[f"right_{j}"]
        poses[:, [li, ri]] = poses[:, [ri, li]]
    ball = k.ball.copy()
    ball[:, 0] = k.width - ball[:, 0]
    return Kick(**{**k.__dict__, "poses": poses, "ball": ball, "facing": -1, "kicking_leg": "left"})


def ball_detections(k: Kick, score: float = 0.9) -> list[np.ndarray]:
    """Per-frame detector-style candidates [M, 5] (xyxy + score) for the true ball, dropping
    frames where it has left the image."""
    out = []
    for cx, cy in k.ball:
        if 0 <= cx <= k.width:
            out.append(np.array([[cx - BALL_R, cy - BALL_R, cx + BALL_R, cy + BALL_R, score]]))
        else:
            out.append(np.zeros((0, 5)))
    return out
