"""v1 (the hackathon version) replayed on cached keypoints, for E2, E3 and E8.

Follows ``kickcoach-2/pose_engine.py`` @ deb3c4e (V1_AUDIT §1). Only the fallback path is
replayed, because v1's kick-candidate gate never fires (V1_AUDIT 2.8): every upload was
answered by ``_fallback_peak_frame``. Per frame, for the legs ["right", "left"]:
- knee angle, when the hip, knee and ankle scores are all >= 0.2, and a causal EMA of it
  (alpha 0.35; a missing angle leaves the EMA where it was);
- ankle speed = one-step pixel displacement x the container fps, when the ankle score is
  >= 0.2;
- knee angular velocity = (angle - last valid angle) x fps, unfiltered (V1_AUDIT 2.4).
Trunk tilt is hip-mid -> shoulder-mid against image vertical, positive toward image right
whatever the kick direction (V1_AUDIT 2.7), with the same EMA. The answer is the frame and
leg with the highest ankle speed in the clip (V1_AUDIT 2.3), reporting the EMA knee angle and
trunk there, plus the raw knee velocity and ankle speed.

Keypoints come from tools/run_v1_replica.py: v1's MoveNet on BGR frames with v1's
letterbox-ignoring mapping (V1_AUDIT 2.2, 2.9).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kickcoach.features.geometry import joint_angle_deg, lean_from_vertical_deg
from kickcoach.pose import skeleton

CACHE_MODEL = "movenet_v1"
# v1 constants (pose_engine.py)
CONF_THRESH = 0.20
SMOOTH_ALPHA = 0.35
LEGS = ("right", "left")  # v1's order when the kicking leg is "auto"


def v1_pixels(out: np.ndarray, width: int, height: int) -> np.ndarray:
    """MoveNet output [17, 3] as (y_norm, x_norm, score) -> [17, 2] (x, y), mapped as v1 did:
    x_norm * width, y_norm * height, ignoring the letterbox padding (V1_AUDIT 2.2)."""
    return np.stack([out[:, 1] * width, out[:, 0] * height], axis=-1)


def ema(x: np.ndarray, alpha: float = SMOOTH_ALPHA) -> np.ndarray:
    """v1's causal EMA. NaN until the first valid value; a NaN input keeps the last state."""
    out = np.full(len(x), np.nan)
    v = None
    for i, xi in enumerate(x):
        if np.isfinite(xi):
            v = float(xi) if v is None else alpha * float(xi) + (1 - alpha) * v
        out[i] = np.nan if v is None else v
    return out


def _pt(poses: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    i = skeleton.INDEX[name]
    return poses[:, i, :2].astype(np.float64), poses[:, i, 2].astype(np.float64)


@dataclass(frozen=True)
class V1Series:
    knee_raw: dict[str, np.ndarray]
    knee_ema: dict[str, np.ndarray]
    knee_vel: dict[str, np.ndarray]  # deg/s at the container fps
    ankle_speed: dict[str, np.ndarray]  # px/s at the container fps
    trunk_ema: np.ndarray  # deg, + = shoulders toward image right


def series(poses: np.ndarray, fps: float) -> V1Series:
    knee_raw, knee_ema, knee_vel, speed = {}, {}, {}, {}
    for leg in LEGS:
        (hip, hc), (knee, kc), (ank, ac) = (
            _pt(poses, f"{leg}_{j}") for j in ("hip", "knee", "ankle")
        )
        ang = joint_angle_deg(hip, knee, ank)
        ang = np.where(np.minimum(np.minimum(hc, kc), ac) >= CONF_THRESH, ang, np.nan)
        knee_raw[leg], knee_ema[leg] = ang, ema(ang)
        # v1 stores every frame's ankle as "previous", whatever its score.
        sp = np.full(len(ank), np.nan)
        sp[1:] = np.linalg.norm(np.diff(ank, axis=0), axis=-1) * fps
        speed[leg] = np.where(ac >= CONF_THRESH, sp, np.nan)
        # ...but the previous knee angle only on frames where the angle was valid.
        kv = np.full(len(ang), np.nan)
        last = np.nan
        for t, a in enumerate(ang):
            if np.isfinite(a):
                if np.isfinite(last):
                    kv[t] = (a - last) * fps
                last = a
        knee_vel[leg] = kv
    pts = [_pt(poses, n) for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")]
    ok = np.min([c for _, c in pts], axis=0) >= CONF_THRESH
    sh_mid, hip_mid = (pts[0][0] + pts[1][0]) / 2, (pts[2][0] + pts[3][0]) / 2
    trunk = np.where(ok, lean_from_vertical_deg(hip_mid, sh_mid, 1), np.nan)
    return V1Series(knee_raw, knee_ema, knee_vel, speed, ema(trunk))


@dataclass(frozen=True)
class V1Answer:
    frame: int
    leg: str
    knee_angle: float  # EMA at the frame
    trunk: float  # EMA, image-signed
    knee_vel: float  # raw, deg/s at the container fps
    ankle_speed: float  # px/s


def kick(poses: np.ndarray, fps: float) -> V1Answer | None:
    """_fallback_peak_frame: the (frame, leg) with the highest ankle speed. v1 updates the best
    only on a strictly higher speed, frame by frame and right leg before left."""
    s = series(poses, fps)
    sp = np.stack([s.ankle_speed[leg] for leg in LEGS], axis=1)  # [T, 2], row-major = v1's order
    if not np.isfinite(sp).any():
        return None
    flat = int(np.nanargmax(sp))  # first occurrence of the maximum
    t, leg = flat // len(LEGS), LEGS[flat % len(LEGS)]
    return V1Answer(
        t,
        leg,
        float(s.knee_ema[leg][t]),
        float(s.trunk_ema[t]),
        float(s.knee_vel[leg][t]),
        float(s.ankle_speed[leg][t]),
    )
