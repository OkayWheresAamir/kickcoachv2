"""Phase 1 audit follow-ups on real clips (V1_AUDIT 2.2, 2.5, and v1's colour order).

Runs v1's own MoveNet weights on full downloaded source videos and reports:
  A. letterbox (2.2): knee angles from v1's mapping minus the corrected mapping;
  B. colour order: v1 fed OpenCV BGR frames; MoveNet expects RGB;
  C. kick selection (2.5): v1's candidate loop, replayed with its exact state-update order,
     plus a counterfactual with that order fixed, to see whether the 10-candidate cap bites.
Writes results/audit/phase1_followups.md and results/audit/v1_candidates.csv.

    uv run python tools/audit_v1/real_clip_followups.py
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from fetch_clips import source_path, video_id  # noqa: E402

from kickcoach.features.geometry import joint_angle_deg  # noqa: E402
from kickcoach.ingest.video import iter_window, probe  # noqa: E402
from kickcoach.pose.models import MoveNet  # noqa: E402

OUT = ROOT / "results" / "audit"
# v1 constants, from kickcoach-2/pose_engine.py
CONF_THRESH, KICK_PPS, KNEE_DPS, COOLDOWN_S, MAX_KICKS = 0.20, 120.0, 80.0, 0.4, 10
L, R = (11, 13, 15), (12, 14, 16)  # hip, knee, ankle
# Clips chosen to cover aspect ratios (4:3, 16:9) and long run-ups before the kick.
CLIPS = ["c01", "c05", "c10", "c19", "c20", "c24"]


def manifest() -> dict[str, dict]:
    return {r["clip_id"]: r for r in csv.DictReader((ROOT / "data" / "manifest.csv").open())}


def run_source(model: MoveNet, src: Path):
    """MoveNet raw outputs for every frame of the full source, for RGB and for BGR input."""
    rgb_out, bgr_out, lbs = [], [], []
    for _, rgb in iter_window(src, 0.0, math.inf):
        o, lb = model.raw(rgb)
        rgb_out.append(o)
        bgr_out.append(model.raw(np.ascontiguousarray(rgb[..., ::-1]))[0])
        lbs.append(lb)
    return np.array(rgb_out), np.array(bgr_out), lbs[0]


def to_v1_pixels(out: np.ndarray, w: int, h: int) -> np.ndarray:
    """v1's scaled_keypoints_from_output: (y_norm*h, x_norm*w, score), padding ignored."""
    px = out.copy()
    px[..., 0] *= h
    px[..., 1] *= w
    return px


def knee_angles(xy: np.ndarray, conf: np.ndarray, leg) -> np.ndarray:
    hip, knee, ankle = leg
    ang = joint_angle_deg(xy[:, hip], xy[:, knee], xy[:, ankle])
    ok = conf[:, [hip, knee, ankle]].min(axis=1) >= CONF_THRESH
    return np.where(ok, ang, np.nan)


def v1_candidates(px: np.ndarray, fps: float, fix_state_order: bool) -> list[int]:
    """Replay pose_engine.analyze_video_file's per-frame loop on v1-format keypoints.

    As written, the per-leg update stores this frame's ankle and knee angle as "prev"
    *before* the detection block recomputes speeds against "prev", so both speeds are 0.
    ``fix_state_order`` computes the speeds before overwriting, i.e. what was intended.
    """
    state = {leg: {"prev_ankle": None, "prev_knee": None} for leg in ("right", "left")}
    cooldown, cands = 0, []
    cooldown_frames = max(1, int(round(COOLDOWN_S * fps)))
    for f in range(len(px)):
        if len(cands) >= MAX_KICKS:
            break
        speeds = {}
        for leg, (hi, ki, ai) in (("right", R), ("left", L)):
            s = state[leg]
            (hy, hx, hc), (ky, kx, kc), (ay, ax, ac) = px[f, hi], px[f, ki], px[f, ai]
            knee = np.nan
            if min(hc, kc, ac) >= CONF_THRESH:
                knee = joint_angle_deg((hx, hy), (kx, ky), (ax, ay))
            if fix_state_order:
                spd = kav = np.nan
                if s["prev_ankle"] is not None and ac >= CONF_THRESH:
                    spd = math.hypot(ax - s["prev_ankle"][0], ay - s["prev_ankle"][1]) * fps
                if s["prev_knee"] is not None and not np.isnan(knee):
                    kav = (knee - s["prev_knee"]) * fps
            s["prev_ankle"] = (ax, ay)
            if not np.isnan(knee):
                s["prev_knee"] = knee
            if not fix_state_order:  # v1's order: recompute against the just-overwritten prev
                spd = kav = np.nan
                if s["prev_ankle"] is not None and ac >= CONF_THRESH:
                    spd = math.hypot(ax - s["prev_ankle"][0], ay - s["prev_ankle"][1]) * fps
                if s["prev_knee"] is not None and not np.isnan(knee):
                    kav = (knee - s["prev_knee"]) * fps
            speeds[leg] = (spd, kav)
        if cooldown > 0:
            cooldown -= 1
            continue
        fired = any(
            (not np.isnan(spd) and spd > KICK_PPS) or (not np.isnan(kav) and abs(kav) > KNEE_DPS)
            for spd, kav in speeds.values()
        )
        if fired:
            cands.append(f)
            cooldown = cooldown_frames
    return cands


def v1_fallback_frame(px: np.ndarray, fps: float) -> int:
    """_fallback_peak_frame: the frame with the highest single-step ankle speed, either leg."""
    best, best_f = -1.0, -1
    for leg in (R, L):
        a = px[:, leg[2]]
        d = np.hypot(np.diff(a[:, 1]), np.diff(a[:, 0])) * fps
        d[np.isnan(d) | (a[1:, 2] < CONF_THRESH)] = -1
        if d.max() > best:
            best, best_f = float(d.max()), int(d.argmax()) + 1
    return best_f


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    man = manifest()
    model = MoveNet()
    lines, rows = [], []
    for c in CLIPS:
        r = man[c]
        src = source_path(video_id(r["url"]))
        info = probe(src)
        w, h, fps = info.width, info.height, info.r_frame_rate
        rgb, bgr, lb = run_source(model, src)

        fixed_xy = lb.to_source(rgb[..., [1, 0]])
        v1_px = to_v1_pixels(rgb, w, h)
        v1_xy = v1_px[..., [1, 0]]
        conf = rgb[..., 2]
        diffs = np.concatenate(
            [knee_angles(v1_xy, conf, leg) - knee_angles(fixed_xy, conf, leg) for leg in (L, R)]
        )
        diffs = diffs[np.isfinite(diffs)]

        moved = np.linalg.norm(lb.to_source(bgr[..., [1, 0]]) - fixed_xy, axis=-1)[
            (conf >= CONF_THRESH) & (bgr[..., 2] >= CONF_THRESH)
        ]
        torso = np.nanmedian(np.linalg.norm(fixed_xy[:, 5] - fixed_xy[:, 11], axis=-1))

        # C uses v1's whole pipeline input: BGR frames and v1's mapping.
        v1_bgr_px = to_v1_pixels(bgr, w, h)
        as_written = v1_candidates(v1_bgr_px, fps, fix_state_order=False)
        fixed_order = v1_candidates(v1_bgr_px, fps, fix_state_order=True)
        fb = v1_fallback_frame(v1_bgr_px, fps)
        start_f, end_f = float(r["start_s"]) * fps, float(r["end_s"]) * fps
        last = fixed_order[-1] if fixed_order else -1
        rows.append(
            {
                "clip_id": c,
                "source": src.name,
                "frames": len(rgb),
                "fps": f"{fps:.3f}",
                "kick_window_frames": f"{start_f:.0f}-{end_f:.0f}",
                "as_written_candidates": len(as_written),
                "fallback_frame": fb,
                "fallback_in_kick_window": start_f <= fb < end_f,
                "fixed_order_candidates": len(fixed_order),
                "fixed_order_frames": " ".join(map(str, fixed_order)),
                "cap_reached_before_window_start": len(fixed_order) >= MAX_KICKS and last < start_f,
            }
        )
        lines.append(
            f"| {c} | {w}x{h} | {len(diffs)} | {np.median(diffs):+.1f} | "
            f"{np.percentile(np.abs(diffs), 90):.1f} | "
            f"{conf[conf > 0].mean():.3f} / {bgr[..., 2][bgr[..., 2] > 0].mean():.3f} | "
            f"{np.median(moved) / torso:.2f} |"
        )
        print(c, rows[-1], flush=True)

    with (OUT / "v1_candidates.csv").open("w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wtr.writeheader()
        wtr.writerows(rows)

    md = [
        "# Phase 1 audit follow-ups on real clips",
        "",
        "Generated by `tools/audit_v1/real_clip_followups.py` with v1's own MoveNet weights",
        "(`3.tflite`, input 192x192, i.e. SinglePose Lightning) on the full source videos.",
        "Knee angles use both legs on every frame where hip, knee and ankle all have score >= 0.2.",
        "",
        "## A. Letterbox mapping (2.2) and B. colour order",
        "",
        "| clip | source size | knee angles | median v1 - fixed (deg) | 90th pct abs diff (deg) "
        "| mean score RGB / BGR | median BGR shift (torso lengths) |",
        "|---|---|---|---|---|---|---|",
        *lines,
        "",
        "## C. v1 kick selection (2.5)",
        "",
        "Replay of `analyze_video_file` on v1-style input (BGR, v1 mapping). 'As written' keeps",
        "v1's state-update order; 'fixed order' computes speeds before overwriting the previous",
        "frame's values. Kick windows are Claude's curation windows from the manifest (they",
        "contain the kick; they are not ground truth).",
        "",
        "| clip | frames | as-written candidates | fallback frame | fallback in kick window "
        "| fixed-order candidates | cap reached before window start |",
        "|---|---|---|---|---|---|---|",
        *[
            f"| {r['clip_id']} | {r['frames']} | {r['as_written_candidates']} | "
            f"{r['fallback_frame']} | {r['fallback_in_kick_window']} | "
            f"{r['fixed_order_candidates']} | {r['cap_reached_before_window_start']} |"
            for r in rows
        ],
        "",
    ]
    (OUT / "phase1_followups.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
