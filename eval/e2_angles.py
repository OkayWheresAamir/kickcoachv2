"""E2: how close are the pipeline's angles to the angles from Aamir's labels?

The truth is each angle computed from Aamir's confirmed keypoints at his labelled frame
(contact, or max backswing) with the pipeline's own geometry code (eval plan, "Ground
truth"). Error = system - truth, in degrees; bias is the mean signed error.

Two views:
1. **End to end**: what each system reports to a user, events and all.
   - ``v1_shipped``: the hackathon version replayed (eval/v1_replica.py): MoveNet fed BGR,
     v1's letterbox-ignoring mapping, causal EMA, and the frame with the highest ankle speed
     as "the kick". It has no backswing feature, and its trunk sign follows the image, not
     the kick direction (V1_AUDIT 2.7).
   - ``v2_pipeline``: kickcoach.pipeline's features (results/pipeline/), on its own events.
2. **At the labelled frames**: events taken from the labels, so only keypoints and cleaning
   differ. A 2x2 of model (MoveNet, RTMPose) x cleaning (raw = no filter and no swap fix,
   the plan's "v1-style"; clean = kickcoach.signal with a break at the labelled contact), plus
   v1's own measurement (its keypoints and EMA).

The winner's curse (JOURNAL Phase 3 §26): ``max_knee_flexion_backswing`` is the smallest
filtered knee angle in the backswing valley, while the truth is the angle at the one frame
Aamir picked. So feature - truth = [series(pipeline frame) - series(label frame)] +
[series(label frame) - truth]. When the labelled frame is inside the search window, the
first term can't be positive. It mixes noise (the minimum of a noisy series sits below its
true minimum) with any frame-choice difference between Aamir and the pipeline; with one
labelled frame per backswing, the two can't be told apart.

Pre-fill anchoring (DEC-013): labels started at RTMPose's raw points, so an angle whose
points were all left in place has zero error for raw RTMPose by construction. Each row
records how many of its points were moved, and the summary repeats the main rows on angles
with at least one moved point.

Cutoff sensitivity (handoff D7): the v2 rows rerun with the low-pass cutoff forced to each
of several values on every clip. The default (DEC-019) is 0.15 cycles/frame, or 18 Hz where
the frame rate is trusted.

    uv run python -m eval.e2_angles
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eval import v1_replica
from eval.e1_pose import UNMOVED_PX, ci95, fmt, frame_angles, load_manifest, select, table
from eval.e1_pose import write_csv as _write_csv
from kickcoach.events import ball_detector
from kickcoach.ingest import load_meta
from kickcoach.labels import schema
from kickcoach.pipeline import analyse
from kickcoach.pose import cache, skeleton
from kickcoach.signal import clean

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FRAMES = ROOT / "data" / "frames"

# feature id -> (labelled frame, frame_angles key)
ANGLES = {
    "knee_angle_contact": ("contact_frame", "knee_angle"),
    "max_knee_flexion_backswing": ("max_backswing_frame", "knee_angle"),
    "trunk_lean_contact": ("contact_frame", "trunk_lean"),
}
NEEDED = {
    "knee_angle": lambda k: (f"{k}_hip", f"{k}_knee", f"{k}_ankle"),
    "trunk_lean": lambda k: ("left_hip", "right_hip", "left_shoulder", "right_shoulder"),
}
AT_FRAME_SYSTEMS = ("v1_measure", "movenet_raw", "movenet_clean", "rtmpose_raw", "rtmpose_clean")
END_TO_END = ("v1_shipped", "v2_pipeline")
CUTOFFS = (None, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3)
N_BOOT = 2000
SEED = 0


def write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        _write_csv(path, rows)


def labelled_clips(labels: dict, manifest: dict) -> list[str]:
    return sorted(
        c
        for c, lab in labels.items()
        if lab.get("purpose") == "eval"
        and lab["clip"].get("kicking_leg") in schema.SIDES
        and lab["clip"].get("kick_direction") in schema.SIDES
        and lab["clip"].get("contact_frame") is not None
        and c in manifest
    )


def truth(lab: dict) -> dict[str, dict]:
    """Per feature: the labelled frame, the angle from the labels, and pre-fill anchoring."""
    clip = lab["clip"]
    leg, facing = clip["kicking_leg"], 1 if clip["kick_direction"] == "right" else -1
    out = {}
    for fid, (frame_key, key) in ANGLES.items():
        t = clip.get(frame_key)
        entry = lab["frames"].get(str(t)) if t is not None else None
        if not entry or not entry.get("confirmed"):
            continue
        pts = entry["points"]
        xy = {n: np.array([p["x"], p["y"]]) for n, p in pts.items() if p["v"] > schema.UNKNOWN}
        value = frame_angles(xy, leg, facing)[key]
        if not np.isfinite(value):
            continue
        needed = NEEDED[key](leg)
        pre = entry.get("prefill", {})
        moved = [
            bool(pre.get(n))
            and np.hypot(pts[n]["x"] - pre[n][0], pts[n]["y"] - pre[n][1]) > UNMOVED_PX
            for n in needed
            if n in pts
        ]
        out[fid] = {
            "frame": t,
            "truth": float(value),
            "points_moved": int(sum(moved)),
            "points_needed": len(needed),
            "all_visible": all(pts.get(n, {}).get("v") == schema.VISIBLE for n in needed),
        }
    return out


def angles_at(xy: np.ndarray, t: int, leg: str, facing: int) -> dict[str, float]:
    """frame_angles at frame ``t`` of a [T, K, 2+] array."""
    return frame_angles(
        {n: xy[t, skeleton.INDEX[n], :2].astype(float) for n in schema.POINTS}, leg, facing
    )


def _clip_inputs(cid: str, meta: dict) -> dict:
    spf = meta["seconds_per_frame"] if meta.get("fps_trusted") else None
    return {"spf": spf, "fps": meta["playback_fps"]}


def at_frame_values(cid: str, lab: dict, meta: dict, poses: dict, cpf="auto") -> dict:
    """{system: {feature: value}} with events from the labels."""
    clip = lab["clip"]
    leg, facing = clip["kicking_leg"], 1 if clip["kick_direction"] == "right" else -1
    c = clip["contact_frame"]
    spf = _clip_inputs(cid, meta)["spf"]
    tracks = {
        "movenet_raw": poses["movenet"][..., :2],
        "movenet_clean": clean(
            poses["movenet"], seconds_per_frame=spf, breaks=(c,), target_cpf=cpf
        ).xy,
        "rtmpose_raw": poses["rtmpose"][..., :2],
        "rtmpose_clean": clean(
            poses["rtmpose"], seconds_per_frame=spf, breaks=(c,), target_cpf=cpf
        ).xy,
    }
    v1 = v1_replica.series(poses["movenet_v1"], meta["playback_fps"])
    out: dict[str, dict] = {s: {} for s in AT_FRAME_SYSTEMS}
    for fid, (frame_key, key) in ANGLES.items():
        t = clip.get(frame_key)
        if t is None:
            continue
        for s, xy in tracks.items():
            out[s][fid] = angles_at(xy, t, leg, facing)[key]
        out["v1_measure"][fid] = v1.knee_ema[leg][t] if key == "knee_angle" else v1.trunk_ema[t]
    return out


def pipeline_values(res: dict) -> dict[str, dict]:
    """{feature: {"value", "frame", "status", "reason"}} from a pipeline result."""
    feats = {f["id"]: f for f in res["features"]}
    out = {}
    for fid in ANGLES:
        f = feats.get(fid)
        out[fid] = {
            "value": f.get("value") if f and f["status"] == "measured" else None,
            "frame": f.get("frame") if f else None,
            "reason": f.get("reason") if f else "clip refused by the quality gate",
        }
    return out


def end_to_end_values(cid: str, meta: dict, poses: dict, res: dict) -> dict[str, dict]:
    v1 = v1_replica.kick(poses["movenet_v1"], meta["playback_fps"])
    v1_vals = {
        "knee_angle_contact": {"value": v1.knee_angle, "frame": v1.frame, "leg": v1.leg},
        "max_knee_flexion_backswing": {
            "value": None,
            "frame": None,
            "reason": "v1 has no backswing feature",
        },
        "trunk_lean_contact": {"value": v1.trunk, "frame": v1.frame, "leg": v1.leg},
    }
    return {"v1_shipped": v1_vals, "v2_pipeline": pipeline_values(res)}


def curse_row(cid: str, lab: dict, meta: dict, rtm: np.ndarray, res: dict, tr: dict) -> list[dict]:
    """Decompose each v2 end-to-end error into an event part and a measurement part, on the
    series the pipeline used (filtered with the break at the *detected* contact)."""
    ev = res["events"]
    clip = lab["clip"]
    leg, facing = clip["kicking_leg"], 1 if clip["kick_direction"] == "right" else -1
    if ev["contact_frame"] is None or ev["kicking_leg"] != leg:
        return []
    spf = _clip_inputs(cid, meta)["spf"]
    xy = clean(rtm, seconds_per_frame=spf, breaks=(ev["contact_frame"],)).xy
    feats = pipeline_values(res)
    rows = []
    for fid, (_, key) in ANGLES.items():
        f, t_true = feats[fid], tr.get(fid)
        if t_true is None or f["value"] is None:
            continue
        at_label = angles_at(xy, t_true["frame"], leg, facing)[key]
        rows.append(
            {
                "clip_id": cid,
                "feature": fid,
                "label_frame": t_true["frame"],
                "pipeline_frame": f["frame"],
                "frame_offset": f["frame"] - t_true["frame"],
                "truth": round(t_true["truth"], 2),
                "feature_value": f["value"],
                "series_at_label_frame": round(float(at_label), 2),
                "event_part": round(f["value"] - float(at_label), 2),
                "measurement_part": round(float(at_label) - t_true["truth"], 2),
                "total_err": round(f["value"] - t_true["truth"], 2),
                "label_frame_in_window": (
                    ev["backswing_window_start"] is not None
                    and ev["backswing_window_start"] <= t_true["frame"] <= ev["contact_frame"]
                )
                if fid == "max_knee_flexion_backswing"
                else None,
            }
        )
    return rows


def error_rows(system: str, values: dict, cid: str, split: str, tr: dict, view: str) -> list[dict]:
    rows = []
    for fid in ANGLES:
        t = tr.get(fid)
        if t is None:
            continue
        v = values.get(fid)
        val = v.get("value") if isinstance(v, dict) else v
        ok = val is not None and np.isfinite(val)
        rows.append(
            {
                "view": view,
                "system": system,
                "clip_id": cid,
                "split": split,
                "feature": fid,
                "label_frame": t["frame"],
                "system_frame": v.get("frame") if isinstance(v, dict) else t["frame"],
                "truth": round(t["truth"], 2),
                "value": round(float(val), 2) if ok else None,
                "err": round(float(val) - t["truth"], 2) if ok else None,
                "points_moved": t["points_moved"],
                "points_needed": t["points_needed"],
                "all_visible": t["all_visible"],
                "reason": v.get("reason") if isinstance(v, dict) and not ok else None,
            }
        )
    return rows


def summary(rows: list[dict]) -> dict:
    err = np.array([r["err"] for r in rows if r["err"] is not None], dtype=float)
    n = len(rows)
    if len(err) == 0:
        return {
            "n": n,
            "answered": 0,
            "mae": np.nan,
            "bias": np.nan,
            "median_abs": np.nan,
            "max_abs": np.nan,
        }
    return {
        "n": n,
        "answered": len(err),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "median_abs": float(np.median(np.abs(err))),
        "max_abs": float(np.max(np.abs(err))),
    }


def bootstrap_mae(rows: list[dict], systems: tuple[str, str], n_boot=N_BOOT, seed=SEED) -> dict:
    """Clip-bootstrap 95% CIs for each system's MAE and for the paired difference, on clips
    where both answered (one value per clip and feature)."""
    by = {
        s: {r["clip_id"]: abs(r["err"]) for r in rows if r["system"] == s and r["err"] is not None}
        for s in systems
    }
    clips = sorted(set(by[systems[0]]) & set(by[systems[1]]))
    if not clips:
        return {}
    a = np.array([by[systems[0]][c] for c in clips])
    b = np.array([by[systems[1]][c] for c in clips])
    draws = np.random.default_rng(seed).integers(0, len(clips), size=(n_boot, len(clips)))
    return {
        "n_clips": len(clips),
        "mae": (float(a.mean()), float(b.mean())),
        "ci": (ci95(a[draws].mean(1)), ci95(b[draws].mean(1))),
        "diff": float((a - b).mean()),
        "diff_ci": ci95((a[draws] - b[draws]).mean(1)),
    }


def sensitivity(labels, manifest, clips, metas, poses, cands) -> list[dict]:
    rows = []
    for cpf in CUTOFFS:
        e2e, at = [], []
        for cid in clips:
            tr = truth(labels[cid])
            res = analyse(poses[cid]["rtmpose"], metas[cid], cands[cid], target_cpf=cpf)
            e2e += error_rows(
                "v2_pipeline", pipeline_values(res), cid, manifest[cid]["split"], tr, "e2e"
            )
            vals = at_frame_values(cid, labels[cid], metas[cid], poses[cid], cpf=cpf)[
                "rtmpose_clean"
            ]
            at += error_rows("rtmpose_clean", vals, cid, manifest[cid]["split"], tr, "at_label")
        for view, rs in (("end_to_end", e2e), ("at_labelled_frames", at)):
            for fid in ANGLES:
                for split in ("tune", "heldout"):
                    s = summary(select(rs, feature=fid, split=split))
                    rows.append(
                        {
                            "cutoff_cpf": "none" if cpf is None else cpf,
                            "view": view,
                            "feature": fid,
                            "split": split,
                            **s,
                        }
                    )
    return rows


def contact_step(labels: dict, clips: list[str]) -> list[dict]:
    """Knee angle at contact minus at contact - 1, both from the labels."""
    out = []
    for cid in clips:
        clip = labels[cid]["clip"]
        c = clip["contact_frame"]
        entries = [labels[cid]["frames"].get(str(t)) for t in (c - 1, c)]
        if not all(e and e.get("confirmed") for e in entries):
            continue
        facing = 1 if clip["kick_direction"] == "right" else -1
        k = [
            frame_angles(
                {n: np.array([p["x"], p["y"]]) for n, p in e["points"].items() if p["v"]},
                clip["kicking_leg"],
                facing,
            )["knee_angle"]
            for e in entries
        ]
        if all(np.isfinite(k)):
            out.append({"clip_id": cid, "knee_change": float(k[1] - k[0])})
    return out


def run(labels: dict, manifest: dict, *, with_sensitivity: bool = True) -> dict:
    clips = labelled_clips(labels, manifest)
    metas = {c: load_meta(FRAMES / c) for c in clips}
    poses = {
        c: {m: cache.load(m, c) for m in ("movenet", "rtmpose", v1_replica.CACHE_MODEL)}
        for c in clips
    }
    for c in clips:
        poses[c]["movenet_v1"] = poses[c].pop(v1_replica.CACHE_MODEL)
    cands = {
        c: ball_detector.load(c) if ball_detector.cache_path(c).exists() else None for c in clips
    }
    results = {c: json.loads((RESULTS / "pipeline" / f"{c}.json").read_text()) for c in clips}
    rows, curse = [], []
    for cid in clips:
        tr = truth(labels[cid])
        split = manifest[cid]["split"]
        for s, vals in end_to_end_values(cid, metas[cid], poses[cid], results[cid]).items():
            rows += error_rows(s, vals, cid, split, tr, "end_to_end")
        for s, vals in at_frame_values(cid, labels[cid], metas[cid], poses[cid]).items():
            rows += error_rows(s, vals, cid, split, tr, "at_labelled_frames")
        curse += [
            {"split": split, **r}
            for r in curse_row(
                cid, labels[cid], metas[cid], poses[cid]["rtmpose"], results[cid], tr
            )
        ]
    v1_kicks = {c: v1_replica.kick(poses[c]["movenet_v1"], metas[c]["playback_fps"]) for c in clips}
    return {
        "clips": clips,
        "rows": rows,
        "curse": curse,
        "sensitivity": sensitivity(labels, manifest, clips, metas, poses, cands)
        if with_sensitivity
        else [],
        "v1_kicks": v1_kicks,
        "contact_step": contact_step(labels, clips),
        "labels": labels,
    }


# --- report ---------------------------------------------------------------------------


def _summary_table(rows: list[dict], systems, view: str) -> list[list[str]]:
    body = []
    for fid in ANGLES:
        for s in systems:
            for split in ("all", "tune", "heldout"):
                rs = select(rows, view=view, system=s, feature=fid)
                if split != "all":
                    rs = select(rs, split=split)
                m = summary(rs)
                if m["n"] == 0:
                    continue
                body.append(
                    [
                        fid,
                        s,
                        split,
                        f"{m['answered']}/{m['n']}",
                        fmt(m["mae"], nd=1),
                        fmt(m["bias"], nd=1),
                        fmt(m["median_abs"], nd=1),
                        fmt(m["max_abs"], nd=1),
                    ]
                )
    return body


HDR = [
    "feature",
    "system",
    "split",
    "answered/n",
    "MAE °",
    "bias °",
    "median abs err °",
    "max abs err °",
]


def report(out: dict, manifest: dict) -> str:
    rows, curse = out["rows"], out["curse"]
    clips = out["clips"]
    n_tune = sum(manifest[c]["split"] == "tune" for c in clips)
    L = [
        "# E2: angle accuracy against the labels",
        "",
        "Generated by `uv run python -m eval.e2_angles`. Truth = the angle from Aamir's confirmed",
        "keypoints at his labelled frame, computed with the pipeline's geometry code. Error =",
        "system − truth; bias = mean signed error. One value per clip and feature.",
        "",
        f"**n:** {len(clips)} clips ({n_tune} tuning, {len(clips) - n_tune} held-out) with a",
        "labelled kicking leg, kick direction and confirmed keypoints at contact and max",
        "backswing. c20, c23, c24 and c25 have no labelled events yet, so they aren't scored.",
        "",
        "## 1. End to end: what each system reports",
        "",
        "`v1_shipped` = the hackathon version replayed (MoveNet fed BGR, v1's letterbox-ignoring",
        "mapping, causal EMA, kick = the frame with the highest ankle speed; trunk signed by image",
        "left/right). `v2_pipeline` = `kickcoach.pipeline` on its own detected events.",
        "",
    ]
    L += table(HDR, _summary_table(rows, END_TO_END, "end_to_end"))
    L += [
        "",
        "Clip-bootstrap 95% CIs (2,000 resamples of clips; paired, clips where both answered):",
        "",
    ]
    body = []
    for fid in ("knee_angle_contact", "trunk_lean_contact"):
        b = bootstrap_mae(select(rows, view="end_to_end", feature=fid), END_TO_END)
        if b:
            body.append(
                [
                    fid,
                    str(b["n_clips"]),
                    f"{b['mae'][0]:.1f} ({b['ci'][0][0]:.1f}–{b['ci'][0][1]:.1f})",
                    f"{b['mae'][1]:.1f} ({b['ci'][1][0]:.1f}–{b['ci'][1][1]:.1f})",
                    f"{b['diff']:+.1f} ({b['diff_ci'][0]:+.1f} to {b['diff_ci'][1]:+.1f})",
                ]
            )
    L += table(
        ["feature", "clips", "v1_shipped MAE (CI)", "v2_pipeline MAE (CI)", "v1 − v2 (CI)"], body
    )
    k = out["v1_kicks"]
    lab = out["labels"]
    off = [
        (c, k[c].frame - lab[c]["clip"]["contact_frame"], k[c].leg == lab[c]["clip"]["kicking_leg"])
        for c in clips
        if k[c] is not None
    ]
    L += [
        "",
        'Why v1 is off: its "kick" frame is the highest one-step ankle speed in the clip. It',
        f"landed within ±2 frames of the labelled contact on {sum(abs(o) <= 2 for _, o, _ in off)}"
        f" of {len(off)} clips and picked the labelled kicking leg on {sum(g for *_, g in off)};",
        "offsets (frames): "
        + ", ".join(f"{c} {o:+d}{'' if g else ' (wrong leg)'}" for c, o, g in off)
        + ".",
        "",
        "## 2. At the labelled frames: model × cleaning",
        "",
        "Events from the labels, so only keypoints and cleaning differ. `raw` = the model's",
        'points as output (no filtering, no swap fix: the plan\'s "v1-style"); `clean` =',
        "`kickcoach.signal` (swap fix, glitch mask, gap fill, low-pass with a break at the",
        "labelled contact). `v1_measure` = v1's keypoints and EMA at the labelled frame.",
        "",
    ]
    L += table(HDR, _summary_table(rows, AT_FRAME_SYSTEMS, "at_labelled_frames"))
    sub = [r for r in rows if r["points_moved"] > 0]
    n_anch = len({(r["clip_id"], r["feature"]) for r in rows if r["points_moved"] == 0})
    n_all = len({(r["clip_id"], r["feature"]) for r in rows})
    L += [
        "",
        "### Pre-fill anchoring",
        "",
        f"{n_anch} of {n_all} labelled angles had every point left at RTMPose's pre-fill, so raw",
        "RTMPose scores 0° there by construction. Repeated on angles with at least one moved",
        "point (this flatters RTMPose less; the moved points are where it was wrong):",
        "",
    ]
    body = []
    for view, systems in (("end_to_end", END_TO_END), ("at_labelled_frames", AT_FRAME_SYSTEMS)):
        for fid in ANGLES:
            for s in systems:
                m = summary(select(sub, view=view, system=s, feature=fid))
                if m["answered"]:
                    body.append(
                        [
                            view,
                            fid,
                            s,
                            f"{m['answered']}/{m['n']}",
                            fmt(m["mae"], nd=1),
                            fmt(m["bias"], nd=1),
                        ]
                    )
    L += table(["view", "feature", "system", "answered/n", "MAE °", "bias °"], body)

    L += [
        "",
        "## 3. Where v2's end-to-end error comes from (and the winner's curse)",
        "",
        "Each v2 error splits into an **event part** (the series at the pipeline's frame minus",
        "the series at the labelled frame) and a **measurement part** (the series at the",
        "labelled frame minus the truth). For max knee flexion the event part can't be positive",
        "when the labelled frame is inside the search window: the pipeline takes the minimum.",
        "",
    ]
    body = []
    for fid in ANGLES:
        rs = select(curse, feature=fid)
        if not rs:
            continue
        ev = np.array([r["event_part"] for r in rs])
        me = np.array([r["measurement_part"] for r in rs])
        tot = np.array([r["total_err"] for r in rs])
        offs = np.array([r["frame_offset"] for r in rs])
        body.append(
            [
                fid,
                str(len(rs)),
                f"{tot.mean():+.1f}",
                f"{np.abs(tot).mean():.1f}",
                f"{ev.mean():+.1f}",
                f"{me.mean():+.1f}",
                f"{np.abs(me).mean():.1f}",
                f"{int((offs == 0).sum())}",
                f"{np.median(offs):+.0f}",
            ]
        )
    L += table(
        [
            "feature",
            "n",
            "bias °",
            "MAE °",
            "event part (mean) °",
            "measurement part (mean) °",
            "measurement MAE °",
            "frame exact",
            "median frame offset",
        ],
        body,
    )
    bs = select(curse, feature="max_knee_flexion_backswing")
    if bs:
        inside = [r for r in bs if r["label_frame_in_window"]]
        L += [
            "",
            f"Max knee flexion: the labelled frame was inside the pipeline's search window on "
            f"{len(inside)} of {len(bs)} clips; there the event part was ≤ 0 on "
            f"{sum(r['event_part'] <= 0 for r in inside)}, mean "
            f"{np.mean([r['event_part'] for r in inside]):+.1f}°. Per clip:",
            "",
        ]
        L += table(
            [
                "clip",
                "label frame",
                "pipeline frame",
                "truth °",
                "feature °",
                "series at label frame °",
                "event part °",
                "measurement part °",
            ],
            [
                [
                    r["clip_id"],
                    str(r["label_frame"]),
                    str(r["pipeline_frame"]),
                    f"{r['truth']:.1f}",
                    f"{r['feature_value']:.1f}",
                    f"{r['series_at_label_frame']:.1f}",
                    f"{r['event_part']:+.1f}",
                    f"{r['measurement_part']:+.1f}",
                ]
                for r in bs
            ],
        )
    step = out["contact_step"]
    if step:
        d = np.array([s["knee_change"] for s in step])
        L += [
            "",
            "Knee angle at contact is sensitive to the contact frame. From the labels alone, on",
            f"the {len(step)} clips with contact − 1 also confirmed, the knee angle changes by a "
            "median "
            f"{np.median(d):+.1f}° (range {d.min():+.1f} to {d.max():+.1f}) over the last frame "
            "before contact ("
            + ", ".join(f"{s['clip_id']} {s['knee_change']:+.1f}" for s in step)
            + "). So a one-frame disagreement about contact, which DEC-015 expects between",
            "reasonable labellers, already moves this feature by a few degrees.",
        ]

    sens = out["sensitivity"]
    if sens:
        L += [
            "",
            "## 4. Sensitivity to the low-pass cutoff (D7)",
            "",
            "v2 rerun with the cutoff forced to each value on every clip (cycles per frame;",
            "`none` = unfiltered). Default: 0.15, or 18 Hz on the two trusted high-speed clips",
            "(0.06 for c13, 0.075 for c19). The default was chosen on the tuning split",
            "(`eval/tune_cutoff.py`), so only the held-out rows are untouched.",
            "MAE °, tune / held-out:",
            "",
        ]
        body = []
        for view in ("end_to_end", "at_labelled_frames"):
            for cpf in CUTOFFS:
                key = "none" if cpf is None else cpf
                cells = [view, str(key)]
                for fid in ANGLES:
                    t = select(sens, cutoff_cpf=key, view=view, feature=fid, split="tune")[0]
                    h = select(sens, cutoff_cpf=key, view=view, feature=fid, split="heldout")[0]
                    cells.append(f"{fmt(t['mae'], nd=1)} / {fmt(h['mae'], nd=1)}")
                body.append(cells)
        L += table(["view", "cutoff", *ANGLES], body)
    return "\n".join(L) + "\n"


def main() -> None:
    labels = schema.load_all()
    manifest = load_manifest()
    out = run(labels, manifest)
    write_csv(RESULTS / "e2.csv", out["rows"])
    write_csv(RESULTS / "e2_decomposition.csv", out["curse"])
    write_csv(RESULTS / "e2_cutoff.csv", out["sensitivity"])
    md = report(out, manifest)
    (RESULTS / "e2_summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
