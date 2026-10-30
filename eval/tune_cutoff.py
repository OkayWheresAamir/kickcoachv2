"""Choose the low-pass cutoff for clips without a trusted frame rate (smooth.CUTOFF_CPF_UNTRUSTED).

Tuning split only: held-out clips never tune a filter (eval plan). For each candidate cutoff,
RTMPose's points are cleaned (kickcoach.signal.clean), and the knee angle and trunk lean at
Aamir's labelled frames are compared with the angles from his labels. Run with and without a
break at the labelled contact frame (Nunome et al. 2006: filtering through impact distorts).
This is a tuning aid, not E2: E2 (Phase 4) uses detected events on every clip.

    uv run python -m eval.tune_cutoff
"""

from __future__ import annotations

import csv

import numpy as np

from eval.e1_pose import RESULTS, frame_angles, load_manifest
from kickcoach.labels import schema
from kickcoach.pose import cache, skeleton
from kickcoach.signal import clean

CANDIDATES = (None, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4)


def main() -> None:
    labels = schema.load_all()
    manifest = load_manifest()
    clips = [
        c
        for c, lab in labels.items()
        if manifest[c]["split"] == "tune"
        and lab["purpose"] == "eval"
        and manifest[c]["capture_fps_known"] != "y"
        and manifest[c]["slow_motion"] == "y"
        and any(e.get("confirmed") for e in lab["frames"].values())
    ]
    rows = []
    for cpf in CANDIDATES:
        for use_break in (False, True):
            errs: dict[str, list[float]] = {"knee_angle": [], "trunk_lean": []}
            for cid in clips:
                lab = labels[cid]
                clip = lab["clip"]
                facing = 1 if clip["kick_direction"] == "right" else -1
                c = clip["contact_frame"]
                out = clean(
                    cache.load("rtmpose", cid),
                    breaks=(c,) if use_break else (),
                    target_cpf=cpf,
                )
                for key, entry in lab["frames"].items():
                    if not entry.get("confirmed"):
                        continue
                    t = int(key)
                    pts = entry["points"]
                    truth = frame_angles(
                        {n: np.array([p["x"], p["y"]]) for n, p in pts.items() if p["v"] > 0},
                        clip["kicking_leg"],
                        facing,
                    )
                    pred = frame_angles(
                        {n: out.xy[t, skeleton.INDEX[n]] for n in schema.POINTS},
                        clip["kicking_leg"],
                        facing,
                    )
                    for a in errs:
                        errs[a].append(pred[a] - truth[a])
            for a, e in errs.items():
                e = np.array(e)
                e = e[~np.isnan(e)]
                rows.append(
                    {
                        "cutoff_cpf": "none" if cpf is None else cpf,
                        "break_at_contact": use_break,
                        "angle": a,
                        "n": len(e),
                        "mae_deg": round(float(np.mean(np.abs(e))), 3),
                        "bias_deg": round(float(np.mean(e)), 3),
                    }
                )
    path = RESULTS / "tune_cutoff.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"tuning clips ({len(clips)}): {', '.join(clips)}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
