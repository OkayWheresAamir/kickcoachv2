"""Phase 1 acceptance check: every manifest clip has frames + metadata, and every model's
cached pose array is [T, 23, 3] with T = the clip's frame count. Writes
results/phase1_pose_summary.csv (per model and clip: frames, share of body/foot points
present). Exit code 1 if anything is missing or mis-shaped.

    uv run python tools/check_phase1.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from kickcoach.pose import cache, skeleton
from kickcoach.pose.models import MODELS

ROOT = Path(__file__).resolve().parents[1]
BODY = [skeleton.INDEX[n] for n in skeleton.KEYPOINTS[5:17]]  # shoulders .. ankles
FEET = [skeleton.INDEX[n] for n in skeleton.KEYPOINTS[17:]]


def main() -> int:
    clips = [r["clip_id"] for r in csv.DictReader((ROOT / "data" / "manifest.csv").open())]
    problems, rows = [], []
    for c in clips:
        fdir = ROOT / "data" / "frames" / c
        if not (fdir / "meta.json").exists():
            problems.append(f"{c}: no meta.json")
            continue
        n = json.loads((fdir / "meta.json").read_text())["n_frames"]
        if len(list(fdir.glob("*.jpg"))) != n:
            problems.append(f"{c}: frame count differs from meta")
        for m in MODELS:
            try:
                p = cache.load(m, c)
            except (FileNotFoundError, ValueError) as e:
                problems.append(f"{m} {c}: {e}")
                continue
            if len(p) != n:
                problems.append(f"{m} {c}: {len(p)} poses for {n} frames")
            ok = np.isfinite(p[..., 0])
            rows.append(
                {
                    "model": m,
                    "clip_id": c,
                    "frames": len(p),
                    "body_present": f"{ok[:, BODY].mean():.3f}",
                    "feet_present": f"{ok[:, FEET].mean():.3f}",
                }
            )
    out = ROOT / "results" / "phase1_pose_summary.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for m in MODELS:
        mr = [r for r in rows if r["model"] == m]
        body = np.mean([float(r["body_present"]) for r in mr])
        feet = np.mean([float(r["feet_present"]) for r in mr])
        print(f"{m}: {len(mr)} clips, body points present {body:.1%}, foot points {feet:.1%}")
    print(f"{len(clips)} clips; problems: {problems or 'none'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
