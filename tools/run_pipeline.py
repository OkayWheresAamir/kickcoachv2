"""Run the measurement pipeline on every manifest clip from the cached RTMPose poses and ball
candidates. Writes results/pipeline/<clip>.json and results/pipeline_summary.csv.

Nothing here reads the labels: the pipeline sees only what a user's upload would give it.
The labelled events are joined into the summary afterwards, for comparison.

uv run python tools/run_pipeline.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from kickcoach.events import ball_detector
from kickcoach.ingest import load_meta
from kickcoach.labels import schema
from kickcoach.pipeline import analyse
from kickcoach.pose import cache

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "pipeline"
POSE_MODEL = "rtmpose"  # chosen by E1 (DEC-016)


def main() -> None:
    with open(ROOT / "data" / "manifest.csv", newline="") as f:
        manifest = list(csv.DictReader(f))
    labels = schema.load_all()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in manifest:
        cid = m["clip_id"]
        meta = load_meta(ROOT / "data" / "frames" / cid)
        cands = ball_detector.load(cid) if ball_detector.cache_path(cid).exists() else None
        res = analyse(cache.load(POSE_MODEL, cid), meta, cands)
        res = {"clip_id": cid, "pose_model": POSE_MODEL, **res}
        (OUT / f"{cid}.json").write_text(json.dumps(res, indent=1) + "\n")
        ev, q = res["events"], res["quality"]
        lab = labels.get(cid, {}).get("clip", {})
        feats = {f["id"]: f for f in res["features"]}
        rows.append(
            {
                "clip_id": cid,
                "split": m["split"],
                "purpose": m["purpose"],
                "view_ok_human": lab.get("view_ok_human"),
                "accepted": q["accepted"],
                "reasons": " | ".join(q["reasons"]),
                "viewpoint_ratio": q["viewpoint_ratio"],
                "keypoint_score": q["keypoint_score"],
                "kicking_leg": ev["kicking_leg"],
                "kicking_leg_label": lab.get("kicking_leg"),
                "kick_direction": ev["kick_direction"],
                "kick_direction_label": lab.get("kick_direction"),
                "contact_frame": ev["contact_frame"],
                "contact_label": lab.get("contact_frame"),
                "contact_method": ev["contact_method"],
                "contact_confidence": ev["contact_confidence"],
                "max_backswing_frame": ev["max_backswing_frame"],
                "max_backswing_label": lab.get("max_backswing_frame"),
                **{
                    fid: (f.get("value") if f["status"] == "measured" else "not_measured")
                    for fid, f in feats.items()
                },
            }
        )
        print(
            f"{cid}: {'ACCEPT' if q['accepted'] else 'REFUSE'} view {q['viewpoint_ratio']} "
            f"leg {ev['kicking_leg']}/{lab.get('kicking_leg')} contact {ev['contact_frame']}/"
            f"{lab.get('contact_frame')} ({ev['contact_method']}, {ev['contact_confidence']}) "
            f"backswing {ev['max_backswing_frame']}/{lab.get('max_backswing_frame')}"
        )
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(ROOT / "results" / "pipeline_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
