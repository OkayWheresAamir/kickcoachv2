"""Trim every manifest clip and extract its frames and metadata.

Writes, per clip (all gitignored):
  data/frames/<clip_id>/000000.jpg ... and meta.json
  data/raw/clips/<clip_id>.mp4   (the trimmed window, frame-aligned with the JPEGs)
and the committed summary data/ingest_summary.csv. Also fills the manifest's playback_fps
column, the only manifest column this script touches.

    uv run python tools/ingest_clips.py [clip_id ...]
"""

from __future__ import annotations

import csv
import sys

from fetch_clips import source_path, video_id
from ytmeta import ROOT

from kickcoach.ingest import ingest_clip

MANIFEST = ROOT / "data" / "manifest.csv"
SUMMARY_COLS = [
    "clip_id",
    "codec",
    "width",
    "height",
    "r_frame_rate",
    "avg_frame_rate",
    "playback_fps",
    "n_frames",
    "constant_frame_rate",
    "repeated_frame_ratio",
    "fps_trusted",
    "fps_reason",
]


def main(only: list[str]) -> None:
    with MANIFEST.open() as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    summary_path = ROOT / "data" / "ingest_summary.csv"
    summary = {}
    if summary_path.exists():
        summary = {r["clip_id"]: r for r in csv.DictReader(summary_path.open())}

    for r in rows:
        if only and r["clip_id"] not in only:
            continue
        src = source_path(video_id(r["url"]))
        if src is None:
            print(f"{r['clip_id']}: source missing, run tools/fetch_clips.py")
            continue
        m = ingest_clip(
            src,
            clip_id=r["clip_id"],
            start_s=float(r["start_s"]),
            end_s=float(r["end_s"]),
            capture_fps_known=r["capture_fps_known"],
            capture_fps=float(r["capture_fps"]) if r["capture_fps"] else None,
            slow_motion=r["slow_motion"],
            frames_dir=ROOT / "data" / "frames" / r["clip_id"],
            clip_path=ROOT / "data" / "raw" / "clips" / f"{r['clip_id']}.mp4",
        )
        r["playback_fps"] = f"{m['playback_fps']:.3f}".rstrip("0").rstrip(".")
        s = m["stream"]
        summary[r["clip_id"]] = {
            "clip_id": r["clip_id"],
            "codec": s["codec"],
            "width": s["width"],
            "height": s["height"],
            "r_frame_rate": f"{s['r_frame_rate']:.3f}",
            "avg_frame_rate": f"{s['avg_frame_rate']:.3f}",
            "playback_fps": r["playback_fps"],
            "n_frames": m["n_frames"],
            "constant_frame_rate": m["constant_frame_rate"],
            "repeated_frame_ratio": f"{m['repeated_frame_ratio']:.3f}",
            "fps_trusted": m["fps_trusted"],
            "fps_reason": m["fps_reason"],
        }
        print(
            f"{r['clip_id']}: {m['n_frames']} frames {s['width']}x{s['height']} "
            f"r={s['r_frame_rate']:.3f} avg={s['avg_frame_rate']:.3f} "
            f"cfr={m['constant_frame_rate']} dup={m['repeated_frame_ratio']:.3f} "
            f"trusted={m['fps_trusted']} ({m['fps_reason']})",
            flush=True,
        )

    with MANIFEST.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        w.writeheader()
        w.writerows(summary[k] for k in sorted(summary))


if __name__ == "__main__":
    main(sys.argv[1:])
