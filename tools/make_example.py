"""Build the page's opt-in example (api/static/example.json): one public evaluation clip run
through the same steps as an upload, from the committed caches. No API calls.

    uv run python tools/make_example.py

The feedback is the committed LLM answer from results/feedback_llm/ (run with the key
blanked, so a cache miss falls back to the template instead of calling Gemini). The clip's
video is not used and nothing in the file is a pixel: the stick figure is the cleaned
keypoints, which are committed anyway (DEC-031).
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from kickcoach.events import ball_detector
from kickcoach.figure import frame_size, pose_figure
from kickcoach.ingest import load_meta
from kickcoach.llm import GeminiClient
from kickcoach.llm.gemini import DEV_MODEL
from kickcoach.pipeline import analyse_with_signal
from kickcoach.pose import cache
from kickcoach.session import POSE_MODEL, assess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "api" / "static" / "example.json"
# c07: an amateur, indoors, filmed side-on; the ball-motion contact matches the label
# (frame 76) and three fault rules fire, so the example shows every part of a result.
CLIP = "c07"


def build(clip_id: str = CLIP) -> dict:
    os.environ["GEMINI_API_KEY"] = ""  # cached answers only (DEC-032)
    meta = load_meta(ROOT / "data" / "frames" / clip_id)
    res, cleaned = analyse_with_signal(
        cache.load(POSE_MODEL, clip_id), meta, ball_detector.load(clip_id)
    )
    stored = json.loads((ROOT / "results" / "pipeline" / f"{clip_id}.json").read_text())
    if {k: v for k, v in stored.items() if k not in ("clip_id", "pose_model")} != res:
        raise SystemExit(f"{clip_id}: differs from results/pipeline/; rerun eval.run_all first")
    res = {"pose_model": POSE_MODEL, **res}
    res["figure"] = pose_figure(cleaned.xy, res, frame_size(meta))
    out = assess(res, client=GeminiClient(DEV_MODEL, ROOT / "results" / "feedback_llm"))
    with open(ROOT / "data" / "manifest.csv", newline="") as f:
        m = next(r for r in csv.DictReader(f) if r["clip_id"] == clip_id)
    return {
        "example": {
            "clip_id": clip_id,
            "source_url": m["url"],
            "window_s": [float(m["start_s"]), float(m["end_s"])],
            "level_guess": m["level_guess"],
            "note": "A public clip from the evaluation set, analysed from the committed pose "
            "and ball caches. Its feedback is the cached LLM answer that passed the validator.",
        },
        "session_id": None,
        **out,
        "history_delta": {},
    }


def main() -> None:
    d = build()
    OUT.write_text(json.dumps(d, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {d['example']['clip_id']}, "
          f"{len(d['faults'])} faults, feedback {d['feedback']['generated_by']}")  # fmt: skip


if __name__ == "__main__":
    main()
