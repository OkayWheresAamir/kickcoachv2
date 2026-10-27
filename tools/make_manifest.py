"""Create data/manifest.csv from the approved rows of data/candidates.csv (run once).

The held-out split is drawn with a fixed seed over *groups* of clips that share a kicker or a
kick, so no person or kick appears in both splits. Columns that only Aamir may fill
(kicking_leg, view_ok_human; DEC-007) are left blank. Refuses to overwrite an existing
manifest, because by then it holds human-entered values.

    uv run python tools/make_manifest.py
"""

from __future__ import annotations

import csv
import random

from ytmeta import ROOT, watch_url

APPROVED = [f"c{i:02d}" for i in range(1, 30)]  # Aamir approved c01-c29, 2026-10-27
# Dropped after download (contact only visible under a drawn overlay); see data/candidates.csv.
DROPPED = {"c21", "c22", "c29"}
SEED = 20261027
HELD_OUT_TARGET = 5  # side-on clips, per docs/handoff/04_EVALUATION_PLAN.md
# Clips that must share a split: same kicker, same kick, or a view twin (from candidates notes).
LINKED = [
    {"c01", "c26"},
    {"c02", "c27"},
    {"c07", "c28"},
    {"c04", "c05"},
    {"c11", "c12"},
    {"c23", "c24"},
]
COLUMNS = [
    "clip_id",
    "url",
    "start_s",
    "end_s",
    "playback_fps",
    "capture_fps_known",
    "capture_fps",
    "slow_motion",
    "kicking_leg",
    "level_guess",
    "view_ok_human",
    "notes",
    "split",
    "purpose",
]


def main() -> None:
    out = ROOT / "data" / "manifest.csv"
    if out.exists():
        raise SystemExit(f"{out} exists; edit it by hand instead of regenerating.")
    cands = {r["cand_id"]: r for r in csv.DictReader((ROOT / "data" / "candidates.csv").open())}
    rows = [cands[c] for c in APPROVED if c not in DROPPED]

    groups: list[set[str]] = [set(g) for g in LINKED]
    grouped = set().union(*groups)
    groups += [{r["cand_id"]} for r in rows if r["cand_id"] not in grouped]
    groups.sort(key=min)  # deterministic order before shuffling
    random.Random(SEED).shuffle(groups)

    held_out: set[str] = set()
    for g in groups:
        n_side_on = sum(cands[c]["group"] == "side_on" for c in held_out | g)
        if n_side_on <= HELD_OUT_TARGET:
            held_out |= g
        if sum(cands[c]["group"] == "side_on" for c in held_out) >= HELD_OUT_TARGET:
            break

    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            stated = r["capture_fps_stated"]
            w.writerow(
                {
                    "clip_id": r["cand_id"],
                    "url": watch_url(r["video_id"]),
                    "start_s": r["start_s"],
                    "end_s": r["end_s"],
                    "capture_fps_known": "y" if stated else "n",
                    "capture_fps": stated,
                    "slow_motion": r["slow_motion"],
                    "level_guess": r["level_guess"],
                    "notes": r["notes"],
                    "split": "heldout" if r["cand_id"] in held_out else "tune",
                    "purpose": "eval" if r["group"] == "side_on" else "gate_only",
                }
            )
    print(f"wrote {out}; held out: {sorted(held_out)}")


if __name__ == "__main__":
    main()
