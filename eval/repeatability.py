"""Labelling repeatability: how far the labeller's blind relabel lands from the first labels.

The eval plan's floor: no model can be shown to be more accurate than the labeller is
consistent. Each confirmed task in data/labels/repeat/ (``tools/label_tool.py --repeat``,
relabelled at least 48 h later without seeing the first labels) is compared with the first
label of the same frame: point distances in pixels and thigh lengths (E1's unit), and the
knee and trunk angles from each set of points with the pipeline's geometry code (E2's).

Both sessions start from the same RTMPose pre-fill (DEC-013), so a point left at the
pre-fill both times agrees by construction. The "moved" rows keep only points (and angles
with a point) moved in at least one session. They are the floor to quote.

    uv run python -m eval.repeatability
    # -> results/repeatability_points.csv, _angles.csv, _summary.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eval.e1_pose import PCK_THRESHOLD, UNMOVED_PX, fmt, frame_angles, table, write_csv
from kickcoach.labels import qa, schema

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ANGLES = ("knee_angle", "trunk_lean")
ANGLE_POINTS = {
    "knee_angle": lambda k: (f"{k}_hip", f"{k}_knee", f"{k}_ankle"),
    "trunk_lean": lambda k: ("left_hip", "right_hip", "left_shoulder", "right_shoulder"),
}


def load_sessions(labels_dir: Path = schema.LABELS_DIR) -> list[dict]:
    return [
        json.loads(p.read_text()) for p in sorted(schema.repeat_dir(labels_dir).glob("r*.json"))
    ]


def _moved(entry: dict, name: str) -> bool:
    p, pre = entry["points"].get(name), entry.get("prefill", {}).get(name)
    if not p or not pre:
        return True  # nothing to compare with, so it can't have agreed by construction
    return bool(np.hypot(p["x"] - pre[0], p["y"] - pre[1]) > UNMOVED_PX)


def _xy(points: dict) -> dict[str, np.ndarray]:
    return {n: np.array([p["x"], p["y"]]) for n, p in points.items() if p["v"] > schema.UNKNOWN}


def compare(labels: dict[str, dict], sessions: list[dict]) -> tuple[list[dict], list[dict]]:
    """Point rows and angle rows for every confirmed repeat task whose frame has a confirmed
    first label. Differences are repeat − first."""
    point_rows, angle_rows = [], []
    for s in sessions:
        for t in s["tasks"]:
            lab = labels.get(t["clip_id"])
            first = lab["frames"].get(str(t["frame"])) if lab else None
            if not t.get("confirmed") or not first or not first.get("confirmed"):
                continue
            key = {"session": s["session_id"], "clip_id": t["clip_id"], "frame": t["frame"]}
            thigh = qa.thigh_length(schema.points_array(first["points"]))
            for name in schema.POINTS:
                a, b = first["points"].get(name), t["points"].get(name)
                if not a or not b or schema.UNKNOWN in (a["v"], b["v"]):
                    continue
                d = float(np.hypot(b["x"] - a["x"], b["y"] - a["y"]))
                point_rows.append(
                    {
                        **key,
                        "point": name,
                        "diff_px": d,
                        "thigh_px": thigh,
                        "diff_thigh": d / thigh if thigh else np.nan,
                        "moved": _moved(first, name) or _moved(t, name),
                    }
                )
            clip = lab["clip"]
            leg, direction = clip.get("kicking_leg"), clip.get("kick_direction")
            if leg not in schema.SIDES or direction not in schema.SIDES:
                continue
            facing = 1 if direction == "right" else -1
            a_ang = frame_angles(_xy(first["points"]), leg, facing)
            b_ang = frame_angles(_xy(t["points"]), leg, facing)
            for ang in ANGLES:
                if np.isnan(a_ang[ang]) or np.isnan(b_ang[ang]):
                    continue
                pts = ANGLE_POINTS[ang](leg)
                angle_rows.append(
                    {
                        **key,
                        "angle": ang,
                        "first_deg": a_ang[ang],
                        "repeat_deg": b_ang[ang],
                        "diff_deg": b_ang[ang] - a_ang[ang],
                        "moved": any(_moved(first, n) or _moved(t, n) for n in pts),
                    }
                )
    return point_rows, angle_rows


def point_stats(rows: list[dict]) -> dict:
    d = np.array([r["diff_thigh"] for r in rows], float)
    px = np.array([r["diff_px"] for r in rows], float)
    if len(rows) == 0:
        return {
            "n": 0,
            "median_px": np.nan,
            "median_thigh": np.nan,
            "p90_thigh": np.nan,
            "pck": np.nan,
        }
    return {
        "n": len(rows),
        "median_px": float(np.median(px)),
        "median_thigh": float(np.nanmedian(d)),
        "p90_thigh": float(np.nanpercentile(d, 90)),
        "pck": float(np.mean(d <= PCK_THRESHOLD)),
    }


def angle_stats(rows: list[dict]) -> dict:
    d = np.array([r["diff_deg"] for r in rows], float)
    if len(rows) == 0:
        return {"n": 0, "mae": np.nan, "bias": np.nan, "max_abs": np.nan}
    return {
        "n": len(rows),
        "mae": float(np.mean(np.abs(d))),
        "bias": float(np.mean(d)),
        "max_abs": float(np.max(np.abs(d))),
    }


def summary_lines(
    sessions: list[dict], point_rows: list[dict], angle_rows: list[dict]
) -> list[str]:
    L = [
        "# Labelling repeatability (the floor)",
        "",
        "Generated by `uv run python -m eval.repeatability`. Aamir relabels sampled frames",
        "blind, at least 48 h after the first labels (`tools/label_tool.py --repeat`).",
        "Difference = relabel − first label. Thigh lengths as in E1; angles with the",
        "pipeline's geometry code, as in E2.",
        "",
    ]
    if not point_rows:
        n_tasks = sum(len(s["tasks"]) for s in sessions)
        n_done = sum(t.get("confirmed", False) for s in sessions for t in s["tasks"])
        L += [
            "**Not measured yet.** "
            + (
                f"{n_done} of {n_tasks} sampled frames are confirmed in {len(sessions)} "
                "session(s); nothing to compare."
                if sessions
                else "No blind relabel session exists."
            )
            + " Run `uv run python tools/label_tool.py --repeat` (about 25 min), then",
            "`uv run python -m eval.run_all`. Until then no error below can be compared with",
            "the labeller's own consistency.",
        ]
        return L
    frames = {(r["clip_id"], r["frame"]) for r in point_rows}
    L += [
        f"**n:** {len(frames)} frames from {len({c for c, _ in frames})} clips, "
        f"{len(point_rows)} body points.",
        "",
        "## Points",
        "",
    ]
    body = []
    for label, sel in (
        ("all", point_rows),
        ("moved in either session", [r for r in point_rows if r["moved"]]),
    ):
        s = point_stats(sel)
        body.append(
            [
                label,
                str(s["n"]),
                fmt(s["median_px"], nd=1),
                fmt(s["median_thigh"]),
                fmt(s["p90_thigh"]),
                fmt(s["pck"], pct=True),
            ]
        )
    L += table(
        ["points", "n", "median px", "median thigh", "p90 thigh", f"within {PCK_THRESHOLD} thigh"],
        body,
    )
    L += ["", "## Angles", ""]
    body = []
    for ang in ANGLES:
        for label, moved_only in (("all", False), ("moved", True)):
            sel = [r for r in angle_rows if r["angle"] == ang and (r["moved"] or not moved_only)]
            s = angle_stats(sel)
            body.append(
                [
                    ang,
                    label,
                    str(s["n"]),
                    fmt(s["mae"], nd=1),
                    fmt(s["bias"], nd=1),
                    fmt(s["max_abs"], nd=1),
                ]
            )
    L += table(["angle", "frames", "n", "MAE °", "bias °", "max abs °"], body)
    L += [
        "",
        "Read with E1 and E2: a model error near these numbers is within the labeller's own",
        "variation, so it can't be called worse than the labels.",
    ]
    return L


def main() -> None:
    sessions = load_sessions()
    point_rows, angle_rows = compare(schema.load_all(), sessions)
    for name, rows in (("points", point_rows), ("angles", angle_rows)):
        path = RESULTS / f"repeatability_{name}.csv"
        if rows:
            write_csv(path, rows)
        else:
            path.unlink(missing_ok=True)  # no stale numbers once the sessions are gone
    out = RESULTS / "repeatability_summary.md"
    out.write_text("\n".join(summary_lines(sessions, point_rows, angle_rows)) + "\n")
    print(
        f"{len(point_rows)} point pairs, {len(angle_rows)} angle pairs; "
        f"wrote {out.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
