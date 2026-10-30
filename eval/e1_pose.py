"""E1: which pose model places keypoints best on kicking frames?

Scores every cached pose model against Aamir's confirmed labels on the same frames:
- error normalised by the frame's thigh length (the mean of the two labelled hip-knee
  lengths), so a far-away player and a close one count alike;
- PCK@0.1: the share of labelled points within 0.1 thigh lengths of the label. A point the
  model gives no coordinates for counts as a miss (eval plan, E1);
- missing rate: labelled points the model gives no coordinates for. Confidence is not used,
  because the three models' scores mean different things (DEC-011).
Broken out by joint group (kicking vs plant leg), frame role (contact is the most blurred),
visibility (visible only vs all labelled) and split (tune vs held-out).

Pre-fill anchoring (DEC-013): labels started at RTMPose's guess, so a point the labeller
left alone agrees with RTMPose by construction. The script counts those, and reports every
model's error on the moved points only. That subset gives RTMPose no such help; if anything
it's biased against RTMPose, because a point got moved *because* RTMPose had it wrong.

Also reported, as supporting evidence for the feature list: the angle each model's raw
points give at the labelled frames, against the labels' angle. No filtering, and the events
come from the labels, so this is not E2 (Phase 4).

Decision rule, fixed before the results were seen: the model with the best PCK@0.1 on the
core points (shoulders, hips, knees, ankles, which all three models output) of the required
frames (max backswing + contact) is chosen if (a) the clip-bootstrap 95% CI of its paired
lead over the runner-up excludes 0, and (b) it also leads on the moved points. Otherwise the
result is "close", and Aamir picks (CLAUDE.md, ask-before list).

    uv run python -m eval.e1_pose
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from kickcoach.features.geometry import joint_angle_deg, lean_from_vertical_deg
from kickcoach.labels import qa, schema
from kickcoach.pose import cache, skeleton

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS = ("movenet", "rtmpose", "mediapipe")
PCK_THRESHOLD = 0.1  # thigh lengths (eval plan, E1)
# Labels are saved rounded to 0.01 px, the pre-fill at full precision, so an untouched point
# differs from its pre-fill by at most ~0.007 px.
UNMOVED_PX = 0.05
REQUIRED_ROLES = schema.REQUIRED_ROLES
CORE_JOINTS = ("shoulder", "hip", "knee", "ankle")
FOOT_JOINTS = ("heel", "big_toe")
N_BOOT = 2000
SEED = 0


def load_manifest(path: Path = ROOT / "data" / "manifest.csv") -> dict[str, dict]:
    with open(path, newline="") as f:
        return {r["clip_id"]: r for r in csv.DictReader(f)}


def joint_of(point: str) -> str:
    return point.split("_", 1)[1]


def group_of(point: str, kicking_leg: str) -> str:
    side, joint = point.split("_", 1)
    if joint == "shoulder":
        return "shoulders"
    leg = "kicking" if side == kicking_leg else "plant"
    return f"{leg}_{'foot' if joint in FOOT_JOINTS else 'leg'}"


def frame_role(roles: list[str]) -> str:
    return "+".join(sorted(roles)) if roles else "other"


def point_rows(labels: dict[str, dict], manifest: dict[str, dict], poses: dict) -> list[dict]:
    """One row per (model, clip, confirmed frame, labelled body point).

    ``poses[model][clip_id]`` is the cached [T, K, 3] array. Only clips with a kicking leg
    are scored, because the groups depend on it.
    """
    rows = []
    for cid, lab in sorted(labels.items()):
        leg = lab["clip"]["kicking_leg"]
        if lab.get("purpose") != "eval" or leg not in schema.SIDES:
            continue
        roles = schema.frame_roles(lab["clip"], lab["n_frames"])
        for key, entry in sorted(lab["frames"].items(), key=lambda kv: int(kv[0])):
            if not entry.get("confirmed"):
                continue
            t = int(key)
            thigh = qa.thigh_length(schema.points_array(entry["points"]))
            if thigh is None:
                continue
            for name in schema.POINTS:
                p = entry["points"].get(name)
                if not p or p["v"] == schema.UNKNOWN:
                    continue
                lab_xy = np.array([p["x"], p["y"]])
                pre = entry.get("prefill", {}).get(name)
                moved_px = float(np.hypot(*(lab_xy - pre[:2]))) if pre else np.nan
                for model in MODELS:
                    pred = poses[model][cid][t, skeleton.INDEX[name]]
                    missing = bool(np.isnan(pred[0]) or np.isnan(pred[1]))
                    err_px = np.nan if missing else float(np.hypot(*(pred[:2] - lab_xy)))
                    rows.append(
                        {
                            "model": model,
                            "clip_id": cid,
                            "split": manifest[cid]["split"],
                            "frame": t,
                            "role": frame_role(roles.get(t, [])),
                            "point": name,
                            "joint": joint_of(name),
                            "group": group_of(name, leg),
                            "v": p["v"],
                            "label_x": p["x"],
                            "label_y": p["y"],
                            "pred_x": None if missing else round(float(pred[0]), 2),
                            "pred_y": None if missing else round(float(pred[1]), 2),
                            "thigh_px": round(thigh, 2),
                            "missing": missing,
                            "err_px": err_px,
                            "err_thigh": err_px / thigh,
                            "moved_px": moved_px,
                            "unmoved": bool(moved_px <= UNMOVED_PX),
                        }
                    )
    return rows


def summarise(rows: list[dict]) -> dict:
    """PCK, missing rate and error over a set of point rows (one model)."""
    n = len(rows)
    if n == 0:
        return {"n": 0, "missing_rate": np.nan, "pck": np.nan, "median_err": np.nan,
                "mean_err": np.nan, "p90_err": np.nan}  # fmt: skip
    err = np.array([r["err_thigh"] for r in rows], dtype=float)
    hit = np.nan_to_num(err, nan=np.inf) < PCK_THRESHOLD
    have = err[~np.isnan(err)]
    return {
        "n": n,
        "missing_rate": float(np.isnan(err).mean()),
        "pck": float(hit.mean()),
        "median_err": float(np.median(have)) if len(have) else np.nan,
        "mean_err": float(np.mean(have)) if len(have) else np.nan,
        "p90_err": float(np.percentile(have, 90)) if len(have) else np.nan,
    }


def select(rows: list[dict], **conds) -> list[dict]:
    """Rows matching every condition; a condition value may be a tuple of allowed values."""
    out = []
    for r in rows:
        ok = True
        for k, v in conds.items():
            allowed = v if isinstance(v, tuple) else (v,)
            if r[k] not in allowed:
                ok = False
                break
        if ok:
            out.append(r)
    return out


def clip_bootstrap_pck(
    rows: list[dict], models: tuple[str, ...], n_boot: int = N_BOOT, seed: int = SEED
) -> dict[str, np.ndarray]:
    """PCK per model over clip resamples. Frames of one clip share a kicker, a camera and a
    lighting setup, so the clip (not the point) is the unit that's resampled."""
    clips = sorted({r["clip_id"] for r in rows})
    hits = {m: np.zeros(len(clips)) for m in models}
    counts = {m: np.zeros(len(clips)) for m in models}
    idx = {c: i for i, c in enumerate(clips)}
    for r in rows:
        if r["model"] in hits:
            i = idx[r["clip_id"]]
            counts[r["model"]][i] += 1
            hits[r["model"]][i] += np.nan_to_num(r["err_thigh"], nan=np.inf) < PCK_THRESHOLD
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(clips), size=(n_boot, len(clips)))
    return {m: hits[m][draws].sum(1) / counts[m][draws].sum(1) for m in models}


def ci95(x: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


# --- raw per-frame angles -------------------------------------------------------------


def frame_angles(xy: dict[str, np.ndarray], kicking_leg: str, facing: int) -> dict[str, float]:
    """The three angle features from one frame's points (NaN where a point is missing)."""
    nan2 = np.array([np.nan, np.nan])
    g = lambda name: xy.get(name, nan2)  # noqa: E731
    k = kicking_leg
    hip_mid = (g("left_hip") + g("right_hip")) / 2
    sh_mid = (g("left_shoulder") + g("right_shoulder")) / 2
    return {
        "knee_angle": float(joint_angle_deg(g(f"{k}_hip"), g(f"{k}_knee"), g(f"{k}_ankle"))),
        "trunk_lean": float(lean_from_vertical_deg(hip_mid, sh_mid, facing)),
        "ankle_angle": float(joint_angle_deg(g(f"{k}_knee"), g(f"{k}_ankle"), g(f"{k}_big_toe"))),
    }


ANGLE_POINTS = {
    "knee_angle": lambda k: (f"{k}_hip", f"{k}_knee", f"{k}_ankle"),
    "trunk_lean": lambda k: ("left_hip", "right_hip", "left_shoulder", "right_shoulder"),
    "ankle_angle": lambda k: (f"{k}_knee", f"{k}_ankle", f"{k}_big_toe"),
}


def angle_rows(labels: dict[str, dict], poses: dict) -> list[dict]:
    """Per (model, confirmed frame, angle): the model's raw angle vs the labels' angle."""
    rows = []
    for cid, lab in sorted(labels.items()):
        clip = lab["clip"]
        leg, direction = clip["kicking_leg"], clip["kick_direction"]
        if lab.get("purpose") != "eval" or leg not in schema.SIDES or direction not in schema.SIDES:
            continue
        facing = 1 if direction == "right" else -1
        roles = schema.frame_roles(clip, lab["n_frames"])
        for key, entry in sorted(lab["frames"].items(), key=lambda kv: int(kv[0])):
            if not entry.get("confirmed"):
                continue
            t = int(key)
            pts = entry["points"]
            lab_xy = {
                n: np.array([p["x"], p["y"]]) for n, p in pts.items() if p["v"] > schema.UNKNOWN
            }
            truth = frame_angles(lab_xy, leg, facing)
            for model in MODELS:
                pose = poses[model][cid][t]
                pred_xy = {n: pose[skeleton.INDEX[n], :2].astype(float) for n in schema.POINTS}
                pred = frame_angles(pred_xy, leg, facing)
                for angle, truth_val in truth.items():
                    needed = ANGLE_POINTS[angle](leg)
                    visible = all(pts.get(n, {}).get("v") == schema.VISIBLE for n in needed)
                    err = pred[angle] - truth_val
                    rows.append(
                        {
                            "model": model,
                            "clip_id": cid,
                            "frame": t,
                            "role": frame_role(roles.get(t, [])),
                            "angle": angle,
                            "all_visible": visible,
                            "label_deg": round(truth_val, 2),
                            "model_deg": None if np.isnan(pred[angle]) else round(pred[angle], 2),
                            "err_deg": None if np.isnan(err) else round(float(err), 2),
                        }
                    )
    return rows


def angle_summary(rows: list[dict]) -> dict:
    err = np.array([r["err_deg"] for r in rows if r["err_deg"] is not None], dtype=float)
    if len(err) == 0:
        return {"n": 0, "mae": np.nan, "bias": np.nan, "median_abs": np.nan, "max_abs": np.nan}
    return {
        "n": len(err),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "median_abs": float(np.median(np.abs(err))),
        "max_abs": float(np.max(np.abs(err))),
    }


# --- left/right swaps -----------------------------------------------------------------


def leg_swaps(labels: dict[str, dict], poses: dict) -> dict[str, list[tuple[str, int]]]:
    """Frames where a model's whole left and right legs are swapped relative to the labels.

    Uses the same rule as the labelling QA (kickcoach.labels.qa.SWAP_RATIO, heuristic): the
    crossed match is under half the straight one. A swap the signal stage would have to fix.
    """
    out: dict[str, list[tuple[str, int]]] = defaultdict(list)
    joints = ("hip", "knee", "ankle")
    ia = {n: i for i, n in enumerate(schema.ALL_POINTS)}
    for cid, lab in sorted(labels.items()):
        if lab.get("purpose") != "eval":
            continue
        for key, entry in lab["frames"].items():
            if not entry.get("confirmed"):
                continue
            arr = schema.points_array(entry["points"])
            for model in MODELS:
                pose = poses[model][cid][int(key)]
                straight = crossed = 0.0
                ok = True
                for j in joints:
                    L, R = arr[ia[f"left_{j}"]], arr[ia[f"right_{j}"]]
                    mL = pose[skeleton.INDEX[f"left_{j}"], :2]
                    mR = pose[skeleton.INDEX[f"right_{j}"], :2]
                    missing = np.isnan(mL).any() or np.isnan(mR).any()
                    if L[2] == schema.UNKNOWN or R[2] == schema.UNKNOWN or missing:
                        ok = False
                        break
                    straight += np.hypot(*(L[:2] - mL)) + np.hypot(*(R[:2] - mR))
                    crossed += np.hypot(*(L[:2] - mR)) + np.hypot(*(R[:2] - mL))
                if ok and crossed < qa.SWAP_RATIO * straight:
                    out[model].append((cid, int(key)))
    return out


# --- report ---------------------------------------------------------------------------


def fmt(x: float, pct: bool = False, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{100 * x:.0f}%" if pct else f"{x:.{nd}f}"


def table(header: list[str], body: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
        *("| " + " | ".join(r) + " |" for r in body),
    ]


def run(labels: dict[str, dict], manifest: dict[str, dict], poses: dict) -> dict:
    rows = point_rows(labels, manifest, poses)
    req = select(rows, role=REQUIRED_ROLES)
    core = select(req, joint=CORE_JOINTS)
    feet = select(req, joint=FOOT_JOINTS)

    summary_rows = []

    def add(subset: str, rs: list[dict]) -> None:
        for m in MODELS:
            s = summarise(select(rs, model=m))
            summary_rows.append({"subset": subset, "model": m, **s})

    add("core, required frames, all labelled", core)
    add("core, required frames, visible only", select(core, v=schema.VISIBLE))
    add("feet, required frames, all labelled", feet)
    add("feet, required frames, visible only", select(feet, v=schema.VISIBLE))
    for g in ("shoulders", "kicking_leg", "plant_leg", "kicking_foot", "plant_foot"):
        add(f"group={g}, required frames", select(req, group=g))
    for role in ("max_backswing", "contact", "pre_contact"):
        add(f"core, role={role}", select(rows, role=role, joint=CORE_JOINTS))
    add("kicking leg at contact", select(rows, role="contact", group="kicking_leg"))
    add(
        "kicking big toe at contact, visible only",
        select(rows, role="contact", group="kicking_foot", joint="big_toe", v=schema.VISIBLE),
    )
    for split in ("tune", "heldout"):
        add(f"core, required frames, split={split}", select(core, split=split))
    add("core, required frames, moved points only", select(core, unmoved=False))
    add("feet, required frames, moved points only", select(feet, unmoved=False))
    for j in (*CORE_JOINTS, *FOOT_JOINTS):
        add(f"joint={j}, required frames", select(req, joint=j))

    boot = clip_bootstrap_pck(core, MODELS)
    point_est = {m: summarise(select(core, model=m))["pck"] for m in MODELS}
    ranked = sorted(MODELS, key=lambda m: -point_est[m])
    best, second = ranked[0], ranked[1]
    diff = boot[best] - boot[second]
    moved_pck = {m: summarise(select(core, model=m, unmoved=False))["pck"] for m in MODELS}
    lead_moved = max(MODELS, key=lambda m: moved_pck[m]) == best
    diff_ci = ci95(diff)
    decision = {
        "ranked": ranked,
        "pck_core": point_est,
        "pck_core_ci95": {m: ci95(boot[m]) for m in MODELS},
        "best": best,
        "runner_up": second,
        "lead": point_est[best] - point_est[second],
        "lead_ci95": diff_ci,
        "pck_core_moved": moved_pck,
        "leads_on_moved": lead_moved,
        "close": not (diff_ci[0] > 0 and lead_moved),
        "n_clips": len({r["clip_id"] for r in core}),
        "n_frames": len({(r["clip_id"], r["frame"]) for r in core}),
        "n_boot": N_BOOT,
    }

    rt = select(rows, model="rtmpose")
    anchoring = {
        "points": len(rt),
        "unmoved": sum(r["unmoved"] for r in rt),
        "by_group": {
            g: (
                sum(r["unmoved"] for r in select(rt, group=g)),
                len(select(rt, group=g)),
            )
            for g in ("shoulders", "kicking_leg", "plant_leg", "kicking_foot", "plant_foot")
        },
        "by_joint": {
            j: (sum(r["unmoved"] for r in select(rt, joint=j)), len(select(rt, joint=j)))
            for j in (*CORE_JOINTS, *FOOT_JOINTS)
        },
        "by_clip": {
            c: (sum(r["unmoved"] for r in select(rt, clip_id=c)), len(select(rt, clip_id=c)))
            for c in sorted({r["clip_id"] for r in rt})
        },
        "moved_px_median": float(np.median([r["moved_px"] for r in rt if not r["unmoved"]])),
    }

    angles = angle_rows(labels, poses)
    angle_table = []
    for angle in ("knee_angle", "trunk_lean", "ankle_angle"):
        for role in ("max_backswing", "contact"):
            for vis_only in (False, True):
                for m in MODELS:
                    rs = select(angles, model=m, angle=angle, role=role)
                    if vis_only:
                        rs = select(rs, all_visible=True)
                    angle_table.append(
                        {
                            "angle": angle,
                            "role": role,
                            "points": "visible only" if vis_only else "all labelled",
                            "model": m,
                            **angle_summary(rs),
                        }
                    )

    swaps = leg_swaps(labels, poses)
    return {
        "rows": rows,
        "summary": summary_rows,
        "decision": decision,
        "anchoring": anchoring,
        "angles": angles,
        "angle_table": angle_table,
        "swaps": {m: swaps.get(m, []) for m in MODELS},
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def report(out: dict) -> str:
    d, a = out["decision"], out["anchoring"]
    s = {(r["subset"], r["model"]): r for r in out["summary"]}
    L = [
        "# E1: pose accuracy on labelled kicking frames",
        "",
        "Generated by `uv run python -m eval.e1_pose`. Labels: Aamir's confirmed frames in",
        "`data/labels/`. Error is in thigh lengths (the frame's mean labelled hip-knee length).",
        f"PCK@{PCK_THRESHOLD}: share of labelled points within {PCK_THRESHOLD} thigh lengths;",
        "a point the model doesn't output counts as a miss. Core = shoulders, hips, knees,",
        "ankles (all models output them). Feet = heel + big toe (MoveNet has none).",
        "Required frames = max backswing + contact.",
        "",
        f"**n:** {d['n_clips']} clips, {d['n_frames']} required frames; "
        f"{len(select(out['rows'], model='rtmpose'))} labelled body points on all confirmed "
        "frames (incl. optional contact − 1).",
        "",
        "## Headline and decision",
        "",
    ]
    body = []
    for m in d["ranked"]:
        r = s[("core, required frames, all labelled", m)]
        lo, hi = d["pck_core_ci95"][m]
        body.append(
            [
                m,
                f"{fmt(r['pck'], True)} ({fmt(lo, True)}–{fmt(hi, True)})",
                fmt(r["median_err"]),
                fmt(r["missing_rate"], True),
                fmt(d["pck_core_moved"][m], True),
                fmt(s[("feet, required frames, all labelled", m)]["pck"], True),
                str(r["n"]),
            ]
        )
    L += table(
        [
            "model",
            "core PCK@0.1 (95% CI, clip bootstrap)",
            "core median err",
            "core missing",
            "core PCK, moved points only",
            "feet PCK",
            "n core points",
        ],
        body,
    )
    lo, hi = d["lead_ci95"]
    L += [
        "",
        f"- Lead of **{d['best']}** over {d['runner_up']}: {100 * d['lead']:+.1f} PCK points "
        f"(95% CI {100 * lo:+.1f} to {100 * hi:+.1f}, {d['n_boot']} clip resamples).",
        f"- Leads on moved points too: {'yes' if d['leads_on_moved'] else 'no'}.",
        f"- **Result: {'CLOSE: Aamir decides' if d['close'] else d['best'] + ' chosen'}** "
        "(rule fixed in advance; see the script's docstring).",
        "",
        "## Breakdowns (PCK@0.1 / median error / missing)",
        "",
    ]
    subsets = [r["subset"] for r in out["summary"] if r["model"] == MODELS[0]]
    body = []
    for sub in subsets:
        cells = [sub, str(s[(sub, MODELS[0])]["n"])]
        for m in MODELS:
            r = s[(sub, m)]
            cells.append(
                f"{fmt(r['pck'], True)} / {fmt(r['median_err'], nd=2)} / "
                f"{fmt(r['missing_rate'], True)}"
            )
        body.append(cells)
    L += table(["subset", "n", *MODELS], body)
    L += [
        "",
        "## Pre-fill anchoring (DEC-013)",
        "",
        f"- {a['unmoved']} of {a['points']} labelled body points "
        f"({100 * a['unmoved'] / a['points']:.0f}%) were left exactly at RTMPose's pre-fill "
        f"(≤ {UNMOVED_PX} px). Moved points moved a median {a['moved_px_median']:.1f} px.",
        "- On an unmoved point RTMPose's error is 0 by construction, so the all-points",
        "  numbers flatter RTMPose. The moved-only rows give it no help (and are, if anything,",
        "  biased against it). A model that leads on both has a robust lead.",
        "",
    ]
    L += table(
        ["joint", "left at pre-fill"],
        [[j, f"{u}/{n} ({100 * u / n:.0f}%)"] for j, (u, n) in a["by_joint"].items() if n],
    )
    L += ["", "By clip (labelling order was c10, c01, the held-out five, then the rest):", ""]
    L += table(
        ["clip", "left at pre-fill"],
        [[c, f"{u}/{n} ({100 * u / n:.0f}%)"] for c, (u, n) in a["by_clip"].items()],
    )
    L += [
        "",
        "## Raw per-frame angle error (supporting evidence, not E2)",
        "",
        "Angle from each model's raw points at the labelled frame minus the angle from the",
        "labels (same geometry code). No filtering; frames come from the labels. Visible only =",
        "frames where every point the angle needs is marked visible.",
        "",
    ]
    body = []
    for r in out["angle_table"]:
        body.append(
            [
                r["angle"],
                r["role"],
                r["points"],
                r["model"],
                str(r["n"]),
                fmt(r["mae"], nd=1),
                fmt(r["bias"], nd=1),
                fmt(r["median_abs"], nd=1),
                fmt(r["max_abs"], nd=1),
            ]
        )
    L += table(["angle", "frame", "points", "model", "n", "MAE °", "bias °", "median |err| °",
                "max |err| °"], body)  # fmt: skip
    L += ["", "## Whole-leg left/right swaps vs the labels", ""]
    L += table(
        ["model", "frames swapped", "which"],
        [
            [m, str(len(v)), ", ".join(f"{c}:{t}" for c, t in v) or "–"]
            for m, v in out["swaps"].items()
        ],
    )
    return "\n".join(L) + "\n"


def main() -> None:
    labels = schema.load_all()
    manifest = load_manifest()
    poses = {
        m: {cid: cache.load(m, cid) for cid in labels if cache.path(m, cid).exists()}
        for m in MODELS
    }
    out = run(labels, manifest, poses)
    write_csv(RESULTS / "e1_points.csv", out["rows"])
    write_csv(RESULTS / "e1.csv", out["summary"])
    write_csv(RESULTS / "e1_angles.csv", out["angles"])
    (RESULTS / "e1_decision.json").write_text(json.dumps(out["decision"], indent=1) + "\n")
    md = report(out)
    (RESULTS / "e1_summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
