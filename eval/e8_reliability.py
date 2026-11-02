"""E8: does v2 eliminate impossible outputs and refuse bad clips?

On all 26 manifest clips (no labels needed except ``view_ok_human``):
1. **Values beyond the sanity bounds** (kickcoach/features/bounds.py), v1 vs v2. v1 is the
   hackathon version replayed (eval/v1_replica.py); it answers every clip. Its knee angle,
   trunk tilt and knee angular velocity are checked against v2's bounds for the same
   quantity. Its ankle speed is in px/s, which no physical bound fits (it scales with
   resolution and camera distance), so it isn't counted. v2 never outputs a value beyond a
   bound by construction, so for v2 the count that matters is how many it caught.
2. **The quality gate vs ``view_ok_human``** (Aamir's label, DEC-007). Positive = "refuse".
   The 0.5 view-ratio threshold was set on the tuning split, which holds all 4 not-OK clips
   (DEC-021), so the tuning numbers are in-sample and the held-out ones can only show false
   refusals. v1 has no gate.
3. **Velocities correctly withheld**: a velocity is allowed only with a trusted capture rate
   of at least 120 fps (handoff D6, DEC-010). Whether the rate is trusted comes from the
   source's own statement (c13: 300 fps, c19: 240 fps) or real-time evidence (c25), recorded
   in the manifest and ingest metadata. v1 reports a velocity on every clip, using the
   container fps.

    uv run python -m eval.e8_reliability
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eval import v1_replica
from eval.e1_pose import fmt, load_manifest, select, table, write_csv
from kickcoach.features.bounds import violation
from kickcoach.features.kick import MIN_VELOCITY_FPS
from kickcoach.ingest import load_meta
from kickcoach.labels import schema
from kickcoach.pose import cache
from kickcoach.quality import gate

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FRAMES = ROOT / "data" / "frames"

# v1 output -> the v2 bound for the same quantity
V1_CHECKS = {
    "knee_angle": "knee_angle_contact",
    "trunk": "trunk_lean_contact",
    "knee_vel": "peak_knee_ext_velocity",
}


def velocity_allowed(meta: dict) -> bool:
    spf = meta.get("seconds_per_frame")
    return bool(meta.get("fps_trusted")) and bool(spf) and 1.0 / spf >= MIN_VELOCITY_FPS


def v1_rows(cid: str, meta: dict) -> list[dict]:
    a = v1_replica.kick(cache.load(v1_replica.CACHE_MODEL, cid), meta["playback_fps"])
    rows = []
    for attr, bound in V1_CHECKS.items():
        v = getattr(a, attr) if a else np.nan
        # v1's velocity is the instantaneous one at its "kick" frame, not a peak extension,
        # so only its size is held to the bound; a negative one (knee bending) is counted
        # separately, as a wrong event rather than an impossible number.
        checked = abs(v) if attr == "knee_vel" else v
        why = violation(bound, checked) if np.isfinite(v) else None
        rows.append(
            {
                "clip_id": cid,
                "system": "v1",
                "quantity": bound,
                "value": round(float(v), 1) if np.isfinite(v) else None,
                "frame": a.frame if a else None,
                "beyond_bound": why is not None,
                "caught_by_bound": False,
                "negative_velocity": attr == "knee_vel" and bool(v < 0),
                "reason": why,
            }
        )
    return rows


def v2_rows(cid: str, res: dict) -> list[dict]:
    rows = []
    for f in res["features"]:
        v = f.get("value")
        numeric = isinstance(v, (int, float))
        rows.append(
            {
                "clip_id": cid,
                "system": "v2",
                "quantity": f["id"],
                "value": v,
                "frame": f.get("frame"),
                # Re-checked here rather than trusted: a measured value must be inside.
                "beyond_bound": bool(
                    numeric and f["id"] in V1_CHECKS.values() and violation(f["id"], v)
                ),
                "caught_by_bound": f["status"] == "not_measured"
                and "sanity bound" in (f.get("reason") or ""),
                "negative_velocity": False,
                "reason": f.get("reason"),
            }
        )
    return rows


def gate_rows(manifest: dict, labels: dict, results: dict) -> list[dict]:
    rows = []
    for cid, m in manifest.items():
        q = results[cid]["quality"]
        rows.append(
            {
                "clip_id": cid,
                "split": m["split"],
                "purpose": m["purpose"],
                "view_ok_human": labels.get(cid, {}).get("clip", {}).get("view_ok_human"),
                "v2_refused": not q["accepted"],
                "viewpoint_ratio": q["viewpoint_ratio"],
                "reasons": " | ".join(q["reasons"]),
                "features_when_refused": len(results[cid]["features"])
                if not q["accepted"]
                else None,
            }
        )
    return rows


def confusion(rows: list[dict], refused_key: str) -> dict:
    rs = [r for r in rows if r["view_ok_human"] in ("y", "n")]
    tp = sum(r[refused_key] and r["view_ok_human"] == "n" for r in rs)
    fp = sum(r[refused_key] and r["view_ok_human"] == "y" for r in rs)
    fn = sum(not r[refused_key] and r["view_ok_human"] == "n" for r in rs)
    tn = sum(not r[refused_key] and r["view_ok_human"] == "y" for r in rs)
    return {
        "n": len(rs),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "false_refusal_rate": fp / (fp + tn) if fp + tn else np.nan,
    }


def velocity_rows(manifest: dict, metas: dict, results: dict) -> list[dict]:
    rows = []
    for cid in manifest:
        meta = metas[cid]
        feats = {f["id"]: f for f in results[cid]["features"]}
        kv = feats.get("peak_knee_ext_velocity", {})
        a = v1_replica.kick(cache.load(v1_replica.CACHE_MODEL, cid), meta["playback_fps"])
        allowed = velocity_allowed(meta)
        v2_reported = kv.get("status") == "measured"
        rows.append(
            {
                "clip_id": cid,
                "allowed": allowed,
                "fps_reason": meta.get("fps_reason"),
                "v2_reported": v2_reported,
                "v2_value": kv.get("value"),
                "v2_reason": kv.get("reason") if not v2_reported else None,
                "v2_correct": v2_reported == allowed,
                "v1_reported": a is not None and np.isfinite(a.knee_vel),
                "v1_value": round(a.knee_vel, 1) if a and np.isfinite(a.knee_vel) else None,
                "v1_fps_used": round(meta["playback_fps"], 2),
                "v1_correct": (a is not None and np.isfinite(a.knee_vel)) == allowed,
            }
        )
    return rows


def run(manifest: dict, labels: dict) -> dict:
    metas = {c: load_meta(FRAMES / c) for c in manifest}
    results = {c: json.loads((RESULTS / "pipeline" / f"{c}.json").read_text()) for c in manifest}
    bounds = []
    for cid in manifest:
        bounds += v1_rows(cid, metas[cid]) + v2_rows(cid, results[cid])
    g = gate_rows(manifest, labels, results)
    for r in g:
        r["v1_refused"] = False
    return {
        "bounds": bounds,
        "gate": g,
        "gate_threshold": gate.MAX_VIEW_RATIO,
        "velocity": velocity_rows(manifest, metas, results),
    }


def report(out: dict) -> str:
    b, g, v = out["bounds"], out["gate"], out["velocity"]
    v1b = select(b, system="v1")
    v2b = select(b, system="v2")
    v2_numeric = [r for r in v2b if isinstance(r["value"], (int, float))]
    caught = [r for r in v2b if r["caught_by_bound"]]
    L = [
        "# E8: reliability",
        "",
        "Generated by `uv run python -m eval.e8_reliability`, on all "
        f"{len(g)} manifest clips. v1 = the hackathon version replayed",
        "(`eval/v1_replica.py`); v2 = `results/pipeline/`.",
        "",
        "## 1. Values beyond the sanity bounds",
        "",
        "Bounds from `kickcoach/features/bounds.py` (cited or marked heuristic there). v1 is",
        "checked on the same quantities: knee angle and trunk tilt at its kick frame, and the",
        "size of its knee angular velocity there (instantaneous, not a peak, so only |value|",
        "is held to the extension bound).",
        "",
    ]
    body = []
    for q in V1_CHECKS.values():
        rs = select(v1b, quantity=q)
        bad = [r for r in rs if r["beyond_bound"]]
        body.append(
            [
                "v1",
                q,
                str(sum(r["value"] is not None for r in rs)),
                str(len(bad)),
                ", ".join(f"{r['clip_id']} {r['value']:g}" for r in bad) or "–",
            ]
        )
    for q in V1_CHECKS.values():
        rs = [r for r in select(v2b, quantity=q) if isinstance(r["value"], (int, float))]
        bad = [r for r in rs if r["beyond_bound"]]
        body.append(
            [
                "v2",
                q,
                str(len(rs)),
                str(len(bad)),
                "–" if not bad else ", ".join(r["clip_id"] for r in bad),
            ]
        )
    L += table(["system", "quantity", "values output", "beyond bound", "which"], body)
    v1_total = sum(r["beyond_bound"] for r in v1b)
    L += [
        "",
        f"- **v1: {v1_total} of {sum(r['value'] is not None for r in v1b)} values beyond a bound.**"
        f" v2: {sum(r['beyond_bound'] for r in v2_numeric)} of {len(v2_numeric)} numeric values"
        " (by construction: a value outside its bound becomes `not_measured`).",
        f"- v1's knee velocity was also negative (the knee bending at what v1 calls the kick) on "
        f"{sum(r['negative_velocity'] for r in v1b)} of {len(g)} clips: a wrong event, not counted"
        " above.",
        f"- v2 values caught by a bound and withheld: {len(caught)}"
        + (": " + "; ".join(f"{r['clip_id']} {r['quantity']}" for r in caught) if caught else "."),
        "- v1's in-bound velocities are still wrong: they use the container's ~30 fps on slow-",
        "  motion video, so they are slowed by the unknown slow-motion factor (section 3).",
        "",
        "## 2. Quality gate vs `view_ok_human`",
        "",
        f"Positive = refuse. View rule: shoulder width / torso length in the kick window > "
        f"{out['gate_threshold']} (heuristic, set on the tuning split). Clips without",
        "`view_ok_human` are left out: "
        + (", ".join(r["clip_id"] for r in g if r["view_ok_human"] not in ("y", "n")) or "none")
        + ".",
        "",
    ]
    body = []
    for system, key in (("v2", "v2_refused"), ("v1 (no gate)", "v1_refused")):
        for split in ("all", "tune", "heldout"):
            rs = g if split == "all" else select(g, split=split)
            c = confusion(rs, key)
            body.append(
                [
                    system,
                    split,
                    str(c["n"]),
                    f"{c['tp']}/{c['tp'] + c['fn']}",
                    f"{c['fp']}/{c['fp'] + c['tn']}",
                    fmt(c["precision"], True),
                    fmt(c["recall"], True),
                    fmt(c["false_refusal_rate"], True),
                ]
            )
    L += table(
        [
            "system",
            "split",
            "n",
            "not-OK refused",
            "OK refused",
            "precision",
            "recall",
            "false refusals",
        ],
        body,
    )
    ok = sorted((r for r in g if r["view_ok_human"] == "y"), key=lambda r: -r["viewpoint_ratio"])
    bad = sorted((r for r in g if r["view_ok_human"] == "n"), key=lambda r: r["viewpoint_ratio"])
    refused = [r for r in g if r["v2_refused"]]
    L += [
        "",
        f"- Margins: the highest side-on ratio is {ok[0]['viewpoint_ratio']:.3f} "
        f"({ok[0]['clip_id']}), the lowest not-OK ratio {bad[0]['viewpoint_ratio']:.3f} "
        f"({bad[0]['clip_id']}), either side of the {out['gate_threshold']} threshold."
        if ok and bad
        else "",
        f"- Every refusal carries a reason and no features: "
        f"{all(r['reasons'] and r['features_when_refused'] == 0 for r in refused)} "
        f"({len(refused)} refusals).",
        "- All 4 not-OK clips are tuning clips and set the threshold, so recall and precision",
        "  here are in-sample. The held-out clips are all side-on: they can only show that the",
        "  gate doesn't refuse good clips.",
        "",
        "## 3. Velocities correctly withheld",
        "",
        f"Allowed only with a trusted capture rate ≥ {MIN_VELOCITY_FPS:.0f} fps (handoff D6).",
        "",
    ]
    n = len(v)
    L += table(
        ["system", "velocity reported", "correct (reported iff allowed)"],
        [
            ["v2", str(sum(r["v2_reported"] for r in v)), f"{sum(r['v2_correct'] for r in v)}/{n}"],
            ["v1", str(sum(r["v1_reported"] for r in v)), f"{sum(r['v1_correct'] for r in v)}/{n}"],
        ],
    )
    L += ["", "Clips where a velocity is allowed, or the timing is otherwise trusted:", ""]
    L += table(
        ["clip", "timing", "allowed", "v2", "v1 (deg/s at the container fps)"],
        [
            [
                r["clip_id"],
                r["fps_reason"] or "",
                "yes" if r["allowed"] else "no",
                f"{r['v2_value']:g} deg/s" if r["v2_reported"] else (r["v2_reason"] or "withheld"),
                f"{r['v1_value']:g}" if r["v1_value"] is not None else "–",
            ]
            for r in v
            if r["allowed"] or (r["fps_reason"] or "").startswith("real-time")
        ],
    )
    return "\n".join(x for x in L) + "\n"


def main() -> None:
    manifest = load_manifest()
    labels = schema.load_all()
    out = run(manifest, labels)
    write_csv(RESULTS / "e8_bounds.csv", out["bounds"])
    write_csv(RESULTS / "e8_gate.csv", out["gate"])
    write_csv(RESULTS / "e8_velocity.csv", out["velocity"])
    md = report(out)
    (RESULTS / "e8_summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
