"""E3: does ball-based contact beat peak foot speed?

Truth: Aamir's contact frame (DEC-015: the first frame the shoe is pressed against the
ball). Error = detected - labelled, in frames. Every method runs on the same cleaned RTMPose
signal the pipeline's event pass uses (kickcoach.signal.clean, no break, cutoff by DEC-019):
- ``ball``: kickcoach.events with the ball (motion, else bracket), as shipped: the bracket
  pick moved BRACKET_LEAD_FRAMES later, inside the bracket. No resting ball near the foot =
  a detection failure (the shipped pipeline then falls back to foot speed).
- ``ball_uncorrected``: the same with the bracket pick left where the speed peak is.
- ``foot_speed``: the handoff's fallback (D9): the frame where one foot is fastest relative
  to the other over the whole clip, also choosing the leg.
- ``foot_speed_given_leg``: the same, told the labelled kicking leg. The ball method finds
  contact without knowing the leg, so this is the fairer comparison for the contact frame.
- ``v2_pipeline``: what the pipeline reports (ball, else foot speed).
- ``v1``: the hackathon version's "kick" frame (eval/v1_replica.py).

The bracket correction (DEC-020's open question). Rule, written before the corrected numbers
were computed (Phase 3's preview had already shown the uncorrected offsets): estimate the
shift as minus the median offset of the tuning bracket clips. Adopt it if (a) leave-one-
clip-out on the tuning bracket clips lowers the mean absolute error, and (b) it doesn't
raise the held-out bracket clips' mean absolute error.

Frame errors are in video frames. Most clips are slow motion of unknown factor, so a frame
is not a fixed time.

    uv run python -m eval.e3_contact
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eval import v1_replica
from eval.e1_pose import fmt, load_manifest, select, table, write_csv
from kickcoach import events
from kickcoach.events import ball_detector
from kickcoach.events.kick import leg_from_speed
from kickcoach.ingest import load_meta
from kickcoach.labels import schema
from kickcoach.pose import cache
from kickcoach.signal import clean

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FRAMES = ROOT / "data" / "frames"
METHODS = ("ball", "ball_uncorrected", "foot_speed", "foot_speed_given_leg", "v2_pipeline", "v1")
# Descriptive only: an error this large means the method found something other than the
# touch (a stride, a follow-through glitch), not an imprecise touch.
GROSS_FRAMES = 5


def clips_with_contact(labels: dict, manifest: dict) -> list[str]:
    return sorted(
        c
        for c, lab in labels.items()
        if lab.get("purpose") == "eval" and lab["clip"].get("contact_frame") is not None
    )


def detect_all(cid: str, lab: dict, meta: dict) -> dict[str, dict]:
    """{method: {"frame", "leg", "detail"}}; frame None = the method gave no answer."""
    spf = meta["seconds_per_frame"] if meta.get("fps_trusted") else None
    xy = clean(cache.load("rtmpose", cid), seconds_per_frame=spf).xy
    cands = ball_detector.load(cid) if ball_detector.cache_path(cid).exists() else None
    out: dict[str, dict] = {}
    for name, lead in (("ball", events.BRACKET_LEAD_FRAMES), ("ball_uncorrected", 0)):
        ev = events.detect(xy, cands, bracket_lead_frames=lead)
        is_ball = ev.contact_method in ("ball_motion", "ball_bracket")
        bc = ev.ball_contact
        out[name] = {
            "frame": ev.contact_frame if is_ball else None,
            "leg": ev.kicking_leg.leg if is_ball and ev.kicking_leg else None,
            "detail": ev.contact_method if is_ball else "; ".join(ev.notes) or "no ball contact",
            "bracket": (bc.frame, (bc.next_seen if bc.next_seen is not None else len(xy)) - 1)
            if is_ball and ev.contact_method == "ball_bracket"
            else None,
        }
        if name == "ball":
            out["v2_pipeline"] = {
                "frame": ev.contact_frame,
                "leg": ev.kicking_leg.leg if ev.kicking_leg else None,
                "detail": ev.contact_method,
            }
    for name, given in (("foot_speed", None), ("foot_speed_given_leg", lab["clip"]["kicking_leg"])):
        kl = leg_from_speed(xy, given)
        out[name] = {
            "frame": kl.kick_frame if kl else None,
            "leg": kl.leg if kl else None,
            "detail": kl.source if kl else "no foot track",
        }
    v1 = v1_replica.kick(cache.load(v1_replica.CACHE_MODEL, cid), meta["playback_fps"])
    out["v1"] = {
        "frame": v1.frame if v1 else None,
        "leg": v1.leg if v1 else None,
        "detail": "highest one-step ankle speed",
    }
    return out


def rows_for(cid: str, split: str, lab: dict, found: dict) -> list[dict]:
    c, leg = lab["clip"]["contact_frame"], lab["clip"]["kicking_leg"]
    rows = []
    for m in METHODS:
        f = found[m]
        rows.append(
            {
                "clip_id": cid,
                "split": split,
                "method": m,
                "label_frame": c,
                "frame": f["frame"],
                "err": None if f["frame"] is None else f["frame"] - c,
                "leg": f["leg"],
                "leg_ok": None if f["leg"] is None else f["leg"] == leg,
                "detail": f["detail"],
                "bracket": f.get("bracket"),
            }
        )
    return rows


def metrics(rows: list[dict]) -> dict:
    n = len(rows)
    err = np.array([r["err"] for r in rows if r["err"] is not None], dtype=float)
    a = np.abs(err)
    out = {"n": n, "failed": n - len(err)}
    if n == 0:
        return out
    # Within-k rates are over all clips: a failure is not within anything.
    out |= {
        "exact": float((a == 0).sum() / n),
        "within1": float((a <= 1).sum() / n),
        "within2": float((a <= 2).sum() / n),
        "gross": int((a > GROSS_FRAMES).sum()),
        "mae": float(a.mean()) if len(a) else np.nan,
        "median_abs": float(np.median(a)) if len(a) else np.nan,
        "median_signed": float(np.median(err)) if len(err) else np.nan,
        "max_abs": float(a.max()) if len(a) else np.nan,
    }
    return out


def bracket_analysis(rows: list[dict]) -> dict:
    """The pre-stated rule for the bracket correction (see the docstring)."""
    br = [r for r in select(rows, method="ball_uncorrected") if r["bracket"] is not None]
    tune = [r for r in br if r["split"] == "tune"]
    held = [r for r in br if r["split"] == "heldout"]

    def shifted(r: dict, k: int) -> int:
        return min(r["frame"] + k, r["bracket"][1])

    k_all = -int(np.median([r["err"] for r in tune])) if tune else 0
    loo = []
    for i, r in enumerate(tune):
        others = [o["err"] for j, o in enumerate(tune) if j != i]
        k = -int(np.median(others)) if others else 0
        loo.append(
            {
                "clip_id": r["clip_id"],
                "k": k,
                "before": r["err"],
                "after": shifted(r, k) - r["label_frame"],
            }
        )
    held_rows = [
        {
            "clip_id": r["clip_id"],
            "k": k_all,
            "before": r["err"],
            "after": shifted(r, k_all) - r["label_frame"],
        }
        for r in held
    ]

    def mae(rs: list[dict], key: str) -> float:
        return float(np.mean([abs(r[key]) for r in rs])) if rs else np.nan

    loo_ok = mae(loo, "after") < mae(loo, "before")
    held_ok = not (mae(held_rows, "after") > mae(held_rows, "before"))
    return {
        "k": k_all,
        "tune": [
            {
                "clip_id": r["clip_id"],
                "bracket": r["bracket"],
                "pick": r["frame"],
                "label": r["label_frame"],
                "err": r["err"],
            }
            for r in tune
        ],
        "loo": loo,
        "heldout": held_rows,
        "loo_mae": (mae(loo, "before"), mae(loo, "after")),
        "heldout_mae": (mae(held_rows, "before"), mae(held_rows, "after")),
        "adopt": bool(loo_ok and held_ok),
        "n_early": sum(r["err"] < 0 for r in tune),
        "n_late": sum(r["err"] > 0 for r in tune),
    }


def run(labels: dict, manifest: dict) -> dict:
    rows = []
    for cid in clips_with_contact(labels, manifest):
        found = detect_all(cid, labels[cid], load_meta(FRAMES / cid))
        rows += rows_for(cid, manifest[cid]["split"], labels[cid], found)
    shipped = {
        c: json.loads((RESULTS / "pipeline" / f"{c}.json").read_text())["events"]["contact_frame"]
        for c in {r["clip_id"] for r in rows}
    }
    # The pipeline's saved output must match what E3 recomputes, or results/ is stale.
    stale = [
        r["clip_id"]
        for r in select(rows, method="v2_pipeline")
        if shipped[r["clip_id"]] != r["frame"]
    ]
    return {"rows": rows, "bracket": bracket_analysis(rows), "stale": stale}


def report(out: dict) -> str:
    rows, b = out["rows"], out["bracket"]
    clips = sorted({r["clip_id"] for r in rows})
    n_held = len({r["clip_id"] for r in rows if r["split"] == "heldout"})
    L = [
        "# E3: contact detection",
        "",
        "Generated by `uv run python -m eval.e3_contact`. Truth: Aamir's contact frame (DEC-015).",
        'Error = detected − labelled, in frames. Within ±k and "gross" (> '
        f"{GROSS_FRAMES} frames off: a different event, not an imprecise touch) count a failure",
        "as a miss. MAE and medians are over the clips a method answered.",
        "",
        f"**n:** {len(clips)} clips ({len(clips) - n_held} tuning, {n_held} held-out) with a",
        "labelled contact. c20, c23, c24 and c25 have none yet; the 3 gate-only rear views are",
        "refused before contact matters, so they aren't scored.",
        "",
        "| method | split | n | failed | exact | ±1 | ±2 | gross | MAE (frames) "
        "| median signed | max abs err |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in METHODS:
        for split in ("all", "tune", "heldout"):
            rs = select(rows, method=m) if split == "all" else select(rows, method=m, split=split)
            s = metrics(rs)
            L.append(
                f"| {m} | {split} | {s['n']} | {s['failed']} | {fmt(s['exact'], True)} | "
                f"{fmt(s['within1'], True)} | {fmt(s['within2'], True)} | {s['gross']} | "
                f"{fmt(s['mae'], nd=2)} | {fmt(s['median_signed'], nd=1)} | "
                f"{fmt(s['max_abs'], nd=0)} |"
            )
    L += ["", "Kicking leg chosen by each method (it decides which leg's angles are reported):", ""]
    body = []
    for m in ("ball", "foot_speed", "v2_pipeline", "v1"):
        rs = [r for r in select(rows, method=m) if r["leg_ok"] is not None]
        body.append([m, f"{sum(r['leg_ok'] for r in rs)}/{len(select(rows, method=m))}"])
    L += table(["method", "labelled leg chosen"], body)
    L += ["", "Per clip (detected frame, error):", ""]
    body = []
    for c in clips:
        cells = [
            c,
            select(rows, clip_id=c)[0]["split"],
            str(select(rows, clip_id=c)[0]["label_frame"]),
        ]
        for m in METHODS:
            r = select(rows, clip_id=c, method=m)[0]
            cells.append("–" if r["frame"] is None else f"{r['frame']} ({r['err']:+d})")
        body.append(cells)
    L += table(["clip", "split", "label", *METHODS], body)

    lo, hi = b["loo_mae"]
    hlo, hhi = b["heldout_mae"]
    L += [
        "",
        "## The bracket correction (DEC-020's open question → DEC-022)",
        "",
        "When the ball is hidden at the touch, the pick is the frame inside the bracket",
        "[last seen at rest, first seen moved − 1] where one foot is fastest relative to the",
        f"other. On the {len(b['tune'])} tuning bracket clips it was early on {b['n_early']} and",
        f"late on {b['n_late']}; the median offset gives a shift of {b['k']:+d} frames, kept",
        "inside the bracket.",
        "",
    ]
    L += table(
        [
            "clip",
            "bracket",
            "uncorrected pick",
            "label",
            "error before",
            "LOO shift",
            "error after (LOO)",
        ],
        [
            [
                r["clip_id"],
                f"{r['bracket'][0]}–{r['bracket'][1]}",
                str(r["pick"]),
                str(r["label"]),
                f"{r['err']:+d}",
                f"{lr['k']:+d}",
                f"{lr['after']:+d}",
            ]
            for r, lr in zip(b["tune"], b["loo"], strict=True)
        ],
    )
    L += [
        "",
        f"- Leave-one-clip-out on tuning: mean |error| {lo:.2f} → {hi:.2f} frames.",
        "- Held-out bracket clips: "
        + (
            ", ".join(f"{r['clip_id']} {r['before']:+d} → {r['after']:+d}" for r in b["heldout"])
            or "none"
        )
        + (f" (mean |error| {hlo:.2f} → {hhi:.2f})." if b["heldout"] else "."),
        f"- **Rule result: {'adopt' if b['adopt'] else 'do not adopt'}.**"
        + (
            " The held-out set can only show harm here: its one bracket clip has a 3-frame"
            " bracket, so the clamp leaves its pick unchanged."
            if len(b["heldout"]) == 1
            else ""
        ),
    ]
    if out["stale"]:
        L += [
            "",
            f"⚠️ results/pipeline/ is stale for {', '.join(out['stale'])}: "
            "rerun tools/run_pipeline.py.",
        ]
    return "\n".join(L) + "\n"


def main() -> None:
    labels = schema.load_all()
    manifest = load_manifest()
    out = run(labels, manifest)
    write_csv(
        RESULTS / "e3.csv",
        [
            {**r, "bracket": r["bracket"] and f"{r['bracket'][0]}-{r['bracket'][1]}"}
            for r in out["rows"]
        ],
    )
    md = report(out)
    (RESULTS / "e3_summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
