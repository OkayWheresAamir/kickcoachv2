"""E4: can a frontier video model measure the same angles? (eval plan, E4 protocol)

Gemini (``gemini-3.8-flash``, the strongest free-tier model that takes video; DEC-003,
DEC-012) gets each trimmed clip, the kicking leg when it is labelled, the exact angle
definitions the pipeline uses, a fixed JSON schema and an explicit "cannot determine"
(eval/e4_prompt.py). Scored against the same labels as E2 and E3.

Request settings (DEC-023), each giving the baseline the most the free tier allows:
- video sampled at min(playback fps, 24): 24 is the API maximum (the 1 fps default would
  miss the kick);
- ``media_resolution`` HIGH: 280 tokens a frame instead of 70. Google's media-resolution
  page says HIGH is for "small details within video frames", and a knee angle on a player
  who fills a fifth of the frame is that;
- ``thinking_level`` HIGH: the 3.8 Flash guide recommends it for "split-second movement
  detection (sports, editing)"; its default is MEDIUM;
- no temperature: the same guide says sampling parameters "are ignored by the backend", so
  the gut-check's temperature 0 never applied. Answers can vary between calls, so a few
  clips are asked twice (``--repeat 1``) to measure that.

Raw responses are cached in results/e4_raw/ and committed, so scoring reruns without calls.

    uv run python -m eval.e4_gemini collect [--clips c01 c02 ...] [--repeat N]
    uv run python -m eval.e4_gemini            # score from the cache only
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from eval.e1_pose import fmt, load_manifest, table, write_csv
from eval.e4_prompt import PROMPT_VERSION, SCHEMA, build_prompt
from kickcoach.labels import schema
from kickcoach.llm import DailyQuotaExhausted, GeminiClient, RateLimited
from kickcoach.llm.gemini import E4_MODEL

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RAW = RESULTS / "e4_raw"
MAX_VIDEO_FPS = 24  # API limit: "threshold must be less than or equal to 24"
MEDIA_RESOLUTION = "MEDIA_RESOLUTION_HIGH"
THINKING_LEVEL = "HIGH"
# Clips asked a second time to see how much an answer changes between identical calls. c01
# is included because its first answer said the video was "a single frozen frame".
REPEAT_CLIPS = ("c01", "c03", "c07", "c10", "c13", "c16")


def clip_meta(cid: str) -> dict:
    return json.loads((ROOT / "data" / "frames" / cid / "meta.json").read_text())


def video_fps(meta: dict) -> int:
    return min(round(meta["playback_fps"]), MAX_VIDEO_FPS)


def request_for(cid: str, labels: dict) -> dict:
    """The exact request E4 sends for a clip (also used to find its cached answer)."""
    meta = clip_meta(cid)
    leg = labels.get(cid, {}).get("clip", {}).get("kicking_leg")
    return {
        "prompt": build_prompt(meta["n_frames"] / meta["playback_fps"], leg),
        "schema": SCHEMA,
        "video": ROOT / "data" / "raw" / "clips" / f"{cid}.mp4",
        "video_fps": video_fps(meta),
        "temperature": None,
        "media_resolution": MEDIA_RESOLUTION,
        "thinking_level": THINKING_LEVEL,
    }


def collect(jobs: list[tuple[str, int]]) -> None:
    """Ask Gemini for every (clip, repeat) not cached yet. The free tier allows 20 requests a
    day for this model, so a run stops at the daily quota; the next run skips what's cached
    and carries on (the quota resets at midnight Pacific)."""
    labels = schema.load_all()
    client = GeminiClient(E4_MODEL, RAW)
    todo = [(c, r) for c, r in jobs if client.cached(**request_for(c, labels), repeat=r) is None]
    print(f"{len(jobs) - len(todo)} of {len(jobs)} requests cached; {len(todo)} to ask", flush=True)
    for i, (cid, r) in enumerate(todo):
        try:
            res = client.generate_json(
                **request_for(cid, labels), repeat=r, tag=f"e4 {cid} {PROMPT_VERSION} r{r}"
            )
        except DailyQuotaExhausted:
            print(f"daily quota used up; {len(todo) - i} requests left for another day", flush=True)
            return
        except RateLimited as e:
            # Recorded as no answer at scoring time; a rerun retries only what's missing.
            print(f"{cid} r{r}: no answer after all retries ({str(e)[:80]})", flush=True)
            continue
        usage = res["response"].get("usage_metadata", {})
        print(
            f"{cid} r{r}: called prompt_tokens={usage.get('prompt_token_count')} "
            f"thoughts={usage.get('thoughts_token_count')} {json.dumps(res['parsed'])}",
            flush=True,
        )


def all_jobs(manifest: dict[str, dict], labels: dict) -> list[tuple[str, int]]:
    """Every E4 request, most useful first: clips with a labelled contact (they get scored),
    then the repeat calls, then the clips without labels (answer and refusal behaviour only)."""
    scored = [
        c
        for c in manifest
        if manifest[c]["purpose"] == "eval"
        and labels.get(c, {}).get("clip", {}).get("contact_frame") is not None
    ]
    rest = [c for c in manifest if c not in scored]
    return [(c, 0) for c in scored] + [(c, 1) for c in REPEAT_CLIPS] + [(c, 0) for c in rest]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="score", choices=("collect", "score"))
    ap.add_argument("--clips", nargs="*")
    ap.add_argument("--repeat", type=int, default=0)
    args = ap.parse_args()
    manifest = load_manifest()
    labels = schema.load_all()
    if args.mode == "collect":
        if args.clips:
            collect([(c, r) for c in args.clips for r in range(args.repeat + 1)])
        else:
            collect(all_jobs(manifest, labels))
    else:
        score_main(manifest, labels)


# heuristic: words that mark a number as an estimate. Used only to count how often Gemini
# states numbers without any hedge (eval plan, E4 metrics).
HEDGE = re.compile(r"approx|about|around|roughly|estimat|appear|seem|likely|probabl|~|unclear")
FIELDS = {  # E4 answer field -> E2 feature it is scored as
    "knee_angle_contact_deg": "knee_angle_contact",
    "max_knee_flexion_deg": "max_knee_flexion_backswing",
    "trunk_lean_contact_deg": "trunk_lean_contact",
}


def _num(x) -> float | None:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def answers(cid: str, labels: dict, client: GeminiClient, repeat: int = 0) -> dict | None:
    """Gemini's parsed answer plus derived frames, or None if this call isn't cached."""
    rec = client.cached(**request_for(cid, labels), repeat=repeat)
    return None if rec is None else parse_answer(rec, clip_meta(cid)["playback_fps"])


def parse_answer(rec: dict, fps: float) -> dict:
    """A cached record -> numbers, frames (time x playback fps), and answer flags."""
    p = rec.get("parsed") or {}
    usage = rec.get("response", {}).get("usage_metadata", {})
    out = {
        "key": rec["key"],
        "parsed_ok": rec.get("parsed") is not None,
        "kicking_leg": p.get("kicking_leg"),
        "reason": p.get("cannot_determine_reason"),
        "notes": p.get("notes") or "",
        "prompt_tokens": usage.get("prompt_token_count"),
        "thought_tokens": usage.get("thoughts_token_count"),
    }
    for f in (*FIELDS, "contact_time_s", "max_backswing_time_s"):
        out[f] = _num(p.get(f))
    for t, fr in (("contact_time_s", "contact_frame"), ("max_backswing_time_s", "backswing_frame")):
        out[fr] = None if out[t] is None else round(out[t] * fps)
    nums = [out[f] for f in (*FIELDS, "contact_time_s")]
    out["n_numbers"] = sum(v is not None for v in nums)
    out["hedged"] = bool(HEDGE.search(f"{out['notes']} {out['reason'] or ''}".lower()))
    return out


def score_rows(manifest: dict, labels: dict, client: GeminiClient) -> list[dict]:
    from eval.e2_angles import pipeline_values, truth

    rows = []
    for cid in manifest:
        a = answers(cid, labels, client)
        lab = labels.get(cid, {})
        clip = lab.get("clip", {})
        has_truth = lab.get("frames") and clip.get("kicking_leg") and clip.get("kick_direction")
        tr = truth(lab) if has_truth else {}
        res = json.loads((RESULTS / "pipeline" / f"{cid}.json").read_text())
        pv = pipeline_values(res)
        row = {
            "clip_id": cid,
            "split": manifest[cid]["split"],
            "purpose": manifest[cid]["purpose"],
            "view_ok_human": clip.get("view_ok_human"),
            "leg_given": clip.get("kicking_leg") is not None,
            "asked": a is not None,
            "parsed_ok": a["parsed_ok"] if a else None,
            "gemini_leg": a["kicking_leg"] if a else None,
            "label_leg": clip.get("kicking_leg"),
            "n_numbers": a["n_numbers"] if a else None,
            "hedged": a["hedged"] if a else None,
            "reason": a["reason"] if a else None,
            "pipeline_accepted": res["quality"]["accepted"],
        }
        for fname, fid in FIELDS.items():
            g = a[fname] if a else None
            t = tr.get(fid)
            p = pv[fid]["value"]
            row[f"{fid}_truth"] = round(t["truth"], 2) if t else None
            row[f"{fid}_gemini"] = g
            row[f"{fid}_gemini_err"] = round(g - t["truth"], 2) if t and g is not None else None
            row[f"{fid}_pipeline"] = p
            row[f"{fid}_pipeline_err"] = round(p - t["truth"], 2) if t and p is not None else None
        for fr, key, pkey in (
            ("contact", "contact_frame", "contact_frame"),
            ("backswing", "backswing_frame", "max_backswing_frame"),
        ):
            lf = clip.get(f"{fr}_frame" if fr == "contact" else "max_backswing_frame")
            g = a[key] if a else None
            p = res["events"][pkey]
            row[f"{fr}_label"] = lf
            row[f"{fr}_gemini"] = g
            row[f"{fr}_gemini_err"] = g - lf if g is not None and lf is not None else None
            row[f"{fr}_pipeline"] = p
            row[f"{fr}_pipeline_err"] = p - lf if p is not None and lf is not None else None
        row["prompt_tokens"] = a["prompt_tokens"] if a else None
        row["thought_tokens"] = a["thought_tokens"] if a else None
        row["notes"] = a["notes"] if a else None
        rows.append(row)
    return rows


def repeat_rows(labels: dict, client: GeminiClient) -> list[dict]:
    rows = []
    for cid in REPEAT_CLIPS:
        a0, a1 = answers(cid, labels, client, 0), answers(cid, labels, client, 1)
        if a0 is None or a1 is None:
            continue
        row = {"clip_id": cid}
        for f in (*FIELDS, "contact_frame", "backswing_frame"):
            x, y = a0[f], a1[f]
            row[f"{f}_r0"], row[f"{f}_r1"] = x, y
            row[f"{f}_diff"] = None if x is None or y is None else round(y - x, 2)
        rows.append(row)
    return rows


def _err_stats(errs: list[float]) -> str:
    e = np.array(errs, dtype=float)
    if len(e) == 0:
        return "– / –"
    return f"{np.mean(np.abs(e)):.1f} / {np.mean(e):+.1f}"


def report(rows: list[dict], reps: list[dict]) -> str:
    asked = [r for r in rows if r["asked"]]
    scored = [r for r in asked if r["contact_label"] is not None and r["purpose"] == "eval"]
    L = [
        "# E4: Gemini as a measuring instrument",
        "",
        "Generated by `uv run python -m eval.e4_gemini` from the cached responses in",
        f"`results/e4_raw/` (no API calls). Model `{E4_MODEL}` (the strongest free-tier model",
        f"that takes video; a Flash model, not Pro), prompt `{PROMPT_VERSION}`",
        "(`eval/e4_prompt.py`): the angle definitions the pipeline uses, the labelled kicking",
        'leg when there is one, a JSON schema and an explicit "cannot determine". Video at',
        f"min(playback fps, {MAX_VIDEO_FPS}) frames/s (so a 30 fps clip loses 1 frame in 5),",
        f"`media_resolution` {MEDIA_RESOLUTION.split('_')[-1]}, `thinking_level` {THINKING_LEVEL};",
        "no temperature (ignored by this model). Frames = Gemini's time × playback fps.",
        "",
        f"**n:** asked on {len(asked)} of {len(rows)} clips; {len(scored)} have labels to score",
        "against.",
        "",
        "## 1. Did it answer?",
        "",
    ]
    body = []
    for name, rs in (("scored clips", scored), ("all asked clips", asked)):
        n = len(rs)
        full = sum(r["n_numbers"] == len(FIELDS) + 1 for r in rs)
        none = sum(r["n_numbers"] == 0 for r in rs)
        nums = [r for r in rs if r["n_numbers"]]
        body.append(
            [
                name,
                str(n),
                f"{full}/{n}",
                f"{none}/{n}",
                f"{sum(not r['hedged'] for r in nums)}/{len(nums)}",
            ]
        )
    L += table(
        [
            "clips",
            "n",
            "all 4 numbers",
            "no numbers (cannot determine)",
            "answers with numbers and no hedge word",
        ],
        body,
    )
    refusals = [r for r in asked if r["n_numbers"] == 0 or r["reason"]]
    if refusals:
        L += ["", "Cannot-determine reasons, verbatim:", ""]
        L += [f"- **{r['clip_id']}** ({r['n_numbers']} numbers): {r['reason']}" for r in refusals]
    not_ok = [r for r in asked if r["view_ok_human"] == "n"]
    if not_ok:
        L += [
            "",
            f"Clips Aamir marked not side-on (the pipeline refuses all {len(not_ok)}): Gemini gave "
            f"numbers on {sum(r['n_numbers'] > 0 for r in not_ok)} of {len(not_ok)} ("
            + ", ".join(f"{r['clip_id']} {r['n_numbers']}" for r in not_ok)
            + " numbers).",
        ]
    asked_leg = [r for r in asked if not r["leg_given"]]
    if asked_leg:
        L += [
            "",
            "Kicking leg where the prompt didn't give it: "
            + ", ".join(f"{r['clip_id']} {r['gemini_leg']}" for r in asked_leg)
            + ".",
        ]

    L += [
        "",
        "## 2. Angle error: Gemini vs the pipeline",
        "",
        'MAE ° / bias °. "Both answered" = the clips where Gemini and the pipeline both gave',
        "a number, which is the fair comparison (eval plan, E4 protocol).",
        "",
    ]
    body = []
    for fid in FIELDS.values():
        for split in ("all", "tune", "heldout"):
            rs = [r for r in scored if split == "all" or r["split"] == split]
            rs = [r for r in rs if r[f"{fid}_truth"] is not None]
            g = [r[f"{fid}_gemini_err"] for r in rs if r[f"{fid}_gemini_err"] is not None]
            both = [
                r
                for r in rs
                if r[f"{fid}_gemini_err"] is not None and r[f"{fid}_pipeline_err"] is not None
            ]
            body.append(
                [
                    fid,
                    split,
                    f"{len(g)}/{len(rs)}",
                    _err_stats(g),
                    _err_stats([r[f"{fid}_gemini_err"] for r in both]),
                    _err_stats([r[f"{fid}_pipeline_err"] for r in both]),
                    str(len(both)),
                ]
            )
    L += table(
        [
            "feature",
            "split",
            "Gemini answered",
            "Gemini MAE / bias",
            "both answered: Gemini",
            "both answered: pipeline",
            "n both",
        ],
        body,
    )
    # A low error needs the numbers to move with the truth: a model that always says ~150°
    # can still have a modest MAE if the truths happen to sit near 150.
    body = []
    for fid in FIELDS.values():
        both = [
            r
            for r in scored
            if r[f"{fid}_truth"] is not None
            and r[f"{fid}_gemini"] is not None
            and r[f"{fid}_pipeline"] is not None
        ]
        if len(both) < 3:
            continue
        t = np.array([r[f"{fid}_truth"] for r in both])
        g = np.array([r[f"{fid}_gemini"] for r in both])
        p = np.array([r[f"{fid}_pipeline"] for r in both])
        body.append(
            [
                fid,
                str(len(both)),
                f"{t.min():.0f}–{t.max():.0f}",
                f"{g.min():.0f}–{g.max():.0f}",
                f"{np.corrcoef(t, g)[0, 1]:+.2f}",
                f"{np.corrcoef(t, p)[0, 1]:+.2f}",
            ]
        )
    L += ["", "Do the numbers move with the truth? (Pearson r, clips where both answered)", ""]
    L += table(["feature", "n", "truth range °", "Gemini range °", "r Gemini", "r pipeline"], body)
    # If Gemini found the right moment, its angle error can't be blamed on timing.
    on_time = [
        r
        for r in scored
        if r["contact_gemini_err"] is not None and abs(r["contact_gemini_err"]) <= 1
    ]
    L += [
        "",
        f"On the {len(on_time)} clips where Gemini's contact time was within ±1 frame of the",
        "label, so a wrong moment can't explain its contact angles (MAE ° / bias °): knee "
        + _err_stats(
            [
                r["knee_angle_contact_gemini_err"]
                for r in on_time
                if r["knee_angle_contact_gemini_err"] is not None
            ]
        )
        + "; trunk "
        + _err_stats(
            [
                r["trunk_lean_contact_gemini_err"]
                for r in on_time
                if r["trunk_lean_contact_gemini_err"] is not None
            ]
        )
        + ".",
    ]

    L += ["", "## 3. Event timing (frames)", ""]
    body = []
    for fr in ("contact", "backswing"):
        rs = [r for r in scored if r[f"{fr}_label"] is not None]
        g = np.array([r[f"{fr}_gemini_err"] for r in rs if r[f"{fr}_gemini_err"] is not None])
        p = np.array([r[f"{fr}_pipeline_err"] for r in rs if r[f"{fr}_pipeline_err"] is not None])
        for name, e in (("Gemini", g), ("pipeline", p)):
            a = np.abs(e)
            body.append(
                [
                    fr,
                    name,
                    f"{len(e)}/{len(rs)}",
                    fmt(float((a <= 1).mean()) if len(e) else np.nan, True),
                    fmt(float((a <= 2).mean()) if len(e) else np.nan, True),
                    fmt(float(a.mean()) if len(e) else np.nan, nd=1),
                    fmt(float(np.median(e)) if len(e) else np.nan, nd=1),
                ]
            )
    L += table(
        [
            "event",
            "system",
            "answered",
            "±1 (of answered)",
            "±2 (of answered)",
            "MAE",
            "median signed",
        ],
        body,
    )

    if reps:
        L += [
            "",
            "## 4. Same question twice",
            "",
            "The model ignores temperature, so identical requests can differ. Second answer −",
            "first, on the clips asked twice:",
            "",
        ]
        cols = [*FIELDS, "contact_frame", "backswing_frame"]
        L += table(
            ["clip", *cols],
            [
                [
                    r["clip_id"],
                    *("–" if r[f"{c}_diff"] is None else f"{r[f'{c}_diff']:+g}" for c in cols),
                ]
                for r in reps
            ],
        )

    def trio(r: dict, fid: str) -> str:
        vals = (r[f"{fid}_truth"], r[f"{fid}_gemini"], r[f"{fid}_pipeline"])
        return " / ".join("–" if v is None else f"{v:.0f}" for v in vals)

    def opt(v) -> str:
        return "–" if v is None else str(v)

    L += ["", "## Per clip", ""]
    L += table(
        [
            "clip",
            "split",
            "knee contact: truth / Gemini / pipeline",
            "max flexion: truth / G / P",
            "trunk: truth / G / P",
            "contact frame: label / Gemini / pipeline",
            "tokens (prompt / thinking)",
        ],
        [
            [
                r["clip_id"],
                r["split"],
                *(trio(r, fid) for fid in FIELDS.values()),
                f"{opt(r['contact_label'])} / {opt(r['contact_gemini'])} / "
                f"{opt(r['contact_pipeline'])}",
                f"{r['prompt_tokens']} / {r['thought_tokens']}",
            ]
            for r in asked
        ],
    )
    return "\n".join(L) + "\n"


def score_main(manifest: dict[str, dict], labels: dict) -> None:
    client = GeminiClient(E4_MODEL, RAW)
    rows = score_rows(manifest, labels, client)
    reps = repeat_rows(labels, client)
    write_csv(RESULTS / "e4.csv", rows)
    if reps:
        write_csv(RESULTS / "e4_repeat.csv", reps)
    md = report(rows, reps)
    (RESULTS / "e4_summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
