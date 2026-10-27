"""Phase 1 early risk check: send 3 trimmed clips to the E4 model and look at what comes back.

Not the evaluation (no labels exist yet). Raw responses are cached in
results/gemini_gutcheck/ and summarised in results/gemini_gutcheck/summary.md.

    uv run python tools/gemini_gutcheck.py [clip_id ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.e4_prompt import PROMPT_VERSION, SCHEMA, build_prompt  # noqa: E402
from kickcoach.llm import GeminiClient, RateLimited  # noqa: E402
from kickcoach.llm.gemini import E4_MODEL  # noqa: E402

OUT = ROOT / "results" / "gemini_gutcheck"
# One lab clip (1080p, slow motion), the 240 fps Ronaldo clip, the real-time outdoor clip.
DEFAULT_CLIPS = ["c10", "c19", "c25"]


def main(clips: list[str]) -> None:
    client = GeminiClient(E4_MODEL, OUT)
    rows = []
    for c in clips:
        meta = json.loads((ROOT / "data" / "frames" / c / "meta.json").read_text())
        duration = meta["n_frames"] / meta["playback_fps"]
        try:
            res = client.generate_json(
                prompt=build_prompt(duration, kicking_leg=None),
                schema=SCHEMA,
                video=ROOT / "data" / "raw" / "clips" / f"{c}.mp4",
                # As many frames as allowed: at the 1 fps default a kick is invisible, and the
                # API rejects fps > 24 ("threshold must be less than or equal to 24").
                video_fps=min(round(meta["playback_fps"]), 24),
                tag=f"gutcheck {c} {PROMPT_VERSION}",
            )
        except RateLimited as e:
            # Report the outage instead of dropping the clip; a rerun retries only this one.
            msg = f"no answer: {str(e)[:40]}... after all retries"
            rows.append((c, "-", False, {"cannot_determine_reason": msg}, None, None))
            print(c, msg, flush=True)
            continue
        p = res["parsed"] or {}
        usage = res["response"].get("usage_metadata", {})
        t = p.get("contact_time_s")
        frame = round(t * meta["playback_fps"]) if isinstance(t, (int, float)) else None
        rows.append((c, res["key"], res["cached"], p, frame, usage.get("prompt_token_count")))
        print(c, "cached" if res["cached"] else "called", json.dumps(p), flush=True)

    lines = [
        "# Gemini gut-check (Phase 1)",
        "",
        f"Model `{E4_MODEL}`, prompt `{PROMPT_VERSION}` (eval/e4_prompt.py), temperature 0,",
        "sampled at min(playback fps, 24) frames per second (the API maximum). No labels exist",
        "yet, so nothing here is scored; contact frame = contact_time_s x playback fps, for",
        "eyeballing only.",
        "",
        "| clip | leg | contact s (frame) | knee deg | trunk deg | cannot-determine reason "
        "| prompt tokens | cache file |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c, key, _cached, p, frame, toks in rows:
        lines.append(
            f"| {c} | {p.get('kicking_leg')} | {p.get('contact_time_s')} ({frame}) | "
            f"{p.get('knee_angle_contact_deg')} | {p.get('trunk_lean_contact_deg')} | "
            f"{p.get('cannot_determine_reason')} | {toks} | {key}.json |"
        )
    lines += ["", "Model notes:", ""]
    lines += [f"- **{c}:** {p.get('notes')}" for c, _k, _c, p, _f, _t in rows]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_CLIPS)
