"""One analysis session, end to end: video -> ingest -> pose -> ball -> pipeline -> norms,
faults, feedback -> the MVP response (handoff 02_MVP_SPEC.md).

``assess`` is the part after the measurements and needs no video or model, so tests and the
manifest runs use it directly. ``analyse_video`` adds the heavy front half for an upload.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from kickcoach.compare import engine, faults, norms
from kickcoach.feedback import generate_feedback
from kickcoach.figure import frame_size, pose_figure
from kickcoach.ingest import ingest_clip, probe
from kickcoach.llm import GeminiClient
from kickcoach.pipeline import analyse_with_signal

POSE_MODEL = "rtmpose"  # chosen by E1 (DEC-016)
# heuristic: an upload is one kick, not a match. 20 s at 30 fps is ~600 frames, ~4 min of
# RTMPose on this laptop's CPU (2-3 frames/s); longer uploads are refused up front.
MAX_FRAMES = 600
SLOW_MOTION = {"real_time": "n", "slow_motion": "y", "unknown": ""}


def assess(
    result: dict,
    *,
    level: str | None = None,
    use_llm: bool = True,
    client: GeminiClient | None = None,
    verified: list[norms.NormRow] | None = None,
) -> dict:
    """Add comparisons, faults and feedback to a pipeline result. A refused clip gets none of
    them: only the gate's reasons (MVP acceptance criterion 4)."""
    verified = norms.verified_norms() if verified is None else verified
    out = dict(result)
    if not result["quality"]["accepted"]:
        out.update(
            comparisons=[], faults=[], inactive_rules=[],
            feedback={"items": [], "generated_by": "none", "validator_passed": None,
                      "attempts": [], "fallback_reason": "the clip was refused: "
                      + "; ".join(result["quality"]["reasons"])},
        )  # fmt: skip
        return out
    comps = engine.compare(result["features"], verified, level)
    fired, inactive = faults.evaluate(result["features"], verified)
    fb = generate_feedback(
        result["features"], fired, comps, verified, level=level, client=client, use_llm=use_llm
    )
    out.update(comparisons=comps, faults=fired, inactive_rules=inactive, feedback=fb)
    return out


def _log(stage: str, done: int | None = None, total: int | None = None) -> None:
    if done is None or done == total:
        print(stage if done is None else f"{stage} {done}/{total}", flush=True)


def _frames(frames_dir: Path, on_frame=None):
    """The ingested frames in order; ``on_frame(done, total)`` after each one is used."""
    from PIL import Image

    paths = sorted(frames_dir.glob("*.jpg"))
    for i, p in enumerate(paths):
        yield np.asarray(Image.open(p).convert("RGB"))
        if on_frame is not None:
            on_frame(i + 1, len(paths))


def measure_video(
    video: Path,
    *,
    kicking_leg: str | None = None,
    capture_fps: float | None = None,
    slow_motion: str = "unknown",
    workdir: Path,
    progress=_log,
) -> tuple[dict, np.ndarray, list[np.ndarray]]:
    """Ingest the whole upload, run the pose model and ball detector, then the pipeline.
    Returns (pipeline result with its stick figure, poses, ball candidates); frames stay in
    ``workdir``. ``progress(stage, done, total)`` reports each stage and frame."""
    import certifi

    from kickcoach.events.ball_detector import BallDetector
    from kickcoach.pose.models import MODELS

    # rtmlib downloads its ONNX weights with urllib; python.org's macOS Python has no CA
    # bundle (STATUS, environment facts), so point it at certifi's.
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    info = probe(video)
    if not info.duration_s:
        raise ValueError("the video has no duration; is it a video file?")
    if info.r_frame_rate and info.duration_s * info.r_frame_rate > MAX_FRAMES:
        raise ValueError(
            f"the video is {info.duration_s:.1f} s long ({info.duration_s * info.r_frame_rate:.0f}"
            f" frames); trim it to the one kick (at most {MAX_FRAMES} frames)"
        )
    frames_dir = workdir / "frames"
    progress("ingest")
    meta = ingest_clip(
        video,
        clip_id="upload",
        start_s=0.0,
        end_s=info.duration_s + 1.0,
        capture_fps_known="y" if capture_fps else "n",
        capture_fps=capture_fps,
        slow_motion=SLOW_MOTION.get(slow_motion, ""),
        frames_dir=frames_dir,
        clip_path=workdir / "clip.mp4",
    )
    n = meta["n_frames"]
    model = MODELS[POSE_MODEL]()
    progress("pose", 0, n)
    poses = model.run_clip(
        _frames(frames_dir, lambda i, t: progress("pose", i, t)), meta["frame_times_s"]
    )
    det = BallDetector()
    progress("ball", 0, n)
    cands = [det.detect(img) for img in _frames(frames_dir, lambda i, t: progress("ball", i, t))]
    progress("measure")
    res, cleaned = analyse_with_signal(poses, meta, cands, kicking_leg=kicking_leg)
    res["figure"] = pose_figure(cleaned.xy, res, frame_size(meta))
    return {"pose_model": POSE_MODEL, **res}, poses, cands


def analyse_video(
    video: Path,
    *,
    kicking_leg: str | None = None,
    capture_fps: float | None = None,
    slow_motion: str = "unknown",
    level: str | None = None,
    use_llm: bool = True,
    keep_dir: Path | None = None,
    progress=_log,
) -> tuple[dict, dict[str, str]]:
    """The full analysis of an upload. The video and its frames are deleted afterwards;
    the pose array and ball candidates are kept in ``keep_dir`` (D16 artifacts), if given.
    Returns (response without session/history, artifact paths)."""
    tmp = Path(tempfile.mkdtemp(prefix="kickcoach_"))
    try:
        res, poses, cands = measure_video(
            video, kicking_leg=kicking_leg, capture_fps=capture_fps, slow_motion=slow_motion,
            workdir=tmp, progress=progress,
        )  # fmt: skip
        artifacts = {}
        if keep_dir is not None:
            keep_dir.mkdir(parents=True, exist_ok=True)
            np.save(keep_dir / "poses.npy", poses)
            (keep_dir / "ball_candidates.json").write_text(
                json.dumps([np.round(c, 2).tolist() for c in cands])
            )
            artifacts = {
                "poses": str(keep_dir / "poses.npy"),
                "ball_candidates": str(keep_dir / "ball_candidates.json"),
            }
        progress("feedback")
        return assess(res, level=level, use_llm=use_llm), artifacts
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
