"""The hosted demo on Modal (DEC-034): the same FastAPI app as `uvicorn api.app:app`, in
public mode, in one CPU container.

    uv run --group deploy modal deploy deploy/modal_app.py   # prints the public URL
    uv run --group deploy modal serve deploy/modal_app.py    # a temporary URL, for testing

One container at most (max_containers=1): the job queue lives in its memory, so every poll
reaches the job, and jobs run one at a time anyway. Nothing persists: public mode stores
nothing, and the container's disk goes when it scales down. No API key is given to it, so
feedback is always the template.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
APP = "/app"  # the repo layout, so the code's parents[] paths find knowledge/ and models/
# The ball detector's weights: the same file and checksum as the README's rebuild steps.
YOLOX_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.onnx"
YOLOX_SHA256 = "21ff6cfdeb53b013bac2249599e55f00bff3cfdfdab37ed7a4620818c1d15b3f"
# Physical cores (2 vCPU each, Modal pricing): c07 (150 frames) took 60-107 s warm on 4.
# ORT_THREADS matches them, since the container reports 20 CPUs; no setting measured clearly
# faster (JOURNAL Phase 7). "0" = onnxruntime's default; set KICKCOACH_ORT_THREADS to override.
CPU = 4.0
ORT_THREADS = os.environ.get("KICKCOACH_ORT_THREADS", "4")
# heuristic: an idle container stays up this long, so a visitor's second kick starts warm.
SCALEDOWN_S = 300


def fetch_weights() -> None:
    """At build time, so a cold start doesn't download: the ball detector, and the RTMPose
    and person-detector ONNX files rtmlib otherwise fetches on first use."""
    import hashlib
    import urllib.request

    from rtmlib import BodyWithFeet

    path = Path(APP, "models", "yolox", "yolox_m.onnx")
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(YOLOX_URL, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != YOLOX_SHA256:
        raise RuntimeError(f"yolox_m.onnx sha256 {digest} != {YOLOX_SHA256}")
    BodyWithFeet(mode="balanced", backend="onnxruntime", device="cpu")  # = RTMPoseFeet's


SOURCE_IGNORE = ["**/__pycache__", "**/*.pyc"]
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")  # opencv-python (rtmlib) links against them
    .uv_sync(str(ROOT), frozen=True, uv_version="0.13.1", extra_options="--no-dev")
    .run_function(fetch_weights)
    .env({"PYTHONPATH": APP, "KICKCOACH_PUBLIC": "1", "KICKCOACH_ORT_THREADS": ORT_THREADS})
    .add_local_dir(ROOT / "kickcoach", f"{APP}/kickcoach", ignore=SOURCE_IGNORE)
    .add_local_dir(ROOT / "api", f"{APP}/api", ignore=SOURCE_IGNORE)
    # The source papers and coaching texts stay local: only quoted excerpts are shipped.
    .add_local_dir(ROOT / "knowledge", f"{APP}/knowledge", ignore=["papers", "coaching_sources"])
)
app = modal.App("kickcoach", image=image)


@app.function(cpu=CPU, memory=4096, max_containers=1, scaledown_window=SCALEDOWN_S, timeout=600)
@modal.concurrent(max_inputs=32)
@modal.asgi_app()
def web():
    import os
    import platform

    from api.app import create_app

    cpu = next((ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo")
                if ln.startswith("model name")), platform.processor())  # fmt: skip
    # Hosts differ, and so do timings (JOURNAL Phase 7); the log says which one this was.
    print(
        f"container start: {cpu}; os.cpu_count() = {os.cpu_count()}; "
        f"KICKCOACH_ORT_THREADS = {os.environ.get('KICKCOACH_ORT_THREADS')}",
        flush=True,
    )

    return create_app()
