"""An optional cap on onnxruntime's threads for rtmlib's models (KICKCOACH_ORT_THREADS).

rtmlib builds its sessions without options, so onnxruntime sizes its thread pool from the
cores it can see: 20 in Modal's 4-core container. Capping it there made no measurable
difference (c07: 60-107 s capped, 60-128 s not; host variance dominates; JOURNAL Phase 7),
so it is a guard and a benchmarking knob, not a speed-up. The cap changes outputs only by
float summation order: at most 0.0004 px on RTMPose keypoints and 3e-6 on YOLOX boxes.
"""

from __future__ import annotations

import os


def cap_threads(tool) -> None:
    """Rebuild an rtmlib tool's onnxruntime session with the cap, if one is set."""
    n = int(os.environ.get("KICKCOACH_ORT_THREADS") or 0)
    if n <= 0:
        return
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = n
    so.inter_op_num_threads = 1
    tool.session = ort.InferenceSession(
        tool.onnx_model, sess_options=so, providers=tool.session.get_providers()
    )
