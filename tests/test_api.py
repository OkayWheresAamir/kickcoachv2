"""The API with a stand-in analyser (no video decoding or pose model): the job flow, response
shape, storage, history, public mode, refusal and running without an API key."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import kickcoach.feedback.generate as gen
import kickcoach.llm.gemini as gemini
from api.app import create_app
from api.jobs import Busy, Jobs
from kickcoach import session

ROOT = Path(__file__).resolve().parents[1]
MVP_KEYS = {"session_id", "quality", "events", "features", "comparisons", "faults", "feedback",
            "history_delta"}  # fmt: skip


@pytest.fixture
def no_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(gemini, "ROOT", tmp_path)  # no .env
    monkeypatch.setattr(gen, "CACHE_DIR", tmp_path / "llm_cache")  # no cached answers


def fake_analyser(clip_id: str, seen: list | None = None):
    def run(video, *, kicking_leg, capture_fps, slow_motion, level, use_llm, keep_dir, progress):
        if seen is not None:
            seen.append({"video": video, "use_llm": use_llm, "keep_dir": keep_dir})
        progress("pose", 1, 2)
        if clip_id == "too_long":
            raise ValueError("trim it to the one kick")
        res = json.loads((ROOT / "results" / "pipeline" / f"{clip_id}.json").read_text())
        if keep_dir is not None:
            keep_dir.mkdir(parents=True, exist_ok=True)
            np.save(keep_dir / "poses.npy", np.zeros((2, 3, 3)))
        return session.assess(res, level=level, use_llm=use_llm), {}

    return run


def client(tmp_path, clip_id="c07", public=False, seen=None) -> TestClient:
    app = create_app(
        tmp_path / "t.db", tmp_path / "sessions", fake_analyser(clip_id, seen), public=public
    )
    return TestClient(app)


def submit(c: TestClient, name="kick.mp4", **form):
    return c.post("/analyse", files={"file": (name, b"not really a video", "video/mp4")},
                  data={"player": "ana", **form})  # fmt: skip


def wait(c: TestClient, job_id: str, timeout_s: float = 10.0) -> dict:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        status = c.get(f"/jobs/{job_id}").json()
        if status["state"] in {"done", "failed"}:
            return status
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} still {status['state']}")


def analyse(c: TestClient, **form) -> dict:
    r = submit(c, **form)
    assert r.status_code == 202, r.text
    status = wait(c, r.json()["job_id"])
    assert status["state"] == "done", status
    return status["result"]


def test_upload_returns_the_mvp_shape_without_a_key(tmp_path, no_key):
    c = client(tmp_path)
    d = analyse(c, level="intermediate")
    assert MVP_KEYS <= set(d)
    assert d["quality"]["accepted"] is True
    assert {f["fault_id"] for f in d["faults"]} == {
        "F-LEAN-BACK", "F-SHALLOW-BACKSWING", "F-PLANT-FAR-BEHIND",
    }  # fmt: skip
    assert d["feedback"]["generated_by"] == "template"
    assert d["feedback"]["validator_passed"] is True
    assert "no GEMINI_API_KEY" in d["feedback"]["fallback_reason"]
    assert all(f["threshold_source"] for f in d["faults"])
    health = c.get("/health").json()
    assert health["feedback_llm"].startswith("no API key") and health["public"] is False


def test_a_job_reports_progress_then_its_result(tmp_path, no_key):
    c = client(tmp_path)
    first = submit(c).json()
    assert first["state"] in {"queued", "running", "done"} and len(first["job_id"]) >= 16
    done = wait(c, first["job_id"])
    assert done["stage"] == "pose" and (done["done"], done["total"]) == (1, 2)
    assert c.get("/jobs/not-a-job").status_code == 404


def test_the_upload_is_deleted_after_analysis(tmp_path, no_key):
    seen = []
    c = client(tmp_path, seen=seen)
    analyse(c)
    assert not seen[0]["video"].exists() and not seen[0]["video"].parent.exists()


def test_sessions_history_and_detail(tmp_path, no_key):
    c = client(tmp_path)
    first = analyse(c, use_llm="false")
    second = analyse(c, use_llm="false")
    assert first["history_delta"] == {}
    assert second["history_delta"]["knee_angle_contact"]["delta"] == 0.0
    listed = c.get("/sessions", params={"player": "ana"}).json()["sessions"]
    assert [s["session_id"] for s in listed] == [second["session_id"], first["session_id"]]
    assert c.get(f"/sessions/{first['session_id']}").json()["session_id"] == first["session_id"]
    assert (tmp_path / "sessions" / str(first["session_id"]) / "poses.npy").exists()
    assert c.get("/sessions/999").status_code == 404


def test_history_sent_by_the_page_is_used_for_the_delta(tmp_path, no_key):
    c = client(tmp_path)
    knee = next(f["value"] for f in analyse(c)["features"] if f["id"] == "knee_angle_contact")
    earlier = [{"knee_angle_contact": knee - 4.0}, {"knee_angle_contact": knee - 2.0}]
    d = analyse(c, history=json.dumps(earlier))
    assert d["history_delta"]["knee_angle_contact"] == {
        "delta": 3.0, "vs_mean_of": 2, "text": "+3.0 vs last 2 sessions",
    }  # fmt: skip
    assert analyse(c, history="[]")["history_delta"] == {}  # sent, empty: not the database's
    assert submit(c, history="{not json").status_code == 422
    assert submit(c, history=json.dumps([1, 2])).status_code == 422


def test_public_mode_stores_nothing_and_never_calls_the_llm(tmp_path, no_key, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "a-key-that-must-not-be-used")
    seen = []
    c = client(tmp_path, public=True, seen=seen)
    d = analyse(c, history=json.dumps([{"knee_angle_contact": 100.0}]))
    assert d["session_id"] is None
    assert seen[0]["use_llm"] is False and seen[0]["keep_dir"] is None
    assert d["feedback"]["generated_by"] == "template"
    assert d["history_delta"]["knee_angle_contact"]["vs_mean_of"] == 1
    assert not (tmp_path / "t.db").exists() and not (tmp_path / "sessions").exists()
    assert c.get("/sessions").status_code == 404 and c.get("/sessions/1").status_code == 404
    health = c.get("/health").json()
    assert health["public"] is True and health["feedback_llm"].startswith("off")


def test_refused_clip_has_reasons_and_no_values(tmp_path, no_key):
    d = analyse(client(tmp_path, "c26"))
    assert d["quality"]["accepted"] is False and d["quality"]["reasons"]
    assert d["features"] == [] and d["faults"] == [] and d["comparisons"] == []
    assert d["feedback"]["items"] == [] and "refused" in d["feedback"]["fallback_reason"]


def test_a_video_the_pipeline_rejects_fails_the_job_with_its_reason(tmp_path, no_key):
    c = client(tmp_path, "too_long")
    status = wait(c, submit(c).json()["job_id"])
    assert status["state"] == "failed"
    assert status["error"] == {"status": 422, "detail": "trim it to the one kick"}


def test_rejects_non_video_and_bad_form(tmp_path, no_key):
    c = client(tmp_path)
    assert submit(c, name="notes.txt").status_code == 415
    assert submit(c, kicking_leg="middle").status_code == 422
    assert submit(c, capture_fps="-5").status_code == 422


def test_a_full_queue_refuses_new_jobs():
    release = threading.Event()
    jobs = Jobs(max_waiting=1)
    running = jobs.submit(lambda job: release.wait(5) and {})
    while running.state != "running":
        time.sleep(0.01)
    queued = jobs.submit(lambda job: {})
    assert jobs.ahead_of(queued) == 1
    with pytest.raises(Busy):
        jobs.submit(lambda job: {})
    release.set()


def test_index_page_and_example_are_served(tmp_path):
    c = client(tmp_path)
    r = c.get("/")
    assert r.status_code == 200 and "KickCoach" in r.text
    ex = c.get("/example").json()
    assert ex["example"]["clip_id"] and ex["quality"]["accepted"] is True
    assert set(ex["figure"]["frames"]) == {"contact", "max_backswing"}
