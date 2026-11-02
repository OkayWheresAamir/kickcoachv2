"""The API with a stand-in analyser (no video decoding or pose model): response shape,
storage, history, refusal and running without an API key."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import kickcoach.feedback.generate as gen
import kickcoach.llm.gemini as gemini
from api.app import create_app
from kickcoach import session

ROOT = Path(__file__).resolve().parents[1]
MVP_KEYS = {"session_id", "quality", "events", "features", "comparisons", "faults", "feedback",
            "history_delta"}  # fmt: skip


@pytest.fixture
def no_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(gemini, "ROOT", tmp_path)  # no .env
    monkeypatch.setattr(gen, "CACHE_DIR", tmp_path / "llm_cache")  # no cached answers


def fake_analyser(clip_id: str):
    def run(video, *, kicking_leg, capture_fps, slow_motion, level, use_llm, keep_dir):
        res = json.loads((ROOT / "results" / "pipeline" / f"{clip_id}.json").read_text())
        keep_dir.mkdir(parents=True, exist_ok=True)
        np.save(keep_dir / "poses.npy", np.zeros((2, 3, 3)))
        return session.assess(res, level=level, use_llm=use_llm), {}

    return run


def client(tmp_path, clip_id="c07") -> TestClient:
    app = create_app(tmp_path / "t.db", tmp_path / "sessions", fake_analyser(clip_id))
    return TestClient(app)


def upload(c: TestClient, name="kick.mp4", **form):
    return c.post("/analyse", files={"file": (name, b"not really a video", "video/mp4")},
                  data={"player": "ana", **form})  # fmt: skip


def test_upload_returns_the_mvp_shape_without_a_key(tmp_path, no_key):
    c = client(tmp_path)
    r = upload(c, level="intermediate")
    assert r.status_code == 200, r.text
    d = r.json()
    assert MVP_KEYS <= set(d)
    assert d["quality"]["accepted"] is True
    assert {f["fault_id"] for f in d["faults"]} == {
        "F-LEAN-BACK", "F-SHALLOW-BACKSWING", "F-PLANT-FAR-BEHIND",
    }  # fmt: skip
    assert d["feedback"]["generated_by"] == "template"
    assert d["feedback"]["validator_passed"] is True
    assert "no GEMINI_API_KEY" in d["feedback"]["fallback_reason"]
    assert all(f["threshold_source"] for f in d["faults"])
    assert c.get("/health").json()["feedback_llm"].startswith("no API key")


def test_sessions_history_and_detail(tmp_path, no_key):
    c = client(tmp_path)
    first = upload(c, use_llm="false").json()
    second = upload(c, use_llm="false").json()
    assert first["history_delta"] == {}
    assert second["history_delta"]["knee_angle_contact"]["delta"] == 0.0
    listed = c.get("/sessions", params={"player": "ana"}).json()["sessions"]
    assert [s["session_id"] for s in listed] == [second["session_id"], first["session_id"]]
    assert c.get(f"/sessions/{first['session_id']}").json()["session_id"] == first["session_id"]
    assert (tmp_path / "sessions" / str(first["session_id"]) / "poses.npy").exists()
    assert c.get("/sessions/999").status_code == 404


def test_refused_clip_has_reasons_and_no_values(tmp_path, no_key):
    d = upload(client(tmp_path, "c26")).json()
    assert d["quality"]["accepted"] is False and d["quality"]["reasons"]
    assert d["features"] == [] and d["faults"] == [] and d["comparisons"] == []
    assert d["feedback"]["items"] == [] and "refused" in d["feedback"]["fallback_reason"]


def test_rejects_non_video_and_bad_form(tmp_path, no_key):
    c = client(tmp_path)
    assert upload(c, name="notes.txt").status_code == 415
    assert upload(c, kicking_leg="middle").status_code == 422
    assert upload(c, capture_fps="-5").status_code == 422


def test_index_page_is_served(tmp_path):
    r = client(tmp_path).get("/")
    assert r.status_code == 200 and "KickCoach" in r.text
