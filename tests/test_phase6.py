"""Phase 6: the repeatability floor, E4's cache lookup without the video, and run_all's
README block."""

import json

import pytest

from eval import e4_gemini, repeatability, run_all
from kickcoach.labels import schema
from kickcoach.llm import GeminiClient

# A right-footed kicker facing image right: thigh vertical, shank horizontal (knee 90 deg).
PTS = {
    "left_shoulder": (200, 100),
    "right_shoulder": (205, 100),
    "left_hip": (200, 300),
    "right_hip": (205, 300),
    "left_knee": (200, 400),
    "right_knee": (205, 400),
    "left_ankle": (200, 500),
    "right_ankle": (305, 400),
}


def _points(moves: dict | None = None) -> dict:
    moves = moves or {}
    return {
        n: {
            "x": float(x + moves.get(n, (0, 0))[0]),
            "y": float(y + moves.get(n, (0, 0))[1]),
            "v": schema.VISIBLE,
        }
        for n, (x, y) in PTS.items()
    }


PREFILL = {n: [float(x), float(y), 0.9] for n, (x, y) in PTS.items()}


def _labels() -> dict:
    return {
        "c01": {
            "clip": {"kicking_leg": "right", "kick_direction": "right"},
            "frames": {"5": {"confirmed": True, "points": _points(), "prefill": PREFILL}},
        }
    }


def _session(points: dict, confirmed: bool = True) -> dict:
    task = {
        "clip_id": "c01",
        "frame": 5,
        "points": points,
        "prefill": PREFILL,
        "confirmed": confirmed,
    }
    return {"session_id": "r1", "tasks": [task]}


def test_repeatability_measures_the_relabel_against_the_first_label():
    # The relabel drops the right ankle 100 px: the shank goes from horizontal to 45 deg.
    pts, angs = repeatability.compare(_labels(), [_session(_points({"right_ankle": (0, 100)}))])
    ankle = next(r for r in pts if r["point"] == "right_ankle")
    hip = next(r for r in pts if r["point"] == "right_hip")
    assert ankle["diff_px"] == pytest.approx(100.0) and ankle["moved"]
    assert ankle["diff_thigh"] == pytest.approx(1.0)  # thigh = 100 px
    assert hip["diff_px"] == 0.0 and not hip["moved"]  # left at the pre-fill both times
    knee = next(r for r in angs if r["angle"] == "knee_angle")
    trunk = next(r for r in angs if r["angle"] == "trunk_lean")
    assert knee["first_deg"] == pytest.approx(90.0)
    assert knee["diff_deg"] == pytest.approx(45.0) and knee["moved"]
    assert trunk["diff_deg"] == pytest.approx(0.0) and not trunk["moved"]


def test_unconfirmed_relabels_are_not_compared_and_the_summary_says_not_measured():
    sessions = [_session(_points(), confirmed=False)]
    pts, angs = repeatability.compare(_labels(), sessions)
    assert pts == [] and angs == []
    text = "\n".join(repeatability.summary_lines(sessions, pts, angs))
    assert "Not measured yet" in text and "0 of 1 sampled frames" in text


def test_e4_finds_its_cached_answer_by_tag_when_the_video_is_absent(tmp_path, monkeypatch):
    req = {
        "prompt": "p",
        "schema": {"type": "object"},
        "video": tmp_path / "missing.mp4",
        "video_fps": 24,
        "temperature": None,
        "media_resolution": "MEDIA_RESOLUTION_HIGH",
        "thinking_level": "HIGH",
    }
    monkeypatch.setattr(e4_gemini, "request_for", lambda cid, labels: req)
    client = GeminiClient("m", tmp_path / "cache")
    want, _ = GeminiClient._request(**{**req, "video": None})
    (tmp_path / "cache").mkdir()

    def save(name, tag, request):
        rec = {"key": name, "tag": tag, "model": "m", "request": request, "parsed": {}}
        (tmp_path / "cache" / f"{name}.json").write_text(json.dumps(rec))

    save("a", e4_gemini.tag_for("c01", 0), {**want, "video_sha256": "abc"})
    save("b", e4_gemini.tag_for("c01", 0), {**want, "prompt": "an older prompt"})
    save("c", e4_gemini.tag_for("c02", 0), {**want, "video_sha256": "abc"})
    assert e4_gemini.cached_record("c01", {}, client)["key"] == "a"
    assert e4_gemini.cached_record("c01", {}, client, repeat=1) is None


def test_run_all_rewrites_only_the_readme_block(tmp_path, monkeypatch):
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{run_all.BLOCK_START}\nold\n{run_all.BLOCK_END}\noutro\n")
    monkeypatch.setattr(run_all, "README", readme)
    assert run_all.update_readme(["| new |"])
    assert readme.read_text() == (
        f"intro\n{run_all.BLOCK_START}\n| new |\n{run_all.BLOCK_END}\noutro\n"
    )
    readme.write_text("no markers\n")
    assert not run_all.update_readme(["| new |"])


def test_only_numbers_count_as_values_for_e8():
    assert run_all.is_number("12.5") and run_all.is_number("-3")
    assert not run_all.is_number("") and not run_all.is_number("proximal_to_distal")
