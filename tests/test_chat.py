"""Phase 9: the coach chat's tools, answer checker, agent loop and turn cache, with fakes
(no API calls). The tools run on E9's fixture players, built from committed pipeline output."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from eval.e9_chat import build_fixture, load_spec, pipeline, refused, tool_choice
from kickcoach.chat import agent
from kickcoach.chat.tools import DECLARATIONS, Tools, typical_errors
from kickcoach.chat.validate import check, grounding
from kickcoach.llm import GeminiClient, NoApiKey
from kickcoach.storage import db


@pytest.fixture(scope="module")
def con(tmp_path_factory):
    path = tmp_path_factory.mktemp("chat") / "fixture.db"
    build_fixture(path, load_spec())
    c = db.connect(path)
    yield c
    c.close()


class FakeEmbedder:
    """Deterministic unit vectors from a text hash: enough to exercise ranking and filters."""

    def vectors(self, inputs):
        rows = [np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest()[:8], 16))
                .normal(size=8) for t in inputs]  # fmt: skip
        m = np.stack(rows)
        return m / np.linalg.norm(m, axis=1, keepdims=True)


def feature(clip, fid):
    return next(f for f in pipeline(clip)["features"] if f["id"] == fid)["value"]


# -- tools ----------------------------------------------------------------------------------
def test_history_trend_is_computed_not_left_to_the_model(con):
    t = Tools(con, "fixture-a").get_history(feature_id="knee_angle_contact", last_n=3)["trend"]
    vals = [p["value"] for p in t["points_oldest_first"]]
    assert vals == [round(feature(c, "knee_angle_contact"), 2) for c in ("c09", "c17", "c07")]
    assert t["change_first_to_last"] == pytest.approx(vals[-1] - vals[0], abs=0.01)
    assert t["step_changes"] == [round(b - a, 2) for a, b in zip(vals[:-1], vals[1:], strict=True)]
    err = typical_errors()["knee_angle_contact"]["value"]
    assert t["change_exceeds_typical_error"] == (abs(t["change_first_to_last"]) > err)


def test_a_feature_never_measured_has_no_trend(con):
    t = Tools(con, "fixture-a").get_history(feature_id="peak_knee_ext_velocity")["trend"]
    assert t["measured_in"] == 0 and "change_first_to_last" not in t
    assert all("not_measured" in p for p in t["points_oldest_first"])


def test_refused_clip_reports_why_and_no_values(con):
    tools = Tools(con, "fixture-b")
    s = tools.get_session_features()
    assert not s["accepted"] and "side-on" in s["refused_because"][0] and "features" not in s
    t = tools.get_history(feature_id="peak_knee_ext_velocity")["trend"]
    assert t["points_oldest_first"][-1] == {**t["points_oldest_first"][-1],
                                            "not_measured": "clip refused"}  # fmt: skip
    assert t["typical_error"]["value"] is None and t["change_exceeds_typical_error"] is None


def test_the_model_cannot_read_another_players_session(con):
    other = Tools(con, "fixture-b").get_history()["sessions"][0]["session_id"]
    assert "error" in Tools(con, "fixture-a").call("get_session_features", {"session_id": other})


def test_bad_calls_come_back_as_errors_not_exceptions(con):
    tools = Tools(con, "fixture-a")
    assert "error" in tools.call("no_such_tool", {})
    assert "error" in tools.call("get_history", {"feature_id": "ball_speed"})
    assert "error" in tools.call("compare_to_norms", {})  # missing feature_id
    assert "error" in tools.call("search_drills", {"query": "  "})
    assert "error" in Tools(con, "nobody").call("get_session_features", {})


def test_compare_to_norms_carries_the_fault_rule_and_its_source(con):
    r = Tools(con, "fixture-a").compare_to_norms("trunk_lean_contact")
    assert r["comparison"]["verdict"] == "no_comparable_norm"
    assert r["fault_rule"]["threshold_source"].startswith("heuristic:")
    r = Tools(con, "fixture-a").compare_to_norms("peak_knee_ext_velocity")
    assert r["status"] == "not_measured" and r["reason"]


def test_search_drills_filters_by_fault_when_given(con):
    tools = Tools(con, "fixture-a", embedder=FakeEmbedder())
    r = tools.search_drills("keep your chest over the ball", fault_id="F-LEAN-BACK")
    assert len(r["passages"]) == 3
    assert all("F-LEAN-BACK" in p["faults"] for p in r["passages"])
    assert "filter" in r["method"]
    assert "filter" not in tools.search_drills("anything")["method"]


def test_declarations_match_the_tools():
    for d in DECLARATIONS:
        assert callable(getattr(Tools, d["name"]))
        assert d["parameters_json_schema"]["type"] == "object"


# -- the answer checker ---------------------------------------------------------------------
def results(con, player="fixture-a"):
    t = Tools(con, player)
    return [t.get_session_features(), t.get_history(feature_id="knee_angle_contact", last_n=3)]


def test_numbers_must_come_from_tool_results(con):
    g = grounding(results(con), "over my last 3 sessions")
    trunk = abs(round(feature("c07", "trunk_lean_contact")))
    assert check(f"Your trunk leaned back {trunk} degrees on 2026-10-25.", g) == []
    assert any("27" in e for e in check("Your trunk leaned back 27 degrees.", g))


def test_ids_must_have_been_returned(con):
    g = grounding(results(con))
    assert any("K-12" in e for e in check("Keep your chest over the ball [K-12].", g))


def test_a_never_measured_feature_needs_a_not_measured_phrase(con):
    g = grounding(results(con))
    assert check("Your extension speed was not measured: the frame rate is unknown.", g) == []
    assert any("not measured" in e for e in check("Your extension speed looked quick.", g))


def test_knee_angle_called_bend_is_rejected_in_chat(con):
    g = grounding(results(con))
    knee = round(feature("c07", "max_knee_flexion_backswing"))
    assert any("interior angle" in e for e in check(f"Your knee bend was {knee} degrees.", g))


def test_empty_answer_is_rejected():
    assert check("  ", grounding([])) == ["the answer is empty"]


# -- the agent loop -------------------------------------------------------------------------
class FakeClient:
    """Plays back scripted turns: a list of function-call lists or answer strings."""

    def __init__(self, turns):
        self.turns, self.requests = list(turns), []

    def generate_turn(self, *, contents, tool_mode=None, **kw):
        self.requests.append({"contents": json.loads(json.dumps(contents)), "mode": tool_mode})
        t = self.turns.pop(0)
        if isinstance(t, Exception):
            raise t
        if isinstance(t, str):
            return {"content": {"role": "model", "parts": [{"text": t}]}, "text": t,
                    "function_calls": []}  # fmt: skip
        parts = [{"function_call": c, "thought_signature": "c2ln"} for c in t]
        return {"content": {"role": "model", "parts": parts}, "text": "", "function_calls": t}


TRUNK = {"name": "get_session_features", "args": {}}


def test_tool_call_then_a_checked_answer(con):
    trunk = abs(round(feature("c07", "trunk_lean_contact")))
    c = FakeClient([[TRUNK], f"You leaned back {trunk} degrees."])
    out = agent.ask("What was my trunk lean?", Tools(con, "fixture-a"), c)
    assert out["generated_by"] == "llm" and out["validator_passed"] and out["model_calls"] == 2
    # the model's turn goes back unchanged (thought signature included), then the result
    sent = c.requests[1]["contents"]
    assert sent[1]["parts"][0]["thought_signature"] == "c2ln"
    assert sent[2]["parts"][0]["function_response"]["response"]["output"]["session_id"]


def test_a_rejected_answer_is_retried_with_the_errors(con):
    trunk = abs(round(feature("c07", "trunk_lean_contact")))
    c = FakeClient([[TRUNK], "You leaned back 27 degrees.", f"You leaned back {trunk} degrees."])
    out = agent.ask("What was my trunk lean?", Tools(con, "fixture-a"), c)
    assert out["generated_by"] == "llm" and len(out["attempts"]) == 2
    assert "27" in c.requests[2]["contents"][-1]["parts"][0]["text"]


def test_two_rejections_give_the_deterministic_answer(con):
    c = FakeClient([[TRUNK], "27 degrees.", "28 degrees."])
    out = agent.ask("What was my trunk lean?", Tools(con, "fixture-a"), c)
    assert out["generated_by"] == "template" and out["validator_passed"] is False
    assert "trunk_lean_contact" in out["answer"] and "27" not in out["answer"]


def test_no_key_gives_the_latest_session_as_a_template(con):
    out = agent.ask("Anything?", Tools(con, "fixture-a"), FakeClient([NoApiKey("no key")]))
    assert out["fallback_reason"] == agent.NO_KEY and "Session" in out["answer"]


def test_tool_rounds_are_capped_then_an_answer_is_forced(con):
    c = FakeClient([[TRUNK]] * agent.MAX_TOOL_ROUNDS + ["Done."])
    agent.ask("Loop?", Tools(con, "fixture-a"), c)
    assert [r["mode"] for r in c.requests] == [None] * agent.MAX_TOOL_ROUNDS + ["NONE"]


# -- the turn cache -------------------------------------------------------------------------
def test_turn_cache_ignores_thought_signatures(tmp_path):
    from google.genai import types

    calls = []

    class Models:
        def generate_content(self, **kw):
            calls.append(kw)
            content = types.Content(
                role="model", parts=[types.Part(text="hi", thought_signature=b"new")]
            )
            cand = types.Candidate(content=content, finish_reason="STOP")
            return types.GenerateContentResponse(candidates=[cand])

    c = GeminiClient("m", tmp_path, api_key="k", min_interval_s=0.0, rate_file=tmp_path / "r")
    c._client = SimpleNamespace(models=Models())
    turn = {"role": "model", "parts": [{"function_call": {"name": "f", "args": {}}}]}
    base = [{"role": "user", "parts": [{"text": "q"}]}]
    a = c.generate_turn(system="s", contents=[*base, {**turn, "parts": [
        {**turn["parts"][0], "thought_signature": "AAA="}]}], tools=[])  # fmt: skip
    b = c.generate_turn(system="s", contents=[*base, {**turn, "parts": [
        {**turn["parts"][0], "thought_signature": "BBB="}]}], tools=[])  # fmt: skip
    assert a["key"] == b["key"] and b["cached"] and len(calls) == 1
    assert a["text"] == "hi" and a["content"]["parts"][0]["thought_signature"]
    # the signature reached the SDK as bytes
    assert calls[0]["contents"][1].parts[0].thought_signature == b"\x00\x00"


# -- E9 scoring helpers ---------------------------------------------------------------------
def test_refusal_needs_the_phrase_and_no_velocity_number():
    fid = "peak_knee_ext_velocity"
    assert refused("Your knee speed wasn't measured: the frame rate is unknown.", fid)
    assert not refused("It was not measured, but roughly 900 deg/s.", fid)
    assert not refused("Your knee straightened quickly.", fid)


def test_tool_choice_scoring():
    q = {"required": [["get_history"]], "args": {"get_history": {"feature_id": "x"}}}
    assert tool_choice(q, [{"name": "get_history", "args": {"feature_id": "x"}}]) == (True, True)
    assert tool_choice(q, [{"name": "get_history", "args": {}}]) == (True, False)
    assert tool_choice({"no_tools": True}, [{"name": "get_history", "args": {}}]) == (False, False)


def test_a_number_written_in_the_wrong_unit_is_rejected(con):
    # E9 run 1, q03: "your plant foot offset went from -0.02 m to -0.32 m" (leg lengths)
    g = grounding(results(con))
    plant = feature("c07", "plant_foot_offset")
    knee = round(feature("c07", "max_knee_flexion_backswing"))
    assert any("length isn't" in e for e in check(f"Your plant foot was at {plant} m.", g))
    assert check(f"Your plant foot was {plant} leg lengths from the ball.", g) == []
    assert check(f"Your backswing knee angle was {knee}°, or {knee} degrees.", g) == []
    assert any("deg/s" in e for e in check(f"It reached {knee} degrees per second.", g))


def test_history_values_and_fault_thresholds_carry_units(con):
    # E9 run 1, q02: "the 90 degree threshold" (89.61 rounded) was rejected because the
    # threshold came without a unit, so it got the strict default tolerance
    t = Tools(con, "fixture-a")
    h, s = t.get_history(), t.get_session_features()
    assert h["units"]["plant_foot_offset"] == "leg_lengths"
    shallow = next(f for f in s["faults"] if f["fault_id"] == "F-SHALLOW-BACKSWING")
    assert shallow["unit"] == "deg"
    g = grounding([h, s])
    thr = round(shallow["threshold"])
    assert check(f"That is above the {thr} degree threshold [F-SHALLOW-BACKSWING].", g) == []
    assert any("F-MADE-UP" in e for e in check("See [F-MADE-UP].", g))


def test_feature_units_match_the_pipeline():
    from pathlib import Path

    from kickcoach.feedback.generate import FEATURE_UNITS

    for p in sorted(Path("results/pipeline").glob("c*.json")):
        for f in json.loads(p.read_text())["features"]:
            assert FEATURE_UNITS[f["id"]] == f["unit"], (p.name, f["id"])


def test_history_says_why_a_value_is_missing(con):
    # E9: with a bare "not measured", the model invented "insufficient markers on shank"
    v = Tools(con, "fixture-a").get_history()["sessions"][0]["values"]["peak_knee_ext_velocity"]
    assert v.startswith("not measured: ") and "frame rate" in v
