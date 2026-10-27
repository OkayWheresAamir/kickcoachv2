import json

import pytest

from kickcoach.llm import GeminiClient, NoApiKey


def _client(tmp_path, **kw):
    return GeminiClient("m", tmp_path / "cache", rate_file=tmp_path / "rate.txt", **kw)


def test_no_key_raises_so_callers_can_fall_back(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    with pytest.raises(NoApiKey):
        _client(tmp_path, api_key="").generate_json(prompt="p", schema={})


def test_cached_response_is_returned_without_calling_the_api(tmp_path):
    c = _client(tmp_path, api_key="")  # an empty key would raise if the API were reached
    req = {"prompt": "p", "schema": {}, "temperature": 0.0, "video_sha256": None, "video_fps": None}
    key = c.cache_key(req)
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / f"{key}.json").write_text(json.dumps({"key": key, "parsed": {"a": 1}}))
    out = c.generate_json(prompt="p", schema={})
    assert out["cached"] and out["parsed"] == {"a": 1}


def test_cache_key_changes_with_anything_that_shapes_the_answer(tmp_path):
    c = _client(tmp_path)
    base = {"prompt": "p", "schema": {}, "temperature": 0.0, "video_sha256": "v", "video_fps": 24}
    keys = {c.cache_key(base)}
    for k, v in [("prompt", "q"), ("temperature", 1.0), ("video_sha256", "w"), ("video_fps", 1)]:
        keys.add(c.cache_key({**base, k: v}))
    keys.add(GeminiClient("other", tmp_path).cache_key(base))
    assert len(keys) == 6


def test_rate_limit_waits_out_the_interval_across_instances(tmp_path):
    now = [1000.0]
    slept = []
    kw = dict(min_interval_s=12.0, clock=lambda: now[0], sleep=slept.append)
    _client(tmp_path, **kw)._wait_turn()
    now[0] += 5.0
    _client(tmp_path, **kw)._wait_turn()  # a second process sharing the rate file
    assert slept == [pytest.approx(7.0)]
