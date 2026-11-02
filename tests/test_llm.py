import json

import pytest

from kickcoach.llm import DailyQuotaExhausted, GeminiClient, NoApiKey, RateLimited


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


class _FakeModels:
    def __init__(self) -> None:
        self.configs = []

    def generate_content(self, model, contents, config):
        self.configs.append(config)

        class _Resp:
            text = '{"a": 1}'

            def model_dump(self, **_):
                return {"usage_metadata": {"prompt_token_count": 3}}

        return _Resp()


def _fake_sdk_client(tmp_path):
    from types import SimpleNamespace

    c = _client(tmp_path, api_key="k", min_interval_s=0.0)
    fake = _FakeModels()
    c._client = SimpleNamespace(models=fake)
    return c, fake


def test_media_resolution_and_thinking_reach_the_request(tmp_path):
    c, fake = _fake_sdk_client(tmp_path)
    out = c.generate_json(
        prompt="p",
        schema={},
        temperature=None,
        media_resolution="MEDIA_RESOLUTION_HIGH",
        thinking_level="HIGH",
    )
    cfg = fake.configs[0]
    assert cfg.temperature is None
    assert cfg.media_resolution.value == "MEDIA_RESOLUTION_HIGH"
    assert cfg.thinking_config.thinking_level.value == "HIGH"
    assert out["parsed"] == {"a": 1} and not out["cached"]
    assert c.cached(
        prompt="p",
        schema={},
        temperature=None,
        media_resolution="MEDIA_RESOLUTION_HIGH",
        thinking_level="HIGH",
    )["parsed"] == {"a": 1}


def test_new_options_leave_old_cache_keys_alone(tmp_path):
    # Responses cached before the options existed (the Phase 1 gut-check) must still be found.
    c, fake = _fake_sdk_client(tmp_path)
    old = {"prompt": "p", "schema": {}, "temperature": 0.0, "video_sha256": None, "video_fps": None}
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / f"{c.cache_key(old)}.json").write_text(json.dumps({"parsed": {"b": 2}}))
    assert c.generate_json(prompt="p", schema={})["parsed"] == {"b": 2}
    assert c.generate_json(prompt="p", schema={}, repeat=1)["parsed"] == {"a": 1}  # new key
    assert len(fake.configs) == 1


def _raising_client(tmp_path, message: str):
    from types import SimpleNamespace

    from google.genai import errors

    class _Models:
        def generate_content(self, **_):
            raise errors.ClientError(429, {"error": {"code": 429, "message": message}})

    slept = []
    c = _client(tmp_path, api_key="k", min_interval_s=0.0, sleep=slept.append)
    c._client = SimpleNamespace(models=_Models())
    return c, slept


def test_daily_quota_stops_at_once(tmp_path):
    c, slept = _raising_client(tmp_path, "quotaId: GenerateRequestsPerDayPerProjectPerModel")
    with pytest.raises(DailyQuotaExhausted):
        c.generate_json(prompt="p", schema={})
    assert slept == []  # no point waiting minutes for a quota that resets at midnight


def test_per_minute_429_is_retried_then_reported(tmp_path):
    c, slept = _raising_client(tmp_path, "quotaId: GenerateRequestsPerMinutePerProjectPerModel")
    with pytest.raises(RateLimited) as e:
        c.generate_json(prompt="p", schema={})
    assert not isinstance(e.value, DailyQuotaExhausted)
    assert len(slept) == 3
