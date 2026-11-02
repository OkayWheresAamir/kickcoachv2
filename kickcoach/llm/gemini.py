"""Gemini calls through one door: cache first, then a cross-process rate limit, then the API.

Only a free-tier key exists (DEC-003), and Google no longer publishes free-tier limits in the
docs (they point to AI Studio), so the spacing between calls is a conservative heuristic and a
429 backs off and retries. Responses are cached by a hash of everything that shapes them, so
committed caches let a fresh clone rerun an evaluation without any calls.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RATE_FILE = ROOT / "data" / "cache" / "gemini_last_call.txt"
# heuristic: 12 s between calls (5 per minute), below any free-tier RPM seen for Flash models.
MIN_INTERVAL_S = 12.0
RETRY_WAITS_S = (30.0, 60.0, 120.0)
# heuristic: a video call with high thinking took 6-8 min on 2026-10-31, and one hung for
# over 30 min with the SDK's default of no timeout; a timeout is retried like a 503.
TIMEOUT_MS = 15 * 60 * 1000

DEV_MODEL = "gemini-3.5-flash-lite"  # cheapest free-tier model (pricing page, 2026-10-28)
E4_MODEL = "gemini-3.8-flash"  # strongest free-tier model; 4-argon and 3.1-pro are paid-only


class NoApiKey(RuntimeError):
    """No GEMINI_API_KEY: callers fall back to templates (the app must run without a key)."""


class RateLimited(RuntimeError):
    pass


class DailyQuotaExhausted(RateLimited):
    """The free tier's requests-per-day quota for this model is used up (20 for
    gemini-3.8-flash on 2026-10-31, from the 429's QuotaFailure). Retrying before it resets
    at midnight Pacific is pointless, so the caller should stop."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class GeminiClient:
    def __init__(
        self,
        model: str,
        cache_dir: Path,
        *,
        api_key: str | None = None,
        min_interval_s: float = MIN_INTERVAL_S,
        rate_file: Path = RATE_FILE,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.cache_dir = Path(cache_dir)
        self._api_key = api_key
        self.min_interval_s = min_interval_s
        self.rate_file = rate_file
        self._clock, self._sleep = clock, sleep
        self._client = None

    # -- cache -----------------------------------------------------------------------------
    def cache_key(self, request: dict) -> str:
        return _sha256(json.dumps({"model": self.model, **request}, sort_keys=True).encode())[:24]

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    # -- rate limit ------------------------------------------------------------------------
    def _wait_turn(self) -> None:
        try:
            last = float(self.rate_file.read_text())
        except (FileNotFoundError, ValueError):
            last = 0.0
        wait = last + self.min_interval_s - self._clock()
        if wait > 0:
            self._sleep(wait)
        self.rate_file.parent.mkdir(parents=True, exist_ok=True)
        self.rate_file.write_text(str(self._clock()))

    # -- API -------------------------------------------------------------------------------
    def _sdk(self):
        if self._client is None:
            key = self._api_key
            if key is None:
                from dotenv import load_dotenv

                load_dotenv(ROOT / ".env")
                key = os.environ.get("GEMINI_API_KEY") or None
            if not key:
                raise NoApiKey("GEMINI_API_KEY is not set")
            from google import genai
            from google.genai import types

            self._client = genai.Client(
                api_key=key, http_options=types.HttpOptions(timeout=TIMEOUT_MS)
            )
        return self._client

    @staticmethod
    def _request(
        *,
        prompt: str,
        schema: dict,
        video: Path | None = None,
        video_fps: float | None = None,
        temperature: float | None = 0.0,
        media_resolution: str | None = None,
        thinking_level: str | None = None,
        repeat: int = 0,
    ) -> tuple[dict, bytes | None]:
        video_bytes = Path(video).read_bytes() if video else None
        request = {
            "prompt": prompt,
            "schema": schema,
            "temperature": temperature,
            "video_sha256": _sha256(video_bytes) if video_bytes else None,
            "video_fps": video_fps,
        }
        # Added only when set, so cache keys made before these options existed stay valid.
        for k, v in (
            ("media_resolution", media_resolution),
            ("thinking_level", thinking_level),
            ("repeat", repeat or None),
        ):
            if v is not None:
                request[k] = v
        return request, video_bytes

    def cached(self, **request_args) -> dict | None:
        """The cached record for a request (generate_json's arguments, minus ``tag``), or None.
        Never calls the API, so scoring runs without a key."""
        request, _ = self._request(**request_args)
        path = self._cache_path(self.cache_key(request))
        return json.loads(path.read_text()) if path.exists() else None

    def generate_json(self, *, tag: str = "", **request_args) -> dict:
        """Return {"parsed", "text", "response", "cached", "key"}; "parsed" is None if the
        model's text is not valid JSON (that is a reportable outcome, not an exception).

        Arguments as ``_request``. ``temperature=None`` leaves it out: Gemini 3.8 Flash
        ignores sampling parameters. ``media_resolution`` / ``thinking_level`` are SDK enum
        names ("MEDIA_RESOLUTION_HIGH", "HIGH"). ``repeat`` > 0 asks the same question again
        under a new cache key, to see how much the answer varies from call to call."""
        request, video_bytes = self._request(**request_args)
        temperature = request["temperature"]
        video_fps = request["video_fps"]
        media_resolution = request.get("media_resolution")
        thinking_level = request.get("thinking_level")
        key = self.cache_key(request)
        path = self._cache_path(key)
        if path.exists():
            return {**json.loads(path.read_text()), "cached": True}

        import httpx
        from google.genai import errors, types

        parts = []
        if video_bytes is not None:
            parts.append(
                types.Part(
                    inline_data=types.Blob(data=video_bytes, mime_type="video/mp4"),
                    video_metadata=types.VideoMetadata(fps=video_fps) if video_fps else None,
                )
            )
        parts.append(types.Part(text=request["prompt"]))
        config = types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
            response_json_schema=request["schema"],
            media_resolution=media_resolution,
            thinking_config=(
                types.ThinkingConfig(thinking_level=thinking_level) if thinking_level else None
            ),
        )
        sdk = self._sdk()
        for attempt, backoff in enumerate((*RETRY_WAITS_S, None)):
            self._wait_turn()
            try:
                resp = sdk.models.generate_content(
                    model=self.model,
                    contents=[types.Content(role="user", parts=parts)],
                    config=config,
                )
                break
            except (errors.APIError, httpx.TimeoutException) as e:
                # 429 = rate limit, 503 = "model is experiencing high demand", or no answer
                # within TIMEOUT_MS: all transient.
                code = getattr(e, "code", "timeout")
                if code not in (429, 503, "timeout"):
                    raise
                if code == 429 and "PerDay" in str(e):
                    raise DailyQuotaExhausted(str(e)) from e
                if backoff is None:
                    raise RateLimited(str(e)) from e
                print(
                    f"{code} from Gemini (attempt {attempt + 1}); waiting {backoff:.0f} s",
                    flush=True,
                )
                self._sleep(backoff)

        text = resp.text or ""
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        record = {
            "key": key,
            "tag": tag,
            "model": self.model,
            "request": {k: v for k, v in request.items() if k != "prompt"}
            | {"prompt": request["prompt"]},
            "text": text,
            "parsed": parsed,
            "response": resp.model_dump(mode="json", exclude_none=True),
        }
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1))
        return {**record, "cached": False}
