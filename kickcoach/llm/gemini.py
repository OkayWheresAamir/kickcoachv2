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

DEV_MODEL = "gemini-3.5-flash-lite"  # cheapest free-tier model (pricing page, 2026-10-28)
E4_MODEL = "gemini-3.8-flash"  # strongest free-tier model; 4-argon and 3.1-pro are paid-only


class NoApiKey(RuntimeError):
    """No GEMINI_API_KEY: callers fall back to templates (the app must run without a key)."""


class RateLimited(RuntimeError):
    pass


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

            self._client = genai.Client(api_key=key)
        return self._client

    def generate_json(
        self,
        *,
        prompt: str,
        schema: dict,
        video: Path | None = None,
        video_fps: float | None = None,
        temperature: float = 0.0,
        tag: str = "",
    ) -> dict:
        """Return {"parsed", "text", "response", "cached", "key"}; "parsed" is None if the
        model's text is not valid JSON (that is a reportable outcome, not an exception)."""
        video_bytes = Path(video).read_bytes() if video else None
        request = {
            "prompt": prompt,
            "schema": schema,
            "temperature": temperature,
            "video_sha256": _sha256(video_bytes) if video_bytes else None,
            "video_fps": video_fps,
        }
        key = self.cache_key(request)
        path = self._cache_path(key)
        if path.exists():
            return {**json.loads(path.read_text()), "cached": True}

        from google.genai import errors, types

        parts = []
        if video_bytes is not None:
            parts.append(
                types.Part(
                    inline_data=types.Blob(data=video_bytes, mime_type="video/mp4"),
                    video_metadata=types.VideoMetadata(fps=video_fps) if video_fps else None,
                )
            )
        parts.append(types.Part(text=prompt))
        config = types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
            response_json_schema=schema,
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
            except errors.APIError as e:
                # 429 = rate limit, 503 = "model is experiencing high demand": both transient.
                if e.code not in (429, 503):
                    raise
                if backoff is None:
                    raise RateLimited(str(e)) from e
                print(f"{e.code} from Gemini (attempt {attempt + 1}); waiting {backoff:.0f} s")
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
            "request": {k: v for k, v in request.items() if k != "prompt"} | {"prompt": prompt},
            "text": text,
            "parsed": parsed,
            "response": resp.model_dump(mode="json", exclude_none=True),
        }
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1))
        return {**record, "cached": False}
