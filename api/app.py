"""KickCoach API (handoff S9, S13): upload one kick, poll the analysis, get the MVP response.

    uv run uvicorn api.app:app --port 8000      # then open http://127.0.0.1:8000/

Runs without GEMINI_API_KEY: feedback then comes from templates (CLAUDE.md rule 5).
Uploaded videos and their frames are deleted after analysis.

Two modes (DEC-034):
- local (default): each session is stored in SQLite with its pose array and ball
  candidates (data/kickcoach.db, data/sessions/<id>/; gitignored), and LLM feedback is used
  when a key is set.
- public (KICKCOACH_PUBLIC=1, the hosted demo): nothing is stored on the server, the page
  keeps the visitor's history in their own browser, and feedback is always the template:
  Gemini's unpaid tier may use inputs to improve Google's products and may not serve users
  in the EEA, Switzerland or the UK (Gemini API terms, effective 23 Mar 2026).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from api.jobs import Busy, Job, JobError, Jobs
from kickcoach import session
from kickcoach.storage import db

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
SESSIONS_DIR = ROOT / "data" / "sessions"
# heuristic: a phone clip of one kick is well under this; it bounds disk use per request.
MAX_UPLOAD_MB = 100
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv"}
# heuristic: more past sessions than the page could usefully send; bounds the request.
MAX_HISTORY = 50
BUSY = "the demo is busy with other kicks; try again in a few minutes"
CHAT_CACHE = ROOT / "data" / "cache" / "chat"
# heuristic: a chat question is a sentence or two; earlier turns are context, not a transcript.
MAX_CHAT_CHARS, MAX_CHAT_TURNS = 500, 10

Analyser = Callable[..., tuple[dict, dict[str, str]]]


def public_mode() -> bool:
    return os.environ.get("KICKCOACH_PUBLIC", "").lower() in {"1", "true", "yes"}


def measured(result: dict) -> dict[str, float]:
    return {
        f["id"]: f["value"]
        for f in result.get("features", [])
        if f["status"] == "measured" and isinstance(f.get("value"), int | float)
    }


def parse_history(raw: str | None) -> list[dict[str, float]] | None:
    """The page's own earlier sessions, newest first: [{feature_id: value}, ...]."""
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(422, "history must be JSON") from e
    if not isinstance(data, list) or len(data) > MAX_HISTORY:
        raise HTTPException(422, f"history must be a list of at most {MAX_HISTORY} sessions")
    out = []
    for s in data:
        if not isinstance(s, dict):
            raise HTTPException(422, "each history entry must be {feature_id: value}")
        out.append(
            {
                str(k)[:64]: float(v)
                for k, v in s.items()
                if isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)
            }
        )
    return out


class ChatTurn(BaseModel):
    role: Literal["user", "model"]
    text: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_CHAT_CHARS)
    player: str = Field("me", max_length=64)
    level: Literal["", "beginner", "intermediate", "advanced"] = ""
    history: list[ChatTurn] = Field(default_factory=list, max_length=MAX_CHAT_TURNS)


def create_app(
    db_path: Path = db.DB_PATH,
    sessions_dir: Path = SESSIONS_DIR,
    analyser: Analyser = session.analyse_video,
    public: bool | None = None,
    chat_client: Callable[[], object] | None = None,
) -> FastAPI:
    public = public_mode() if public is None else public
    app = FastAPI(title="KickCoach", version="0.3")
    jobs = Jobs()

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/example")
    def example() -> FileResponse:
        return FileResponse(STATIC / "example.json")

    @app.get("/health")
    def health() -> dict:
        from kickcoach.llm import GeminiClient, NoApiKey
        from kickcoach.llm.gemini import DEV_MODEL

        if public:
            llm = "off on the public demo: template feedback"
        else:
            try:
                GeminiClient(DEV_MODEL, Path(tempfile.gettempdir()))._sdk()
                llm = "available"
            except NoApiKey:
                llm = "no API key: template feedback"
        return {
            "status": "ok",
            "public": public,
            "feedback_llm": llm,
            "max_upload_mb": MAX_UPLOAD_MB,
            "max_frames": session.MAX_FRAMES,
            "waiting": jobs.waiting(),
        }

    def respond(result: dict, sid: int | None, delta: dict) -> dict:
        return {
            "session_id": sid,
            **{k: v for k, v in result.items() if k != "pose_model"},
            "pose_model": result.get("pose_model"),
            "history_delta": delta,
        }

    def finish(result: dict, *, player, level, clip_name, inputs, work, earlier) -> dict:
        if public:
            return respond(result, None, db.delta_vs(measured(result), earlier or []))
        con = db.connect(db_path)
        try:
            sid = db.save_session(
                con, player=player, level=level or None, clip_name=clip_name,
                result=result, inputs=inputs,
            )  # fmt: skip
            delta = (
                db.history_delta(con, player, sid)
                if earlier is None
                else db.delta_vs(measured(result), earlier)
            )
            response = respond(result, sid, delta)
            keep = sessions_dir / str(sid)
            if work.exists():
                shutil.copytree(work, keep, dirs_exist_ok=True)
            keep.mkdir(parents=True, exist_ok=True)
            (keep / "response.json").write_text(json.dumps(response, indent=1))
            with con:
                con.executemany(
                    "INSERT INTO artifacts VALUES (?, ?, ?)",
                    [(sid, p.stem, str(p)) for p in keep.iterdir()],
                )
        finally:
            con.close()
        return response

    # Sync endpoint: FastAPI runs it in a worker thread, so writing the upload to disk
    # doesn't block the event loop. The analysis itself runs on the job thread.
    @app.post("/analyse", status_code=202)
    def analyse_upload(
        file: Annotated[UploadFile, File()],
        player: Annotated[str, Form(max_length=64)] = "me",
        level: Annotated[Literal["", "beginner", "intermediate", "advanced"], Form()] = "",
        kicking_leg: Annotated[Literal["auto", "left", "right"], Form()] = "auto",
        capture_fps: Annotated[float | None, Form(gt=0)] = None,
        slow_motion: Annotated[Literal["unknown", "real_time", "slow_motion"], Form()] = "unknown",
        use_llm: Annotated[bool, Form()] = True,
        history: Annotated[str | None, Form()] = None,
    ) -> dict:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            raise HTTPException(415, f"expected a video file ({', '.join(sorted(VIDEO_SUFFIXES))})")
        earlier = parse_history(history)
        if jobs.waiting() >= jobs.max_waiting:
            raise HTTPException(503, BUSY)
        tmp = Path(tempfile.mkdtemp(prefix="kickcoach_upload_"))
        video = tmp / f"upload{suffix}"
        try:
            size = 0
            with open(video, "wb") as out:
                while chunk := file.file.read(1 << 20):
                    size += len(chunk)
                    if size > MAX_UPLOAD_MB << 20:
                        raise HTTPException(413, f"the video is over {MAX_UPLOAD_MB} MB")
                    out.write(chunk)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        llm = use_llm and not public
        inputs = {
            "kicking_leg": kicking_leg, "capture_fps": capture_fps,
            "slow_motion": slow_motion, "use_llm": llm,
        }  # fmt: skip

        def run(job: Job) -> dict:
            try:
                work = tmp / "keep"
                try:
                    result, _ = analyser(
                        video,
                        kicking_leg=None if kicking_leg == "auto" else kicking_leg,
                        capture_fps=capture_fps,
                        slow_motion=slow_motion,
                        level=level or None,
                        use_llm=llm,
                        keep_dir=None if public else work,
                        progress=job.progress,
                    )
                except ValueError as e:
                    raise JobError(422, str(e)) from e
                return finish(
                    result, player=player, level=level, clip_name=file.filename,
                    inputs=inputs, work=work, earlier=earlier,
                )  # fmt: skip
            finally:
                shutil.rmtree(tmp, ignore_errors=True)  # the upload and its frames

        try:
            job = jobs.submit(run)
        except Busy:
            shutil.rmtree(tmp, ignore_errors=True)
            raise HTTPException(503, BUSY) from None
        return jobs.status(job)

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job (results are held for 30 minutes)")
        return jobs.status(job)

    @app.get("/sessions")
    def sessions(player: str | None = None, limit: int = 30) -> dict:
        if public:
            raise HTTPException(404, "the public demo keeps no sessions on the server")
        con = db.connect(db_path)
        try:
            return {"sessions": db.list_sessions(con, player, min(limit, 200))}
        finally:
            con.close()

    @app.get("/sessions/{session_id}")
    def session_detail(session_id: int) -> dict:
        path = sessions_dir / str(session_id) / "response.json"
        if public or not path.exists():
            raise HTTPException(404, f"no session {session_id}")
        return json.loads(path.read_text())

    # The coach chat (Phase 9, DEC-043). Local mode only: it needs the LLM, which the public
    # demo can't use (DEC-034). Without a key it answers with a template.
    @app.post("/chat")
    def chat(req: ChatRequest) -> dict:
        if public:
            raise HTTPException(
                403, "the coach chat needs the LLM, which the public demo doesn't use (DEC-034)"
            )
        from kickcoach.chat.agent import ask
        from kickcoach.chat.tools import Tools
        from kickcoach.llm import GeminiClient
        from kickcoach.llm.gemini import DEV_MODEL

        client = chat_client() if chat_client else GeminiClient(DEV_MODEL, CHAT_CACHE)
        con = db.connect(db_path)
        try:
            out = ask(req.message, Tools(con, req.player, level=req.level or None), client,
                      history=[t.model_dump() for t in req.history])  # fmt: skip
        finally:
            con.close()
        return {
            "answer": out["answer"],
            "generated_by": out["generated_by"],
            "validator_passed": out["validator_passed"],
            "fallback_reason": out["fallback_reason"],
            "tool_calls": [{"name": c["name"], "args": c["args"]} for c in out["tool_calls"]],
            "model_calls": out["model_calls"],
        }

    return app


app = create_app()
