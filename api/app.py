"""KickCoach API (handoff S9): upload one kick, get the MVP response; list past sessions.

    uv run uvicorn api.app:app --port 8000      # then open http://127.0.0.1:8000/

Runs without GEMINI_API_KEY: feedback then comes from templates (CLAUDE.md rule 5).
Uploaded videos and their frames are deleted after analysis; only the pose array, the ball
candidates and the response are kept, under data/sessions/<id>/ (gitignored).
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from kickcoach import session
from kickcoach.storage import db

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
SESSIONS_DIR = ROOT / "data" / "sessions"
# heuristic: a phone clip of one kick is well under this; it bounds disk use per request.
MAX_UPLOAD_MB = 100
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv"}

Analyser = Callable[..., tuple[dict, dict[str, str]]]


def create_app(
    db_path: Path = db.DB_PATH,
    sessions_dir: Path = SESSIONS_DIR,
    analyser: Analyser = session.analyse_video,
) -> FastAPI:
    app = FastAPI(title="KickCoach", version="0.2")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health() -> dict:
        from kickcoach.llm import GeminiClient, NoApiKey
        from kickcoach.llm.gemini import DEV_MODEL

        try:
            GeminiClient(DEV_MODEL, Path(tempfile.gettempdir()))._sdk()
            llm = "available"
        except NoApiKey:
            llm = "no API key: template feedback"
        return {"status": "ok", "feedback_llm": llm}

    # Sync endpoint: FastAPI runs it in a worker thread, so the CPU-bound analysis doesn't
    # block the event loop.
    @app.post("/analyse")
    def analyse_upload(
        file: Annotated[UploadFile, File()],
        player: Annotated[str, Form()] = "me",
        level: Annotated[Literal["", "beginner", "intermediate", "advanced"], Form()] = "",
        kicking_leg: Annotated[Literal["auto", "left", "right"], Form()] = "auto",
        capture_fps: Annotated[float | None, Form(gt=0)] = None,
        slow_motion: Annotated[Literal["unknown", "real_time", "slow_motion"], Form()] = "unknown",
        use_llm: Annotated[bool, Form()] = True,
    ) -> dict:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            raise HTTPException(415, f"expected a video file ({', '.join(sorted(VIDEO_SUFFIXES))})")
        tmp = Path(tempfile.mkdtemp(prefix="kickcoach_upload_"))
        try:
            video = tmp / f"upload{suffix}"
            size = 0
            with open(video, "wb") as out:
                while chunk := file.file.read(1 << 20):
                    size += len(chunk)
                    if size > MAX_UPLOAD_MB << 20:
                        raise HTTPException(413, f"the video is over {MAX_UPLOAD_MB} MB")
                    out.write(chunk)
            work = tmp / "keep"
            try:
                result, _ = analyser(
                    video,
                    kicking_leg=None if kicking_leg == "auto" else kicking_leg,
                    capture_fps=capture_fps,
                    slow_motion=slow_motion,
                    level=level or None,
                    use_llm=use_llm,
                    keep_dir=work,
                )
            except ValueError as e:
                raise HTTPException(422, str(e)) from e
            inputs = {
                "kicking_leg": kicking_leg, "capture_fps": capture_fps,
                "slow_motion": slow_motion, "use_llm": use_llm,
            }  # fmt: skip
            con = db.connect(db_path)
            try:
                sid = db.save_session(
                    con, player=player, level=level or None, clip_name=file.filename,
                    result=result, inputs=inputs,
                )  # fmt: skip
                keep = sessions_dir / str(sid)
                if work.exists():
                    shutil.copytree(work, keep, dirs_exist_ok=True)
                response = {
                    "session_id": sid,
                    **{k: v for k, v in result.items() if k != "pose_model"},
                    "pose_model": result.get("pose_model"),
                    "history_delta": db.history_delta(con, player, sid),
                }
                keep.mkdir(parents=True, exist_ok=True)
                (keep / "response.json").write_text(json.dumps(response, indent=1))
                artifacts = {p.stem: str(p) for p in keep.iterdir()}
                with con:
                    con.executemany(
                        "INSERT INTO artifacts VALUES (?, ?, ?)",
                        [(sid, kind, path) for kind, path in artifacts.items()],
                    )
            finally:
                con.close()
            return response
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @app.get("/sessions")
    def sessions(player: str | None = None, limit: int = 30) -> dict:
        con = db.connect(db_path)
        try:
            return {"sessions": db.list_sessions(con, player, min(limit, 200))}
        finally:
            con.close()

    @app.get("/sessions/{session_id}")
    def session_detail(session_id: int) -> dict:
        path = sessions_dir / str(session_id) / "response.json"
        if not path.exists():
            raise HTTPException(404, f"no session {session_id}")
        return json.loads(path.read_text())

    return app


app = create_app()
