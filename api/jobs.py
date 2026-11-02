"""Analysis jobs: an upload is queued, analysed by one worker thread, and polled by its id.

An analysis takes one to a few minutes, longer than hosts allow one HTTP request to stay
open (Modal: 150 s), so the page polls instead of waiting. One analysis already saturates
the CPU, so jobs run one at a time; two at once would only make both slower and double the
memory.
"""

from __future__ import annotations

import logging
import queue
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# heuristic: bounds a visitor's wait (each clip takes ~1.5-3 min on the demo host).
MAX_WAITING = 4
# heuristic: how long a finished result is held for the page to collect it.
KEEP_S = 30 * 60


class Busy(Exception):
    pass


class JobError(Exception):
    """A failure to show the visitor as it is (e.g. "trim the video to the one kick")."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status, self.detail = status, detail


@dataclass
class Job:
    id: str
    run: Callable[[Job], dict]
    state: str = "queued"  # queued | running | done | failed
    stage: str | None = None
    done: int | None = None
    total: int | None = None
    result: dict | None = None
    error: dict | None = None
    created: float = field(default_factory=time.monotonic)
    started: float | None = None
    finished: float | None = None

    def progress(self, stage: str, done: int | None = None, total: int | None = None) -> None:
        self.stage, self.done, self.total = stage, done, total


class Jobs:
    def __init__(self, max_waiting: int = MAX_WAITING, keep_s: float = KEEP_S) -> None:
        self.max_waiting, self.keep_s = max_waiting, keep_s
        self._jobs: dict[str, Job] = {}
        self._queue: queue.Queue[Job] = queue.Queue()
        self._lock = threading.Lock()
        threading.Thread(target=self._work, daemon=True, name="kickcoach-jobs").start()

    def submit(self, run: Callable[[Job], dict]) -> Job:
        with self._lock:
            self._prune()
            if self.waiting() >= self.max_waiting:
                raise Busy
            job = Job(secrets.token_urlsafe(12), run)
            self._jobs[job.id] = job
        self._queue.put(job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            self._prune()
            return self._jobs.get(job_id)

    def waiting(self) -> int:
        return sum(j.state == "queued" for j in self._jobs.values())

    def ahead_of(self, job: Job) -> int:
        """Jobs that will run before this one (including one running now)."""
        if job.state != "queued":
            return 0
        return sum(
            j.state == "running" or (j.state == "queued" and j.created < job.created)
            for j in list(self._jobs.values())
        )

    def status(self, job: Job) -> dict:
        now = time.monotonic()
        out = {
            "job_id": job.id,
            "state": job.state,
            "stage": job.stage,
            "done": job.done,
            "total": job.total,
            "ahead": self.ahead_of(job),
            "elapsed_s": round(now - job.started, 1) if job.started else 0.0,
        }
        if job.state == "done":
            out["result"] = job.result
        if job.state == "failed":
            out["error"] = job.error
        return out

    def _work(self) -> None:
        while True:
            job = self._queue.get()
            job.state, job.started = "running", time.monotonic()
            try:
                job.result = job.run(job)
                job.state = "done"
            except JobError as e:
                job.error, job.state = {"status": e.status, "detail": e.detail}, "failed"
            except Exception:  # keep the worker alive for the next visitor
                log.exception("job %s failed", job.id)
                job.error = {"status": 500, "detail": "the analysis failed on the server"}
                job.state = "failed"
            finally:
                job.finished = time.monotonic()

    def _prune(self) -> None:
        now = time.monotonic()
        for jid in [
            j.id for j in self._jobs.values() if j.finished and now - j.finished > self.keep_s
        ]:
            del self._jobs[jid]
