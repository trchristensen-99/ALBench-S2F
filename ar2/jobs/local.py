"""Local backend: one evaluator subprocess per GPU id (also used on CPU for smoke tests).

State lives only in memory plus ``runs/<run_id>/job.pid``: after an orchestrator crash a resubmitted run finds the pid file, sees the process alive (or metrics.json
present) and does not start a second job.
"""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

from ar2.jobs.base import State, evaluator_command, metrics_exist


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:  # a zombie that nobody has waited for still answers kill 0
        out = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True
        ).stdout.strip()
        return bool(out) and not out.startswith("Z")
    except OSError:
        return True


class LocalBackend:
    def __init__(self, gpus: list[int | None] | None = None, env: dict[str, str] | None = None):
        self.gpus: list[int | None] = list(gpus) if gpus else [None]
        self.env = env or {}
        self._queue: list[tuple[str, Path]] = []
        self._procs: dict[str, tuple[subprocess.Popen, int | None]] = {}
        self._specs: dict[str, Path] = {}
        self._cancelled: set[str] = set()

    # ---- helpers ---------------------------------------------------------------------------------------------------------------
    def _free_gpu(self) -> tuple[bool, int | None]:
        busy = {g for _, g in self._procs.values()}
        for g in self.gpus:
            if g not in busy:
                return True, g
        return False, None

    def _start(self, run_id: str, spec: Path, gpu: int | None) -> None:
        env = {**os.environ, **self.env}
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log = open(spec.parent / "log.txt", "ab")
        p = subprocess.Popen(
            [*evaluator_command(), "--spec", str(spec)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        (spec.parent / "job.pid").write_text(str(p.pid))
        self._procs[run_id] = (p, gpu)

    def _fill(self) -> None:
        while self._queue:
            ok, gpu = self._free_gpu()
            if not ok:
                return
            run_id, spec = self._queue.pop(0)
            if run_id in self._cancelled or metrics_exist(spec):
                continue
            self._start(run_id, spec, gpu)

    # ---- contract --------------------------------------------------------------------------------------------------------------
    def submit(self, run_id: str, spec_path: Path) -> str:
        spec_path = Path(spec_path)
        self._specs[run_id] = spec_path
        self._cancelled.discard(run_id)
        if (
            run_id in self._procs
            or any(r == run_id for r, _ in self._queue)
            or metrics_exist(spec_path)
        ):
            return run_id
        pid_file = spec_path.parent / "job.pid"
        if pid_file.exists():  # an orphan from a crashed orchestrator may still be running
            try:
                if _alive(int(pid_file.read_text())):
                    return run_id
            except ValueError:
                pass
        self._queue.append((run_id, spec_path))
        self._fill()
        return run_id

    def poll(self, job_id: str) -> State:
        spec = self._specs.get(job_id)
        if spec is not None and metrics_exist(spec):
            self._procs.pop(job_id, None)
            self._fill()
            return "done"
        if job_id in self._procs:
            p, _ = self._procs[job_id]
            if p.poll() is None:
                return "running"
            self._procs.pop(job_id)
            self._fill()
            return "done" if spec is not None and metrics_exist(spec) else "failed"
        if any(r == job_id for r, _ in self._queue):
            self._fill()
            return "running" if job_id in self._procs else "pending"
        if spec is not None:
            pid_file = spec.parent / "job.pid"
            if pid_file.exists():
                try:
                    if _alive(int(pid_file.read_text())):
                        return "running"
                except ValueError:
                    pass
        return "failed"

    def cancel(self, job_id: str) -> None:
        self._cancelled.add(job_id)
        self._queue = [(r, s) for r, s in self._queue if r != job_id]
        if job_id in self._procs:
            p, _ = self._procs.pop(job_id)
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except OSError:
                pass

    def wait_all(self, timeout_s: float = 600) -> None:
        """Block until every started and queued job has ended (tests and smoke runs)."""
        import time

        t0 = time.time()
        while (self._procs or self._queue) and time.time() - t0 < timeout_s:
            for r in list(self._procs):
                self.poll(r)
            self._fill()
            time.sleep(0.05)
