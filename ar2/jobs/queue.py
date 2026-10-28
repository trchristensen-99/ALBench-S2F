"""File-queue backend: the orchestrator writes jobs, external workers (``python -m ar2.jobs.worker``) claim and run them.

Layout under ``ws/queue``: ``pending/<id>.json`` (written atomically; idempotent per id), ``running/<id>.json`` (claimed by atomic rename; ``running/<id>.hb`` is touched
every 30 s while the job runs), ``done/<id>.json`` (the worker's receipt), ``gate/<id>.json`` (result of a ``gate_speed`` probe). On a cluster the workers are ordinary
GPU jobs that live for hours and pull many runs, so there is no scheduler latency per run. A job is done when ``metrics.json`` exists, not when the receipt does.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from ar2.jobs.base import State, metrics_exist, read_json, write_json_atomic


class QueueBackend:
    def __init__(self, queue_dir: str | Path, stale_claim_s: float = 600.0):
        self.q = Path(queue_dir)
        for sub in ("pending", "running", "done", "gate"):
            (self.q / sub).mkdir(parents=True, exist_ok=True)
        self.stale_claim_s = stale_claim_s
        self._specs: dict[str, Path] = {}

    # ---- paths -----------------------------------------------------------------------------------------------------------------
    def _p(self, sub: str, job_id: str, ext: str = "json") -> Path:
        return self.q / sub / f"{job_id}.{ext}"

    # ---- training runs ---------------------------------------------------------------------------------------------------------
    def submit(self, run_id: str, spec_path: Path, *, force: bool = False) -> str:
        spec_path = Path(spec_path)
        self._specs[run_id] = spec_path
        if metrics_exist(spec_path):
            return run_id
        if force:
            self._p("done", run_id).unlink(missing_ok=True)
        if (
            self._p("pending", run_id).exists()
            or self._p("running", run_id).exists()
            or self._p("done", run_id).exists()
        ):
            return run_id
        write_json_atomic(
            self._p("pending", run_id),
            {
                "job_id": run_id,
                "kind": "train",
                "spec_path": str(spec_path.resolve()),
                "submitted_at": time.time(),
            },
        )
        return run_id

    def poll(self, job_id: str) -> State:
        spec = self._specs.get(job_id)
        if spec is not None and metrics_exist(spec):
            return "done"
        if self._p("running", job_id).exists():
            return "running"
        if self._p("pending", job_id).exists():
            return "pending"
        rec = read_json(self._p("done", job_id))
        if rec is not None:
            if rec.get("timeout"):
                return "timeout"
            return "done" if spec is not None and metrics_exist(spec) else "failed"
        return "failed"

    def cancel(self, job_id: str) -> None:
        self._p("pending", job_id).unlink(missing_ok=True)
        if self._p("running", job_id).exists():
            self._p("running", job_id, "cancel").write_text("cancel")

    # ---- gate speed probes -----------------------------------------------------------------------------------------------------
    def submit_gate(self, gate_id: str, spec: dict) -> str:
        if (
            self._p("gate", gate_id).exists()
            or self._p("pending", gate_id).exists()
            or self._p("running", gate_id).exists()
        ):
            return gate_id
        write_json_atomic(
            self._p("pending", gate_id),
            {"job_id": gate_id, "kind": "gate_speed", "spec": spec, "submitted_at": time.time()},
        )
        return gate_id

    def gate_result(self, gate_id: str) -> dict | None:
        return read_json(self._p("gate", gate_id))

    def gate_state(self, gate_id: str) -> State:
        if self._p("gate", gate_id).exists():
            return "done"
        if self._p("running", gate_id).exists():
            return "running"
        if self._p("pending", gate_id).exists():
            return "pending"
        return "failed" if self._p("done", gate_id).exists() else "pending"

    # ---- housekeeping ----------------------------------------------------------------------------------------------------------
    def requeue_stale(self, now: float | None = None) -> list[str]:
        """Move claimed jobs whose worker stopped heart-beating back to pending (the worker died, was pre-empted, or the node went away)."""
        now = now or time.time()
        out = []
        for f in sorted((self.q / "running").glob("*.json")):
            job_id = f.stem
            hb = self._p("running", job_id, "hb")
            last = hb.stat().st_mtime if hb.exists() else f.stat().st_mtime
            if now - last > self.stale_claim_s:
                try:
                    os.rename(f, self._p("pending", job_id))
                except OSError:
                    continue
                hb.unlink(missing_ok=True)
                out.append(job_id)
        return out

    def counts(self) -> dict[str, int]:
        return {s: len(list((self.q / s).glob("*.json"))) for s in ("pending", "running", "done")}
