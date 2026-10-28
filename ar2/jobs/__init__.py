"""Job backends: local subprocesses (tests, single machine) and a file queue with external workers (cluster)."""

from __future__ import annotations

from pathlib import Path

from ar2.jobs.base import JobBackend


def make_backend(cfg: dict, ws: Path) -> JobBackend:
    kind = cfg.get("jobs", {}).get("backend", "local")
    if kind == "local":
        from ar2.jobs.local import LocalBackend

        return LocalBackend(cfg.get("jobs", {}).get("gpus"))
    if kind == "queue":
        from ar2.jobs.queue import QueueBackend

        return QueueBackend(ws / "queue", cfg.get("jobs", {}).get("stale_claim_s", 600))
    raise ValueError(f"unknown jobs.backend {kind!r}")
