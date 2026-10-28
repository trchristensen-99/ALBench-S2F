"""Job backend contract (spec 4.6) and the small helpers every backend shares.

A job is ``python -m ar2.evaluator.train --spec runs/<run_id>/spec.json``. The spec is self-contained, so any scheduler can run it. A run counts as done only when its
``metrics.json`` exists (the evaluator writes it last, by temporary name and rename).
"""

from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path
from typing import Literal, Protocol

State = Literal["pending", "running", "done", "failed", "timeout"]


class JobBackend(Protocol):
    def submit(self, run_id: str, spec_path: Path) -> str:
        """Returns the job id; idempotent on run_id (a second submit of the same run never starts a second job)."""

    def poll(self, job_id: str) -> State: ...

    def cancel(self, job_id: str) -> None: ...


def evaluator_command() -> list[str]:
    """The command that runs one evaluator job, without the trailing ``--spec <path>``. ``AR2_EVAL_CMD`` overrides it (tests use a fake evaluator)."""
    env = os.environ.get("AR2_EVAL_CMD")
    return shlex.split(env) if env else [sys.executable, "-m", "ar2.evaluator.train"]


def run_dir_of(spec_path: Path) -> Path:
    return Path(spec_path).parent


def metrics_exist(spec_path: Path) -> bool:
    return (run_dir_of(spec_path) / "metrics.json").exists()


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def write_json_atomic(path: Path, obj: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def speed_factor_for(gpu_name: str | None) -> float:
    """Time-cap multiplier by device: a full H100 is 1, a V100 about 5, a MIG slice (1g.12gb) about 8 (measured on this cluster for the menu retrains)."""
    n = (gpu_name or "").upper()
    if "MIG" in n or "1G." in n:
        return 8.0
    if "V100" in n:
        return 5.0
    return 1.0
