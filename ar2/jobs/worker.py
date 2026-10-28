"""Queue worker: claims jobs from ``ws/queue/pending`` and runs them on this machine's GPU (or CPU), until the queue stays empty for ``--idle-exit-s`` or ``ws/STOP`` exists.

    python -m ar2.jobs.worker --queue ws/queue [--idle-exit-s 1800] [--poll-s 5]

Start as many as there are GPUs. On Slurm, run it inside an ordinary GPU job; if the job is pre-empted the orchestrator requeues the claimed run when its heartbeat goes stale.
The worker patches ``speed_factor`` (H100 1, V100 5, MIG slice 8) into each training spec before running it, so time caps are expressed in H100 time on every device.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path

from ar2.jobs.base import evaluator_command, read_json, speed_factor_for, write_json_atomic

HEARTBEAT_S = 30.0


def gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 - no torch or no driver: run on CPU
        pass
    return None


class Worker:
    def __init__(
        self,
        queue: Path,
        idle_exit_s: float = 1800.0,
        poll_s: float = 5.0,
        heartbeat_s: float = HEARTBEAT_S,
    ):
        self.q = Path(queue)
        self.stop_file = self.q.parent / "STOP"
        self.idle_exit_s, self.poll_s, self.heartbeat_s = idle_exit_s, poll_s, heartbeat_s
        for sub in ("pending", "running", "done", "gate"):
            (self.q / sub).mkdir(parents=True, exist_ok=True)
        self.gpu = gpu_name()
        self.speed = speed_factor_for(self.gpu)
        self.host = socket.gethostname()

    # ---- claim -----------------------------------------------------------------------------------------------------------------
    def claim(self) -> Path | None:
        pend = sorted(
            (self.q / "pending").glob("*.json"), key=lambda p: (p.stat().st_mtime, p.name)
        )
        for f in pend:
            dst = self.q / "running" / f.name
            try:
                os.rename(f, dst)  # atomic: exactly one worker wins
            except OSError:
                continue
            (self.q / "running" / f"{f.stem}.hb").write_text(self.host)
            return dst
        return None

    # ---- run -------------------------------------------------------------------------------------------------------------------
    def _heartbeat(self, job_id: str, done: threading.Event, proc_holder: list) -> None:
        hb = self.q / "running" / f"{job_id}.hb"
        cancel = self.q / "running" / f"{job_id}.cancel"
        while not done.wait(self.heartbeat_s if not cancel.exists() else 0.1):
            hb.touch()
            if cancel.exists() and proc_holder:
                try:
                    os.killpg(proc_holder[0].pid, signal.SIGTERM)
                except OSError:
                    pass
                return

    def run_train(self, rec: dict) -> dict:
        spec_path = Path(rec["spec_path"])
        spec = read_json(spec_path) or {}
        spec["speed_factor"] = self.speed
        write_json_atomic(spec_path, spec)
        log = open(spec_path.parent / "log.txt", "ab")
        log.write(
            f"--- worker {self.host} gpu={self.gpu} speed_factor={self.speed} {time.strftime('%F %T')}\n".encode()
        )
        log.flush()
        p = subprocess.Popen(
            [*evaluator_command(), "--spec", str(spec_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        done = threading.Event()
        t = threading.Thread(target=self._heartbeat, args=(rec["job_id"], done, [p]), daemon=True)
        t.start()
        try:
            rc = p.wait()
        finally:
            done.set()
            t.join(timeout=2)
            log.close()
        return {"exit": rc, "host": self.host, "gpu": self.gpu, "speed_factor": self.speed}

    def run_gate_speed(self, rec: dict) -> dict:
        from ar2.gatekeeper import run_gate

        s = rec["spec"]
        res = run_gate(
            Path(s["candidate_dir"]),
            s["cfg"],
            hparams=s.get("hparams"),
            device="cuda" if self.gpu else "cpu",
            speed_probe=True,
            speed_factor=self.speed,
            time_cap_s=s.get("time_cap_s"),
        )
        out = res.__dict__ if hasattr(res, "__dict__") else dict(res)
        write_json_atomic(self.q / "gate" / f"{rec['job_id']}.json", out)
        return {"exit": 0, "host": self.host, "gpu": self.gpu}

    def process(self, running: Path) -> None:
        rec = read_json(running) or {}
        job_id = rec.get("job_id", running.stem)
        result: dict = {}
        try:
            if rec.get("kind") == "gate_speed":
                result = self.run_gate_speed(rec)
            else:
                result = self.run_train(rec)
        except Exception as e:  # noqa: BLE001 - report, never crash the worker on one bad job
            result = {"exit": -1, "error": f"{type(e).__name__}: {e}", "host": self.host}
        result["finished_at"] = time.time()
        write_json_atomic(self.q / "done" / f"{job_id}.json", result)
        running.unlink(missing_ok=True)
        (self.q / "running" / f"{job_id}.hb").unlink(missing_ok=True)
        (self.q / "running" / f"{job_id}.cancel").unlink(missing_ok=True)

    def loop(self) -> int:
        idle_since = time.time()
        n = 0
        while True:
            if self.stop_file.exists():
                break
            claimed = self.claim()
            if claimed is None:
                if time.time() - idle_since > self.idle_exit_s:
                    break
                time.sleep(self.poll_s)
                continue
            self.process(claimed)
            n += 1
            idle_since = time.time()
        return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--queue", required=True)
    ap.add_argument("--idle-exit-s", type=float, default=1800.0)
    ap.add_argument("--poll-s", type=float, default=5.0)
    a = ap.parse_args(argv)
    w = Worker(Path(a.queue), a.idle_exit_s, a.poll_s)
    print(f"worker {w.host} gpu={w.gpu} speed_factor={w.speed} queue={a.queue}", flush=True)
    n = w.loop()
    print(f"worker done: {n} jobs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
