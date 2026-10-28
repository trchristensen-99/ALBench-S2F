"""B1: job backends. The local backend (one subprocess per GPU slot), the file queue with a worker (atomic claim, heartbeat, stale requeue), idempotent submission."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import b1_helpers as h
import pytest
from fake_eval import val_r_for

from ar2.jobs import base, make_backend
from ar2.jobs.base import write_json_atomic
from ar2.jobs.local import LocalBackend
from ar2.jobs.queue import QueueBackend
from ar2.jobs.worker import Worker


@pytest.fixture(autouse=True)
def fake_evaluator(monkeypatch):
    monkeypatch.setenv("AR2_EVAL_CMD", h.FAKE_EVAL_CMD)


def test_local_backend_runs_jobs_and_is_idempotent(tmp_path):
    be = LocalBackend([None, None])
    specs = [h.write_spec(tmp_path, f"r_{i}") for i in range(4)]
    for i, sp in enumerate(specs):
        assert be.submit(f"r_{i}", sp) == f"r_{i}"
        be.submit(f"r_{i}", sp)  # second submit never starts a second job
    be.wait_all(60)
    assert [be.poll(f"r_{i}") for i in range(4)] == ["done"] * 4
    for sp in specs:
        assert (sp.parent / "metrics.json").exists() and (sp.parent / "curves.csv").exists()
    t = (specs[0].parent / "metrics.json").stat().st_mtime
    be.submit("r_0", specs[0])  # metrics exist: nothing is rerun
    time.sleep(0.2)
    assert (specs[0].parent / "metrics.json").stat().st_mtime == t


def test_local_backend_cancel_and_failed_state(tmp_path, monkeypatch):
    monkeypatch.setenv("AR2_EVAL_CMD", "python3 -c 'import sys; sys.exit(3)' --")
    be = LocalBackend([None])
    sp = h.write_spec(tmp_path, "r_bad")
    be.submit("r_bad", sp)
    be.wait_all(30)
    assert be.poll("r_bad") == "failed"  # the process ended without metrics.json
    be.cancel("r_bad")


def test_queue_submit_is_idempotent_and_worker_claims_atomically(tmp_path):
    q = tmp_path / "ws" / "queue"
    be = QueueBackend(q)
    sp = h.write_spec(tmp_path / "ws", "r_q1")
    be.submit("r_q1", sp)
    be.submit("r_q1", sp)
    assert len(list((q / "pending").glob("*.json"))) == 1 and be.poll("r_q1") == "pending"
    w1, w2 = Worker(q, idle_exit_s=0.2, poll_s=0.05), Worker(q, idle_exit_s=0.2, poll_s=0.05)
    got = []
    ts = [threading.Thread(target=lambda w=w: got.append(w.claim())) for w in (w1, w2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sorted(x is not None for x in got) == [False, True]  # exactly one worker won the rename
    assert be.poll("r_q1") == "running"


def test_worker_runs_jobs_to_done_and_patches_speed_factor(tmp_path):
    q = tmp_path / "ws" / "queue"
    be = QueueBackend(q)
    specs = {f"r_{i}": h.write_spec(tmp_path / "ws", f"r_{i}") for i in range(3)}
    for rid, sp in specs.items():
        be.submit(rid, sp)
    w = Worker(q, idle_exit_s=0.3, poll_s=0.05, heartbeat_s=0.1)
    n = w.loop()
    assert n == 3
    for rid, sp in specs.items():
        assert be.poll(rid) == "done" and (sp.parent / "metrics.json").exists()
        spec = base.read_json(sp)
        assert spec["speed_factor"] == 1.0  # no GPU here: CPU counts as 1
    assert be.counts() == {"pending": 0, "running": 0, "done": 3}
    be.submit("r_0", specs["r_0"])  # finished runs stay finished
    assert be.counts()["pending"] == 0


def test_worker_exits_on_stop_file(tmp_path):
    q = tmp_path / "ws" / "queue"
    QueueBackend(q)
    (tmp_path / "ws" / "STOP").write_text("")
    assert Worker(q, idle_exit_s=60, poll_s=0.05).loop() == 0


def test_stale_claims_are_requeued_but_live_ones_are_not(tmp_path):
    q = tmp_path / "ws" / "queue"
    be = QueueBackend(q, stale_claim_s=60)
    be.submit("r_a", h.write_spec(tmp_path / "ws", "r_a"))
    w = Worker(q, idle_exit_s=0.1, poll_s=0.05)
    running = w.claim()
    assert running is not None and be.poll("r_a") == "running"
    assert be.requeue_stale() == []  # fresh heartbeat
    old = time.time() - 3600
    os.utime(q / "running" / "r_a.hb", (old, old))
    assert be.requeue_stale() == ["r_a"] and be.poll("r_a") == "pending"


def test_queue_cancel_and_failed_receipt(tmp_path):
    q = tmp_path / "ws" / "queue"
    be = QueueBackend(q)
    sp = h.write_spec(tmp_path / "ws", "r_c")
    be.submit("r_c", sp)
    be.cancel("r_c")
    assert be.poll("r_c") == "failed"
    be.submit("r_d", h.write_spec(tmp_path / "ws", "r_d"))
    (q / "pending" / "r_d.json").rename(
        q / "done" / "r_d.json"
    )  # a worker that crashed after the receipt, without metrics
    assert be.poll("r_d") == "failed"
    be.submit(
        "r_d",
        base.run_dir_of(Path(tmp_path / "ws" / "runs" / "r_d" / "spec.json")) / "spec.json",
        force=True,
    )
    assert be.poll("r_d") == "pending"


def test_speed_factor_by_device():
    assert base.speed_factor_for("NVIDIA H100 NVL") == 1.0
    assert base.speed_factor_for("Tesla V100-SXM2-32GB") == 5.0
    assert base.speed_factor_for("NVIDIA H100 NVL MIG 1g.12gb") == 8.0
    assert base.speed_factor_for(None) == 1.0


def test_make_backend(tmp_path):
    from ar2 import config as cfgmod

    cfg = cfgmod.load_config(profile="deployment")
    assert isinstance(make_backend(cfg, tmp_path), QueueBackend)
    assert isinstance(make_backend(cfgmod.load_config(), tmp_path), LocalBackend)
    assert val_r_for({"code_dir": "x/f", "seed": 0, "recipe": h.recipe()}) < 1.0
