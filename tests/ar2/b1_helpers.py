"""Shared fixtures for the B1 tests: a workspace with a mock-friendly config, a scripted controller, a fake gate and tagger, an in-process job backend."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fake_eval

from ar2 import config as cfgmod
from ar2.gatekeeper import GateResult
from ar2.jobs.base import write_json_atomic
from ar2.orchestrator import Orchestrator
from ar2.state import Action

FAKE_EVAL_CMD = f"{sys.executable} {Path(__file__).with_name('fake_eval.py')}"


def recipe(lr: float = 0.002, bs: int = 512, **kw) -> dict:
    r = {
        "optimizer": "adamw",
        "lr": lr,
        "weight_decay": 0.01,
        "batch_size": bs,
        "schedule": "onecycle",
        "warmup_frac": 0.05,
        "max_epochs": 40,
        "patience": 8,
        "grad_clip": 1.0,
        "loss": {"name": "mse"},
        "rc_augment": True,
        "ema": None,
        "amp": True,
        "tasks": ["K562"],
        "hparams": {},
    }
    r.update(kw)
    return r


def make_ws(tmp: Path, **overrides) -> Path:
    cfg = cfgmod.load_config(
        overrides={
            "proposer.backend": "mock",
            "proposer.scratch_root": str(tmp / "scratch"),
            "data.D": 30000,
            "data.pool_id": "pool_test",
            "data.train_manifest": "train.json",
            "data.val_loop_manifest": "val_loop.json",
            "data.val_select_manifest": "val_select.json",
            "controller.max_parallel_jobs": 4,
            "controller.max_parallel_calls": 2,
            "controller.step_sleep_s": 0,
            **overrides,
        }
    )
    ws = tmp / "ws"
    Orchestrator.init_workspace(ws, cfg, check_proposer=False, make_stress=False).close()
    return ws


class FakeBackend:
    """Runs the fake evaluator in this process at submit time (deterministic and fast); the same idempotency contract as the real backends."""

    def __init__(self, delay_steps: int = 0):
        self.jobs: dict[str, Path] = {}
        self.delay = delay_steps
        self.submitted: list[str] = []

    def submit(self, run_id: str, spec_path: Path, *, force: bool = False) -> str:
        spec_path = Path(spec_path)
        self.jobs[run_id] = spec_path
        if (spec_path.parent / "metrics.json").exists():
            return run_id
        self.submitted.append(run_id)
        fake_eval.run(str(spec_path))
        return run_id

    def poll(self, job_id: str) -> str:
        sp = self.jobs.get(job_id)
        return "done" if sp is not None and (sp.parent / "metrics.json").exists() else "failed"

    def cancel(self, job_id: str) -> None:
        pass


def fake_gate(cdir, cfg, **kw):
    from ar2 import gatekeeper as gk

    res = GateResult(
        True,
        None,
        "pass",
        "",
        "",
        {"n_params": 1000, "throughput_seq_per_s": 5000.0, "amp_safe": True},
    )
    gk.write_gate_json(cdir, res)
    return res


class FakeController:
    """A tiny scripted policy over the snapshot, enough to drive Phase A: reference anchors, three first-look recipes, confirmation on seeds 1 to 3, stop."""

    def __init__(self, cfg: dict, stop_after_anchors: bool = True):
        self.cfg = cfg
        self.guard_state: dict = {}
        self.stop_after_anchors = stop_after_anchors
        self.calls = 0

    def decide(self, snap, free_jobs, free_calls):
        self.calls += 1
        acts: list[Action] = []
        fams = snap.families
        for a in snap.anchors:
            fs = fams.get(a)
            if (fs is None or fs.n_candidates == 0) and free_calls > 0:
                acts.append(
                    Action(
                        mode="EXPLORE",
                        target={"family": a, "anchor": True, "source": "reference", "sketch": None},
                        phase="A",
                        category="anchor",
                        reason=f"register anchor {a}",
                    )
                )
                free_calls -= 1
        for c in snap.new_candidates_unplanned:
            acts.append(
                Action(
                    mode="TUNE",
                    target={"code_hash": c["code_hash"], "family": c["family"]},
                    recipes=[recipe(0.001), recipe(0.002), recipe(0.004)],
                    seeds=[0, 0, 0],
                    purpose="first_look",
                    category="anchor",
                    phase="A",
                    reason="first look",
                )
            )
        for fs in fams.values():
            if (
                fs.n_candidates
                and fs.first_look_done >= 3
                and fs.first_look_open == 0
                and fs.n_confirm == 0
                and fs.confirm_open == 0
                and fs.best_code_hash
            ):
                acts.append(
                    Action(
                        mode="CONFIRM",
                        target={
                            "code_hash": fs.best_code_hash,
                            "family": fs.family,
                            "recipe_hash": fs.best_recipe_hash,
                            "target": "family",
                        },
                        recipes=[fs.best_recipe] * 3,
                        seeds=[1, 2, 3],
                        purpose="confirm",
                        category="anchor",
                        phase="A",
                        reason="confirm",
                    )
                )
        done = all((fams.get(a) is not None and fams[a].n_confirm >= 3) for a in snap.anchors)
        if done and self.stop_after_anchors:
            acts.append(
                Action(
                    mode="NOTE",
                    target={"kind": "stop"},
                    reason="anchors confirmed",
                    executable=False,
                )
            )
        return acts


def fake_tag(proposer, code, taxonomy, cfg, workdir):
    return {
        "family": "cnn_plain",
        "core_block": "conv",
        "uses_attention": False,
        "uses_recurrence": False,
        "normalisation": "batch",
        "activation": "relu",
        "pooling": "max",
        "positional_info": "none",
        "downsampling_factor": 1,
        "justification": "fake",
    }


def new_orch(ws: Path, **kw) -> Orchestrator:
    cfg, _ = cfgmod.effective_config(ws)
    return Orchestrator(
        ws,
        controller=kw.pop("controller", None) or FakeController(cfg),
        backend=kw.pop("backend", None) or FakeBackend(),
        gate_fn=fake_gate,
        tag_fn=fake_tag,
        **kw,
    )


def ledger_keys(ws: Path) -> tuple[set, float, int]:
    from ar2.memory.db import DB

    db = DB(ws / "ar2.db")
    runs = db.runs_for()
    keys = {(r["code_hash"], r["recipe_hash"], r["seed"], r["purpose"]) for r in runs}
    total = db.spend()
    n = len(runs)
    ids = {r["run_id"] for r in runs}
    assert len(ids) == n
    db.close()
    return keys, round(total, 9), n


def write_spec(tmp: Path, name: str = "r_x", **kw) -> Path:
    d = tmp / "runs" / name
    d.mkdir(parents=True, exist_ok=True)
    spec = {
        "run_id": name,
        "code_dir": str(tmp / "cand" / "fam"),
        "recipe": recipe(),
        "seed": 0,
        "purpose": "first_look",
        "out_dir": str(d),
        "time_cap_s": 100.0,
        **kw,
    }
    write_json_atomic(d / "spec.json", spec)
    return d / "spec.json"
