"""The orchestrator: single writer of the ledger, owner of ``ar2 step`` (spec 3.2).

One pass (``step``): verify the frozen config and templates, apply live ``balance`` overrides, requeue stale jobs, ingest finished runs (metrics.json) and finished LLM
tasks, build the controller's snapshot from the ledger, call ``Controller.decide`` and execute the actions. Every step is idempotent: runs are keyed by ``run_id`` (the
UNIQUE constraint of the ledger), result files are written by temporary name and rename, and a crash between "decision logged" and "run submitted" only costs a repeat of
the decision.

LLM work (SKETCH, EXPLORE, REFINE, REPAIR, tagging) runs in a thread pool; a thread only writes files under ``ws/calls`` and ``ws/candidates`` and returns plain data, and the
main thread alone writes the database. Candidate code is never imported here: only the gatekeeper's sandbox and the evaluator subprocess execute it.
"""

from __future__ import annotations

import concurrent.futures as cf
import csv
import dataclasses
import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ar2 import config as cfgmod
from ar2.jobs.base import read_json, write_json_atomic
from ar2.memory import archive
from ar2.memory.archive import TAXONOMY
from ar2.memory.db import DB
from ar2.state import Action, Snapshot


class FrozenError(RuntimeError):
    """The config or the prompt templates changed under a running search (spec 2.x, 12.2, test T12)."""


@dataclass
class StepReport:
    step: int
    ingested: int = 0
    submitted_runs: int = 0
    started_calls: int = 0
    finished_calls: int = 0
    actions: list[str] = field(default_factory=list)
    paused_until: float | None = None
    done: bool = False
    note: str = ""


def make_run_id(code_hash: str, recipe_hash: str, pool_id: str, seed: int, purpose: str) -> str:
    return (
        "r_"
        + hashlib.sha1(
            f"{code_hash}|{recipe_hash}|{pool_id}|{seed}|{purpose}".encode()
        ).hexdigest()[:12]
    )


def _sha1(s: str, n: int = 10) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:n]


def read_curves(path: Path, max_epochs: int, checkpoints=archive.CHECKPOINTS) -> dict:
    """From curves.csv (columns ``epoch`` and ``val_r``): best validation r up to each checkpoint and the mean r gain per epoch over the last 5 epochs."""
    out: dict[str, Any] = {"checkpoint_best": {}, "improvement_per_epoch": 0.0}
    try:
        rows = list(csv.DictReader(open(path)))
    except OSError:
        return out
    pts = []
    for r in rows:
        try:
            pts.append((int(float(r["epoch"])), float(r.get("val_r", r.get("val_pearson", "nan")))))
        except (KeyError, ValueError):
            continue
    pts = [(e, v) for e, v in pts if v == v]
    if not pts:
        return out
    for c in checkpoints:
        upto = [v for e, v in pts if e <= max(1, round(c * max_epochs))]
        if upto:
            out["checkpoint_best"][str(c)] = max(upto)
    tail = [v for _, v in pts[-6:]]
    if len(tail) >= 2:
        out["improvement_per_epoch"] = (tail[-1] - tail[0]) / (len(tail) - 1)
    return out


class Orchestrator:
    """Owns one workspace. Dependencies are injectable so the whole loop runs without GPU, network or CLI (tests, simulation)."""

    def __init__(
        self,
        ws: str | Path,
        *,
        proposer: Any = None,
        controller: Any = None,
        backend: Any = None,
        gate_fn: Callable | None = None,
        tag_fn: Callable | None = None,
        allow_config_change: bool = False,
        new_arm: bool = False,
        threads: int | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self.ws = Path(ws)
        self.db = DB(self.ws / "ar2.db")
        self.clock = clock
        self.frozen = cfgmod.load_frozen(self.ws)
        self._check_frozen(allow_config_change, new_arm)
        self.cfg, pending = cfgmod.effective_config(self.ws)
        self.proposer = proposer
        self.controller = controller
        self.backend = backend
        self.gate_fn = gate_fn
        self.tag_fn = tag_fn
        self.pool = cf.ThreadPoolExecutor(
            max_workers=max(1, threads or self.cfg["controller"]["max_parallel_calls"])
        )
        self.inflight: dict[cf.Future, dict] = {}
        self.pause_until = 0.0
        self.consecutive_call_failures = 0
        self._log_overrides(pending)

    # ------------------------------------------------------------------------------------------------------------------ lazy dependencies
    def _proposer(self):
        if self.proposer is None:
            from ar2.proposer import make_proposer

            self.proposer = make_proposer(self.cfg)
        return self.proposer

    def _controller(self):
        if self.controller is None:
            from ar2.controller import Controller

            self.controller = Controller(self.cfg)
        return self.controller

    def _backend(self):
        if self.backend is None:
            from ar2.jobs import make_backend

            self.backend = make_backend(self.cfg, self.ws)
        return self.backend

    def _gate(self, *a, **k):
        if self.gate_fn is None:
            from ar2.gatekeeper import run_gate

            self.gate_fn = run_gate
        return self.gate_fn(*a, **k)

    def _tag(self, code: str, workdir: Path) -> dict:
        if self.tag_fn is None:
            from ar2 import tagger

            self.tag_fn = tagger.tag
        return self.tag_fn(self._proposer(), code, self._taxonomy(), self.cfg, workdir)

    def _taxonomy(self) -> list[str]:
        return list(TAXONOMY)

    # ------------------------------------------------------------------------------------------------------------------ workspace creation
    @staticmethod
    def init_workspace(
        ws: str | Path,
        cfg: dict,
        *,
        check_proposer: bool = True,
        proposer: Any = None,
        make_stress: bool = True,
    ) -> "Orchestrator":
        """Create the directory layout, freeze the config and the prompt templates, write the meta table, build the stress set once, verify the data contract."""
        ws = Path(ws)
        if (ws / "config.yaml").exists():
            raise FileExistsError(f"{ws} is already initialised")
        for sub in ("candidates", "runs", "calls", "queue", "meta", "stress", "prompt_proposals"):
            (ws / sub).mkdir(parents=True, exist_ok=True)
        if proposer is not None or (check_proposer and cfg["proposer"]["backend"] != "mock"):
            p = proposer
            if p is None:
                from ar2.proposer import make_proposer

                p = make_proposer(cfg)
            if hasattr(p, "check"):
                p.check()
        h = cfgmod.save_config(ws, cfg)
        db = DB(ws / "ar2.db")
        db.meta_set("config_hash", h)
        try:
            from ar2 import views

            db.meta_set("prompt_version", views.prompt_version(cfg))
        except ImportError:
            db.meta_set("prompt_version", "unavailable")
        db.meta_set("pinned_model", cfg["proposer"].get("model") or "")
        db.meta_set("created_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
        db.meta_set("step", "0")
        db.close()
        o = Orchestrator(ws, proposer=proposer)
        if make_stress and cfg["data"].get("val_loop_manifest"):
            o._setup_data()
        return o

    def _setup_data(self) -> None:
        """Data contract at start-up: the three roles only, no training sequence (or its reverse complement) in a validation set, one stress set per search."""
        from ar2.data import loader
        from ar2.evaluator import stress

        d = self.cfg["data"]
        train = loader.load_split(d["train_manifest"], "train")
        val = loader.load_split(d["val_loop_manifest"], "val_loop")
        others, names = [val.seqs], ["val_loop"]
        if d.get("val_select_manifest"):
            others.append(loader.load_split(d["val_select_manifest"], "val_select").seqs)
            names.append("val_select")
        loader.verify_no_overlap(train.seqs, *others, names=tuple(names))
        meta = getattr(train, "meta", {}) or {}
        self.db.meta_set(
            "pool_id", d.get("pool_id") or meta.get("pool_id") or Path(d["train_manifest"]).stem
        )
        path = self.ws / "stress" / "stress_seqs.npy"
        if not path.exists():
            stress.save_stress_set(path, stress.make_stress_set(list(val.seqs)))

    # ------------------------------------------------------------------------------------------------------------------ freeze checks
    def _check_frozen(self, allow_config_change: bool, new_arm: bool) -> None:
        h = cfgmod.config_hash(self.frozen)
        stored = self.db.meta_get("config_hash")
        if stored and h != stored:
            if not allow_config_change:
                raise FrozenError(
                    f"config.yaml hash {h} differs from the frozen {stored}; pass --allow-config-change to accept"
                )
            self.db.meta_set("config_hash", h)
        try:
            from ar2 import views

            now = views.prompt_version(self.frozen)
        except ImportError:
            return
        was = self.db.meta_get("prompt_version")
        if was and was not in ("unavailable", now):
            if not new_arm:
                raise FrozenError(
                    f"prompt templates changed ({was} -> {now}); a template change is a new arm: restart with --new-arm (spec 12.2)"
                )
            self.db.meta_set("prompt_version", now)
            self.db.meta_set(f"prompt_arm_{int(self.clock())}", f"{was} -> {now}")

    def _log_overrides(self, pending: list[dict]) -> None:
        if not pending:
            return
        step = int(self.db.meta_get("step", "0") or 0)
        for rec in pending:
            self.db.insert_decision(
                step,
                "SET",
                rec["key"],
                f"live change {rec['old']} -> {rec['value']}",
                self.db.archive_hash(),
                rec,
            )
        cfgmod.mark_overrides_applied(self.ws)

    # ------------------------------------------------------------------------------------------------------------------ paths and ids
    @property
    def pool_id(self) -> str:
        return self.db.meta_get("pool_id") or self.cfg["data"].get("pool_id") or "pool"

    def cdir(self, code_hash: str) -> Path:
        return self.ws / "candidates" / code_hash

    def rdir(self, run_id: str) -> Path:
        return self.ws / "runs" / run_id

    # ------------------------------------------------------------------------------------------------------------------ run creation
    def _family_runs(self, family: str) -> list[dict]:
        out = []
        for r in self.db.runs_for(family=family):
            if r["status"] in archive.OPEN:
                continue
            info = archive.run_info(r)
            out.append(
                {
                    "schedule": info.schedule,
                    "checkpoint_best": info.checkpoint_best,
                    "exploded": info.exploded,
                }
            )
        return out

    def run_row(
        self,
        *,
        code_hash: str,
        recipe: dict,
        seed: int,
        purpose: str,
        phase: str,
        category: str,
        meta: dict | None = None,
        cap_multiplier: float = 1.0,
    ) -> dict:
        from ar2 import tuner

        cand = self.db.candidate(code_hash)
        rh = tuner.recipe_hash(recipe)
        m = dict(meta or {})
        if cap_multiplier != 1.0:
            m["cap_multiplier"] = cap_multiplier
        return {
            "run_id": make_run_id(code_hash, rh, self.pool_id, seed, purpose),
            "code_hash": code_hash,
            "recipe": recipe,
            "recipe_hash": rh,
            "pool_id": self.pool_id,
            "D": int(self.cfg["data"]["D"]),
            "seed": seed,
            "purpose": purpose,
            "phase": phase,
            "category": category,
            "family": (cand or {}).get("family"),
            "schedule": recipe.get("schedule", "onecycle"),
            "meta": m,
        }

    def create_runs(self, rows: list[dict]) -> list[str]:
        """Record a family of runs atomically, then write each spec and submit it. Both steps are idempotent, so a crash anywhere between them is repaired by
        ``_materialize_pending`` on the next step."""
        self.db.upsert_runs(rows)
        for r in rows:
            self._materialize(r["run_id"])
        return [r["run_id"] for r in rows]

    def create_run(self, **kw) -> str:
        return self.create_runs([self.run_row(**kw)])[0]

    def _materialize(self, run_id: str) -> None:
        """Make sure a recorded, unfinished run has its spec.json on disk and has been handed to the job backend."""
        from ar2 import tuner

        r = self.db.run(run_id)
        if r is None or r["status"] not in archive.OPEN:
            return
        spec_path = self.rdir(run_id) / "spec.json"
        if not spec_path.exists():
            meta = r.get("meta") or {}
            mult = float(meta.get("cap_multiplier", 1.0) or 1.0)
            final_run = r["purpose"] == "final"
            et = tuner.early_term_spec(
                r["purpose"],
                r.get("schedule") or "onecycle",
                self._family_runs(r.get("family") or ""),
                self.cfg,
            )
            dd = self.cfg["data"]
            spec = {
                "run_id": run_id,
                "code_dir": str(self.cdir(r["code_hash"])),
                "recipe": r["recipe"],
                "seed": int(r["seed"]),
                "purpose": r["purpose"],
                "pool_id": r["pool_id"],
                "D": int(r["D"]),
                "out_dir": str(self.rdir(run_id)),
                "train_manifest": dd["train_manifest"],
                "val_loop_manifest": dd["val_loop_manifest"],
                "val_select_manifest": dd.get("val_select_manifest") if final_run else None,
                "time_cap_s": tuner.time_cap_s(int(r["D"]), self.cfg, multiplier=mult),
                "cap_extension_used": mult > 1.0,
                "speed_factor": 1.0,
                "stress_path": str(self.ws / "stress" / "stress_seqs.npy"),
                "save_preds": bool(self.cfg["evaluator"].get("save_preds", True)),
                "save_ckpt": r["purpose"] in ("confirm", "final"),
                "label_noise_sd": float(dd.get("label_noise_sd", 0.0)),
                "rc_average": bool(self.cfg["evaluator"]["rc_average"]),
                "early_term": et,
            }
            write_json_atomic(spec_path, spec)
        self._backend().submit(run_id, spec_path)

    def _materialize_pending(self) -> None:
        for r in self.db.runs_for(status="submitted"):
            self._materialize(r["run_id"])

    def _recover_oom(self) -> None:
        """An out-of-memory run whose replacement was never created (a crash between the two writes) gets its replacement now (spec 8.6)."""
        from ar2 import tuner

        for r in self.db.runs_for(status="oom"):
            meta = dict(r.get("meta") or {})
            if meta.get("replaced_by") or meta.get("replaces"):
                continue
            fb = tuner.handle_failure(
                "oom",
                r["recipe"],
                is_replacement=False,
                max_infra_retries=int(self.cfg["jobs"]["infra_retries"]),
            )
            if fb["action"] == "replace":
                row = self.run_row(
                    code_hash=r["code_hash"],
                    recipe=fb["recipe"],
                    seed=int(r["seed"]),
                    purpose=r["purpose"],
                    phase=r["phase"],
                    category=r["category"],
                    meta={
                        **{
                            k: v
                            for k, v in meta.items()
                            if k not in ("checkpoint_best", "improvement_per_epoch")
                        },
                        "replaces": r["run_id"],
                    },
                )
                self.create_runs([row])
                meta["replaced_by"] = row["run_id"]
                self.db.update_run(r["run_id"], meta=meta)

    # ------------------------------------------------------------------------------------------------------------------ ingest
    def ingest_runs(self) -> int:
        n = 0
        be = self._backend()
        for r in self.db.runs_for(pending=True):
            spec_path = self.rdir(r["run_id"]) / "spec.json"
            if not spec_path.exists():
                continue
            mp = self.rdir(r["run_id"]) / "metrics.json"
            state = be.poll(r["run_id"])
            if mp.exists():
                m = read_json(mp)
                if m is not None:
                    self._ingest_metrics(r, m)
                    n += 1
                continue
            if state == "running" and r["status"] == "submitted":
                self.db.update_run(r["run_id"], status="running")
            elif state in ("failed", "timeout"):
                self._infra_failure(r, f"job {state} without metrics.json")
                n += 1
            elif state == "done":
                continue  # metrics.json will be visible on the next poll (rename not yet visible)
        return n

    def _infra_failure(self, r: dict, why: str) -> None:
        meta = dict(r.get("meta") or {})
        attempt = int(r.get("attempt") or 0)
        if attempt < int(self.cfg["jobs"]["infra_retries"]):
            meta.setdefault("infra_log", []).append(why)
            self.db.update_run(r["run_id"], attempt=attempt + 1, status="submitted", meta=meta)
            sp = self.rdir(r["run_id"]) / "spec.json"
            be = self._backend()
            if hasattr(be, "submit") and "force" in be.submit.__code__.co_varnames:
                be.submit(r["run_id"], sp, force=True)
            else:
                be.submit(r["run_id"], sp)
        else:
            meta.setdefault("infra_log", []).append(why)
            self.db.finish_run(
                r["run_id"], status="error", stop_reason="infrastructure", cost_equiv=0.0, meta=meta
            )

    def _ingest_metrics(self, r: dict, m: dict) -> None:
        from ar2 import tuner

        rd = self.rdir(r["run_id"])
        recipe = r["recipe"] if isinstance(r["recipe"], dict) else {}
        max_epochs = int(recipe.get("max_epochs", self.cfg["evaluator"]["max_epochs"]))
        status = m.get("status", "ok")
        if status == "error":
            sp = rd / "metrics.json"
            try:
                sp.rename(rd / f"metrics.error{int(r.get('attempt') or 0)}.json")
            except OSError:
                pass
            self._infra_failure(r, str(m.get("error", "error")))
            return
        meta = dict(r.get("meta") or {})
        meta.update(read_curves(rd / "curves.csv", max_epochs))
        epochs_run = int(m.get("epochs_run", 0) or 0)
        cost = float(m.get("cost_equiv", epochs_run / max_epochs if max_epochs else 1.0))
        if status in ("ok", "time_cap") and m.get("stop_reason") not in ("early_stop_tuner",):
            cost = min(cost, 1.0) if epochs_run else cost
        diag = {k: v for k, v in m.items() if k not in ("run_id",)}
        diag["improvement_per_epoch"] = meta.get("improvement_per_epoch", 0.0)
        diag["max_epochs"] = max_epochs
        self.db.finish_run(
            r["run_id"],
            status=status,
            stop_reason=m.get("stop_reason"),
            val_r=m.get("val_r"),
            val_mse=m.get("val_mse"),
            val_spearman=m.get("val_spearman"),
            train_r=m.get("train_r"),
            diagnostics=diag,
            epochs_run=epochs_run,
            cost_equiv=cost,
            wall_time_s=m.get("wall_time_s"),
            meta=meta,
        )
        if r["purpose"] == "final" and m.get("val_select_r") is not None:
            from ar2.memory import final

            final.write_final_score(
                self.db.path,
                r["code_hash"],
                r["recipe_hash"],
                int(r["seed"]),
                float(m["val_select_r"]),
                float(m.get("val_r") or 0.0),
            )
        if status == "oom":
            self._recover_oom()

    # ------------------------------------------------------------------------------------------------------------------ LLM tasks
    def _prompt_ctx(self, snap: Snapshot, mode: str, action: Action):
        from ar2 import views

        c = self.cfg
        adapt = action.adaptation or {}
        fams = snap.families
        coverage = []
        for fam in sorted(set(fams) | set(self._taxonomy())):
            fs = fams.get(fam)
            n = fs.n_candidates if fs else 0
            note = ""
            if fs and fs.state in ("parked", "retired", "tuned_only") and fs.triage:
                note = str(fs.triage.get("diagnosis", fs.triage))[:120]
            coverage.append(
                {
                    "family": fam,
                    "candidates": n,
                    "state": (fs.state if fs and n else "empty"),
                    "note": note + (" (anchor)" if fam in snap.anchors and n else ""),
                }
            )
        target_family = (action.target or {}).get("family") or (
            (action.target or {}).get("sketch") or {}
        ).get("family")
        cap = int(adapt.get("extend_do_not_repeat") or 12)
        near = sorted(
            snap.candidate_summaries, key=lambda s: (s["family"] != target_family, s["family"])
        )[:cap]
        lessons = self.db.lessons()
        lg = [x["statement"] for x in lessons if x["scope"] == "global"][:15]
        lf = [x["statement"] for x in lessons if x["scope"] == target_family][:15]
        fam_view = None
        if mode == "REFINE":
            fam_view = self._family_view(snap, (action.target or {}).get("family"))
        failure = None
        if adapt.get("failure_view"):
            failure = self._failure_view()
        from ar2 import tuner

        return views.ViewContext(
            D=int(c["data"]["D"]),
            n_recipes=int(c["tuner"]["recipes_per_family"]),
            max_params=int(c["gate"]["max_params"]),
            time_cap_min=tuner.time_cap_s(int(c["data"]["D"]), c) / 60.0,
            noise_floor=float(c["controller"]["noise_floor"]),
            taxonomy=self._taxonomy(),
            coverage=coverage,
            near=near,
            lessons_global=lg,
            lessons_family=lf,
            family=fam_view,
            failure_view=failure,
            extra_constraints=list(
                adapt.get("extra_constraints")
                or (action.target or {}).get("extra_constraints")
                or []
            ),
            forced_families=adapt.get("forced_families")
            or (action.target or {}).get("forced_families"),
            speed_note=(
                "the parent's measured throughput and the time cap apply; keep the model fast"
                if adapt.get("speed_note")
                else None
            ),
            max_prompt_tokens=int(c["proposer"]["max_prompt_tokens"]),
        )

    def _failure_view(self) -> list[dict]:
        desc = {
            "build": "build_model failed",
            "shape": "wrong output shape or non-finite output",
            "params": "parameter count outside the limits",
            "coverage": "output ignores part of the 200 nt input",
            "batch_leak": "eval output depends on the other sequences in the batch",
            "learn": "cannot fit 64 random examples",
            "speed": "too slow for the time cap",
            "constant": "constant output",
            "files": "missing files",
            "determinism": "two builds with the same seed differ",
            "unsafe": "forbidden import or call",
            "duplicate": "identical to an existing candidate",
        }
        counts: dict[str, int] = {}
        for c in self.db.candidates():
            if c.get("gate_code"):
                counts[c["gate_code"]] = counts.get(c["gate_code"], 0) + 1
        return [
            {"code": k, "description": desc.get(k, k)}
            for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:3]
        ]

    def _family_view(self, snap: Snapshot, family: str | None) -> dict | None:
        fs = snap.families.get(family or "")
        if fs is None or fs.best_code_hash is None:
            return None
        cand = self.db.candidate(fs.best_code_hash) or {}
        sig = snap.sigma_seed or self.cfg["controller"]["sigma_seed_prior"]
        se = sig / max(fs.n_confirm, 1) ** 0.5
        runs = [x for x in fs.recipe_results if x["val_r"] is not None]
        rs = {}
        if runs:
            best = max(runs, key=lambda x: x["val_r"])
            rec = (
                next(
                    (
                        r["recipe"]
                        for r in self.db.runs_for(code_hash=best["code_hash"])
                        if r["recipe_hash"] == best["recipe_hash"]
                    ),
                    {},
                )
                or {}
            )
            rs = {
                "n": len(runs),
                "lo": min(x["val_r"] for x in runs),
                "hi": max(x["val_r"] for x in runs),
                "best_lr": rec.get("lr"),
                "best_weight_decay": rec.get("weight_decay"),
            }
        changes = []
        for h in fs.refinement_history:
            child = self.db.candidate(h["child"]) or {}
            card = child.get("card") if isinstance(child.get("card"), dict) else {}
            changes.append(
                {
                    "text": card.get("what_changed", "a change"),
                    "effect": h.get("effect"),
                    "se": h.get("se"),
                    "accepted": h.get("accepted"),
                    "n_seeds": 3,
                }
            )
        top = snap.top_confirmed_mean
        diag = dict(fs.last_diagnostics)
        d = {
            k: diag.get(k)
            for k in (
                "train_r",
                "val_r",
                "best_epoch",
                "max_epochs",
                "stop_reason",
                "pred_sd_ratio",
                "mse_by_activity_quintile",
                "mse_by_gc_tercile",
            )
        }
        return {
            "family": fs.family,
            "confirmed_mean": fs.confirmed_mean,
            "confirmed_se": se,
            "top_mean": top,
            "top_se": se,
            "n_seeds": fs.n_confirm,
            "params": cand.get("n_params"),
            "throughput": cand.get("throughput"),
            "diagnostics": {k: v for k, v in d.items() if v is not None},
            "recipe_sensitivity": rs,
            "earlier_changes": changes,
        }

    def _scratch(self, call_id: str) -> Path:
        root = Path(self.cfg["proposer"]["scratch_root"]) / self.ws.name
        p = root / call_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _do_call(
        self,
        call_id: str,
        mode: str,
        prompt: str,
        expect: str,
        assignment: dict,
        parent_code: str | None,
        ref_counter: int,
        parent_card: dict | None = None,
        workdir_mode: str | None = None,
    ) -> tuple[Any, dict]:
        """One proposer call: files under calls/<id> and a scratch directory outside the repository. Returns (ProposerResult, call row).

        ``parent_card`` is written as card.json into the scratch directory of REFINE / REPAIR calls (they edit it instead of inventing one). ``workdir_mode`` is the mode the
        scratch directory is prepared for when it differs from ``mode`` (the follow-up call after an invalid EXPLORE / REFINE reply)."""
        from ar2 import views

        cdir = self.ws / "calls" / call_id
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "prompt.md").write_text(prompt)
        wd = self._scratch(call_id)
        views.prepare_workdir(
            wd,
            (workdir_mode or mode) if (workdir_mode or mode) in ("REFINE", "REPAIR") else "EXPLORE",
            parent_code=parent_code,
            reference_code=views.reference_model_code(ref_counter),
            parent_card=_clean_card(parent_card),
        )
        link = cdir / "workdir"
        if not link.exists():
            try:
                link.symlink_to(wd, target_is_directory=True)
            except OSError:
                pass
        t0 = time.time()
        res = self._proposer().call(prompt, wd, expect, int(self.cfg["proposer"]["timeout_s"]))
        resp = {
            "ok": res.ok,
            "error": res.error,
            "model_id": res.model_id,
            "cli_version": res.cli_version,
            "usage": res.usage,
            "exit_code": res.exit_code,
            "duration_s": res.duration_s or time.time() - t0,
            "files": sorted(res.files),
            "parsed": res.parsed if _jsonable(res.parsed) else str(res.parsed)[:2000],
            "stdout": (res.raw_stdout or "")[-20000:],
            "stderr": (res.raw_stderr or "")[-5000:],
            "missing": list(getattr(res, "missing", []) or []),
            "proposer_changed": bool(getattr(res, "proposer_changed", False)),
        }
        write_json_atomic(cdir / "response.json", resp)
        row = {
            "call_id": call_id,
            "mode": mode,
            "assignment": assignment,
            "prompt_version": self.db.meta_get("prompt_version"),
            "prompt_path": str(cdir / "prompt.md"),
            "response_path": str(cdir / "response.json"),
            "model_id": res.model_id,
            "cli_version": res.cli_version,
            "status": "ok" if res.ok else "failed",
            "duration_s": resp["duration_s"],
            "tokens_in": int((res.usage or {}).get("input_tokens", 0) or 0),
            "tokens_out": int((res.usage or {}).get("output_tokens", 0) or 0),
        }
        return res, row

    def _speed_probe(self, cdir: Path, gate_cfg: dict, time_cap_s: float) -> dict | None:
        """Throughput probe on a GPU: via the queue workers when the queue backend is in use, in this process's sandbox otherwise."""
        be = self._backend()
        if hasattr(be, "submit_gate"):
            gid = "g_" + _sha1(f"{cdir}|{time.time()}")
            be.submit_gate(
                gid,
                {
                    "candidate_dir": str(cdir),
                    "cfg": gate_cfg,
                    "hparams": None,
                    "time_cap_s": time_cap_s,
                },
            )
            t0 = time.time()
            while time.time() - t0 < 900:
                r = be.gate_result(gid)
                if r is not None:
                    return r
                time.sleep(3)
            return None
        try:  # the local backend probes only when this machine has a GPU: a CPU projection against a GPU time cap would reject every candidate
            import torch

            device = "cuda" if torch.cuda.is_available() else None
        except ImportError:
            device = None
        if device is None:
            return None
        res = self._gate(cdir, gate_cfg, speed_probe=True, device=device, time_cap_s=time_cap_s)
        return dataclasses.asdict(res) if dataclasses.is_dataclass(res) else dict(res)

    def _candidate_task(
        self,
        kind: str,
        action: Action,
        prompt: str,
        assignment: dict,
        known_hashes: set[str],
        others: list[dict],
        ref_counter: int,
        repair_prompt_fn,
    ) -> dict:
        """EXPLORE / REFINE: one LLM call, then the gatekeeper with up to ``max_repairs`` repair calls, speed probe, tagging, near-duplicate check. Pure function of its
        inputs plus files; returns the ledger rows for the main thread to write."""
        from ar2 import gatekeeper as gk
        from ar2 import tuner

        out: dict[str, Any] = {
            "calls": [],
            "candidates": [],
            "events": [],
            "paused": None,
            "assignment": assignment,
            "kind": kind,
        }
        sketch_id = _sketch_id(action)
        anchor_src = (action.target or {}).get("source") == "reference"
        phase = action.phase or "M"
        parent = (action.target or {}).get("parent_code_hash")
        max_rep = int(self.cfg["gate"]["max_repairs"])
        cap_s = tuner.time_cap_s(int(self.cfg["data"]["D"]), self.cfg)
        call_i = 0

        def new_call_id(mode: str) -> str:
            nonlocal call_i
            call_i += 1
            return "c_" + _sha1(
                f"{self.ws}|{action.mode}|{json.dumps(assignment, sort_keys=True, default=str)}|{mode}|{call_i}|{self.clock()}"
            )

        try:
            if anchor_src:
                fam = (action.target or {}).get("family")
                src = reference_path(fam).read_text()
                card = {
                    "family_claimed": fam,
                    "summary": f"reference {fam} model",
                    "hypothesis": "anchor",
                    "suggested_recipe": {},
                }
            else:
                res, row = self._do_call(
                    new_call_id(kind),
                    kind,
                    prompt,
                    "candidate",
                    assignment,
                    self._parent_code(parent) if kind == "REFINE" else None,
                    ref_counter,
                    parent_card=self._parent_card(parent) if kind == "REFINE" else None,
                )
                out["calls"].append(row)
                if not res.ok or "model.py" not in res.files:
                    miss = (
                        (getattr(res, "missing", None) or ["model.py"])
                        if res.ok
                        else [res.error or "call failed"]
                    )
                    res, row = self._do_call(
                        new_call_id("REPAIR"),
                        "REPAIR",
                        prompt
                        + "\n\nYour previous reply was invalid: "
                        + "; ".join(map(str, miss))
                        + ". Write the missing files.",
                        "candidate",
                        {**assignment, "followup": True},
                        self._parent_code(parent) if kind == "REFINE" else None,
                        ref_counter,
                        parent_card=self._parent_card(parent) if kind == "REFINE" else None,
                        workdir_mode=kind,
                    )
                    out["calls"].append(row)
                    if not res.ok or "model.py" not in res.files:
                        out["events"].append({"kind": "call_failed", "detail": res.error})
                        return out
                src = res.files["model.py"]
                card = (
                    res.parsed
                    if isinstance(res.parsed, dict)
                    else (json.loads(res.files["card.json"]) if "card.json" in res.files else {})
                )
            attempts = 0
            while True:
                h = gk.code_hash(src)
                d = self.cdir(h)
                d.mkdir(parents=True, exist_ok=True)
                (d / "model.py").write_text(src)
                write_json_atomic(d / "card.json", card)
                gate = self._gate(
                    d,
                    self.cfg,
                    existing_hashes=known_hashes - {h} if h not in known_hashes else known_hashes,
                    device="cpu",
                    time_cap_s=cap_s,
                )
                if gate.ok and gate.measured is not None and not anchor_src:
                    pr = self._speed_probe(d, self.cfg, cap_s)
                    if pr is not None:
                        gate.measured.update((pr.get("measured") or {}))
                        if not pr.get("ok", True):
                            gate = gk.GateResult(
                                False,
                                pr.get("code") or "speed",
                                pr.get("action", "repair"),
                                pr.get("detail", ""),
                                pr.get("traceback", ""),
                                gate.measured,
                            )
                gk.write_gate_json(d, gate)
                if gate.ok:
                    break
                out["candidates"].append(
                    {
                        "code_hash": h,
                        "family_claimed": card.get("family_claimed"),
                        "gate_status": "superseded"
                        if gate.action == "repair" and attempts < max_rep
                        else "rejected",
                        "gate_code": gate.code,
                        "gate_detail": gate.detail[:2000],
                        "repair_attempts": attempts,
                        "n_params": (gate.measured or {}).get("n_params"),
                        "source": src,
                        "card": card,
                        "phase": phase,
                        "sketch_id": sketch_id,
                        "parent_code_hash": parent,
                        "hparams": None,
                        "origin": kind,
                    }
                )
                if gate.action != "repair" or attempts >= max_rep:
                    out["events"].append({"kind": "gate_rejected", "code": gate.code})
                    return out
                attempts += 1
                rp, rassign = repair_prompt_fn(src, gate, attempts)
                res, row = self._do_call(
                    new_call_id("REPAIR"),
                    "REPAIR",
                    rp,
                    "candidate",
                    rassign,
                    src,
                    ref_counter,
                    parent_card=card,
                )
                out["calls"].append(row)
                if not res.ok or "model.py" not in res.files:
                    out["events"].append({"kind": "repair_failed", "detail": res.error})
                    return out
                src = res.files["model.py"]
                card = res.parsed if isinstance(res.parsed, dict) else card
            # passed the gate: tag, near-duplicate check
            hp, tun = archive.extract_hparams(src)
            if anchor_src:
                from ar2 import tagger

                tags = tagger.rule_check(src, {"family": (action.target or {}).get("family")})
                tags["family"] = (action.target or {}).get("family")
            else:
                tags = self._tag(src, self._scratch(new_call_id("TAG")))
            tags["n_params"] = (gate.measured or {}).get("n_params")
            same = [
                {"code_hash": o["code_hash"], "tags": o["tags"], "hparams": o["hparams"]}
                for o in others
                if o.get("family") == tags.get("family")
            ]
            dup = None if anchor_src else gk.near_duplicate(tags, hp, same)
            status = "ok"
            code = None
            if dup and kind != "REFINE":
                status, code = "rejected", "near_duplicate"
            write_json_atomic(d / "tags.json", tags)
            out["candidates"].append(
                {
                    "code_hash": h,
                    "family": tags.get("family"),
                    "family_claimed": card.get("family_claimed"),
                    "tags": tags,
                    "card": card,
                    "gate_status": status,
                    "gate_code": code,
                    "gate_detail": (f"near duplicate of {dup}" if dup else ""),
                    "repair_attempts": attempts,
                    "n_params": (gate.measured or {}).get("n_params"),
                    "throughput": (gate.measured or {}).get("throughput_seq_per_s"),
                    "amp_safe": 1 if (gate.measured or {}).get("amp_safe", True) else 0,
                    "source": src,
                    "phase": phase,
                    "sketch_id": sketch_id,
                    "parent_code_hash": parent,
                    "hparams": {"hparams": hp, "tunable": tun},
                    "origin": kind,
                }
            )
            out["landed_family"] = tags.get("family")
        except Exception as ex:  # noqa: BLE001 - one failed task must not stop the search
            out["events"].append({"kind": "task_error", "detail": f"{type(ex).__name__}: {ex}"})
            try:
                from ar2.proposer.base import ProposerPaused

                if isinstance(ex, ProposerPaused):
                    out["paused"] = ex.retry_after_s or 300
            except ImportError:
                pass
        return out

    def _parent_card(self, code_hash: str | None) -> dict | None:
        c = self.db.candidate(code_hash) if code_hash else None
        return c["card"] if c and isinstance(c.get("card"), dict) else None

    def _parent_code(self, code_hash: str | None) -> str | None:
        if not code_hash:
            return None
        c = self.db.candidate(code_hash)
        return c["source"] if c else None

    def _sketch_task(
        self, action: Action, prompt: str, ref_counter: int, allowed: list[str] | None
    ) -> dict:
        out: dict[str, Any] = {
            "calls": [],
            "sketches": [],
            "events": [],
            "paused": None,
            "kind": "SKETCH",
            "assignment": {},
        }
        cid = "c_" + _sha1(f"{self.ws}|SKETCH|{self.clock()}")
        try:
            res, row = self._do_call(cid, "SKETCH", prompt, "sketches", {}, None, ref_counter)
            out["calls"].append(row)
            if res.ok and isinstance(res.parsed, list):
                for i, s in enumerate(res.parsed):
                    if not isinstance(s, dict):
                        continue
                    if (
                        allowed
                        and s.get("family") not in allowed
                        and not str(s.get("family", "")).startswith("other:")
                    ):
                        continue
                    out["sketches"].append(
                        {
                            "sketch_id": f"s_{cid[2:]}_{i}",
                            "call_id": cid,
                            "family": s.get("family"),
                            "name": s.get("name"),
                            "body": s,
                        }
                    )
            else:
                out["events"].append({"kind": "call_failed", "detail": res.error})
        except Exception as ex:  # noqa: BLE001
            out["events"].append({"kind": "task_error", "detail": f"{type(ex).__name__}: {ex}"})
            try:
                from ar2.proposer.base import ProposerPaused

                if isinstance(ex, ProposerPaused):
                    out["paused"] = ex.retry_after_s or 300
            except ImportError:
                pass
        return out

    def _start_llm(self, action: Action, snap: Snapshot) -> bool:
        """Submit one LLM action to the pool. Returns False when it is a duplicate of work already in flight."""
        from ar2 import views

        key = (
            action.mode,
            (action.target or {}).get("family"),
            (action.target or {}).get("parent_code_hash"),
            _sketch_id(action),
        )
        if any(v["key"] == key for v in self.inflight.values()):
            return False
        ctx = self._prompt_ctx(snap, action.mode, action)
        ref = len(self.db.calls()) + (1 if (action.adaptation or {}).get("rotate_reference") else 0)
        if action.mode == "SKETCH":
            prompt = views.render("SKETCH", ctx, {})
            allowed = (action.target or {}).get("forced_families")
            fut = self.pool.submit(self._sketch_task, action, prompt, ref, allowed)
        elif action.mode in ("EXPLORE", "REFINE"):
            sketch = (action.target or {}).get("sketch")
            anchor_src = (action.target or {}).get("source") == "reference"
            if action.mode == "EXPLORE" and not anchor_src:
                prompt = views.render(
                    "EXPLORE",
                    ctx,
                    {
                        "sketch": sketch
                        or {
                            "family": (action.target or {}).get("family"),
                            "name": "implement a new architecture",
                        }
                    },
                )
                assignment = {"family": (action.target or {}).get("family"), "sketch": sketch}
                if sketch and sketch.get("id"):
                    self.db.set_sketch_status(sketch["id"], "assigned")
            elif action.mode == "REFINE":
                prompt = views.render(
                    "REFINE",
                    ctx,
                    {"angle": (action.target or {}).get("angle", "capacity (width and depth)")},
                )
                assignment = {
                    "family": (action.target or {}).get("family"),
                    "angle": (action.target or {}).get("angle"),
                    "parent": (action.target or {}).get("parent_code_hash"),
                }
            else:
                prompt, assignment = (
                    "",
                    {"family": (action.target or {}).get("family"), "anchor": True},
                )
            known = self.db.all_code_hashes()
            others = [
                {
                    "code_hash": c["code_hash"],
                    "family": c.get("family"),
                    "tags": c.get("tags") or {},
                    "hparams": (c.get("hparams") or {}).get("hparams", {})
                    if isinstance(c.get("hparams"), dict)
                    else {},
                }
                for c in self.db.candidates(gate_status="ok")
            ]

            def repair_prompt(src: str, gate: Any, attempt: int, _ctx=ctx):
                a = {
                    "gate_code": gate.code,
                    "gate_description": gate.detail[:300],
                    "measured": gate.measured,
                    "traceback": gate.traceback,
                }
                return views.render("REPAIR", _ctx, a), {
                    **assignment,
                    "repair_attempt": attempt,
                    "gate_code": gate.code,
                }

            fut = self.pool.submit(
                self._candidate_task,
                action.mode,
                action,
                prompt,
                assignment,
                known,
                others,
                ref,
                repair_prompt,
            )
        else:
            return False
        self.inflight[fut] = {"key": key, "mode": action.mode, "action": action, "t": self.clock()}
        return True

    def _note_proposer_version(self, row: dict) -> None:
        """Spec 6.5: a reported model id that differs from the pinned one, or changes mid-run, is recorded as ``proposer_changed`` in meta (the report splits results by it)."""
        mid, pinned = row.get("model_id"), self.db.meta_get("pinned_model")
        resp = read_json(Path(row["response_path"])) if row.get("response_path") else None
        flagged = bool(resp and resp.get("proposer_changed"))
        seen = json.loads(self.db.meta_get("proposer_ids", "[]") or "[]")
        if mid and mid not in seen:
            seen.append(mid)
            self.db.meta_set("proposer_ids", seen)
            flagged = flagged or (len(seen) > 1)
        if flagged or (mid and pinned and mid != pinned and pinned not in ("", "mock-model-1")):
            self.db.meta_set(
                "proposer_changed",
                json.dumps({"model_id": mid, "pinned": pinned, "call_id": row.get("call_id")}),
            )

    def _near_duplicate_now(self, c: dict) -> str | None:
        from ar2 import gatekeeper as gk

        hp = (
            (c.get("hparams") or {}).get("hparams", {})
            if isinstance(c.get("hparams"), dict)
            else {}
        )
        others = [
            {
                "code_hash": o["code_hash"],
                "tags": o.get("tags") or {},
                "hparams": (
                    (o.get("hparams") or {}).get("hparams", {})
                    if isinstance(o.get("hparams"), dict)
                    else {}
                ),
            }
            for o in self.db.candidates(family=c.get("family"), gate_status="ok")
            if o["code_hash"] != c["code_hash"]
        ]
        return gk.near_duplicate(c.get("tags") or {}, hp, others)

    def _finish_llm(self) -> int:
        """Apply the results of finished LLM tasks to the ledger (main thread only)."""
        n = 0
        for fut in [f for f in self.inflight if f.done()]:
            info = self.inflight.pop(fut)
            try:
                res = fut.result()
            except Exception as ex:  # noqa: BLE001
                res = {
                    "calls": [],
                    "events": [{"kind": "task_error", "detail": f"{type(ex).__name__}: {ex}"}],
                    "paused": None,
                }
            for row in res.get("calls", []):
                self.db.insert_call(row)
                self._note_proposer_version(row)
            ok_call = any(c["status"] == "ok" for c in res.get("calls", []))
            self.consecutive_call_failures = (
                0 if ok_call else self.consecutive_call_failures + (1 if res.get("calls") else 0)
            )
            if res.get("paused"):
                self.pause_until = self.clock() + float(res["paused"])
            for s in res.get("sketches", []):
                self.db.insert_sketches([s])
            action = info["action"]
            cands = res.get("candidates", [])
            for c in cands:
                origin = c.pop("origin", "explore")
                card = dict(c.get("card") or {})
                card["_ar2"] = {
                    "origin": "anchor"
                    if (action.target or {}).get("source") == "reference"
                    else ("refine" if origin == "REFINE" else "explore"),
                    "angle": (action.target or {}).get("angle"),
                }
                c["card"] = card
                if (
                    c.get("gate_status") == "ok"
                    and origin == "EXPLORE"
                    and card["_ar2"]["origin"] != "anchor"
                ):
                    dup = self._near_duplicate_now(
                        c
                    )  # authoritative check, in the single writer, against everything recorded so far
                    if dup:
                        c["gate_status"], c["gate_code"], c["gate_detail"] = (
                            "rejected",
                            "near_duplicate",
                            f"near duplicate of {dup}",
                        )
                self.db.insert_candidate({k: v for k, v in c.items()})
            first_call = res.get("calls", [{}])[0] if res.get("calls") else {}
            if first_call.get("call_id") and info["mode"] in ("EXPLORE", "REFINE", "SKETCH"):
                fam = res.get("landed_family")
                size = len(self.db.candidates(family=fam)) if fam else 0
                outcome = {
                    "assigned_family": (action.target or {}).get("family"),
                    "landed_family": fam,
                    "landed_family_size": size,
                    "first_attempt_gate_ok": bool(cands) and cands[0].get("gate_status") in ("ok",),
                    "duplicate": any(
                        c.get("gate_code") in ("duplicate", "near_duplicate") for c in cands
                    ),
                    "speed_fail": any(c.get("gate_code") == "speed" for c in cands),
                    "collapse": bool(fam)
                    and size > int(self.cfg["adaptation"]["collapse_family_size"]),
                }
                self.db.update_call(
                    first_call["call_id"],
                    assignment={**(res.get("assignment") or {}), "outcome": outcome},
                )
            if info["mode"] == "EXPLORE" and _sketch_id(action):
                self.db.set_sketch_status(
                    _sketch_id(action),
                    "used" if any(c.get("gate_status") == "ok" for c in cands) else "failed",
                )  # a rejected sketch is not retried
            for ev in res.get("events", []):
                self.db.insert_decision(
                    int(self.db.meta_get("step", "0") or 0),
                    "EVENT",
                    info["mode"],
                    ev.get("kind", "event"),
                    self.db.archive_hash(),
                    ev,
                )
            n += 1
        return n

    # ------------------------------------------------------------------------------------------------------------------ action execution
    def _execute(self, action: Action, snap: Snapshot, rep: StepReport) -> None:
        mode, t = action.mode, action.target or {}
        if mode == "NOTE" or not action.executable:
            self._apply_note(action)
            return
        if mode in ("TUNE", "CONFIRM", "TUNE_DEEP", "SCREEN", "FINAL"):
            if mode == "SCREEN" and t.get("kind") == "adopt":
                self._apply_note(
                    Action(
                        mode="NOTE", target={"kind": "screen_adoption", **t}, reason=action.reason
                    )
                )
                return
            meta0 = {"action_mode": mode}
            if mode == "CONFIRM":
                meta0["role"] = t.get("target", "family")
            if mode == "TUNE_DEEP":
                meta0["batch"] = t.get("batch", 0)
            if mode == "SCREEN":
                meta0.update({"anchor": t.get("anchor")})
            rows = []
            for i, recipe in enumerate(action.recipes):
                mult = 1.0
                cm = t.get("cap_multipliers")
                if cm and i < len(cm):
                    mult = float(cm[i])
                if t.get("cap_multiplier"):
                    mult = float(t["cap_multiplier"])
                meta_i = dict(meta0)
                if (
                    mode == "SCREEN"
                    and isinstance(t.get("options"), list)
                    and i < len(t["options"])
                ):
                    meta_i["option"] = t["options"][i]
                ch = t.get("code_hash")
                apr = t.get("anchors_per_recipe")
                if (
                    apr and i < len(apr)
                ):  # the screen's combination runs: one recipe per anchor, each on that anchor's best model
                    meta_i["anchor"] = apr[i]
                    ch = snap.families[apr[i]].best_code_hash
                rows.append(
                    self.run_row(
                        code_hash=ch,
                        recipe=recipe,
                        seed=int(action.seeds[i]) if i < len(action.seeds) else 0,
                        purpose=action.purpose
                        or {
                            "TUNE": "tune",
                            "CONFIRM": "confirm",
                            "TUNE_DEEP": "deep_tune",
                            "SCREEN": "screen",
                            "FINAL": "final",
                        }[mode],
                        phase=action.phase or ("D" if mode == "FINAL" else "M"),
                        category=action.category or "explore",
                        meta=meta_i,
                        cap_multiplier=mult,
                    )
                )
            if rows:
                self.create_runs(rows)
                rep.submitted_runs += len(rows)
            return
        if mode in ("SKETCH", "EXPLORE", "REFINE", "PROCEDURE"):
            if self.clock() < self.pause_until and not (
                mode == "EXPLORE" and t.get("source") == "reference"
            ):  # registering a reference anchor needs no LLM
                rep.paused_until = self.pause_until
                return
            if mode == "PROCEDURE":
                return  # LLM-written procedures (phase P) are not wired in this version (documented stub)
            if self._start_llm(action, snap):
                rep.started_calls += 1
            return
        if mode == "REPAIR":
            return  # gate repair happens inside the candidate task (up to max_repairs calls); the controller's repair queue stays empty

    def _apply_note(self, action: Action) -> None:
        """Side effects of NOTE decisions. The decision row itself is the ledger fact (outcomes, family transitions); this only mirrors what must be read back quickly."""
        t = action.target or {}
        kind = t.get("kind")
        if kind == "guard_state" and isinstance(t.get("state"), dict):
            self.db.meta_set("guard_state", t["state"])
        elif kind == "stop":
            self.db.meta_set("stop_requested", "1")
        elif kind == "menu":
            self.db.meta_set("final_menu_done", "1")
        elif kind == "screen_adoption":
            adopted = t.get("adopted") or []
            self.db.meta_set(
                "recipe_defaults",
                {
                    "loss": {
                        "name": next(
                            (o for o in adopted if o in ("huber", "hetero", "corr", "binned")),
                            "mse",
                        )
                    },
                    "ema": 0.999 if "ema" in adopted else None,
                    "screen_done": True,
                },
            )
        elif kind in ("refinement_outcome", "deep_outcome") and t.get("accepted"):
            self.db.meta_set("last_progress_spend", str(self.db.spend("M")))
        elif (
            kind == "family_state"
            and t.get("to") == "confirmed"
            and t.get("family") not in archive.ANCHORS
        ):
            self.db.meta_set("last_progress_spend", str(self.db.spend("M")))

    def _already_recorded(self, action: Action) -> bool:
        """Outcome notes are facts; the controller may emit one again before it sees the first in the snapshot, so a repeat is dropped (idempotent step)."""
        t = action.target or {}
        if action.mode != "NOTE" or t.get("kind") not in (
            "refinement_outcome",
            "deep_outcome",
            "screen_adoption",
            "menu",
        ):
            return False
        key = (t.get("kind"), t.get("family"), t.get("child"), t.get("batch"))
        for d in self.db.decisions("NOTE"):
            p = d.get("payload")
            if (
                isinstance(p, dict)
                and (p.get("kind"), p.get("family"), p.get("child"), p.get("batch")) == key
            ):
                return True
        return False

    # ------------------------------------------------------------------------------------------------------------------ the step
    def step(self) -> StepReport:
        self.cfg, pending = cfgmod.effective_config(self.ws)
        self._log_overrides(pending)
        step = int(self.db.meta_get("step", "0") or 0) + 1
        self.db.meta_set("step", str(step))
        rep = StepReport(step=step)
        be = self._backend()
        if hasattr(be, "requeue_stale"):
            be.requeue_stale()
        self._materialize_pending()
        rep.ingested = self.ingest_runs()
        self._recover_oom()
        rep.finished_calls = self._finish_llm()
        archive.rebuild(self.db, self.cfg)
        if self.consecutive_call_failures >= int(self.cfg["proposer"]["halt_after_failures"]):
            rep.note = "halt: consecutive failed LLM calls"
            raise RuntimeError(
                f"{self.consecutive_call_failures} consecutive failed LLM calls; state is resumable (ar2 run)"
            )
        targets = [
            {
                "mode": v["mode"],
                "family": (v["action"].target or {}).get("family"),
                "code_hash": (v["action"].target or {}).get("parent_code_hash"),
            }
            for v in self.inflight.values()
        ]
        snap = archive.build_snapshot(
            self.db,
            self.cfg,
            step,
            calls_in_flight=len(self.inflight),
            inflight_targets=targets,
            stop_requested=(self.db.meta_get("stop_requested") == "1")
            or (self.ws / "STOP").exists(),
            taxonomy=self._taxonomy(),
        )
        ctrl = self._controller()
        free_jobs = max(0, int(self.cfg["controller"]["max_parallel_jobs"]) - snap.runs_in_flight)
        free_calls = max(0, int(self.cfg["controller"]["max_parallel_calls"]) - len(self.inflight))
        actions = [
            a for a in ctrl.decide(snap, free_jobs, free_calls) if not self._already_recorded(a)
        ]
        for a in actions:
            payload = {
                "target": a.target,
                "purpose": a.purpose,
                "category": a.category,
                "phase": a.phase,
                "n_recipes": len(a.recipes),
                "adaptation": a.adaptation,
                "expected_cost": a.expected_cost,
            }
            self.db.insert_decision(
                step,
                a.mode,
                a.target,
                a.reason,
                snap.archive_hash,
                payload if a.mode != "NOTE" else {**(a.target or {}), "reason": a.reason},
            )
            rep.actions.append(a.mode)
        for a in actions:
            self._execute(a, snap, rep)
        gs = getattr(ctrl, "guard_state", None)
        if isinstance(gs, dict) and gs:
            self.db.meta_set("guard_state", gs)
        if (
            self.db.meta_get("stop_requested") == "1"
            and not self.inflight
            and not self.db.runs_for(pending=True)
        ):
            (self.ws / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
            rep.done = True
        return rep

    def run(
        self,
        max_steps: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] | None = None,
    ) -> StepReport:
        """Loop ``step`` until the controller's stop rule fires (then ``ws/DONE`` exists) or ``max_steps``. ``log`` gets one line per step that did something."""
        n = 0
        idle_steps = 0
        last = None
        while max_steps is None or n < max_steps:
            last = self.step()
            n += 1
            busy = bool(
                last.ingested or last.started_calls or last.submitted_runs or last.finished_calls
            )
            if log and (busy or last.done):
                acts = ",".join(a for a in last.actions if a != "NOTE") or "-"
                log(
                    f"{time.strftime('%F %T')} step {last.step}: actions {acts}; ingested {last.ingested}, "
                    f"runs submitted {last.submitted_runs}, calls started {last.started_calls}/finished {last.finished_calls}; "
                    f"spend {self.db.spend():.1f} run-eq, {len(self.inflight)} calls in flight"
                    + (" DONE" if last.done else "")
                )
            if last.done:
                break
            idle = (
                not busy
                and not self.inflight
                and not self.db.runs_for(pending=True)
                and self.clock() >= self.pause_until
            )
            idle_steps = idle_steps + 1 if idle else 0
            if idle_steps >= int(self.cfg["controller"].get("stall_steps", 60)):
                (self.ws / "STALLED").write_text(
                    f"{time.strftime('%F %T')}: no run, call or decision for {idle_steps} steps; check the budgets (ar2 status)\n"
                )
                last.note = "stalled: nothing in flight and nothing to do (see ws/STALLED)"
                if log:
                    log(f"{time.strftime('%F %T')} {last.note}")
                break
            if not (last.ingested or last.started_calls or last.finished_calls):
                # submitting runs is not progress: a step that only re-issues the same submissions must not spin (it did, at one step per second, when a recipe could not run)
                sleep(float(self.cfg["controller"]["step_sleep_s"]))
        return last  # type: ignore[return-value]

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.db.close()


def _clean_card(card: dict | None) -> dict | None:
    """The card as the proposer sees it: without the orchestrator's private ``_ar2`` bookkeeping key."""
    if card is None:
        return None
    return {k: v for k, v in card.items() if k != "_ar2"}


def _sketch_id(action: Action) -> str | None:
    return ((action.target or {}).get("sketch") or {}).get("id")


def _jsonable(x: Any) -> bool:
    try:
        json.dumps(x)
        return True
    except (TypeError, ValueError):
        return False


def reference_path(family: str) -> Path:
    return Path(__file__).resolve().parent / "proposer" / "reference_models" / f"{family}.py"
