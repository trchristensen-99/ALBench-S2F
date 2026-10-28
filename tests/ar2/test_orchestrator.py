"""B1: the orchestrator loop without GPU, network or CLI: a scripted controller, a fake gate and tagger, a fake evaluator.

T5 (crash safety), T12 (template freeze), T16 (live balance change), idempotent ingest, OOM replacement, infrastructure retries, LLM tasks with a fake proposer,
phase D handling of val_select."""

from __future__ import annotations

import json
import os
import random
import shutil
import threading
import time as _time
from pathlib import Path

import b1_helpers as h
import pytest

from ar2 import config as cfgmod
from ar2.jobs.local import LocalBackend
from ar2.memory import archive, final
from ar2.memory.db import DB
from ar2.orchestrator import FrozenError, Orchestrator
from ar2.proposer.base import ProposerPaused, ProposerResult
from ar2.state import Action


def run_to_done(o: Orchestrator, max_steps: int = 60):
    last = None
    for _ in range(max_steps):
        last = o.step()
        if last.done:
            break
    return last


def test_phase_a_end_to_end(tmp_path):
    ws = h.make_ws(tmp_path)
    o = h.new_orch(ws)
    rep = run_to_done(o)
    assert rep.done and (ws / "DONE").exists()
    db = o.db
    runs = db.runs_for()
    assert len(runs) == 3 * (3 + 3) and all(r["status"] == "ok" for r in runs)
    fams = {r["family"]: r for r in db.families()}
    assert set(fams) == {"cnn_residual", "rnn_hybrid", "transformer_hybrid"} and all(
        r["n_confirm"] == 3 for r in fams.values()
    )
    assert abs(db.spend() - 18.0) < 1e-6 or db.spend() <= 18.0  # 30 of 40 epochs per run
    # every candidate was filed under the family the tagger assigned (rules for the reference models), not what the card claimed
    assert {c["family"] for c in db.candidates()} == set(fams)
    assert all(c["gate_status"] == "ok" for c in db.candidates())
    # decisions are logged with the archive hash, including NOTE rows
    dec = db.decisions()
    assert (
        any(d["mode"] == "EXPLORE" for d in dec)
        and any(d["mode"] == "TUNE" for d in dec)
        and any(d["mode"] == "CONFIRM" for d in dec)
    )
    assert all(d["archive_hash"] for d in dec)
    # run directories: spec, metrics (written last), curves
    r0 = next(r for r in runs if r["purpose"] == "first_look")
    rd = ws / "runs" / r0["run_id"]
    assert (
        (rd / "spec.json").exists()
        and (rd / "metrics.json").exists()
        and (rd / "curves.csv").exists()
    )
    spec = json.loads((rd / "spec.json").read_text())
    assert spec["val_select_manifest"] is None and "val_select" not in json.dumps(
        {k: v for k, v in spec.items() if k != "val_select_manifest"}
    )
    # confirmation runs keep checkpoints, screening runs do not
    assert (
        json.loads(
            (
                ws
                / "runs"
                / [r for r in runs if r["purpose"] == "confirm"][0]["run_id"]
                / "spec.json"
            ).read_text()
        )["save_ckpt"]
        is True
    )
    assert spec["save_ckpt"] is False and spec["time_cap_s"] == pytest.approx(1800 * 0.3**0.72)
    # curves were turned into early-termination references and an improvement rate
    m = r0["meta"]
    assert "0.5" in m["checkpoint_best"] and "improvement_per_epoch" in m


def _ledger(ws):
    return h.ledger_keys(ws)


def test_T5_crash_safety_20_random_kill_points(tmp_path):
    """Kill the orchestrator at 20 random points (an exception between 'decision logged' and 'run submitted', and at every other create_run / ingest boundary); the
    resumed search must end with the same runs as an uninterrupted one: none missing, none duplicated, the same cost."""
    ref = h.make_ws(tmp_path / "ref")
    o = h.new_orch(ref)
    run_to_done(o)
    ref_keys, ref_cost, ref_n = _ledger(ref)
    o.close()
    rng = random.Random(5)
    for trial in range(20):
        ws = h.make_ws(tmp_path / f"t{trial}")
        crash_at = rng.randint(1, ref_n + 12)
        where = rng.choice(["create_run", "ingest", "finish_llm"])
        n = {"c": 0}
        o = h.new_orch(ws)
        orig = getattr(
            Orchestrator,
            {"create_run": "create_run", "ingest": "_ingest_metrics", "finish_llm": "_finish_llm"}[
                where
            ],
        )

        def bomb(self, *a, _orig=orig, _n=n, _at=crash_at, **k):
            _n["c"] += 1
            if _n["c"] == _at:
                raise KeyboardInterrupt("simulated kill")
            return _orig(self, *a, **k)

        setattr(
            Orchestrator,
            {"create_run": "create_run", "ingest": "_ingest_metrics", "finish_llm": "_finish_llm"}[
                where
            ],
            bomb,
        )
        try:
            try:
                run_to_done(o, 80)
            except KeyboardInterrupt:
                pass
        finally:
            setattr(
                Orchestrator,
                {
                    "create_run": "create_run",
                    "ingest": "_ingest_metrics",
                    "finish_llm": "_finish_llm",
                }[where],
                orig,
            )
        o2 = h.new_orch(ws)  # a fresh process: new objects, same workspace
        run_to_done(o2, 80)
        keys, cost, nn = _ledger(ws)
        assert keys == ref_keys and nn == ref_n and abs(cost - ref_cost) < 1e-9, (
            trial,
            where,
            crash_at,
        )
        o2.close()


def test_ingest_is_idempotent_and_results_only_count_once(tmp_path):
    ws = h.make_ws(tmp_path)
    o = h.new_orch(ws)
    run_to_done(o)
    spend = o.db.spend()
    for _ in range(3):
        assert o.ingest_runs() == 0
    assert o.db.spend() == spend


def test_T12_template_freeze(tmp_path, monkeypatch):
    import ar2.prompts as prompts

    d = tmp_path / "prompts_v1"
    shutil.copytree(prompts.PROMPT_DIR, d)
    monkeypatch.setattr(prompts, "PROMPT_DIR", d)
    ws = h.make_ws(tmp_path)
    h.new_orch(ws).close()  # unchanged templates: fine
    (d / "core.md").write_text((d / "core.md").read_text() + "\nAn extra sentence added mid-run.\n")
    with pytest.raises(FrozenError):
        h.new_orch(ws)
    o = h.new_orch(ws, new_arm=True)  # an explicit new-arm flag records the new version instead
    assert any(
        k.startswith("prompt_arm_")
        for k in [r["key"] for r in o.db.con.execute("SELECT key FROM meta").fetchall()]
    )
    o.close()
    h.new_orch(ws).close()  # and the new version is now the frozen one


def test_config_change_is_refused_unless_told(tmp_path):
    ws = h.make_ws(tmp_path)
    cfg = cfgmod.load_frozen(ws)
    cfg["controller"]["budget"]["main"] = 7
    (ws / "config.yaml").write_text(__import__("yaml").safe_dump(cfg))
    with pytest.raises(FrozenError):
        h.new_orch(ws)
    o = h.new_orch(ws, allow_config_change=True)
    o.close()


def test_T16_live_balance_change_applies_next_step_and_is_logged(tmp_path):
    ws = h.make_ws(tmp_path)
    o = h.new_orch(
        ws, controller=h.FakeController(cfgmod.load_frozen(ws), stop_after_anchors=False)
    )
    o.step()
    with pytest.raises(cfgmod.ConfigError):
        cfgmod.set_live(ws, "controller.budget.main", 1)
    cfgmod.set_live(ws, "balance.explore_fraction", 0.25)
    assert o.cfg["balance"]["explore_fraction"] == 0.6  # not applied until the next step
    o.step()
    assert o.cfg["balance"]["explore_fraction"] == 0.25
    sets = o.db.decisions("SET")
    assert (
        len(sets) == 1
        and sets[0]["target"] == "balance.explore_fraction"
        and "0.6 -> 0.25" in sets[0]["reason"]
    )
    o.step()
    assert len(o.db.decisions("SET")) == 1  # logged once


class OneShot:
    """A controller that issues a fixed list of actions at the first step, then nothing."""

    def __init__(self, make):
        self.make, self.done, self.guard_state = make, False, {}

    def decide(self, snap, free_jobs, free_calls):
        if self.done:
            return []
        self.done = True
        return self.make(snap)


def _anchor_candidates(o):
    # register one anchor through the orchestrator (reference model, no LLM call)
    acts = [
        Action(
            mode="EXPLORE",
            target={
                "family": "cnn_residual",
                "anchor": True,
                "source": "reference",
                "sketch": None,
            },
            phase="A",
            category="anchor",
            reason="anchor",
        )
    ]
    c = OneShot(lambda snap: acts)
    o.controller = c
    o.step()
    for _ in range(50):
        if o.db.candidates(gate_status="ok"):
            break
        o.step()
        import time

        time.sleep(0.02)
        o._finish_llm()
    return o.db.candidates(gate_status="ok")[0]["code_hash"]


def test_oom_is_replaced_with_half_batch_and_infra_errors_retry_without_counting(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("AR2_EVAL_CMD", h.FAKE_EVAL_CMD)
    ws = h.make_ws(tmp_path)
    be = LocalBackend([None, None, None, None])
    o = h.new_orch(ws, backend=be)
    ch = _anchor_candidates(o)
    recipes = [h.recipe(bs=1024), h.recipe(lr=0.0123), h.recipe(lr=0.0456), h.recipe(lr=0.003)]
    acts = [
        Action(
            mode="TUNE",
            target={"code_hash": ch, "family": "cnn_residual"},
            recipes=recipes,
            seeds=[0] * 4,
            purpose="first_look",
            category="anchor",
            phase="A",
            reason="x",
        )
    ]
    o.controller = OneShot(lambda snap: acts)
    for _ in range(40):
        o.step()
        be.wait_all(30)
        if not o.db.runs_for(pending=True) and len(o.db.runs_for()) >= 5:
            o.step()
            break
    runs = {r["run_id"]: r for r in o.db.runs_for()}
    oom = [r for r in runs.values() if r["status"] == "oom"]
    assert len(oom) == 1 and oom[0]["meta"].get("replaced_by") in runs
    repl = runs[oom[0]["meta"]["replaced_by"]]
    assert (
        repl["recipe"]["batch_size"] == 512
        and repl["meta"]["replaces"] == oom[0]["run_id"]
        and repl["status"] == "ok"
    )
    err = [r for r in runs.values() if r["recipe"]["lr"] == 0.0123][0]
    assert (
        err["status"] == "ok" and err["attempt"] == 1
    )  # one infrastructure failure, resubmitted, then ok
    assert [r for r in runs.values() if r["recipe"]["lr"] == 0.0456][0]["status"] == "diverged"
    # derived counts: the replaced oom run holds no recipe slot, the replacement and the diverged run do
    fam = {r["family"]: r for r in o.db.families()}["cnn_residual"]
    assert fam["n_runs"] == 5
    st = archive.derive_families(o.db, o.cfg)["cnn_residual"]
    assert st.first_look_done == 4 and st.first_look_open == 0


# ------------------------------------------------------------------------------------------------------------------------------------------------ LLM tasks
GOOD_SRC = (
    "import torch\nimport torch.nn as nn\n\nHPARAMS = {'width': WIDTH, 'depth': 3}\nTUNABLE = {'width': [32, 128]}\n\n\n"
    "def build_model(out_dim, **hp):\n    return nn.Sequential(nn.Flatten(), nn.Linear(800, out_dim, bias=BIAS))  # TAGHERE\n"
)
CARD = {
    "family_claimed": "cnn_plain",
    "summary": "a tiny linear model",
    "hypothesis": "h",
    "suggested_recipe": {"lr": 0.001},
}


_LOCK = threading.Lock()


class ScriptedProposer:
    """Answers EXPLORE / REFINE / REPAIR / SKETCH calls from a script and records them; a source containing DEFECT fails the fake gate."""

    def __init__(self, first_defect=False, fail=False, pause_first=None, same_hparams=False):
        self.workdirs = []
        self.calls, self.first_defect, self.fail, self.pause_first, self.n, self.same = (
            [],
            first_defect,
            fail,
            pause_first,
            0,
            same_hparams,
        )

    def call(self, prompt, workdir, expect, timeout_s):
        with _LOCK:  # calls come from pool threads: the counter must not race (two threads would write identical sources)
            self.n += 1
            n = self.n
            self.calls.append(expect)
            self.workdirs.append(sorted(q.name for q in Path(workdir).iterdir()))
        if self.pause_first is not None and n == 1:
            raise ProposerPaused("rate limited", self.pause_first)
        if self.fail:
            return ProposerResult(ok=False, error="boom", model_id="scripted")
        if expect == "sketches":
            sk = [
                {
                    "family": "unet",
                    "name": f"sk{i}",
                    "mechanism": "m",
                    "why_for_mpra": "w",
                    "nearest_existing": "n",
                    "difference": "d",
                    "recipe_needs": "r",
                }
                for i in range(10)
            ]
            return ProposerResult(ok=True, parsed=sk, model_id="scripted")
        src = GOOD_SRC.replace("TAGHERE", f"v{n}").replace(
            "WIDTH", str(64 if self.same else 64 + 16 * n)
        ).replace("BIAS", str(n % 2 == 0)) + (
            "# DEFECT\n" if (self.first_defect and n == 1) else ""
        )
        return ProposerResult(
            ok=True,
            files={"model.py": src, "card.json": json.dumps(CARD)},
            parsed=dict(CARD),
            model_id="scripted",
            usage={"input_tokens": 10, "output_tokens": 5},
        )


def gate_with_defects(cdir, cfg, **kw):
    from ar2 import gatekeeper as gk

    src = (Path(cdir) / "model.py").read_text()
    if "DEFECT" in src:
        res = gk.GateResult(
            False, "shape", "repair", "output has the wrong shape", "Traceback...", {"n_params": 5}
        )
    else:
        res = gk.GateResult(
            True,
            None,
            "pass",
            "",
            "",
            {"n_params": 1000, "throughput_seq_per_s": 5000.0, "amp_safe": True},
        )
    gk.write_gate_json(cdir, res)
    return res


def _explore_action(i=0):
    sk = {"id": f"s_test_{i}", "family": "unet", "name": "sk", "mechanism": "m"}
    return Action(
        mode="EXPLORE",
        target={"family": "unet", "sketch": sk, "anchor": False, "source": "llm"},
        phase="M",
        category="explore",
        reason="explore",
    )


def _orch(tmp_path, proposer, controller, **kw):
    ws = h.make_ws(tmp_path, **kw.pop("overrides", {}))
    o = Orchestrator(
        ws,
        controller=controller,
        proposer=proposer,
        backend=h.FakeBackend(),
        gate_fn=kw.pop("gate_fn", gate_with_defects),
        tag_fn=h.fake_tag,
        **kw,
    )
    o.db.insert_sketches(
        [{"sketch_id": "s_test_0", "family": "unet", "name": "sk", "body": {"mechanism": "m"}}]
    )
    return o


def _drain(o, n=60):
    for _ in range(n):
        o.step()
        if not o.inflight:
            break
        _time.sleep(0.02)
    o._finish_llm()


def test_explore_repair_loop_logs_every_call_and_files_the_repaired_candidate(tmp_path):
    prop = ScriptedProposer(first_defect=True)
    o = _orch(tmp_path, prop, OneShot(lambda snap: [_explore_action()]))
    _drain(o)
    cands = o.db.candidates()
    by = {c["gate_status"]: c for c in cands}
    assert (
        set(by) == {"superseded", "ok"}
        and by["superseded"]["gate_code"] == "shape"
        and by["ok"]["repair_attempts"] == 1
    )
    assert (
        by["ok"]["family"] == "cnn_plain"
        and by["ok"]["family_claimed"] == "cnn_plain"
        and by["ok"]["sketch_id"] == "s_test_0"
    )
    assert by["ok"]["hparams"]["hparams"] == {"width": 96, "depth": 3} and by["ok"]["hparams"][
        "tunable"
    ] == {"width": [32, 128]}
    assert [c["mode"] for c in o.db.calls()] == ["EXPLORE", "REPAIR"] or sorted(
        c["mode"] for c in o.db.calls()
    ) == ["EXPLORE", "REPAIR"]
    assert prop.calls == ["candidate", "candidate"]
    # the repair call works on the failing code and its card (a real CLI refused to create card.json from nothing); the first call has neither
    assert "parent_model.py" not in prop.workdirs[0] and "card.json" not in prop.workdirs[0]
    assert "parent_model.py" in prop.workdirs[1] and "card.json" in prop.workdirs[1]
    assert {s["sketch_id"]: s["status"] for s in o.db.sketches()}["s_test_0"] == "used"
    c0 = o.db.calls("EXPLORE")[0]
    assert (
        c0["model_id"] == "scripted"
        and c0["tokens_in"] == 10
        and (o.ws / "calls" / c0["call_id"] / "prompt.md").exists()
    )
    assert c0["assignment"]["outcome"]["landed_family"] == "cnn_plain"
    # the gate wrote gate.json next to each candidate
    assert all((o.cdir(c["code_hash"]) / "gate.json").exists() for c in cands)
    # the candidate is now visible to the controller as unplanned
    snap = archive.build_snapshot(o.db, o.cfg, 1)
    assert [u["code_hash"] for u in snap.new_candidates_unplanned] == [by["ok"]["code_hash"]]


def test_unrepairable_gate_failure_rejects_and_closes_the_sketch(tmp_path):
    from ar2 import gatekeeper as gk

    def reject(cdir, cfg, **kw):
        res = gk.GateResult(False, "unsafe", "reject", "import os", "", {})
        gk.write_gate_json(cdir, res)
        return res

    o = _orch(
        tmp_path, ScriptedProposer(), OneShot(lambda snap: [_explore_action()]), gate_fn=reject
    )
    _drain(o)
    c = o.db.candidates()
    assert len(c) == 1 and c[0]["gate_status"] == "rejected" and c[0]["gate_code"] == "unsafe"
    assert {s["sketch_id"]: s["status"] for s in o.db.sketches()}["s_test_0"] == "failed"
    assert archive.build_snapshot(o.db, o.cfg, 1).new_candidates_unplanned == []


def test_near_duplicate_after_tagging_is_rejected_for_explore(tmp_path):
    prop = ScriptedProposer(same_hparams=True)
    acts = {"n": 0}

    def make(snap):
        acts["n"] += 1
        return [_explore_action(acts["n"])] if acts["n"] <= 2 else []

    o = _orch(tmp_path, prop, h.FakeController.__new__(h.FakeController))  # replaced below
    o.controller = type("C", (), {"guard_state": {}, "decide": lambda self, s, j, c: make(s)})()
    o.db.insert_sketches(
        [
            {"sketch_id": "s_test_1", "family": "unet", "name": "sk", "body": {}},
            {"sketch_id": "s_test_2", "family": "unet", "name": "sk", "body": {}},
        ]
    )
    for _ in range(6):
        o.step()
        _drain(o)
    st = [c["gate_status"] for c in o.db.candidates()]
    assert (
        st.count("ok") == 1 and st.count("rejected") == 1
    )  # same tags and the same HPARAMS: the second is a near duplicate
    assert [c for c in o.db.candidates() if c["gate_status"] == "rejected"][0][
        "gate_code"
    ] == "near_duplicate"


def test_pause_stops_new_calls_but_training_continues(tmp_path):
    t = [1000.0]
    prop = ScriptedProposer(pause_first=120.0)
    o = _orch(tmp_path, prop, OneShot(lambda snap: [_explore_action()]), clock=lambda: t[0])
    _drain(o)
    assert o.pause_until == pytest.approx(1120.0) and o.db.candidates() == []
    o.controller = OneShot(lambda snap: [_explore_action()])
    rep = o.step()
    assert rep.started_calls == 0 and rep.paused_until == pytest.approx(
        1120.0
    )  # still inside the back-off: no new call
    # training is not paused: a TUNE action still creates and ingests runs
    ch = _anchor_candidates(o)
    o.controller = OneShot(
        lambda snap: [
            Action(
                mode="TUNE",
                target={"code_hash": ch, "family": "cnn_residual"},
                recipes=[h.recipe()],
                seeds=[0],
                purpose="first_look",
                category="anchor",
                phase="A",
                reason="t",
            )
        ]
    )
    o.step()
    o.step()
    assert any(r["status"] == "ok" for r in o.db.runs_for())
    t[0] = 1200.0
    o.controller = OneShot(lambda snap: [_explore_action()])
    assert o.step().started_calls == 1
    _drain(o)
    assert [c["gate_status"] for c in o.db.candidates()] == [
        "ok",
        "ok",
    ]  # the anchor and the explored candidate


def test_consecutive_call_failures_halt_the_run_resumably(tmp_path):
    prop = ScriptedProposer(fail=True)
    o = _orch(
        tmp_path,
        prop,
        h.FakeController.__new__(h.FakeController),
        overrides={"proposer.halt_after_failures": 2},
    )
    o.controller = type(
        "C", (), {"guard_state": {}, "decide": lambda self, s, j, c: [_explore_action(s.step)]}
    )()
    with pytest.raises(RuntimeError, match="consecutive failed LLM calls"):
        for _ in range(40):
            o.step()
            _time.sleep(0.02)
    assert o.db.calls() and all(c["status"] == "failed" for c in o.db.calls())
    assert len(o.db.decisions("EVENT")) >= 1


def test_phase_d_specs_carry_val_select_only_for_final_runs(tmp_path, monkeypatch):
    ws = h.make_ws(tmp_path)
    o = h.new_orch(ws)
    ch = _anchor_candidates(o)
    acts = [
        Action(
            mode="FINAL",
            target={
                "code_hash": ch,
                "family": "cnn_residual",
                "recipe_hash": "x",
                "val_select": True,
            },
            recipes=[h.recipe()] * 3,
            seeds=[4, 5, 6],
            purpose="final",
            category="final",
            phase="D",
            reason="d",
        ),
        Action(
            mode="TUNE",
            target={"code_hash": ch, "family": "cnn_residual"},
            recipes=[h.recipe(0.003)],
            seeds=[0],
            purpose="tune",
            category="explore",
            phase="M",
            reason="t",
        ),
    ]
    o.controller = OneShot(lambda snap: acts)
    o.step()
    o.step()
    for r in o.db.runs_for():
        spec = json.loads((o.ws / "runs" / r["run_id"] / "spec.json").read_text())
        assert (spec["val_select_manifest"] == "val_select.json") == (r["purpose"] == "final")
    fin = final.read_final_scores(o.db.path)
    assert sorted(x["seed"] for x in fin) == [4, 5, 6] and all(
        x["val_select_r"] < x["val_loop_r"] for x in fin
    )
    assert all(not hasattr(o.db, n) for n in ("final_scores", "read_final_scores"))


def test_invalid_first_reply_gets_one_followup_call(tmp_path):
    class Flaky(ScriptedProposer):
        def call(self, prompt, workdir, expect, timeout_s):
            if self.n == 0:  # the first reply has no model.py
                self.n += 1
                self.calls.append(expect)
                self.workdirs.append(sorted(q.name for q in Path(workdir).iterdir()))
                return ProposerResult(
                    ok=True, files={}, parsed=None, model_id="scripted", missing=["model.py"]
                )
            return super().call(prompt, workdir, expect, timeout_s)

    prop = Flaky()
    o = _orch(tmp_path, prop, OneShot(lambda snap: [_explore_action()]))
    _drain(o)
    assert [c["mode"] for c in o.db.calls()] == ["EXPLORE", "REPAIR"] or sorted(
        c["mode"] for c in o.db.calls()
    ) == ["EXPLORE", "REPAIR"]
    assert [c["gate_status"] for c in o.db.candidates()] == ["ok"]
    assert o.db.calls("REPAIR")[0]["assignment"]["followup"] is True


def test_run_stops_with_a_stalled_marker_when_there_is_nothing_to_do(tmp_path):
    ws = h.make_ws(tmp_path, **{"controller.stall_steps": 4})
    o = Orchestrator(
        ws,
        controller=OneShot(lambda snap: []),
        backend=h.FakeBackend(),
        gate_fn=h.fake_gate,
        tag_fn=h.fake_tag,
    )
    lines = []
    rep = o.run(max_steps=50, sleep=lambda s: None, log=lines.append)
    assert (ws / "STALLED").exists() and "stalled" in rep.note and not (ws / "DONE").exists()
    assert rep.step <= 6 and lines and "stalled" in lines[-1]
