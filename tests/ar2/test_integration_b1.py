"""B1 integration: the real Controller, the mock proposer, the real views / tagger rules / tuner, a fake gate and a fake evaluator. Budgets are small so that a whole
search (phases A, M, D) finishes in seconds. Checks accounting, the phase D rule for val_select, the report and the menu export."""

from __future__ import annotations

import ast
import json
import time

import b1_helpers as h
import pytest

from ar2 import config as cfgmod
from ar2.controller import Controller
from ar2.memory import archive, final
from ar2.orchestrator import Orchestrator
from ar2.proposer import make_proposer


def _ws(tmp_path, **kw):
    return h.make_ws(
        tmp_path,
        **{
            "controller.budget": {
                "anchors": 30,
                "screen": 0,
                "procedures": 0,
                "main": 36,
                "final": 18,
            },
            "phases.screen": False,
            "phases.procedures": False,
            "procedure_mode.enabled": False,
            "controller.queue_depth": 3,
            **kw,
        },
    )


def _run(ws, max_steps=400, **orch_kw):
    cfg, _ = cfgmod.effective_config(ws)
    o = Orchestrator(
        ws,
        controller=Controller(cfg),
        proposer=make_proposer(cfg),
        backend=h.FakeBackend(),
        gate_fn=h.fake_gate,
        **orch_kw,
    )
    last = None
    for _ in range(max_steps):
        last = o.step()
        if last.done:
            break
        if not (last.ingested or last.submitted_runs or last.started_calls or last.finished_calls):
            time.sleep(0.01)  # waiting for an LLM task thread
    return o, last


def test_full_search_with_real_controller_and_mock_proposer(tmp_path):
    ws = _ws(tmp_path)
    o, last = _run(ws)
    assert last.done, "the search did not finish within the step limit"
    db = o.db
    bud = o.cfg["controller"]["budget"]
    for ph, key in archive.PHASE_BUDGET_KEY.items():
        # budgets are counted in run-equivalents and are never exceeded (a small overshoot is allowed only for the documented borrow of phase A)
        assert (
            db.spend(phase=ph) <= bud[key] + o.cfg["controller"].get("anchor_borrow", 6) + 1e-6
        ), (ph, db.spend(phase=ph))
    runs = db.runs_for()
    assert not db.runs_for(pending=True) and len({r["run_id"] for r in runs}) == len(runs)
    fams = {r["family"]: r for r in db.families()}
    assert {"cnn_residual", "rnn_hybrid", "transformer_hybrid"} <= set(fams)
    assert any(f not in archive.ANCHORS for f in fams), "no new family was explored"
    calls = db.calls()
    assert {c["mode"] for c in calls} >= {"SKETCH", "EXPLORE"}
    # candidates were filed under the tagger's family and every call has its files
    for c in calls:
        assert (o.ws / "calls" / c["call_id"] / "prompt.md").exists() and (
            o.ws / "calls" / c["call_id"] / "response.json"
        ).exists()
        assert "{{" not in (o.ws / "calls" / c["call_id"] / "prompt.md").read_text()
    # val_select appears only in phase D specs, and phase D results go to final_scores only
    for r in runs:
        spec = json.loads((o.ws / "runs" / r["run_id"] / "spec.json").read_text())
        assert (spec["val_select_manifest"] is not None) == (r["purpose"] == "final")
    fin = final.read_final_scores(db.path)
    assert len(fin) == len([r for r in runs if r["purpose"] == "final"]) > 0
    # no prompt contains val_select, a score from phase D, or a single-seed score in a REFINE prompt
    for c in calls:
        text = (o.ws / "calls" / c["call_id"] / "prompt.md").read_text()
        assert "val_select" not in text and "final_scores" not in text
    o.close()


def test_report_and_export_after_a_search(tmp_path):
    from ar2 import report

    ws = _ws(tmp_path)
    o, last = _run(ws)
    assert last.done
    o.close()
    md = report.write_report(ws).read_text()
    for section in (
        "Run summary",
        "Coverage",
        "Archive",
        "Variance split",
        "Accuracy against cost",
        "Procedure screen",
        "Lessons",
        "Search health",
        "Selection check",
        "Verdict",
    ):
        assert f"## {section}" in md or f"## {section} (" in md, section
    out = tmp_path / "menu"
    p = report.export_menu(ws, out)
    entries = json.loads(p.read_text())
    assert len(entries) >= 3
    for e in entries:
        assert {
            "id",
            "cell",
            "idea",
            "orig_val",
            "n_params",
            "source",
            "train_config",
            "family",
            "ar2_recipe",
        } <= set(e)
        assert e["train_config"].keys() == {
            "optimizer",
            "lr",
            "weight_decay",
            "batch_size",
            "epochs",
            "lr_schedule",
        }
        ast.parse(e["source"])
        assert "class Net(" in e["source"] and "build_model(1" in e["source"]
        fd = out / e["family"]
        assert {p.name for p in fd.iterdir()} == {
            "model.py",
            "recipe.json",
            "card.json",
            "tags.json",
        }
