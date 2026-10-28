"""B1: the final report (spec 16.4) and the menu export that feeds the repository's per-size retraining."""

from __future__ import annotations

import json
from pathlib import Path

import b1_helpers as h
import pytest

from ar2 import report
from ar2.memory import archive
from ar2.orchestrator import Orchestrator

REF = Path(__file__).resolve().parents[2] / "ar2" / "proposer" / "reference_models"


def _search(tmp_path):
    ws = h.make_ws(tmp_path)
    o = h.new_orch(ws)
    for _ in range(60):
        if o.step().done:
            break
    o.close()
    return ws


def test_report_sections_variance_split_and_flat_verdict(tmp_path):
    ws = _search(tmp_path)
    md = report.write_report(ws).read_text()
    for sec in (
        "Run summary",
        "Coverage",
        "Archive (confirmed families)",
        "Variance split",
        "Accuracy against cost",
        "Procedure screen",
        "Lessons",
        "Search health",
        "Selection check",
        "Verdict",
    ):
        assert f"## {sec}" in md, sec
    assert "between families (confirmed means)" in md and "across seeds" in md
    assert (
        "Phase D has not scored any family on val_select." in md
    )  # no FINAL runs in this scripted search
    assert "Nothing is confirmed" not in md and "cnn_residual" in md
    # three anchor families only: fewer than the 5 the menu needs, so all three are exported and flagged as the top confirmed ones
    from ar2 import config as cfgmod
    from ar2.memory.db import DB

    db = DB(ws / "ar2.db")
    mf = archive.menu_families(db, cfgmod.load_frozen(ws))
    assert len(mf) == 3
    db.close()


def test_export_menu_layout_and_train_config(tmp_path):
    ws = _search(tmp_path)
    p = report.export_menu(ws, tmp_path / "menu")
    entries = json.loads(p.read_text())
    assert {e["family"] for e in entries} == {"cnn_residual", "rnn_hybrid", "transformer_hybrid"}
    for e in entries:
        assert e["train_config"] == {
            "optimizer": "adamw",
            "lr": e["ar2_recipe"]["lr"],
            "weight_decay": e["ar2_recipe"]["weight_decay"],
            "batch_size": 512,
            "epochs": 40,
            "lr_schedule": "onecycle",
        }
        assert e["orig_val"] == e["ar2_confirmed_mean"] and e["id"].startswith("ar2_" + e["family"])


def test_train_config_mapping():
    r = h.recipe(lr=0.0005, bs=256, schedule="cosine_warmup", optimizer="muon", max_epochs=30)
    assert report.to_train_config(r) == {
        "optimizer": "muon",
        "lr": 0.0005,
        "weight_decay": 0.01,
        "batch_size": 256,
        "epochs": 30,
        "lr_schedule": "cosine",
    }


@pytest.mark.parametrize("family", ["cnn_residual", "rnn_hybrid", "transformer_hybrid"])
def test_exported_source_builds_with_the_repositorys_menu_builder(family):
    """The wrapper must define ``Net()`` with no arguments and one value per sequence: exactly what albench.arch_search.validate.build_and_check (and so every
    menu retraining) requires of a menu entry."""
    pytest.importorskip("torch")
    from albench.arch_search.validate import build_and_check

    src = report._rename_net((REF / f"{family}.py").read_text()) + report._WRAPPER.format(
        hp=repr({})
    )
    net, verdict = build_and_check(src, seq_len=200, max_params=40_000_000, max_fwd_ms=1e9)
    assert net is not None, verdict.reason
    assert verdict.n_params > 0
