"""B1: the command line (init, set, status, rebuild, report, export-menu, plan, simulate) and the start-up data contract wired through `ar2 init`."""

from __future__ import annotations

import json

import b1_helpers as h
import numpy as np
import pytest

from ar2 import cli
from ar2 import config as cfgmod
from ar2.data import loader, manifest
from ar2.orchestrator import Orchestrator


def _write_pool(path, seqs, labels):
    np.savez(
        path, sequences=np.asarray(seqs, dtype=object), oracle_labels=np.asarray(labels, np.float32)
    )


def _seqs(rng, n):
    return ["".join(rng.choice(list("ACGT"), 200)) for _ in range(n)]


def _manifests(tmp_path, leak: bool = False, leak_rc: bool = False):
    rng = np.random.default_rng(0)
    tr, vl, vs = _seqs(rng, 300), _seqs(rng, 80), _seqs(rng, 40)
    if leak:
        vl[0] = tr[5]
    if leak_rc:
        vs[0] = loader.revcomp(tr[7])
    out = {}
    for name, seqs, role in (
        ("train", tr, "train"),
        ("val_loop", vl, "val_loop"),
        ("val_select", vs, "val_select"),
    ):
        _write_pool(tmp_path / f"{name}.npz", seqs, rng.normal(size=len(seqs)))
        m = manifest.make_manifest(
            str(tmp_path / f"{name}.npz"),
            role,
            pool_id="pool_cli",
            D=300 if role == "train" else None,
        )
        manifest.save_manifest(tmp_path / f"{name}.json", m)
        out[name] = str(tmp_path / f"{name}.json")
    return out


def _init(tmp_path, ws, extra=(), man=None):
    man = man or _manifests(tmp_path)
    args = [
        "init",
        "--ws",
        str(ws),
        "--set",
        "proposer.backend=mock",
        "--set",
        f"proposer.scratch_root={tmp_path / 'scratch'}",
        "--set",
        "data.D=300",
        "--set",
        f"data.train_manifest={man['train']}",
        "--set",
        f"data.val_loop_manifest={man['val_loop']}",
        "--set",
        f"data.val_select_manifest={man['val_select']}",
        *extra,
    ]
    return cli.main(args)


def test_init_builds_workspace_stress_set_and_freezes(tmp_path, capsys):
    ws = tmp_path / "ws"
    assert _init(tmp_path, ws) == 0
    assert (
        (ws / "config.yaml").exists()
        and (ws / "stress" / "stress_seqs.npy").exists()
        and (ws / "ar2.db").exists()
    )
    from ar2.memory.db import DB

    db = DB(ws / "ar2.db")
    assert (
        db.meta_get("config_hash")
        and db.meta_get("prompt_version").startswith("v1+")
        and db.meta_get("pool_id") == "pool_cli"
    )
    assert len(np.load(ws / "stress" / "stress_seqs.npy", allow_pickle=True)) == 4000
    with pytest.raises(FileExistsError):
        _init(tmp_path, ws)


@pytest.mark.parametrize("kw", [{"leak": True}, {"leak_rc": True}])
def test_T1_init_refuses_a_pool_that_shares_a_sequence_or_its_reverse_complement(tmp_path, kw):
    man = _manifests(tmp_path, **kw)
    with pytest.raises(loader.LeakageError):
        _init(tmp_path, tmp_path / "ws", man=man)


def test_init_requires_model_and_manifests(tmp_path):
    with pytest.raises(SystemExit, match="proposer.model"):
        cli.main(["init", "--ws", str(tmp_path / "w1"), "--allow-missing-data"])
    with pytest.raises(SystemExit, match="data.train_manifest"):
        cli.main(["init", "--ws", str(tmp_path / "w2"), "--set", "proposer.backend=mock"])


def test_set_status_rebuild_report_export_and_plan(tmp_path, capsys, monkeypatch):
    ws = h.make_ws(tmp_path)
    o = h.new_orch(ws)
    for _ in range(60):
        if o.step().done:
            break
    o.close()
    assert cli.main(["set", "--ws", str(ws), "balance.deep_tune_share", "0.5"]) == 0
    assert (
        cli.main(["set", "--ws", str(ws), "controller.budget.main", "1"]) == 2
    )  # outside balance: refused
    capsys.readouterr()
    assert cli.main(["status", "--ws", str(ws)]) == 0
    out = capsys.readouterr().out
    assert (
        "spend by phase" in out
        and "cnn_residual" in out
        and "deep_tune_share" in out
        and "balance.deep_tune_share=0.5" in out
    )
    assert cli.main(["rebuild", "--ws", str(ws)]) == 0
    assert cli.main(["report", "--ws", str(ws)]) == 0
    assert (ws / "report.md").read_text().startswith("# ar2 search report")
    assert cli.main(["export-menu", "--ws", str(ws), "--out", str(tmp_path / "menu")]) == 0
    assert len(json.loads((tmp_path / "menu" / "menu_ar2.json").read_text())) == 3
    capsys.readouterr()
    assert cli.main(["plan", "--ws", str(ws)]) == 0
    assert "explore" in capsys.readouterr().out.lower()
    assert cli.main(["import-legacy", "--ws", str(ws), str(tmp_path)]) == 2  # documented stub


def test_simulate_command(capsys):
    assert (
        cli.main(
            [
                "simulate",
                "--seeds",
                "1",
                "--kind",
                "flat",
                "--sweep",
                "balance.explore_fraction=0.4,0.6",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "flat" in out and "0.4" in out and "menu families" in out
