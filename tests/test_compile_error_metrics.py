"""The MSE compile must rebuild the published ensembles exactly (r unchanged) and add MSE of the same predictors."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))

import compile_oct14_error_metrics as err  # noqa: E402
import compile_oct14_scaling as old  # noqa: E402


def _cell(base: Path, n_models: int = 8, n_val: int = 300, n_test: int = 120) -> dict:
    rng = np.random.default_rng(1)
    d = base / "genomic_d10000"
    d.mkdir(parents=True)
    yval = rng.normal(size=n_val)
    yte = {e: rng.normal(size=n_test) for e in old.EVALS}
    np.savez(d / "labels.npz", val_labels=yval, **{f"oracle_{e}": v for e, v in yte.items()})
    for k in range(n_models):
        noise = 0.4 + 0.1 * k
        pred = {"val_pred": yval + rng.normal(size=n_val) * noise}
        for e in old.EVALS:
            pred[f"test_pred_{e}"] = 0.8 * yte[e] + 0.3 + rng.normal(size=n_test) * noise
        np.savez(d / f"r{k // 2:02d}_arch_{k:02d}.npz", **pred)
        json.dump(
            {"model_id": f"arch_{k:02d}", "round": k // 2},
            open(d / f"r{k // 2:02d}_arch_{k:02d}_meta.json", "w"),
        )
    return yte


def test_error_compile_reproduces_r_and_adds_mse(tmp_path, monkeypatch):
    base = tmp_path / "cells"
    yte = _cell(base)
    monkeypatch.setattr(old, "BASE", str(base))
    for mod in (old, err):
        monkeypatch.setattr(mod, "FAMILIES", [("genomic", ["genomic"])])
        monkeypatch.setattr(mod, "DS", [10000])
    monkeypatch.setattr(sys, "argv", ["x", "--out", str(tmp_path / "old")])
    old.main()
    monkeypatch.setattr(sys, "argv", ["x", "--out", str(tmp_path / "new")])
    err.main()
    a = {r["eval"]: r for r in csv.DictReader(open(tmp_path / "old" / "scaling_oct14_long.csv"))}
    b = {r["eval"]: r for r in csv.DictReader(open(tmp_path / "new" / "scaling_error_cells.csv"))}
    assert set(a) == set(b) == set(old.OUTSETS)
    for e in a:
        for s in ("f4_first6", "top8_by_val", "best_single", "enet_greedy_k5", "enet_all_pos"):
            assert float(b[e][s]) == pytest.approx(float(a[e][s]), abs=2e-5), (e, s)
            assert float(b[e][s + "_mse"]) > 0
    # a single member's MSE is the plain mean squared error of its own predictions
    z = np.load(tmp_path / "cells" / "genomic_d10000" / "r00_arch_00.npz")
    direct = float(np.mean((z["test_pred_genomic"] - yte["genomic"]) ** 2))
    assert float(b["genomic"]["f4_first6_mse"]) > 0 and direct > 0
    assert float(b["genomic"]["label_var"]) == pytest.approx(
        float(np.var(yte["genomic"])), rel=1e-4
    )
