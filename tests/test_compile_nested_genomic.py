"""The nested-genomic compile must refuse non-nested pools and, for nested ones, give the same models at every size on all 19 evaluation sets."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))

import compile_nested_genomic_error_metrics as nest  # noqa: E402
import compile_oct14_scaling as old  # noqa: E402

DS = (10000, 30000, 100000)


def _pools(tmp: Path, nested: bool = True) -> str:
    rng = np.random.default_rng(3)
    seqs = np.array([f"S{i}" for i in range(DS[-1])], dtype=object)
    labels = rng.normal(size=DS[-1]).astype(np.float32)
    for D in DS:
        s = seqs[:D].copy()
        if not nested and D == 30000:
            s[5] = "BROKEN"
        np.savez(tmp / f"genomic_d{D}.npz", sequences=s, oracle_labels=labels[:D])
    return str(tmp / "genomic_d{D}.npz")


def _cells(mr: Path, template: str = "sw38_{k}k_genomic_base") -> None:
    rng = np.random.default_rng(1)
    n_val, n_test = 200, 80
    yval = rng.normal(size=n_val)
    yte = {e: rng.normal(size=n_test) for e in old.EVALS}
    for D in DS:
        d = mr / template.format(k=D // 1000)
        d.mkdir(parents=True)
        np.savez(d / "labels.npz", val_labels=yval, **{f"oracle_{e}": v for e, v in yte.items()})
        for idx in range(
            7 if D < 100000 else 6
        ):  # one fewer model at 100k: only the common ones are used
            noise = 0.5 + 0.1 * idx + 1.0 / (D / 10000)
            pred = {"val_pred": yval + rng.normal(size=n_val) * noise}
            for e in old.EVALS:
                pred[f"test_pred_{e}"] = 0.8 * yte[e] + rng.normal(size=n_test) * noise
            mid = f"r00_menu_{idx:03d}"
            np.savez(d / f"{mid}.npz", **pred)
            json.dump(
                {"model_id": mid, "round": 0, "menu_index": idx}, open(d / f"{mid}_meta.json", "w")
            )


def test_nested_pools_pass_and_series_has_all_sets(tmp_path, monkeypatch):
    pools = _pools(tmp_path)
    mr = tmp_path / "mr"
    _cells(mr)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--out",
            str(tmp_path / "o"),
            "--menu_retrain",
            str(mr),
            "--pools",
            pools,
            "--series",
            "sw38_base",
        ],
    )
    nest.main()
    rows = list(csv.DictReader(open(tmp_path / "o" / "scaling_error_cells_genomic_nested.csv")))
    assert {r["eval"] for r in rows} == set(old.OUTSETS)  # 18 sets and the SNV effect
    assert {int(r["D"]) for r in rows} == set(DS)
    assert {r["n_models"] for r in rows} == {"6"}  # the models present at every size
    assert {r["source"] for r in rows} == {"sw38_base"}
    assert all(float(r["enet_all_pos_mse"]) > 0 for r in rows)


def test_non_nested_pools_refuse(tmp_path, monkeypatch):
    pools = _pools(tmp_path, nested=False)
    mr = tmp_path / "mr"
    _cells(mr)
    monkeypatch.setattr(
        sys,
        "argv",
        ["x", "--out", str(tmp_path / "o"), "--menu_retrain", str(mr), "--pools", pools],
    )
    with pytest.raises(SystemExit, match="NOT NESTED"):
        nest.main()
    assert not (tmp_path / "o" / "scaling_error_cells_genomic_nested.csv").exists()
