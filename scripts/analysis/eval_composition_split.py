"""Report every eval-set metric split into in-support vs OOD-composition (by GC).

Why: students are scored against the oracle across eval sets that reach GC
fractions the genomic manifold never occupies, where the oracle extrapolates.
This script recomputes each reservoir's ENSEMBLE metric on each eval set, then
partitions it by whether the sequence's GC is inside the genomic support band,
so the reservoir ranking can be read without the extrapolation tail.

Ensemble = ElasticNetCV fit on stacked per-model validation predictions, applied
to each eval set's per-model predictions (same recipe as the cadence analysis).

Usage:
    python scripts/analysis/eval_composition_split.py \
        --cells outputs/cadence/genomic_llm2x30 outputs/cadence/random_llm2x30 \
        --battery data/k562/test_sets_ag_s2_chrsplit \
        --sets genomic gc_matched random_32k motif_planted ood
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import ElasticNetCV

from albench.composition import (
    GENOMIC_GC_HI,
    GENOMIC_GC_LO,
    gc_content_batch,
    split_support_metrics,
)

# eval-set name -> (per-model npz pred key, labels.npz oracle key, battery file stem)
SET_SPEC = {
    "genomic": ("test_pred_genomic", "oracle_genomic", "genomic"),
    "gc_matched": ("test_pred_gc_matched", "oracle_gc_matched", "gc_matched"),
    "random_32k": ("test_pred_random_32k", "oracle_random_32k", "random_32k"),
    "motif_planted": ("test_pred_motif_planted", "oracle_motif_planted", "motif_planted"),
    "zoonomia": ("test_pred_zoonomia", "oracle_zoonomia", "zoonomia"),
    "ood": ("test_pred_ood", "oracle_ood", "ood"),
    "dinuc_shuffle": ("test_pred_dinuc_shuffle", "oracle_dinuc_shuffle", "dinuc_shuffle"),
    "gc_sweep": ("test_pred_gc_sweep", "oracle_gc_sweep", "gc_sweep"),
}


def _gc_cache(battery: Path, stem: str, cache: dict[str, np.ndarray]) -> np.ndarray | None:
    if stem in cache:
        return cache[stem]
    f = battery / f"{stem}_oracle.npz"
    if not f.exists():
        return None
    seqs = np.load(f, allow_pickle=True)["sequences"]
    gc = gc_content_batch(str(s) for s in seqs)
    cache[stem] = gc
    return gc


def _ensemble_weights(cell: Path, val_labels: np.ndarray) -> tuple[list[Path], np.ndarray, float]:
    """Fit ElasticNetCV on stacked val preds -> val_labels. Returns (metas, weights, intercept)."""
    npzs = sorted(glob.glob(str(cell / "r*_s0.npz")))
    cols, used = [], []
    for p in npzs:
        d = np.load(p, allow_pickle=True)
        if "val_pred" not in d:
            continue
        v = d["val_pred"]
        if v.shape != val_labels.shape or not np.all(np.isfinite(v)):
            continue
        cols.append(v)
        used.append(Path(p))
    if not cols:
        raise RuntimeError(f"no usable val preds in {cell}")
    X = np.column_stack(cols)
    enet = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 0.99], cv=3, n_jobs=2, max_iter=5000)
    enet.fit(X, val_labels)
    return used, enet.coef_, float(enet.intercept_)


def analyse_cell(cell: Path, battery: Path, sets: list[str], gc_cache: dict) -> dict:
    labels = np.load(cell / "labels.npz", allow_pickle=True)
    val_labels = labels["val_labels"].astype(np.float64)
    used, w, b = _ensemble_weights(cell, val_labels)

    # Preload each used model's npz once.
    preds = [np.load(p, allow_pickle=True) for p in used]
    out: dict[str, dict] = {}
    for s in sets:
        if s not in SET_SPEC:
            continue
        pred_key, label_key, stem = SET_SPEC[s]
        if label_key not in labels:
            continue
        gc = _gc_cache(battery, stem, gc_cache)
        if gc is None:
            continue
        # stack this set's per-model preds in the same model order as weights
        cols = []
        ok = True
        for d in preds:
            if pred_key not in d:
                ok = False
                break
            cols.append(d[pred_key].astype(np.float64))
        if not ok:
            continue
        ens = np.column_stack(cols) @ w + b
        target = labels[label_key].astype(np.float64)
        n = min(len(ens), len(target), len(gc))
        out[s] = split_support_metrics(ens[:n], target[:n], gc[:n])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--battery", default="data/k562/test_sets_ag_s2_chrsplit")
    ap.add_argument(
        "--sets", nargs="+", default=["genomic", "gc_matched", "random_32k", "motif_planted", "ood"]
    )
    ap.add_argument("--lo", type=float, default=GENOMIC_GC_LO)
    ap.add_argument("--hi", type=float, default=GENOMIC_GC_HI)
    ap.add_argument("--out", default=None, help="optional JSON dump path")
    args = ap.parse_args()

    battery = Path(args.battery)
    gc_cache: dict[str, np.ndarray] = {}
    results: dict[str, dict] = {}

    header = f"{'cell':<28} {'eval_set':<14} {'all_r':>7} {'inS_r':>7} {'ood_r':>7} {'ood_frac':>8} {'inS_n':>8}"
    print(f"\nsupport band GC in [{args.lo}, {args.hi}]")
    print(header)
    print("-" * len(header))
    for c in args.cells:
        cell = Path(c)
        name = cell.name
        try:
            res = analyse_cell(cell, battery, args.sets, gc_cache)
        except Exception as e:  # noqa: BLE001
            print(f"{name:<28} ERROR: {e}")
            continue
        results[name] = res
        for s, m in res.items():
            print(
                f"{name:<28} {s:<14} "
                f"{m['all']['pearson_r']:>7.4f} {m['in_support']['pearson_r']:>7.4f} "
                f"{m['ood']['pearson_r']:>7.4f} {m['ood']['frac']:>8.3f} {m['in_support']['n']:>8d}"
            )

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
