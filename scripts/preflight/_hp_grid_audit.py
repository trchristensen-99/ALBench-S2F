"""High-dimensional HP grid audit:
1. Aggregate all preflight results into single DataFrame.
2. Per (arch, D), find best config + check for edge optima.
3. Find sparse cells in (arch × D × HP) grid.
4. Detect HP couplings (pairs that depend on each other).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")
RESULTS = REPO / "results/preflight"


def _load_result(p: Path):
    try:
        d = json.loads(p.read_text())
    except Exception:
        return None
    arch = d.get("arch") or d.get("model_name")
    if not arch:
        return None
    # Extract metrics
    mr = d.get("metrics") or d.get("test_metrics") or d
    pearson = mr.get("test_pearson") or mr.get("test_pearson_r") or mr.get("pearson")
    if pearson is None and isinstance(mr, dict):
        for k, v in mr.items():
            if isinstance(v, dict) and "pearson_r" in v:
                pearson = v.get("pearson_r")
                break
    if pearson is None:
        return None
    row = {
        "path": str(p.relative_to(REPO)),
        "task": p.relative_to(RESULTS).parts[0],
        "arch": arch,
        "d_train": d.get("d_train"),
        "seed": d.get("seed"),
        "lr": d.get("lr") or d.get("learning_rate"),
        "batch_size": d.get("batch_size"),
        "dropout": d.get("dropout") or d.get("dropout_rate"),
        "weight_decay": d.get("weight_decay") or d.get("wd"),
        "aug": d.get("aug") or d.get("augmentation"),
        "epochs": d.get("epochs"),
        "test_pearson": pearson,
    }
    # Capture all HP fields
    for k, v in d.items():
        if k in ("metrics", "test_metrics", "history", "predictions", "best_val"):
            continue
        if isinstance(v, (int, float, str, bool, type(None))) and k not in row:
            row[k] = v
    return row


rows = []
for p in RESULTS.rglob("result.json"):
    r = _load_result(p)
    if r:
        rows.append(r)

df = pd.DataFrame(rows)
print(f"Loaded {len(df)} HP results across {df.arch.nunique()} architectures")
print(f"  archs: {sorted(df.arch.unique())}")
print(f"  D values: {sorted(df.d_train.dropna().unique())}")

out = RESULTS / "hp_grid_audit.csv"
df.to_csv(out, index=False)
print(f"\nSaved {out}")

# ── Best per (arch, D) ─────────────────────────────────────────────────────
print("\n=== Best per (arch, D) [test_pearson] ===")
by_ad = df.groupby(["arch", "d_train"])
best = by_ad["test_pearson"].max().reset_index().sort_values(["arch", "d_train"])
for arch in best.arch.unique():
    print(f"\n  {arch}:")
    sub = best[best.arch == arch]
    for _, row in sub.iterrows():
        d = int(row.d_train) if not pd.isna(row.d_train) else "—"
        print(f"    D={d:>7}: best test_pearson = {row.test_pearson:.4f}")

# ── HP grid coverage gaps ──────────────────────────────────────────────────
print("\n=== Coverage gaps: (arch, D) with <5 configs tested ===")
counts = by_ad.size().reset_index(name="n_configs").sort_values("n_configs")
sparse = counts[counts.n_configs < 5]
for _, row in sparse.iterrows():
    arch = row.arch
    d = int(row.d_train) if not pd.isna(row.d_train) else "—"
    print(f"  {arch}, D={d}: only {row.n_configs} configs tested")

# ── Edge analysis: best configs at grid boundary ──────────────────────────
print("\n=== Edge analysis (best config at grid edge per arch) ===")
for arch in df.arch.unique():
    sub = df[df.arch == arch]
    best_idx = sub["test_pearson"].idxmax()
    best_row = sub.loc[best_idx]
    print(f"\n  {arch}: test_pearson={best_row.test_pearson:.4f}")
    for hp in ["lr", "batch_size", "dropout", "weight_decay", "aug"]:
        if hp not in sub.columns or sub[hp].isna().all():
            continue
        vals = sorted(sub[hp].dropna().unique())
        if len(vals) < 2:
            continue
        v = best_row.get(hp)
        if v is None or pd.isna(v):
            continue
        try:
            idx = vals.index(v) if v in vals else None
        except Exception:
            idx = None
        if idx is None:
            continue
        at_edge = idx == 0 or idx == len(vals) - 1
        marker = " ← EDGE" if at_edge else ""
        print(f"    {hp}: best={v}  (grid={vals}){marker}")
