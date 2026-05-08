"""High-dimensional HP grid audit:
1. Aggregate all preflight results into single DataFrame.
2. Per (arch, D), find best config + check for edge optima.
3. Find sparse cells in (arch × D × HP) grid.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")
RESULTS = REPO / "results/preflight"


def _load_result(p: Path):
    try:
        d = json.loads(p.read_text())
    except Exception:
        return None
    if "arch" not in d:
        return None
    hp = d.get("hp", {})
    row = {
        "path": str(p.relative_to(REPO)),
        "task": p.relative_to(RESULTS).parts[0],
        "arch": d["arch"],
        "d_train": d.get("d_train"),
        "seed": d.get("seed"),
        "epochs": d.get("epochs"),
        "aug": d.get("augmentations"),
        "best_val_mse": d.get("best_val_mse"),
        "test_mse": d.get("test_mse_at_best_val"),
        "best_epoch": d.get("best_epoch"),
        "n_params": d.get("n_params"),
        "lr": hp.get("lr"),
        "batch_size": hp.get("batch_size"),
        "weight_decay": hp.get("weight_decay"),
        "in_channels": hp.get("in_channels"),
        "hidden_dim": hp.get("hidden_dim"),
        "cnn_filters": hp.get("cnn_filters"),
        "dropout_cnn": hp.get("dropout_cnn"),
        "dropout_lstm": hp.get("dropout_lstm"),
        "dropout": hp.get("dropout") or hp.get("dropout_rate"),
        "k": hp.get("k"),
    }
    return row


rows = []
for p in RESULTS.rglob("result.json"):
    r = _load_result(p)
    if r and r.get("test_mse") is not None:
        rows.append(r)

df = pd.DataFrame(rows)
print(f"Loaded {len(df)} HP results across {df.arch.nunique()} architectures")
print(f"  archs: {sorted(df.arch.unique())}")
print(f"  D values: {sorted(int(d) for d in df.d_train.dropna().unique())}")

out = RESULTS / "hp_grid_audit.csv"
df.to_csv(out, index=False)
print(f"\nSaved {out}\n")

# ── Best per (arch, D) ─────────────────────────────────────────────────────
print("=== Best test_mse per (arch, D) ===")
best_idx = df.groupby(["arch", "d_train"])["test_mse"].idxmin()
best = df.loc[best_idx].sort_values(["arch", "d_train"])
for arch in sorted(best.arch.unique()):
    print(f"\n  {arch}:")
    sub = best[best.arch == arch]
    for _, row in sub.iterrows():
        d = int(row.d_train) if not pd.isna(row.d_train) else "—"
        hp_summary = []
        for k in [
            "lr",
            "batch_size",
            "dropout_cnn",
            "dropout_lstm",
            "dropout",
            "weight_decay",
            "aug",
        ]:
            v = row.get(k)
            if v is not None and not pd.isna(v):
                hp_summary.append(f"{k}={v}")
        print(f"    D={d:>7}: test_mse={row.test_mse:.4f}  ({', '.join(hp_summary)})")

# ── Coverage gaps ──────────────────────────────────────────────────────────
print("\n=== Coverage gaps: (arch × D) cells with <5 configs ===")
counts = df.groupby(["arch", "d_train"]).size().reset_index(name="n")
sparse = counts[counts.n < 5].sort_values(["arch", "d_train"])
for _, row in sparse.iterrows():
    d = int(row.d_train) if not pd.isna(row.d_train) else "—"
    print(f"  {row.arch:<15} D={d:>7}: only {row.n} configs")

# ── Edge analysis: best at grid boundary ───────────────────────────────────
print("\n=== Edge analysis: best HP at grid boundary per (arch, D) ===")
EDGE_HPS = ["lr", "batch_size", "dropout_cnn", "dropout_lstm", "dropout", "weight_decay"]
for arch in sorted(df.arch.unique()):
    for d in sorted(df[df.arch == arch].d_train.dropna().unique()):
        sub = df[(df.arch == arch) & (df.d_train == d)]
        if len(sub) < 4:
            continue
        bidx = sub["test_mse"].idxmin()
        best_row = sub.loc[bidx]
        edges_found = []
        for hp in EDGE_HPS:
            if hp not in sub.columns:
                continue
            vals = sorted(sub[hp].dropna().unique())
            if len(vals) < 2:
                continue
            v = best_row.get(hp)
            if v is None or pd.isna(v):
                continue
            try:
                idx = list(vals).index(v)
            except ValueError:
                continue
            if idx == 0:
                edges_found.append(f"{hp}={v} (LOW edge, grid={vals})")
            elif idx == len(vals) - 1:
                edges_found.append(f"{hp}={v} (HIGH edge, grid={vals})")
        if edges_found:
            print(f"\n  {arch}, D={int(d):>7} (best test_mse={best_row.test_mse:.4f}):")
            for e in edges_found:
                print(f"    EDGE: {e}")
