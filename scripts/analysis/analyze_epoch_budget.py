"""Attribute best_epoch across the epoch-budget diagnostic to decide whether the current
epochs=60 / patience=10 budget is (roughly) optimal.

Reads every outputs/epoch_diagnostic/<reservoir>/d<D>/seed*/r00_random_*_meta.json and
tabulates best_epoch / epochs_trained / early_stopped against the HP axes (lr, capacity,
block_class, batch, optimizer) plus reservoir and D.

The budget is too small if many runs are CENSORED — i.e. they ran the full 60 epochs
without early-stopping (early_stopped=False) and their best epoch sits near the ceiling,
meaning val performance might still have been improving. The budget is comfortable if most
runs early-stop well before 60 and the censored fraction is small.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DIAG = REPO / "outputs/epoch_diagnostic"
EPOCHS = 60
PATIENCE = 10
EDGE = EPOCHS - PATIENCE  # best_epoch at/after this had < patience epochs of headroom


def load_rows():
    rows = []
    for f in glob.glob(str(DIAG / "*/d*/seed*/r00_random_*_meta.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if "best_epoch" not in d:
            continue
        hp = d.get("hp", {})
        rel = Path(f).relative_to(DIAG)
        rows.append(
            {
                "reservoir": rel.parts[0],
                "D": int(rel.parts[1][1:]),
                "best_epoch": int(d["best_epoch"]),
                "epochs_trained": int(d["epochs_trained"]),
                "early_stopped": bool(d["early_stopped"]),
                "best_val_pearson": float(d.get("best_val_pearson", np.nan)),
                "lr": hp.get("lr"),
                "n_layers": hp.get("n_layers"),
                "width_base": hp.get("width_base"),
                "block_class": hp.get("block_class"),
                "batch_size": hp.get("batch_size"),
                "optimizer": hp.get("optimizer"),
            }
        )
    return rows


def _fmt(vals):
    a = np.array(vals, dtype=float)
    return f"n={len(a):>3d}  median={np.median(a):>4.1f}  mean={a.mean():>4.1f}  p90={np.percentile(a, 90):>4.1f}  max={a.max():>4.0f}"


def _censored(rows):
    # ran the full budget (not early-stopped) → best_epoch may be censored at the ceiling
    return [r for r in rows if not r["early_stopped"]]


def group_report(rows, key, order=None):
    keys = order or sorted({r[key] for r in rows}, key=lambda x: (x is None, x))
    print(f"\n── best_epoch by {key} ──")
    for k in keys:
        sub = [r for r in rows if r[key] == k]
        if not sub:
            continue
        be = [r["best_epoch"] for r in sub]
        es = sum(r["early_stopped"] for r in sub)
        cen = len(_censored(sub))
        print(
            f"  {str(k):>10}: {_fmt(be)}  early_stop={es}/{len(sub)} ({100 * es / len(sub):>3.0f}%)  ran_full_60={cen}"
        )


def lr_bin(lr):
    if lr is None:
        return None
    if lr < 5e-4:
        return "lo(<5e-4)"
    if lr < 2e-3:
        return "mid"
    return "hi(>=2e-3)"


def main():
    rows = load_rows()
    if not rows:
        print(f"no meta files under {DIAG}")
        return
    n = len(rows)
    be_all = [r["best_epoch"] for r in rows]
    es_all = sum(r["early_stopped"] for r in rows)
    censored = _censored(rows)
    edge_runs = [r for r in rows if r["best_epoch"] >= EDGE]

    print(f"=== Epoch-budget diagnostic: {n} configs (budget={EPOCHS}, patience={PATIENCE}) ===")
    print(f"\nOverall best_epoch: {_fmt(be_all)}")
    print(f"early_stopped: {es_all}/{n} ({100 * es_all / n:.0f}%)")
    print(
        f"ran full {EPOCHS} without early-stop (CENSORED candidates): {len(censored)}/{n} "
        f"({100 * len(censored) / n:.0f}%)"
    )
    print(
        f"best_epoch >= {EDGE} (within patience of the ceiling): {len(edge_runs)}/{n} "
        f"({100 * len(edge_runs) / n:.0f}%)"
    )
    if censored:
        print(f"  └ among censored runs, best_epoch: {_fmt([r['best_epoch'] for r in censored])}")

    group_report(rows, "D", order=sorted({r["D"] for r in rows}))
    group_report(rows, "reservoir")
    group_report(rows, "optimizer")
    group_report(rows, "block_class")

    # binned numeric axes
    print("\n── best_epoch by lr bin ──")
    for b in ["lo(<5e-4)", "mid", "hi(>=2e-3)"]:
        sub = [r for r in rows if lr_bin(r["lr"]) == b]
        if sub:
            print(f"  {b:>10}: {_fmt([r['best_epoch'] for r in sub])}")
    print("\n── best_epoch by n_layers ──")
    for k in sorted({r["n_layers"] for r in rows if r["n_layers"] is not None}):
        sub = [r for r in rows if r["n_layers"] == k]
        print(f"  {k:>10}: {_fmt([r['best_epoch'] for r in sub])}")
    print("\n── best_epoch by batch_size ──")
    for k in sorted({r["batch_size"] for r in rows if r["batch_size"] is not None}):
        sub = [r for r in rows if r["batch_size"] == k]
        print(f"  {k:>10}: {_fmt([r['best_epoch'] for r in sub])}")

    # verdict
    cen_frac = len(censored) / n
    p90 = np.percentile(be_all, 90)
    print("\n=== VERDICT ===")
    print(
        f"censored fraction = {100 * cen_frac:.0f}%  |  p90 best_epoch = {p90:.0f}  |  ceiling = {EPOCHS}"
    )
    if cen_frac <= 0.15 and p90 <= EPOCHS - PATIENCE:
        print("→ Budget looks COMFORTABLE: most runs early-stop with margin; little censoring.")
    elif cen_frac <= 0.30:
        print("→ Budget is BORDERLINE: a non-trivial minority run the full budget — inspect the")
        print("  axes above (esp. D=300k / low-lr / large-batch) for systematic censoring.")
    else:
        print("→ Budget likely TOO SMALL: many runs hit the ceiling without early-stopping —")
        print("  consider raising epochs and/or patience for the censored regimes.")

    # per-D censored fraction (the decisive split)
    print("\nPer-D censored fraction:")
    for D in sorted({r["D"] for r in rows}):
        sub = [r for r in rows if r["D"] == D]
        c = len(_censored(sub))
        print(f"  D={D:>7}: {c}/{len(sub)} ran full {EPOCHS} ({100 * c / len(sub):.0f}%)")


if __name__ == "__main__":
    main()
