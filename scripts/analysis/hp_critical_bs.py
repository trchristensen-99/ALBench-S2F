"""Empirical critical batch size B_crit per D, using the behavioral definition:
the LARGEST batch size whose best-achievable val_pearson is within EPS of the
global per-D max. Below B_crit, raising bs is free (cheaper, same val).
At ~B_crit, val starts dropping. Cross-D comparison reveals B_crit(D) scaling.

Also extract two efficiency views:
  - epochs_to_target: median epochs_trained needed to reach target val per bs.
    If a bs needs MORE epochs (per-step learning is lower), it's beyond B_crit.
  - cost-frontier: (median train_time, top_val) per bs; the frontier knee is B_crit.

Pools all reservoirs at a given D (genomic + motif_planted_v2 + dinuc_shuffle)."""

import glob
import json
import os

import numpy as np

ROOT = "outputs/hp_step1_bakeoff_e100"
EPS = 0.005
BS_GRID = [16, 32, 64, 128, 256, 512, 1024]


def load_D(D):
    rows = []
    for f in glob.glob(os.path.join(ROOT, f"k562_*_d{D}/seed*/*/r*_meta.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        hp = d.get("hp", {}) or {}
        bs = hp.get("batch_size")
        if bs is None:
            continue
        rows.append(
            dict(
                bs=int(bs),
                t=float(d.get("train_time_sec", 0)),
                ep=int(d.get("epochs_trained", 0)),
                val=float(d.get("val_pearson", float("nan"))),
            )
        )
    return rows


def per_bs(rows, key, fn):
    out = {}
    for bs in BS_GRID:
        sel = [r for r in rows if r["bs"] == bs and np.isfinite(r[key])]
        if not sel:
            continue
        out[bs] = float(fn([r[key] for r in sel]))
    return out


def b_crit(rows, eps=EPS):
    """B_crit = largest bs whose top val is within eps of the per-D max top val."""
    top = per_bs(rows, "val", lambda v: max(v))
    if not top:
        return None, {}
    gmax = max(top.values())
    keepers = [bs for bs in sorted(top) if top[bs] >= gmax - eps]
    return (max(keepers) if keepers else None), top


def analyze(D):
    rows = load_D(D)
    if not rows:
        print(f"\n=== D={D}: no rows")
        return
    top = per_bs(rows, "val", max)
    p90 = per_bs(rows, "val", lambda v: np.percentile(v, 90))
    med_t = per_bs(rows, "t", np.median)
    med_ep = per_bs(rows, "ep", np.median)
    n = per_bs(rows, "t", len)
    bcrit, _ = b_crit(rows)
    print(f"\n=== D = {D}  (n={len(rows)})  B_crit ≈ {bcrit}  (top within {EPS})")
    print(
        f"  {'bs':>5s} {'n':>5s} {'top':>7s} {'p90':>7s} {'med_t':>7s} {'med_ep':>7s} {'top_efficiency':>16s}"
    )
    # efficiency = top_val / median_time  (higher = better val per second)
    for bs in BS_GRID:
        if bs not in top:
            continue
        eff = top[bs] / med_t[bs] if med_t.get(bs, 0) > 0 else float("nan")
        mark = " ←" if bs == bcrit else ""
        print(
            f"  {bs:>5d} {int(n[bs]):>5d} {top[bs]:>7.4f} {p90[bs]:>7.4f} {med_t[bs]:>7.0f} {med_ep[bs]:>7.0f} {eff * 1e4:>16.3f}{mark}"
        )
    return bcrit


def main():
    bcrits = {}
    for D in [30000, 300000]:
        b = analyze(D)
        if b:
            bcrits[D] = b

    if len(bcrits) >= 2:
        Ds = sorted(bcrits)
        b0, b1 = bcrits[Ds[0]], bcrits[Ds[1]]
        if b0 and b1 and b1 > b0:
            alpha = np.log(b1 / b0) / np.log(Ds[1] / Ds[0])
            print(f"\n=== Scaling: B_crit({Ds[0]}) = {b0}, B_crit({Ds[1]}) = {b1}")
            print(f"  Fit  B_crit(D) ∝ D^alpha  →  alpha = {alpha:.3f}")
            print(f"  Projection (rounded to power-of-2):")
            for D in [10000, 30000, 100000, 300000, 1_000_000, 3_000_000]:
                proj = b0 * (D / Ds[0]) ** alpha
                p2 = 2 ** int(round(np.log2(proj)))
                print(f"    D={D:>9d}  B_crit ≈ {proj:7.1f}  → {p2}")
            print("\n  WARNING — based on only 2 D anchors; need 100k + 1M to validate slope.")


if __name__ == "__main__":
    main()
