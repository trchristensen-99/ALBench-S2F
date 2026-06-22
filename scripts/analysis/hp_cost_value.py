"""Are the slow HP configs actually winning val? Pareto-style scan across all
D=300k models: bin by batch_size and depth, report mean train_time and best val
Pearson per bin so we can decide which axis ranges are dominated (slow AND
not better) and could be pruned from the HP space."""

import glob
import json
import os
import sys

import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "300000"
ROOT = f"outputs/hp_step1_bakeoff_e100/k562_genomic_d{D}"
print(f"=== D = {D} ===")


def all_rows():
    out = []
    for f in glob.glob(os.path.join(ROOT, "seed*", "*", "r*_meta.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        hp = d.get("hp", {}) or {}
        bs = d.get("block_sizes", hp.get("block_sizes"))
        out.append(
            dict(
                t=d.get("train_time_sec", 0),
                ep=d.get("epochs_trained", 0),
                val=d.get("val_pearson", float("nan")),
                bs_train=hp.get("batch_size"),
                depth=len(bs) if isinstance(bs, list) else None,
                params=int(sum(bs)) if isinstance(bs, list) else None,
                lr=hp.get("lr") or hp.get("learning_rate"),
                strat=os.path.basename(os.path.dirname(f)),
            )
        )
    return out


def bin_report(rows, key, bins, label):
    print(f"\n=== {label}  (n={len(rows)} models)")
    print(
        f"  {'bin':<14s} {'n':>4s} {'mean_t_s':>9s} {'p90_t_s':>9s} {'top_val':>8s} {'mean_val':>9s}"
    )
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = [r for r in rows if r.get(key) is not None and lo <= r[key] < hi]
        if not sel:
            continue
        t = np.array([r["t"] for r in sel])
        v = np.array([r["val"] for r in sel])
        v = v[np.isfinite(v)]
        rng = f"[{lo},{hi})"
        print(
            f"  {rng:<14s} {len(sel):>4d} {t.mean():>9.0f} {np.percentile(t, 90):>9.0f} {v.max() if len(v) else float('nan'):>8.4f} {v.mean() if len(v) else float('nan'):>9.4f}"
        )


def main():
    rows = all_rows()
    bin_report(rows, "bs_train", [16, 33, 65, 129, 257, 513, 1025, 4097], "by batch_size")
    bin_report(rows, "depth", [1, 3, 5, 7, 9, 11, 13], "by depth (block count)")
    bin_report(
        rows, "params", [0, 100, 200, 400, 800, 1600, 5000], "by param-proxy (sum of block sizes)"
    )
    print("\n=== top-10 val_pearson models — are they slow? ===")
    rows.sort(key=lambda r: -(r["val"] if np.isfinite(r["val"]) else -1))
    print(
        f"  {'rank':>4s} {'val':>7s} {'t_s':>7s} {'ep':>4s} {'bs':>5s} {'depth':>6s} {'strat':>20s}"
    )
    for i, r in enumerate(rows[:10]):
        print(
            f"  {i + 1:>4d} {r['val']:>7.4f} {r['t']:>7.0f} {r['ep']:>4d} {str(r['bs_train']):>5s} {str(r['depth']):>6s} {r['strat']:>20s}"
        )
    print("\n=== bottom: how SLOW for how little gain? ===")
    finite = [r for r in rows if np.isfinite(r["val"])]
    finite.sort(key=lambda r: -r["t"])
    print(f"  10 slowest models:")
    print(f"  {'val':>7s} {'t_s':>7s} {'ep':>4s} {'bs':>5s} {'depth':>6s} {'strat':>20s}")
    for r in finite[:10]:
        print(
            f"  {r['val']:>7.4f} {r['t']:>7.0f} {r['ep']:>4d} {str(r['bs_train']):>5s} {str(r['depth']):>6s} {r['strat']:>20s}"
        )


if __name__ == "__main__":
    main()
