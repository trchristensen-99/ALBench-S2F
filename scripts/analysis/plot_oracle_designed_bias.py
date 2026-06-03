#!/usr/bin/env python3
"""Compare two AG_S2 oracle ensembles to quantify designed-sequence bias.

Both oracles score the *same* chr-split test/control battery (produced by
`scripts/generate_ag_s2_test_labels.py --oracle-dir ...`), so predictions align
row-for-row and we can compute paired per-sequence deltas.

  canonical  = oracle trained WITH the designed high-activity sequences
               (default: outputs/oracle_full856k_clean/s2 scores)
  compare    = oracle trained WITHOUT them
               (default: outputs/oracle_no_designed/s2 scores)

For each set (genomic in-dist, ood designed, random) it produces, per set:
  • overlaid oracle_mean histograms (canonical vs compare)
  • paired-delta histogram  (compare - canonical)
  • scatter canonical vs compare
For SNV it compares delta_mean (alt-ref) distributions.

Outputs (default results/diagnostics/oracle_designed_bias/):
  designed_bias_summary.csv   — per-set paired stats (mean/abs delta, corr, bias dir)
  oracle_designed_bias.png    — multi-panel figure

Run::

    python scripts/analysis/plot_oracle_designed_bias.py \
        --canonical-dir <dir_with_full856k_clean_scores> \
        --compare-dir   <dir_with_no_designed_scores>
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parents[2]

# (filename, value-key) for the row-aligned single-prediction sets.
POINT_SETS = [
    ("genomic in-dist", "genomic_oracle.npz", "oracle_mean"),
    ("ood designed", "ood_oracle.npz", "oracle_mean"),
    ("random 10k", "random_10k_oracle.npz", "oracle_mean"),
]


def _load_vec(d: Path, fname: str, key: str) -> np.ndarray | None:
    p = d / fname
    if not p.exists():
        return None
    with np.load(p, allow_pickle=True) as z:
        if key not in z:
            return None
        return np.asarray(z[key], dtype=np.float64)


def _paired_stats(canon: np.ndarray, comp: np.ndarray) -> dict:
    n = min(len(canon), len(comp))
    canon, comp = canon[:n], comp[:n]
    finite = np.isfinite(canon) & np.isfinite(comp)
    canon, comp = canon[finite], comp[finite]
    delta = comp - canon  # compare(no_designed) minus canonical(designed-incl)
    r = (
        pearsonr(canon, comp)[0]
        if canon.size >= 2 and canon.std() > 0 and comp.std() > 0
        else float("nan")
    )
    return {
        "n": int(canon.size),
        "canon_mean": float(canon.mean()) if canon.size else float("nan"),
        "compare_mean": float(comp.mean()) if comp.size else float("nan"),
        "mean_delta": float(delta.mean()) if delta.size else float("nan"),
        "mean_abs_delta": float(np.abs(delta).mean()) if delta.size else float("nan"),
        "p90_abs_delta": float(np.quantile(np.abs(delta), 0.90)) if delta.size else float("nan"),
        "pearson_r": float(r),
        "frac_canon_higher": float((delta < 0).mean()) if delta.size else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical-dir",
        type=Path,
        default=REPO / "outputs" / "oracle_bias_compare" / "full856k_clean",
        help="Dir with *_oracle.npz scored by the designed-INCLUDED oracle.",
    )
    parser.add_argument(
        "--compare-dir",
        type=Path,
        default=REPO / "outputs" / "oracle_bias_compare" / "no_designed",
        help="Dir with *_oracle.npz scored by the no-designed oracle.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "results" / "diagnostics" / "oracle_designed_bias",
    )
    parser.add_argument("--canon-label", default="full856k_clean (designed-incl)")
    parser.add_argument("--compare-label", default="no_designed")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── collect aligned vectors per set ──────────────────────────────────
    rows, panels = [], []
    for label, fname, key in POINT_SETS:
        canon = _load_vec(args.canonical_dir, fname, key)
        comp = _load_vec(args.compare_dir, fname, key)
        if canon is None or comp is None:
            print(f"  SKIP {label}: missing {fname} in one dir")
            continue
        stats = _paired_stats(canon, comp)
        stats["set"] = label
        rows.append(stats)
        panels.append((label, canon, comp))

    # SNV delta (alt - ref) comparison, if present.
    snv_canon = _load_vec(args.canonical_dir, "snv_oracle.npz", "delta_mean")
    snv_comp = _load_vec(args.compare_dir, "snv_oracle.npz", "delta_mean")
    if snv_canon is not None and snv_comp is not None:
        stats = _paired_stats(snv_canon, snv_comp)
        stats["set"] = "snv delta (alt-ref)"
        rows.append(stats)
        panels.append(("snv delta (alt-ref)", snv_canon, snv_comp))

    if not rows:
        raise SystemExit(
            f"No comparable *_oracle.npz found in {args.canonical_dir} and {args.compare_dir}"
        )

    # ── summary CSV ──────────────────────────────────────────────────────
    csv_path = args.out_dir / "designed_bias_summary.csv"
    fields = [
        "set",
        "n",
        "canon_mean",
        "compare_mean",
        "mean_delta",
        "mean_abs_delta",
        "p90_abs_delta",
        "pearson_r",
        "frac_canon_higher",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print("wrote", csv_path)

    # ── figure: 3 columns (hist overlay | paired delta | scatter) × n sets ─
    n = len(panels)
    fig, axes = plt.subplots(n, 3, figsize=(15, 3.6 * n), squeeze=False)
    for i, (label, canon, comp) in enumerate(panels):
        m = min(len(canon), len(comp))
        c, p = canon[:m], comp[:m]
        fin = np.isfinite(c) & np.isfinite(p)
        c, p = c[fin], p[fin]
        delta = p - c

        ax0 = axes[i][0]
        lo = float(min(c.min(), p.min())) if c.size else 0.0
        hi = float(max(c.max(), p.max())) if c.size else 1.0
        bins = np.linspace(lo, hi, 60)
        ax0.hist(c, bins=bins, alpha=0.55, label=args.canon_label, color="#3B6CB7")
        ax0.hist(p, bins=bins, alpha=0.55, label=args.compare_label, color="#C2185B")
        ax0.set_title(f"{label} — label distribution")
        ax0.set_xlabel("oracle prediction")
        ax0.set_ylabel("count")
        ax0.legend(fontsize=8)

        ax1 = axes[i][1]
        ax1.hist(delta, bins=60, color="#7B3FA0", alpha=0.8)
        ax1.axvline(0, color="k", lw=1, ls="--")
        ax1.axvline(
            float(delta.mean()), color="#E08214", lw=1.5, label=f"mean Δ={delta.mean():.3f}"
        )
        ax1.set_title(f"{label} — paired Δ ({args.compare_label} − canonical)")
        ax1.set_xlabel("Δ prediction")
        ax1.set_ylabel("count")
        ax1.legend(fontsize=8)

        ax2 = axes[i][2]
        ax2.scatter(c, p, s=3, alpha=0.25, color="#2E7D32")
        lim = [lo, hi]
        ax2.plot(lim, lim, "k--", lw=1)
        r = pearsonr(c, p)[0] if c.size >= 2 and c.std() > 0 and p.std() > 0 else float("nan")
        ax2.set_title(f"{label} — r={r:.4f}")
        ax2.set_xlabel(args.canon_label)
        ax2.set_ylabel(args.compare_label)

    fig.suptitle(
        "Designed-sequence oracle bias: WITH vs WITHOUT designed high-activity seqs",
        fontsize=13,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    png = args.out_dir / "oracle_designed_bias.png"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(args.out_dir / "oracle_designed_bias.pdf", bbox_inches="tight")
    print("wrote", png)


if __name__ == "__main__":
    main()
