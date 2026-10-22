"""Standard ALBench figures: scaling curves (Pearson + MSE) and the eval matrix.

Reads outputs/figures/eval_matrix.csv (written by eval_matrix.py) and emits a
consistent figure set. Styling comes from albench.plotting so results produced by
different people drop into the same deck without restyling.

    python scripts/analysis/plot_scaling_results.py                    # all figures
    python scripts/analysis/plot_scaling_results.py --evals genomic ood
    python scripts/analysis/plot_scaling_results.py --matrix_csv mine.csv --out_dir figs/

Scaling panels are drawn twice on purpose:
  * raw     -- Pearson r (and MSE) against log D. What people expect to read.
  * scaled  -- (1 - r) and MSE on log-log, where a power law is a STRAIGHT line
               and the fitted slope is the scaling exponent. Curvature that the
               raw panel hides is obvious here, so a bad power-law fit cannot be
               reported as a clean one.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from albench.plotting import (apply_style, color_for, eval_label,
                              label_scale_axis, legend_by_family)

SUFFIX = ""
DEFAULT_EVALS = ["genomic", "ood", "snv_delta"]
# EXACT one-to-one reservoir/eval pairs. Each matched eval set was generated from
# ONE reservoir config, so only that arm has a true twin:
#   motif_planted <- motif_ct_enriched settings (ct_enriched vocab, cluster_at=0.99)
#   gc_matched    <- the gc_matched arm
#   zoonomia      <- zoonomia_relaxed settings (distance=all, clade=all, mcf=0.80)
# Sibling variants (motif_ct_dense, motif_jaspar, the EvoAug ladder) are NOT twins
# of these sets and must not be outlined as if they were -- scoring them against a
# sibling's twin is the confound this layout exists to avoid. They stay in the
# matrix as ordinary rows; giving each its own eval set is the proper fix.
DIAG_PAIRS = [
    ("random", "random_32k"),
    ("genomic", "genomic"),
    ("gc_matched", "gc_matched"),
    ("dinuc_shuffle", "dinuc_shuffle"),
    ("motif_ct_enriched", "motif_planted"),
    ("zoonomia_relaxed", "zoonomia"),
    ("mutagenesis", "sub_low"),
]
# eval sets with no reservoir twin at all, kept after the diagonal
UNMATCHED_COLS = ["sub_high", "ins_high", "del_high", "translocation", "inversion",
                  "ood", "snv_delta"]


def diag_order(arms):
    """(row_order, col_order). Paired arms first, in pair order, so their matched
    cell lands on the diagonal; every other arm follows, grouped by family."""
    rows, cols, used = [], [], set()
    for arm, col in DIAG_PAIRS:
        if arm in arms and arm not in used:
            rows.append(arm); used.add(arm)
            if col not in cols:
                cols.append(col)
    rest = sorted(a for a in arms if a not in used)
    rows.extend(rest)
    return rows, cols + [c for c in UNMATCHED_COLS if c not in cols]


HEATMAP_COLS = ["genomic", "motif_planted", "gc_matched", "zoonomia", "random_32k",
                "dinuc_shuffle", "sub_high", "ins_high", "del_high",
                "translocation", "inversion", "ood", "snv_delta"]


def load(csv_path, complete_only=True):
    rows = list(csv.DictReader(open(csv_path)))
    if complete_only:
        rows = [r for r in rows if r.get("complete") == "1"]
    return rows


def series(rows, evals):
    """{arm: {eval: {D: (r, mse)}}}"""
    out = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        for e in evals:
            v, m = r.get(e), r.get(f"{e}_mse")
            if v:
                out[r["arm"]][e][int(r["D"])] = (float(v), float(m) if m else np.nan)
    return out


def fit_exponent(D, err):
    """err = a * D^(-b); returns (b, r2) from OLS in log space."""
    x, y = np.log10(D), np.log10(np.maximum(err, 1e-9))
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return -b, (1 - ss_res / ss_tot if ss_tot > 0 else np.nan)


def plot_curves(data, evals, out_dir, metric):
    """metric: 'pearson' | 'mse'. Two rows: raw value, then power-law view."""
    idx = 0 if metric == "pearson" else 1
    fig, axes = plt.subplots(2, len(evals), figsize=(4.6 * len(evals), 8.0), squeeze=False)
    arms = sorted(data)
    for j, e in enumerate(evals):
        ax_raw, ax_pow = axes[0][j], axes[1][j]
        for arm in arms:
            pts = sorted(data[arm][e].items())
            if len(pts) < 2:
                continue
            D = np.array([p[0] for p in pts], float)
            v = np.array([p[1][idx] for p in pts], float)
            if not np.isfinite(v).all():
                continue
            c = color_for(arm)
            ax_raw.plot(D, v, "o-", color=c, label=arm)
            err = (1.0 - v) if metric == "pearson" else v
            ax_pow.plot(D, err, "o-", color=c, label=arm)
        label_scale_axis(ax_raw)
        ax_raw.set_ylabel(("Pearson r" if metric == "pearson" else "MSE") + f"  [{eval_label(e)}]")
        ax_raw.set_title(f"{eval_label(e)}: {'Pearson r' if metric=='pearson' else 'MSE'} vs scale")
        label_scale_axis(ax_pow)
        ax_pow.set_yscale("log")
        ax_pow.set_ylabel("1 - Pearson r" if metric == "pearson" else "MSE (log)")
        ax_pow.set_title(f"{eval_label(e)}: power-law view (straight = power law)")
    legend_by_family(axes[0][-1], arms, ncol=2, loc="best")
    fig.suptitle(f"Scaling of {'Pearson r' if metric=='pearson' else 'MSE'} "
                 f"— top: as measured; bottom: log-log, slope = scaling exponent", y=1.0)
    fig.tight_layout()
    p = Path(out_dir) / f"scaling_{metric}{SUFFIX}.png"
    fig.savefig(p); plt.close(fig)
    print(f"  wrote {p}")


def plot_heatmap(rows, D, out_dir, metric="pearson", diagonal=False, pairs_only=False):
    sub = [r for r in rows if int(r["D"]) == D]
    if len(sub) < 2:
        return
    if pairs_only:
        # every row has a twin and every column is somebody twin, so the matrix is
        # square and the diagonal is the whole question -- nothing else to scan
        present = {r["arm"] for r in sub}
        pairs = [(a, c) for a, c in DIAG_PAIRS if a in present]
        order = [a for a, _ in pairs]
        cols = [c for _, c in pairs]
        pos = {a: i for i, a in enumerate(order)}
        sub = [r for r in sub if r["arm"] in pos]
        sub.sort(key=lambda r: pos[r["arm"]])
    elif diagonal:
        order, cols = diag_order([r["arm"] for r in sub])
        cols = [c for c in cols if any(r.get(c) for r in sub)]
        pos = {a: i for i, a in enumerate(order)}
        sub.sort(key=lambda r: pos.get(r["arm"], 999))
    else:
        cols = [c for c in HEATMAP_COLS if any(r.get(c) for r in sub)]
        sub.sort(key=lambda r: -float(r.get("genomic") or 0))
    key = (lambda r, c: r.get(c)) if metric == "pearson" else (lambda r, c: r.get(f"{c}_mse"))
    M = np.array([[float(key(r, c)) if key(r, c) else np.nan for c in cols] for r in sub])
    cw, ch = (1.55, 0.92) if pairs_only else (0.95, 0.44)
    fig, ax = plt.subplots(figsize=(cw * len(cols) + 3.2, ch * len(sub) + 2.4))
    # z-score WITHIN each column: eval sets differ hugely in difficulty, so raw
    # shading would only ever show which eval is easy, not which arm is better.
    Z = (M - np.nanmean(M, 0)) / (np.nanstd(M, 0) + 1e-9)
    if metric == "mse":
        Z = -Z                                  # lower MSE is better -> keep red = good
    im = ax.imshow(Z, cmap="RdBu_r", vmin=-2.2, vmax=2.2, aspect="auto")
    tick_fs = 15 if pairs_only else 8
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([eval_label(c) for c in cols], rotation=35, ha="right", fontsize=tick_fs)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels([r["arm"] for r in sub], fontsize=tick_fs)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center",
                        fontsize=15 if pairs_only else 6.2,
                        color="white" if abs(Z[i, j]) > 1.3 else "black")
    ax.set_xlabel("eval set (test distribution)", fontsize=15 if pairs_only else 10)
    ax.set_ylabel("reservoir (training distribution)", fontsize=15 if pairs_only else 10)
    ax.set_title(f"D={D:,} added, base=0, deploy ensemble n=5\n"
                 f"cell = {'Pearson r' if metric=='pearson' else 'MSE'}; "
                 f"shading = z within column (red = better)", fontsize=15 if pairs_only else 11)
    if diagonal or pairs_only:
        # outline the matched pair in each row so the diagonal is unmistakable
        arm_of = [r["arm"] for r in sub]
        for i, a in enumerate(arm_of):
            if pairs_only:
                ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False,
                                           edgecolor="#111", linewidth=2.4))
                continue
            for pair_arm, col in DIAG_PAIRS:
                if a == pair_arm and col in cols:
                    j = cols.index(col)
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                               edgecolor="#111", linewidth=2.0))
                    break
    cb = fig.colorbar(im, ax=ax, shrink=0.75, label="z within eval set")
    if pairs_only:
        cb.ax.tick_params(labelsize=15); cb.set_label("z within eval set", size=15)
    fig.tight_layout()
    p = Path(out_dir) / f"heatmap_{metric}_d{D}{SUFFIX}.png"
    fig.savefig(p); plt.close(fig)
    print(f"  wrote {p}")


def write_table(rows, evals, out_dir):
    p = Path(out_dir) / f"scaling_table{SUFFIX}.md"
    with open(p, "w") as fh:
        fh.write("| arm | D | pool | " + " | ".join(f"{e} r | {e} MSE" for e in evals) + " |\n")
        fh.write("|" + "---|" * (3 + 2 * len(evals)) + "\n")
        for r in sorted(rows, key=lambda r: (r["arm"], int(r["D"]))):
            cells = []
            for e in evals:
                cells += [r.get(e, "") or "-", r.get(f"{e}_mse", "") or "-"]
            fh.write(f"| {r['arm']} | {int(r['D']):,} | {r['pool']} | " + " | ".join(cells) + " |\n")
    print(f"  wrote {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix_csv", default=str(REPO / "outputs/figures/eval_matrix.csv"))
    ap.add_argument("--out_dir", default=str(REPO / "outputs/figures"))
    ap.add_argument("--evals", nargs="*", default=DEFAULT_EVALS)
    ap.add_argument("--arms", nargs="*", default=None,
                    help="restrict to these arms (same subset is applied to curves AND heatmaps)")
    ap.add_argument("--pairs_only", action="store_true",
                    help="square matrix: only arms with a true twin, only their matched columns")
    ap.add_argument("--diagonal", action="store_true",
                    help="order both axes so matched reservoir/eval pairs sit on the diagonal")
    ap.add_argument("--suffix", default="", help="appended to output filenames")
    ap.add_argument("--include_partial", action="store_true",
                    help="include cells whose model pool is not full (NOT comparable)")
    args = ap.parse_args()

    global SUFFIX
    SUFFIX = args.suffix
    apply_style()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    rows = load(args.matrix_csv, complete_only=not args.include_partial)
    if args.arms:
        keep = set(args.arms)
        rows = [r for r in rows if r["arm"] in keep]
        missing = keep - {r["arm"] for r in rows}
        if missing:
            print(f"  [warn] requested arms with no complete data: {sorted(missing)}")
    if not rows:
        raise SystemExit(f"no rows in {args.matrix_csv} (all partial? try --include_partial)")
    data = series(rows, args.evals)

    for metric in ("pearson", "mse"):
        plot_curves(data, args.evals, args.out_dir, metric)
        for D in sorted({int(r["D"]) for r in rows}, reverse=True):
            plot_heatmap(rows, D, args.out_dir, metric, diagonal=args.diagonal, pairs_only=args.pairs_only)
    write_table(rows, args.evals, args.out_dir)

    print("\n  scaling exponents (err = a*D^-b):")
    for arm in sorted(data):
        for e in args.evals:
            pts = sorted(data[arm][e].items())
            if len(pts) < 3:
                continue
            D = np.array([p[0] for p in pts], float)
            b, r2 = fit_exponent(D, 1.0 - np.array([p[1][0] for p in pts], float))
            print(f"    {arm:<22} {e:<12} b={b:.3f}  R2={r2:.2f}")


if __name__ == "__main__":
    main()
