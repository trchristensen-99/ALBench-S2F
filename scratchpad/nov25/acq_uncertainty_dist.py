"""Activity of uncertainty-selected sequences, against the two controls that explain it.

fig14 showed the selected corpora shifting upward in activity as the candidate pool grows.
That is only interesting if uncertainty is doing something other than ranking by activity,
and two reference curves decide it, both on the SAME activity bins so the panels are read
together:

  the uncertainty-vs-activity trend   mean bank uncertainty for sequences in each activity
                                      bin. Labels are log2FC and regression error grows with
                                      magnitude, so if this rises monotonically then ranking
                                      by raw ensemble spread partly re-ranks by activity and
                                      the fig14 shift is largely mechanical.
  the genomic activity distribution   what real CREs look like. The selected corpora are
                                      only "too active" relative to something, and the
                                      natural target is the distribution we actually want a
                                      model to work on.

Needs one GPU pass to score the candidate pool with the bank; everything else is histograms.

    python scratchpad/nov25/acq_uncertainty_dist.py --out outputs/figures/nov25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

POOLS = REPO / "outputs/acq52/pools"
NESTED = REPO / "outputs/grid_pools/k562_nested"
A_SIZES = [30_000, 60_000, 100_000, 300_000, 1_000_000]
BLUE, VERM, GREEN, ORANGE, GREY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#6b6b6b"
# One binning for every panel, so the histograms and the trend line share an x axis.
BINS = np.linspace(-3.0, 8.0, 45)
CENTRES = 0.5 * (BINS[:-1] + BINS[1:])

plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.18,
        "legend.frameon": False,
        "savefig.bbox": "tight",
    }
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/figures/nov25")
    ap.add_argument("--bank", default="outputs/wed41/bench38_ln_e500_wd2_30k_c20")
    ap.add_argument("--trend-pool", default="motif_shared_core_d300000.npz")
    ap.add_argument("--trend-n", type=int, default=200_000)
    a = ap.parse_args()
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)

    # --- selected corpora -----------------------------------------------------
    sel: list[tuple[int, np.ndarray]] = []
    for sz in A_SIZES:
        f = POOLS / f"poolA_{sz}_d30000.npz"
        if f.exists():
            sel.append((sz, np.asarray(np.load(f, allow_pickle=True)["oracle_labels"], float)))
    if not sel:
        print("no selected corpora yet")
        return 1

    # --- control 1: genomic activity -------------------------------------------
    gz = np.load(NESTED / "genomic_d300000.npz", allow_pickle=True)
    gen_y = np.asarray(gz["oracle_labels"], float)

    # --- control 2: bank uncertainty against activity ---------------------------
    from albench.acquisition.legnet_bank import LegNetBank

    z = np.load(NESTED / a.trend_pool, allow_pickle=True)
    seqs = [str(s) for s in z["sequences"]]
    y = np.asarray(z["oracle_labels"], float)
    if len(seqs) > a.trend_n:
        idx = np.sort(np.random.default_rng(0).choice(len(seqs), a.trend_n, replace=False))
        seqs = [seqs[i] for i in idx]
        y = y[idx]
    bank = LegNetBank(str(REPO / a.bank))
    unc = np.asarray(bank.uncertainty(seqs), float).ravel()
    print(
        f"scored {len(seqs):,} for the trend; pearson(unc, activity) = "
        f"{np.corrcoef(unc, y)[0, 1]:+.4f}",
        flush=True,
    )

    b = np.digitize(y, BINS) - 1
    ok = (b >= 0) & (b < len(CENTRES))
    trend = np.full(len(CENTRES), np.nan)
    for i in range(len(CENTRES)):
        m = ok & (b == i)
        if m.sum() >= 50:
            trend[i] = unc[m].mean()

    # --- figure ----------------------------------------------------------------
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.4, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1], "hspace": 0.12},
    )

    ax = axes[0]
    shades = plt.cm.Blues(np.linspace(0.35, 0.95, len(sel)))
    for (sz, ys), colour in zip(sel, shades, strict=True):
        h, _ = np.histogram(ys, bins=BINS, density=True)
        ax.plot(CENTRES, h, "-", color=colour, lw=1.9, label=f"selected from {sz // 1000}k")
    hg, _ = np.histogram(gen_y, bins=BINS, density=True)
    ax.plot(CENTRES, hg, "--", color=VERM, lw=2.1, label="genomic CREs (control)")
    ax.set_ylabel("density")
    ax.set_title("(a) activity of the selected 30k, against real CREs")
    ax.legend(fontsize=8, ncols=2)

    ax = axes[1]
    ax.plot(CENTRES, trend, "o-", color=GREEN, ms=4)
    ax.set_xlabel("oracle activity (log2FC)")
    ax.set_ylabel("mean bank uncertainty")
    ax.set_title("(b) control: is uncertainty just activity?")
    ax.annotate(
        f"pearson(uncertainty, activity) = {np.corrcoef(unc, y)[0, 1]:+.3f}",
        xy=(0.98, 0.06),
        xycoords="axes fraction",
        ha="right",
        fontsize=9,
        color=GREY,
    )

    fig.savefig(out / "fig17_uncertainty_vs_activity.png")
    plt.close(fig)
    print("fig17 written")
    for sz, ys in sel:
        print(
            f"  {sz:>9,}  mean {ys.mean():6.3f}  frac below genomic median "
            f"{float((ys < np.median(gen_y)).mean()):.3f}"
        )
    print(f"  genomic    mean {gen_y.mean():6.3f}  median {np.median(gen_y):6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
