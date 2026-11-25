"""What uncertainty selection actually buys, in activity, as the candidate pool grows.

Companion to fig5. fig5 answers "does the student get better"; this answers "what changed
about the corpus", which is the mechanism and is readable before any student finishes.

Both panels come from the ab52 selection step alone -- the 30k corpora on disk and their
oracle labels -- so they need no GPU and no training.

    (a) mean activity of the selected 30k against the candidate-pool size, with the pool's
        own mean as the flat reference. The 30k point is a no-op selection and must sit
        exactly on that reference; it is the built-in control for the whole panel.
    (b) the full activity DISTRIBUTION of each selected corpus, as a violin. A mean can rise
        because the whole distribution shifted or because a tail was added, and those are
        different corpora; only the shape distinguishes them.

    python scratchpad/nov25/acq_activity_figure.py --out outputs/figures/nov25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
POOLS = REPO / "outputs/acq52/pools"
A_SIZES = [30_000, 60_000, 100_000, 300_000, 1_000_000]
BLUE, VERM, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#6b6b6b"

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
    a = ap.parse_args()
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)

    sizes, sel_mean, pool_mean, dists = [], [], [], []
    for sz in A_SIZES:
        f = POOLS / f"poolA_{sz}_d30000.npz"
        m = POOLS / f"poolA_{sz}_d30000_mix.json"
        if not f.exists() or not m.exists():
            continue
        y = np.asarray(np.load(f, allow_pickle=True)["oracle_labels"], float)
        j = json.loads(m.read_text())
        sizes.append(sz)
        sel_mean.append(float(y.mean()))
        pool_mean.append(float(j["pool_label_mean"]))
        dists.append(y)
    if len(sizes) < 2:
        print(f"only {len(sizes)} pool(s) selected so far")
        return 1

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.3), gridspec_kw={"wspace": 0.22})

    ax = axes[0]
    ax.plot(sizes, sel_mean, "o-", color=BLUE, lw=1.9, ms=6, label="selected 30k")
    ax.plot(sizes, pool_mean, "s--", color=GREY, lw=1.3, ms=5, label="candidate pool")
    ax.plot([sizes[0]], [sel_mean[0]], "o", mfc="white", mec=BLUE, mew=1.8, zorder=5)
    ax.annotate(
        "no-op selection:\nselected = pool",
        xy=(sizes[0], sel_mean[0]),
        xytext=(14, 18),
        textcoords="offset points",
        fontsize=8,
        color=GREY,
        arrowprops={"arrowstyle": "->", "color": GREY, "lw": 0.8},
    )
    ax.set_xscale("log")
    ax.set_xlabel("candidate sequences available to the rule")
    ax.set_ylabel("mean oracle activity (log2FC)")
    ax.set_title("(a) activity bought by uncertainty selection")
    ax.legend()

    ax = axes[1]
    parts = ax.violinplot(dists, positions=range(len(sizes)), widths=0.8, showextrema=False)
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(GREY if i == 0 else BLUE)
        b.set_alpha(0.45 if i == 0 else 0.7)
        b.set_edgecolor("black")
        b.set_linewidth(0.4)
    # Medians drawn separately: violinplot's own median line is thin and gets lost against
    # the body at this figure size.
    ax.plot(range(len(sizes)), [float(np.median(d)) for d in dists], "o", color="black", ms=4)
    ax.axhline(pool_mean[0], color=GREY, lw=1.1, ls="--")
    ax.text(len(sizes) - 0.45, pool_mean[0], " pool mean", fontsize=8, va="bottom", ha="right")
    ax.set_xticks(range(len(sizes)), [f"{s // 1000}k" for s in sizes])
    ax.set_xlabel("candidate pool size")
    ax.set_ylabel("oracle activity of the selected 30k")
    ax.set_title("(b) distribution of each selected corpus")

    fig.savefig(out / "fig14_selected_activity.png")
    plt.close(fig)
    print(f"fig14 written ({len(sizes)} pools)")
    for s, sm, pm, d in zip(sizes, sel_mean, pool_mean, dists, strict=True):
        print(
            f"  {s:>9,}  selected mean {sm:6.3f} (pool {pm:5.3f})  "
            f"sd {d.std():.3f}  p90 {np.quantile(d, 0.9):6.3f}  "
            f"frac>2 {float((d > 2).mean()):.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
