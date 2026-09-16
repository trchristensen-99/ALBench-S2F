"""Figures for the additive reservoir scaling curves.

Built to run on PARTIAL data: points still training are simply absent, and every
panel says how many of its cells are present so a sparse figure cannot be mistaken
for a complete one. Run it as late as possible before a meeting.

Panels:
  1. scaling curves  — added sequences vs test metric, one line per reservoir,
                       one panel per starting point
  2. zoonomia A/B    — the phylogenetic variant against its own flat control
  3. coverage grid   — which cells exist, so the gaps are explicit
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Figure conventions. Large type throughout -- these are read projected, not on a
# laptop. The rest follows the feedback that has actually been given in meetings:
#   - replicate error bars, because "training on one dataset measures the dataset,
#     not the method"; the subset-order seeds ARE the replicates
#   - the MEASURED noise floor drawn as a band, so a reader can see at a glance
#     whether a gap is interpretable rather than taking the ranking on faith
#   - several eval sets per figure, because no single eval should carry a claim
#   - the genomic arm drawn as an explicit reference line, since "beats real CREs
#     at matched D" is the comparison that matters
# ---------------------------------------------------------------------------
BIG = {
    "font.size": 15,
    "axes.titlesize": 17,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "figure.titlesize": 19,
    "axes.linewidth": 1.3,
    "lines.linewidth": 2.4,
    "lines.markersize": 8,
    "grid.alpha": 0.3,
}
# Measured on the 105-cell screen: 2 x median within-config SD. Gaps below this
# cannot be separated from replicate noise.
NOISE_FLOOR = 0.0044

CELL = re.compile(r"^(?P<res>.+?)__base(?P<base>\d+)__add(?P<add>\d+)__s(?P<seed>\d+)$")


def load(root: Path, metric: str | None = None) -> dict:
    """metric=None loads every eval set: {metric: {(res, base, add): [r, ...]}}."""
    out = defaultdict(lambda: defaultdict(list))
    for res_json in root.rglob("result.json"):
        cell = None
        for part in res_json.parts:
            if CELL.match(part):
                cell = CELL.match(part)
                break
        if cell is None:
            continue
        try:
            d = json.loads(res_json.read_text())
        except Exception:
            continue
        for name, mm in d.get("test_metrics", {}).items():
            if not isinstance(mm, dict):
                continue
            r = mm.get("pearson_r")
            if isinstance(r, (int, float)) and np.isfinite(r):
                key = (cell["res"], int(cell["base"]), int(cell["add"]))
                out[name][key].append(float(r))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="outputs/curves/train")
    ap.add_argument("--metric", default="in_dist")
    ap.add_argument(
        "--exclude",
        default="zoonomia,zoonomia_flat,mutagenesis_zoonomia,evoaug_zoonomia",
        help="Comma-separated arms to drop. Defaults to the zoonomia family: the "
        "current implementation mutates HUMAN cCREs under conservation-derived "
        "rates, which is not the arm that was asked for (actual ortholog sequences "
        "from other mammals). Plotting it under that name would misrepresent it.",
    )
    ap.add_argument("--out-dir", default="outputs/curves/figures")
    ap.add_argument(
        "--max-add",
        type=int,
        default=30000,
        help="Drop increments above this. Defaults to the ACTIVE plan (+10k, +30k): "
        "the larger increments were deferred mid-run, so only some arms have them, "
        "and an arm whose line simply extends further reads as an arm that is "
        "winning. Raise it once the larger points are complete for every arm.",
    )
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(BIG)

    per_metric = load(REPO / args.root)
    all_metrics = sorted(per_metric)
    data = per_metric.get(args.metric, {})
    if not per_metric:
        print(f"no results yet under {args.root}", file=sys.stderr)
        return 1

    drop = {x.strip() for x in args.exclude.split(",") if x.strip()}
    reservoirs = sorted({k[0] for k in data} - drop)
    if drop & {k[0] for k in data}:
        print(f"  excluded arms: {sorted(drop & {k[0] for k in data})}")
    bases = sorted({k[1] for k in data})
    adds = sorted({k[2] for k in data})
    dropped = [a for a in adds if a > args.max_add]
    adds = [a for a in adds if a <= args.max_add]
    if dropped:
        print(f"  excluded increments above {args.max_add:,}: {dropped} "
              f"(deferred mid-run; only some arms have them)")
    n_cells = len(data)
    n_runs = sum(len(v) for v in data.values())
    print(f"{n_runs} runs across {n_cells} cells | {len(reservoirs)} reservoirs "
          f"| starts {bases} | increments {adds}")

    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("tab20")
    colour = {r: cmap(i % 20) for i, r in enumerate(reservoirs)}

    # ---- panel 1: curves, one panel per eval set, at the largest baseline ----
    EVALS = [e for e in ("in_dist", "snv_delta", "ood") if any(e == mm for mm in all_metrics)]
    if not EVALS:
        EVALS = [args.metric]
    for b in bases:
        fig, axes = plt.subplots(1, len(EVALS), figsize=(7.2 * len(EVALS), 7.4), squeeze=False)
        for ax, ev in zip(axes[0], EVALS):
            dset = per_metric[ev]
            ref = None
            for r in reservoirs:
                xs, ys, es = [], [], []
                for a in adds:
                    v = dset.get((r, b, a))
                    if not v:
                        continue
                    xs.append(a)
                    ys.append(float(np.mean(v)))
                    es.append(float(np.std(v)) if len(v) > 1 else 0.0)
                if not xs:
                    continue
                if r == "genomic":
                    ref = (xs, ys)
                    ax.errorbar(xs, ys, yerr=es, marker="s", color="black", lw=3.2,
                                capsize=4, zorder=5, label="genomic (real CREs)")
                else:
                    ax.errorbar(xs, ys, yerr=es, marker="o", color=colour[r], lw=2.2,
                                capsize=3, alpha=0.9, label=r)
            # the measured noise floor, anchored on the best curve
            if ref and ref[1]:
                last = max(adds)
                at_last = [float(np.mean(v)) for k, v in dset.items()
                           if k[1] == b and k[2] == last and v]
                top = max(at_last) if at_last else max(
                    float(np.mean(v)) for k, v in dset.items() if k[1] == b and v)
                ax.axhspan(top - NOISE_FLOOR, top, color="0.55", alpha=0.20, zorder=0)
                ax.text(0.02, 0.03,
                        f"grey band = measured noise floor ({NOISE_FLOOR:.4f})\n"
                        "differences inside it are not interpretable",
                        transform=ax.transAxes, fontsize=11, va="bottom")
            ax.set_xscale("log")
            ax.set_xlabel("sequences added")
            ax.set_ylabel(f"test Pearson r")
            n_here = sum(len(v) for k, v in dset.items() if k[1] == b)
            ax.set_title(f"{ev}   ({n_here} runs)")
            ax.grid(True, which="both", lw=0.6)
        h, lab = axes[0][-1].get_legend_handles_labels()
        fig.legend(h, lab, fontsize=12, ncol=min(5, max(2, len(lab) // 3)),
                   loc="lower center", bbox_to_anchor=(0.5, 0.005), frameon=False)
        start_lbl = "from scratch" if b == 0 else f"{b:,} genomic baseline"
        fig.suptitle(f"Adding reservoir data on top of {start_lbl}", fontweight="bold")
        # leave real room at the bottom: the legend sat on the x-axis label before
        fig.tight_layout(rect=(0, 0.17, 1, 1))
        f = out / f"curves_base{b}.png"
        fig.savefig(f, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {f}")

    # ---- panel 2: zoonomia against its own control -----------------------
    pair = [p for p in ("zoonomia", "zoonomia_flat") if p in reservoirs]
    if len(pair) == 2:
        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        for r, style in zip(pair, ("-o", "--s")):
            for b in bases:
                xs = [a for a in adds if data.get((r, b, a))]
                ys = [float(np.mean(data[(r, b, a)])) for a in xs]
                if xs:
                    ax.plot(xs, ys, style, lw=1.5, ms=5,
                            label=f"{r} (start {b:,})", alpha=0.85)
        ax.set_xscale("log")
        ax.set_xlabel("sequences added")
        ax.set_ylabel(f"test Pearson r ({args.metric})")
        ax.set_title("Zoonomia: phylogenetic rates vs the flat control\n"
                     "(flat DISCARDS the conservation profile)")
        ax.grid(alpha=0.3, lw=0.5)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "zoonomia_vs_flat.png", dpi=180)
        print(f"  wrote {out / 'zoonomia_vs_flat.png'}")
    else:
        print("  zoonomia pair not both present yet; skipping that panel")

    # ---- panel 3: coverage, so gaps are explicit -------------------------
    grid = np.zeros((len(reservoirs), len(bases) * len(adds)))
    cols = [(b, a) for b in bases for a in adds]
    for i, r in enumerate(reservoirs):
        for j, (b, a) in enumerate(cols):
            grid[i, j] = len(data.get((r, b, a), []))
    fig, ax = plt.subplots(figsize=(1.1 + 0.42 * len(cols), 0.34 * len(reservoirs) + 1.6))
    ax.imshow(grid, cmap="Greens", vmin=0, vmax=max(2, grid.max()), aspect="auto")
    ax.set_yticks(range(len(reservoirs)), reservoirs, fontsize=7)
    ax.set_xticks(range(len(cols)), [f"{b//1000}k+{a//1000}k" for b, a in cols],
                  fontsize=7, rotation=90)
    for i in range(len(reservoirs)):
        for j in range(len(cols)):
            ax.text(j, i, int(grid[i, j]), ha="center", va="center", fontsize=6,
                    color="white" if grid[i, j] > 1 else "#444")
    ax.set_title(f"seeds completed per cell ({int(grid.sum())} of "
                 f"{len(reservoirs) * len(cols) * 2} runs)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "coverage.png", dpi=180)
    print(f"  wrote {out / 'coverage.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
