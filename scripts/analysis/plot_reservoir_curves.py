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

CELL = re.compile(r"^(?P<res>.+?)__base(?P<base>\d+)__add(?P<add>\d+)__s(?P<seed>\d+)$")


def load(root: Path, metric: str) -> dict:
    out = defaultdict(list)
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
        m = d.get("test_metrics", {}).get(metric, {})
        r = m.get("pearson_r")
        if not isinstance(r, (int, float)):
            continue
        out[(cell["res"], int(cell["base"]), int(cell["add"]))].append(float(r))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="outputs/curves/train")
    ap.add_argument("--metric", default="in_dist")
    ap.add_argument("--out-dir", default="outputs/curves/figures")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load(REPO / args.root, args.metric)
    if not data:
        print(f"no results yet under {args.root}", file=sys.stderr)
        return 1

    reservoirs = sorted({k[0] for k in data})
    bases = sorted({k[1] for k in data})
    adds = sorted({k[2] for k in data})
    n_cells = len(data)
    n_runs = sum(len(v) for v in data.values())
    print(f"{n_runs} runs across {n_cells} cells | {len(reservoirs)} reservoirs "
          f"| starts {bases} | increments {adds}")

    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("tab20")
    colour = {r: cmap(i % 20) for i, r in enumerate(reservoirs)}

    # ---- panel 1: curves, one panel per starting point -------------------
    fig, axes = plt.subplots(1, len(bases), figsize=(5.2 * len(bases), 4.4), squeeze=False)
    for ax, b in zip(axes[0], bases):
        present = 0
        for r in reservoirs:
            xs, ys, es = [], [], []
            for a in adds:
                v = data.get((r, b, a))
                if not v:
                    continue
                xs.append(a)
                ys.append(float(np.mean(v)))
                es.append(float(np.std(v)) if len(v) > 1 else 0.0)
                present += 1
            if xs:
                ax.errorbar(xs, ys, yerr=es, marker="o", ms=4, lw=1.4, capsize=2,
                            color=colour[r], label=r)
        ax.set_xscale("log")
        ax.set_xlabel("sequences added")
        ax.set_ylabel(f"test Pearson r ({args.metric})")
        total = len(reservoirs) * len(adds)
        ax.set_title(f"start = {b:,}" + ("  (from scratch)" if b == 0 else "")
                     + f"\n{present}/{total} cells present")
        ax.grid(alpha=0.3, lw=0.5)
    axes[0][-1].legend(fontsize=6, ncol=2, loc="lower right")
    fig.suptitle(f"Additive reservoir scaling — {args.metric} "
                 f"({n_runs} runs; missing points are still training)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"curves_{args.metric}.png", dpi=180)
    print(f"  wrote {out / f'curves_{args.metric}.png'}")

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
