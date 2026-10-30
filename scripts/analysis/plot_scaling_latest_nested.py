#!/usr/bin/env python
"""Scaling of every reservoir from the latest NESTED-pool cells (Oct 28 night, ``sc31_`` cells): 1 - Pearson r against training-set size on the
genomic test set, the designed high-activity set and random sequences, one line per reservoir, log-log.

Why these cells: every size of a reservoir is a prefix of the same pool (10k inside 30k inside 100k, canonical pool, seed 42), so each curve is
monotone by construction, and all reservoirs went through the same procedure (menu v1: 20 fixed architectures, patience 16, validation V0c, bare
200-bp students). The ensemble used is the one of the FIRST 6 menu models at every size, so every point is an ensemble of the same size (the 100k
cells only trained 6 models). Source: ``scale31`` of ``data_oct30.json`` (written by ``analyze_oct30.py``).

    python scripts/analysis/plot_scaling_latest_nested.py --data data_oct30.json --out outdir
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SETS = [
    ("genomic", "genomic test (chr7/13)"),
    ("ood", "designed high-activity"),
    ("random_32k", "random sequences"),
]
FAMILY_COLOR = {
    "genomic": "#1b7837",
    "random": "#2166ac",
    "motif": "#d95f02",
    "perturbation": "#7b3294",
    "model": "#636363",
}
STYLES = [("-", "o"), ("--", "s"), (":", "^"), ("-.", "v")]
SIZES = ("10000", "30000", "100000")


def collect(data: dict, which: str = "first6"):
    cells = {}
    for res, v in data.items():
        for D in SIZES:
            s = v["sizes"].get(D)
            if s and which in s:
                for ev, r in s[which]["ensemble"].items():
                    cells[(res, int(D), ev)] = (r, s[which]["n_models"])
    return cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw = json.load(open(a.data))["scale31"]
    cells = collect(raw)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ress = list(raw)
    fams = {r: raw[r]["family"] for r in ress}
    labels = {r: raw[r]["label"] for r in ress}
    nonmono = []
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), constrained_layout=True)
    rows, exps = [], []
    used = {}
    for ax, (ev, name) in zip(axes, SETS):
        for res in ress:
            col = FAMILY_COLOR.get(fams[res], "#444444")
            k = used.setdefault(fams[res], [])
            if res not in k:
                k.append(res)
            ls, mk = STYLES[k.index(res) % len(STYLES)]
            pts = [
                (int(D), 1.0 - cells[(res, int(D), ev)][0])
                for D in SIZES
                if (res, int(D), ev) in cells
            ]
            if len(pts) < 2:
                continue
            if any(b[1] > a[1] for a, b in zip(pts, pts[1:])):
                nonmono.append((res, ev))
            ax.plot(
                *zip(*pts),
                ls=ls,
                marker=mk,
                ms=6,
                lw=1.8,
                color=col,
                label=f"{fams[res]}: {labels[res]}",
            )
            b = -float(
                np.polyfit(np.log10([p[0] for p in pts]), np.log10([p[1] for p in pts]), 1)[0]
            )
            exps.append(
                {
                    "reservoir": res,
                    "label": labels[res],
                    "family": fams[res],
                    "eval": ev,
                    "exponent_b": round(b, 4),
                }
            )
            for D in SIZES:
                if (res, int(D), ev) in cells:
                    r, n = cells[(res, int(D), ev)]
                    rows.append(
                        {
                            "reservoir": res,
                            "label": labels[res],
                            "family": fams[res],
                            "D": int(D),
                            "eval": ev,
                            "n_models": n,
                            "pearson_r": round(r, 5),
                            "one_minus_r": round(1 - r, 5),
                        }
                    )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([int(D) for D in SIZES])
        ax.set_xticklabels([f"{int(D) // 1000}k" for D in SIZES])
        ax.minorticks_off()
        ax.yaxis.set_major_locator(
            matplotlib.ticker.LogLocator(base=10, subs=(1, 1.5, 2, 3, 5, 7), numticks=20)
        )
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2g}"))
        ax.grid(alpha=0.25, which="both")
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("training sequences")
    axes[0].set_ylabel("1 - Pearson r")
    h, lab = axes[-1].get_legend_handles_labels()
    fig.legend(h, lab, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9.5, frameon=False)
    fig.suptitle(
        "1 - Pearson r against training-set size, latest nested-pool cells (10k inside 30k inside 100k of one pool per reservoir)\n"
        "same procedure for every reservoir: menu v1, ensemble of the first 6 models at every size, bare 200-bp students",
        fontsize=13,
    )
    fig.savefig(out / "scaling_1minusr_latest_nested_3_eval_sets.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    for name, rr in (
        ("scaling_latest_nested_cells.csv", rows),
        ("scaling_latest_nested_exponents.csv", exps),
    ):
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rr[0]))
            w.writeheader()
            w.writerows(rr)
    print("curves:", len(exps), "non-monotone curves:", nonmono if nonmono else "none")
    print(
        "exponent range:",
        min(e["exponent_b"] for e in exps),
        max(e["exponent_b"] for e in exps),
        "NaN" if any(math.isnan(e["exponent_b"]) for e in exps) else "",
    )


if __name__ == "__main__":
    main()
