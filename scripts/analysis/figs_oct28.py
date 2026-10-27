#!/usr/bin/env python
"""Figures for the Oct-28 results (how many configurations to train per dataset size). Reads data.json written by analyze_oct27.py (kcurve).

python scripts/analysis/figs_oct28.py --data outputs/justification_oct22/data.json --out outputs/justification_oct22
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_oct22 as a  # noqa: E402

plt = a.plt
BLUE, ORANGE, GREEN, RED, GREY, PURPLE, TEAL = (
    a.BLUE,
    a.ORANGE,
    a.GREEN,
    a.RED,
    a.GREY,
    a.PURPLE,
    a.TEAL,
)
RULES = [
    ("val", "ranked by search validation r", BLUE),
    ("first2", "best 2 of every run first, then by search validation r", GREEN),
    ("random", "random order (mean of 60)", GREY),
]
EVN = {"genomic": "genomic test", "ood": "designed set"}


def fig_kcurve(d, out):
    K = d.get("kcurve")
    if not K:
        return
    Ds = (10000, 30000, 100000)
    fig, axs = plt.subplots(2, 3, figsize=(12.6, 7.0), sharex="col")
    for j, D in enumerate(Ds):
        cells = {k: v for k, v in K.items() if k.startswith("menu33|") and k.endswith(f"|{D}")}
        for i, e in enumerate(("genomic", "ood")):
            ax = axs[i, j]
            if not cells:
                continue
            for rule, lab, col in RULES:
                per = []
                for v in cells.values():
                    c = np.mean([p[rule]["top8"][e] for p in v["pools"]], 0)
                    ref = np.mean([p["val"]["top8"][e][-1] for p in v["pools"]], 0)
                    per.append(c - ref)
                n = min(len(x) for x in per)
                m = np.mean([x[:n] for x in per], 0)
                ax.plot(
                    np.arange(1, n + 1),
                    m,
                    "-",
                    color=col,
                    lw=2.0 if rule == "first2" else 1.4,
                    label=lab,
                )
                if rule == "first2":
                    for x in per:
                        ax.plot(np.arange(1, n + 1), x[:n], "-", color=col, lw=0.7, alpha=0.35)
            nn = []
            for v in cells.values():
                c = np.mean([p["first2"]["nnls"][e] for p in v["pools"]], 0)
                ref = np.mean([p["val"]["top8"][e][-1] for p in v["pools"]], 0)
                nn.append(c - ref)
            n = min(len(x) for x in nn)
            ax.plot(
                np.arange(1, n + 1),
                np.mean([x[:n] for x in nn], 0),
                "--",
                color=TEAL,
                lw=1.5,
                label="non-negative weights over all K (best 2 first)",
            )
            ax.axhspan(-0.005, 0.005, color=GREY, alpha=0.15, lw=0)
            ax.axhline(0, color=GREY, lw=0.8)
            lim = (-0.05, 0.012) if e == "genomic" else (-0.12, 0.04)
            ax.set_ylim(*lim)
            if i == 0:
                ax.set_title(f"{D // 1000}k training sequences", fontsize=10.5)
            if j == 0:
                ax.set_ylabel(f"{EVN[e]}\nr minus r of the best-8 ensemble of all", fontsize=9)
            if i == 1:
                ax.set_xlabel("configurations trained (K)")
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(
        h, lab, loc="lower center", ncol=2, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.01)
    )
    fig.suptitle(
        "Final 8-member ensemble (the 8 best of the K configurations trained, by validation r) against K\nmean over genomic, random and motif pools (thin lines: each reservoir); shaded band: +/-0.005, the noise floor",
        fontsize=10,
        y=0.995,
    )
    fig.subplots_adjust(bottom=0.15, top=0.9)
    a.savefig(fig, "kcurve", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/justification_oct22/data.json")
    ap.add_argument("--out", default="outputs/justification_oct22")
    args = ap.parse_args()
    d = json.load(open(args.data))
    for fn in (fig_kcurve,):
        try:
            fn(d, args.out)
            print("ok", fn.__name__, flush=True)
        except Exception as ex:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print("FAILED", fn.__name__, ex, flush=True)


if __name__ == "__main__":
    main()
