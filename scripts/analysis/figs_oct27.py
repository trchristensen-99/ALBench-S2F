#!/usr/bin/env python
"""Figures for the Oct-27 results: core reservoir set, loop-element contrasts, training-recipe sensitivity. Reads data.json written by analyze_oct26.py / analyze_oct27.py.

python scripts/analysis/figs_oct27.py --data outputs/justification_oct22/data.json --old outputs/justification/data.json --out outputs/justification_oct22
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
EVN = {"genomic": "genomic test", "ood": "designed set", "random_32k": "random sequences"}
COL = {
    "genomic": BLUE,
    "random": ORANGE,
    "motif_ct_enriched": GREEN,
    "evoaug_paper2023": PURPLE,
    "evo2": TEAL,
}
NAME = {
    "genomic": "genomic",
    "random": "random",
    "motif_ct_enriched": "motif (CT-enriched)",
    "evoaug_paper2023": "evoaug",
    "evo2": "evo2",
}


def fig_coreset(d, old, out):
    ms = old.get("menu_scaling", {})
    U = {**d.get("unseen", {}).get("results", {}), **d.get("unseen_menu", {})}
    A = d.get("anchor", {})
    if not ms:
        return
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.1))
    for ax, e in zip(axs, ("genomic", "ood", "random_32k")):
        for st in ("evoaug_paper2023", "evo2", "genomic", "random", "motif_ct_enriched"):
            if st not in ms:
                continue
            Ds = sorted(int(k) * 1000 for k in ms[st])
            ys = [ms[st][str(D // 1000)][e] for D in Ds]
            xs, vs = list(Ds), list(ys)
            if st in A and st in ("random", "motif_ct_enriched"):
                for D in (600000, 1000000):
                    if str(D) in A[st]["points"]:
                        xs.append(D)
                        vs.append(A[st]["points"][str(D)][e])
                if st == "motif_ct_enriched" and "300000" in A[st]["points"] and 300000 not in xs:
                    xs.insert(len(Ds), 300000)
                    vs.insert(len(Ds), A[st]["points"]["300000"][e])
            thick = st in ("genomic", "random", "motif_ct_enriched")
            ax.plot(
                xs,
                vs,
                "o-" if thick else "s--",
                color=COL[st],
                lw=2.0 if thick else 1.2,
                ms=5 if thick else 4,
                alpha=1 if thick else 0.75,
                label=NAME[st],
            )
        sc = [
            (D, U[f"motif_shared_core|{D}"]["menu"][e])
            for D in (10000, 30000, 100000)
            if f"motif_shared_core|{D}" in U
        ]
        if sc:
            ax.plot(*zip(*sc), "D-", color=RED, lw=1.6, ms=4.5, label="motif shared core")
        for fam in (
            "motif_syntax_core",
            "mutagenesis",
            "gc_matched",
            "dinuc_shuffle",
            "zoonomia_orthologs",
        ):
            pts = [
                (D, U[f"{fam}|{D}"]["menu"][e])
                for D in (10000, 30000, 100000)
                if f"{fam}|{D}" in U and U[f"{fam}|{D}"].get("menu")
            ]
            if pts:
                ax.plot(*zip(*pts), "-", color=GREY, lw=0.8, alpha=0.55)
                ax.text(
                    pts[-1][0] * 1.04,
                    pts[-1][1],
                    fam.replace("_", " ").replace("orthologs", "").strip(),
                    fontsize=6,
                    color=GREY,
                    va="center",
                )
        ax.set_xscale("log")
        ax.set_xticks([10000, 30000, 100000, 300000, 1000000])
        ax.set_xticklabels(["10k", "30k", "100k", "300k", "1M"])
        ax.set_title(EVN[e], fontsize=10)
        ax.set_xlabel("training sequences")
        if e == "ood":
            ax.set_ylim(0.0, 0.7)
    axs[0].set_ylabel("menu ensemble r")
    axs[0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle(
        "Reservoirs by size and evaluation set (grey: other families tested at 10k to 100k)",
        fontsize=10.5,
        y=1.02,
    )
    a.savefig(fig, "coreset", out)


def fig_loop(d, out):
    L = d.get("loopctl")
    if not L or not L.get("pairs"):
        return
    rows = []
    for k, v in L["pairs"].items():
        g = v["genomic"]
        rows.append(
            (
                k.replace(" minus default", " vs default")
                .replace("(no coordinator, no reflector)", "")
                .replace("  ", " "),
                g["mean"],
                g["lo"],
                g["hi"],
                g["n"],
                BLUE if k.endswith("minus default") else ORANGE,
            )
        )
    for k, v in L.get("guidance", {}).items():
        g = v["genomic"]
        rows.append((f"no guidance vs guidance, {k}", g["mean"], g["lo"], g["hi"], g["n"], PURPLE))
    fig, ax = plt.subplots(figsize=(8.8, 0.45 * len(rows) + 1.4))
    ys = np.arange(len(rows))[::-1]
    for y, (lab, m, lo, hi, n, c) in zip(ys, rows):
        ax.plot([lo, hi], [y, y], color=c, lw=2.2)
        ax.plot([m], [y], "o", color=c, ms=6)
    ax.axvline(0, color=GREY, lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r[0]} (n={r[4]})" for r in rows], fontsize=8)
    ax.set_xlabel("ensemble r difference, genomic test (paired by pool, 95% bootstrap interval)")
    ax.set_title("Loop elements: each contrast against the loop without it", fontsize=10.5)
    a.savefig(fig, "loop_elements", out)


def fig_recipe(d, out):
    R = d.get("recipe")
    if not R:
        return
    cells = {k: v for k, v in R.items() if k != "summary"}
    arms = [
        ("pat4", "patience 4"),
        ("pat16", "patience 16"),
        ("ep05", "epochs x0.5"),
        ("ep2", "epochs x2"),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    w = 0.2
    for j, (m, lab) in enumerate(arms):
        vals = [
            v[m]["r"]["genomic"] - v["default"]["r"]["genomic"] if m in v else np.nan
            for v in cells.values()
        ]
        ax.bar(
            np.arange(len(cells)) + (j - 1.5) * w,
            vals,
            w,
            label=lab,
            color=[BLUE, TEAL, ORANGE, RED][j],
        )
    ax.axhline(0, color=GREY, lw=1)
    ax.axhspan(-0.003, 0.003, color=GREY, alpha=0.12)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(
        [k.replace("_", " ").replace("motif ct enriched", "motif") for k in cells], fontsize=8.5
    )
    ax.set_ylabel("ensemble r change, genomic test")
    ax.legend(fontsize=8, ncol=4)
    ax.set_title(
        "Training-recipe sensitivity of the retrained menu (shaded: +/-0.003)", fontsize=10.5
    )
    a.savefig(fig, "recipe_sensitivity", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/justification_oct22/data.json")
    ap.add_argument("--old", default="outputs/justification/data.json")
    ap.add_argument("--out", default="outputs/justification_oct22")
    args = ap.parse_args()
    d = json.load(open(args.data))
    old = json.load(open(args.old)) if os.path.exists(args.old) else {}
    for fn, kw in (
        (fig_coreset, (d, old, args.out)),
        (fig_loop, (d, args.out)),
        (fig_recipe, (d, args.out)),
    ):
        try:
            fn(*kw)
            print("ok", fn.__name__, flush=True)
        except Exception as ex:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print("FAILED", fn.__name__, ex, flush=True)


if __name__ == "__main__":
    main()
