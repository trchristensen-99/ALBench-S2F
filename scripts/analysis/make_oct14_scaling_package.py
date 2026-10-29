# ruff: noqa
#!/usr/bin/env python
"""Package the Oct 13-14 reservoir-scaling results for a collaborator: tables (long CSV, one wide table per metric), a five-family plot of the deploy ensemble, the original Oct-14 figures and a README.

python scripts/analysis/make_oct14_scaling_package.py --table <scaling_oct14_long.csv> --figs ~/Downloads/meeting_figures_oct14 --out ~/Downloads/scaling_oct13_14_for_yash
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FAMILIES = [
    (
        "genomic",
        "Genomic sequences",
        ["genomic", "zoonomia_flank_primates", "zoonomia_flank_mammals"],
    ),
    ("random", "Random sequences", ["random", "gc_matched", "dinuc_shuffle"]),
    ("motif", "Planted motifs", ["motif_jaspar", "motif_ct_enriched", "motif_grammar"]),
    ("augmentation", "Augmentation of genomic", ["evoaug_paper2023", "evoaug_prm010"]),
    ("model-based", "Model-generated", ["d3_steps64", "evo2"]),
]
NICE = {
    "genomic": "genomic (reference)",
    "zoonomia_flank_primates": "Zoonomia, primates",
    "zoonomia_flank_mammals": "Zoonomia, mammals",
    "random": "uniform random",
    "gc_matched": "GC-matched random",
    "dinuc_shuffle": "dinucleotide shuffle",
    "motif_jaspar": "motifs (JASPAR)",
    "motif_ct_enriched": "motifs, CT-enriched",
    "motif_grammar": "motif grammar",
    "evoaug_paper2023": "EvoAug (paper settings)",
    "evoaug_prm010": "EvoAug (p = 0.10)",
    "d3_steps64": "D3 diffusion, 64 steps",
    "evo2": "Evo2 regeneration",
}
COL = ["#1d6fa5", "#c4571a", "#2a7d4f", "#6a4fa3", "#b0382c"]
DS = [10000, 30000, 100000]


def load(p):
    rows = list(csv.DictReader(open(p)))
    out = {}
    for r in rows:
        out[(r["reservoir"], int(r["D"]), r["eval"])] = {
            k: (float(v) if k not in ("family", "reservoir", "eval") and v not in ("", None) else v)
            for k, v in r.items()
        }
    return out


def plot(T, path):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8.6), sharex=True)
    evs = [("genomic", "genomic test (real sequences)"), ("random_32k", "random sequences")]
    for i, (ev, evt) in enumerate(evs):
        lo = min(T[k]["enet_greedy_k5"] for k in T if k[2] == ev) - 0.02
        hi = max(T[k]["enet_greedy_k5"] for k in T if k[2] == ev) + 0.02
        for j, (fam, famt, ress) in enumerate(FAMILIES):
            ax = axes[i, j]
            for k, res in enumerate(ress):
                xs, ys, ns = [], [], []
                for D in DS:
                    t = T.get((res, D, ev))
                    if t:
                        xs.append(D)
                        ys.append(t["enet_greedy_k5"])
                        ns.append(int(t["n_models"]))
                ls = ["-", "--", ":"][k]
                ax.plot(xs, ys, ls, color=COL[j], lw=2.2, alpha=0.9, label=NICE[res])
                for x, y, n in zip(xs, ys, ns):
                    ax.scatter(
                        [x],
                        [y],
                        s=60,
                        facecolor=COL[j] if n >= 20 else "white",
                        edgecolor=COL[j],
                        lw=1.8,
                        zorder=3,
                    )
            ax.set_xscale("log")
            ax.set_xticks(DS)
            ax.set_xticklabels(["10k", "30k", "100k"])
            ax.set_ylim(lo, hi)
            ax.grid(alpha=0.35)
            ax.legend(fontsize=8.5, frameon=False, loc="lower right")
            if i == 0:
                ax.set_title(famt, fontsize=14, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"ensemble r on {evt}", fontsize=12)
            if i == 1:
                ax.set_xlabel("training-set size D", fontsize=12)
    fig.suptitle(
        "Reservoir scaling, 10k to 100k: 13 reservoirs in 5 families, deploy ensemble of the time (greedy top-5 ElasticNetCV)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.005,
        "oracle_v3 labels; HP search llm2x20 (20 rounds x 2 proposals). Open marker = fewer than 20 models in the cell (immature, mostly 100k). Pools are independently sampled at each D (not nested), so a size step also changes the sample.",
        ha="center",
        fontsize=10.5,
        color="#5a6878",
    )
    fig.tight_layout(rect=[0, 0.025, 1, 0.97])
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def wide(T, metric, ev):
    lines = [
        "| family | reservoir | 10k | 30k | 100k | models (10k / 30k / 100k) |",
        "|---|---|---|---|---|---|",
    ]
    for fam, _, ress in FAMILIES:
        for res in ress:
            vals, ns = [], []
            for D in DS:
                t = T.get((res, D, ev))
                vals.append(f"{t[metric]:.3f}" if t else "-")
                ns.append(str(int(t["n_models"])) if t else "-")
            lines.append(
                f"| {fam} | {NICE[res]} | " + " | ".join(vals) + " | " + " / ".join(ns) + " |"
            )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--figs", required=True)
    ap.add_argument(
        "--methods_figs", default=os.path.expanduser("~/Downloads/ALBench-methods/figures")
    )
    ap.add_argument(
        "--prelim", default=os.path.expanduser("~/Downloads/scaling_curves_preliminary_oct14.png")
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(f"{a.out}/figures", exist_ok=True)
    os.makedirs(f"{a.out}/tables", exist_ok=True)
    T = load(a.table)
    shutil.copy(a.table, f"{a.out}/tables/scaling_all_cells_long.csv")
    plot(T, f"{a.out}/figures/scaling_by_family_deploy_ensemble.png")
    for f, src in (
        ("F4_reservoir_scaling.png", a.figs),
        ("F14_ensemble_poolsize.png", a.figs),
        ("F14b_ensemble_poolsize_randomeval.png", a.figs),
        ("F23_scaling_law.png", a.methods_figs),
    ):
        p = os.path.join(src, f)
        if os.path.exists(p):
            shutil.copy(p, f"{a.out}/figures/{f}")
    if os.path.exists(a.prelim):
        shutil.copy(a.prelim, f"{a.out}/figures/scaling_curves_preliminary_oct14.png")
    with open(f"{a.out}/tables/scaling_tables.md", "w") as fh:
        for metric, mt in (
            (
                "enet_greedy_k5",
                "Deploy ensemble of the time: prefilter 12 by validation, greedy forward to 5 models, positive ElasticNetCV (up to the first 40 models of the cell)",
            ),
            (
                "f4_first6",
                "F4 metric: mean single-model r of the first 6 models of the search (equal search budget at every size and reservoir)",
            ),
        ):
            for ev, evt in (
                ("genomic", "genomic test (chr7/13, real sequences)"),
                ("random_32k", "random sequences (32k)"),
                ("ood", "designed high-activity sequences"),
            ):
                fh.write(
                    f"### {mt}\n\nEvaluated on: {evt}; Pearson r against oracle_v3\n\n{wide(T, metric, ev)}\n\n"
                )
    print("done", a.out)


if __name__ == "__main__":
    main()
