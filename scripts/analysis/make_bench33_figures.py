# ruff: noqa
#!/usr/bin/env python
"""Slide figures of the Oct-28 benchmark (reads data_bench33.json from analyze_bench33.py).

python scripts/analysis/make_bench33_figures.py --in <dir with data_bench33.json> --out slide_figures/benchmarks
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.size": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.dpi": 100,
    }
)
BLUE, ORANGE, GREEN, PURPLE, GREY, DARK, RED = (
    "#1b6ca8",
    "#c4571a",
    "#2a8f5b",
    "#7a5195",
    "#8a949a",
    "#1c2530",
    "#b03a48",
)
SETN = {
    "genomic": "genomic test (chr7/13)",
    "ood": "designed high-activity (OOD)",
    "random_32k": "random sequences",
}
SETS = ("genomic", "ood", "random_32k")


def pick(arms, prefix, n_min=3):
    """The configuration of `prefix` (ln_ / ag2_) with the best mean single-model V0c validation Pearson among those with enough models."""
    c = [(k, v) for k, v in arms.items() if k.startswith(prefix) and v["n_models"] >= n_min]
    return max(c, key=lambda kv: kv[1]["single"]["val_r_mean"])[0] if c else None


def rows_for(arms):
    """(label, color, value-by-set, error-by-set or None, n models) for the arms of the figure, in reading order."""
    out = []
    ln, ln_pub = pick(arms, "ln_"), arms.get("ln_pub")
    ag = pick(arms, "ag2_", 1)
    ag1 = pick(arms, "ag1_", 1)

    def single(v):
        return {s: v["single"]["mean"][s] for s in SETS}, {s: v["single"]["sd"][s] for s in SETS}

    def ens(v, kind, n=None):
        n = str(v["n_models"] if n is None else n)
        return {s: v["ens"][n][kind][s] for s in SETS}

    if ln_pub and ln_pub["n_models"]:
        m, e = single(ln_pub)
        out.append(
            (
                f"MPRA-LegNet, published recipe\n(single, mean of {ln_pub['n_models']} seeds)",
                GREY,
                m,
                e,
                1,
            )
        )
    if ln:
        v = arms[ln]
        m, e = single(v)
        out.append(
            (f"MPRA-LegNet, {ln[3:]}\n(single, mean of {v['n_models']})", "#9fb3bc", m, e, 1)
        )
        out.append(
            (
                f"MPRA-LegNet ×{v['n_models']}\n(mean of members)",
                BLUE,
                ens(v, "mean_pred"),
                None,
                v["n_models"],
            )
        )
        out.append(
            (
                f"MPRA-LegNet ×{v['n_models']}\n(stacked)",
                "#0d3f66",
                ens(v, "stack"),
                None,
                v["n_models"],
            )
        )
    if ag1:
        v = arms[ag1]
        m, e = single(v)
        out.append(
            (f"AlphaGenome-FT S1 only\n(single, mean of {v['n_models']})", "#d9b8e6", m, e, 1)
        )
    if ag:
        v = arms[ag]
        m, e = single(v)
        out.append(
            (f"AlphaGenome-FT S2 (from S1)\n(single, mean of {v['n_models']})", ORANGE, m, e, 1)
        )
        if v["n_models"] >= 2:
            out.append(
                (
                    f"AlphaGenome-FT S2 ×{v['n_models']}\n(stacked)",
                    "#8f3a0e",
                    ens(v, "stack"),
                    None,
                    v["n_models"],
                )
            )
    mv = arms.get("menu_v1")
    if mv:
        m, e = single(mv)
        out.append((f"menu v1, one architecture\n(mean of {mv['n_models']})", "#8fc5a7", m, e, 1))
        out.append(
            (f"menu v1 ×{mv['n_models']} (stacked)", GREEN, ens(mv, "stack"), None, mv["n_models"])
        )
    return out


def fig_main(data, out):
    Ds = [D for D in ("30000", "10000") if D in data["arms"] and data["arms"][D]]
    fig, axs = plt.subplots(len(Ds), 3, figsize=(19, 5.2 * len(Ds)), squeeze=False)
    for ri, D in enumerate(Ds):
        rows = rows_for(data["arms"][D])
        for ci, s in enumerate(SETS):
            ax = axs[ri][ci]
            ys = [r[2][s] for r in rows]
            es = [r[3][s] if r[3] else 0 for r in rows]
            for i, (y, e, r) in enumerate(zip(ys, es, rows)):
                ax.hlines(i, 0, y, color=r[1], lw=1.2, alpha=0.35)
                ax.errorbar(
                    y,
                    i,
                    xerr=e if e else None,
                    fmt="o",
                    ms=9,
                    color=r[1],
                    ecolor=DARK,
                    elinewidth=1.2,
                    capsize=3,
                    zorder=3,
                )
                ax.text(y + (e or 0) + 0.006, i, f"{y:.3f}", va="center", fontsize=9)
            ax.set_yticks(
                range(len(rows)),
                [r[0] for r in rows] if ci == 0 else [""] * len(rows),
                fontsize=8.5,
            )
            ax.invert_yaxis()
            span = max(ys) - min(ys)
            ax.set_xlim(max(0, min(ys) - 0.25 * span - 0.02), max(ys) + 0.25 * span + 0.03)
            ax.set_title(f"D = {int(D) // 1000}k: {SETN[s]}", fontsize=11)
            ax.set_xlabel("Pearson r against oracle_v3")
    best = {}
    for D in Ds:
        rows = rows_for(data["arms"][D])
        best[D] = {r[0].split("\n")[0]: r[2]["genomic"] for r in rows}
    fig.suptitle(
        "Genomic ensembles against single and many MPRA-LegNet and AlphaGenome-FT-MPRA models, same 30k / 10k genomic pool, oracle_v3 labels (error bars: sd over seeds)",
        fontsize=12,
        y=1.0,
    )
    fig.tight_layout()
    os.makedirs(out, exist_ok=True)
    fig.savefig(
        os.path.join(out, "bench_genomic_ensembles.png"),
        dpi=170,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print("wrote bench_genomic_ensembles.png")


def fig_growth(data, out):
    """Ensemble r against the number of members, for the arms that have several sizes (stacked)."""
    Ds = [D for D in ("30000", "10000") if D in data["arms"] and data["arms"][D]]
    fig, axs = plt.subplots(len(Ds), 3, figsize=(16, 4.3 * len(Ds)), squeeze=False)
    for ri, D in enumerate(Ds):
        arms = data["arms"][D]
        ln, ag = pick(arms, "ln_"), pick(arms, "ag2_", 2)
        spec = [
            ("menu_v1", "menu v1", GREEN),
            (ln, "MPRA-LegNet", BLUE),
            (ag, "AlphaGenome-FT S2", ORANGE),
        ]
        for ci, s in enumerate(SETS):
            ax = axs[ri][ci]
            for key, lab, col in spec:
                if not key or key not in arms:
                    continue
                v = arms[key]
                ks = [1] + [int(k) for k in sorted(v["ens"], key=int)]
                ys = [v["single"]["mean"][s]] + [
                    v["ens"][str(k)]["stack"].get(s, v["ens"][str(k)]["mean_pred"][s])
                    for k in ks[1:]
                ]
                ax.plot(ks, ys, "o-", color=col, lw=2, label=f"{lab} (n={v['n_models']})")
            ax.set_xscale("log")
            ax.set_xticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
            ax.set_xlabel("models in the ensemble")
            ax.set_title(f"D = {int(D) // 1000}k: {SETN[s]}", fontsize=11)
            if ci == 0:
                ax.set_ylabel("Pearson r (single = mean over members)")
            ax.legend(frameon=False, fontsize=8.5)
    fig.suptitle(
        "How ensemble r grows with the number of models, by architecture family (stacked on V0c; random subsets of the available seeds)",
        fontsize=12,
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(out, "bench_ensemble_growth.png"),
        dpi=170,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print("wrote bench_ensemble_growth.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = json.load(open(os.path.join(a.inp, "data_bench33.json")))
    fig_main(data, a.out)
    fig_growth(data, a.out)


if __name__ == "__main__":
    main()
