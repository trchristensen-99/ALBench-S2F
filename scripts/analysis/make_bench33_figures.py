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


def pick_ag(arms, prefix):
    """The AlphaGenome recipe with the most trained models (ties: best mean single-model V0c Pearson): the one whose ensemble is shown."""
    c = [(k, v) for k, v in arms.items() if k.startswith(prefix)]
    return (
        max(c, key=lambda kv: (kv[1]["n_models"], kv[1]["single"]["val_r_mean"]))[0] if c else None
    )


def rows_for(arms, include_ag=True):
    """(label, color, value-by-set, error-by-set or None, n models) for the arms of the figure, in reading order."""
    out = []
    ln, ln_pub = pick(arms, "ln_"), arms.get("ln_pub")
    ag = pick_ag(arms, "ag2_") if include_ag else None
    ag1 = pick_ag(arms, "ag1_") if include_ag else None

    def single(v):
        return {s: v["single"]["mean"][s] for s in SETS}, {s: v["single"]["sd"][s] for s in SETS}

    def ens(v, kind, n=None):
        n = str(v["n_models"] if n is None else n)
        return {s: v["ens"][n][kind][s] for s in SETS}

    if ln_pub and ln_pub["n_models"] and ln != "ln_pub":
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
        for k2, v2 in arms.items():
            if k2.startswith("ag2_") and k2 != ag and v2["n_models"] >= 1:
                m2, _ = single(v2)
                lab2 = f"AlphaGenome-FT S2, {k2[4:]} recipe\n(single, {v2['n_models']} seed{'s' if v2['n_models'] > 1 else ''})"
                out.append((lab2, "#e3a27a", m2, None, 1))
    mv = arms.get("menu_v1")
    if mv:
        m, e = single(mv)
        out.append((f"menu v1, one architecture\n(mean of {mv['n_models']})", "#8fc5a7", m, e, 1))
        out.append(
            (f"menu v1 ×{mv['n_models']} (stacked)", GREEN, ens(mv, "stack"), None, mv["n_models"])
        )
    return out


def fig_main(data, out, include_ag=True, name="bench_genomic_ensembles.png"):
    Ds = [D for D in ("30000", "10000") if D in data["arms"] and data["arms"][D]]
    fig, axs = plt.subplots(len(Ds), 3, figsize=(19, 5.2 * len(Ds)), squeeze=False)
    for ri, D in enumerate(Ds):
        rows = rows_for(data["arms"][D], include_ag)
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
    fig.suptitle(
        "Genomic ensembles against single and many MPRA-LegNet"
        + (
            " and AlphaGenome-FT-MPRA models"
            if include_ag
            else " models (AlphaGenome-FT omitted to zoom in)"
        )
        + ", same 30k / 10k genomic pool, oracle_v3 labels (error bars: sd over seeds)",
        fontsize=12,
        y=1.0,
    )
    fig.tight_layout()
    os.makedirs(out, exist_ok=True)
    fig.savefig(
        os.path.join(out, name),
        dpi=170,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print("wrote", name)


def fig_growth(data, out):
    """Ensemble r against the number of members, for the arms that have several sizes (stacked)."""
    Ds = [D for D in ("30000", "10000") if D in data["arms"] and data["arms"][D]]
    fig, axs = plt.subplots(len(Ds), 3, figsize=(16, 4.3 * len(Ds)), squeeze=False)
    for ri, D in enumerate(Ds):
        arms = data["arms"][D]
        ln, ag = pick(arms, "ln_"), pick_ag(arms, "ag2_")
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
            ax.legend(frameon=False, fontsize=8.5, loc="center right")
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


TRAIN_TEXT = """## Training details of every arm

All arms train on the nested canonical genomic pool of the size (`outputs/grid_pools/k562_nested/genomic_d{D}.npz`, replicate 0, oracle_v3 labels), are
early-stopped / checkpoint-selected on V0c (`outputs/val_variants/val_V0.npz`, oracle_v3 labels, 29,449 chr19/21/X sequences; never trained on) and are
scored against oracle_v3 on the 18 test sets of `scaling_hp_search.load_all_test_sets()` (the same ones as the menu retrains). Predictions are reverse-complement
averaged; forward-only predictions are stored too (`fwd_*` in each npz). Training seeds were drawn once from `os.urandom` (`seeds.json` in this folder) and are
recorded in every meta file; none is pinned. The ensemble "stack" is the positive ElasticNetCV of `albench.ensemble.stack.fit_zscored_stack` fitted on V0c;
"mean" is the plain mean of the members. Subsets of 5 and 10 members are random subsets (8 draws, mean of the draws).

**Menu v1** (the reference, `outputs/menu_retrain_v0/sc31_{10,30}k_genomic`): 20 LLM-written architectures (`albench/ensemble/menus/menu_v1.json`), Muon, lr 2e-3,
weight decay 1e-4, batch 256, one-cycle, 60 to 80 epochs, patience 16, shift and reverse-complement augmentation, one training seed per architecture; validated
during training on `chr_val_ref_only.npz` (the same sequences as V0c with different oracle labels, r = 0.983), so its stack is re-fitted on V0c here.

**MPRA-LegNet** (`scripts/program_oct22/bench33_legnet.py`): the published architecture (`external/human_legnet/model.py`: stem 64 / ks 11, four EffBlocks ks 9, widths
80 96 112 128, MaxPool 2 each, resize factor 4, SiLU, 1.32M parameters) trained from scratch with the published optimiser (AdamW, weight decay 0.1, OneCycleLR max_lr
0.01 from max_lr/25, pct_start 0.3, MSE, fp16 autocast, gradient clipping 1, best epoch by V0c Pearson). Shift (uniform -10..10 on the real plasmid flank) and
reverse complement (p 0.5) augmentation. Three schedules: `pub` batch 1024 for 25 epochs (the published K562 values), `e100` batch 1024 for 100 epochs, `b256` batch 256
for 60 epochs; the one with the best mean single-model V0c Pearson is the "tuned" arm in the figure. 20 seeds per schedule and size.

**AlphaGenome-FT-MPRA** (`scripts/program_oct22/bench33_ag.py`, the recipe of `experiments/train_oracle_s2_v2.py` that trained the oracle members): AlphaGenome encoder
(all_folds weights) with the `boda-flatten-512-512` head (1 track, dropout 0.1), 600 bp input = 200 bp insert + 200 bp real plasmid flank each side. Stage 1: encoder
frozen, head only, AdamW lr 1e-3, weight decay 1e-6, batch 128, up to 60 epochs, patience 6, shift +-15 on the real flank (p 0.5) and reverse complement (p 0.5). Stage 2: the
Stage 1 checkpoint is written to disk, loaded back, and the job asserts that the loaded model reproduces the Stage 1 validation Pearson before any Stage 2 update
(`s2_start_val_after_loading_s1` in each meta file); then every encoder block is unfrozen. Recipe `oracle`: encoder lr 1e-4, head lr 1e-3, constant, up to 60 epochs, patience 4.
Recipe `winner` (memory `project_ag_s2_winner`): encoder lr 1.2e-4, head lr 5e-4, 3 head-only warm-up epochs, cosine decay, 60 epochs. Stage 1 checkpoints:
`outputs/bench33/s1_ckpt/{recipe}_{D}k_s{index}/best_model/checkpoint`. Hardware: H100 NVL (AlphaGenome stage 2 needs about 55 GB), one model per job.
"""


def write_readme(data, out, extra=""):
    lines = [
        "# Benchmark: genomic ensembles against MPRA-LegNet and AlphaGenome-FT-MPRA (Oct 28)",
        "",
        extra,
        "",
        TRAIN_TEXT,
        "## Results (Pearson r against oracle_v3)",
        "",
    ]
    for D in ("30000", "10000"):
        arms = data["arms"].get(D)
        if not arms:
            continue
        lines += [
            f"### D = {int(D) // 1000}k",
            "",
            "| arm | models | single (mean ± sd) genomic / OOD / random | ensemble, plain mean | ensemble, stacked on V0c | r vs experiment, genomic test (single / ensemble) |",
            "|---|---|---|---|---|---|",
        ]
        for k, v in arms.items():
            n = str(v["n_models"])
            g = lambda d, s: d.get(s, float("nan"))  # noqa: E731
            sm, sd = v["single"]["mean"], v["single"]["sd"]
            e = v["ens"][n]
            single = " / ".join(f"{g(sm, s):.3f}±{(g(sd, s) or 0):.3f}" for s in SETS)
            em = " / ".join(f"{g(e['mean_pred'], s):.3f}" for s in SETS)
            es = " / ".join(f"{g(e['stack'], s):.3f}" for s in SETS) if e["stack"] else "-"
            lines.append(
                f"| {v['label']} | {v['n_models']} | {single} | {em} | {es} | {v['single']['exp_genomic_mean']:.3f} / {e.get('exp_genomic_mean_pred', float('nan')):.3f} |"
            )
        lines.append("")
    open(os.path.join(out, "README.md"), "w").write("\n".join(lines) + "\n")
    print("wrote README.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = json.load(open(os.path.join(a.inp, "data_bench33.json")))
    fig_main(data, a.out)
    fig_main(data, a.out, include_ag=False, name="bench_zoom_without_alphagenome.png")
    fig_growth(data, a.out)
    write_readme(
        data,
        a.out,
        extra=open(os.path.join(a.inp, "readme_extra.md")).read()
        if os.path.exists(os.path.join(a.inp, "readme_extra.md"))
        else "",
    )


if __name__ == "__main__":
    main()
