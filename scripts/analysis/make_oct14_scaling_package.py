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
SETS_ALL = [
    "genomic",
    "ood",
    "random_32k",
    "gc_sweep",
    "dinuc_shuffle",
    "snv_ref",
    "snv_alt",
    "snv_delta",
    "sub_low",
    "sub_med",
    "sub_high",
    "ins_low",
    "ins_med",
    "ins_high",
    "del_low",
    "del_med",
    "del_high",
    "translocation",
    "inversion",
]
SET_TITLE = {
    "genomic": "genomic test (chr7/13)",
    "ood": "designed high-activity",
    "random_32k": "random sequences",
    "gc_sweep": "GC sweep",
    "dinuc_shuffle": "dinucleotide shuffle",
    "snv_ref": "SNV, reference allele",
    "snv_alt": "SNV, alternative allele",
    "snv_delta": "SNV effect (alt - ref)",
    "sub_low": "substitutions, low",
    "sub_med": "substitutions, medium",
    "sub_high": "substitutions, high",
    "ins_low": "insertions, low",
    "ins_med": "insertions, medium",
    "ins_high": "insertions, high",
    "del_low": "deletions, low",
    "del_med": "deletions, medium",
    "del_high": "deletions, high",
    "translocation": "translocation",
    "inversion": "inversion",
}


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


def contact_sheet(T, metric, path):
    fig, axes = plt.subplots(4, 5, figsize=(22, 15), sharex=True)
    for ax, ev in zip(axes.ravel(), SETS_ALL):
        for j, (fam, famt, ress) in enumerate(FAMILIES):
            for k, res in enumerate(ress):
                xs = [D for D in DS if (res, D, ev) in T]
                ys = [T[(res, D, ev)][metric] for D in xs]
                ax.plot(
                    xs,
                    ys,
                    ["-", "--", ":"][k],
                    color=COL[j],
                    lw=1.6,
                    marker="o",
                    ms=4,
                    alpha=0.9,
                    label=famt if k == 0 else None,
                )
        ax.set_xscale("log")
        ax.set_xticks(DS)
        ax.set_xticklabels(["10k", "30k", "100k"])
        ax.set_title(SET_TITLE[ev], fontsize=12)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(SETS_ALL) :]:
        ax.axis("off")
    h, l = axes[0, 0].get_legend_handles_labels()
    axes[3, 4].legend(
        h,
        l,
        loc="center",
        fontsize=12,
        frameon=False,
        title="line colour = family\n(solid / dashed / dotted = reservoirs of the family)",
    )
    name = {
        "enet_greedy_k5": "deploy ensemble (greedy top-5 ElasticNetCV)",
        "f4_first6": "mean single model of the first 6 models",
    }[metric]
    fig.suptitle(
        f"Pearson r against oracle_v3 on every evaluation set, 13 reservoirs, D = 10k / 30k / 100k: {name}",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def wide_csv(T, metric, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "reservoir", "D", "n_models"] + SETS_ALL)
        for fam, _, ress in FAMILIES:
            for res in ress:
                for D in DS:
                    if (res, D, "genomic") not in T:
                        continue
                    w.writerow(
                        [fam, res, D, int(T[(res, D, "genomic")]["n_models"])]
                        + [
                            round(T[(res, D, e)][metric], 4) if (res, D, e) in T else ""
                            for e in SETS_ALL
                        ]
                    )


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


README = """# Reservoir scaling, 10k to 100k: results of Oct 13-14 (for Yash)

What this is: how well a student model trained on D = 10k, 30k or 100k sequences from each training-sequence generator ("reservoir") predicts held-out data, for 13 reservoirs in 5 families, with
the AutoResearch hyperparameter search and ensembling we used on Oct 13-14. This is the *earlier* setup; it is not the current procedure (see "What has changed since" at the end).

## Files
- `figures/F4_reservoir_scaling.png` - the main Oct-14 plot: student r against D, one line per reservoir, on real genomic test sequences (left) and random sequences (right).
- `figures/scaling_by_family_deploy_ensemble.png` - the same cells grouped in the 5 families, with the deploy ensemble of the time (genomic test and random sequences).
- `figures/scaling_all_eval_sets_deploy_ensemble.png`, `figures/scaling_all_eval_sets_first6_single.png` - every evaluation set (19 panels), 13 reservoirs, 10k / 30k / 100k.
- `figures/scaling_curves_preliminary_oct14.png` - the preliminary version of Oct 14 (top-8 mean, open markers = partial cells).
- `figures/F14_ensemble_poolsize.png`, `F14b_...randomeval.png` - how many candidate models the 5-model ensemble needs (Oct 14).
- `figures/F23_scaling_law.png` - later (Oct 16), three reservoirs only, with fitted scaling laws on mature cells.
- `tables/scaling_all_cells_long.csv` - every number: one row per reservoir x D x evaluation set (39 cells x 19 sets = 741 rows), all five metrics.
- `tables/scaling_wide_<metric>.csv` - one table per metric: a row per reservoir and D, a column per evaluation set.
- `tables/scaling_tables.md` - readable tables (13 reservoirs x 3 sizes) for the main evaluation sets.

## Evaluation sets (all scored as Pearson r against oracle_v3 labels; 19 columns in the tables)
- `genomic` - held-out real genomic sequences (chr7/chr13, n = 31,435); `ood` - designed high-activity sequences (out of distribution); `random_32k` - uniformly random sequences (n = 32,000).
- `gc_sweep` (random sequences across GC content), `dinuc_shuffle` (dinucleotide-shuffled genomic sequences).
- `snv_ref`, `snv_alt` - single-nucleotide variants, reference and alternative allele; `snv_delta` - the variant EFFECT: predicted alt minus ref against the oracle's alt minus ref.
- `sub_low/med/high`, `ins_low/med/high`, `del_low/med/high` - substitution, insertion and deletion sets at three strengths; `translocation`, `inversion` - structural edits.

## Setup
- Labels: oracle_v3 (10 AlphaGenome-encoder members on chromosome folds). Training: for each reservoir and D, an LLM-driven hyperparameter / architecture search (`llm2x20`: 20 rounds of 2 proposals, Claude Sonnet via the bridge) trained up to 40 models (200-bp sequence input, validation on chr19/21/X).
  Cells finished at different times; the number of models in each cell is in the tables (6 to 60).
- Metrics (columns of the CSV): `f4_first6` = mean single-model r of the first 6 models of the search (rounds 0-2): an equal search budget at every size and reservoir, so this is the fair comparison across cells and is what F4 plots.
  `top8_by_val` = mean single-model r of the 8 models with the best validation r (preliminary plot). `best_single` = the best model by validation. `enet_greedy_k5` = the deploy ensemble of that period: prefilter the 12 best models by validation r,
  greedy forward selection to 5 models, positive ElasticNetCV fitted on validation (up to the first 40 models). `enet_all_pos` = positive ElasticNetCV over all models of the cell (up to 40).

## Families
| family | reservoirs |
|---|---|
| genomic sequences | genomic (reference), Zoonomia primates, Zoonomia mammals |
| random sequences | uniform random, GC-matched random, dinucleotide shuffle |
| planted motifs | motifs (JASPAR), motifs CT-enriched, motif grammar |
| augmentation of genomic | EvoAug (paper settings), EvoAug (p = 0.10) |
| model-generated | D3 diffusion (64 steps), Evo2 regeneration |

## What the numbers show (first-6 metric, so every cell has the same search budget)
- Every reservoir improves monotonically from 10k to 100k on the genomic test and on random sequences. Gain from 10k to 100k on the genomic test: +0.12 (motifs, JASPAR) to +0.19 (Zoonomia mammals); on random sequences +0.18 (motifs, JASPAR) to +0.29 (Zoonomia mammals).
- Genomic test: the motif reservoirs lead at every size (CT-enriched motifs 0.636 / 0.708 / 0.770 at 10k / 30k / 100k; JASPAR motifs 0.633 / 0.698 / 0.755), followed at 30k and 100k by Evo2 (0.694 / 0.750); at 10k EvoAug (paper settings, 0.609) comes next.
  The genomic reservoir itself is not the best training source for the genomic test (0.585 / 0.682 / 0.725); Zoonomia mammals is the lowest at 10k and 30k.
- Random sequences: the random reservoir leads at every size (0.564 / 0.671 / 0.757), followed by GC-matched random (0.505 / 0.648 / 0.748) and the motif reservoirs; genomic and the Zoonomia sets are among the lowest (genomic 0.394 / 0.565 / 0.621). The best training set depends on what it is tested on.
- Designed high-activity set: low for everyone (at most 0.35 at 100k); JASPAR motifs lead at 10k and 100k (0.168 / 0.348), the genomic reservoir is the lowest at 10k (0.029).
- SNV effect (alt minus ref): motif reservoirs lead at 100k (CT-enriched 0.583, JASPAR 0.567, random 0.563) and JASPAR at 10k (0.416); the Zoonomia sets are the lowest at 100k (0.49) and genomic is among the lowest at 10k (0.272).
- GC sweep is easy for every reservoir (0.82 to 0.94); GC-matched random leads at 10k and 100k (0.895 / 0.942).
- The deploy ensemble adds 0.03 to 0.10 (mean 0.06) over the mean single model on the genomic test and 0.05 to 0.16 (mean 0.10) on random sequences; part of that is that the ensemble draws on more than 6 models.

## Caveats (please read before quoting numbers)
1. Pools are independently sampled at each D, not nested (10k is not a subset of 30k; synthetic reservoirs are fully disjoint). The curves therefore mix the effect of more data with a different sample. Nested pools were adopted on Oct 14 and used from then on.
2. Cells differ in how many models were trained (see `n_models`; many 100k cells have 6 to 14, genomic 10k and 30k have 60). Ensemble numbers (`enet_*`) therefore favour cells with more models; for example the genomic reference gains between 10k and 30k partly because its 30k cell has 60 models.
   Use `f4_first6` to compare reservoirs, the ensembles to see absolute levels.
3. One search run per cell, no replicates. Run-to-run noise of a single search is about 0.01 in ensemble r, so differences below that between neighbouring reservoirs are not meaningful.
4. The oracle is itself an AlphaGenome-based ensemble; scores are against its labels (agreement of the oracle with the experiment on held-out chromosomes is r 0.93).
5. The students of this setup saw the 200-bp insert only (no plasmid flank) and used the early protocol (patience 8, validation labels of an earlier oracle labelling, LegNet-style search space).

## What has changed since Oct 14
- Pools are strictly nested and independent per reservoir; the search now uses a fixed 20-architecture menu retrained at every size; validation is relabelled with the current oracle (V0c); patience 16.
- On Oct 29 we found that giving the student the real plasmid flank as input (20 bases each side) lifts single-model r by about +0.07 on the genomic test and by +0.04 to +0.12 on every other evaluation set (largest for the SNV effect); the rankings above have not yet been re-measured with it (running).
- Current scaling results (10k to 1M, 11 reservoirs, nested pools, menu v1) are on the Evidence Ledger page and in `slide_figures/scaling/` of the repository.

"""


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
    contact_sheet(T, "enet_greedy_k5", f"{a.out}/figures/scaling_all_eval_sets_deploy_ensemble.png")
    contact_sheet(T, "f4_first6", f"{a.out}/figures/scaling_all_eval_sets_first6_single.png")
    for m in ("enet_greedy_k5", "f4_first6", "top8_by_val", "best_single", "enet_all_pos"):
        wide_csv(T, m, f"{a.out}/tables/scaling_wide_{m}.csv")
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
    with open(f"{a.out}/README.md", "w") as fh:
        fh.write(README + open(f"{a.out}/tables/scaling_tables.md").read())
    print("done", a.out)


if __name__ == "__main__":
    main()
