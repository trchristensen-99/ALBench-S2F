# ruff: noqa
#!/usr/bin/env python
"""Slide-ready headline figures (16:9, 2667 x 1500) for the Evidence Ledger's top section: one message per figure, the claim as the title, error bars on every estimate, the noise floor drawn where a
difference is claimed. Reads the analysis JSONs in slide_figures/benchmarks and writes slide_figures/deck/*.png. Re-run after new cells finish; missing arms are skipped and the point counts in the labels
say how many configurations stand behind each estimate.

    python scripts/analysis/make_deck_figures.py [--data slide_figures/benchmarks] [--out slide_figures/deck]
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, LINE = "#14202e", "#5a6878", "#dde3ea"
BLUE, ORANGE, GREEN, PURPLE, GREY, RED = (
    "#1d6fa5",
    "#c4571a",
    "#2a7d4f",
    "#6a4fa3",
    "#8b97a6",
    "#b0382c",
)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 15,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": LINE,
        "grid.linewidth": 0.9,
        "axes.axisbelow": True,
    }
)
NAMES = {
    "lrx05": "learning rate x0.5",
    "lrx2": "learning rate x2",
    "bs128": "batch size 128 (x0.5)",
    "bs512": "batch size 512 (x2)",
    "wd1e3": "weight decay 1e-3",
    "adamw": "AdamW lr 1e-3 wd 1e-2 (not Muon)",
    "lgrecipe": "LegNet recipe (AdamW lr 1e-2 wd 0.1)",
    "flank_shift": "shift over real flank, ±10",
    "c0s30": "shift over real flank, ±30",
    "ema": "weight averaging (decay 0.9995)",
    "bn": "BatchNorm recalculation only",
    "emabn": "weight averaging + BN recalculation",
    "ema999bn": "weight averaging 0.999 + BN recalc.",
    "c4s10": "4 real bases each side",
    "c8s10": "8 real bases each side",
    "c12s10": "12 real bases each side",
    "c20s10": "20 real bases each side",
    "c20s0": "20 bases, no shift",
    "c20s30": "20 bases, shift ±30",
    "c60s10": "60 real bases each side",
    "c100s10": "100 real bases each side",
    "c200s10": "200 bases each side (the oracle's 600 bp)",
    "c20s10_emabn": "20 bases + weight averaging + BN recalc.",
}


def slide(title, sub, footer, figsize=(13.333, 7.5)):
    fig = plt.figure(figsize=figsize)
    fig.text(
        0.04,
        0.945,
        "\n".join(textwrap.wrap(title, 56)),
        fontsize=25,
        fontweight="bold",
        color=INK,
        va="top",
        linespacing=1.15,
    )
    fig.text(
        0.04,
        0.835,
        "\n".join(textwrap.wrap(sub, 100)),
        fontsize=14.5,
        color=MUTED,
        va="top",
        linespacing=1.3,
    )
    fig.text(
        0.04,
        0.012,
        "\n".join(textwrap.wrap(footer, 150)),
        fontsize=11,
        color=MUTED,
        va="bottom",
        linespacing=1.25,
    )
    return fig


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print("wrote", path)


def ctx_note(P, key):
    n = P["n_models"].get(key, 0)
    return n


def fig_flank(P, out):
    pr, floor = P["paired"], P["seed_floor"]["genomic"]
    arms = [
        ("flank_shift", 0),
        ("c4s10", 4),
        ("c8s10", 8),
        ("c12s10", 12),
        ("c20s10", 20),
        ("c60s10", 60),
        ("c100s10", 100),
        ("c200s10", 200),
    ]
    arms = [(k, f) for k, f in arms if k in pr and pr[k]["n"] >= 1]
    if not any(f > 0 for _, f in arms):
        return
    fig = slide(
        "Giving the student the real plasmid flank adds about +0.07 test r",
        "Paired change in test r (same architecture, same seed, same 30k genomic training pool) when the 200-bp insert is shown with extra real plasmid sequence on each side. The oracle that labels every sequence sees 200 bases each side.",
        "menu v1 configurations that can take wider input (15 of 20); error bars = standard error over configurations; grey band = twice the seed-to-seed standard error. Genomic test chr7/13, oracle_v3 labels; experimental labels agree.",
    )
    ax = fig.add_axes([0.075, 0.20, 0.89, 0.50])
    xs = list(range(len(arms)))
    ax.axhspan(
        -2 * floor["se"],
        2 * floor["se"],
        color=GREY,
        alpha=0.25,
        lw=0,
        label="seed-to-seed noise (2 SE)",
    )
    ax.axhline(0, color=INK, lw=1)
    for key, col, mk, lab in (
        ("genomic", BLUE, "o", "genomic test (oracle labels)"),
        ("random_32k", PURPLE, "s", "random sequences"),
        ("genomic_measured", ORANGE, "D", "genomic test vs experimental labels"),
    ):
        m = np.array([pr[k][key]["mean"] for k, _ in arms])
        se = np.array([0 if np.isnan(pr[k][key]["se"]) else pr[k][key]["se"] for k, _ in arms])
        off = {"genomic": -0.12, "random_32k": 0.0, "genomic_measured": 0.12}[key]
        ax.errorbar(
            np.array(xs) + off,
            m,
            yerr=se,
            fmt=mk,
            color=col,
            ms=10,
            capsize=4,
            lw=2,
            label=lab,
            mfc=col if key != "genomic_measured" else "white",
            mew=2,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [
            f"{('no extra bases' + chr(10) + '(shift on real flank)') if f == 0 else f} "
            + (f"\nn={pr[k]['n']}" if f else f"\nn={pr[k]['n']}")
            for k, f in arms
        ],
        fontsize=12.5,
    )
    ax.set_xlabel("real flank bases per side given as input", fontsize=15, labelpad=8)
    ax.set_ylabel("change in test Pearson r", fontsize=15)
    ax.legend(loc="lower right", fontsize=12.5, frameon=False)
    save(fig, os.path.join(out, "deck_flank_context.png"))


def fig_scatter(P, out):
    M = P["models"]
    if not any(m["arm"].startswith("c") and m["arm"][1].isdigit() for m in M):
        return
    grp = {
        "no context (menu v1 and recipe variants)": ([], GREY),
        "shift on real flank, no extra input": ([], BLUE),
        "weight averaging": ([], GREEN),
        "real flank as input": ([], ORANGE),
    }
    for m in M:
        a = m["arm"]
        if a in (
            "base",
            "base_s1",
            "lrx05",
            "lrx2",
            "bs128",
            "bs512",
            "wd1e3",
            "adamw",
            "lgrecipe",
        ):
            grp["no context (menu v1 and recipe variants)"][0].append(m)
        elif a in ("flank_shift", "c0s30"):
            grp["shift on real flank, no extra input"][0].append(m)
        elif a in ("ema", "bn", "emabn", "ema999bn"):
            grp["weight averaging"][0].append(m)
        elif a.startswith("c") and a[1].isdigit():
            grp["real flank as input"][0].append(m)
    fig = slide(
        "Single models with flank context reach the collaborator's range",
        "Each dot is one trained model (30k genomic training pool). Models that see the real flank have the validation-to-test gap of the collaborator's best model; no earlier model of ours came near it.",
        "x: validation r on V0c (chr19/21/X, oracle labels); y: genomic test r (chr7/13, oracle labels). Star: collaborator's best single model as reported (own harness, files not yet compared).",
    )
    ax = fig.add_axes([0.075, 0.18, 0.89, 0.52])
    for lab, (ms, col) in grp.items():
        if ms:
            ax.scatter(
                [m["val"] for m in ms],
                [m["genomic"] for m in ms],
                s=46,
                color=col,
                alpha=0.8,
                label=f"{lab} (n={len(ms)})",
                edgecolor="white",
                lw=0.5,
            )
    ax.scatter(
        [0.854],
        [0.8165],
        marker="*",
        s=420,
        color=RED,
        zorder=5,
        label="collaborator's best single (val 0.854, test 0.8165)",
    )
    ax.set_xlabel("validation Pearson r (V0c)", fontsize=15)
    ax.set_ylabel("genomic test Pearson r", fontsize=15)
    ax.legend(loc="lower right", fontsize=12.5, frameon=False)
    save(fig, os.path.join(out, "deck_single_models.png"))


def fig_forest(P, out):
    pr, floor = P["paired"], P["seed_floor"]["genomic"]
    order = [
        "lrx05",
        "lrx2",
        "bs128",
        "bs512",
        "wd1e3",
        "lgrecipe",
        "adamw",
        "c0s30",
        "flank_shift",
        "bn",
        "ema",
        "ema999bn",
        "emabn",
        "c4s10",
        "c8s10",
        "c12s10",
        "c20s0",
        "c20s30",
        "c20s10",
        "c60s10",
        "c100s10",
        "c200s10",
        "c20s10_emabn",
    ]
    rows = [k for k in order if k in pr and pr[k]["n"] >= 1]
    fig = slide(
        "What moves a single model: what it sees (+0.07) and weight averaging (+0.01); the training recipe does not",
        "Paired change in genomic test r against the unchanged menu v1 (same 20 architectures, same seed, V0c validation). Dots right of the grey band beat seed-to-seed noise.",
        "error bars = standard error over configurations; n = configurations in the arm; grey band = twice the seed-to-seed standard error (second seed of the unchanged recipe minus the first).",
        figsize=(13.333, 8.6),
    )
    ax = fig.add_axes([0.35, 0.13, 0.62, 0.62])
    y = np.arange(len(rows))[::-1]
    grpcol = {"recipe": GREY, "flank": BLUE, "averaging": GREEN, "context": ORANGE, "combined": RED}
    ax.axvspan(-2 * floor["se"], 2 * floor["se"], color=GREY, alpha=0.25, lw=0)
    ax.axvline(0, color=INK, lw=1)
    for yy, k in zip(y, rows):
        m, se = pr[k]["genomic"]["mean"], pr[k]["genomic"]["se"]
        ax.errorbar(
            [m],
            [yy],
            xerr=[0 if np.isnan(se) else se],
            fmt="o",
            color=grpcol[P["group"][k]],
            ms=9,
            capsize=3,
            lw=2,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{NAMES[k]}" + (f"  (n={pr[k]['n']})" if pr[k]["n"] < 20 else "") for k in rows],
        fontsize=12.5,
    )
    ax.set_xlabel("change in genomic test r against the unchanged menu", fontsize=14)
    ax.grid(axis="y", alpha=0)
    save(fig, os.path.join(out, "deck_what_moves_a_model.png"))


def fig_ladder(P, B, out):
    E = P["ensembles"]
    ctxm = [m for m in P["models"] if m["arm"][0] == "c" and m["arm"][1].isdigit()]
    rows = [
        ("MPRA-LegNet, one model (tuned)", 0.757, 0.720, GREY, True),
        ("MPRA-LegNet, 20-seed ensemble", 0.782, 0.742, GREY, True),
        ("Our menu v1 (20 architectures), ensemble", 0.782, 0.743, BLUE, True),
        ("All our earlier 30k runs pooled (1,297 models)", 0.798, 0.758, BLUE, True),
    ]
    if ctxm:
        b = max(ctxm, key=lambda m: m["val"])
        rows.append(
            (
                f"Ours with flank context: best single by validation (of {len(ctxm)} models so far)",
                b["genomic"],
                b["genomic_measured"],
                ORANGE,
                True,
            )
        )
    if "c20s10" in E and E["c20s10"]["n"] >= 10:
        e = E["c20s10"]
        rows.append(
            (
                f"Ours with flank context: ensemble of {e['n']} architectures",
                e["genomic"],
                e["genomic_measured"],
                ORANGE,
                True,
            )
        )
    if "pool_all_menu_arms" in E and ctxm:
        e = E["pool_all_menu_arms"]
        rows.append(
            (
                f"Ours, every 30k model including flank context ({e['n']})",
                e["genomic"],
                e["genomic_measured"],
                ORANGE,
                True,
            )
        )
    rows += [
        ("Collaborator's search: best single (own harness)", 0.8165, None, RED, False),
        ("Collaborator's search: top-5 ensemble", 0.8317, None, RED, False),
    ]
    fig = slide(
        "With the flank as input our models reach the collaborator's range; the old menu sat 0.05 below",
        "Genomic test r (chr7/13, 30k genomic training sequences) against the oracle labels (filled) and the experimental labels (open). AlphaGenome-FT-MPRA scores 0.96 / 0.92 because it fine-tunes the oracle's own backbone.",
        "Collaborator numbers as reported on Slack (test set and label column not yet confirmed identical to ours). Ours: ensembles are positive ElasticNet stacks fitted on V0c; no test set was used for any choice.",
        figsize=(13.333, 8.2),
    )
    ax = fig.add_axes([0.40, 0.14, 0.57, 0.58])
    y = np.arange(len(rows))[::-1]
    for yy, (lab, o, m, col, ok) in zip(y, rows):
        ax.plot([0.68, o], [yy, yy], color=col, lw=1.4, alpha=0.45)
        ax.scatter(
            [o],
            [yy],
            s=170,
            color=col,
            zorder=3,
            edgecolor="white",
            lw=1,
            marker="o" if ok else "*" if False else "o",
        )
        ax.text(o + 0.004, yy + 0.18, f"{o:.3f}", fontsize=12.5, color=INK, va="bottom")
        if m is not None and not np.isnan(m):
            ax.scatter(
                [m], [yy], s=130, facecolor="white", edgecolor=col, lw=2.2, zorder=3, marker="D"
            )
    ax.set_yticks(y)
    ax.set_yticklabels(["\n".join(textwrap.wrap(r[0], 46)) for r in rows], fontsize=12)
    ax.set_xlim(0.68, 0.87)
    ax.set_xlabel("Pearson r on the genomic test set", fontsize=14)
    ax.grid(axis="y", alpha=0)
    save(fig, os.path.join(out, "deck_benchmark_ladder.png"))


def fig_fairness(P, out):
    F = P.get("fairness", {})
    full = {r: v for r, v in F.items() if v["complete"]}
    if (
        len(full) < 4
    ):  # not enough reservoirs yet: a status slide keeps the numbering and the question visible
        fig = slide(
            "Does flank context change which training strategy looks best? Being measured now",
            "The same menu configurations are retrained on 30k sequences from ten other training-sequence generators, with and without 20 real flank bases of input. The plot appears here as the cells finish.",
            "Fairness check for the corpus recommendation: if context lifts every strategy about equally, the comparison between strategies is unchanged; if it favours some, the ranking is redone with context.",
        )
        ax = fig.add_axes([0.075, 0.18, 0.89, 0.52])
        ax.axis("off")
        names = [
            "random",
            "motif CT-enriched",
            "EvoAug",
            "Zoonomia orthologs",
            "motif shared core",
            "random GC sweep",
            "mutagenesis 10%",
            "D3 diffusion 64 steps",
            "D3 diffusion 256 steps",
            "Evo2 regeneration",
        ]
        for k, nm in enumerate(names):
            ax.text(
                0.02 + (k // 5) * 0.5,
                0.9 - (k % 5) * 0.18,
                f"{nm}",
                fontsize=17,
                color=INK,
                va="top",
            )
            ax.text(
                0.02 + (k // 5) * 0.5,
                0.9 - (k % 5) * 0.18 - 0.07,
                "15 architectures, with vs without context: training",
                fontsize=12.5,
                color=MUTED,
                va="top",
            )
        save(fig, os.path.join(out, "deck_fairness_flank.png"))
        return
    fig = slide(
        "Flank context lifts every reservoir; the ranking of strategies is what we check next",
        "Ensemble r of the same menu configurations trained on 30k sequences from each strategy, without (x) and with (y) 20 real flank bases of input. Points above the diagonal gain from the context.",
        "15 configurations that can take wider input; stacks fitted on each run's own validation labels; test sets: genomic chr7/13, designed high-activity, random 32k (oracle labels).",
    )
    sets = [
        ("genomic", "genomic test"),
        ("ood", "designed high-activity"),
        ("random_32k", "random sequences"),
    ]
    for j, (s, t) in enumerate(sets):
        ax = fig.add_axes([0.06 + j * 0.325, 0.19, 0.27, 0.50])
        xs = np.array([v["without"][s] for v in full.values()])
        ys = np.array([v["with"][s] for v in full.values()])
        lo, hi = min(xs.min(), ys.min()) - 0.02, max(xs.max(), ys.max()) + 0.02
        ax.plot([lo, hi], [lo, hi], color=GREY, lw=1.2)
        ax.scatter(xs, ys, s=70, color=BLUE, edgecolor="white")
        for r, x, y in zip(full, xs, ys):
            ax.annotate(
                r.replace("_", " ")[:14],
                (x, y),
                fontsize=8.5,
                xytext=(4, 3),
                textcoords="offset points",
                color=MUTED,
            )
        rk = P.get("fairness_rank", {}).get(s)
        ax.set_title(t + (f"  (rank correlation {rk:.2f})" if rk is not None else ""), fontsize=13)
        ax.set_xlabel("without context", fontsize=12.5)
        if j == 0:
            ax.set_ylabel("with 20 flank bases", fontsize=12.5)
    save(fig, os.path.join(out, "deck_fairness_flank.png"))


def fig_manager(Mg, out):
    runs, pa = Mg["runs"], Mg["paired"]
    arms = [
        ("c15", "coordinator only", GREY),
        ("s15", "+ explore-then-exploit schedule", BLUE),
        ("e15", "+ schedule and ensemble status", GREEN),
    ]
    fig = slide(
        "An explore-then-exploit manager helps a little (+0.006) but not by more than the noise",
        "One AutoResearch run = 30 trained models on a replicate 30k genomic pool; six runs per arm on the same six pools. Dots are the run's stacked-ensemble genomic test r; lines join paired runs.",
        "paired difference (schedule minus coordinator-only): +0.0062, standard error 0.0032, 4 of 6 runs better; ensemble-status arm: -0.0003 (20 models). Noise floor of a single run about 0.006.",
    )
    ax = fig.add_axes([0.075, 0.21, 0.5, 0.49])
    reps = sorted(runs["c15"])
    for r in reps:
        ys = [runs[a][r]["stack30"]["genomic"] for a, _, _ in arms if r in runs[a]]
        ax.plot(range(len(ys)), ys, color=GREY, lw=1.2, alpha=0.7)
    for i, (a, lab, col) in enumerate(arms):
        ys = [runs[a][r]["stack30"]["genomic"] for r in reps if r in runs[a]]
        ax.scatter([i] * len(ys), ys, s=80, color=col, zorder=3, edgecolor="white")
        ax.errorbar(
            [i + 0.18],
            [np.mean(ys)],
            yerr=[np.std(ys, ddof=1) / np.sqrt(len(ys))],
            fmt="D",
            color=INK,
            ms=9,
            capsize=4,
        )
    ax.set_xticks(range(3))
    ax.set_xticklabels(["\n".join(textwrap.wrap(l, 18)) for _, l, _ in arms], fontsize=12.5)
    ax.set_ylabel("genomic test r of the run's ensemble", fontsize=14)
    ax.set_xlim(-0.4, 2.5)
    ax2 = fig.add_axes([0.66, 0.21, 0.31, 0.49])
    items = [
        ("schedule vs coordinator-only\n(30 models)", pa["s15_minus_c15_stack30"]),
        ("schedule vs coordinator-only\n(first 20 models)", pa["s15_minus_c15_stack20"]),
        ("ensemble status vs coordinator-only\n(30 models)", pa["e15_minus_c15_stack30"]),
    ]
    for i, (l, p) in enumerate(items[::-1]):
        ax2.errorbar(
            [p["genomic"]["mean"]],
            [i],
            xerr=[p["genomic"]["se"]],
            fmt="o",
            color=BLUE,
            ms=10,
            capsize=4,
            lw=2,
        )
    ax2.axvline(0, color=INK, lw=1)
    ax2.axvspan(-0.004, 0.004, color=GREY, alpha=0.25, lw=0)
    ax2.set_yticks(range(3))
    ax2.set_yticklabels([l for l, _ in items[::-1]], fontsize=11.5)
    ax2.set_xlabel("paired change in test r", fontsize=13)
    ax2.grid(axis="y", alpha=0)
    save(fig, os.path.join(out, "deck_manager.png"))


CTX_COLS = [
    ("shift_only", "0*"),
    ("c4", "4"),
    ("c8", "8"),
    ("c12", "12"),
    ("c20", "20"),
    ("c60", "60"),
    ("c100", "100"),
    ("c200", "200"),
]
SET_NAMES = {
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


def fig_context_abs(C, out):
    A = C["arms"]
    cols = [(k, lab) for k, lab in CTX_COLS if k in A and A[k]["n"] >= 2]
    if len(cols) < 3:
        return
    fig = slide(
        "More context helps up to about 20 bases per side, then it stops",
        "Mean single-model test r (solid) and the ensemble of the same architectures (open) with the given amount of real plasmid flank in the input, against the same architectures without it (grey). Genomic test, designed high-activity, random sequences and the SNV effect.",
        "menu v1, 30k genomic training pool, oracle_v3 labels; each point uses the architectures that have finished for that arm (n under the tick), the grey reference is the unchanged menu on the same architectures; ensembles are positive ElasticNet stacks fitted on V0c. 0* = no extra bases, shift over real flank only.",
    )
    sets = ["genomic", "ood", "random_32k", "snv_delta"]
    for j, st in enumerate(sets):
        ax = fig.add_axes([0.06 + j * 0.24, 0.20, 0.20, 0.50])
        xs = np.arange(len(cols))
        single = np.array([A[k]["single"][st] for k, _ in cols])
        ref = np.array(
            [
                A[k]["single"][st] - (A[k]["delta"][st]["mean"] if "delta" in A[k] else 0.0)
                for k, _ in cols
            ]
        )
        ax.plot(xs, ref, "--o", color=GREY, ms=5, lw=1.6, label="no context")
        ax.plot(xs, single, "-o", color=BLUE, ms=7, lw=2.4, label="with context, mean single model")
        ens = [
            (i, A[k]["ens"][st], A[k]["ens_base_same"][st])
            for i, (k, _) in enumerate(cols)
            if "ens" in A[k]
        ]
        if ens:
            ax.plot(
                [e[0] for e in ens],
                [e[1] for e in ens],
                "-D",
                mfc="white",
                color=ORANGE,
                ms=7,
                lw=2,
                label="with context, ensemble",
            )
            ax.plot(
                [e[0] for e in ens],
                [e[2] for e in ens],
                ":D",
                mfc="white",
                color=GREY,
                ms=5,
                lw=1.4,
                label="no context, ensemble",
            )
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{lab}\nn={A[k]['n']}" for k, lab in cols], fontsize=8.5)
        ax.set_title(SET_NAMES[st], fontsize=13)
        ax.set_xlabel("real flank bases per side", fontsize=11)
        if j == 0:
            ax.set_ylabel("Pearson r against the oracle", fontsize=12)
            ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    save(fig, os.path.join(out, "deck_context_effect_absolute.png"))


def fig_context_heat(C, out):
    A = C["arms"]
    cols = [(k, lab) for k, lab in CTX_COLS if k in A and A[k]["n"] >= 2]
    if len(cols) < 3:
        return
    rows = [
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
    M = np.array([[A[k]["delta"][r]["mean"] for k, _ in cols] for r in rows])
    fig = slide(
        "Every evaluation set gains from context, the SNV effect most (+0.12 with 20 bases per side)",
        "Paired change in single-model Pearson r (same architecture, same seed) from adding real plasmid flank to the input, on all 18 evaluation sets of the battery and the SNV effect. Darker = larger gain.",
        "menu v1, 30k genomic training pool, oracle_v3 labels; n = architectures that have finished for the arm (15 can take wider input); the first column is the shift over real flank without extra input.",
        figsize=(13.333, 9.0),
    )
    ax = fig.add_axes([0.25, 0.11, 0.72, 0.60])
    im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=-0.01, vmax=max(0.13, np.nanmax(M)))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(
                j,
                i,
                f"{M[i, j]:+.3f}",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if M[i, j] > 0.075 else INK,
            )
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{lab}\nn={A[k]['n']}" for k, lab in cols], fontsize=10)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([SET_NAMES[r] for r in rows], fontsize=11)
    ax.set_xlabel(
        "real flank bases per side given as input (first column: shift over real flank, no extra input)",
        fontsize=11,
        labelpad=6,
    )
    ax.xaxis.set_label_position("bottom")
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axhline(7.5, color="white", lw=3)
    ax.axhline(4.5, color="white", lw=3)
    save(fig, os.path.join(out, "deck_context_effect_allsets.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="slide_figures/benchmarks")
    ap.add_argument("--out", default="slide_figures/deck")
    a = ap.parse_args()
    P = json.load(open(f"{a.data}/data_pk35.json"))
    Mg = json.load(open(f"{a.data}/data_mgr34.json"))
    fig_flank(P, a.out)
    cp = f"{a.data}/data_context_allsets.json"
    if os.path.exists(cp):
        C = json.load(open(cp))
        fig_context_abs(C, a.out)
        fig_context_heat(C, a.out)
    fig_scatter(P, a.out)
    fig_forest(P, a.out)
    fig_ladder(P, None, a.out)
    fig_fairness(P, a.out)
    fig_manager(Mg, a.out)


if __name__ == "__main__":
    main()
