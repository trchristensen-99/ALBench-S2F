# ruff: noqa
#!/usr/bin/env python
"""Slide figures that justify the procedure (Oct 28). Reads the analysis JSONs fetched from the cluster, writes PNGs.

python scripts/analysis/make_slide_figures.py --in <dir with data_*.json, _cadence_genomic.csv> --out slide_figures/justification [--only rounds,patience,...]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.dpi": 100,
    }
)
BLUE, ORANGE, GREEN, PURPLE, GREY, DARK = (
    "#1b6ca8",
    "#c4571a",
    "#2a8f5b",
    "#7a5195",
    "#8a949a",
    "#1c2530",
)
SETN = {"genomic": "genomic test", "ood": "designed OOD", "random_32k": "random sequences"}
SETS = ("genomic", "ood", "random_32k")


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name), dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def J(inp, f):
    return json.load(open(os.path.join(inp, f)))


# ------------------------------------------------------------------------------------------------------------------------------------ rounds
def fig_rounds(inp, out):
    d = J(inp, "data_rounds.json")
    per, pooled = d["per_run"], d["pooled"]
    NS = [10, 20, 30, 40, 60]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axs[0]
    for v in per.values():
        ys = [v[str(n)]["stack_all"]["genomic"] if str(n) in v else np.nan for n in NS]
        ax.plot(NS, ys, color=BLUE, alpha=0.18, lw=1)
    mean = [
        np.mean([v[str(n)]["stack_all"]["genomic"] for v in per.values() if str(n) in v])
        for n in NS
    ]
    sd = [
        np.std([v[str(n)]["stack_all"]["genomic"] for v in per.values() if str(n) in v], ddof=1)
        for n in NS
    ]
    ax.fill_between(
        NS,
        np.array(mean) - sd,
        np.array(mean) + sd,
        color=BLUE,
        alpha=0.18,
        lw=0,
        label="±1 SD across the 15 runs",
    )
    ax.plot(NS, mean, "o-", color=BLUE, lw=2.6, label="mean of 15 runs")
    ax.set_xlabel("proposals in the run (rounds × 2)")
    ax.set_ylabel("ensemble r, genomic test")
    ax.set_title("One run: ensemble of its own models", fontsize=11)
    ax.axvline(20, color=ORANGE, ls="--", lw=1.4)
    ax.text(20.6, ax.get_ylim()[1] - 0.001, "10 rounds\n(used)", color=ORANGE, fontsize=9, va="top")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax = axs[1]
    cols = [BLUE, GREEN, PURPLE, GREY, ORANGE]
    for (fam, v), c in zip(pooled.items(), cols):
        xs = [n for n in (10, 20, 30, 60) if str(n) in v]
        ax.plot(
            xs,
            [v[str(n)]["stack_all"]["genomic"] for n in xs],
            "o-",
            color=c,
            lw=1.8,
            label=fam.replace("son55_none", "no guidance").replace("son55_wide", "son55 wide"),
        )
    runs_by_fam = {}
    for name, v in per.items():
        runs_by_fam.setdefault(name.rsplit("/rep", 1)[0].rsplit("/", 1)[-1], []).append(v)
    for (fam, v), c in zip(pooled.items(), cols):
        fk = fam.rsplit("/", 1)[-1]
        for rv in runs_by_fam.get(fk, []):
            xs = [n for n in (10, 20, 30, 60) if str(n) in rv]
            ax.scatter(
                xs,
                [rv[str(n)]["stack_all"]["genomic"] for n in xs],
                color=c,
                s=14,
                alpha=0.35,
                lw=0,
            )
    ax.set_xlabel("proposals per run (line: 3 runs pooled; dots: each run alone)")
    ax.set_title("Three runs pooled, as the menu does", fontsize=11)
    ax.axvline(20, color=ORANGE, ls="--", lw=1.4)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax = axs[2]
    bv = [np.mean([v[str(n)]["best_val"] for v in per.values() if str(n) in v]) for n in NS]
    bt = [np.mean([v[str(n)]["best_test"] for v in per.values() if str(n) in v]) for n in NS]
    for key, mu, col in (("best_val", bv, PURPLE), ("best_test", bt, GREEN)):
        sd_ = np.array(
            [np.std([v[str(n)][key] for v in per.values() if str(n) in v], ddof=1) for n in NS]
        )
        ax.fill_between(NS, np.array(mu) - sd_, np.array(mu) + sd_, color=col, alpha=0.15, lw=0)
    ax.plot(NS, bv, "o-", color=PURPLE, label="best single, validation r (±1 SD)")
    ax.plot(NS, bt, "s-", color=GREEN, label="best single, test r (genomic; ±1 SD)")
    ax.set_xlabel("proposals in the run")
    ax.set_title("Best single model", fontsize=11)
    ax.axvline(20, color=ORANGE, ls="--", lw=1.4)
    ax.legend(frameon=False, fontsize=9)
    g20 = np.mean(
        [
            v["30"]["stack_all"]["genomic"] - v["20"]["stack_all"]["genomic"]
            for v in per.values()
            if "30" in v
        ]
    )
    gp = np.mean(
        [v["30"]["stack_all"]["genomic"] - v["20"]["stack_all"]["genomic"] for v in pooled.values()]
    )
    nup = sum(
        v["30"]["stack_all"]["genomic"] > v["20"]["stack_all"]["genomic"]
        for v in per.values()
        if "30" in v
    )
    fig.suptitle(
        f"More rounds help one run slightly (+{g20:.4f} ensemble r from 10 to 15 rounds, {nup} of 15 runs up; run-to-run SD about 0.01) and the pooled menu even less (+{gp:.4f}); 30k genomic pool, 15 runs of 30 rounds",
        fontsize=12,
        y=1.04,
    )
    save(fig, out, "justify_rounds.png")


# ------------------------------------------------------------------------------------------------------------------------------------ patience
def fig_patience(inp, out):
    m = J(inp, "data_main.json")["recipe2"]
    r3 = J(inp, "data_oct31.json").get("recipe3", {})
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.0), gridspec_kw={"width_ratios": [1.1, 1, 1.1]})
    ax = axs[0]
    sizes = {"10k": 1, "30k": 2, "100k": 3}
    colr = {"genomic": BLUE, "random": PURPLE, "motif_ct_enriched": GREEN}
    allv, ratio = {}, []
    for name, v in m.items():
        if name == "summary":
            continue
        res = re.match(r"(.+)_(\d+k)$", name)
        if not res:
            continue
        st, sz = res.group(1), res.group(2)
        ds = [p["delta"]["genomic"] for p in v["pools"]]
        ratio += [p["minutes_ratio"] for p in v["pools"] if p.get("minutes_ratio")]
        allv.setdefault(sz, []).extend(ds)
        ax.scatter(
            np.full(len(ds), sizes[sz]) + (list(colr).index(st) - 1) * 0.16,
            ds,
            color=colr.get(st, GREY),
            s=30,
            alpha=0.9,
            label=st.replace("motif_ct_enriched", "motif") if sz == "30k" else None,
        )
    for sz, xs in sizes.items():
        if sz in allv:
            ax.hlines(np.mean(allv[sz]), xs - 0.35, xs + 0.35, color=DARK, lw=2.2)
            ax.text(
                xs,
                -0.0155,
                f"mean {np.mean(allv[sz]):+.4f}\n(n={len(allv[sz])})",
                ha="center",
                fontsize=9,
            )
    ax.axhline(0, color=GREY, lw=1)
    ax.set_ylim(-0.017, 0.0135)
    ax.set_xticks(list(sizes.values()), list(sizes.keys()))
    ax.set_xlim(0.5, 3.5)
    ax.set_ylabel("patience 16 minus 8\n(ensemble r, genomic test)")
    ax.set_title("Replicate pools: 16 beats 8", fontsize=11)
    ax.legend(
        frameon=False,
        fontsize=8,
        loc="center right",
        title="pool source",
        title_fontsize=8,
        bbox_to_anchor=(1.0, 0.43),
    )
    sm = m["summary"]
    ax.text(
        0.02,
        0.97,
        f"all pools {sm['mean']:+.4f} (SE {sm['se']:.4f}), {sm['positive']} of {sm['n']} positive\ntraining time x{np.mean(ratio):.2f}",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
    )
    ax = axs[1]
    P = [8, 16, 24, 32]
    cc = {
        "genomic_10k": BLUE,
        "genomic_30k": "#4d94c9",
        "genomic_100k": "#0d3f66",
        "random_30k": PURPLE,
        "motif_30k": GREEN,
    }
    for cell, row in r3.items():
        ys = [
            row[str(p)]["ensemble"]["genomic"] - row["8"]["ensemble"]["genomic"]
            if str(p) in row
            else np.nan
            for p in P
        ]
        ax.plot(P, ys, "o-", color=cc.get(cell, GREY), label=cell.replace("_", " "), lw=1.8)
    ax.axvline(16, color=ORANGE, ls="--", lw=1.4)
    ax.axhline(0, color=GREY, lw=1)
    ax.set_xticks(P)
    ax.set_xlabel("early-stopping patience")
    ax.set_ylabel("change vs patience 8\n(ensemble r, genomic test)")
    ax.set_title("Most of the gain is by 16; 24 and 32 are mixed", fontsize=11)
    ax.set_ylim(-0.0005, 0.0175)
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 0.86))
    d24 = np.mean(
        [r3[c]["24"]["ensemble"]["genomic"] - r3[c]["16"]["ensemble"]["genomic"] for c in r3]
    )
    d32 = np.mean(
        [r3[c]["32"]["ensemble"]["genomic"] - r3[c]["16"]["ensemble"]["genomic"] for c in r3]
    )
    ax.text(
        0.03,
        0.97,
        f"mean change from 16: to 24 {d24:+.4f},\nto 32 {d32:+.4f} (single run per point)",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
    )
    ax = axs[2]
    w = 0.27
    cells = list(r3)
    for j, p in enumerate((16, 24, 32)):
        ax.bar(
            np.arange(len(cells)) + (j - 1) * w,
            [r3[c][str(p)]["ensemble"]["ood"] - r3[c]["8"]["ensemble"]["ood"] for c in cells],
            w,
            color=[ORANGE, "#9fb3bc", "#6c808b"][j],
            label=f"patience {p}",
        )
    ax.axhline(0, color=GREY, lw=1)
    ax.set_xticks(range(len(cells)), [c.replace("_", " ") for c in cells], rotation=20, fontsize=9)
    ax.set_ylabel("change vs patience 8\n(ensemble r, designed OOD)")
    npos = sum(r3[c]["16"]["ensemble"]["ood"] > r3[c]["8"]["ensemble"]["ood"] for c in cells)
    ax.set_title(
        f"Designed set: patience 16 is positive in {npos} of {len(cells)} cells", fontsize=11
    )
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.subplots_adjust(wspace=0.38)
    fig.suptitle(
        f"Early-stopping patience: 16 is clearly better than 8 ({sm['mean']:+.4f}, {sm['positive']} of {sm['n']} replicate pools); beyond 16 the single-run gains are inconsistent, so we keep 16",
        fontsize=12,
        y=1.02,
    )
    save(fig, out, "justify_patience.png")


# ------------------------------------------------------------------------------------------------------------------------------------ proposals per round
def fig_cadence(inp, out):
    rows = list(csv.DictReader(open(os.path.join(inp, "_cadence_genomic.csv"))))
    arms = [
        ("llm1x60", "1 proposal × 60 rounds\n(serial feedback)", GREY),
        ("llm2x30", "2 proposals × 30 rounds\n(used, 2 per round)", ORANGE),
        ("llm6x10", "6 proposals × 10 rounds", BLUE),
        ("llmctx", "2 × 30 with rich context", GREEN),
        ("multi6", "6 classical samplers\n(no LLM baseline)", PURPLE),
    ]
    res = ["genomic", "motif_ct_enriched", "random"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), sharey=False)
    for ax, r in zip(axs, res):
        sub = {x["variant"]: x for x in rows if x["reservoir"] == r}
        xs, ys, ns, cs = [], [], [], []
        for i, (a, lab, c) in enumerate(arms):
            if a in sub:
                (
                    xs.append(i),
                    ys.append(float(sub[a]["genomic_top8"])),
                    ns.append(int(sub[a]["n"])),
                    cs.append(c),
                )
        bars = ax.bar(xs, ys, color=cs)
        for b, n in zip(bars, ns):
            if n < 55:
                b.set_hatch("///"), b.set_alpha(0.6)
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.001,
                f"n={n}",
                ha="center",
                fontsize=8.5,
            )
        ax.set_ylim(min(ys) - 0.02, max(ys) + 0.012)
        ax.set_xticks(range(len(arms)), [a[1] for a in arms], rotation=35, ha="right", fontsize=8.5)
        ax.set_title(
            f"{r.replace('motif_ct_enriched', 'motif CT-enriched')} reservoir", fontsize=11
        )
        ax.set_ylabel("ensemble r, genomic test (top-8 greedy)")
    fig.suptitle(
        "Cadence study at a budget of about 60 models (Oct 13 harness, so absolute r is lower than the final pipeline): 2 and 6 proposals per round are indistinguishable;\nserial proposing (1 per round) is worse where it finished (motif, random). Hatched = fewer than 55 models finished, so weak evidence",
        fontsize=11.5,
        y=1.09,
    )
    save(fig, out, "justify_proposals_per_round.png")


def fig_evaluator(inp, out):
    L = J(inp, "data_main.json")["loopctl"]
    arms = [
        "default single call",
        "separate coder (no coordinator, no reflector)",
        "6 ideas, 2 picked at random",
        "6 ideas, 2 picked by the coordinator",
        "coordinator + reflector",
        "explore-style proposer in the agent loop",
    ]
    lab = [
        "default",
        "coder only",
        "random 2 of 6",
        "evaluator 2 of 6",
        "evaluator +\nreflector",
        "explore-style",
    ]
    col = [GREY, BLUE, "#9fb3bc", ORANGE, "#e08a5c", PURPLE]
    fig, axs = plt.subplots(1, 4, figsize=(18, 5.2), gridspec_kw={"width_ratios": [1, 1, 1, 1.5]})
    nruns = min(L["arms"][a]["n"] for a in arms)
    for k, (ax, s) in enumerate(zip(axs[:3], SETS)):
        ys = [L["arms"][a]["mean"][k] for a in arms]
        ax.bar(range(len(arms)), ys, color=col, alpha=0.85)
        runs_path = os.path.join(inp, "data_loopctl.json")
        if os.path.exists(runs_path):
            LR = json.load(open(runs_path))["loopctl"]["arms"]
            for ai, aname in enumerate(arms):
                rv = np.array([r[k] for r in LR[aname]["runs"]])
                ax.scatter(
                    ai + np.linspace(-0.22, 0.22, len(rv)),
                    rv,
                    s=12,
                    color="black",
                    alpha=0.55,
                    zorder=3,
                    lw=0,
                )
                ax.errorbar(
                    ai, rv.mean(), yerr=rv.std(ddof=1), color="black", lw=1.4, capsize=3, zorder=4
                )
            ys = [np.mean([r[k] for r in LR[a_]["runs"]]) for a_ in arms]
            lo = min(min(r[k] for r in LR[a_]["runs"]) for a_ in arms)
            hi = max(max(r[k] for r in LR[a_]["runs"]) for a_ in arms)
            ax.set_ylim(lo - 0.006, hi + 0.004)
        if not os.path.exists(os.path.join(inp, "data_loopctl.json")):
            ax.set_ylim(min(ys) - 0.01, max(ys) + 0.006)
        ax.set_xticks(
            range(len(arms)), lab, rotation=40, ha="right", fontsize=9, rotation_mode="anchor"
        )
        ax.set_title(SETN[s], fontsize=11)
        ax.set_ylabel(
            f"ensemble r (bar: mean of {nruns} runs; dots: runs; ±1 SD)" if k == 0 else ""
        )
    ax = axs[3]
    keys = list(L["pairs"])
    want = [
        ("6 ideas, 2 picked by the coordinator minus default", "evaluator + coder vs default"),
        (
            "coordinator minus random pick (same over-generation)",
            "evaluator + coder vs random pick\n+ coder (same 6 ideas)",
        ),
        (
            "coordinator minus separate coder alone",
            "evaluator + coder vs coder only\n(is the evaluator worth it?)",
        ),
        ("random pick minus separate coder alone", "random pick + coder vs coder only"),
    ]
    cols = [BLUE, ORANGE, GREEN]
    for ni, (key, nm) in enumerate(want):
        key = next(k for k in keys if k.startswith(key[:28]) and k.endswith(key[-12:]))
        for j, s in enumerate(SETS):
            v = L["pairs"][key][s]
            yy = (len(want) - 1 - ni) * 5 - j
            ax.errorbar(
                v["mean"],
                yy,
                xerr=[[v["mean"] - v["lo"]], [v["hi"] - v["mean"]]],
                fmt="o",
                color=cols[j],
                capsize=3,
                label=SETN[s] if ni == 0 else None,
            )
        ax.text(
            -0.0345,
            (len(want) - 1 - ni) * 5 + 0.7,
            nm,
            ha="left",
            va="bottom",
            fontsize=9,
            weight="bold",
        )
    ax.axvline(0, color=GREY, lw=1)
    ax.set_yticks([])
    ax.set_xlim(-0.036, 0.034)
    ax.set_ylim(-2.6, 18.2)
    ax.set_xlabel("paired difference in ensemble r (95% interval)")
    ax.set_title("What does the evaluator add?", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    fig.suptitle(
        f"Evaluator agent (picks 2 of 6 proposed ideas); every arm except the default also uses a separate coder; {nruns} runs per arm at 30k genomic: the combination beats the default one-call loop and a random pick,\nbut adds little over the coder alone (intervals include zero); the reflector adds nothing more; small samples, no multiple-comparison correction",
        fontsize=11,
        y=1.08,
    )
    fig.subplots_adjust(wspace=0.28)
    save(fig, out, "justify_evaluator_agent.png")


# ------------------------------------------------------------------------------------------------------------------------------------ ar2
def fig_ar2(inp, out):
    d31, d32 = J(inp, "data_oct31.json"), J(inp, "data_oct32.json")
    a2 = d31["ar2"]
    cells = list(a2)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, s in zip(axs, SETS):
        w = 0.2
        for j, (k, lab, c) in enumerate(
            (
                ("ar2_all_seeds", "AutoResearch v2 menu", ORANGE),
                ("v1_first_K", "menu v1, same number of configs", "#6aa6d1"),
                ("v1", "menu v1 (20)", BLUE),
                ("v0", "menu v0 (30)", GREY),
            )
        ):
            ys = [a2[c_][k]["ensemble"][s] if a2[c_].get(k) else np.nan for c_ in cells]
            ax.bar(np.arange(len(cells)) + (j - 1.5) * w, ys, w, color=c, label=lab)
        ax.set_xticks(
            range(len(cells)),
            [c_.replace("_", " ") for c_ in cells],
            rotation=30,
            ha="right",
            fontsize=9,
        )
        allv = [
            a2[c_][k]["ensemble"][s]
            for c_ in cells
            for k in ("ar2_all_seeds", "v1_first_K", "v1", "v0")
            if a2[c_].get(k)
        ]
        ax.set_ylim(min(allv) - 0.03, max(allv) + 0.012)
        ax.set_title(SETN[s], fontsize=11)
        ax.set_ylabel("ensemble Pearson r")
    axs[0].legend(frameon=False, fontsize=8, loc="upper left")
    gaps = [
        a2[c_]["ar2_all_seeds"]["ensemble"]["genomic"] - a2[c_]["v1_first_K"]["ensemble"]["genomic"]
        for c_ in cells
        if a2[c_].get("ar2_all_seeds") and a2[c_].get("v1_first_K")
    ]
    fig.suptitle(
        f"AutoResearch v2 menus retrained with our per-D procedure: on genomic test they sit {-max(gaps):.3f} to {-min(gaps):.3f} below the same number of menu-v1 configurations at all {len(gaps)} pools\n(AutoResearch v2 menus hold 4 to 5 configurations, trained over seeds; 100k pools use the menu from the 30k search)",
        fontsize=11.5,
        y=1.08,
    )
    save(fig, out, "ar2_ensembles.png")
    # search summary
    fam = d32["ar2_30k"]["families"]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1.3, 1, 1.1]})
    ax = axs[0]
    fam = sorted(fam, key=lambda f: -(f["confirmed_mean"] or f["screen_best"] or 0))
    stc = {"confirmed": GREEN, "confirming": "#7bbf9a", "retired": GREY, "parked": "#c9d3d8"}
    for i, f in enumerate(fam):
        v = f["confirmed_mean"] or f["screen_best"]
        ax.barh(i, v, color=stc.get(f["state"], GREY), xerr=f["confirmed_sd"] or 0, capsize=3)
        ax.text(
            v + (f["confirmed_sd"] or 0) + 0.004,
            i,
            f"{f['state']}, {f['n_runs']} runs",
            va="center",
            fontsize=8.5,
        )
    ax.set_yticks(range(len(fam)), [f["family"].replace("_", " ") for f in fam], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0.6, 0.88)
    ax.set_xlabel("validation r (confirmed mean of 3 seeds; else best single run)")
    ax.set_title("Genomic 30k search: families tried", fontsize=11)
    ax = axs[1]
    vs = {
        "genomic 10k": (0.0250, 0.0555, 0.0081),
        "genomic 30k": (0.0483, 0.0264, 0.0040),
        "random 30k": (0.0565, 0.0543, 0.0115),
        "motif 30k": (0.0401, 0.1116, 0.0044),
    }
    x = np.arange(len(vs))
    for j, (lab, c) in enumerate(
        (
            ("between families", BLUE),
            ("within families, across recipes", ORANGE),
            ("across seeds", GREY),
        )
    ):
        ax.bar(x + (j - 1) * 0.26, [v[j] for v in vs.values()], 0.26, color=c, label=lab)
    ax.set_xticks(x, list(vs), rotation=20)
    ax.set_ylabel("standard deviation of validation r")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Variance split in each search", fontsize=11)
    ax = axs[2]
    arch = ["menu-v1 config 1", "menu-v1 config 2"]
    ours, ar2 = [0.7659, 0.7630], [0.7359, 0.7193]
    ax.bar(np.arange(2) - 0.18, ours, 0.36, color=BLUE, label="our loop (Muon, shifts)")
    ax.bar(np.arange(2) + 0.18, ar2, 0.36, color=ORANGE, label="AutoResearch v2 loop")
    for i in range(2):
        ax.text(i + 0.18, ar2[i] + 0.002, f"{ar2[i] - ours[i]:+.3f}", ha="center", fontsize=9)
    ax.set_ylim(0.68, 0.78)
    ax.set_xticks(range(2), arch)
    ax.set_ylabel("validation r, genomic 10k")
    ax.set_title("Evaluator parity: not met", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    top = sorted(
        [
            (f["confirmed_mean"] or f["screen_best"])
            for f in fam
            if f["state"] in ("confirmed", "confirming")
        ],
        reverse=True,
    )
    fig.suptitle(
        f"AutoResearch v2 at genomic 30k: the top {len(top)} families are within {top[0] - top[-1]:.3f} validation r; recipe spread within a family beats the spread between families at genomic 10k and motif 30k;\nand its evaluator trains our two best configurations 0.03 to 0.04 below our own loop (the spec asks for at most 0.005)",
        fontsize=11,
        y=1.08,
    )
    save(fig, out, "ar2_search_summary.png")


# ------------------------------------------------------------------------------------------------------------------------------------ plateaus
def kcurves(K, D, rule="first2", method="top8"):
    cells = {k: v for k, v in K.items() if k.startswith("menu33|") and k.endswith(f"|{D}")}
    out = {}
    for e in SETS:
        per = []
        for v in cells.values():
            c = np.mean([p[rule][method][e] for p in v["pools"]], 0)
            ref = np.mean([p["val"]["top8"][e][-1] for p in v["pools"]], 0)
            per.append(c - ref)
        if per:
            n = min(len(x) for x in per)
            out[e] = np.mean([x[:n] for x in per], 0)
    return out


def fig_plateau(inp, out):
    K = J(inp, "data_main.json")["kcurve"]
    path = os.path.join(inp, "data_kspread.json")
    KS = json.load(open(path)) if os.path.exists(path) else {}
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.9))
    cols = {10000: "#6aa6d1", 30000: BLUE, 100000: "#0d3f66"}
    k12, k20 = {}, {}
    xs = np.arange(1, 31)
    for ax, e in zip(axs, SETS):
        for D, c in cols.items():
            cells = [v for k, v in KS.items() if k.endswith(f"|{D}")]
            if cells:

                def avg(key):
                    return np.mean([np.asarray(v[e][key]) for v in cells], 0)

                n = min(len(v[e]["mean"]) for v in cells)
                ax.fill_between(xs[:n], avg("p05")[:n], avg("p95")[:n], color=c, alpha=0.13, lw=0)
                ax.fill_between(xs[:n], avg("p25")[:n], avg("p75")[:n], color=c, alpha=0.22, lw=0)
                ax.plot(xs[:n], avg("mean")[:n], "-", color=c, lw=1.9, label=f"{D // 1000}k")
                if e == "genomic":
                    k12[D] = (float(avg("mean")[11]), float(avg("sd")[11]))
                    k20[D] = (float(avg("mean")[19]), float(avg("sd")[19]))
            cur = kcurves(K, D)
            if e in cur:
                ax.plot(
                    np.arange(1, len(cur[e]) + 1),
                    cur[e],
                    "--",
                    color=c,
                    lw=1.1,
                    alpha=0.9,
                    label="our ordering (best two of each search run first)"
                    if D == 30000 and e == "genomic"
                    else None,
                )
        ax.axhline(0, color=GREY, lw=1)
        ax.axvline(12, color=GREY, ls=":", lw=1.4)
        ax.axvline(20, color=ORANGE, ls="--", lw=1.4)
        ax.set_xlabel("configurations trained, K")
        ax.set_title(SETN[e], fontsize=11)
        ax.set_ylabel("ensemble r (best 8 of K) minus (best 8 of all 30)" if e == "genomic" else "")
        ax.set_ylim(-0.06, 0.012)
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=c, lw=2, label=f"{D // 1000}k, random draws of K")
        for D, c in cols.items()
    ]
    handles.append(
        Line2D(
            [0], [0], color=DARK, lw=1.2, ls="--", label="our ordering (best two of each run first)"
        )
    )
    axs[2].legend(handles=handles, frameon=False, loc="lower right", fontsize=8.5)
    axs[1].text(11.7, -0.057, "12: from 300k", color=GREY, fontsize=9, ha="right")
    axs[1].text(20.3, -0.057, "20: up to 100k", color=ORANGE, fontsize=9)
    if k12:
        fig.suptitle(
            "Number of configurations trained, K, drawn at random from the 30 retrained ones (300 draws x 3 pools per reservoir)\n"
            f"Genomic test, gap to using all 30 at K = 12 / K = 20: 10k {-k12[10000][0]:.4f} / {-k20[10000][0]:.4f}, 30k {-k12[30000][0]:.4f} / {-k20[30000][0]:.4f}, 100k {-k12[100000][0]:.4f} / {-k20[100000][0]:.4f}; "
            f"SD between draws at 30k {k12[30000][1]:.4f} / {k20[30000][1]:.4f}\n"
            "Bands: 25-75% and 5-95% over draws, pools and the three reservoirs; our ordering is ahead of random draws at small K",
            fontsize=10.5,
            y=1.13,
        )
    save(fig, out, "justify_models_plateau.png")
    R = J(inp, "data_main.json")["runsrc"]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    srcc = {"genomic": BLUE, "random": PURPLE, "motif": GREEN}
    gains, spread = [], []
    for ax, tgt in zip(axs, ("genomic", "random", "motif")):
        if tgt not in R:
            continue
        at15 = []
        for src, c in srcc.items():
            pts = sorted(
                (int(k.split("|")[1]), v["mean"][0])
                for k, v in R[tgt]["single"].items()
                if k.startswith(src + "|")
            )
            if pts:
                ax.plot(*zip(*pts), "o-", color=c, lw=1.8, label=f"runs searched on {src}")
                sds = sorted(
                    (int(k.split("|")[1]), v["sd"][0])
                    for k, v in R[tgt]["single"].items()
                    if k.startswith(src + "|")
                )
                ax.fill_between(
                    [x for x, _ in pts],
                    [y - d for (_, y), (_, d) in zip(pts, sds)],
                    [y + d for (_, y), (_, d) in zip(pts, sds)],
                    color=c,
                    alpha=0.12,
                    lw=0,
                )
                at15 += [y for x, y in pts if x == 15]
        bal = sorted((int(k), v["mean"][0]) for k, v in R[tgt].get("balanced", {}).items())
        if bal:
            bsd = sorted((int(k), v["sd"][0]) for k, v in R[tgt].get("balanced", {}).items())
            ax.fill_between(
                [x for x, _ in bal],
                [y - d for (_, y), (_, d) in zip(bal, bsd)],
                [y + d for (_, y), (_, d) in zip(bal, bsd)],
                color=ORANGE,
                alpha=0.15,
                lw=0,
            )
            ax.plot(*zip(*bal), "s--", color=ORANGE, lw=1.8, label="balanced across reservoirs")
            d = dict(bal)
            if 15 in d and 45 in d:
                gains.append(d[45] - d[15])
        if at15:
            spread.append(max(at15) - min(at15))
        ax.set_xlabel("search runs pooled")
        ax.set_title(f"target: {tgt} pool", fontsize=11)
        ax.set_ylabel("ensemble r, genomic test" if tgt == "genomic" else "")
    axs[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(
        f"Number of search runs pooled (each keeps its 3 best models, so 15 runs = 45 retrained configurations): no elbow, returns diminish slowly; tripling from 15 to 45 balanced runs adds {min(gains):+.3f} to {max(gains):+.3f}.\nAt 15 runs the reservoir the runs searched on moves the result by up to {max(spread):.3f}; searching on the target's own reservoir does not win. Bands: ±1 SD over 40 random draws of runs",
        fontsize=11,
        y=1.08,
    )
    save(fig, out, "justify_runs_plateau.png")


# ------------------------------------------------------------------------------------------------------------------------------------ what the menu is
def fig_priors(inp, out, menu_path):
    menu = json.load(open(menu_path))
    fams = {}
    for e in menu:
        fams.setdefault(e["family"], []).append(e)
    names = sorted(fams, key=lambda f: -len(fams[f]))
    pal = [GREEN, BLUE, PURPLE, GREY, ORANGE, "#b08a2e"]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1, 1.2, 1]})
    ax = axs[0]
    ax.barh(range(len(names)), [len(fams[f]) for f in names], color=pal[: len(names)])
    ax.set_yticks(
        range(len(names)),
        [n.replace(" (genomic + random + motif)", "\n(genomic + random + motif)") for n in names],
        fontsize=9,
    )
    ax.invert_yaxis()
    ax.set_xlabel("configurations in menu v1")
    ax.set_title("Which search run each came from", fontsize=11)
    ax = axs[1]
    for f, c in zip(names, pal):
        ax.scatter(
            [e["n_params"] / 1e6 for e in fams[f]],
            [float(e["orig_val"]) for e in fams[f]],
            color=c,
            s=42,
            label=f.split(" (")[0],
        )
    ax.set_xscale("log")
    ax.set_xticks([0.2, 0.5, 1, 2, 4], ["0.2", "0.5", "1", "2", "4"])
    ax.minorticks_off()
    ax.set_xlabel("parameters (millions)")
    ax.set_ylabel("validation r in its own search")
    p = np.array([e["n_params"] for e in menu], float) / 1e6
    v = np.array([float(e["orig_val"]) for e in menu])
    ax.set_title(
        f"Size spans {p.min():.1f}M to {p.max():.1f}M; r vs size: rho = {np.corrcoef(np.argsort(np.argsort(p)), np.argsort(np.argsort(v)))[0, 1]:+.2f}",
        fontsize=11,
    )
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    ax = axs[2]
    ax.axis("off")
    tc = menu[0]["train_config"]
    ep = {}
    for e in menu:
        ep[e["train_config"]["epochs"]] = ep.get(e["train_config"]["epochs"], 0) + 1
    same = all(
        all(
            e["train_config"][k] == tc[k]
            for k in ("optimizer", "lr", "weight_decay", "batch_size", "lr_schedule")
        )
        for e in menu
    )
    lines = [
        "Training recipe",
        f"all {len(menu)} configurations share it" if same else "recipes differ",
        "",
        f"optimizer        {tc['optimizer']}",
        f"learning rate    {tc['lr']}",
        f"weight decay     {tc['weight_decay']}",
        f"batch size       {tc['batch_size']}",
        f"schedule         {tc['lr_schedule']}",
        "epochs           " + ", ".join(f"{k} ({v})" for k, v in sorted(ep.items())),
        "patience         16 (set by us at retraining)",
    ]
    ax.text(0.0, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=10.5)
    fig.suptitle(
        "What is in the menu: 20 architectures from 18 search runs; what varies between members is the architecture, not the recipe",
        fontsize=12,
        y=1.03,
    )
    save(fig, out, "menu_composition.png")


def fig_prompt_loop(inp, out):
    """How the proposer prompt is built, updated and stored (read from albench/arch_search/propose.py and arch_autoresearch.py)."""
    import textwrap

    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(16, 8.2))
    ax.set_xlim(0, 16), ax.set_ylim(0, 8.8), ax.axis("off")

    def box(x, y, w, h, title, items, fc, ec, wrap):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.6
            )
        )
        ax.text(x + 0.15, y + h - 0.15, title, fontsize=11.5, weight="bold", va="top", color=DARK)
        body = "\n".join(
            textwrap.fill(i, wrap, initial_indent="• ", subsequent_indent="   ") for i in items
        )
        ax.text(x + 0.15, y + h - 0.65, body, fontsize=9.2, va="top", color=DARK, linespacing=1.4)

    def arrow(x0, y0, x1, y1, txt="", dx=0.0, dy=0.1):
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=1.8, color=GREY)
        )
        if txt:
            ax.text(
                (x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, txt, fontsize=9, ha="center", color=GREY
            )

    W, Y1, H1 = 3.8, 3.1, 4.25
    X = [0.2, 4.2, 8.2, 12.2]
    box(
        X[0],
        Y1,
        W,
        H1,
        "1  Prompt, rebuilt each round",
        [
            "system: design any architecture you can write in torch; return exactly N proposals as IDEA, WHY, code",
            "task: K562 MPRA regression, D training sequences",
            "contract: Net() with no arguments, one-hot in, one scalar out, parameter and CPU-time caps, import whitelist",
            "history: best 18 architectures so far (validation r, parameters, time) and up to 8 discarded proposals with the reason",
            "optional guidance, reference code, round note",
        ],
        "#e8f0f7",
        BLUE,
        40,
    )
    box(
        X[1],
        Y1,
        W,
        H1,
        "2  Proposer model",
        [
            "claude-sonnet-5-5 for the menu runs",
            "N = 2 proposals per round, 10 rounds",
            "returns the idea, why it might work, and the full Net source",
            "no temperature control through the CLI",
        ],
        "#f6ece4",
        ORANGE,
        40,
    )
    box(
        X[2],
        Y1,
        W,
        H1,
        "3  Gate and parse",
        [
            "every code block is kept, even without an idea line",
            "discarded on a contract violation: too many parameters, forward too slow, wrong shape, output ignores input, forbidden import",
            "the discard and its reason go into the next prompt",
        ],
        "#f3eef7",
        PURPLE,
        40,
    )
    box(
        X[3],
        Y1,
        3.6,
        H1,
        "4  Train and score",
        [
            "each accepted proposal is trained on the D sequences",
            "scored on the validation set only: no test feedback",
            "validation r, parameters and time are added to the history",
        ],
        "#eaf5ef",
        GREEN,
        37,
    )
    box(
        8.2,
        0.3,
        7.6,
        2.45,
        "5  What is stored",
        [
            "per model, rNN_meta.json: idea, why, source code, validation r, parameters, epochs, time",
            "per run, prompts/prompt_rNN.txt for the first round and the explore-phase round only; the other prompts are rebuilt from the stored models, which is also how a crashed run resumes",
            "nothing is edited by hand between rounds",
        ],
        "#f1f3f4",
        GREY,
        84,
    )
    box(
        0.2,
        0.3,
        7.8,
        2.45,
        "6  Between runs",
        [
            "each run starts from the same prompt with an empty history",
            "runs are independent; the menu pools the best architectures of 18 runs",
            "a cross-D hand-off (results from a smaller D pasted into the prompt) exists as an option and is not used for the menu",
        ],
        "#fbf4e0",
        "#b08a2e",
        88,
    )
    for i in range(3):
        arrow(X[i] + W, Y1 + H1 / 2, X[i + 1], Y1 + H1 / 2)
    arrow(X[3] + 1.8, Y1, X[3] + 1.8, 2.75, "write", dx=0.4, dy=0.0)
    ax.annotate(
        "",
        xy=(2.1, Y1 + H1),
        xytext=(14.0, Y1 + H1),
        arrowprops=dict(arrowstyle="-|>", lw=1.9, color=BLUE, connectionstyle="arc3,rad=0.07"),
    )
    ax.text(
        8.0,
        7.72,
        "validation r and discards update the history, which becomes the next round's prompt",
        fontsize=10,
        color=BLUE,
        ha="center",
    )
    ax.text(
        0.2,
        8.35,
        "Our AutoResearch loop (albench/arch_search, the one that produced menu v1), not the AutoResearch v2 loop",
        fontsize=11.5,
        weight="bold",
        color=DARK,
        ha="left",
        va="bottom",
    )
    save(fig, out, "prompt_loop.png")


def fig_recipes(inp, out):
    """Why the 20 menu configurations share one training recipe (data from analyze_recipe_choices.py)."""
    R = J(inp, "data_recipes.json")
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.9), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    ax = axs[0]
    settings = [
        ("optimizer", "muon", "Muon"),
        ("lr", "0.002", "lr 2e-3"),
        ("weight_decay", "0.0001", "wd 1e-4"),
        ("batch_size", "256", "batch 256"),
        ("lr_schedule", "onecycle", "one-cycle"),
    ]
    y = np.arange(len(settings))
    allv = [R["share_all"][k].get(v, 0) for k, v, _ in settings]
    topv = [R["share_top3"][k].get(v, 0) for k, v, _ in settings]
    ax.barh(y - 0.2, allv, 0.38, color=GREY, label=f"all {R['n_candidates']:,} proposed candidates")
    ax.barh(
        y + 0.2,
        topv,
        0.38,
        color=ORANGE,
        label=f"the 3 best candidates of each of {R['n_runs']} runs",
    )
    for yy, a_, t_ in zip(y, allv, topv):
        ax.text(a_ + 0.01, yy - 0.2, f"{a_:.0%}", va="center", fontsize=9)
        ax.text(t_ + 0.01, yy + 0.2, f"{t_:.0%}", va="center", fontsize=9)
    ax.set_yticks(y, [x[2] for x in settings])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("share of candidates using the menu's setting")
    ax.set_title("The proposer varies the recipe; selection keeps one", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.45, -0.14))
    ax = axs[1]
    d = np.array(sorted(x["diff"] for x in R["muon_minus_adamw_by_run"]))
    ax.scatter(np.arange(len(d)), d, s=16, color=PURPLE)
    ax.axhline(0, color=GREY, lw=1)
    ax.axhline(d.mean(), color=ORANGE, ls="--", lw=1.4)
    ax.text(
        0.03,
        0.95,
        f"mean {d.mean():+.3f}; Muon ahead in {int((d > 0).sum())} of {len(d)} runs",
        transform=ax.transAxes,
        fontsize=9.5,
        va="top",
    )
    ax.set_xlabel("search runs, sorted")
    ax.set_ylabel("Muon minus AdamW, mean validation r within a run")
    ax.set_title("Within every run, the same comparison", fontsize=11)
    ax = axs[2]
    names = ["0-2", "3-6", "7-11"]
    for key, col, lab in (
        ("muon", PURPLE, "Muon"),
        ("lr0.002", BLUE, "lr 2e-3"),
        ("wd1e-4", GREEN, "wd 1e-4"),
        ("onecycle", ORANGE, "one-cycle"),
    ):
        ax.plot(names, [R["share_by_round"][n][key] for n in names], "o-", color=col, label=lab)
    ax.set_xlabel("round of the search")
    ax.set_ylabel("share of that round's candidates")
    ax.set_title("The proposer drifts toward it as it sees results", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5)
    fig.suptitle(
        "Why the menu shares one recipe: it is the empirical winner among 1,712 proposed recipes, but the settings come as a package and are not separated\n"
        f"Muon candidates use weight decay 1e-4 in {R['wd_by_optimizer']['muon'].get('0.0001', 0) / sum(R['wd_by_optimizer']['muon'].values()):.0%} of cases and AdamW candidates use 0.01 or more in {sum(v for k, v in R['wd_by_optimizer']['adamw'].items() if float(k) >= 0.01) / sum(R['wd_by_optimizer']['adamw'].values()):.0%}, so optimizer and weight decay are confounded;\nthe prompt names Muon and lr 2e-3; the recipe was not re-tuned per dataset size except patience and epochs",
        fontsize=11,
        y=1.06,
    )
    save(fig, out, "justify_recipes.png")


def fig_cadence_budget(inp, out):
    """Proposals per round at an equal number of trained models (analyze_cadence_budget.py); the three reservoirs are the replicates."""
    D = J(inp, "data_cadence_budget.json")
    arms = [
        ("llm1x60", "1 per round (serial)", GREY),
        ("llm2x30", "2 per round (used)", ORANGE),
        ("llm6x10", "6 per round", BLUE),
        ("multi6", "6 classical samplers (no LLM)", PURPLE),
    ]
    res = ["genomic", "motif_ct_enriched", "random"]
    rcol = {"genomic": "#1b6ca8", "motif_ct_enriched": "#2a8f5b", "random": "#7a5195"}

    def curve(arm, r):
        c = D.get(f"{r}_{arm}")
        return {k: v for k, v in c["curve"]} if c else {}

    fig, axs = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axs[0]
    for arm, lab, col in arms:
        per = {r: curve(arm, r) for r in res}
        ks = sorted({k for c in per.values() for k in c})
        m, lo, hi, kk = [], [], [], []
        for k in ks:
            v = [c[k] for c in per.values() if k in c]
            if len(v) == len(res):
                kk.append(k)
                m.append(np.mean(v))
                se = np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
                lo.append(np.mean(v) - se)
                hi.append(np.mean(v) + se)
        if kk:
            ax.fill_between(kk, lo, hi, color=col, alpha=0.18, lw=0)
            ax.plot(kk, m, "o-", color=col, lw=2, ms=4, label=lab)
    ax.set_xlabel("models trained")
    ax.set_ylabel("deploy ensemble r, genomic test\n(mean over the 3 reservoirs; band ±1 SE)")
    ax.set_title(
        "Same number of models; points only where all three reservoirs have that many", fontsize=11
    )
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax = axs[1]
    base = {r: curve("llm2x30", r) for r in res}
    stats = {}
    for arm, lab, col in arms:
        if arm == "llm2x30":
            continue
        ks, m, se_, nn = [], [], [], []
        for k in range(6, 61, 6):
            d = [curve(arm, r)[k] - base[r][k] for r in res if k in curve(arm, r) and k in base[r]]
            if len(d) >= 2 or (k <= 12 and len(d) >= 1):
                ks.append(k)
                m.append(np.mean(d))
                se_.append(np.std(d, ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0)
                nn.append(len(d))
        if ks:
            m, se_ = np.array(m), np.array(se_)
            ax.fill_between(ks, m - se_, m + se_, color=col, alpha=0.18, lw=0)
            ax.plot(
                ks,
                m,
                "o-",
                color=col,
                lw=2,
                ms=4,
                label=f"{lab} (n = {min(nn)} to {max(nn)} reservoirs)",
            )
            for r in res:
                kk = [k for k in ks if k in curve(arm, r) and k in base[r]]
                ax.plot(
                    kk,
                    [curve(arm, r)[k] - base[r][k] for k in kk],
                    "-",
                    color=col,
                    lw=0.7,
                    alpha=0.4,
                )
    ax.axhline(0, color=DARK, lw=1.2)
    ax.set_xlabel("models trained")
    ax.set_ylabel(
        "difference from 2 per round, same reservoir\n(thick: mean ±1 SE; thin: each reservoir)"
    )
    ax.set_title("Paired by reservoir", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.suptitle(
        "Proposals per round at an equal number of trained models (Oct 13 cadence study, one run per arm and reservoir; harness older than the final pipeline)\n"
        "Reservoirs are the only replicates here; the serial arm finished only 9 to 17 models, so it appears at small budgets only",
        fontsize=11,
        y=1.04,
    )
    save(fig, out, "justify_proposals_per_round.png")


def fig_runsbudget(inp, out):
    """One long run or several shorter ones at the same total number of models (analyze_runs_budget.py)."""
    D = J(inp, "data_runs_budget.json")
    rs = sorted({v["r"] for v in D.values()})
    cm = plt.get_cmap("viridis")
    col = {r: cm(i / max(len(rs) - 1, 1) * 0.9) for i, r in enumerate(rs)}
    fig, axs = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for j, e in enumerate(SETS):
        for r in rs:
            cells = sorted((v for v in D.values() if v["r"] == r), key=lambda v: v["M"])
            xs = [v["M"] for v in cells]
            m = np.array([v[e]["mean"] for v in cells])
            sd = np.array([v[e]["sd"] for v in cells])
            lab = f"{r} run{'s' if r > 1 else ''}"
            axs[0, j].fill_between(xs, m - sd, m + sd, color=col[r], alpha=0.14, lw=0)
            axs[0, j].plot(xs, m, "o-", color=col[r], lw=1.8, ms=4, label=lab)
            axs[1, j].plot(xs, sd, "o-", color=col[r], lw=1.8, ms=4, label=lab)
        axs[0, j].set_title(SETN[e], fontsize=11)
        axs[0, j].set_ylabel("ensemble r\n(mean; band ±1 SD)" if j == 0 else "")
        axs[1, j].set_ylabel("SD of ensemble r\nover draws of runs" if j == 0 else "")
        axs[1, j].set_xlabel("total models in the ensemble")
        axs[1, j].set_xticks(sorted({v["M"] for v in D.values()}))
    axs[0, 0].legend(frameon=False, fontsize=9, title="split over", loc="lower right")
    g = {r: [v for v in D.values() if v["r"] == r and v["M"] == 60] for r in rs}
    fig.suptitle(
        "One long run or several shorter ones, at the same total number of models (the 8 guided 30-round runs on 30k genomic with at least 60 usable candidates; each run contributes its first M/r candidates; stacked like the menu)\n"
        "Draws are random subsets of the same 8 runs, so the SD understates run-to-run variability when many runs are used; bands show spread, not significance\n"
        + "At 60 models, genomic test (mean ± SD over draws): "
        + ", ".join(
            f"{r} run{'s' if r > 1 else ''} {D[f'60|{r}']['genomic']['mean']:.4f} ± {D[f'60|{r}']['genomic']['sd']:.4f}"
            for r in (1, 3, 5)
            if f"60|{r}" in D
        ),
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out, "justify_runs_vs_models.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--menu", default="albench/ensemble/menus/menu_v1.json")
    ap.add_argument(
        "--only",
        default="rounds,patience,cadence,evaluator,ar2,plateau,priors,loop,recipes,runsbudget",
    )
    a = ap.parse_args()
    only = a.only.split(",")
    if "rounds" in only:
        fig_rounds(a.inp, a.out)
    if "patience" in only:
        fig_patience(a.inp, a.out)
    if "cadence" in only:
        if os.path.exists(os.path.join(a.inp, "data_cadence_budget.json")):
            fig_cadence_budget(a.inp, a.out)
        else:
            fig_cadence(a.inp, a.out)
    if "runsbudget" in only and os.path.exists(os.path.join(a.inp, "data_runs_budget.json")):
        fig_runsbudget(a.inp, a.out)
    if "evaluator" in only:
        fig_evaluator(a.inp, a.out)
    if "ar2" in only:
        fig_ar2(a.inp, a.out)
    if "plateau" in only:
        fig_plateau(a.inp, a.out)
    if "priors" in only:
        fig_priors(a.inp, a.out, a.menu)
    if "loop" in only:
        fig_prompt_loop(a.inp, a.out)
    if "recipes" in only:
        fig_recipes(a.inp, a.out)


if __name__ == "__main__":
    main()
