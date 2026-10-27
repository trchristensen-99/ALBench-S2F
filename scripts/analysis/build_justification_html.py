# ruff: noqa
"""Build the evidence-ledger page from outputs/justification/{data.json,figs/*.png}.

To add a result: (1) add or extend a figure in make_justification.py, (2) add a card (or edit one) in CARDS below, (3) add a CHANGELOG line,
(4) rerun both scripts and republish. Numbers in card text come from data.json, so they refresh on every run; the *status chip* and the
interpretation are written by hand because they are judgements.
    python build_justification_html.py --src outputs/justification
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

STATUS = {  # key -> (label, meaning)
    "supported": ("Supported", "replicated or large effect; we would defend it to the PI"),
    "provisional": ("Provisional", "evidence points one way but n is small or confounded"),
    "running": ("Running", "experiment in progress; plot fills in as cells finish"),
    "open": ("Open", "known gap or contradiction; decision needed"),
    "planned": ("Planned", "designed or sketched, not yet run"),
    "carried": ("Carried over", "decided earlier; evidence lives in the earlier artifact"),
}
NARROW = {"arch_vs_legnet.png", "replicates.png", "kb_prompts.png"}
OLD_ARTIFACT = "https://claude.ai/artifact/98NSVuNvp3F41LHBdGgvPp"

CHANGELOG = [
    (
        "2026-10-27 20:00",
        "Added the Oct 27 results: the core reservoir set per dataset size, which loop elements earn their place (separate coder, coordinator against a random pick, reflector, guidance), consistency of the coordinator effect across sizes and reservoirs, the extrapolation under error-based metrics, training-recipe sensitivity (patience, epochs), the oracle-relabelled validation set adopted as standard and tested at 10k and 100k, HepG2 and the differential at 10k and 100k, reduced menus and ensemble methods across reservoirs, a 600k recipe retune, and a table of how far each assumption has been tested across sizes and reservoirs.",
    ),
    (
        "2026-10-26 20:00",
        "Added the Oct 25-26 results: assay noise, 600k and 1M anchors, acquisition (reduced selection ensemble, one round against three), HepG2 and differential students over nine pools, explore and coordinator prompt grids, nine seeds, the seed ladder and the Zoonomia rebuild. The page now opens with the procedure, parameter by parameter (prompt, rounds, runs, epochs, early stopping, top-k, reservoirs per D, seeds, ensembling and the rest) with a reservoir-by-D coverage matrix; the assumptions table gains a targets, acquisition and robustness group.",
    ),
    (
        "2026-10-23 15:00",
        "Added the results of the Oct 22 program: run-count curves, architecture source, unseen reservoirs (search against menu at 10k, 30k and, for three strategies, 100k) and the mixture sweep. Three runs per reservoir leave 0.005 to 0.012 of ensemble r on the table; the architecture source and corpus mixing change the genomic-test ensemble by less than the noise margin; a separate search on the strategy does not beat the retrained menu at equal model counts. Corrected the oracle accuracy line (0.931 for the current oracle) and the library size (4M sequences).",
    ),
    (
        "2026-10-22 18:00",
        "Restructured: the page now opens with the assumptions to justify (with the goal and the simplest recipe so far), then results by experiment, with live progress for the running experiments.",
    ),
    (
        "2026-10-22 17:00",
        "Corrected the reservoir-mixing claim: the union used the genomic-derived architectures and was not count-controlled; with equal counts mixing gains +0.005 over genomic alone and motif alone does as well or better. Launched the architecture-source, run-count-curve, unseen-reservoir and mixture experiments.",
    ),
    (
        "2026-10-22 14:30",
        "Independent pools: three independent draws beat three seeds on one pool in all 12 comparisons (menu, 30k). Per-reservoir search again matches the retrained menu at 100k. Models trained on several reservoirs ensemble better than genomic-only. Nested genomic ladder lands within 0.01 of the old ladder. Opus +0.005 over Sonnet 5.5, not significant. Added a setup-choices table at the top.",
    ),
    (
        "2026-10-21 21:20",
        "All 16 context cells, 15 family runs, 300k menu (3 seeds) and the baselines are done. Results: no benefit from cross-D context; a fresh search per reservoir matches the retrained menu at 30k (within 0.006); one instance saturates faster at 100k than at 30k; menu-ensemble scaling in the corrected loop reaches 0.863 at 300k (genomic). Only the Opus and Sonnet 4.5 proposer cells still run.",
    ),
    (
        "2026-10-21 09:40",
        "Experiment audit: ctx prompts carry the context (T/J/X yes, none no), family cells use the right pools, 300k models were not time-capped; 16 of 2,040 models had exploded predictions on some eval set. Dispatcher: LLM cap 14 to 22, default-queue overflow now goes to slow_nice, 300k seeds 2-3 queued.",
    ),
    (
        "2026-10-21 09:30",
        "Overnight results: guidance beats wide-open for the proposer (N is 0.02 behind at 50 models; P = W); prior-box baselines match the LLM in HP-only; context hand-off shows no gain yet; menu ranks agree across reservoirs near the retest ceiling except random; menu at 300k is 23/33 done.",
    ),
    (
        "2026-10-21 05:25",
        "Menu retraining at 300k queued on idle H100s (6 shards) to test whether 100k rankings carry to 300k.",
    ),
    (
        "2026-10-21 00:55",
        "Menu retrained on random / motif / evo2 / evoaug at 30k: rank agreement with genomic is 0.4-0.6, about the genomic retest ceiling (0.55). Second-seed retests for each family are queued.",
    ),
    (
        "2026-10-20 23:00",
        "First version. Ten evidence figures, sixteen decisions. Context-transfer (16 cells), reservoir-family AutoResearch (12), "
        "menu-by-reservoir retraining (12) and the repaired baselines launched or still running.",
    ),
]


def f3(x, n=3):
    return "n/a" if x is None or x != x else f"{x:.{n}f}"


def sgn(x, n=3):
    return "n/a" if x is None or x != x else f"{x:+.{n}f}"


def _rng(vals):
    return (min(vals), max(vals)) if vals else (float("nan"), float("nan"))


def oct22_cards(d: dict) -> list[dict]:
    """Result cards for the program launched Oct 22 (data from outputs/justification_oct22/data.json, merged in main())."""
    o = d.get("oct22", {})
    if not o:
        return []
    out = []
    # ---------------------------------------------------------------- run-count curves
    R = o.get("runs", {})
    if R:
        pts, short = [], []
        for lbl, r in R.items():
            n = r["n_runs"]
            m = lambda k, lbl=lbl: R[lbl]["curve"][str(k)]["mean"][0]  # noqa: E731
            short.append(m(n) - m(3))
            pts.append(
                f"{lbl} ({n} runs, top 3 models from each): one run {m(1):.3f}, three runs {m(3):.3f}, five runs {m(5):.3f}, all {n} runs {m(n):.3f}. "
                f"90% of the gain from one run to all of them is reached at {r['k90']} runs, and a further run adds under 0.001 beyond {r['k_marginal_lt_0p001']} runs."
            )
        lo, hi = _rng(short)
        pts.append(
            f"Three runs sit {lo:.3f} to {hi:.3f} below the ensemble of all runs (the noise margin is 0.005), so three pooled runs are on the steep part of the curve at every size."
        )
        pts.append(
            "The fast NNLS ensemble used for the many subsets reproduces the repo's ElasticNet ensemble to within 0.0003 on the full sets (for example genomic 30k, 15 runs: "
            + f"{R['genomic 30k']['curve']['15']['mean'][0]:.4f} against {R['genomic 30k']['enet_all_runs']['genomic']:.4f})."
        )
        out.append(
            dict(
                id="runcount",
                group="Results from the Oct 22 program",
                status="provisional",
                figs=["run_count_curves.png"],
                q="How many independent search runs does an ensemble need before it stops improving?",
                choice=f"Use about {min(r['k90'] for r in R.values())} to {max(r['k90'] for r in R.values())} independent runs per reservoir and size (top 3 models from each) where the budget allows; the 90% point depends on the reservoir and size. Three runs leave {lo:.3f} to {hi:.3f} of ensemble r on the table.",
                points=pts,
                caveat="Subsets are drawn from the 10 to 15 runs available (up to 60 random subsets per size of subset), so subsets overlap and the right end of each curve is a single ensemble, not independent samples. Genomic test set shown; the curves for the designed set and random sequences are in data.json.",
                change="If a comparison only needs the ranking of strategies at equal run counts, three runs per arm are still fair; the cost of fewer runs is a lower absolute level, about 0.01 at 30k.",
            )
        )
    # ---------------------------------------------------------------- architecture source
    A = o.get("archsrc", {}).get("results", {})
    AD = o.get("archsrc", {}).get("diffs", {})
    if A:
        pts, spreads = [], []
        names = {
            "MG": "genomic only",
            "MR": "random only",
            "MGR": "genomic+random",
            "MF": "5 reservoirs",
            "MX": "searched on mixed pools",
        }
        for t in ("genomic", "random", "motif", "grm"):
            row = [
                (c, A[f"{c}|{t}"]["enet"]["genomic"])
                for c in ("MG", "MR", "MGR", "MF", "MX")
                if f"{c}|{t}" in A
            ]
            if row:
                spreads.append(max(v for _, v in row) - min(v for _, v in row))
                pts.append(
                    f"Retrained on the {t} pool, ensemble r on genomic test: "
                    + ", ".join(f"{c} ({names[c]}) {v:.3f}" for c, v in row)
                    + f". Spread {spreads[-1]:.3f}."
                )
        dl = [
            f"{k.split('|')[0].replace('-MG', '')} against MG on the {k.split('|')[1]} pool {v['genomic_eval_enet']:+.4f} (bootstrap interval {v['genomic_eval_boot'][1]:+.4f} to {v['genomic_eval_boot'][2]:+.4f})"
            for k, v in AD.items()
        ]
        if dl:
            pts.append(
                "Differences from the genomic-only menu on genomic test: " + "; ".join(dl) + "."
            )
        ood = [abs(v["ood_enet"]) for v in AD.values()]
        if ood:
            pts.append(
                f"On the designed (out-of-distribution) set the sources differ by up to {max(ood):.3f} in either direction. That set is much noisier (one pool, one seed per model), and the differences have no consistent sign."
            )
        pts.append(
            "Each source contributes the same number of candidates: the top 3 of each of 15 independent runs (45 models), all retrained once per target pool with the same recipes and cap."
        )
        ok = bool(spreads) and max(spreads) <= 0.006
        out.append(
            dict(
                id="archsource",
                group="Results from the Oct 22 program",
                status="supported" if ok else "provisional",
                figs=["arch_source.png"],
                q="Are architectures searched on genomic data biased toward genomic data?",
                choice=(
                    "Searching on genomic data alone is enough: the source of the architectures changes the ensemble by no more than the noise margin on the genomic test set."
                    if ok
                    else "Not settled, but small: searches on mixed corpora (MX) or on several reservoirs (MF) are nominally ahead of genomic-only by up to 0.005 on some pools, with bootstrap intervals that mostly include zero; random-only searches are never ahead."
                ),
                points=pts,
                caveat="One pool and one seed per model for each target, so differences under about 0.005 are noise. The first genomic and random target pools are also the pools of run 1 of the genomic-only and random-only searches (3 of 45 candidates), a small in-sample advantage for those two sources.",
                change="A random-only or mixed source that beat genomic-only by more than 0.005 on both the genomic and designed sets, with a bootstrap interval excluding zero, would argue for searching on more than genomic data.",
            )
        )
    # ---------------------------------------------------------------- unseen reservoirs
    U = o.get("unseen", {}).get("results", {})
    DOD = o.get("unseen", {}).get("dod", {})
    if U:
        pts, worse = [], []
        for D in (10000, 30000, 100000):
            rows = [(k.split("|")[0], v) for k, v in U.items() if k.endswith(f"|{D}")]
            if not rows:
                continue
            eq = {
                st: v["menu_equal_count"]["genomic"] - v["bespoke_all_nnls"]["genomic"]
                for st, v in rows
            }
            fu = {st: v["menu"]["genomic"] - v["bespoke_all"]["genomic"] for st, v in rows}
            lo, hi = _rng(list(eq.values()))
            flo, fhi = _rng(list(fu.values()))
            nb = rows[0][1]["n_bespoke_models"]
            nm = sorted({v["n_menu_models"] for _, v in rows})
            worse += [
                f"{st.replace('_', ' ')} at {D // 1000}k" for st, x in eq.items() if -x > 0.005
            ]
            pts.append(
                f"{D // 1000}k, menu cut to the same {nb} models as the three searches: menu minus own search {lo:+.3f} to {hi:+.3f} ("
                + ", ".join(f"{st.replace('_', ' ')} {x:+.3f}" for st, x in eq.items())
                + f"); with the whole menu ({nm[0]} to {nm[-1]} models) {flo:+.3f} to {fhi:+.3f}."
            )
        for D in (10000, 30000, 100000):
            vs = [(k.split("|")[0], v) for k, v in DOD.items() if k.endswith(f"|{D}|equal_count")]
            if vs:
                pts.append(
                    f"Difference of differences at {D // 1000}k (gap genomic minus X with each searched separately, minus the same gap with the menu, equal model counts; genomic test, bootstrap interval in brackets): "
                    + ", ".join(
                        f"{st.replace('_', ' ')} {v['point']['genomic']:+.3f} [{v['boot_genomic_eval'][1]:+.3f}, {v['boot_genomic_eval'][2]:+.3f}]"
                        for st, v in vs
                    )
                    + "."
                )
        wid = {
            D: [
                (v["boot_genomic_eval"][2] - v["boot_genomic_eval"][1]) / 2
                for k, v in DOD.items()
                if k.endswith(f"|{D}|equal_count") and "boot_genomic_eval" in v
            ]
            for D in (10000, 30000, 100000)
        }
        wid = {D: w for D, w in wid.items() if w}
        if wid:
            pts.append(
                "With three runs and three pools per arm the bootstrap intervals on these differences are about plus or minus "
                + ", ".join(f"{sum(w) / len(w):.3f} at {D // 1000}k" for D, w in wid.items())
                + "; a bias toward genomic of up to about that size cannot be excluded, and the point estimates have no consistent sign."
                + (
                    (
                        " At 100k all the intervals lie within +-"
                        + f"{max(max(abs(v['boot_genomic_eval'][1]), abs(v['boot_genomic_eval'][2])) for k, v in DOD.items() if k.endswith('|100000|equal_count')):.3f}"
                        + "."
                    )
                    if 100000 in wid
                    else ""
                )
            )
        pts.append(
            "Reading: a negative difference of differences means the genomic-derived menu favours genomic more than separate searches do (the signature of a menu biased toward the strategy it was built from); a positive one means the opposite."
        )
        out.append(
            dict(
                id="unseen",
                group="Results from the Oct 22 program",
                status="provisional",
                figs=["unseen_search_vs_menu.png"],
                q="Is a strategy the search was not run on handicapped when the menu is retrained on it?",
                choice=(
                    "No evidence of a handicap: a separate three-run search on the strategy itself did not beat the genomic-derived menu on any strategy at any size tested; at equal model counts the menu is ahead by "
                    + f"{min(x['menu_equal_count']['genomic'] - x['bespoke_all_nnls']['genomic'] for x in U.values()):+.3f} to {max(x['menu_equal_count']['genomic'] - x['bespoke_all_nnls']['genomic'] for x in U.values()):+.3f}."
                    if not worse
                    else "Mixed: own search beat the menu by more than the noise margin, at equal model counts, for "
                    + ", ".join(worse)
                    + "."
                ),
                points=pts,
                caveat="Three runs and three pools per strategy and size, one seed per menu model. Genomic test set shown; designed and random sets are in data.json. The 100k comparison covers three strategies plus the genomic reference.",
                change="A strategy where own search beat the menu by more than the 0.005 noise margin at equal model counts would show the menu is missing something for that strategy.",
            )
        )
    # ---------------------------------------------------------------- mixtures
    MX = o.get("mix", {})
    if MX:
        rows, ctl = MX["rows"], MX["controls"]
        c3 = MX.get("controls_3pools", {})
        names = {"G": "genomic", "R": "random", "M": "motif", "E": "evoaug"}
        top = sorted(rows, key=lambda n: -rows[n]["enet"]["genomic"])[:5]
        pts = [
            "Single-reservoir menus at 30k (genomic test / designed / random): "
            + "; ".join(
                f"{names[k]} {v['genomic']:.3f} / {v['ood']:.3f} / {v['random_32k']:.3f}"
                for k, v in ctl.items()
            )
            + ".",
            "Best mixtures on genomic test: "
            + ", ".join(f"{n} {rows[n]['enet']['genomic']:.3f}" for n in top)
            + ". The best mixture on each set against the best single reservoir: "
            + "; ".join(
                f"{lbl} {max(r['enet'][e] for r in rows.values()):.3f} against {max(v[e] for v in ctl.values()):.3f}"
                for lbl, e in (
                    ("genomic test", "genomic"),
                    ("designed", "ood"),
                    ("random", "random_32k"),
                )
            )
            + ".",
            f"Ensemble across the best three mixtures chosen on validation ({', '.join(MX['best3_by_val'])}; {MX.get('n_models_best3', '?')} models): {MX['ensemble_best3']['genomic']:.3f} genomic test, {MX['ensemble_best3']['ood']:.3f} designed, {MX['ensemble_best3']['random_32k']:.3f} random. "
            f"Across all {len(rows)} mixtures ({MX.get('n_models_all', '?')} models): {MX['ensemble_all_mixtures']['genomic']:.3f}, {MX['ensemble_all_mixtures']['ood']:.3f}, {MX['ensemble_all_mixtures']['random_32k']:.3f}.",
        ]
        if c3:
            pts.append(
                "Model-count match: three independent pools of one reservoir pooled (genomic test / designed / random) give "
                + "; ".join(
                    f"{names[k]} {v['enet']['genomic']:.3f} / {v['enet']['ood']:.3f} / {v['enet']['random_32k']:.3f} with {v['n_models']} models"
                    for k, v in c3.items()
                )
                + ". An ensemble over several mixtures is a bigger ensemble, so compare it with these and not with a single 33-model menu."
            )
        comp = {n: [k for k in r["spec"] if k in ctl] for n, r in rows.items()}
        full = [n for n, r in rows.items() if len(comp[n]) == len(r["spec"])]
        if full:
            beat = {
                e: sum(1 for n in full if rows[n]["enet"][e] > max(ctl[k][e] for k in comp[n]))
                for e in ("genomic", "ood", "random_32k")
            }
            gains = [
                rows[n]["gain_vs_single_mean"]["genomic"]
                for n in full
                if "gain_vs_single_mean" in rows[n]
            ]
            pts.append(
                f"Of the {len(full)} mixtures made only of genomic, random, motif and evoaug components, {beat['genomic']} beat their best single component on genomic test, {beat['ood']} on designed and {beat['random_32k']} on random. "
                f"On genomic test they gain {sum(gains) / len(gains):+.3f} on average ({min(gains):+.3f} to {max(gains):+.3f}) over the weight-averaged single-reservoir results."
            )
        pts.append(
            "Each mixture is one 30k corpus (menu of 33 architectures, one seed) built from the first replicate pools of its components, so it is compared with the single-reservoir menus on the same pools."
        )
        best_beats = any(
            max(r["enet"][e] for r in rows.values()) > max(v[e] for v in ctl.values()) + 0.005
            for e in ("genomic", "ood", "random_32k")
        )
        out.append(
            dict(
                id="mixsweep",
                group="Results from the Oct 22 program",
                status="provisional",
                figs=["mixture_sweep.png"],
                q="Does training on a mixed corpus beat the best single reservoir?",
                choice=(
                    "Not on these data: no single mixture beat the best single reservoir on any evaluation set by more than the noise margin; mixtures mostly land between their components."
                    if not best_beats
                    else "Some mixtures beat the best single reservoir on at least one evaluation set by more than the noise margin; see the figure."
                ),
                points=pts,
                caveat="30k only, one seed per mixture, 21 mixtures chosen by hand (pairs, trios, uneven splits, within-family mixes). The 21 are not independent of each other, and the best of 21 on any one set is optimistically biased.",
                change="A mixture that beat the best single reservoir on both the genomic and designed sets, repeated on a second pool, would argue for mixing.",
            )
        )
    return out


def oct23_cards(d: dict) -> list[dict]:
    o = d.get("oct22", {})
    out = []
    G = "Results from the Oct 22 program"
    # ------------------------------------------------------------ top-k per run and runs from several reservoirs
    R2 = o.get("runs2")
    if R2:
        tk, mx, five = R2["topk"], R2["mixed"], R2.get("five", {})
        pts = []
        for src in ("genomic", "random", "motif"):
            c = tk[src]
            n = len(c["3"])
            pts.append(
                f"{src} 30k, ensemble r on genomic test at 1 / 5 / {n} runs: top 1 per run "
                + " / ".join(f"{c['1'][i][0]:.3f}" for i in (0, 4, n - 1))
                + "; top 3 "
                + " / ".join(f"{c['3'][i][0]:.3f}" for i in (0, 4, n - 1))
                + "; all 20 "
                + " / ".join(f"{c['20'][i][0]:.3f}" for i in (0, 4, n - 1))
                + "."
            )
        n15 = {src: tk[src]["3"][14] for src in tk}
        pts.append(
            "Runs drawn evenly from genomic, random and motif reservoirs (top 3 each), at 3 / 9 / 15 / 45 total runs: genomic test "
            + " / ".join(f"{mx[t][0]:.3f}" for t in ("3", "9", "15", "45"))
            + "; designed set "
            + " / ".join(f"{mx[t][1]:.3f}" for t in ("3", "9", "15", "45"))
            + "; random sequences "
            + " / ".join(f"{mx[t][2]:.3f}" for t in ("3", "9", "15", "45"))
            + "."
        )
        pts.append(
            "The same 15 runs from a single reservoir (genomic test / designed / random): "
            + "; ".join(
                f"{s_} {n15[s_][0]:.3f} / {n15[s_][1]:.3f} / {n15[s_][2]:.3f}" for s_ in n15
            )
            + f". So the mixed ensemble at 15 runs is {mx['15'][0] - n15['genomic'][0]:+.3f} above genomic-only on genomic test and {mx['15'][1] - n15['genomic'][1]:+.3f} on the designed set, and {mx['15'][0] - n15['motif'][0]:+.3f} against motif-only (genomic test), {mx['15'][1] - n15['random'][1]:+.3f} against random-only (designed), {mx['15'][2] - n15['random'][2]:+.3f} against random-only (random sequences)."
        )
        if five:
            pts.append(
                "The planned structure, 1 / 2 / 3 runs from each of five reservoirs (genomic, random, motif, evo2, evoaug; 5 / 10 / 15 runs): genomic test "
                + " / ".join(f"{five[t][0]:.3f}" for t in ("5", "10", "15"))
                + "; designed "
                + " / ".join(f"{five[t][1]:.3f}" for t in ("5", "10", "15"))
                + "; random "
                + " / ".join(f"{five[t][2]:.3f}" for t in ("5", "10", "15"))
                + "."
            )
        out.append(
            dict(
                id="runcount2",
                group=G,
                status="provisional",
                figs=["runcount_topk.png", "runcount_mixed_sources.png"],
                q="How many models should we keep from each run, and does it matter which reservoirs the runs come from?",
                choice="Keep the top 3 models of each run: 5, 8 or all 20 add nothing once a few runs are pooled, and keeping only the best costs about 0.015 to 0.019 with one run and 0.001 to 0.003 with five. Runs drawn from several reservoirs give a hedged ensemble: above genomic-only runs on the genomic and designed sets, below the best single reservoir on each set.",
                points=pts,
                caveat="Models from different reservoirs were trained on different pools, so mixing here is at the level of models, not of training corpora. Subsets of the 15 runs per reservoir overlap, so the curves are smooth but not independent; the 5-reservoir structure has only three runs per reservoir.",
                change="If the final ensemble is scored on one target (say genomic held-out sequences) the best single reservoir for that target still beats the mixture; the mixture is the choice when several targets matter.",
            )
        )
    # ------------------------------------------------------------ why random wins the designed set
    W = o.get("oodwhy")
    U = o.get("unseen", {}).get("results", {})
    if W:
        dz, st = W["designed"], W.get("student", {})
        pts = [
            f"Composition: the designed sequences have GC {dz['gc'][0]:.3f} and CpG {dz['cpg'][0]:.3f} per dinucleotide. Random pool: GC {W['random']['gc'][0]:.3f}, CpG {W['random']['cpg'][0]:.3f}; GC-matched random: {W['gc_matched']['gc'][0]:.3f}, {W['gc_matched']['cpg'][0]:.3f}; motif pool: {W['motif']['gc'][0]:.3f}, {W['motif']['cpg'][0]:.3f}; genomic: {W['genomic']['gc'][0]:.3f}, {W['genomic']['cpg'][0]:.3f} (genomic sequence is CpG-depleted).",
            "Coverage: the fraction of designed sequences whose (GC, CpG) cell holds at least 10 pool sequences is "
            + ", ".join(
                f"{k} {W[k]['designed_gc_cpg_support']:.2f}"
                for k in ("random", "gc_matched", "motif", "genomic", "evoaug")
            )
            + ".",
            f"Activity tails do not explain it: the fraction of pool labels above 3 is "
            + ", ".join(
                f"{k} {W[k]['frac_label_above']['3.0']:.3f}" for k in ("genomic", "random", "motif")
            )
            + f", against a designed-set mean of {dz['label_mean']:.1f} (5th to 95th percentile {dz['label_q'][0]:.1f} to {dz['label_q'][4]:.1f}). Random has no more high-activity sequences than genomic; the designed set is far beyond every pool.",
            "Students (three independent pools pooled), r on all designed sequences / top quartile only: "
            + "; ".join(
                f"trained on {k} {v['r']:.3f} / {v['r_top_quartile']:.3f}" for k, v in st.items()
            )
            + ". Every student underpredicts the designed set (mean prediction "
            + ", ".join(f"{v['mean_pred']:.1f}" for v in st.values())
            + f" against {dz['label_mean']:.1f}), so the difference is in ranking among designed sequences, not in reaching the right level.",
        ]
        if U:
            g = lambda k: U.get(f"{k}|30000", {}).get("menu", {}).get("ood")
            if all(
                g(k) is not None
                for k in ("genomic", "gc_matched", "random", "dinuc_shuffle", "motif_shared_core")
            ):
                pts.append(
                    f"Controls at 30k (designed-set r of the menu ensemble): genomic {g('genomic'):.3f}; GC-matched random, which carries the CpG-rich composition but no genomic structure, {g('gc_matched'):.3f}; dinucleotide shuffles of genomic sequence, which keep genomic CpG depletion, {g('dinuc_shuffle'):.3f}; uniform random {g('random'):.3f}; planted-motif pool (shared core) {g('motif_shared_core'):.3f}. "
                    f"Matching composition alone recovers {g('gc_matched') - g('genomic'):+.3f} of random's {g('random') - g('genomic'):+.3f}; pools with planted transcription-factor motifs reach as high as random."
                )
        out.append(
            dict(
                id="oodwhy",
                group=G,
                status="provisional",
                figs=["ood_why_random.png"],
                q="Why does a random-trained student do so well on the designed (high-activity) set?",
                choice="The designed sequences look compositionally like random DNA (GC about 0.5, CpG-rich), and pools with planted transcription-factor motifs do as well as random on them, so students trained on random or motif-planted data rank them better; it is not because random data reaches higher activities.",
                points=pts,
                caveat="Correlational: composition, motif content and coverage move together across pools, and only the GC-matched and dinucleotide-shuffle controls separate them partly. The designed set was itself optimised against a model, and the oracle's response to GC and CpG is stronger than the experimental response (Inoue and Sahu comparisons), so part of random's advantage may be distilling the oracle's own composition effects. Composition statistics use the 22,862-sequence version of the designed set.",
                change="A CpG-enriched, dinucleotide-preserving genomic control (mono-nucleotide shuffles) that reached random's designed-set r would put the whole effect on composition; measured designed sequences with real MPRA labels would separate oracle artefact from biology.",
            )
        )
    # ------------------------------------------------------------ extrapolation
    X = o.get("extrap")
    if X:
        pts = []
        for res_, row in X.items():
            e, od = row["genomic"], row["ood"]
            p1, p4 = e["pred"]["1000000"], e["pred"]["4000000"]
            q1, q4 = od["pred"]["1000000"], od["pred"]["4000000"]
            pts.append(
                f"{res_.replace('_', ' ')} (fit to {e['sizes'][0] // 1000}k to {e['sizes'][-1] // 1000}k, exponent {e['exponent']:.2f}): genomic test {e['r'][-1]:.3f} at {e['sizes'][-1] // 1000}k, {p1[0]:.3f} [{p1[1]:.3f}, {p1[2]:.3f}] at 1M, {p4[0]:.3f} [{p4[1]:.3f}, {p4[2]:.3f}] at 4M; designed set {q1[0]:.3f} at 1M, {q4[0]:.3f} [{q4[1]:.3f}, {q4[2]:.3f}] at 4M."
            )
        g = X["genomic"]["genomic"]
        e3 = g["r"]
        loc = np_ln = None
        import math

        b_loc = math.log((1 - e3[-2]) / (1 - e3[-1])) / math.log(g["sizes"][-1] / g["sizes"][-2])
        pts.append(
            f"The exponent is not constant: for genomic, the local exponent between {g['sizes'][-2] // 1000}k and {g['sizes'][-1] // 1000}k is {b_loc:.2f} against {g['exponent']:.2f} for the fit to all sizes, so the straight-line extrapolation may be a little optimistic. Adding an error floor of 0.01 or 0.03 lowers the 4M prediction for genomic test by "
            + " and ".join(
                f"{g['pred']['4000000'][0] - v[1]:.3f}" for v in g["floor_sensitivity"].values()
            )
            + "."
        )
        pts.append(
            "Genomic sequence cannot reach 1M (about 315k exist), so its curve is a hypothetical for sequences like it; random and the motif reservoirs can be scaled."
        )
        out.append(
            dict(
                id="extrap",
                group=G,
                status="open",
                figs=["extrapolation.png"],
                q="What do the scaling fits predict at 1M and 4M training sequences?",
                choice="Treat these as a range, not a forecast: on the genomic test set most reservoirs sit between 0.86 and 0.91 at 1M and between 0.89 and 0.94 at 4M; the designed set stays far lower (0.54 to 0.74 at 1M).",
                points=pts,
                caveat="Straight-line fits in log-log space to three or four sizes. The intervals only carry measurement noise of 0.004 per point, not uncertainty about the functional form, which is larger; fits to 10k to 100k only (motif, evo2, evoaug) extrapolate over a factor of 40 to 400.",
                change="One measured 1M point for the finalists would cut the extrapolation uncertainty more than any further point below 300k.",
            )
        )
    # ------------------------------------------------------------ reference LegNet
    L = o.get("legref_summary")
    if L:
        pts = []
        for k, v in L.items():
            pts.append(
                f"{k}k ({v['n_pools']} pools, {v['mean_n']:.0f} entries): the tuned LegNet's mean rank is {v['mean_rank']:.1f}; it is {v['ref_minus_median']:+.3f} against the median entry, the mean of the top 5 entries is {v['top5_minus_ref']:+.3f} above it and the best entry {v['best_minus_ref']:+.3f} above it; it lands in the top quarter on {v['frac_ref_top_quartile']:.0%} of pools."
            )
        pts.append(
            "For scale: the best of 32 equally good models, each with a retraining SD of about 0.009, is about +0.018 above the typical model, which is the size of the best-minus-reference gap at 30k, so the best entry's lead is what picking a maximum from noisy entries gives."
        )
        out.append(
            dict(
                id="legref",
                group=G,
                status="supported",
                figs=["legnet_reference.png"],
                q="Is a free-form searched architecture better than a well-tuned LegNet?",
                choice="No, not on average: from 30k up the tuned LegNet sits at the median of the searched entries; at 10k it is above the median. The searched models matter as a diverse ensemble, not as better individual models.",
                points=pts,
                caveat="One retraining per entry on most pools, so single-model ranks are noisy (retest ceilings 0.5 to 0.7); the reference is one configuration of LegNet, tuned earlier on genomic data. Genomic test set; the same ranks on the designed and random sets are in data.json.",
                change="A tuned LegNet that was beaten by the median searched entry on pools from other reservoirs would change this.",
            )
        )
    # ------------------------------------------------------------ new experiments
    out.append(
        dict(
            id="program2",
            group="Experiments launched Oct 22",
            status="running",
            figs=[],
            q="What was launched on Oct 23 after the first review?",
            choice="Searches on mixed corpora, prompt styles, a precision check, and three workstreams that are still being built.",
            points=[
                "Architecture and recipe search ON mixed pools: 15 runs on corpora of 10k genomic + 10k random + 10k motif sequences (independent pool per run), then the top 3 per run retrained on genomic, random, motif and mixed pools against the genomic-only, random-only, mixed-run and 5-reservoir sources.",
                "Prompt styles in the free-architecture loop, 9 runs each on genomic 30k pools r1 to r9, paired with the 9 default-prompt runs on the same pools: explore, reflect (the proposer writes a short reflection that is fed back) and explore+reflect.",
                "Precision check: the 33-architecture menu retrained on genomic r1 30k under bf16, fp16 and tf32 against the fp32 baseline. Precision is set by the harness for every model alike, so no proposal can choose its own speed-up.",
                "Multi-agent search loop (proposer, coder, evaluator, reflector): built and smoke-tested; a 5-run pilot on genomic 30k pools r1 to r5, paired with the default-loop runs on the same pools, is queued. It costs about four times the LLM tokens per run.",
                "Validation-set variants: four oracle-labelled validation sets are built (the current genomic set relabelled with the current oracle; perturbed genomic; a 14-family diverse set; a union), similarity-filtered against all evaluation sets and training pools. Phase 1 of the retraining test (the 33-architecture menu on genomic, random and motif 30k pools under each validation set, 45 jobs, about 36 H100-hours) is queued.",
                "HepG2 oracle for the differential objective: an earlier HepG2 oracle without the designed sequences exists (held-out Pearson 0.916); a full-pool oracle with the K562 protocol passed its dry run (stage 1 held-out Pearson 0.895 on fold 0) and is training; a labelling script for K562, HepG2 and the differential is ready.",
            ],
            caveat="Search runs share the account's 5-hour usage limit and pause and resume on their own after each reset.",
            change="A prompt style or a mixed-pool search that beats the default by more than the 0.005 noise margin on pooled runs would change the search recipe.",
        )
    )
    out.append(
        dict(
            id="integrity",
            group=G,
            status="open",
            figs=[],
            q="What data-integrity problems did the validation-set workstream find?",
            choice="Two, both fixable: the stored validation labels are not from the current oracle, and the Zoonomia sampler let validation chromosomes into its training pools.",
            points=[
                "Validation labels: the stored validation set (genomic chr19, chr21, chrX) has no oracle stamp, and against a fresh labelling with the current oracle its labels correlate at 0.983, with a mean offset of +0.07 that grows with GC (+0.055 to +0.112). Every experiment so far validated on slightly different labels from its training pools and evaluation sets. The effect on early stopping and ensemble weights is probably small and is measured by one arm of the validation-set experiment.",
                "Zoonomia: the sampler excluded only the test chromosomes (chr7, chr13), so about 14% of its windows (9.5% of base pairs) came from chr19, chr21 and chrX, the validation chromosomes. Test sets were never affected. Zoonomia pools used so far had validation sequences or their orthologs in training; their validation-based choices are slightly favourable, and the pools are rebuilt with the corrected default (a regression test now pins it).",
                "Smaller: two files in the newer evaluation-set folder share sequence with the validation chromosomes; they are not part of the benchmark and were excluded from the similarity index. Mutagenesis and EvoAug variants that take Zoonomia as a base use a rates file with the same narrow exclusion.",
            ],
            caveat="Both were found by the validation-set workstream while building similarity checks; neither changes the test-set numbers, and the unseen-reservoir comparison for Zoonomia (the weakest strategy) is the result most exposed.",
            change="If the relabelled-validation arm changes pool contrasts by more than the 0.005 noise margin, earlier comparisons need re-running on the corrected validation labels.",
        )
    )
    return out


def oct25_cards(d: dict) -> list[dict]:
    o = d.get("oct22", {})
    out = []
    G = "Results from the Oct 22 program"
    f = lambda x, n=3: f"{x:.{n}f}"  # noqa: E731
    # ------------------------------------------------------------ prompt styles
    P = o.get("prompts")
    if P:
        arms, df = P["arms"], P["diffs"]
        pts = [
            "Mean single-run ensemble r on genomic test over nine paired pools (genomic / designed / random): "
            + "; ".join(
                f"{k} {f(v['mean'][0])} / {f(v['mean'][1])} / {f(v['mean'][2])}"
                for k, v in arms.items()
            )
            + ".",
            "Paired difference from the default prompt, genomic test (mean, bootstrap interval, runs ahead of nine): "
            + "; ".join(
                f"{k} {df[k]['genomic']['mean']:+.4f} [{df[k]['genomic']['lo']:+.4f}, {df[k]['genomic']['hi']:+.4f}], {df[k]['genomic']['wins']} of {df[k]['genomic']['n']}"
                for k in df
            )
            + ". Uncorrected paired t-test p: "
            + ", ".join(f"{k} {df[k]['genomic']['p']:.3f}" for k in df)
            + ".",
            "Designed set / random sequences, mean difference: "
            + "; ".join(
                f"{k} {df[k]['ood']['mean']:+.4f} / {df[k]['random_32k']['mean']:+.4f}" for k in df
            )
            + ". Pooled triples of runs (genomic test, three triples): "
            + "; ".join(f"{k} {df[k]['pooled3_genomic']['mean']:+.4f}" for k in df)
            + ".",
            "Individual models are better with the new prompts: mean validation r of a run's 20 models is "
            + ", ".join(f"{k} {arms[k]['mean_val']:.4f}" for k in arms)
            + "; prediction diversity within a run (one minus mean pairwise correlation) "
            + ", ".join(f"{k} {arms[k]['diversity']:.3f}" for k in arms)
            + ".",
            "Earlier two-run comparisons suggested explore +0.021 and reflect +0.017 over the default; with nine paired runs the effects are about a quarter of that.",
        ]
        best = max(df, key=lambda k: df[k]["genomic"]["mean"])
        out.append(
            dict(
                id="prompts",
                group=G,
                status="provisional",
                figs=["prompt_styles.png"],
                q="Do the explore and reflect prompt styles improve the search?",
                choice=f"A small improvement at best: {best} gains {df[best]['genomic']['mean']:+.3f} in single-run ensemble r on genomic 30k over the default, about half the 0.01 that would change the recipe; the effect is clearer in individual models than in the ensemble.",
                points=pts,
                caveat="Nine paired runs on genomic 30k pools, three comparisons with no multiplicity correction (with a Bonferroni correction none is below 0.05). The explore arm is the only one whose interval excludes zero. The generality of the effect across reservoirs and sizes is being tested (explore against default on five reservoirs at 10k and 30k).",
                change="Gains above 0.01 that repeat across reservoirs and sizes would justify changing the default prompt.",
            )
        )
    # ------------------------------------------------------------ multi-agent pilot
    A = o.get("agent")
    if A:
        dg, ood, rnd = A["diff"]["genomic"], A["diff"]["ood"], A["diff"]["random_32k"]
        cs = A.get("cost_summaries") or []
        cpm = sum(c.get("calls_per_model", 0) for c in cs) / len(cs) if cs else None
        tpm = sum(c.get("tokens_incl_overhead_per_model", 0) for c in cs) / len(cs) if cs else None
        pts = [
            f"{A['n_pairs']} paired runs on genomic 30k pools. Single-run ensemble r (genomic test): default {f(sum(x[0] for x in A['default']) / A['n_pairs'])}, multi-agent {f(sum(x[0] for x in A['agent']) / A['n_pairs'])}; difference {dg['mean']:+.4f} [{dg['lo']:+.4f}, {dg['hi']:+.4f}], ahead in {dg['wins']} of {dg['n']} pairs. Designed set {ood['mean']:+.4f}, random sequences {rnd['mean']:+.4f}.",
            f"Pooled over the five runs: default {f(A['pooled5']['default'][0], 4)}, multi-agent {f(A['pooled5']['agent'][0], 4)} (genomic test).",
            f"Individual models: mean validation r {A['mean_val']['default']:.4f} for default against {A['mean_val']['agent']:.4f} for multi-agent ({A['mean_val']['diff']['mean']:+.4f}); best model {A['max_val']['default']:.4f} against {A['max_val']['agent']:.4f}; prediction diversity {A['diversity']['default']:.3f} against {A['diversity']['agent']:.3f}.",
        ]
        if cpm:
            pts.append(
                f"Cost: about {cpm:.1f} LLM calls per trained model, about {tpm / 1000:.0f}k tokens per model including each call's fixed overhead, roughly four times the single-call loop."
            )
        out.append(
            dict(
                id="agentpilot",
                group=G,
                status="provisional",
                figs=["agent_pilot.png"],
                q="Does the multi-agent loop (proposer, coder, evaluator, reflector) beat the single-call loop?",
                choice=f"Not in the pilot: the ensemble difference is {dg['mean']:+.4f} on {A['n_pairs']} paired pools at about four times the LLM cost; individual models are slightly better but the ensemble is not.",
                points=pts,
                caveat="Five pairs resolve a difference of about 0.01 and no less. The agent arm also differs from the default in more than roles (reworded setup note, diversity figures in the evaluator report, a noise note for the reflector). Variants with a coordinator role, without the reflector and with an explore proposer are running with six paired runs each.",
                change="A variant whose pooled-run ensemble beats the default by more than 0.005 across pools would justify its token cost.",
            )
        )
    # ------------------------------------------------------------ precision
    Q = o.get("precision")
    if Q:
        pts = [
            f"The 33-architecture menu on genomic r1 30k, paired by architecture against fp32 (ensemble {Q['fp32']['ensemble']['genomic']:.4f}): "
            + "; ".join(
                f"{m} {Q[m]['mean_dtest']:+.4f} per model (SE {Q[m]['se']:.4f}), ensemble {Q[m]['ensemble']['genomic']:.4f}"
                for m in ("tf32", "bf16", "fp16")
            )
            + ".",
            "By optimizer, mean per-model test r change (muon, 31 of 33 architectures / adamw, 2): "
            + "; ".join(
                f"{m} {Q[m]['by_optimizer'].get('muon', {}).get('mean', float('nan')):+.4f} / {Q[m]['by_optimizer'].get('adamw', {}).get('mean', float('nan')):+.4f}"
                for m in ("tf32", "bf16", "fp16")
            )
            + ". fp16 diverged (non-finite loss at epoch 24) on one of 33 architectures, which no other mode did.",
        ]
        if "speed" in Q:
            pts.append(
                "Controlled speed on one H100, nine architectures, same data and seed, four epochs (fp32 time divided by mode time; median, range): "
                + "; ".join(
                    f"{m} {Q['speed'][m]['median']:.2f}x ({Q['speed'][m]['min']:.2f} to {Q['speed'][m]['max']:.2f})"
                    for m in ("tf32", "bf16", "fp16")
                )
                + "."
            )
        pts.append(
            "The earlier loop-decomposition test found fp16 costing 0.006 and a July benchmark found it 2.3 times faster at batch 512 on a bigger model; neither pattern appears for these 30k models."
        )
        speed_ok = "speed" in Q and max(v["median"] for v in Q["speed"].values()) >= 1.3
        out.append(
            dict(
                id="precision",
                group="Compute and limits",
                status="supported",
                figs=["precision_ab.png"],
                q="Should the harness train in reduced precision to speed up the search?",
                choice=(
                    (
                        "Keep fp32 as the single harness-wide setting: reduced precision is not faster for these models (speed "
                        + ", ".join(
                            f"{m} {Q['speed'][m]['median']:.2f}x" for m in ("tf32", "bf16", "fp16")
                        )
                        + " of fp32) and accuracy changes are within noise (about 0.004 per model)."
                        if "speed" in Q
                        else "Keep fp32 as the single harness-wide setting unless a larger benchmark shows a real speed gain: accuracy changes are within noise (about 0.004 per model) in every mode."
                    )
                    if not speed_ok
                    else "A reduced-precision mode is faster without a measurable accuracy cost; enforce it harness-wide so no proposal chooses its own."
                ),
                points=pts,
                caveat="One pool, one seed per architecture, so single-mode accuracy changes under about 0.004 cannot be resolved. Models of 0.5M to 3M parameters at 30k sequences are launch- and data-bound; larger models or 300k pools may behave differently. Precision is set through AL_PRECISION in the harness for every model alike.",
                change="A mode that is 1.5 times faster or more on the 100k-300k stage with no change in ensemble r beyond 0.005 would be worth enforcing.",
            )
        )
    # ------------------------------------------------------------ validation variants
    V = o.get("valvar")
    if V:
        pools = V["pools"]
        avg = lambda pool, arm: sum(pools[pool]["arms"][arm]) / len(pools[pool]["arms"][arm])  # noqa: E731
        pts = [
            "Ensemble r averaged over the 18 evaluation sets, by validation set used for early stopping and ensemble weights (V0 stored labels, mean of three seeds; V0c the same sequences relabelled by the current oracle; V1 perturbed genomic; V2 diverse families; V3 union): "
            + "; ".join(
                f"{pl}: "
                + ", ".join(f"{a_} {avg(pl, a_):.4f}" for a_ in ("V0", "V0c", "V1", "V2", "V3"))
                for pl in pools
            )
            + ".",
            "Seed SD of that average: "
            + ", ".join(f"{pl} {pools[pl]['v0_seed_sd']}" for pl in pools)
            + "; rerunning the identical validation set (V0x) moved it by "
            + ", ".join(f"{avg(pl, 'V0x') - avg(pl, 'V0'):+.4f}" for pl in pools)
            + ", so differences under about 0.003 are not resolved.",
            "Relabelling the stored validation set with the current oracle (V0 to V0c) changes the ensemble by "
            + ", ".join(f"{pl} {avg(pl, 'V0c') - avg(pl, 'V0'):+.4f}" for pl in pools)
            + ". Against that fair baseline the variants change it by (V1 / V2 / V3): "
            + "; ".join(
                f"{pl} {avg(pl, 'V1') - avg(pl, 'V0c'):+.4f} / {avg(pl, 'V2') - avg(pl, 'V0c'):+.4f} / {avg(pl, 'V3') - avg(pl, 'V0c'):+.4f}"
                for pl in pools
            )
            + ".",
            "Ranking architectures: Spearman correlation across the 33 models between validation score and test r, change when the same trained models are scored on a variant instead of the genomic set, against the corrected baseline V0c (V1 / V2 / V3): "
            + "; ".join(
                f"{pl} {pools[pl]['rank_delta']['V0c']['V1'][0]:+.3f} / {pools[pl]['rank_delta']['V0c']['V2'][0]:+.3f} / {pools[pl]['rank_delta']['V0c']['V3'][0]:+.3f}"
                for pl in pools
                if "V0c" in pools[pl]["rank_delta"]
            )
            + "; seed-to-seed ceiling "
            + ", ".join(f"{pl} {pools[pl]['noise_ceiling']}" for pl in pools)
            + ".",
        ]
        if V.get("contrasts"):
            c = V["contrasts"]
            pts.append(
                "Pool contrasts (ensemble mean over 18 sets) under each validation set: "
                + "; ".join(
                    f"{k}: "
                    + ", ".join(
                        f"{a_} {v:+.4f}"
                        for a_, v in vals.items()
                        if a_ in ("V0", "V0c", "V1", "V2", "V3")
                    )
                    for k, vals in c.items()
                    if "minus" in k and "difference" not in k
                )
                + ". The ordering of reservoirs does not change under any validation set; the size of the gap moves by up to about 0.01."
            )
        out.append(
            dict(
                id="valvar",
                group=G,
                status="provisional",
                figs=[],
                q="Does the choice of validation set change the results?",
                choice="Fixing the label mismatch matters most; replacing the genomic validation set by perturbed or diverse sets does not give a reliable gain, and no choice changes which reservoir wins.",
                points=pts,
                caveat="One pool each for genomic, random and motif at 30k, one seed per variant arm (three for V0), so differences under about 0.003 in the 18-set average are noise. The variants were similarity-filtered against all evaluation sets and training pools. Cross-scoring (the same trained models scored on each validation set) is the cleaner comparison for ranking; the second seed (phase 2) has not been run.",
                change="A validation variant that changed the pool contrasts by more than 0.01 on a second seed, or ranked architectures better than the relabelled genomic set with an interval excluding zero in all three pools, would change the recommendation.",
            )
        )
    # ------------------------------------------------------------ oracle verification and HepG2
    out.append(
        dict(
            id="oracle",
            group=G,
            status="supported",
            figs=[],
            q="Is the current oracle the proper ten-member chromosome-split ensemble?",
            choice="Yes: ten AlphaGenome-encoder models, each trained on eight chromosome groups, validated on one and tested on one.",
            points=[
                "Members: 10 (outputs/oracle_v3/s2/fold_0 to fold_9), each a stage-1 head plus a stage-2 fully unfrozen encoder, trained on the same 856,252-sequence pool. Training sizes per member are about 678,000 train, 88,000 validation and 90,000 test sequences (79%, 10%, 10%).",
                "Folds: the 24 chromosomes are grouped into 10 folds of 2 to 3 chromosomes (for example chr7, chr13 and chr20 form fold 2; chr15, chr19 and chr21 fold 4; chr1 and chrX fold 0). Member k is tested on fold k, validated on fold k+1 and trained on the other eight; reference and alternate alleles of a variant always sit in the same fold.",
                "The 22,962 designed sequences have no chromosome and are assigned to folds at random, so for those sequences the held-out member is held out by sequence, not by chromosome.",
                "Held-out test Pearson against experiment per member 0.9227 to 0.9394, mean 0.9311 (SD 0.0051). Labels for synthetic pools and evaluation sets are the mean of the ten members; for real held-out sequences (chr7 and chr13 genomic, SNVs, designed) only the member that did not train on them is used.",
            ],
            caveat="Genomic-reservoir pools are drawn from sequences inside the oracle's training pool, so their labels are in-sample for eight of ten members. The stored student validation labels were not from this oracle until the validation-set work found it (see the integrity card).",
            change="A member trained with the wrong fold map, or labels for held-out sets that included members that had trained on them, would change this.",
        )
    )
    out.append(
        dict(
            id="hepg2",
            group=G,
            status="provisional",
            figs=[],
            q="Is the HepG2 side of the differential objective ready?",
            choice="The HepG2 oracle is trained with the same protocol as K562; relabelling the pools and evaluation sets with HepG2 and differential labels is the next step.",
            points=[
                "A full-pool HepG2 oracle (same fold map, same two-stage recipe, separate head per fold) has all ten folds: held-out Pearson 0.9130, 0.9138, 0.9118, 0.9094, 0.9315, 0.9176, 0.9198, 0.9214, 0.9033 and 0.9118, mean 0.9153 (K562: 0.9311). On identical held-out reference and alternate rows the two cell types score 0.917 (K562) and 0.916 (HepG2), so the gap is mostly the designed sequences.",
                "An earlier HepG2 oracle that never saw the designed sequences (mean 0.916) exists and was not used by any code.",
                "Labels: K562 and HepG2 true labels correlate at 0.80 on the measured library; the differential (K562 minus HepG2) has SD 0.73, an independent-noise ceiling of about 0.68, and the existing oracles reach 0.73 on held-out rows, with the sign right in 92% of rows where the true differential exceeds 1.",
                "Designed K562 sequences average 3.96 in K562 and -0.37 in HepG2 (K562-HepG2 correlation 0.10 on that set), so they are strongly K562-specific.",
            ],
            caveat="The labelling script (K562, HepG2 and differential, with provenance) reproduces the stored K562 labels at r = 0.999999 on CPU; its GPU run on the 18 evaluation sets and the 30k replicate pools is not yet done, so differential results for strategies are not available.",
            change="HepG2 held-out accuracy on synthetic pools far below 0.9, or a differential that the students cannot learn above the noise ceiling, would limit the differential comparisons.",
        )
    )
    # ------------------------------------------------------------ what is running now
    out.append(
        dict(
            id="program3",
            group="Experiments launched Oct 22",
            status="running",
            figs=[],
            q="What was launched on Oct 25?",
            choice="Search-loop variants with a coordinator role, an explore proposer and the explore prompt across the reservoir grid.",
            points=[
                "Agent loop with a COORDINATOR: the proposer over-generates six ideas, a coordinator chooses the two most promising (high expected validation r, different from what is trained, unlikely to fail), then the coder, evaluator and reflector run as before. Six paired runs on genomic 30k pools r1 to r6; a variant without the reflector isolates the coordinator.",
                "Agent loop with an explore-style proposer (six runs).",
                "Explore prompt against the default prompt on five reservoirs (genomic, random, motif cell-type-enriched, evoaug, evo2) at 10k and 30k, three to six paired runs per cell (66 runs).",
                "A multi-agent loop run costs about four times the LLM tokens of a single-call run; the whole set is roughly 15 million tokens and is paced by the account's five-hour usage limit, so results arrive over a day or two.",
            ],
            caveat="100k is excluded for GPU cost. Evo2 has no nested or replicate pools, so both arms use one legacy pool; paired comparisons are still valid there, runs are not independent draws of the pool.",
            change="A variant that beats the default by more than 0.005 across reservoirs would change the search recipe.",
        )
    )
    return out


def oct25b_cards(d: dict) -> list[dict]:
    o = d.get("oct22", {})
    out = []
    G = "Results from the Oct 22 program"
    C, Rd, Cv, Ax = o.get("cost"), o.get("reduced"), o.get("converge"), o.get("archexp")
    if C and Rd:
        pts = [
            "Cost of retraining the 33-architecture menu once (H100-equivalent GPU-hours, one seed): "
            + ", ".join(
                f"{k // 1000}k {C[str(k)]['menu_gpu_h_one_seed']:.1f}"
                for k in (10000, 30000, 100000, 300000)
                if str(k) in C
            )
            + f". Time grows as D^{C['fit_exponent']:.2f}; extrapolated: "
            + ", ".join(f"{int(k) // 1000}k {v:.0f}" for k, v in C["pred_gpu_h_one_seed"].items())
            + ". Three seeds or independent pools triple these.",
            "If only K of the 33 architectures are retrained at the next size, chosen by their validation r at the previous size, the ensemble loses (K = 5 / 8 / 12, genomic test; full menu in brackets): "
            + "; ".join(
                f"{int(D) // 1000}k {Rd[D]['K']['5']['prev_val'][0] - Rd[D]['full'][0]:+.4f} / {Rd[D]['K']['8']['prev_val'][0] - Rd[D]['full'][0]:+.4f} / {Rd[D]['K']['12']['prev_val'][0] - Rd[D]['full'][0]:+.4f} ({Rd[D]['full'][0]:.4f})"
                for D in Rd
            )
            + ".",
            "Random choices of the same K lose more at K = 8 ("
            + ", ".join(
                f"{int(D) // 1000}k {Rd[D]['K']['8']['random'][0] - Rd[D]['full'][0]:+.4f}"
                for D in Rd
            )
            + "); picking by the same size's own validation r (which needs the full retrain) is no better than picking by the previous size's (K = 8: "
            + ", ".join(
                f"{int(D) // 1000}k {Rd[D]['K']['8']['own_val'][0] - Rd[D]['K']['8']['prev_val'][0]:+.4f}"
                for D in Rd
            )
            + " difference).",
            "So at the top of the ladder about eight architectures chosen on the previous size's validation r recover essentially the whole ensemble (0.0003 below at 300k), at about a quarter of the cost.",
        ]
        if Cv:
            pts.append(
                "Training is not cut short at any size: "
                + "; ".join(
                    f"{int(k) // 1000}k {v['stopped_by_patience']:.0%} of models stop by patience, {v['cap_hit']:.0%} hit the time cap (median {v['median_epochs']:.0f} of {v['median_planned']:.0f} planned epochs)"
                    for k, v in Cv.items()
                )
                + "."
            )
        out.append(
            dict(
                id="highd",
                group=G,
                status="provisional",
                figs=[],
                q="What does a run at a higher dataset size cost, and how small can the retrained menu be?",
                choice="Search once at a small size, carry the recipes (not text context), and at larger sizes retrain only about eight menu architectures chosen on the previous size's validation r; retrain the full menu only for the final ensembles.",
                points=pts,
                caveat="One retraining per architecture at each size for the genomic reservoir, so differences under about 0.003 are noise; whether a fresh search at 300k would beat the carried menu has not been tested (three runs at 300k are queued). Cost extrapolation assumes the measured scaling continues.",
                change="A fresh search at 300k that beat the carried menu by more than 0.005 at equal model counts would argue for searching again at the largest sizes.",
            )
        )
    if Ax:
        e = Ax["exponents"]
        pts = [
            f"Scaling exponent of each of the {e['n']} architectures alone (genomic reservoir, 10k to 300k, 1 minus r against D on log axes): mean {e['mean']:.3f}, SD {e['sd']:.3f}, 5th to 95th percentile {e['lo']:.3f} to {e['hi']:.3f}; the ensemble's exponent is {e['ensemble']:.3f}. Across reservoirs and evaluation sets the ensemble exponents span 0.11 to 0.31, several times the spread between architectures."
        ]
        for es, lbl in (
            ("genomic", "genomic test"),
            ("ood", "designed set"),
            ("random_32k", "random sequences"),
        ):
            pr = Ax["reservoir_order"][es]
            pts.append(
                f"Reservoir ordering architecture by architecture at 30k ({lbl}; share of the {pr['motif > genomic']['n']} architectures that agree): "
                + "; ".join(f"{k} {v['frac']:.0%}" for k, v in pr.items())
                + "."
            )
        out.append(
            dict(
                id="archexp",
                group=G,
                status="supported",
                figs=[],
                q="Do the scaling and reservoir conclusions hold architecture by architecture?",
                choice="Yes: individual architectures scale with nearly the same exponent as the ensemble, and the reservoir orderings that the ensembles show hold in nearly every individual architecture.",
                points=pts,
                caveat="Exponents from four sizes per architecture, one to three retrainings each; reservoir orderings from one pool per reservoir at 30k.",
                change="Architectures whose exponents or reservoir orderings disagree with the ensemble would show the ensemble hides an architecture-specific effect.",
            )
        )
    out.append(
        dict(
            id="program4",
            group="Experiments launched Oct 22",
            status="running",
            figs=[],
            q="What else was launched on Oct 25?",
            choice="A 300k search check, HepG2 relabelling and H100-first routing.",
            points=[
                "Three AutoResearch runs of 20 models on genomic 300k replicate pools (H100 only), to compare with the retrained menu at 300k at equal model counts, as at 10k, 30k and 100k. Each model takes about 35 minutes on an H100, so a run is about a day of training.",
                "HepG2 and differential relabelling of the 18 evaluation sets and the 30k replicate pools with the new HepG2 oracle (about 1.6 GPU-hours).",
                "Routing: every search run may use H100s first (V100 as overflow); the default queue is reserved for H100 starts so they are not blocked by V100 jobs; at most 40 search runs are active at once.",
            ],
            caveat="The 300k runs are the slowest item in the program; other users' jobs decide when H100s are free.",
            change="If the 300k search ties the carried menu, the per-size search can be dropped at every size above 30k.",
        )
    )
    return out


def build(d: dict) -> list[dict]:
    L, M, I, T = d.get("loop", {}), d.get("menu", {}), d.get("instances", {}), d.get("topk", {})
    cad, kb, rep = d.get("cadence", {}), d.get("kb", {}), d.get("replicates", [])
    rr = M.get("reference_rank", {})
    sp = {k: M.get(k, {}) for k in ("spearman_10k_30k", "spearman_30k_100k", "spearman_10k_100k")}
    ens = M.get("ensembles", {})
    nat, arch = L.get("native_reported", [0, 0, 0]), L.get("arch_reported", [0, 0, 0])
    wins = sum(1 for r in rep if r["pooled"] > max(r["singles"])) if rep else 0
    gains = [r["pooled"] - max(r["singles"]) for r in rep]
    meansing = [r["pooled"] - sum(r["singles"]) / len(r["singles"]) for r in rep]
    top5 = T.get("ens", [0] * 7)
    cadg = lambda r, a: cad.get(f"{r}|{a}", [float("nan"), 0])
    ctx = d.get("ctx100_progress", {})
    c100 = d.get("ctx100", {})
    n_ctx = sum(sum(v) for v in ctx.values())
    fam = d.get("family", {}).get("progress", {})
    n_fam = {
        f: sum(v for k, v in fam.items() if k.startswith(f + "|"))
        for f in ("random", "motif_ct_enriched", "evo2", "evoaug_paper2023")
    }
    pa = d.get("prompt_arms", {}).get("summary", {})
    pe = {k.split()[0]: v.get("ens_at_n", []) for k, v in pa.items()}
    mean = lambda v: sum(v) / len(v) if v else float("nan")
    bp = d.get("baselines_progress", {})
    bt = {k: sum(v) for k, v in bp.items()}
    bv = d.get("baselines_vs_llm", {})
    rt = M.get("retest", {})
    mc = d.get("mix_controlled", {}).get("rows", {})
    ind = d.get("indep30", {})
    i100 = d.get("indep100_genomic", {})
    f100 = d.get("family100", {}).get("ens", {})
    un = d.get("family_union", {})
    nl = d.get("nested_ladder", {})
    tk100 = d.get("topk100", {})
    pr = d.get("proposer", {}).get("summary", {})
    ms = d.get("menu_scaling", {})
    fe = d.get("family_ens", {})
    sat = d.get("saturation100", {})
    m3 = d.get("menu300", {})
    c100 = d.get("ctx100", {})
    g = lambda D, e="genomic": ms.get("genomic", {}).get(str(D), {}).get(e, float("nan"))  # noqa: E731
    per_dbl = lambda a, b, e="genomic": (g(b, e) - g(a, e)) / (__import__("math").log2(b / a))  # noqa: E731
    B = d.get("budget", {})
    mf = d.get("menu_fam", {})

    cards = [
        dict(
            id="loop",
            group="Training and evaluation",
            status="supported",
            figs=["loop_decomposition.png"],
            q="Why did earlier scaling runs report r = 0.69-0.70 for a LegNet that scores 0.73 in the architecture loop?",
            choice="Train every model, from every source, in one common loop and report RC-averaged predictions.",
            points=[
                f"Same network, same HPs, three configs: reported test r {f3(min(nat))}-{f3(max(nat))} in the old loop vs {f3(min(arch))}-{f3(max(arch))} in the arch loop ({sgn(L.get('total_gap'))} on average).",
                f"Two parts: reverse-complement averaging at test time {sgn(L.get('rc_avg'))}; the loop itself {sgn(L.get('loop_fwd_matched'))} on matched forward-only predictions.",
                f"Inside the loop, the per-sample crop-shift ({sgn(L['flips'].get('A_flip_shift'))}) and fp16 ({sgn(L['flips'].get('A_flip_amp'))}) cost accuracy; early-stop rule, gradient clip and the time cap do not matter (within noise).",
            ],
            caveat="Every scaling-law number measured before Oct 20 used the old loop and forward-only predictions. The shape of the curves probably survives; the absolute levels do not. The retrained-menu ensembles re-measure the ladder in the common loop (see the scaling card below), and a nested genomic pool reproduces the old curve within about 0.01.",
            change="A config where the old loop wins would mean the ingredients interact with the architecture and we need per-architecture loops.",
        ),
        dict(
            id="arch",
            group="Training and evaluation",
            status="supported",
            figs=["arch_vs_legnet.png"],
            q="Does a freely-designed architecture beat a well-tuned LegNet once both are trained the same way?",
            choice="Do not claim an architecture advantage for free-form search. Its value is diversity for the ensemble and a wide, assumption-free search.",
            points=[
                f"LegNet-eff HP-search winners: {', '.join(f3(v) for v in d['arch_vs_legnet']['LegNet-eff configs'])}. Top-3 free-form architectures: {', '.join(f3(v) for v in d['arch_vs_legnet']['free-form architectures'])}.",
                "Single seeds scatter by about +-0.005, so these are indistinguishable.",
                f"Retraining the 33-model menu three times each puts the reference LegNet at rank {rr.get('10k', '?')} / {rr.get('30k', '?')} / {rr.get('100k', '?')} of 33 at 10k / 30k / 100k. With one or two seeds it had looked like 2nd-3rd at 30k and 100k, so rankings within this pack are seed-noise dominated and no architecture, LegNet included, is distinguishable from the pack.",
            ],
            caveat="Genomic reservoir at 30k only; 3 seeds per architecture.",
        ),
        dict(
            id="menu",
            group="Dataset-size ladder",
            status="supported",
            figs=["menu_across_D.png"],
            q="Can we pick architectures at a small D and reuse them at a larger D?",
            choice="No. Retrain the whole candidate pool at every D and ensemble there; do not pre-select the top few at 10k.",
            points=[
                f"Rank correlation of the same {M.get('n_entries', 33)} architectures between D: 10k vs 30k, validation {f3(sp['spearman_10k_30k'].get('val'), 2)} / test {f3(sp['spearman_10k_30k'].get('test'), 2)}; 30k vs 100k, validation {f3(sp['spearman_30k_100k'].get('val'), 2)} / test {f3(sp['spearman_30k_100k'].get('test'), 2)}.",
                f"Ensemble of the whole retrained menu: {', '.join(f3(v) for v in ens.get('all', []))} at 10k / 30k / 100k. Top-5 chosen at 10k and retrained: {', '.join(f3(v) for v in ens.get('top5_at_10k', []))}, i.e. {', '.join(sgn(a - b) for a, b in zip(ens.get('top5_at_10k', []), ens.get('all', [])))} against the whole pool. HP-only search (old loop): {', '.join(f3(v) for v in ens.get('hp_only', []))}.",
                (
                    f"Within-D retest ceiling (one seed against the mean of the others): 10k {rt['10k']:.2f}, 30k {rt['30k']:.2f}, 100k {rt['100k']:.2f}. The 10k-to-30k agreement is far below these, so the ranking really changes; 30k-to-100k sits at them."
                    if all(rt.get(k) is not None for k in ("10k", "30k", "100k"))
                    else "Retest ceilings pending."
                ),
                "Retrainings per architecture: "
                + ", ".join(f"{k} {v}" for k, v in M.get("seeds_per_D", {}).items())
                + ".",
                (
                    "On the nested genomic pool (one seed per size) the agreement is 10k-30k "
                    + f"{nl['spearman'].get('10k-30k', float('nan')):.2f}, 30k-100k {nl['spearman'].get('30k-100k', float('nan')):.2f}, 100k-300k {nl['spearman'].get('100k-300k', float('nan')):.2f}"
                    + ", against 0.13 / 0.66 / 0.43 on the old three-seed ladder. The pattern does not repeat, so rank agreement across sizes is low to moderate everywhere and mostly noise-limited; it is not evidence that small-D picks transfer or fail to."
                    if nl.get("spearman")
                    else "Nested-pool rank agreement pending."
                ),
            ],
            caveat="Three retrainings per architecture at every D. The practical cost of choosing the top 5 at 10k anyway is small (a few thousandths in ensemble r), so the case for retraining everything is about robustness, not size. The HP-only comparison mixes harnesses.",
            change="Three seeds on the nested pools would settle whether any rank carries over between sizes; until then retraining everything is the safe choice.",
        ),
        dict(
            id="menufam",
            group="Dataset-size ladder",
            status="supported",
            figs=["menu_families.png"],
            q="Do the same architectures do well on random, motif, evo2 and evoaug data, or does each reservoir need its own search?",
            choice="Menu is reservoir-agnostic: retrain the same pool on each reservoir's data.",
            points=[
                "All "
                + str(mf.get("n_menu", 33))
                + " architectures retrained at 30k on each reservoir, three seeds each.",
                "Rank agreement with genomic (Spearman of test r): "
                + (
                    ", ".join(
                        f"{k.replace('_', ' ')} {v:.2f}"
                        for k, v in mf.get("spearman_vs_genomic", {}).items()
                    )
                    or "pending"
                )
                + (
                    f". Retest ceiling (genomic, one retraining vs the mean of the others): {mf['retest_spearman_genomic']:.2f}."
                    if "retest_spearman_genomic" in mf
                    else "."
                ),
                "Menu ensemble test r (genomic eval): "
                + ", ".join(
                    f"{r['res'].replace('_', ' ')} {r['ensemble']:.3f}" for r in mf.get("rows", [])
                )
                + ". Motif-enriched is best, random worst; the best single model trails each ensemble by 0.02-0.03.",
                "Rankings agree across reservoirs at about the retest ceiling for motif, evo2 and evoaug. Random is the exception (0.29): what works on random sequences differs more.",
            ],
            caveat="30k only; the retest ceiling comes from genomic. Runs on V100 as well as H100 with GPU-scaled time caps.",
            change="If a larger-D check shows random-reservoir rankings diverging further, random needs its own pool.",
        ),
        dict(
            id="menu300",
            group="Dataset-size ladder",
            status="supported",
            figs=["menu_300k.png"],
            q="Can the architecture pool chosen at 100k be reused at 300k, or does 300k need its own retraining?",
            choice="Retrain the whole pool at 300k. Rankings at 100k carry over only partly.",
            points=[
                f"All {m3.get('done', 0)} architectures retrained at 300k, three seeds each (none hit the 120-minute cap; median 32 minutes)."
            ]
            + (
                [
                    f"Rank agreement 100k vs 300k: test {m3['spearman_test_100k_300k']:.2f}, validation {m3['spearman_val_100k_300k']:.2f}; 30k vs 300k test {m3['spearman_test_30k_300k']:.2f}; {m3.get('top5_overlap', '?')} of the top 5 are shared. The within-D retest ceiling at 300k is {m3.get('retest') or float('nan'):.2f}, clearly above this, so rankings keep changing with D."
                ]
                if "spearman_test_100k_300k" in m3
                else ["Rank agreement fills in as architectures finish."]
            )
            + (
                [
                    f"Ensemble of the retrained menu at 300k: genomic {m3['ensemble']['genomic']:.3f}, out-of-distribution {m3['ensemble']['ood']:.3f}."
                ]
                if m3.get("ensemble")
                else []
            ),
            caveat="Genomic reservoir only at 300k. Both sizes use three-seed means.",
            change="If a search run at 300k found architectures well outside this pool, retraining the old pool would not be enough there.",
        ),
        dict(
            id="indep",
            group="Search and pooling",
            status="supported",
            figs=["independent_pools_30k.png"],
            q="Should every independent search run get its own labelled pool, or should all runs share one?",
            choice="Give each independent run its own pool (a separate labelled draw), nested across sizes inside it. Keep one nested corpus for the final retrain.",
            points=[
                "Menu test at 30k (33 architectures): three seeds on one pool (A) against three independent pools (B), equal model counts (99 each). Pooled ensemble, genomic evaluation, A vs B: "
                + "; ".join(
                    f"{k.replace('_', ' ')} {v['A']['pooled']['genomic']:.3f} vs {v['B']['pooled']['genomic']:.3f} ({v['B']['pooled']['genomic'] - v['A']['pooled']['genomic']:+.3f})"
                    for k, v in ind.items()
                )
                + ".",
                "Independent pools win in every reservoir on all three evaluation sets (12 of 12 comparisons). Out-of-distribution gain "
                + ", ".join(
                    f"{v['B']['pooled']['ood'] - v['A']['pooled']['ood']:+.3f}"
                    for v in ind.values()
                )
                + "; random-sequence gain "
                + ", ".join(
                    f"{v['B']['pooled']['random_32k'] - v['A']['pooled']['random_32k']:+.3f}"
                    for v in ind.values()
                )
                + ".",
                "Spread of one replicate's ensemble across the three replicates (SD): seeds on one pool "
                + ", ".join(
                    f"{__import__('statistics').stdev(v['A']['ens']):.4f}" for v in ind.values()
                )
                + "; separate pools "
                + ", ".join(
                    f"{__import__('statistics').stdev(v['B']['ens']):.4f}" for v in ind.values()
                )
                + ". Shared-pool replicates understate run-to-run variance.",
                "Architecture rankings repeat about equally well across seeds and across pools (Spearman "
                + ", ".join(f"{v['A']['rank']:.2f}/{v['B']['rank']:.2f}" for v in ind.values())
                + " for seeds/pools), so ranking noise is mostly training noise.",
                (
                    f"AutoResearch at genomic 100k: three runs with their own pools pool to {i100['independent']['pooled']:.3f} (single runs {min(i100['independent']['single']):.3f}-{max(i100['independent']['single']):.3f}) against {i100['shared']['pooled']:.3f} for runs sharing the old 100k pool (single runs {min(i100['shared']['single']):.3f}-{max(i100['shared']['single']):.3f}). This comparison is confounded: the shared pool was the old, biased genomic 100k set."
                    if i100
                    else "100k comparison pending."
                ),
            ],
            caveat="The clean test is the 30k menu comparison (same code, same pools, only the data draw differs). The 100k search comparison points the same way but cannot separate independence from the old pool's bias. The final retrain at each size should still use one nested corpus.",
            change="If nine seeds on one pool matched three pools of three seeds, the gain would be model count and not data diversity.",
        ),
        dict(
            id="mix",
            group="Search and pooling",
            status="provisional",
            figs=["mix_controlled.png", "family_union.png"],
            q="Do models trained on several reservoirs ensemble better than models trained on genomic data only?",
            choice="Not settled. After matching model counts the gain from mixing is small (+0.005 over genomic alone on genomic sequences, none out of distribution) and motif data alone does as well or better. The mixture sweep and architecture-source test now running decide it.",
            points=[
                "What the first comparison was: the same 31-33 architectures (the genomic-derived menu), retrained on each reservoir's data, then ensembled. The architectures did not come from searches on the other reservoirs, and the first union had 165 models against 33 for genomic alone, so it was not count-controlled.",
                (
                    "Count-controlled version (same architectures, k independent pools of one reservoir against one pool from each of k different reservoirs; genomic eval): "
                    + "; ".join(
                        f"k = {k}: mixed {r['mixed']['genomic']:.4f}, same-reservoir mean {r['same_mean']['genomic']:.4f}, genomic only {r['same']['genomic']['genomic']:.4f}, motif only {r['same']['motif_ct_enriched']['genomic']:.4f}"
                        for k, r in mc.items()
                        if k != "1"
                    )
                    + "."
                    if mc
                    else "Count-controlled comparison pending."
                ),
                (
                    "Out of distribution the mixed ensemble and the same-reservoir mean are equal ("
                    + ", ".join(
                        f"{r['mixed']['ood']:.3f} vs {r['same_mean']['ood']:.3f}"
                        for k, r in mc.items()
                        if k != "1"
                    )
                    + "), and on random sequences mixing is below the same-reservoir mean, so mixing is not a free improvement."
                    if mc
                    else ""
                ),
            ],
            caveat="Model-level mixing of genomic-derived architectures only. Whether architectures searched on several reservoirs, or a training corpus that mixes reservoirs, does better is what the running experiments test.",
            change="A mixed-corpus or mixed-search configuration that beats motif-only by more than the run-to-run noise (about 0.005) would settle it in favour of mixing.",
        ),
        dict(
            id="program",
            group="Experiments launched Oct 22",
            status="supported",
            figs=[],
            q="What did the Oct 22 program run, and which assumption does each experiment test?",
            choice="Four experiments launched Oct 22 on independent nested pools. All of them have finished; their results are in the Oct 22 results cards above.",
            points=[
                (
                    "Progress (done of total cells, running now): "
                    + "; ".join(
                        f"{k} {v['done']} of {v['total']} ({v['running']} running)"
                        for k, v in d.get("program_status", {}).get("groups", {}).items()
                    )
                )
                if d.get("program_status")
                else "Progress pending.",
                "Architecture source (assumptions C2, A5): 15 independent search runs on genomic only, 15 on random only, a genomic+random mix (8 + 7) and the existing 5-reservoir mix; top 3 per run, equal candidate counts, each arm retrained on genomic, random and motif data.",
                "Run-count curves (A5): ensemble accuracy against the number of independent runs, genomic at 10k, 30k and 100k, random and motif at 30k.",
                "Unseen reservoirs (C1, C3): three bespoke search runs each on motif_shared_core, motif_syntax_core, zoonomia_orthologs, mutagenesis, gc_matched and dinuc_shuffle, against the genomic-derived menu retrained on the same pools.",
                "Mixture sweep (C4): 21 mixed training corpora at 30k, 33 architectures each, then an ensemble across the best mixtures.",
            ],
            caveat="About 105 search runs plus a few thousand retrainings. Search runs stop at each 5-hour usage limit and resume on their own after the reset.",
            change="Any arm that beats the fixed menu by more than the run-to-run noise (about 0.005) would argue for per-reservoir search.",
        ),
        dict(
            id="inst",
            group="Search and pooling",
            status="supported",
            figs=["instances_and_topk.png"],
            q="How many models should one AutoResearch instance train, how many should we keep, and how many instances do we need?",
            choice="About 20-30 models per instance, keep the top 8 by validation, pool several independent instances.",
            points=[
                f"One instance's ensemble saturates around 20-40 models (mean of runs: {', '.join(f3(v) for v in I.get('mean', []))} at n = {I.get('n')}).",
                f"Pooling the top-k of three instances: k = 1: {f3(top5[0])}, k = 3: {f3(top5[2])}, k = 8: {f3(top5[4])}, all: {f3(top5[6])}. Single instances with all their models: {f3(min(T.get('singles', [0])))}-{f3(max(T.get('singles', [0])))}.",
                f"Between-instance spread of the final ensemble: SD {f3(I.get('sd_between_runs'), 4)}.",
                (
                    "At 100k on independent pools (20 models per run, three runs pooled), keeping the top k from each run gives genomic "
                    + ", ".join(
                        f"k={k}: {v:.4f}"
                        for k, v in tk100.get("genomic", {}).items()
                        if k in ("1", "3", "5", "8", "20")
                    )
                    + "; motif, random and evoaug flatten by k = 2-3 as well. Three to five per run is enough at 100k."
                    if tk100.get("genomic")
                    else "100k top-k pending."
                ),
            ],
            caveat="Three Sonnet-5.5 instances at 30k; three per reservoir at 100k.",
        ),
        dict(
            id="rep",
            group="Search and pooling",
            status="supported",
            figs=["replicates.png"],
            q="Is pooling independent runs actually better than a single run, and by how much?",
            choice="Always run several independent instances per reservoir and pool them.",
            points=[
                f"Three independent runs, 3 reservoirs x 3 D: the pooled ensemble beats the best single run in {wins} of {len(rep)} cells (gain over the best single {sgn(min(gains) if gains else float('nan'))} to {sgn(max(gains) if gains else float('nan'))}; over the average single {sgn(min(meansing) if meansing else float('nan'))} to {sgn(max(meansing) if meansing else float('nan'))}).",
                "Single-run SD is about 0.009, the same size as many between-strategy gaps: any single-run comparison below that is noise.",
            ],
            caveat="HP-only runs in the old loop. The replicate structure (not the absolute r) is what carries over.",
        ),
        dict(
            id="llm",
            group="Proposer and prompt",
            status="provisional",
            figs=["llm_vs_classical.png"],
            q="Is an LLM proposer better than classical samplers, and if so is it because of search or because of its prior?",
            choice="Use the LLM with 2 proposals x 30 rounds, but do not claim it out-searches classical methods.",
            points=[
                f"Cadence study, genomic: LLM 2x30 {f3(cadg('genomic', 'llm2x30')[0])}, 6x10 {f3(cadg('genomic', 'llm6x10')[0])}, 1x60 {f3(cadg('genomic', 'llm1x60')[0])} (incomplete), classical multi6 {f3(cadg('genomic', 'multi6')[0])}; the LLM led by +0.02 to +0.045, but its proposal quality is flat from round 0, so the edge is its prior.",
                "Repaired baselines (6 runs each, HP-only, genomic 30k, ensemble of the first 20 models): "
                + "; ".join(
                    f"{lab} mean {mean(bv.get(k, [])):.3f} (range {min(bv[k]):.3f}-{max(bv[k]):.3f}, n = {len(bv[k])})"
                    for k, lab in (
                        ("random_broad", "random in the full box"),
                        ("prior_marginal", "random in the LLM-like prior box"),
                        ("prior_tpe", "TPE in that box"),
                        ("llm_first30", "LLM 2x30 first 20"),
                    )
                    if bv.get(k)
                )
                + ".",
                "Sampling the box the LLM tends to propose from, with no LLM at all, matches the LLM in this HP-only setting; plain random sampling in the full box is about 0.004-0.005 behind, inside the single-run SD of 0.009.",
            ],
            caveat="HP-only, one reservoir, one size; n = 5-6 runs per baseline and 2 LLM runs with 20 models. The case for the LLM therefore rests on what the baselines cannot do: write new architectures, and propose with knowledge of what has been tried.",
            change="If baselines also matched the LLM on the free-architecture task (which they cannot run), the LLM would be unnecessary.",
        ),
        dict(
            id="kb",
            group="Proposer and prompt",
            status="supported",
            figs=["kb_prompts.png"],
            q="Does the knowledge base or a particular prompt wording make the proposer better?",
            choice="Keep the default prompt; do not spend more budget on prompt wording.",
            points=[
                f"Knowledge base on vs off, 5 cells: mean {sgn(kb.get('mean'))}, individual differences {', '.join(sgn(x) for x in kb.get('diffs', []) if x is not None)}.",
                "Six prompt styles x 2 reps at genomic 30k: explore and reflect sit about +0.01 above the default, critic below, but each style's two reps straddle the single-run noise.",
            ],
            caveat="HP-only harness. Effects below about 0.01 are not detectable at one run per cell.",
        ),
        dict(
            id="prompt",
            group="Proposer and prompt",
            status="supported",
            figs=["prompt_arms.png"],
            q="Should the free-architecture proposer be given guidance (known-good pieces, a reference LegNet, a build-then-explore schedule) or be left wide open?",
            choice="Give it the known-good guidance and the ensemble note, without a phase schedule (arm W). Leaving it wide open (N) is measurably worse.",
            points=[
                "Arms: P = guidance + reference + build-on-what-works for 6 rounds then explore; W = guidance only; N = nothing. Three independent runs per arm.",
                f"At equal n = {d.get('prompt_arms', {}).get('n_eq', '?')} models, mean ensemble test r: P {f3(mean(pe.get('P', [])))}, W {f3(mean(pe.get('W', [])))}, N {f3(mean(pe.get('N', [])))}; N is {sgn(mean(pe.get('N', [])) - mean(pe.get('W', [])))} from W, about four times the between-run SD of 0.005.",
                "Top-5 validation r per run: P 0.823-0.833, W 0.830-0.833, N 0.793-0.817. The build-then-explore phases add nothing over plain guidance.",
            ],
            caveat="Sonnet 5.5, genomic 30k only; 3 reps per arm. The wide-open preference is kept as the default for everything the guidance does not cover (architecture families, depth, width).",
            change="If N matched W on another reservoir, we would drop the guidance there too.",
        ),
        dict(
            id="proposer",
            group="Proposer and prompt",
            status="supported",
            figs=["proposer_models.png"],
            q="Which model should write the architectures?",
            choice="Sonnet 5.5. Opus 5.5 is about +0.005 ahead, which is not enough to justify its cost; Sonnet 4.5 is slightly behind.",
            points=[
                "Same prompt (build-on-what-works), genomic 30k, three runs each for Sonnet 5.5, Sonnet 4.5 and Opus 5.5; all nine runs are complete.",
                f"Ensemble test r at equal n = {d.get('proposer', {}).get('n_eq', '?')} models (mean over runs): "
                + "; ".join(
                    f"{k} {mean(v.get('ens_at_n', [])):.4f}"
                    for k, v in pr.items()
                    if v.get("ens_at_n")
                )
                + ". Run-to-run SD is about 0.002-0.005, so Opus is ahead by under two standard errors.",
                "Models per run: "
                + "; ".join(
                    f"{k.replace('|', ' run ')} {v}"
                    for k, v in d.get("proposer", {}).get("cells", {}).items()
                )
                + ".",
            ],
            caveat="Genomic 30k only. A +0.005 gain would matter if confirmed on another reservoir.",
            change="A gain of 0.01 or more for Opus at equal models on a second reservoir would justify its higher token cost.",
        ),
        dict(
            id="ctx",
            group="Dataset-size ladder",
            status="supported",
            figs=["context_transfer.png"],
            q="When AutoResearch moves from 30k to 100k, should the proposer see what worked at 30k?",
            choice="No. Start every D's search from the same prompt; handing results across sizes does not help.",
            points=[
                "Four arms at genomic 100k, 15 rounds x 2 proposals, four independent runs each: no context, top architectures by validation from 30k (T), T plus a trajectory summary (J), top architectures from 10k as the stale control (X).",
                "Ensemble test r at "
                + str(c100.get("n_eq", 30))
                + " models per run (mean, runs): "
                + "; ".join(
                    f"{k} {mean(c100[k]):.4f} (n = {len(c100[k])})"
                    for k in ("none", "T", "J", "X")
                    if c100.get(k)
                )
                + ".",
                "T and J are within +0.002 of no context. The stale 10k context scores highest (about +0.0035, under 2 standard errors), which is the opposite of what useful context would predict. Earlier 30k pilot and the replicate of the one 100k gain also showed nothing.",
            ],
            caveat="One J run and one T run are missing from the counts because the exploded-prediction screen removed models. Genomic reservoir only.",
            change="A gain of 0.01 or more for T or J across runs would justify the hand-off.",
        ),
        dict(
            id="fam",
            group="Search and pooling",
            status="supported",
            figs=["families.png", "families_100k.png"],
            q="Does a separate AutoResearch search on each reservoir beat reusing the genomic-derived architecture pool?",
            choice="Reuse the pool: search on genomic data and retrain the pool on each reservoir's data. A fresh per-reservoir search matches it at 30k and at 100k but costs LLM budget.",
            points=[
                "30k: five reservoirs x three independent runs (60 models each, guidance-only prompt). Mean validation r of each run's 5 best models: "
                + "; ".join(
                    f"{k.replace('_', ' ')} {min(v):.3f}-{max(v):.3f}"
                    for k, v in d.get("family", {}).get("top5", {}).items()
                )
                + ". Runs on the same reservoir agree to about 0.003.",
                "30k pooled ensemble test r (top 8 of 3 runs / all models / retrained menu): "
                + "; ".join(
                    f"{k.replace('_', ' ')} {v['ar_top8']:.3f} / {v['ar_all']:.3f} / {v['menu']:.3f}"
                    for k, v in fe.items()
                    if v.get("menu") is not None
                )
                + ". Search and menu are within 0.006 everywhere.",
                "100k, each run on its own pool, 20 models per run, top 8 of 3 pooled, against the retrained menu (genomic eval): "
                + "; ".join(
                    f"{k.replace('_', ' ')} {v['ar_top8']:.3f} vs {v['menu']:.3f}"
                    for k, v in f100.items()
                    if v.get("menu") is not None
                )
                + ". Again within 0.004, with the menu a little ahead out of distribution.",
                "So a search on one reservoir finds nothing that the retrained pool does not already have, on any of the four reservoirs tested, at either size.",
            ],
            caveat="Tested on one representative member per family (motif_ct_enriched, evoaug_paper2023, random, genomic); other members of each family were not tested. The menu was distilled from genomic searches at 30k, so this is a transfer result.",
            change="If other members of a family (for example motif_jaspar or motif_grammar) showed a gap between search and menu, the family representatives would be too narrow.",
        ),
        dict(
            id="ens",
            group="Ensembling",
            status="carried",
            figs=[],
            q="How should the pooled candidates be combined?",
            choice="Non-negative ElasticNet on the validation predictions of all models (enet_all_pos), z-scored per model so that one diverged model cannot zero the fit.",
            points=[
                "Beat greedy top-K selection by about +0.01 in the earlier ensemble-construction study; uncertainty weighting and diversity (DPP) selection gave no gain.",
                "The z-scored version was added after one exploded model (predictions ~1e20) zeroed the raw ElasticNet.",
                "Audit of 2,040 saved free-form and menu models: 16 (0.8%) output |prediction| > 1e3 on at least one evaluation set while validating normally (the worst reached 1e24). The architecture loop and menu retraining only check the validation predictions, so the ensemble step must screen every evaluation set before fitting; this screen is now in the ledger generator and has to be in the final pipeline.",
            ],
            caveat=f"Earlier experiments are in the previous artifact: {OLD_ARTIFACT}",
        ),
        dict(
            id="sched",
            group="Dataset-size ladder",
            status="provisional",
            figs=["saturation_100k.png"],
            q="How many rounds should each D get as the ladder climbs?",
            choice="Fewer rounds at larger D: about 20 models (10 rounds of 2) at 100k and above, 60 models at 10k-30k.",
            points=(
                (
                    [
                        f"Ensemble gain of one instance from 6 to 30 models: {sat['mean100'][-1] - sat['mean100'][0]:.4f} at 100k ({sat.get('n100', '?')} instances) vs {sat['mean30'][-1] - sat['mean30'][0]:.4f} at 30k ({sat.get('n30', '?')} instances).",
                        f"The last 12 models (18 to 30) add {sat['mean100'][-1] - sat['mean100'][3]:.4f} at 100k vs {sat['mean30'][-1] - sat['mean30'][3]:.4f} at 30k, at or below single-run noise (about 0.003).",
                    ]
                )
                if sat.get("mean100")
                else ["Saturation curve pending."]
            )
            + [
                "Pooling several instances adds more than extra rounds in one instance (see the pooling card)."
            ],
            caveat="300k not measured; saturation is judged by the ensemble of one instance only. The 30k curve has three instances and the 100k curve fourteen.",
            change="If a 300k instance were still climbing at 20 models, the schedule would need a longer 300k stage.",
        ),
        dict(
            id="remeasure",
            group="Dataset-size ladder",
            status="provisional",
            figs=["menu_scaling.png"],
            q="How does accuracy scale with dataset size when everything is trained in the corrected loop?",
            choice="Report scaling from the retrained menu ensembles (33 architectures, up to three seeds), not from the old single-loop HP-search cells.",
            points=[
                f"Genomic reservoir, ensemble test r at 10k / 30k / 100k / 300k: {g(10):.3f} / {g(30):.3f} / {g(100):.3f} / {g(300):.3f}. Gain per doubling: {per_dbl(10, 30):.3f}, {per_dbl(30, 100):.3f}, {per_dbl(100, 300):.3f}, so returns diminish but the curve is not flat at 300k.",
                f"Out-of-distribution sequences, same sizes: {g(10, 'ood'):.3f} / {g(30, 'ood'):.3f} / {g(100, 'ood'):.3f} / {g(300, 'ood'):.3f}; this curve keeps accelerating (gain per doubling {per_dbl(10, 30, 'ood'):.3f}, {per_dbl(30, 100, 'ood'):.3f}, {per_dbl(100, 300, 'ood'):.3f}).",
                "Other reservoirs at 10k / 30k / 100k (genomic eval): "
                + "; ".join(
                    f"{k.replace('_', ' ')} {ms[k]['10']['genomic']:.3f} / {ms[k]['30']['genomic']:.3f} / {ms[k]['100']['genomic']:.3f}"
                    for k in ms
                    if k != "genomic" and all(x in ms[k] for x in ("10", "30", "100"))
                )
                + ". Motif-enriched is highest on genomic sequences; random is lowest there but highest out of distribution at 100k ("
                + (
                    f"{ms['random']['100']['ood']:.3f}"
                    if "random" in ms and "100" in ms["random"]
                    else "n/a"
                )
                + ").",
            ],
            caveat=(
                "The old genomic pools were not nested and differed in label distribution by size. Repeating the ladder on a properly nested genomic pool (one seed) gives "
                + "; ".join(
                    f"{k}k {nl['ens'][k]['genomic']:.3f}"
                    for k in ("10", "30", "100", "300")
                    if k in nl.get("ens", {})
                )
                + ", within about 0.01 of the old ladder at every size, and the out-of-distribution curve agrees to within 0.02. The old scaling conclusions stand. "
                if nl.get("ens")
                else ""
            )
            + "These are menu ensembles, not AutoResearch cells, and non-genomic reservoirs have one seed at 10k and 100k. The old HP-only numbers (0.686 / 0.743 / 0.780 on genomic) are lower at every size.",
            change="A full AutoResearch re-run per size would be needed only if the menu is thought to understate what search finds at 100k and 300k.",
        ),
        dict(
            id="budget",
            group="Compute and limits",
            status="supported",
            figs=["llm_budget.png"],
            q="Can we afford the remaining programme within the LLM usage limit?",
            choice="Dispatcher pauses on the session limit, resumes after the reset, and caps concurrent LLM cells at 14.",
            points=[
                "One usage-limit stop on Oct 21 at 11:03 with 18 concurrent LLM cells; the dispatcher paused, probed after the 12:10 reset and resumed at 12:12 without help. The cap is now 15 cells (about 0.65M tokens an hour), which leaves room for interactive use.",
                f"HPC cells used about {B.get('raw_tokens', 0) / 1e6:.0f}M raw tokens ({B.get('output_tokens', 0) / 1e6:.1f}M output) over the last {B.get('hours', '?')} active hours ({B.get('calls', '?')} calls).",
                "The limit was hit at 20:10 on Oct 20 (HPC 3.0M plus laptop 1.7M raw tokens in the window). Remaining programme is roughly 2,100 calls.",
            ],
            caveat="Cap is weighted and unpublished; numbers are raw token counts and do not include cache reads.",
        ),
    ]
    from ledger_oct26 import oct26_cards
    from ledger_oct27 import oct27_cards, patch_cards

    all_cards = (
        oct22_cards(d) + oct23_cards(d) + oct25_cards(d) + oct25b_cards(d) + oct26_cards(d) + cards
    )
    all_cards += oct27_cards(d)
    patch_cards(all_cards, d)
    return all_cards


CSS = """
.sub3{margin:22px 0 6px;font-size:15px}
table.cov{width:auto}
table.cov td.cv{text-align:center;min-width:46px}
td.cv-x{background:rgba(42,125,79,.16)}
td.cv-q{background:rgba(106,79,163,.16)}
td.cv-pool{background:rgba(217,130,43,.16)}
td.cv-cap{background:rgba(139,151,166,.2)}
.note{font-size:12.5px;opacity:.8;max-width:78ch}
:root{--bg:#f5f7f9;--surface:#ffffff;--ink:#14202e;--muted:#5a6878;--line:#dde3ea;--accent:#1d6fa5;--accent-ink:#ffffff;
--ok:#1f6f45;--ok-bg:#e3f2ea;--warn:#8a5a00;--warn-bg:#fbefd2;--run:#1d6fa5;--run-bg:#e1eef8;--open:#9c3a2e;--open-bg:#f8e3df;--plan:#4d5a69;--plan-bg:#e8ecf1;--carry:#5b4a99;--carry-bg:#ebe7f6;
--fig:#ffffff;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;--sans:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0e151d;--surface:#16202b;--ink:#e6ecf2;--muted:#9aa8b8;--line:#263341;--accent:#5eb0e5;--accent-ink:#0e151d;
--ok:#6fcf9a;--ok-bg:#16392a;--warn:#f0c05a;--warn-bg:#3b2f10;--run:#7cc0ee;--run-bg:#173048;--open:#f19a8c;--open-bg:#43201b;--plan:#b2bdca;--plan-bg:#222d3a;--carry:#b9a9f0;--carry-bg:#2c2646}}
:root[data-theme="dark"]{--bg:#0e151d;--surface:#16202b;--ink:#e6ecf2;--muted:#9aa8b8;--line:#263341;--accent:#5eb0e5;--accent-ink:#0e151d;
--ok:#6fcf9a;--ok-bg:#16392a;--warn:#f0c05a;--warn-bg:#3b2f10;--run:#7cc0ee;--run-bg:#173048;--open:#f19a8c;--open-bg:#43201b;--plan:#b2bdca;--plan-bg:#222d3a;--carry:#b9a9f0;--carry-bg:#2c2646}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 var(--sans)}
a{color:var(--accent)}
.wrap{max-width:1080px;margin:0 auto;padding:28px 16px 80px}
header{display:flex;flex-direction:column;gap:8px;padding-bottom:18px;border-bottom:1px solid var(--line)}
h1{margin:0;font-size:28px;line-height:1.15;letter-spacing:-.01em;text-wrap:balance}
.sub{margin:0;color:var(--muted);max-width:68ch}
.stamp{font:12px var(--mono);color:var(--muted)}
.summary{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 6px}
button.filter{font:inherit;font-size:13px;border:1px solid var(--line);background:var(--surface);color:var(--ink);padding:5px 11px;border-radius:999px;cursor:pointer}
button.filter[aria-pressed="true"]{background:var(--ink);color:var(--bg);border-color:var(--ink)}
button.filter:focus-visible,summary:focus-visible,a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.legend{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:4px 18px;margin:6px 0 22px;font-size:12.5px;color:var(--muted)}
.chip{display:inline-block;font:600 11px/1 var(--sans);letter-spacing:.04em;text-transform:uppercase;padding:4px 8px;border-radius:5px;white-space:nowrap}
.s-supported{background:var(--ok-bg);color:var(--ok)}.s-provisional{background:var(--warn-bg);color:var(--warn)}.s-running{background:var(--run-bg);color:var(--run)}
.s-open{background:var(--open-bg);color:var(--open)}.s-planned{background:var(--plan-bg);color:var(--plan)}.s-carried{background:var(--carry-bg);color:var(--carry)}
.flow{display:flex;flex-wrap:wrap;gap:0;margin:0 0 30px;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:var(--surface)}
.flow a{flex:1 1 150px;display:flex;flex-direction:column;gap:6px;padding:12px 14px;text-decoration:none;color:var(--ink);border-right:1px solid var(--line);border-bottom:1px solid var(--line);margin:0 -1px -1px 0}
.flow a:hover{background:var(--bg)}
.flow b{font-size:13px}.flow span.t{font-size:12px;color:var(--muted)}
h2.grp{margin:34px 0 12px;font-size:12px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);font-weight:700}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;margin:0 0 18px;overflow:hidden}
.card>.hd{display:flex;gap:12px;align-items:flex-start;justify-content:space-between;padding:16px 18px 6px}
.card h3{margin:0;font-size:17px;line-height:1.3;text-wrap:balance}
.choice{margin:0;padding:0 18px 12px;color:var(--ink)}
.choice b{color:var(--muted);font-weight:600;font-size:12px;letter-spacing:.05em;text-transform:uppercase;display:block;margin-bottom:2px}
.figure{margin:0;padding:10px 14px;background:var(--fig);border-top:1px solid var(--line);border-bottom:1px solid var(--line);overflow-x:auto}
.figure img{display:block;width:100%;height:auto;max-width:1040px;margin:0 auto;min-width:560px}
.figure img.narrow{max-width:640px}
.body{padding:12px 18px 16px;display:grid;gap:10px}
.body ul{margin:0;padding-left:18px;display:grid;gap:6px}
.body li{max-width:80ch}
.meta{font-size:13.5px;color:var(--muted);max-width:80ch}
.meta b{color:var(--ink)}
.num{font-variant-numeric:tabular-nums}
.log{margin:34px 0 0;border:1px solid var(--line);border-radius:10px;background:var(--surface);padding:14px 18px}
.log h2{margin:0 0 8px;font-size:15px}.log ol{margin:0;padding-left:18px;display:grid;gap:6px;font-size:13.5px}
.log time{font:12px var(--mono);color:var(--muted)}
.tblwrap{overflow-x:auto;background:var(--surface);border:1px solid var(--line);border-radius:10px;margin:0 0 26px}
.choices table{width:100%;border-collapse:collapse;font-size:13px;min-width:900px}
.choices th{text-align:left;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);padding:10px 12px;border-bottom:1px solid var(--line)}
.choices td{vertical-align:top;padding:10px 12px;border-bottom:1px solid var(--line);max-width:46ch}
.choices tr:last-child td{border-bottom:0}.choices tr.gr td{background:var(--bg);font-weight:700;font-size:12px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);white-space:normal}.goal{max-width:78ch;margin:0 0 10px}.choices td:first-child{font-weight:600;min-width:210px}
footer{margin-top:26px;font-size:12.5px;color:var(--muted)}
@media (max-width:640px){h1{font-size:23px}.card>.hd{flex-direction:column-reverse;gap:8px}}
@media (prefers-reduced-motion:no-preference){.card{transition:border-color .15s}.card:hover{border-color:var(--accent)}}
"""

JS = """
const btns=[...document.querySelectorAll('button.filter')];
function apply(k){document.querySelectorAll('.card').forEach(c=>{c.hidden=!(k==='all'||c.dataset.status===k)});
document.querySelectorAll('h2.grp').forEach(h=>{const s=[...document.querySelectorAll('.card[data-group="'+h.dataset.group+'"]')];h.hidden=s.every(c=>c.hidden)});
btns.forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.k===k)))}
btns.forEach(b=>b.addEventListener('click',()=>apply(b.dataset.k)));
"""


def _o22_overrides(d: dict):
    """Rows of the assumptions table and the choices table that the Oct-22/23 program changes: ({assumption text: fields}, {aspect: fields})."""
    o = d.get("oct22", {})
    ov, oc = {}, {}
    R = o.get("runs", {})
    if R:
        short = {
            l: r["curve"][str(r["n_runs"])]["mean"][0] - r["curve"]["3"]["mean"][0]
            for l, r in R.items()
        }
        lo, hi = _rng(list(short.values()))
        k90 = [r["k90"] for r in R.values()]
        ov["Three independent runs per reservoir are enough."] = dict(
            test="Pooled vs single runs (done); ensemble against run count at 10k, 30k and 100k, random and motif at 30k (done).",
            status="open",
            ev=f"Not quite: three runs sit {lo:.3f} to {hi:.3f} below the ensemble of all runs (10 to 15 runs). 90% of the gain from one run to all of them is reached at {min(k90)} to {max(k90)} runs.",
            card="runcount",
        )
        oc["Independent runs per reservoir"] = dict(
            choice=f"About {min(k90)} to {max(k90)} pooled (top 3 each; the 90% point depends on reservoir and size); three if the budget forces it, at about 0.01 lower ensemble r.",
            ev=f"Run-count curves (10 to 15 runs per size): three runs are {lo:.3f} to {hi:.3f} below all runs; 90% of the gain by {min(k90)} to {max(k90)} runs. Earlier: pooling three beats the best single run in 8 of 9 cells.",
        )
    A = o.get("archsrc", {}).get("results", {})
    AD = o.get("archsrc", {}).get("diffs", {})
    if A:
        sp = []
        for t in ("genomic", "random", "motif", "grm"):
            v = [
                A[f"{c}|{t}"]["enet"]["genomic"]
                for c in ("MG", "MR", "MGR", "MF", "MX")
                if f"{c}|{t}" in A
            ]
            if v:
                sp.append(max(v) - min(v))
        ok = max(sp) <= 0.006
        n_src = len({k.split("|")[0] for k in A})
        ov["Architectures searched on one reservoir are not biased toward that reservoir."] = dict(
            test=f"{n_src} architecture sources (15 runs each: genomic only, random only, genomic+random, the 5-reservoir mix; top 3 per run), retrained on genomic, random and motif pools (done).",
            status="supported" if ok else "provisional",
            ev=f"On the genomic test set the source changes the ensemble by at most {max(sp):.3f} (spread across sources, per target pool; noise margin 0.005)."
            + ("" if ok else " Above the margin on at least one pool."),
            card="archsource",
        )
    U = o.get("unseen", {}).get("results", {})
    DOD = o.get("unseen", {}).get("dod", {})
    if U:
        parts, bad = [], False
        for D in (10000, 30000, 100000):
            eq = [
                v["menu_equal_count"]["genomic"] - v["bespoke_all_nnls"]["genomic"]
                for k, v in U.items()
                if k.endswith(f"|{D}")
            ]
            if eq:
                lo2, hi2 = _rng(eq)
                parts.append(f"{D // 1000}k {lo2:+.3f} to {hi2:+.3f}")
                bad = bad or lo2 < -0.005
        dd = [v["point"]["genomic"] for k, v in DOD.items() if k.endswith("|equal_count")]
        dlo, dhi = _rng(dd)
        ev = (
            "Menu minus own three-run search at equal model counts (genomic test): "
            + "; ".join(parts)
            + f". Difference of differences {dlo:+.3f} to {dhi:+.3f}."
        )
        ov[
            "A search on genomic data alone finds the architectures every other reservoir needs."
        ] = dict(
            test="Own three-run search vs the retrained menu on 8 strategies at 30k, 6 at 10k and 3 at 100k, equal model counts (done).",
            status="provisional" if (bad or dlo < -0.005) else "supported",
            ev=ev,
            card="unseen",
        )
        ov["One representative per family stands for its family."] = dict(
            test="Six strategies the search never saw (two motif sets, Zoonomia orthologs, mutagenesis, GC-matched, dinucleotide shuffle) plus random and motif, at 10k, 30k and (three of them) 100k (done); Zoonomia rebuild, D3 and in-silico evolution not run.",
            status="provisional",
            ev=ev,
            card="unseen",
        )
    MX = o.get("mix", {})
    if MX:
        rows, ctl = MX["rows"], MX["controls"]
        diffs = [
            max(r["enet"][e] for r in rows.values()) - max(v[e] for v in ctl.values())
            for e in ("genomic", "ood", "random_32k")
        ]
        ov["Mixing reservoirs, in the ensemble or the corpus, beats the best single reservoir."] = (
            dict(
                test="Count-controlled model-level mixing (done); 21 mixed corpora at 30k with single-reservoir controls (done).",
                status="provisional",
                ev=f"Best mixture minus best single reservoir at 30k: {diffs[0]:+.3f} genomic test, {diffs[1]:+.3f} designed, {diffs[2]:+.3f} random (noise margin 0.005); an ensemble over several mixtures gains mainly from having more models.",
                card="mixsweep",
            )
        )
        oc["Reservoirs feeding the final ensemble"] = dict(
            ev=f"Genomic plus motif captured most of the model-level gain; the 21-corpus sweep found no mixture above the best single reservoir by more than {max(diffs):+.3f} on any set (best mixture minus best single: genomic {diffs[0]:+.3f}, designed {diffs[1]:+.3f}, random {diffs[2]:+.3f})."
        )
    X = o.get("extrap")
    if X and "genomic" in X:
        g = X["genomic"]["genomic"]
        ov.setdefault("Conclusions at 10k to 300k carry to the 3.2M library.", {}).update(
            a="Conclusions at 10k to 300k carry to the 4M-sequence libraries.",
            ev=f"Power-law fits of the menu ensembles predict genomic-test r of {g['pred']['1000000'][0]:.3f} at 1M and {g['pred']['4000000'][0]:.3f} at 4M for the genomic reservoir (intervals carry only measurement noise); designed-set r stays far lower. The local exponent is falling, so the fits may be optimistic.",
            card="extrap",
        )
    L = o.get("legref_summary")
    if L:
        ov[
            "A free-form architecture search adds diversity to the pool, not a better single architecture."
        ] = dict(
            test="Free-form vs tuned LegNet in the same loop (genomic 30k); the tuned LegNet as an entry of the retrained menu on every pool (done).",
            status="supported",
            ev="Across "
            + str(sum(v["n_pools"] for v in L.values()))
            + " retrained pools the tuned LegNet is at the median of the searched entries from 30k up (mean rank "
            + ", ".join(f"{v['mean_rank']:.0f} of {v['mean_n']:.0f} at {k}k" for k, v in L.items())
            + "); the best entry's lead is what the maximum of 32 noisy entries gives.",
            card="legref",
        )
    ov["Oracle labels are a faithful stand-in for the experiment."] = dict(
        test="Earlier out-of-fold oracle-vs-experiment analysis; per-fold test Pearson of the current oracle (verified Oct 22).",
        status="carried",
        ev="Current oracle (oracle_v3, 10 chromosome-held-out folds): test Pearson 0.931 on average (SD 0.005). An earlier out-of-fold analysis of the previous oracle gave 0.95 on genomic sequences, 0.84 on the designed set and 0.39 on SNV effects; those three are not recomputed for the current oracle.",
    )
    ov.setdefault("Conclusions at 10k to 300k carry to the 3.2M library.", {})["a"] = (
        "Conclusions at 10k to 300k carry to the 4M-sequence libraries."
    )
    return ov, oc


def build_choices(d: dict) -> list[dict]:
    I, T = d.get("instances", {}), d.get("topk", {})
    c100 = d.get("ctx100", {})
    ind = d.get("indep30", {})
    f100 = d.get("family100", {}).get("ens", {})
    fe = d.get("family_ens", {})
    un = d.get("family_union", {})
    sat = d.get("saturation100", {})
    tk100 = d.get("topk100", {})
    pr = d.get("proposer", {}).get("summary", {})
    pa = {
        k.split()[0]: v.get("ens_at_n", [])
        for k, v in d.get("prompt_arms", {}).get("summary", {}).items()
    }
    m = lambda v: sum(v) / len(v) if v else float("nan")  # noqa: E731
    gain = [v["B"]["pooled"]["genomic"] - v["A"]["pooled"]["genomic"] for v in ind.values()]
    rows = [
        dict(
            aspect="Lower D informing higher D",
            choice="No hand-off. Retrain the whole candidate pool at every D and start each D's search from the same prompt.",
            ev=(
                "Context from 30k or a trajectory summary changes 100k ensembles by at most +0.002 (the useless 10k context scored highest, +0.0035). Picking the top 5 at 10k and retraining costs 0.001-0.006 against retraining everything. Architecture rankings agree across sizes only weakly."
            ),
            status="supported",
            card="ctx",
        ),
        dict(
            aspect="Rounds per D",
            choice="About 30 rounds x 2 models at 10k-30k, about 10 rounds x 2 at 100k and above.",
            ev=(
                f"One instance gains {sat['mean30'][-1] - sat['mean30'][0]:.4f} from 6 to 30 models at 30k and {sat['mean100'][-1] - sat['mean100'][0]:.4f} at 100k; the last 12 models add at or below noise at 100k. 300k not measured."
                if sat.get("mean100")
                else "pending"
            ),
            status="provisional",
            card="sched",
        ),
        dict(
            aspect="Proposals per round",
            choice="2 per round.",
            ev="Cadence study (one run per cell): 2 x 30 beat 6 x 10 and 1 x 60 on genomic and motif, tied on random. Per-round proposal quality is flat, so the edge is the LLM prior, not feedback.",
            status="provisional",
            card="llm",
        ),
        dict(
            aspect="Independent runs per reservoir",
            choice="Three, pooled.",
            ev="Pooling three runs beats the best single run in 8 of 9 cells and the average run in 9 of 9; between-run SD about 0.005. More than three has not been tested.",
            status="provisional",
            card="rep",
        ),
        dict(
            aspect="Independent pool per run",
            choice="Yes: each run gets its own labelled pool, nested across sizes inside it.",
            ev=(
                "At equal model counts, pooling three independent pools beats pooling three seeds on one pool by "
                + ", ".join(f"{g:+.3f}" for g in gain)
                + " (genomic, random, motif, evoaug), 12 of 12 comparisons positive across evaluation sets."
                if ind
                else "pending"
            ),
            status="supported",
            card="indep",
        ),
        dict(
            aspect="Top models kept per run",
            choice="Top 5 by validation from each run.",
            ev=(
                "At 30k the top 8 of each of three runs equals the whole pool. At 100k, k = 3 to 5 is within 0.001 of keeping all 20."
                if tk100.get("genomic")
                else "At 30k the top 8 equals the whole pool."
            ),
            status="supported",
            card="inst",
        ),
        dict(
            aspect="Reservoir families to search",
            choice="Search on genomic only, then retrain the pool on each reservoir's data.",
            ev=(
                "Fresh per-reservoir searches match the retrained pool within 0.006 at 30k and within 0.004 at 100k ("
                + "; ".join(
                    f"{k.replace('_', ' ')} {v['ar_top8'] - v['menu']:+.3f}"
                    for k, v in f100.items()
                    if v.get("menu") is not None
                )
                + " at 100k). Tested on one representative per family only."
                if f100
                else "pending"
            ),
            status="supported",
            card="fam",
        ),
        dict(
            aspect="Reservoirs feeding the final ensemble",
            choice="Several; genomic plus motif captures most of the gain.",
            ev=(
                "Ensemble on genomic held-out sequences: genomic only "
                + f"{un['genomic only']['genomic']:.3f}, genomic + motif {un['genomic + motif']['genomic']:.3f}, all five {un['all five']['genomic']:.3f}. Out of distribution the best single reservoir beats the union. Ensembles of single-reservoir models, not a mixed corpus."
                if un
                else "pending"
            ),
            status="provisional",
            card="mix",
        ),
        dict(
            aspect="Proposer model",
            choice="Sonnet 5.5.",
            ev="Equal-n ensemble: "
            + ", ".join(
                f"{k} {m(v.get('ens_at_n', [])):.4f}" for k, v in pr.items() if v.get("ens_at_n")
            )
            + ". Opus is ahead by under two standard errors.",
            status="supported",
            card="proposer",
        ),
        dict(
            aspect="Prompt",
            choice="Guidance (known-good pieces, ensemble note), no phase schedule.",
            ev=f"At 50 models: guidance + phases {m(pa.get('P', [])):.3f}, guidance only {m(pa.get('W', [])):.3f}, nothing {m(pa.get('N', [])):.3f}.",
            status="supported",
            card="prompt",
        ),
        dict(
            aspect="Ensemble method",
            choice="Non-negative ElasticNet on validation predictions of all models, with a screen for exploded predictions on every evaluation set.",
            ev="Beat greedy top-K by about +0.01 earlier; 0.8% of free-form models explode on a shifted evaluation set.",
            status="carried",
            card="ens",
        ),
        dict(
            aspect="Training and evaluation loop",
            choice="One common loop with reverse-complement averaging for every model.",
            ev="Same network scores 0.69-0.70 in the old loop and 0.73 in the common loop; reverse-complement averaging is +0.014 of that.",
            status="supported",
            card="loop",
        ),
        dict(
            aspect="Pools",
            choice="Nested canonical pools and independent nested replicates; legacy per-size pools only to reproduce old numbers.",
            ev=(
                "A nested genomic ladder (one seed) lands within about 0.01 of the old ladder at every size, so the earlier scaling numbers stand."
                if d.get("nested_ladder", {}).get("ens")
                else "nested ladder pending"
            ),
            status="supported",
            card="remeasure",
        ),
    ]
    oc = _o22_overrides(d)[1]
    for r_ in rows:
        r_.update(oc.get(r_["aspect"], {}))
    return rows


def build_assumptions(d: dict) -> list[dict]:
    """The assumptions behind the simplest procedure, each with why it matters, the experiment that tests it, its status and the result so far."""
    ch = {r["aspect"]: r for r in build_choices(d)}
    al = d.get("arch_vs_legnet", {})
    ev = lambda k: ch[k]["ev"]  # noqa: E731
    g = lambda grp, label: d.get("program_status", {}).get("groups", {}).get(label, {})  # noqa: E731
    prog = lambda label: (
        lambda x: (
            f"{x.get('done', 0)} of {x.get('total', 0)} done, {x.get('running', 0)} running"
            if x
            else "not started"
        )
    )(g(None, label))
    rows = [
        # --- model side
        dict(
            grp="A. Marginalizing over architecture and training choices",
            a="All models, whatever their source, are trained and scored in one common loop.",
            why="Otherwise loop differences masquerade as architecture or data effects.",
            test="Same network in both loops; flip each loop ingredient.",
            status="supported",
            ev=ev("Training and evaluation loop"),
            card="loop",
        ),
        dict(
            grp="A. Marginalizing over architecture and training choices",
            a="A free-form architecture search adds diversity to the pool, not a better single architecture.",
            why="Decides how much search is worth buying.",
            test="Free-form vs tuned LegNet in the same loop; menu retrains.",
            status="supported",
            ev=(
                "Tuned LegNet 0.730-0.732, top free-form 0.724-0.732 (genomic 30k, single models)."
                if al
                else "see card"
            ),
            card="arch",
        ),
        dict(
            grp="A. Marginalizing over architecture and training choices",
            a="The proposer model and prompt do not change the conclusions.",
            why="The search is the source of the architecture sample.",
            test="Three prompt arms; three proposer models, three runs each.",
            status="supported",
            ev=ev("Prompt") + " " + ev("Proposer model"),
            card="prompt",
        ),
        dict(
            grp="A. Marginalizing over architecture and training choices",
            a="An ensemble of searched models stands in for the best achievable model, so scaling results are not tied to one architecture.",
            why="The scaling claims must hold across architectures.",
            test="Non-negative ElasticNet vs greedy; pooled-run ensembles; exploded-model screen.",
            status="carried",
            ev=ev("Ensemble method"),
            card="ens",
        ),
        dict(
            grp="A. Marginalizing over architecture and training choices",
            a="Three independent runs per reservoir are enough.",
            why="Number of runs sets compute and what noise remains.",
            test="Pooled vs single runs (done); ensemble against run count at 10k, 30k, 100k (running).",
            status="provisional",
            ev=ev("Independent runs per reservoir")
            + " Run-count curves: "
            + prog("genomic 30k search runs")
            + " (genomic 30k).",
            card="rep",
        ),
        dict(
            grp="A. Marginalizing over architecture and training choices",
            a="Each independent run should have its own labelled pool.",
            why="Shared pools understate variance and give up data diversity.",
            test="Menu on three seeds of one pool vs three independent pools, equal model counts.",
            status="supported",
            ev=ev("Independent pool per run"),
            card="indep",
        ),
        dict(
            grp="A. Marginalizing over architecture and training choices",
            a="About 20 to 60 models per run, 2 proposals per round, and the top 3 to 5 per run are enough.",
            why="These set the LLM budget at every dataset size.",
            test="Saturation curves at 30k and 100k; top-k curves; cadence study.",
            status="provisional",
            ev=ev("Rounds per D")
            + " "
            + ev("Top models kept per run")
            + " "
            + ev("Proposals per round"),
            card="sched",
        ),
        # --- ladder
        dict(
            grp="B. The dataset-size ladder",
            a="Nothing learned at a smaller D has to be handed to a larger D.",
            why="Lets every D be run independently and cheaply.",
            test="Context hand-off arms at 100k; top-5-from-10k retrains.",
            status="supported",
            ev=ev("Lower D informing higher D"),
            card="ctx",
        ),
        dict(
            grp="B. The dataset-size ladder",
            a="Retraining the whole candidate pool at each D is the robust choice, because architecture rankings are noise-limited.",
            why="Decides whether architectures chosen at one D can be reused.",
            test="Rank agreement across D with 3 seeds (done); nested 3-seed ladder (planned).",
            status="provisional",
            ev="Agreement across sizes is low to moderate and noise-limited; the nested one-seed ladder did not repeat the old three-seed pattern.",
            card="menu",
        ),
        dict(
            grp="B. The dataset-size ladder",
            a="Nested pools give an undistorted scaling curve.",
            why="Otherwise D is confounded with which sequences were drawn.",
            test="Audit of legacy pools; nested genomic ladder vs old ladder.",
            status="supported",
            ev=ev("Pools"),
            card="remeasure",
        ),
        dict(
            grp="B. The dataset-size ladder",
            a="Conclusions at 10k to 300k carry to the 3.2M library.",
            why="The recommendation is for a library 10 times larger than the largest tested size.",
            test="Menu at 300k (done); a search at 300k and a fitted extrapolation (not run).",
            status="open",
            ev="Menu ensembles still rise at 300k (genomic 0.863, gain per doubling 0.023 and falling); out-of-distribution accuracy is still accelerating.",
            card="menu300",
        ),
        # --- reservoir generality
        dict(
            grp="C. Generality across reservoirs",
            a="A search on genomic data alone finds the architectures every other reservoir needs.",
            why="Would let one search serve all reservoirs and sizes.",
            test="Bespoke search vs retrained menu on four reservoirs at 30k and 100k; six unseen reservoirs (running).",
            status="supported",
            ev=ev("Reservoir families to search")
            + " Unseen reservoirs: "
            + prog("unseen-reservoir search runs")
            + ".",
            card="fam",
        ),
        dict(
            grp="C. Generality across reservoirs",
            a="Architectures searched on one reservoir are not biased toward that reservoir.",
            why="A genomic-only search could favour genomic-like data.",
            test="15 genomic-only, 15 random-only, a mixed and the 5-reservoir search arms, all retrained on genomic, random and motif data (running).",
            status="running",
            ev="Search runs: genomic "
            + prog("genomic 30k search runs")
            + "; random "
            + prog("random 30k search runs")
            + ". Retrains start when the runs finish.",
            card="program",
        ),
        dict(
            grp="C. Generality across reservoirs",
            a="One representative per family stands for its family.",
            why="Four reservoirs were tested; the families are larger.",
            test="Six unseen reservoirs including motif variants, zoonomia, mutagenesis, gc_matched, dinuc_shuffle (running).",
            status="running",
            ev="Unseen-reservoir menu retrains: " + prog("unseen-reservoir menu retrains") + ".",
            card="program",
        ),
        dict(
            grp="C. Generality across reservoirs",
            a="Mixing reservoirs, in the ensemble or the corpus, beats the best single reservoir.",
            why="Decides whether the library should mix sources.",
            test="Count-controlled model-level mixing (done); 21 mixed corpora (running).",
            status="provisional",
            ev=ev("Reservoirs feeding the final ensemble")
            + " Mixed corpora: "
            + prog("mixture-corpus retrains")
            + ".",
            card="mix",
        ),
        # --- measurement
        dict(
            grp="D. Measurement validity",
            a="Differences under about 0.005 in ensemble r are noise, so 'no improvement' means within that margin.",
            why="Defines what counts as proving a more complicated approach does not help.",
            test="Replicate spreads and retest ceilings.",
            status="supported",
            ev="Between-run SD of a pooled ensemble is about 0.002 to 0.005; a single run's SD is about 0.009.",
            card="rep",
        ),
        dict(
            grp="D. Measurement validity",
            a="Oracle labels are a faithful stand-in for the experiment.",
            why="Every number here is against oracle labels.",
            test="Earlier oracle-vs-experiment ceilings; no new test.",
            status="carried",
            ev="Established in earlier work (oracle ceiling on held-out genomic sequences about 0.95); not retested in this program.",
            card="ens",
        ),
        dict(
            grp="D. Measurement validity",
            a="Models that explode on shifted evaluation sets are screened out before ensembling.",
            why="One such model can ruin an ensemble.",
            test="Audit of 2,040 saved models.",
            status="supported",
            ev="16 of 2,040 (0.8%) exploded on some evaluation set; now screened in the generator and in new runs.",
            card="ens",
        ),
    ]
    from ledger_oct26 import extra_assumption_rows

    rows += extra_assumption_rows(d)
    ov = _o22_overrides(d)[0]
    for r_ in rows:
        r_.update(ov.get(r_["a"], {}))
    return rows


def render(cards, d):
    esc = html.escape
    counts = {k: sum(1 for c in cards if c["status"] == k) for k in STATUS}
    out = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>ALBench Evidence Ledger</title>",
        f'<style>{CSS}</style></head><body><div class="wrap">',
    ]
    out.append(
        "<header><h1>ALBench Evidence Ledger</h1>"
        '<p class="sub">Every choice in the training-corpus pipeline, the experiment that justifies it, and how much weight it can bear. '
        "Figures regenerate from the experiment outputs; running experiments fill in as they finish.</p>"
        f'<div class="stamp">data as of {esc(d.get("generated", "?"))} EDT &middot; {len(cards)} decisions</div></header>'
    )
    out.append(
        '<div class="summary"><button class="filter" data-k="all" aria-pressed="true">All '
        + str(len(cards))
        + "</button>"
        + "".join(
            f'<button class="filter" data-k="{k}" aria-pressed="false">{STATUS[k][0]} {n}</button>'
            for k, n in counts.items()
            if n
        )
        + "</div>"
    )
    out.append(
        '<div class="legend">'
        + "".join(
            f'<div><span class="chip s-{k}">{STATUS[k][0]}</span> {esc(STATUS[k][1])}</div>'
            for k in STATUS
        )
        + "</div>"
    )
    from ledger_oct26 import coverage_html, spec_rows

    spec = spec_rows(d)
    from ledger_oct27 import robustness_html, spec_patch

    spec_patch(spec, d)
    sp = [
        '<section class="choices" id="spec"><h2 class="grp">The procedure, parameter by parameter</h2>'
        '<p class="goal">Every setting of the HP-optimisation and ensembling pipeline in the order it is applied, what the recipe currently uses, the experiment behind it, and what is still open. '
        "The status chip is a judgement; the link goes to the evidence card.</p>"
        '<div class="tblwrap"><table><thead><tr><th>Parameter</th><th>Setting in the recipe</th><th>Evidence</th><th>Status</th><th>Still open</th></tr></thead><tbody>'
    ]
    last_ = None
    for r in spec:
        if r["stage"] != last_:
            sp.append(f'<tr class="gr"><td colspan="5">{esc(r["stage"])}</td></tr>')
            last_ = r["stage"]
        link = f'<a href="#{r["card"]}">{esc(r["p"])}</a>'
        sp.append(
            f'<tr><td>{link}</td><td>{esc(r["s"])}</td><td>{esc(r["ev"])}</td><td><span class="chip s-{r["st"]}">{STATUS[r["st"]][0]}</span></td><td>{esc(r["open"])}</td></tr>'
        )
    sp.append("</tbody></table></div>")
    sp.append(
        '<h3 id="coverage" class="sub3">Which reservoirs at which D</h3>'
        + coverage_html(d)
        + robustness_html(d)
        + "</section>"
    )
    out.append("".join(sp))
    rows = build_assumptions(d)
    parts = [
        '<section class="choices" id="assumptions"><h2 class="grp">Assumptions to justify</h2>'
        '<p class="goal"><b>Goal.</b> Take the simplest approach that approximates ideal performance across reservoirs and dataset sizes, while marginalizing over factors such as architecture so the scaling results are general. '
        "A more complicated approach has to beat it by more than the noise (about 0.005 in ensemble r) to earn its cost.</p>"
        '<p class="goal"><b>Simplest recipe so far.</b> One search at 30k on genomic data (several independent runs, each on its own pool: three runs fall 0.005 to 0.012 short of the ensemble of 10 to 15 runs, and 6 to 11 reach 90% of that gain; 20 models per run with the default prompt, the top 3 of each kept), the pooled 33-architecture menu retrained at every size and reservoir (about 8 architectures from 300k up, three seeds at 100k and below), combined by a non-negative ElasticNet ensemble fitted on validation predictions (the oracle-relabelled validation set, V0c). '
        "Nothing more elaborate has beaten it by more than noise: per-reservoir and mixed searches, context hand-off, explore and reflect prompts, adaptive acquisition rounds and a full selection ensemble all tie. The coordinator stage pools to about +0.003 across the grid with no detectable heterogeneity, still below the 0.005 a change must clear.</p>"
        '<div class="tblwrap"><table><thead><tr><th>Assumption</th><th>Why it matters</th><th>How it is tested</th><th>Status</th><th>Result so far</th></tr></thead><tbody>'
    ]
    last = None
    for r in rows:
        if r["grp"] != last:
            parts.append(f'<tr class="gr"><td colspan="5">{esc(r["grp"])}</td></tr>')
            last = r["grp"]
        parts.append(
            f'<tr><td><a href="#{r["card"]}">{esc(r["a"])}</a></td><td>{esc(r["why"])}</td><td>{esc(r["test"])}</td><td><span class="chip s-{r["status"]}">{STATUS[r["status"]][0]}</span></td><td>{esc(r["ev"])}</td></tr>'
        )
    parts.append("</tbody></table></div></section>")
    out.append("".join(parts))
    out.append(
        '<nav class="flow" aria-label="Pipeline">'
        + "".join(
            f'<a href="#{c["id"]}"><span class="chip s-{c["status"]}">{STATUS[c["status"]][0]}</span><b>{esc(t)}</b><span class="t">{esc(s)}</span></a>'
            for t, s, cid in [
                ("D ladder", "10k, 30k, 100k, 300k", "menu"),
                ("Proposer", "Sonnet 5.5, 2 x 30 rounds", "proposer"),
                ("5 reservoir families", "independent instances each", "fam"),
                ("Top by validation", "8 per instance, pooled", "inst"),
                ("Retrain at every D", "same pool, each dataset", "menu"),
                ("Ensemble", "enet_all_pos", "ens"),
            ]
            for c in [next(x for x in cards if x["id"] == cid)]
        )
        + "</nav>"
    )
    seen = []
    for c in cards:
        if c["group"] not in seen:
            seen.append(c["group"])
    ORDER = [
        "Experiments launched Oct 22",
        "Results from the Oct 22 program",
        "Results from the Oct 25-26 program",
        "Results from the Oct 27 program",
        "Training and evaluation",
        "Proposer and prompt",
        "Search and pooling",
        "Dataset-size ladder",
        "Compute and limits",
    ]
    seen = [g_ for g_ in ORDER if g_ in seen] + [g_ for g_ in seen if g_ not in ORDER]
    for g in seen:
        out.append(f'<h2 class="grp" data-group="{esc(g)}">{esc(g)}</h2>')
        for c in [x for x in cards if x["group"] == g]:
            st = c["status"]
            out.append(
                f'<section class="card" id="{c["id"]}" data-status="{st}" data-group="{esc(g)}"><div class="hd"><h3>{esc(c["q"])}</h3><span class="chip s-{st}">{STATUS[st][0]}</span></div>'
            )
            out.append(f'<p class="choice"><b>Choice</b>{esc(c["choice"])}</p>')
            for f in c["figs"]:
                out.append(
                    f'<figure class="figure"><img{" class=narrow" if f in NARROW else ""} src="figs/{f}" alt="{esc(c["q"])}" loading="lazy"></figure>'
                )
            out.append(
                '<div class="body"><ul>'
                + "".join(f'<li class="num">{esc(p)}</li>' for p in c["points"])
                + "</ul>"
            )
            out.append(
                f'<div class="meta"><b>Caveat.</b> {esc(c["caveat"]).replace(esc(OLD_ARTIFACT), f"<a href={chr(34)}{OLD_ARTIFACT}{chr(34)}>previous artifact</a>")}</div>'
            )
            if c.get("change"):
                out.append(
                    f'<div class="meta"><b>What would change the decision.</b> {esc(c["change"])}</div>'
                )
            if c.get("see"):
                out.append(
                    f'<div class="meta"><b>Figure.</b> <a href="figs/{c["see"]}">{esc(c["see"])}</a></div>'
                )
            out.append("</div></section>")
    out.append(
        '<div class="log"><h2>Update log</h2><ol>'
        + "".join(f"<li><time>{esc(t)}</time> {esc(m)}</li>" for t, m in reversed(CHANGELOG))
        + "</ol></div>"
    )
    out.append(
        f'<footer>Generated by scripts/analysis/make_justification.py and build_justification_html.py. Earlier HP-search decisions: <a href="{OLD_ARTIFACT}">ALBench Design Decisions</a>.</footer>'
    )
    out.append(f"</div><script>{JS}</script></body></html>")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="outputs/justification")
    a = ap.parse_args()
    src = Path(a.src)
    d = json.load(open(src / "data.json"))
    if (src / "program_status.json").exists():
        d["program_status"] = json.load(open(src / "program_status.json"))
    o22 = src.parent / "justification_oct22"
    if (o22 / "data.json").exists():
        d["oct22"] = json.load(open(o22 / "data.json"))
        import shutil

        for f in (o22 / "figs").glob("*.png"):
            shutil.copy(f, src / "figs" / f.name)
    cards = build(d)
    for c_ in cards:
        if c_["id"] in ("program2", "program3", "program4"):
            c_["status"] = "carried"
            c_["choice"] = (
                "Launch record: these cells have finished and their results are in the cards below. "
                + c_["choice"]
            )
    if d.get("oct22", {}).get("live"):
        d["generated"] = "2026-" + d["oct22"]["live"]["stamp"]
    (src / "index.html").write_text(render(cards, d))
    print("wrote", src / "index.html", len(cards), "cards")


if __name__ == "__main__":
    main()
