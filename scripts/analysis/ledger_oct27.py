# ruff: noqa
"""Oct 27 additions to the evidence ledger: core reservoir set, loop elements, coordinator consistency, extrapolation metrics, training-recipe sensitivity,
validation labels and HepG2 across sizes, reduced menus and ensemble methods across reservoirs, 600k retune, and the robustness-across-covariates table.

Imported by build_justification_html.py through patch_builder7.py. Everything is computed from d['oct22'] (outputs/justification_oct22/data.json); a card whose data
are missing is simply not built, so the page can be rebuilt at any time.
"""

from __future__ import annotations

G27 = "Results from the Oct 27 program"
EV3 = ("genomic", "ood", "random_32k")
RESN = {
    "genomic": "genomic",
    "random": "random",
    "motif_ct_enriched": "motif (CT-enriched)",
    "evoaug_paper2023": "evoaug",
    "evo2": "evo2",
}
EVN = {"genomic": "genomic test", "ood": "designed set", "random_32k": "random sequences"}


def _m(v):
    return sum(v) / len(v) if v else float("nan")


def _ci(g):
    return f"{g['mean']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}], {g['wins']} of {g['n']} ahead"


def oct27_cards(d: dict) -> list[dict]:
    o = d.get("oct22", {})
    out = []
    # ------------------------------------------------------------ core reservoir set
    ms = d.get("menu_scaling", {})
    U = o.get("unseen", {}).get("results", {})
    UM = o.get("unseen_menu", {})
    if ms:

        def lv(st, D, e):
            return ms.get(st, {}).get(str(D // 1000), {}).get(e)

        pts = [
            "Ensemble r of the retrained 33-architecture menu by reservoir (mean of up to three seeds), genomic test / designed set / random sequences. "
            + "; ".join(
                f"{D // 1000}k: "
                + ", ".join(
                    f"{RESN[st]} " + "/".join(f"{lv(st, D, e):.3f}" for e in EV3)
                    for st in ("genomic", "random", "motif_ct_enriched", "evoaug_paper2023", "evo2")
                    if lv(st, D, "genomic") is not None
                )
                for D in (10000, 30000, 100000)
            )
            + "."
        ]
        if o.get("anchor"):
            A = o["anchor"]
            pts.append(
                "Above 100k (nested pool 1, single seed, architectures common to every size; genomic is capped by its pool): "
                + "; ".join(
                    f"{RESN[st]} "
                    + ", ".join(
                        f"{int(k) // 1000}k {v['genomic']:.3f}/{v['ood']:.3f}/{v['random_32k']:.3f}"
                        for k, v in sorted(A[st]["points"].items(), key=lambda kv: int(kv[0]))
                        if int(k) >= 300000
                    )
                    for st in ("random", "motif_ct_enriched")
                    if st in A
                )
                + "."
            )
        fam = []
        for st in (
            "motif_shared_core",
            "motif_syntax_core",
            "mutagenesis",
            "gc_matched",
            "dinuc_shuffle",
            "zoonomia_orthologs",
        ):
            row = []
            for D in (10000, 30000, 100000):
                v = (UM.get(f"{st}|{D}") or U.get(f"{st}|{D}") or {}).get("menu")
                if v:
                    row.append(f"{D // 1000}k " + "/".join(f"{v[e]:.3f}" for e in EV3))
            if row:
                fam.append(f"{st.replace('_', ' ')} " + ", ".join(row))
        if fam:
            pts.append(
                "Unseen families (menu retrained, mean of three pools): " + "; ".join(fam) + "."
            )

        # random minus motif by size and evaluation set (10k-100k from the menu table, 300k and up from the nested single-seed pool)
        def rm(D, e):
            if (
                D <= 100000
                and lv("random", D, e) is not None
                and lv("motif_ct_enriched", D, e) is not None
            ):
                return lv("random", D, e) - lv("motif_ct_enriched", D, e)
            A_ = o.get("anchor", {})
            if (
                "random" in A_
                and "motif_ct_enriched" in A_
                and str(D) in A_["random"]["points"]
                and str(D) in A_["motif_ct_enriched"]["points"]
            ):
                return (
                    A_["random"]["points"][str(D)][e] - A_["motif_ct_enriched"]["points"][str(D)][e]
                )
            return None

        Ds_ = [
            D
            for D in (10000, 30000, 100000, 300000, 600000, 1000000)
            if rm(D, "genomic") is not None
        ]
        sc = {
            D: (UM.get(f"motif_shared_core|{D}") or {}).get("menu") for D in (10000, 30000, 100000)
        }
        SHARED_TXT = "Motif shared core: retrain not available."
        if all(sc.values()):

            def trio(fn, e):
                return "/".join(f"{fn(D, e):.3f}" for D in (10000, 30000, 100000))

            SHARED_TXT = (
                "Motif shared core against random and motif (CT-enriched), 10k / 30k / 100k. Shared core comes from the unseen-family retrains and the other two from the menu table, so the series use different pool draws and differences under about 0.01 are within their offset (genomic itself reads 0.827 in the menu table and 0.838 in the retrains at 100k). "
                + "; ".join(
                    f"{EVN[e]}: shared core {trio(lambda D, e_: sc[D][e_], e)}, random {trio(lambda D, e_: lv('random', D, e_), e)}, motif CT-enriched {trio(lambda D, e_: lv('motif_ct_enriched', D, e_), e)}"
                    for e in EV3
                )
                + ". It is ahead of both on the designed set at every size, level with motif CT-enriched on the genomic test from 30k up, and behind random on random sequences by 0.00 to 0.03."
            )
        UA = {
            st: UM.get(f"{st}|300000", {}).get("menu")
            for st in ("genomic", "random", "motif_ct_enriched", "evoaug_paper2023")
        }
        EVOAUG300_TXT = "Evoaug at 300k: not available."
        if all(UA.values()):
            EVOAUG300_TXT = (
                "At 300k (nested pool 1, single seed, 33 architectures) evoaug reads "
                + "/".join(f"{UA['evoaug_paper2023'][e]:.3f}" for e in EV3)
                + " against motif CT-enriched "
                + "/".join(f"{UA['motif_ct_enriched'][e]:.3f}" for e in EV3)
                + ", random "
                + "/".join(f"{UA['random'][e]:.3f}" for e in EV3)
                + " and genomic "
                + "/".join(f"{UA['genomic'][e]:.3f}" for e in EV3)
                + f": it is level with motif on the genomic test ({UA['evoaug_paper2023']['genomic'] - UA['motif_ct_enriched']['genomic']:+.3f}) and behind it on the designed set ({UA['evoaug_paper2023']['ood'] - UA['motif_ct_enriched']['ood']:+.3f}) and random sequences ({UA['evoaug_paper2023']['random_32k'] - UA['motif_ct_enriched']['random_32k']:+.3f})."
            )
        pts += [
            "Random minus motif (CT-enriched) by size: "
            + "; ".join(
                f"{EVN[e]} "
                + ", ".join(
                    f"{D // 1000 if D < 1000000 else 1}{'k' if D < 1000000 else 'M'} {rm(D, e):+.3f}"
                    for D in Ds_
                )
                for e in EV3
            )
            + ". Motif leads on the genomic test at every size by about 0.03 to 0.05; random leads on random sequences by 0.01 to 0.04 and on the designed set by about 0.05 at 100k, but that lead shrinks to nothing by 600k, where the two are level. Both are clear of genomic on the designed set (by 0.1 or more at 100k).",
            "At 10k and 30k the two are close on the designed set, so the choice is the genomic test (motif) against random sequences (random). At 100k random has a clear lead on the designed set (+0.05). From 300k up that lead is gone and random keeps only +0.01 to +0.02 on random sequences, so motif (CT-enriched) alone is nearly on the front there, with random as a cheap guard for fully random designs.",
            "Evoaug and evo2 track genomic within about 0.01 on every evaluation set and size (evoaug is 0.005 to 0.015 above genomic on the genomic test); mutagenesis, motif syntax core and dinucleotide shuffle sit in the same cluster; GC-matched lies between genomic and random (at 100k: 0.810 / 0.503 / 0.857); Zoonomia is far below everything (0.70 at 30k). None of these adds a corner that motif (CT-enriched) and random do not already cover.",
            SHARED_TXT,
            EVOAUG300_TXT,
            "Count-controlled mixtures (card on mixing): genomic plus motif captures most of the mixing gain on the genomic test and the best single reservoir beats any union on the designed set, which is the same two-corner picture.",
        ]
        out.append(
            dict(
                id="coreset",
                group=G27,
                status="provisional",
                figs=["coreset.png"],
                q="Which reservoirs belong in the core set at each dataset size?",
                choice="Up to 100k: random, a motif reservoir, and genomic as the only reservoir of real sequences (up to its 315k cap) and the check that oracle-labelled synthetic data behave like real data. Of the two motif reservoirs, shared core is ahead of CT-enriched on the designed set at every size tested (+0.06 / +0.06 / +0.08) and level on the genomic test from 30k up, so it is the stronger candidate where it was measured; it was not built above 100k. From 300k up, motif CT-enriched alone is nearly on the front (evoaug ties it on the genomic test but trails by 0.06 on the designed set) and random shrinks to a guard. Everything else tested (evoaug, evo2, mutagenesis, syntax-core motifs, dinucleotide shuffle, GC-matched, Zoonomia) is covered by this set or worse.",
                points=pts,
                caveat="Levels are menu ensembles on one replicate pool family; the genomic test set rewards genomic-like data, the designed and random sets reward coverage, so the weight on each is a library-design decision, not a result. Shared core is compared with random and CT-enriched across pool series (one pool draw each, three seeds); unseen families were not retrained above 100k; evoaug is single-seed at 300k; 600k and 1M exist for random and motif CT-enriched only.",
                change="A same-series run (shared core, CT-enriched and random from the same replicate pools at 30k and 100k, plus a shared-core pool at 300k) would settle which motif reservoir goes in the core: if shared core stays ahead on the designed set it replaces CT-enriched, otherwise the two are interchangeable.",
            )
        )
    # ------------------------------------------------------------ loop elements
    L = o.get("loopctl")
    if L and L.get("pairs"):
        pr = L["pairs"]
        pts = [
            "All arms use the same six genomic 30k pools, paired with the default single-call loop. Ensemble r on the genomic test set, difference from default (mean, bootstrap interval, runs ahead of six): "
            + "; ".join(
                f"{k.replace(' minus default', '')} {_ci(v['genomic'])}"
                for k, v in pr.items()
                if k.endswith("minus default")
            )
            + ".",
            "Component contrasts: "
            + "; ".join(
                f"{k}: {_ci(v['genomic'])}"
                for k, v in pr.items()
                if not k.endswith("minus default")
            )
            + ".",
        ]
        if L.get("arms"):
            pts.append(
                "Mean validation r of a run's 20 models: "
                + "; ".join(
                    f"{k} {v['mean_val']:.4f}" for k, v in L["arms"].items() if v.get("mean_val")
                )
                + "."
            )

        def gm(k):
            return pr.get(k, {}).get("genomic", {}).get("mean")

        k_sc, k_co, k_cr = (
            "separate coder (no coordinator, no reflector) minus default",
            "6 ideas, 2 picked by the coordinator minus default",
            "coordinator + reflector minus default",
        )
        if all(gm(k) is not None for k in (k_sc, k_co, k_cr)):
            rs = pr[k_sc]["random_32k"]["mean"]
            pts.append(
                f"Reading the contrasts: a separate coder alone gains {gm(k_sc):+.4f} over the single-call loop ({rs:+.4f} on random sequences); the coordinator adds {gm(k_co) - gm(k_sc):+.4f} on top of the separate coder and the reflector {gm(k_cr) - gm(k_co):+.4f} more; the coordinator's picks beat random picks of the same six ideas by {gm('coordinator minus random pick (same over-generation)') or 0:+.4f}, but random picks do not beat coding the first two ideas ({gm('random pick minus separate coder alone') or 0:+.4f}). At genomic 30k the whole gain of the agent loop is therefore the separate coder; coordination and reflection add nothing detectable once it is there, and the pooled coordinator effect in the prompt grid (+0.0027) should be read as an agent-loop effect."
            )
        gd = L.get("guidance", {})
        if gd:
            pts.append(
                "Prompt content, guidance block removed against the default (known-good pieces), paired by pool, genomic test: "
                + "; ".join(f"{k} {_ci(v['genomic'])}" for k, v in gd.items())
                + f". Mean {_m([v['genomic']['mean'] for v in gd.values()]):+.4f}, with no pool ahead in any of the {sum(v['genomic']['n'] for v in gd.values())} pairs: about {-_m([v['genomic']['mean'] for v in gd.values()]) / max(gm(k_sc) or 1e-9, 1e-9):.0f} times the gain of the separate coder."
            )
        out.append(
            dict(
                id="loopels",
                group=G27,
                status="provisional",
                figs=["loop_elements.png"],
                q="Which elements of the search loop earn their place: proposer, coder, evaluator, coordinator, reflector, guidance?",
                choice="Judge each element by its contrast with the loop without it. Keep the guidance block: removing it loses 0.016 to 0.020 on every reservoir-size cell tested, with no pool ahead. Keep the coder as its own agent call (about +0.005 over the single-call loop at genomic 30k, six of six pools ahead). The coordinator and the reflector add nothing measurable on top of it (+0.0003 and +0.0002); the coordinator picks better than random (+0.004), but over-generating and picking is no better than coding the first two ideas, so both are optional. The evaluator is deterministic code (build check, training, validation scoring), not an agent.",
                points=pts,
                caveat="Six paired pools at genomic 30k per arm, so single contrasts have intervals of about +/-0.005; the separate-coder and random-pick contrasts exist at genomic 30k only (the prompt grid shows the coordinator arms at +0.005 on genomic and random 30k and about 0 on motif), so whether the separate-coder gain holds on other reservoirs is the open test; combinations beyond those listed (for example coordinator without over-generation, or a reflector that sees test-side numbers) were not run. The guidance arms use random 30k, motif 30k and genomic 10k pools.",
                change="A contrast whose interval excludes zero by more than 0.003, repeated on a second reservoir, would justify keeping that element; the rest can be dropped for simplicity.",
            )
        )
    # ------------------------------------------------------------ recipe sensitivity
    Rc = o.get("recipe")
    if Rc and any(k != "summary" for k in Rc):
        cells = {k: v for k, v in Rc.items() if k != "summary"}
        pts = [
            "Retrained menu, one seed, same pool, same architectures, only the training recipe changed: patience 4 or 16 epochs instead of 8, or the epoch budget (and so the one-cycle schedule) halved or doubled. Ensemble r on the genomic test set, difference from the default recipe: "
            + "; ".join(
                f"{k.replace('_', ' ')}: "
                + ", ".join(
                    f"{m} {v[m]['r']['genomic'] - v['default']['r']['genomic']:+.4f}"
                    for m in ("pat4", "pat16", "ep05", "ep2")
                    if m in v
                )
                for k, v in cells.items()
            )
            + "."
        ]
        sm = Rc.get("summary", {})
        if sm:
            pts.append(
                "Across cells: "
                + "; ".join(
                    f"{m} mean {x['mean']:+.4f} (range {x['min']:+.4f} to {x['max']:+.4f}, {x['n']} cells)"
                    for m, x in sm.items()
                )
                + "."
            )
        ep = [
            (
                k,
                {
                    m: v[m].get("epochs")
                    for m in ("default", "pat4", "pat16", "ep05", "ep2")
                    if m in v
                },
            )
            for k, v in cells.items()
        ][:2]
        if ep:
            pts.append(
                "Mean epochs actually trained: "
                + "; ".join(
                    f"{k.replace('_', ' ')}: "
                    + ", ".join(f"{m} {e:.0f}" for m, e in x.items() if e)
                    for k, x in ep
                )
                + "."
            )
        r16 = [
            v["pat16"]["r"]["genomic"] - v["default"]["r"]["genomic"]
            for v in cells.values()
            if "pat16" in v
        ]
        tm = [
            v["pat16"]["minutes"] / v["default"]["minutes"]
            for v in cells.values()
            if "pat16" in v and v["default"].get("minutes")
        ]
        out.append(
            dict(
                id="recipe",
                group=G27,
                status="provisional",
                figs=["recipe_sensitivity.png"],
                q="Are the epoch budget and early-stopping patience tuned, or just defaults?",
                choice=(
                    f"Patience 8 is slightly short at small D. Patience 16 is ahead in all {len(r16)} cells ({min(r16):+.4f} to {max(r16):+.4f}, mean {_m(r16):+.4f}), also on the designed set and random sequences at 10k and 30k, for {min(tm):.1f} to {max(tm):.1f} times the training time per model; the gain shrinks with size (+0.011 at 10k, +0.002 to +0.007 at 30k, +0.001 at 100k). Patience 4 loses up to 0.012. Halving or doubling the epoch budget moves the ensemble by less than 0.005 with mixed sign. Provisional recommendation: patience 16 for retrains at 10k and 30k (above the coordinator's +0.003 and costing only minutes), patience 8 from 100k up; one seed per arm, so a second seed should confirm before the standard changes."
                ),
                points=pts,
                caveat="One seed per arm; the epoch change also rescales the learning-rate schedule. Patience and epochs were changed for every architecture alike, not tuned per architecture.",
                change="A second seed in which patience 16 again beats 8 by more than 0.003 at 10k and 30k would make it the default there; a gain at 100k and above would extend it.",
            )
        )
    # ------------------------------------------------------------ validation labels across D
    V = o.get("valdep")
    if V and any(k not in ("summary", "order") for k in V):
        cells = {k: v for k, v in V.items() if k not in ("summary", "order")}
        sm = V.get("summary", {})
        pts = [
            "Oracle-relabelled validation set (V0c) minus the default validation set, ensemble r on the genomic test set, one seed per arm: "
            + "; ".join(
                f"{k.replace('_', ' ')} {v['delta']['genomic']:+.4f}" for k, v in cells.items()
            )
            + ".",
            "Designed set / random sequences: "
            + "; ".join(
                f"{k.replace('_', ' ')} {v['delta']['ood']:+.4f} / {v['delta']['random_32k']:+.4f}"
                for k, v in cells.items()
            )
            + ".",
        ]
        if sm:
            pts.append(
                "Mean over cells: "
                + "; ".join(
                    f"{EVN[e]} {x['mean']:+.4f}"
                    + (f" (SE {x['se']:.4f})" if x.get("se") else "")
                    + f", {x['positive']} of {x['n']} positive"
                    for e, x in sm.items()
                )
                + "."
            )
        if V.get("order"):
            pts.append(
                "Reservoir order on the genomic test set (best first), default / V0c: "
                + "; ".join(
                    f"{k}: {' > '.join(RESN[s] for s in v['default'])} / {' > '.join(RESN[s] for s in v['v0c'])}"
                    for k, v in V["order"].items()
                )
                + "."
            )
        out.append(
            dict(
                id="valdep",
                group=G27,
                status="supported",
                figs=[],
                q="Does relabelling the validation set with the current oracle change results at other sizes?",
                choice="Decision: V0c (the same real chr19/21/X sequences scored by oracle_v3) is the standard validation set from now on, because it is the one set in the same label space as every pool. Results from the old set stay usable: across nine cells it moves the genomic-test ensemble by +0.002 on average (single cells -0.002 to +0.011) and the designed set by up to 0.027 in either direction (single-seed noise), and the reservoir ordering on the genomic test is the same at 10k, 30k and 100k.",
                points=pts,
                caveat="One seed per arm; V0 and V0c share sequences and differ only by labels (r 0.983, mean offset +0.067).",
                change="Not applicable: this is a correctness choice; the check is that nothing downstream reorders.",
            )
        )
    # ------------------------------------------------------------ HepG2 across sizes
    Hd = o.get("hgdep")
    if Hd and any(k != "summary" for k in Hd):
        cells = {k: v for k, v in Hd.items() if k != "summary"}
        sm = Hd.get("summary", {})
        pts = [
            "Direct differential student minus difference of a K562 and a HepG2 student at 10k and 100k (one pool per reservoir), genomic test: "
            + "; ".join(
                f"{k.replace('_', ' ')} {v['direct']['genomic']['diff'] - v['two_students']['genomic']:+.3f}"
                for k, v in cells.items()
            )
            + ".",
            "Same contrast on the designed set / random sequences: "
            + "; ".join(
                f"{k.replace('_', ' ')} {v['direct']['ood']['diff'] - v['two_students']['ood']:+.3f} / {v['direct']['random_32k']['diff'] - v['two_students']['random_32k']:+.3f}"
                for k, v in cells.items()
            )
            + ".",
        ]
        if sm:
            pts.append(
                "Across these cells: "
                + "; ".join(
                    f"{EVN[e]} mean {x['mean']:+.3f} (range {x['min']:+.3f} to {x['max']:+.3f}), {x['positive']} of {x['n']} positive"
                    for e, x in sm.items()
                )
                + ". At 30k it was +0.028 / +0.115 / +0.040 in nine of nine runs."
            )
        out.append(
            dict(
                id="hgdep",
                group=G27,
                status="supported",
                figs=[],
                q="Does the direct differential student win at 10k and 100k as well?",
                choice="Keep the direct differential student. It beats the difference of two students on the designed set and on random sequences in every one of the six cells (mean +0.12 and +0.04) and on the genomic test in five of six (mean +0.020; +0.028 at 30k in nine of nine runs). The genomic-test gain shrinks with size (10k: +0.06 genomic, +0.04 motif; 100k: +0.006 to +0.021) and reverses once, for random 10k (-0.023).",
                points=pts,
                caveat="One pool per reservoir and size; K562 arms are the V0c-validated retrains.",
                change="A size at which two students beat the direct student on all three sets would make the choice size-dependent; so far only random at 10k, on the genomic test, does.",
            )
        )
    # ------------------------------------------------------------ reduced menus (all reservoirs) and ensemble methods
    Rd = o.get("reduced_all")
    if Rd:

        def cell(st, D, K, key):
            return Rd.get(st, {}).get(str(D), {}).get("K", {}).get(str(K), {}).get(key)

        allk8 = [
            cell(st, D, 8, "prev_val")
            for st in Rd
            for D in (30000, 100000, 300000)
            if cell(st, D, 8, "prev_val") is not None
        ]
        allk12 = [
            cell(st, D, 12, "prev_val")
            for st in Rd
            for D in (30000, 100000, 300000)
            if cell(st, D, 12, "prev_val") is not None
        ]
        allr8 = [
            cell(st, D, 8, "random")
            for st in Rd
            for D in (30000, 100000, 300000)
            if cell(st, D, 8, "random") is not None
        ]
        pts = [
            "Retraining only K of the menu's architectures, chosen by their validation r at the previous size, on nested pool 1 (single seed, ensemble r on the genomic test set, loss against the full menu), K = 5 / 8 / 12: "
            + "; ".join(
                f"{RESN[st]} "
                + ", ".join(
                    f"{D // 1000}k {cell(st, D, 5, 'prev_val'):+.4f}/{cell(st, D, 8, 'prev_val'):+.4f}/{cell(st, D, 12, 'prev_val'):+.4f}"
                    for D in (30000, 100000, 300000)
                    if cell(st, D, 8, "prev_val") is not None
                )
                for st in Rd
            )
            + ".",
            f"Across the nine cells K = 8 loses {-_m(allk8):.4f} on average (worst {-min(allk8):.4f}), K = 12 loses {-_m(allk12):.4f} (worst {-min(allk12):.4f}); eight random architectures lose {-_m(allr8):.4f}. Against eight random architectures, the previous-size choice is ahead by {_m(allk8) - _m(allr8):+.4f} on average, by 0.003 or more in {sum(1 for st in Rd for D in (30000, 100000, 300000) if cell(st, D, 8, 'prev_val') is not None and cell(st, D, 8, 'prev_val') - cell(st, D, 8, 'random') >= 0.003)} of nine cells and slightly behind in {sum(1 for st in Rd for D in (30000, 100000, 300000) if cell(st, D, 8, 'prev_val') is not None and cell(st, D, 8, 'prev_val') - cell(st, D, 8, 'random') < -0.0002)}: the ranking recovers about half of what a random reduction loses, and most of the saving comes from retraining fewer architectures.",
            "Earlier figures for genomic only (legacy pools, three seeds): about 0.002 at 30k and 100k and 0.0002 at 300k; the nested single-seed numbers here are larger at 30k, which is the more conservative reading.",
        ]
        out.append(
            dict(
                id="reducedall",
                group=G27,
                status="supported",
                figs=[],
                q="Does retraining a reduced menu at large D hold for random and motif pools too?",
                choice="Retrain about 12 architectures from 300k up, chosen by the previous size's validation r (loss 0.001 or less at 300k, up to 0.004 at 30k and 100k), or 8 if cost forces it (loss up to 0.006, on average half of what eight random architectures lose).",
                points=pts,
                caveat="Single seed and one pool per cell; architecture rankings are noise-limited, which is why the selection adds little.",
                change="A selection rule that beats random by more than 0.003 at K = 8 consistently would justify a ranking-based reduction.",
            )
        )
    E = o.get("ensmethods")
    if E:
        cells = list(E.values())
        mean = lambda m: _m([c["genomic"][m] for c in cells])  # noqa: E731
        lo = lambda m: min(c["genomic"]["nnls"] - c["genomic"][m] for c in cells)  # noqa: E731
        hi = lambda m: max(c["genomic"]["nnls"] - c["genomic"][m] for c in cells)  # noqa: E731
        pts = [
            f"{len(cells)} cells (genomic, random, motif at 10k, 30k, 100k, 300k; one seed, 31 to 33 models each), ensemble r on the genomic test set. Mean over cells: plain mean of all models {mean('mean_all'):.4f}, mean of the 8 best by validation {mean('mean_top8'):.4f}, greedy forward selection {mean('greedy'):.4f}, non-negative weights {mean('nnls'):.4f}, best single model by validation {mean('best_single_by_val'):.4f}.",
            f"Non-negative weights minus the other methods: plain mean {lo('mean_all'):+.4f} to {hi('mean_all'):+.4f} (largest at 10k), top-8 mean {lo('mean_top8'):+.4f} to {hi('mean_top8'):+.4f}, greedy {lo('greedy'):+.4f} to {hi('greedy'):+.4f}. Greedy selection with replacement ties the non-negative fit; both beat averaging everything, and the gap to the single model chosen by validation is {lo('best_single_by_val'):+.4f} to {hi('best_single_by_val'):+.4f} (largest at 10k).",
            "The earlier +0.01 over greedy top-K selection refers to a fixed-K greedy; a forward selection with replacement (Caruana) closes that gap.",
        ]
        out.append(
            dict(
                id="ensmethods",
                group=G27,
                status="supported",
                figs=[],
                q="Does the ensembling method matter across reservoirs and sizes?",
                choice=f"Keep non-negative weights on validation predictions; greedy forward selection with replacement is an equivalent alternative, a mean of the 8 best by validation is within {max(abs(lo('mean_top8')), abs(hi('mean_top8'))):.3f} of it at these sizes, and unweighted averaging of all models costs {lo('mean_all'):.3f} to {hi('mean_all'):.3f} (most at 10k).",
                points=pts,
                caveat="Weights are fitted on the same validation set that selects early stopping; one seed.",
                change="A method that beats non-negative weights by more than 0.003 across cells would replace it.",
            )
        )
    # ------------------------------------------------------------ 600k retune
    Ht = o.get("hptune")
    if Ht and Ht.get("arms"):
        pts = [
            "Three architectures (the best three on random 300k by validation) trained on the random 600k pool with one recipe change each, against their default-recipe run on the same pool: "
            + "; ".join(
                f"{arm}: " + ", ".join(f"{v['d_test']:+.4f}" for v in out_.values())
                for arm, out_ in Ht["arms"].items()
            )
            + " (change in test r on the genomic test set, per architecture).",
            "Default-recipe 600k runs: "
            + ", ".join(
                f"test {v['test']:.4f} ({v['epochs']} epochs, {v['minutes']:.0f} min)"
                for v in Ht["base"].values()
            )
            + ".",
        ]
        _arm = {a_: _m([v["d_test"] for v in out_.values()]) for a_, out_ in Ht["arms"].items()}
        _all = [v["d_test"] for out_ in Ht["arms"].values() for v in out_.values()]
        HPT_TXT = (
            "No single recipe change helps these architectures at 600k: mean change in single-model test r "
            + ", ".join(f"{a_} {x:+.4f}" for a_, x in _arm.items())
            + f" (individual runs {min(_all):+.4f} to {max(_all):+.4f}, mixed sign). The default recipe is therefore not what is holding 600k back under one-change retunes, and the 6 to 16% shortfall against the power law is left as data scaling (or an interaction between architecture and recipe that these runs do not reach)."
        )
        out.append(
            dict(
                id="hptune",
                group=G27,
                status="provisional",
                figs=[],
                q="Is the shortfall against the extrapolation a recipe that was not retuned at 600k?",
                choice=HPT_TXT,
                points=pts,
                caveat="Three architectures, one seed, one change at a time; the real search space (architecture and recipe jointly) is larger.",
                change="A recipe change that gains 0.005 or more at 600k would mean the menu should be retuned at large D.",
            )
        )
    return out


def patch_cards(cards: list[dict], d: dict) -> None:
    """Append the Oct 27 findings to existing cards (coordinator consistency, extrapolation metrics, the search-versus-measurement note)."""
    o = d.get("oct22", {})
    by = {c["id"]: c for c in cards}
    C = o.get("coord")
    if C and "promptgrid" in by:
        r = C.get("coordinator")
        if r:
            ov = r["overall"]
            by["promptgrid"]["points"].append(
                f"Consistency of the coordinator effect across the grid (random-effects meta-analysis over {len(r['cells'])} cells, pooled within-cell SD {r['sd_pool']:.4f}): {ov['random']:+.4f} (SE {ov['se_random']:.4f}), between-cell SD {ov['tau']:.4f}, I-squared {ov['I2']:.0%}, heterogeneity p {ov['p_heterogeneity']:.2f}; {r['sign_consistency']['positive']} of {r['sign_consistency']['of']} cells positive; "
                + "; ".join(
                    f"{k.replace('size_', '')} {r[k]['mean']:+.4f} (SE {r[k]['se']:.4f})"
                    for k in ("size_30k", "size_10k")
                    if k in r
                )
                + (
                    f"; 30k minus 10k {r['size_difference_30k_minus_10k']['mean']:+.4f} (p {r['size_difference_30k_minus_10k']['p']:.2f})"
                    if "size_difference_30k_minus_10k" in r
                    else ""
                )
                + ". The pooled effect is small and shows no detectable heterogeneity with this many pairs, but by cell it is about +0.005 on genomic and random at 30k (intervals above zero), +0.004 on genomic at 10k, and about 0 on motif at both sizes and on random at 10k. The loop-element card shows that at genomic 30k this is the effect of coding in a separate call, not of the coordinator choosing ideas."
            )
        e = C.get("explore")
        if e:
            ov = e["overall"]
            by["promptgrid"]["points"].append(
                f"Explore prompt, same analysis: {ov['random']:+.4f} (SE {ov['se_random']:.4f}), {e['sign_consistency']['positive']} of {e['sign_consistency']['of']} cells positive"
                + (
                    f"; 30k minus 10k {e['size_difference_30k_minus_10k']['mean']:+.4f} (p {e['size_difference_30k_minus_10k']['p']:.2f}), so it hurts at 10k and is neutral at 30k."
                    if "size_difference_30k_minus_10k" in e
                    else "."
                )
            )
    X = o.get("extrap_metrics")
    if X and "anchor" in by:
        P = []
        for st, v in X.items():
            f = v["fit"]["genomic"]
            P.append(
                f"{RESN.get(st, st)}: "
                + "; ".join(
                    f"{m.replace('_', ' ')} {f[m]['rel_error'].get('600000', float('nan')):+.1%} / {f[m]['rel_error'].get('1000000', float('nan')):+.1%}"
                    for m in ("one_minus_r", "mse_calibrated", "mse_raw", "one_minus_spearman")
                )
            )
        by["anchor"]["points"].append(
            "Same comparison under other metrics (measured error relative to the power-law prediction, 600k / 1M, genomic test; positive = worse than predicted): "
            + " | ".join(P)
            + ". The shortfall appears in every metric (6 to 16%), so it is not an artefact of correlation as the yardstick; error-based metrics extrapolate slightly better than 1 - r but still overpredict."
        )
        by["anchor"]["points"].append(
            "How the points were measured: every size from 10k to 1M is the same fixed menu (architectures and recipes from the 30k genomic searches, 33 entries, 29 common to all sizes) retrained on that size's pool; no search or recipe tuning was repeated at the larger sizes (the 300k search check ties the menu, and the 600k retune card tests the recipe directly). So the comparison is like for like across sizes, and the shortfall is either data scaling or a recipe that stops being optimal at large D (the retune card finds no one-change recipe that recovers it)."
        )
    if "agentpilot" in by:
        by["agentpilot"]["caveat"] = by["agentpilot"]["caveat"].replace(
            "are running with six paired runs each.",
            "were run afterwards with six paired runs each (prompt-grid and loop-element cards): the arms with a separate coder gain about +0.005 at genomic 30k.",
        )
    # spec rows are patched in spec_rows_patch


def spec_patch(rows: list[dict], d: dict) -> None:
    o = d.get("oct22", {})
    by = {r["p"]: r for r in rows}
    if "Validation set (early stopping, ensemble weights, proposer feedback)" in by:
        r = by["Validation set (early stopping, ensemble weights, proposer feedback)"]
        r["s"] = (
            "Real genomic chr19 / 21 / X sequences (29,449), one set for every model and reservoir, labelled by oracle_v3 (V0c) from now on. The stored labels of the old set (V0) come from a different oracle (r 0.983, offset +0.067) while every pool is oracle_v3."
        )
        r["st"] = "supported"
        r["open"] = (
            "Old-set results stay usable (small shifts, same ordering); model-guided and similarity-filtered validation sets were not built."
        )
        V = o.get("valdep")
        if V and V.get("summary"):
            s = V["summary"]["genomic"]
            r["ev"] += (
                f" Across sizes and reservoirs V0c shifts ensemble r by {s['mean']:+.4f} on average ({s['n']} cells)."
            )
        r["card"] = "valdep" if V else "valvar"
    Rd = o.get("reduced_all")
    if Rd and "What is retrained at each D" in by:
        k12 = [Rd[st][D]["K"]["12"]["prev_val"] for st in Rd for D in Rd[st]]
        k8 = [Rd[st][D]["K"]["8"]["prev_val"] for st in Rd for D in Rd[st]]
        r = by["What is retrained at each D"]
        r["s"] = (
            "The whole menu on that reservoir's pool at every size up to 100k; about 12 architectures (8 if cost forces it) from 300k up, chosen by the previous size's validation r."
        )
        r["ev"] = (
            f"Nine cells (genomic, random, motif at 30k, 100k, 300k): K = 12 loses {-_m(k12):.4f} on average (worst {-min(k12):.4f}), K = 8 {-_m(k8):.4f} (worst {-min(k8):.4f}); random choices of 8 lose about 0.005, so the ranking recovers about half of that loss. Menu cost per seed (H100-hours) rises about as D^0.91."
        )
        r["card"] = "reducedall"
    E = o.get("ensmethods")
    if E and "Method" in by:
        _lo = lambda m: min(c["genomic"]["nnls"] - c["genomic"][m] for c in E.values())  # noqa: E731
        _hi = lambda m: max(c["genomic"]["nnls"] - c["genomic"][m] for c in E.values())  # noqa: E731
        by["Method"]["ev"] = (
            f"Across {len(E)} cells (genomic, random, motif, 10k to 300k): non-negative weights tie greedy forward selection ({_lo('greedy'):+.4f} to {_hi('greedy'):+.4f}), beat the plain mean of all models by {_lo('mean_all'):.4f} to {_hi('mean_all'):.4f} and the top-8 mean by {_lo('mean_top8'):+.4f} to {_hi('mean_top8'):+.4f}."
        )
        by["Method"]["st"] = "supported"
        by["Method"]["card"] = "ensmethods"
    C = o.get("coord")
    if C and "Prompt" in by and C.get("coordinator"):
        ov = C["coordinator"]["overall"]
        by["Prompt"]["open"] = (
            f"Agent loop with coordinator: {ov['random']:+.4f} (SE {ov['se_random']:.4f}) over the single-call loop, pooled over {len(C['coordinator']['cells'])} reservoir-size cells; at genomic 30k the gain is the separate coder (+0.0047) and the coordinator and reflector add +0.0003 and +0.0002. Guidance removal costs 0.016 to 0.020."
        )
    Rc = o.get("recipe")
    if Rc and "Epochs" in by:
        sm = Rc.get("summary", {})
        if sm:
            by["Epochs"]["ev"] += (
                " Recipe study: "
                + "; ".join(f"{m} {x['mean']:+.4f}" for m, x in sm.items())
                + " (mean change in ensemble r over cells)."
            )
            by["Epochs"]["open"] = (
                "One seed per arm; patience and epochs changed for all architectures alike."
            )
            by["Epochs"]["card"] = "recipe"
            if "Early stopping and time cap" in by:
                by["Early stopping and time cap"]["ev"] += (
                    " Patience 16 beat 8 in all five cells tested (+0.001 at 100k to +0.011 at 10k) for about 1.0 to 1.6 times the training time; patience 4 lost up to 0.012; epoch budget x0.5 / x2 moved the ensemble by less than 0.005."
                )
                by["Early stopping and time cap"]["open"] = (
                    "Patience 16 for 10k and 30k retrains is a provisional upgrade pending a second seed; per-architecture tuning not tried."
                )
                by["Early stopping and time cap"]["card"] = "recipe"
    L = o.get("loopctl")
    if L and "Prompt" in by:
        by["Prompt"]["card"] = "loopels"


def robustness_html(d: dict) -> str:
    """Which assumptions were tested at which sizes and reservoirs (and what was added to close the gaps)."""
    o = d.get("oct22", {})
    rows = [
        (
            "Fresh search matches the carried menu",
            "genomic 30k / 100k / 300k; random, motif, evoaug 30k / 100k; evo2 30k; six unseen families 10k / 30k (three at 100k)",
            "10k, 30k, 100k, 300k",
            "evoaug and evo2 at 300k; unseen families above 100k",
        ),
        (
            "Pooling independent runs beats single runs",
            "genomic 10k / 30k / 100k; random 30k; motif 30k",
            "10k to 100k",
            "random and motif at 10k and 100k; evoaug, evo2",
        ),
        ("Top 3 of each run is enough", "genomic 30k / 100k", "30k, 100k", "random and motif; 10k"),
        (
            "Default prompt (explore adds nothing)",
            "genomic, random, motif, evoaug, evo2",
            "10k and 30k",
            "100k and above (search runs are expensive there)",
        ),
        (
            "Agent loop over single-call loop (+0.003 pooled; the separate coder, not the coordinator)",
            "coordinator arms: genomic, random, motif, evoaug, evo2 at 30k; genomic, random, motif at 10k; component controls at genomic 30k only",
            "10k and 30k",
            "component controls on other reservoirs; 100k; evoaug and evo2 at 10k",
        ),
        (
            "Guidance block helps (-0.016 to -0.020 without it)",
            "genomic 30k (earlier); random 30k, motif 30k, genomic 10k",
            "10k and 30k",
            "100k; evoaug and evo2",
        ),
        (
            "Training recipe: patience 8 (patience 16 is +0.001 to +0.011 better, most at small D)",
            "genomic, random, motif at 30k; genomic at 10k and 100k",
            "10k, 30k, 100k",
            "random and motif at 10k and 100k; a second seed; 300k and up",
        ),
        (
            "Recipe still optimal at 600k (one-change retune)",
            "random 600k, three architectures",
            "600k",
            "motif; 1M; genomic cannot (pool cap)",
        ),
        (
            "Non-negative ensemble weights",
            "genomic, random, motif (now across 10k to 300k)",
            "10k to 300k",
            "evoaug, evo2, unseen families",
        ),
        (
            "Reduced menu from 300k up",
            "genomic, random, motif (now)",
            "30k, 100k, 300k",
            "above 300k; other reservoirs",
        ),
        (
            "Three seeds per architecture",
            "genomic, random, motif (nine seeds at 30k)",
            "30k (three-seed ladder at 10k and 100k)",
            "300k and above",
        ),
        (
            "Label noise does not matter",
            "genomic, random, motif",
            "10k, 30k, 100k",
            "300k and above",
        ),
        (
            "Validation set labelled by the current oracle",
            "genomic, random, motif",
            "10k, 30k, 100k",
            "300k",
        ),
        (
            "K562 menu serves HepG2 and the differential",
            "genomic, random, motif",
            "10k, 30k (nine runs), 100k",
            "300k; evoaug",
        ),
        (
            "Reduced selection ensemble and one round for acquisition",
            "one 30k genomic base, candidates from random, motif and evoaug",
            "30k to 100k",
            "other bases and sizes: the from-30k and from-300k runs",
        ),
        (
            "Scaling beyond 300k",
            "random and motif",
            "600k and 1M measured",
            "genomic and evoaug cannot (pool cap) or were not run",
        ),
    ]
    head = "<tr><th>Assumption</th><th>Reservoirs tested</th><th>Sizes tested</th><th>Not yet tested</th></tr>"
    body = "".join(
        f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{e}</td></tr>" for a, b, c, e in rows
    )
    return (
        '<h3 id="robustness" class="sub3">How far each assumption has been tested across sizes and reservoirs</h3><div class="tblwrap"><table><thead>'
        + head
        + "</thead><tbody>"
        + body
        + '</tbody></table></div><p class="note">Maintained by hand from the experiment directories; rows 5 to 8 and 13 to 14 were filled by the experiments launched overnight on Oct 27.</p>'
    )
