# ruff: noqa
"""Oct 25-26 additions to the evidence ledger: result cards, the procedure specification table and extra assumption rows.

Imported by build_justification_html.py (patch_builder6.py wires it in). All numbers come from d['oct22'] (outputs/justification_oct22/data.json).
"""

from __future__ import annotations

G26 = "Results from the Oct 25-26 program"
EV3 = ("genomic", "ood", "random_32k")
RESN = {"genomic": "genomic", "random": "random", "motif_ct_enriched": "motif (CT-enriched)"}


def _m(v):
    return sum(v) / len(v) if v else float("nan")


def oct26_cards(d: dict) -> list[dict]:
    o = d.get("oct22", {})
    out = live_card(d)
    f3 = lambda x: f"{x:.3f}"  # noqa: E731
    s3 = lambda x: f"{x:+.3f}"  # noqa: E731
    # ---------------------------------------------------------------- assay noise
    N = o.get("noise")
    if N:
        rows = []
        for st in ("genomic", "random", "motif_ct_enriched"):
            if st not in N:
                continue
            drops = [
                N[st][k]["0"]["r"]["genomic"] - N[st][k]["1"]["r"]["genomic"]
                for k in ("10000", "30000", "100000")
                if k in N[st] and "1" in N[st][k]
            ]
            d2 = (
                N[st]["30000"]["0"]["r"]["genomic"] - N[st]["30000"]["2"]["r"]["genomic"]
                if "2" in N[st].get("30000", {})
                else None
            )
            ex = N[st].get("exponent", {}).get("genomic", {})
            rows.append((st, min(drops), max(drops), d2, ex.get("0"), ex.get("1")))
        order30 = lambda s: sorted(
            ("genomic", "random", "motif_ct_enriched"),
            key=lambda k: -N[k]["30000"][s]["r"]["genomic"],
        )  # noqa: E731
        pts = [
            "Labels with the measurement error of the real assay (standard errors drawn from the Gosai K562 table, mean squared error 0.156 against a label variance of about 1; reliability about 0.89) were added to the training and validation labels; the test sets keep clean oracle labels. Same 33-architecture menu, one seed, replicate pool 1.",
            "Cost in ensemble r on the genomic test set, clean minus noisy, over 10k / 30k / 100k: "
            + "; ".join(f"{RESN[st]} {lo:.3f} to {hi:.3f}" for st, lo, hi, _, _, _ in rows)
            + ". Twice the real error at 30k: "
            + "; ".join(f"{RESN[st]} {d2:.3f}" for st, _, _, d2, _, _ in rows if d2 is not None)
            + ".",
            "Scaling exponent b in (1 - r) ~ D^-b on the genomic test set, clean / noisy: "
            + "; ".join(f"{RESN[st]} {e0:.3f} / {e1:.3f}" for st, _, _, _, e0, e1 in rows if e0)
            + ". Noise flattens the curve slightly; it does not change its shape.",
            "Reservoir ordering on the genomic test set at 30k, best first: clean "
            + " > ".join(RESN[k] for k in order30("0"))
            + "; noisy "
            + " > ".join(RESN[k] for k in order30("1"))
            + ".",
            "The designed (out-of-distribution) set loses more at 100k: "
            + "; ".join(
                f"{RESN[st]} {N[st]['100000']['0']['r']['ood'] - N[st]['100000']['1']['r']['ood']:.3f}"
                for st in N
                if "100000" in N[st]
            )
            + ".",
        ]
        out.append(
            dict(
                id="noise",
                group=G26,
                status="supported",
                figs=["assay_noise.png"],
                q="Do the conclusions survive measurement noise in the labels?",
                choice="Yes. Realistic assay error costs about 0.005 to 0.010 in ensemble r and flattens the scaling exponent by 0.01 to 0.02; the reservoir ordering and the shape of the curves are unchanged. The synthetic-label study can be read as a study of real data.",
                points=pts,
                caveat="One seed and one replicate pool per cell; heteroscedastic noise with errors independent of activity (the real standard errors may correlate with expression). Real data also differ from oracle labels in other ways that noise does not model.",
                change="A loss above 0.02 at the real error level, or a reordering of reservoirs, would mean the oracle-label study overstates what is learnable from measured data.",
            )
        )
    # ---------------------------------------------------------------- high-D anchors
    A = o.get("anchor")
    if A:
        pts = []
        gaps = []
        for st in ("random", "motif_ct_enriched"):
            if st not in A or not A[st].get("pred"):
                continue
            P, pr = A[st]["points"], A[st]["pred"]["genomic"]
            row = [
                f"{RESN[st]}: "
                + ", ".join(
                    f"{int(k) // 1000}k {v['genomic']:.3f}"
                    for k, v in sorted(P.items(), key=lambda kv: int(kv[0]))
                )
            ]
            for D in ("600000", "1000000"):
                if D in P:
                    gaps.append(P[D]["genomic"] - pr[D])
                    row.append(
                        f"{int(D) // 1000}k measured {P[D]['genomic']:.3f} vs {pr[D]:.3f} extrapolated ({P[D]['genomic'] - pr[D]:+.3f})"
                    )
            pts.append(
                "; ".join(row)
                + f". Fit exponent {pr['exponent']:.3f}; {A[st]['n_arch_common']} architectures common to every size."
            )
        if gaps:
            pts.insert(
                0,
                f"Ensemble r on the genomic test set for the full menu retrained on random and motif pools of 10k to 1M (genomic cannot go above its 314,981-sequence pool). A power law fitted to 10k-300k overpredicts the measured 600k and 1M points by {min(-g for g in gaps):.3f} to {max(-g for g in gaps):.3f}.",
            )
        for st in ("random", "motif_ct_enriched"):
            if st in A and A[st].get("pred", {}).get("ood"):
                P, pr = A[st]["points"], A[st]["pred"]["ood"]
                if "1000000" in P:
                    pts.append(
                        f"Designed set, {RESN[st]}: 1M measured {P['1000000']['ood']:.3f} vs {pr['1000000']:.3f} extrapolated."
                    )
        out.append(
            dict(
                id="anchor",
                group=G26,
                status="provisional",
                figs=["anchor_points.png"],
                q="Does the extrapolation from 10k-300k hold at 600k and 1M?",
                choice="Not quite: accuracy keeps rising but about 0.01 below the power-law extrapolation, so read the 3.2M library as 'still improving, with returns diminishing faster than a power law'. Use the measured 600k and 1M points as anchors, not the fit.",
                points=pts,
                caveat="Common-architecture ensembles (a few architectures fail the build-time speed check at some sizes, so the common set is 29 of 33) and one seed per size on replicate pool 1. Genomic is capped by the pool, so the anchors cover the two synthetic reservoirs only.",
                change="If the final numbers close the gap (within 0.005), the extrapolation is fine; if it widens, plan the 3.2M library on measured points only.",
            )
        )
    # ---------------------------------------------------------------- acquisition
    Q, Q3 = o.get("acq"), o.get("acq3", {})
    if Q and Q.get("arms"):
        a = Q["arms"]
        R = a["rand"]["ens16"]
        full = [a[k]["ens16"]["genomic"] for k in ("fullmean", "fullstd") if k in a]
        red = [a[k]["ens16"]["genomic"] for k in ("k8mean", "k8std", "k8rmean", "k8rstd") if k in a]
        allm = [a[k]["ens16"] for k in a if k != "rand"]
        pts = [
            "Setting: 30k genomic base, 300k labelled candidates (100k each from random, motif CT-enriched and evoaug), 70k added, final 8-architecture menu with two seeds trained at 100k. Selection scores come from the menu retrained on the base: highest ensemble mean, or highest across-architecture standard deviation.",
            "Ensemble r (genomic test / designed set / random sequences): random selection "
            + " / ".join(f3(R[e]) for e in EV3)
            + "; model-based selection ranges "
            + " / ".join(
                f"{min(x[e] for x in allm):.3f} to {max(x[e] for x in allm):.3f}" for e in EV3
            )
            + ". Model-based selection gains "
            + ", ".join(
                f"{s3(min(x[e] - R[e] for x in allm))} to {s3(max(x[e] - R[e] for x in allm))}"
                for e in EV3
            )
            + " over random.",
            f"Selection ensemble: the full 33-architecture menu ({f3(_m(full))} genomic), the 8 best by validation ({f3(_m([a[k]['ens16']['genomic'] for k in ('k8mean', 'k8std') if k in a]))}) and 8 random architectures ({f3(_m([a[k]['ens16']['genomic'] for k in ('k8rmean', 'k8rstd') if k in a]))}) are indistinguishable (spread {max(red + full) - min(red + full):.3f}, seed noise of an 8-model ensemble about {Q.get('seed_sd_estimate', 0.006):.3f}). Overlap of the picks (Jaccard) between full and 8-best selection: 0.87 for the mean score, 0.60-0.67 for the standard-deviation score.",
            "Mean-score selection picks high-activity sequences (mean label 1.8 against 0.6 at random) from the random and motif sources first.",
        ]
        if Q3.get("arms"):
            for crit, v in Q3["arms"].items():
                pts.append(
                    f"Three rounds of 23.3k against one round of 70k, {crit} score (genomic / designed / random): three rounds "
                    + " / ".join(f3(v["three_round"][e]) for e in EV3)
                    + ", one round "
                    + " / ".join(f3(v["one_round"][e]) for e in EV3)
                    + f"; overlap of the picks {v['jaccard_selected']:.2f}."
                )
            pts.append(
                "Adaptive rounds therefore buy nothing here, and cost more training; the mean-score variant is worse on random sequences, plausibly because repeatedly taking the highest-scoring sequences narrows the training set."
            )
        out.append(
            dict(
                id="acqsel",
                group=G26,
                status="provisional",
                figs=["acq_selection.png"],
                q="Can acquisition use a reduced selection ensemble, and are several adaptive rounds worth it?",
                choice="Select with an 8-architecture ensemble (as good as the full menu) and in one round; model-based selection beats random by about 0.01 on genomic test sequences and by about 0.12 on the designed set.",
                points=pts,
                caveat="One base, one candidate pool and one replicate per arm; the candidate pool mixes three reservoirs, so the large designed-set gain is probably the motif and evoaug sequences it contains. Only two criteria were tried and the final models were the 8-architecture menu, not the full one.",
                change="A different candidate pool or a second replicate in which three rounds beat one by more than 0.005 would reopen the round count.",
            )
        )
    # ---------------------------------------------------------------- HepG2 / differential
    H = o.get("hg")
    if H and H.get("summary"):
        S = H["summary"]
        runs = {k: v for k, v in H.items() if isinstance(v, dict) and "direct" in v}
        mk = lambda t, e: _m([v["direct"][e][t] for v in runs.values()])  # noqa: E731
        pts = [
            f"Setting: the K562-optimised 33-architecture menu retrained on HepG2 labels and on the differential label (K562 minus HepG2) at 30k, three replicate pools of genomic, random and motif ({S['n_runs']} runs), validated on the same real genomic sequences relabelled by the matching oracle ensembles.",
            "Mean ensemble r on the genomic test set: K562 students "
            + f3(mk("k562", "genomic"))
            + ", HepG2 students "
            + f3(mk("hepg2", "genomic"))
            + ", direct differential students "
            + f3(mk("diff", "genomic"))
            + "; a K562 student scored on HepG2 labels reaches only "
            + f3(_m([v["k562_student_as"]["genomic"]["hepg2"] for v in runs.values()]))
            + ", so the menu transfers but the weights and models must be retrained on the new target.",
            "A single student trained on the differential label beats the difference of a K562 and a HepG2 student by "
            + "; ".join(
                f"{e.replace('_32k', '')} {S['direct_minus_two_students'][e]['mean']:+.3f} [{S['direct_minus_two_students'][e]['lo']:+.3f}, {S['direct_minus_two_students'][e]['hi']:+.3f}], {S['direct_minus_two_students'][e]['wins']} of {S['direct_minus_two_students'][e]['n']} runs"
                for e in EV3
            )
            + " (genomic test, designed set, random sequences).",
            "Agreement of single-architecture rankings between K562 and HepG2 targets (Spearman, genomic test): mean "
            + f"{S['arch_rank_spearman_mean']['genomic']:.2f}"
            + ", from 0.2 to 0.8 across runs; the ensemble transfers better than any single architecture's rank.",
        ]
        out.append(
            dict(
                id="hg",
                group=G26,
                status="supported",
                figs=["hepg2_differential.png"],
                q="Does the K562-tuned menu serve HepG2 and the differential target?",
                choice="Yes for HepG2 (retrain the same menu on HepG2 labels; no new search). For cell-type specificity, train a student directly on the differential label rather than subtracting two students: +0.03 on genomic test sequences and +0.12 on the designed set, in all nine runs.",
                points=pts,
                caveat="30k only, K562-chosen menu, oracle-derived labels for both cells (the HepG2 oracle is a partial-pool ensemble whose designed rows are unseen). A shared-trunk two-output model was not tried.",
                change="If a search run on the differential label found architectures that beat the K562 menu by more than 0.01 there, the menu would need a per-target search.",
            )
        )
    # ---------------------------------------------------------------- prompt grids
    Gd, Ag, Ag2 = o.get("grid"), o.get("agent_grid"), o.get("agent2")
    if Gd and Gd.get("pooled"):
        p = Gd["pooled"]["genomic"]
        pts = [
            f"Explore-style prompt against the default prompt, paired by replicate pool, five reservoirs at 30k (genomic 9, random 6, motif 6, evoaug 3, evo2 3 pairs) and at 10k (genomic 6, random 6, motif 6, evoaug 3, evo2 3 pairs): pooled over {p['n']} paired runs the ensemble difference on genomic test sequences is {p['mean']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}], {p['wins']} of {p['n']} runs ahead (designed set {Gd['pooled']['ood']['mean']:+.4f}, random sequences {Gd['pooled']['random_32k']['mean']:+.4f}).",
            "Cell by cell, genomic 30k is the only clear gain ("
            + f"{Gd['cells']['genomic_30k']['genomic']['mean']:+.4f}, {Gd['cells']['genomic_30k']['genomic']['wins']} of {Gd['cells']['genomic_30k']['genomic']['n']}"
            + "); the evo2 30k pairs share one legacy pool, so that row is weaker. The 10k cells are flat or negative for every reservoir.",
        ]
        status = "supported"
        choice = "Keep the default prompt. The explore-style prompt's gain on genomic 30k does not generalise (pooled difference about zero over 51 paired runs)."
        if Ag2 and Ag2.get("arms"):
            ar = Ag2["arms"]
            pts.append(
                "Agent loop (proposer, coordinator that keeps the best 2 of 6 ideas, coder, evaluator, optional reflector) on the six genomic 30k pools: "
                + "; ".join(
                    f"{k} {v['genomic']['mean']:+.4f} [{v['genomic']['lo']:+.4f}, {v['genomic']['hi']:+.4f}], {v['genomic']['wins']} of {v['genomic']['n']}"
                    for k, v in ar.items()
                )
                + ". The coordinator is the one component that helps; the reflector adds nothing and the explore-style proposer does nothing."
            )
        if Ag and Ag.get("pooled"):
            pg = Ag["pooled"]["genomic"]
            pts.append(
                f"Coordinator without reflector across reservoirs and sizes ({pg['n']} paired runs): pooled {pg['mean']:+.4f} [{pg['lo']:+.4f}, {pg['hi']:+.4f}], {pg['wins']} of {pg['n']} ahead. "
                + "; ".join(
                    f"{k.replace('_', ' ')} {v['genomic']['mean']:+.4f} (n={v['genomic']['n']})"
                    for k, v in Ag["cells"].items()
                )
                + "."
            )
            status = "provisional"
            choice = (
                f"Keep the default prompt and single-call loop for now. The explore-style prompt does nothing across the grid ({p['mean']:+.4f} over {p['n']} paired runs). The coordinator helps a little: +0.005 on genomic and random 30k, smaller on evoaug and evo2 30k, nothing on motif or at 10k, "
                f"pooled {pg['mean']:+.4f} [{pg['lo']:+.4f}, {pg['hi']:+.4f}] over {pg['n']} paired runs ({pg['wins']} ahead, p {pg['p']:.2f}). That is below the 0.005 a change must clear, at about three times the proposer tokens, so it stays optional."
            )
        elif Ag2:
            status = "provisional"
            choice += " The coordinator (+0.005, 6 of 6 pairs on genomic 30k) is being tested on the same grid."
        out.append(
            dict(
                id="promptgrid",
                group=G26,
                status=status,
                figs=["prompt_grid.png"],
                q="Do the prompt and agent variants hold up across reservoirs and sizes?",
                choice=choice,
                points=pts,
                caveat="All 54 runs of the coordinator grid are in. Paired by replicate pool but each cell has 3 to 9 pairs; single-run ensembles have a standard deviation of about 0.009, so only pooled results are interpretable. The coordinator spends about three times the proposer tokens per round.",
                change="A coordinator gain that holds across the grid (pooled interval above zero, above about 0.003) would justify adopting it; otherwise the single-call default stays.",
            )
        )
    # ---------------------------------------------------------------- seeds and ladder
    Sd, Ld = o.get("seeds"), o.get("ladder")
    if Sd:
        pts = [
            "Nine seeds of the 33-architecture menu on one 30k pool (seeds 0-2 existed; six more were added), ensemble r on the genomic test set against the number of seeds ensembled (mean and SD over random subsets): "
            + "; ".join(
                f"{RESN[st]} "
                + ", ".join(
                    f"{k} seed{'s' if k != '1' else ''} {v['curve'][k]['genomic']['mean']:.4f}"
                    for k in ("1", "3", "9")
                    if k in v["curve"]
                )
                for st, v in Sd.items()
            )
            + ".",
            "SD of the ensemble r between single-seed ensembles (nine) against between independent pools (one seed each): "
            + "; ".join(
                f"{RESN[st]} {v['between_seed_sd']['genomic']:.4f} vs {v['between_pool_sd']['genomic']:.4f}"
                for st, v in Sd.items()
                if v.get("between_pool_sd")
            )
            + ". Seeds on one pool carry about as much run-to-run noise as separate pools, but they cannot supply the data diversity that independent pools do.",
        ]
        if Ld:
            pts.append(
                "Scaling exponent b on the genomic test set from 10k, 30k and 100k, per seed (SD across seeds) and for the three-seed ensemble: "
                + "; ".join(
                    f"{RESN[st]} {', '.join(f'{x:.3f}' for x in v['exponent']['genomic']['per_seed'])} (SD {v['exponent']['genomic']['sd'] or float('nan'):.3f}), ensemble {v['exponent']['genomic']['ensemble3']:.3f}"
                    for st, v in Ld.items()
                )
                + ". A single seed fixes the exponent to about +/-0.01."
            )
        gs = {
            st: (
                v["curve"]["3"]["genomic"]["mean"] - v["curve"]["1"]["genomic"]["mean"],
                v["curve"]["9"]["genomic"]["mean"] - v["curve"]["3"]["genomic"]["mean"],
            )
            for st, v in Sd.items()
            if "9" in v["curve"]
        }
        G13 = (min(x[0] for x in gs.values()), max(x[0] for x in gs.values()))
        G39 = (min(x[1] for x in gs.values()), max(x[1] for x in gs.values()))
        FR3 = _m([x[0] / (x[0] + x[1]) for x in gs.values()])
        out.append(
            dict(
                id="seeds",
                group=G26,
                status="provisional",
                figs=["seeds_nine.png"],
                q="How many seeds per architecture, and are seeds a substitute for independent pools?",
                choice=f"Three seeds capture about {FR3:.0%} of the nine-seed gain (+{G13[0]:.4f} to +{G13[1]:.4f} over one seed); six more add only +{G39[0]:.4f} to +{G39[1]:.4f}. Beyond three, spend the budget on independent pools or more architectures. Scaling exponents carry about +/-0.01 of seed noise.",
                points=pts,
                caveat="30k, replicate pool 1, genomic eval; the exponent figures use three seeds at 10k, 30k and 100k (genomic seeds 0-2 from the nested ladder).",
                change="A reservoir where nine seeds beat three by more than 0.005 would argue for more seeds at that size.",
            )
        )
    # ---------------------------------------------------------------- 300k search against the menu
    G3 = o.get("g300")
    if G3:
        r = lambda x: " / ".join(f"{x[e]:.4f}" for e in EV3)  # noqa: E731
        sa, st3, mn, mr = (
            G3["search_all"],
            G3["search_top3_per_run"],
            G3["menu_all"],
            G3["menu_random_same_count"],
        )
        out.append(
            dict(
                id="g300",
                group=G26,
                status="supported",
                figs=[],
                q="Does the menu carried from 30k still match a fresh search at 300k?",
                choice="Yes. Three fresh genomic searches at 300k tie the retrained menu: carrying the architectures and recipes up the ladder costs nothing measurable at 300k, so the search need not be repeated per size.",
                points=[
                    f"Three independent AutoResearch runs of 20 models each on genomic 300k pools (default prompt, 120-minute per-model cap), against the 33-architecture menu retrained on genomic 300k. Single-run ensembles on the genomic test set: "
                    + ", ".join(f"{x['genomic']:.4f}" for x in G3["single_run"])
                    + ".",
                    f"Ensemble r (genomic test / designed set / random sequences): the top 3 of each run pooled ({st3['n']} models) {r(st3['r'])}; all {sa['n']} search models {r(sa['r'])}; the full menu ({mn['n']} models) {r(mn['r'])}; {mr['n']} randomly chosen menu architectures {r(mr['r'])}.",
                    f"All-model comparison: search minus menu {sa['r']['genomic'] - mn['r']['genomic']:+.4f} (genomic), {sa['r']['ood'] - mn['r']['ood']:+.4f} (designed set), {sa['r']['random_32k'] - mn['r']['random_32k']:+.4f} (random sequences), with 1.8 times as many search models. At equal count (9 models) search is {st3['r']['genomic'] - mr['r']['genomic']:+.4f} ahead of random menu architectures, about one noise unit; the menu's own choice of 8 by previous-size validation r closes that gap (see the high-D planning card).",
                    "This repeats the 10k, 30k and 100k result at the largest tested size, one reservoir.",
                ],
                caveat="One reservoir (genomic, the one the menu was distilled from), three runs, one seed of the menu; the other reservoirs were not searched at 300k.",
                change="A search at 300k on a synthetic reservoir that beat the retrained menu by more than 0.005 would argue for per-reservoir search at the high sizes.",
            )
        )
    # ---------------------------------------------------------------- Zoonomia rebuild
    Z = o.get("zoo")
    if Z:
        rr = [
            (k, v)
            for k, v in Z.items()
            if k != "pool_overlap" and isinstance(v, dict) and "old" in v
        ]
        dg = [v["new"]["genomic"] - v["old"]["genomic"] for _, v in rr]
        ov = Z.get("pool_overlap", {})
        out.append(
            dict(
                id="zoo",
                group=G26,
                status="supported",
                figs=[],
                q="Did removing validation chromosomes from the Zoonomia reservoir change its result?",
                choice="No. The rebuilt Zoonomia pools (validation chromosomes excluded, as the sampler now does by default) give the same poor ensemble as before; Zoonomia stays the weakest reservoir at every size.",
                points=[
                    f"Ensemble r on the genomic test set, rebuilt minus old, over 10k and 30k and three replicate pools ({len(rr)} comparisons): mean {_m(dg):+.3f}, range {min(dg):+.3f} to {max(dg):+.3f}; sign changes across pools, so there is no systematic shift.",
                    "Absolute level: rebuilt 30k genomic-test r "
                    + ", ".join(f"{v['new']['genomic']:.3f}" for k, v in rr if k.startswith("30k"))
                    + " (random reservoir at 30k is about 0.76).",
                    (
                        f"Rebuilt and old pools share {ov['r1']['shared']:,} of {ov['r1']['n_new']:,} sequences (replicate 1); the rest are new draws."
                        if ov.get("r1")
                        else "Pools were regenerated with the same seeds."
                    ),
                ],
                caveat="Three pools per size, one seed. The old pools also had genomic-test chromosomes available to the sampler; both versions are now excluded by default.",
                change="Not applicable: the rebuild was a correctness fix and the conclusion is unchanged.",
            )
        )
    return out


def live_card(d: dict) -> list[dict]:
    L = d.get("oct22", {}).get("live")
    if not L:
        return []
    g = L["groups"]
    pts = [f"Snapshot {L['stamp']}: {L['mine']}.", "Cluster: " + L["free"] + "."]
    for k, (done, tot, run, pend, wait) in g.items():
        if done < tot:
            pts.append(
                f"{k}: {done} of {tot} cells done, {run} running, {pend} queued, {wait} waiting for a slot."
            )
    fin = [k for k, v in g.items() if v[0] == v[1]]
    pts.append("Finished: " + "; ".join(fin) + ".")
    if all(v[0] == v[1] for v in g.values()):
        return [
            dict(
                id="program5",
                group=G26,
                status="supported",
                figs=[],
                q="Is anything from the Oct 25-27 program still running?",
                choice=f"No. Every group finished by the snapshot of {L['stamp']}; the cards of this page contain all of its results.",
                points=[
                    f"Snapshot {L['stamp']}: {L['mine']}.",
                    "Finished: " + "; ".join(fin) + ".",
                ],
                caveat="Replicates beyond those listed (a second seed for the recipe study, the separate-coder arm on other reservoirs, a shared-core pool above 100k) have not been started.",
                change="",
            )
        ]
    pts.append(
        "Estimate from this snapshot: the non-LLM groups (nine seeds, seed ladder, replicate HepG2 students, the last 1M shard) are all submitted and should finish within about 2 to 4 hours, the 100k unseen-family and evoaug 300k retrains queued behind them add roughly 3 to 5 hours; the coordinator LLM grid depends on the usage-limit windows (about two hours of running per window) and should take 4 to 8 hours."
    )
    return [
        dict(
            id="program5",
            group=G26,
            status="running",
            figs=[],
            q="What is running now, and when will it finish?",
            choice="Everything left is either a replicate that tightens an existing result or a coverage fill-in; none of it is expected to change a recommendation unless a card above says so.",
            points=pts,
            caveat="The cluster's H100 queue is shared and often full, and the V100 caps (20 slow-queue plus 2 fast per user) bound throughput, so estimates move by hours.",
            change="",
        )
    ]


def spec_rows(d: dict) -> list[dict]:
    """The procedure, parameter by parameter: what the recipe sets, the evidence, and what is still open."""
    o = d.get("oct22", {})
    R, C, Rd, Cv = o.get("runs", {}), o.get("cost", {}), o.get("reduced", {}), o.get("converge", {})
    k90 = [r["k90"] for r in R.values()] if R else [6, 11]
    gap3 = (
        [r["curve"][str(r["n_runs"])]["mean"][0] - r["curve"]["3"]["mean"][0] for r in R.values()]
        if R
        else [0.01]
    )
    red8 = {D: Rd[D]["K"]["8"]["prev_val"][0] - Rd[D]["full"][0] for D in Rd} if Rd else {}
    pat = [Cv[k]["stopped_by_patience"] for k in Cv] if Cv else [1.0]
    S = o.get("seeds") or {}
    seedtxt = (
        "; ".join(
            f"{RESN[s]} {v['curve']['1']['genomic']['mean']:.4f} (1) to {v['curve']['3']['genomic']['mean']:.4f} (3) to {v['curve']['9']['genomic']['mean']:.4f} (9 seeds)"
            for s, v in S.items()
        )
        if S
        else "pending"
    )
    N = o.get("noise") or {}
    ax = o.get("archexp", {}).get("exponents", {})
    P = o.get("grid", {}).get("pooled", {}).get("genomic")
    HG = (o.get("hg") or {}).get("summary", {}).get("direct_minus_two_students", {}).get("genomic")
    AS = o.get("archsrc", {}).get("results", {})
    sp = []
    for t in ("genomic", "random", "motif", "grm"):
        v = [
            AS[f"{c}|{t}"]["enet"]["genomic"]
            for c in ("MG", "MR", "MGR", "MF", "MX")
            if f"{c}|{t}" in AS
        ]
        if len(v) > 1:
            sp.append(max(v) - min(v))
    ASRC = max(sp) if sp else float("nan")
    sat = d.get("saturation100", {})
    SATTXT = (
        f"One run gains {sat['mean30'][-1] - sat['mean30'][0]:.4f} from 6 to 30 models at 30k and {sat['mean100'][-1] - sat['mean100'][0]:.4f} at 100k; the last 12 models add at or below single-run noise."
        if sat.get("mean100")
        else "see card"
    )
    ag = (o.get("agent_grid") or {}).get("pooled", {}).get("genomic")
    COORD = (
        f"Coordinator stage (+0.005 on genomic and random 30k, none on motif or at 10k): pooled {ag['mean']:+.4f} [{ag['lo']:+.4f}, {ag['hi']:+.4f}] over {ag['n']} paired runs; below the 0.005 bar, optional."
        if ag
        else "Coordinator variant is being tested on the same grid."
    )
    cost = {
        k: C[k]["menu_gpu_h_one_seed"] for k in ("10000", "30000", "100000", "300000") if k in C
    }
    rows = [
        # ---- data and labels
        dict(
            stage="1. Data and labels",
            p="Oracle (the label source)",
            s="oracle_v3: 10 AlphaGenome-encoder members on chromosome folds (8 train / 1 validation / 1 test per member); synthetic labels are the mean of the 10; real held-out sets use the member that never saw them.",
            ev="K562 test r per member 0.931 (SD 0.005 across the 10 folds); HepG2 per-member mean 0.915. Verified against the saved folds.",
            st="supported",
            card="oracle",
            open="Only one oracle family; a second family was not built (no clean K562 Enformer or Borzoi oracle under the same folds exists).",
        ),
        dict(
            stage="1. Data and labels",
            p="Training pools",
            s="Nested prefixes inside each independent labelled pool (seed 42 + r); a size comparison varies only the amount of data. Genomic reservoir is capped at 314,981 labelled sequences.",
            ev="Audit of the legacy per-size pools (not nested, label distribution differs by size); the nested ladder reproduces the old scaling within 0.01.",
            st="supported",
            card="remeasure",
            open="Genomic cannot be tested above ~300k.",
        ),
        dict(
            stage="1. Data and labels",
            p="Validation set (early stopping, ensemble weights, proposer feedback)",
            s="Real genomic chr19 / 21 / X sequences (29,449), one set for every model and every reservoir. Plan: use the oracle_v3-relabelled version (V0c): the stored labels are a different oracle (r 0.983, offset +0.067) while every pool is oracle_v3.",
            ev="Five validation sets compared on three pools: the label fix is the only change that mattered; perturbed, diverse and mixed validation sets gave no reliable win. Reservoir ordering is unchanged across sets.",
            st="provisional",
            card="valvar",
            open="Adopt V0c as the default and re-check the ensemble weights it implies; model-guided and similarity-filtered validation sets not built.",
        ),
        dict(
            stage="1. Data and labels",
            p="Test sets",
            s="Held-out chr7 / chr13 genomic sequences (31,435), a designed out-of-distribution set, 32,000 random sequences; ensemble r on each.",
            ev="Used throughout; the designed set is where reservoirs differ most (and K562 r is lowest).",
            st="carried",
            card="ens",
            open="Real measured data for the designed and random sets are not available beyond Gosai's own.",
        ),
        # ---- search
        dict(
            stage="2. Architecture and hyperparameter search",
            p="Proposer model",
            s="Claude Sonnet 5.5.",
            ev="Equal-n ensemble within two standard errors of Opus.",
            st="supported",
            card="proposer",
            open="Future models not tested.",
        ),
        dict(
            stage="2. Architecture and hyperparameter search",
            p="Prompt",
            s="Default single-call prompt: guidance block of known-good pieces, ensemble-aware note, reference LegNet and axis list, validation-only feedback. No phase schedule, no explore or reflect styles.",
            ev=(
                "Guidance beats a bare prompt by 0.02 at 50 models. Explore-style prompt: +0.005 on genomic 30k but "
                + (
                    f"{P['mean']:+.4f} [{P['lo']:+.4f}, {P['hi']:+.4f}] pooled over {P['n']} paired runs on five reservoirs at 10k and 30k"
                    if P
                    else "pending"
                )
                + "."
            ),
            st="supported",
            card="promptgrid",
            open=COORD,
        ),
        dict(
            stage="2. Architecture and hyperparameter search",
            p="Rounds and proposals per round",
            s="10 rounds x 2 proposals = 20 models per run (30 rounds at 10k-30k in the earlier design).",
            ev=SATTXT,
            st="provisional",
            card="sched",
            open="Checked at 300k (three fresh searches tie the menu); the saturation curve itself was not remeasured there.",
        ),
        dict(
            stage="2. Architecture and hyperparameter search",
            p="Independent runs per reservoir",
            s=f"Pool several independent runs: the ledger evidence says {min(k90)} to {max(k90)} runs reach 90% of the gain (the current menu was distilled from 15 runs; 3 runs sit {min(gap3):.3f} to {max(gap3):.3f} below the ensemble of all).",
            ev="Run-count curves on 10 to 15 runs at 10k, 30k and 100k (genomic) and 30k (random, motif).",
            st="provisional",
            card="runcount",
            open="Whether the extra runs should be on different reservoirs or on different pools of one reservoir.",
        ),
        dict(
            stage="2. Architecture and hyperparameter search",
            p="Pool per run",
            s="Each run gets its own independently drawn labelled pool, nested across sizes inside it.",
            ev="Pooling three independent pools beats three seeds on one pool in 12 of 12 comparisons.",
            st="supported",
            card="indep",
            open="",
        ),
        dict(
            stage="2. Architecture and hyperparameter search",
            p="Reservoirs searched, and at which D",
            s="One search at 30k, on genomic data (the five-family search produced the same menu quality); carried to every other reservoir and size without a text hand-off.",
            ev=f"Per-reservoir searches match the retrained menu within 0.006 (30k) and 0.004 (100k); architecture source (genomic / random / mixed / five-reservoir searches) changes the result by at most {ASRC:.3f}; handing lower-D context to a 100k search changes the ensemble by +0.002 to +0.0035 at most.",
            st="supported",
            card="fam",
            open="One representative per family; six unseen families tested at menu level only; the 300k check (genomic) ties.",
        ),
        dict(
            stage="2. Architecture and hyperparameter search",
            p="What a proposal contains",
            s="A free-form network class plus its own training recipe (optimizer, learning rate, weight decay, batch size, epochs, schedule), at most 40M parameters, passing a build check (shape, finite output, forward under 8 s on CPU for a batch of 8).",
            ev="Tuned LegNet sits at the median of the searched models (rank 15 of 33 at 30k); free-form search adds diversity, not a better single architecture.",
            st="supported",
            card="arch",
            open="The CPU speed check is node-dependent and silently drops 0 to 6 menu members per run (architecture 2 always); analyses use common architectures.",
        ),
        # ---- training
        dict(
            stage="3. Training one model",
            p="Loss and augmentation (every model alike)",
            s="Loss is the mean of the squared errors of the forward-strand and the reverse-complement-strand predictions; random shift of up to 10 bp; gradient clipping 1.0; fp32 (fp16, bf16 and tf32 gave no speed-up). Validation and test predictions average the two strands.",
            ev="Reverse-complement averaging is +0.014 of the old-loop gap; precision A/B shows no gain.",
            st="supported",
            card="loop",
            open="Augmentation strength not varied.",
        ),
        dict(
            stage="3. Training one model",
            p="Epochs",
            s="Set by the proposal (menu: 60 to 100; the most common is 60 with OneCycle at batch 256; planned median 70).",
            ev=f"Training ends early: median {Cv['100000']['median_epochs'] if Cv else float('nan'):.0f} epochs actually run at 100k; {min(pat):.0%} to {max(pat):.0%} of models stop by patience rather than the epoch limit at every size.",
            st="supported",
            card="highd",
            open="No experiment varies the epoch budget itself; patience dominates it.",
        ),
        dict(
            stage="3. Training one model",
            p="Early stopping and time cap",
            s="Stop after 8 epochs without a validation-MSE gain of 1e-4; restore the best-validation weights. Per-model time cap in H100-equivalent minutes: 15 (10k), 25 (30k), 60 (100k), 120 (300k), 240 (600k), 360 (1M), 5x on a V100 in menu retrains; no cap was hit at 10k to 300k.",
            ev="Convergence audit of 393 retrained models; no cap hits.",
            st="supported",
            card="highd",
            open="Patience 8 and the 1e-4 threshold were not varied.",
        ),
        # ---- menu
        dict(
            stage="4. The menu and its retraining",
            p="Top models kept per run",
            s="Top 3 by validation among a run's first 20 models; identical code from different runs counts once; 33 architectures in the current menu.",
            ev="At 30k the top 8 of each of three runs equals the whole pool; at 100k keeping 3 to 5 is within 0.001 of keeping all 20.",
            st="supported",
            card="inst",
            open="Top-k was measured on runs of the recipe's own size; run-count dependence interacts with this.",
        ),
        dict(
            stage="4. The menu and its retraining",
            p="What is retrained at each D",
            s="The whole menu on that reservoir's pool at every size up to 100k; about 8 architectures chosen by the previous size's validation r at 300k and above.",
            ev=(
                "Choosing 8 of 33 by previous-size validation r loses "
                + ", ".join(f"{int(D) // 1000}k {v:+.4f}" for D, v in red8.items())
                + " (random choices of 8 lose 0.003 to 0.008)."
                if red8
                else "pending"
            )
            + (
                " Menu cost per seed (H100-hours): "
                + ", ".join(f"{int(k) // 1000}k {v:.1f}" for k, v in cost.items())
                + ", about D^0.91."
                if cost
                else ""
            ),
            st="supported",
            card="highd",
            open="Fresh searches at 300k tie the carried menu (genomic only).",
        ),
        dict(
            stage="4. The menu and its retraining",
            p="Seeds per architecture",
            s="Three at 100k and below (done); one at 300k and above for cost.",
            ev="Ensemble r by seeds on one 30k pool: " + seedtxt + ".",
            st="provisional",
            card="seeds",
            open="Replicate pools 2 and 3 for the ladder were not run.",
        ),
        dict(
            stage="4. The menu and its retraining",
            p="Reservoirs at a given D",
            s="Core set, the same at every D: genomic (up to 300k), random, motif (CT-enriched); evoaug_paper2023 as the augmentation family at 10k-100k; evo2 and six unseen families (motif shared / syntax core, Zoonomia, mutagenesis, GC-matched, dinucleotide shuffle) at 10k-30k (three at 100k) as checks.",
            ev="Coverage in the matrix below; genomic is capped by its pool, so 600k and 1M anchors use random and motif only.",
            st="provisional",
            card="coverage",
            open="Confirm the core set. Queued: evoaug at 300k and two unseen families at 100k. Not covered: evo2 and the other unseen families above 100k.",
        ),
        dict(
            stage="4. The menu and its retraining",
            p="Scaling beyond the tested sizes",
            s="Measure, do not extrapolate: random and motif anchors at 600k and 1M.",
            ev="A 10k-300k power law overpredicts the measured 600k and 1M points by about 0.01.",
            st="provisional",
            card="anchor",
            open="Genomic and evoaug anchors impossible or not run.",
        ),
        # ---- ensemble
        dict(
            stage="5. Ensembling",
            p="Method",
            s="Non-negative ElasticNet on z-scored validation predictions of every trained column (architectures x seeds) for that reservoir and size; weights refit per deployment. A non-negative least-squares version (within 0.0003) is used inside resampling loops.",
            ev="Beat greedy top-K by about +0.01; uncertainty weighting and diversity selection gave no gain.",
            st="carried",
            card="ens",
            open="",
        ),
        dict(
            stage="5. Ensembling",
            p="Screens and test-time averaging",
            s="Models whose predictions explode on any evaluation set (0.8% of free-form models) are dropped; every prediction averages forward and reverse-complement.",
            ev="Audit of 2,040 saved models.",
            st="supported",
            card="ens",
            open="",
        ),
        # ---- acquisition & targets
        dict(
            stage="6. Acquisition and targets",
            p="Selection model and rounds",
            s="Select once, with an 8-architecture ensemble, by highest ensemble mean (or SD); do not iterate.",
            ev="8 best, 8 random and the full menu select equally well; one round of 70k is at least as good as three of 23.3k; model-based selection beats random by about 0.01 (genomic) and 0.12 (designed set).",
            st="provisional",
            card="acqsel",
            open="One base, one candidate pool; the from-30k and from-300k arms are the real test.",
        ),
        dict(
            stage="6. Acquisition and targets",
            p="Other cell types and the differential",
            s="Retrain the same menu on HepG2 labels; for specificity train directly on the K562 - HepG2 label.",
            ev=(
                "Direct differential student beats subtracting two students by "
                + (
                    f"{HG['mean']:+.3f} [{HG['lo']:+.3f}, {HG['hi']:+.3f}] (9 of 9 runs, genomic test)"
                    if HG
                    else "pending"
                )
                + "; K562 menu transfers (HepG2 r 0.77 to 0.80)."
            ),
            st="supported",
            card="hg",
            open="Shared-trunk two-output model not tried.",
        ),
        dict(
            stage="6. Acquisition and targets",
            p="Label noise",
            s="Nothing to correct: real assay error costs 0.005 to 0.010 and keeps the ordering.",
            ev="Gosai-sized noise added to labels at 10k, 30k, 100k.",
            st="supported",
            card="noise",
            open="One seed per cell.",
        ),
        dict(
            stage="7. Measurement",
            p="Noise floor for 'no improvement'",
            s="About 0.005 in ensemble r (single run SD 0.009 to 0.011; pooled ensembles 0.002 to 0.005; an 8-model ensemble 0.006).",
            ev="Replicate spreads, nine-seed ensembles, between-pool SD 0.001 to 0.003.",
            st="supported",
            card="rep",
            open="",
        ),
    ]
    return rows


def coverage_html(d: dict) -> str:
    """Reservoir x D matrix of what has been retrained and ensembled (hand-maintained from the experiment directories)."""
    cols = ["10k", "30k", "100k", "300k", "600k", "1M"]
    M = [
        ("genomic", ["x", "x", "x", "x", "cap", "cap"]),
        ("random", ["x", "x", "x", "x", "x", "x"]),
        ("motif CT-enriched", ["x", "x", "x", "x", "x", "x"]),
        ("evoaug (paper 2023)", ["x", "x", "x", "x", "", ""]),
        ("evo2 (legacy per-size pools)", ["x", "x", "x", "", "", ""]),
        ("motif syntax core, mutagenesis, GC-matched", ["x", "x", "x", "", "", ""]),
        ("motif shared core, dinucleotide shuffle", ["x", "x", "x", "", "", ""]),
        ("Zoonomia (rebuilt)", ["x", "x", "", "", "", ""]),
    ]
    sym = {
        "x": "retrained and ensembled",
        "q": "queued on the cluster",
        "pool": "pool built, menu not retrained",
        "cap": "not possible: genomic pool has 314,981 sequences",
        "": "not run",
    }
    head = "<tr><th>Reservoir</th>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
    body = "".join(
        "<tr><td>"
        + n
        + "</td>"
        + "".join(
            f'<td class="cv cv-{v or "none"}" title="{sym[v]}">{ {"x": "&#10003;", "q": "queued", "pool": "pool", "cap": "cap", "": "&ndash;"}[v] }</td>'
            for v in vs
        )
        + "</tr>"
        for n, vs in M
    )
    return (
        '<div class="tblwrap"><table class="cov"><thead>'
        + head
        + "</thead><tbody>"
        + body
        + "</tbody></table></div>"
        '<p class="note">&#10003; retrained with the full 33-architecture menu (genomic 300k, random 300k and motif 300k on the nested replicate pool 1; 600k and 1M with 17 shards each). evoaug 300k is one seed on the nested pool; "cap" = impossible; the three-replicate unseen families use pools r1 to r3.</p>'
    )


def extra_assumption_rows(d: dict) -> list[dict]:
    o = d.get("oct22", {})
    H = (o.get("hg") or {}).get("summary", {}).get("direct_minus_two_students", {}).get("genomic")
    rows = [
        dict(
            grp="E. Targets, acquisition and robustness",
            a="The pipeline works unchanged for other cell types and for the differential target.",
            why="The library is for cell-type specificity, not one cell line.",
            test="K562-tuned menu retrained on HepG2 and on K562 minus HepG2 labels, nine pools.",
            status="supported",
            ev=(
                "Direct differential student +"
                + f"{H['mean']:.3f}"
                + " over subtracting two students, 9 of 9 runs; HepG2 r 0.77 to 0.80."
                if H
                else "see card"
            ),
            card="hg",
        ),
        dict(
            grp="E. Targets, acquisition and robustness",
            a="Measurement noise in real labels does not change the conclusions.",
            why="Every number here is against noise-free oracle labels.",
            test="Gosai-sized and doubled label noise at 10k, 30k and 100k.",
            status="supported",
            ev="Costs 0.005 to 0.010 (0.03 at twice the error); ordering and curve shape unchanged.",
            card="noise",
        ),
        dict(
            grp="E. Targets, acquisition and robustness",
            a="A reduced selection ensemble and one acquisition round are enough.",
            why="Intermediate rounds would otherwise need the full menu at every size.",
            test="Full vs 8-best vs 8-random selection; 3 x 23.3k vs 1 x 70k.",
            status="provisional",
            ev="All three selection ensembles tie; three rounds are no better than one.",
            card="acqsel",
        ),
        dict(
            grp="E. Targets, acquisition and robustness",
            a="Conclusions extrapolate beyond 300k.",
            why="The library is 3.2M sequences.",
            test="Measured 600k and 1M points for random and motif.",
            status="provisional",
            ev="Still rising, about 0.01 below the 10k-300k power law.",
            card="anchor",
        ),
    ]
    return rows
