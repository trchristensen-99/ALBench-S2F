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
    ("2026-10-22 14:30", "Independent pools: three independent draws beat three seeds on one pool in all 12 comparisons (menu, 30k). Per-reservoir search again matches the retrained menu at 100k. Models trained on several reservoirs ensemble better than genomic-only. Nested genomic ladder lands within 0.01 of the old ladder. Opus +0.005 over Sonnet 5.5, not significant. Added a setup-choices table at the top."),
    ("2026-10-21 21:20", "All 16 context cells, 15 family runs, 300k menu (3 seeds) and the baselines are done. Results: no benefit from cross-D context; a fresh search per reservoir matches the retrained menu at 30k (within 0.006); one instance saturates faster at 100k than at 30k; menu-ensemble scaling in the corrected loop reaches 0.863 at 300k (genomic). Only the Opus and Sonnet 4.5 proposer cells still run."),
    ("2026-10-21 09:40", "Experiment audit: ctx prompts carry the context (T/J/X yes, none no), family cells use the right pools, 300k models were not time-capped; 16 of 2,040 models had exploded predictions on some eval set. Dispatcher: LLM cap 14 to 22, default-queue overflow now goes to slow_nice, 300k seeds 2-3 queued."),
    ("2026-10-21 09:30", "Overnight results: guidance beats wide-open for the proposer (N is 0.02 behind at 50 models; P = W); prior-box baselines match the LLM in HP-only; context hand-off shows no gain yet; menu ranks agree across reservoirs near the retest ceiling except random; menu at 300k is 23/33 done."),
    ("2026-10-21 05:25", "Menu retraining at 300k queued on idle H100s (6 shards) to test whether 100k rankings carry to 300k."),
    ("2026-10-21 00:55", "Menu retrained on random / motif / evo2 / evoaug at 30k: rank agreement with genomic is 0.4-0.6, about the genomic retest ceiling (0.55). Second-seed retests for each family are queued."),
    ("2026-10-20 23:00", "First version. Ten evidence figures, sixteen decisions. Context-transfer (16 cells), reservoir-family AutoResearch (12), "
                         "menu-by-reservoir retraining (12) and the repaired baselines launched or still running."),
]


def f3(x, n=3):
    return "n/a" if x is None or x != x else f"{x:.{n}f}"


def sgn(x, n=3):
    return "n/a" if x is None or x != x else f"{x:+.{n}f}"


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
    n_fam = {f: sum(v for k, v in fam.items() if k.startswith(f + "|")) for f in ("random", "motif_ct_enriched", "evo2", "evoaug_paper2023")}
    pa = d.get("prompt_arms", {}).get("summary", {})
    pe = {k.split()[0]: v.get("ens_at_n", []) for k, v in pa.items()}
    mean = lambda v: sum(v) / len(v) if v else float("nan")
    bp = d.get("baselines_progress", {})
    bt = {k: sum(v) for k, v in bp.items()}
    bv = d.get("baselines_vs_llm", {})
    rt = M.get("retest", {})
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
        dict(id="loop", group="Training and evaluation", status="supported", figs=["loop_decomposition.png"],
             q="Why did earlier scaling runs report r = 0.69-0.70 for a LegNet that scores 0.73 in the architecture loop?",
             choice="Train every model, from every source, in one common loop and report RC-averaged predictions.",
             points=[f"Same network, same HPs, three configs: reported test r {f3(min(nat))}-{f3(max(nat))} in the old loop vs {f3(min(arch))}-{f3(max(arch))} in the arch loop ({sgn(L.get('total_gap'))} on average).",
                     f"Two parts: reverse-complement averaging at test time {sgn(L.get('rc_avg'))}; the loop itself {sgn(L.get('loop_fwd_matched'))} on matched forward-only predictions.",
                     f"Inside the loop, the per-sample crop-shift ({sgn(L['flips'].get('A_flip_shift'))}) and fp16 ({sgn(L['flips'].get('A_flip_amp'))}) cost accuracy; early-stop rule, gradient clip and the time cap do not matter (within noise)."],
             caveat="Every scaling-law number measured before Oct 20 used the old loop and forward-only predictions. The shape of the curves probably survives; the absolute levels do not. The retrained-menu ensembles re-measure the ladder in the common loop (see the scaling card below), and a nested genomic pool reproduces the old curve within about 0.01.",
             change="A config where the old loop wins would mean the ingredients interact with the architecture and we need per-architecture loops."),
        dict(id="arch", group="Training and evaluation", status="supported", figs=["arch_vs_legnet.png"],
             q="Does a freely-designed architecture beat a well-tuned LegNet once both are trained the same way?",
             choice="Do not claim an architecture advantage for free-form search. Its value is diversity for the ensemble and a wide, assumption-free search.",
             points=[f"LegNet-eff HP-search winners: {', '.join(f3(v) for v in d['arch_vs_legnet']['LegNet-eff configs'])}. Top-3 free-form architectures: {', '.join(f3(v) for v in d['arch_vs_legnet']['free-form architectures'])}.",
                     "Single seeds scatter by about +-0.005, so these are indistinguishable.",
                     f"Retraining the 33-model menu three times each puts the reference LegNet at rank {rr.get('10k', '?')} / {rr.get('30k', '?')} / {rr.get('100k', '?')} of 33 at 10k / 30k / 100k. With one or two seeds it had looked like 2nd-3rd at 30k and 100k, so rankings within this pack are seed-noise dominated and no architecture, LegNet included, is distinguishable from the pack."],
             caveat="Genomic reservoir at 30k only; 3 seeds per architecture."),
        dict(id="menu", group="Dataset-size ladder", status="supported", figs=["menu_across_D.png"],
             q="Can we pick architectures at a small D and reuse them at a larger D?",
             choice="No. Retrain the whole candidate pool at every D and ensemble there; do not pre-select the top few at 10k.",
             points=[f"Rank correlation of the same {M.get('n_entries', 33)} architectures between D: 10k vs 30k, validation {f3(sp['spearman_10k_30k'].get('val'), 2)} / test {f3(sp['spearman_10k_30k'].get('test'), 2)}; 30k vs 100k, validation {f3(sp['spearman_30k_100k'].get('val'), 2)} / test {f3(sp['spearman_30k_100k'].get('test'), 2)}.",
                     f"Ensemble of the whole retrained menu: {', '.join(f3(v) for v in ens.get('all', []))} at 10k / 30k / 100k. Top-5 chosen at 10k and retrained: {', '.join(f3(v) for v in ens.get('top5_at_10k', []))}, i.e. {', '.join(sgn(a - b) for a, b in zip(ens.get('top5_at_10k', []), ens.get('all', [])))} against the whole pool. HP-only search (old loop): {', '.join(f3(v) for v in ens.get('hp_only', []))}.",
                     (f"Within-D retest ceiling (one seed against the mean of the others): 10k {rt['10k']:.2f}, 30k {rt['30k']:.2f}, 100k {rt['100k']:.2f}. The 10k-to-30k agreement is far below these, so the ranking really changes; 30k-to-100k sits at them." if all(rt.get(k) is not None for k in ("10k", "30k", "100k")) else "Retest ceilings pending."),
                     "Retrainings per architecture: " + ", ".join(f"{k} {v}" for k, v in M.get("seeds_per_D", {}).items()) + ".",
                     ("On the nested genomic pool (one seed per size) the agreement is 10k-30k " + f"{nl['spearman'].get('10k-30k', float('nan')):.2f}, 30k-100k {nl['spearman'].get('30k-100k', float('nan')):.2f}, 100k-300k {nl['spearman'].get('100k-300k', float('nan')):.2f}" + ", against 0.13 / 0.66 / 0.43 on the old three-seed ladder. The pattern does not repeat, so rank agreement across sizes is low to moderate everywhere and mostly noise-limited; it is not evidence that small-D picks transfer or fail to." if nl.get("spearman") else "Nested-pool rank agreement pending.")],
             caveat="Three retrainings per architecture at every D. The practical cost of choosing the top 5 at 10k anyway is small (a few thousandths in ensemble r), so the case for retraining everything is about robustness, not size. The HP-only comparison mixes harnesses.",
             change="Three seeds on the nested pools would settle whether any rank carries over between sizes; until then retraining everything is the safe choice."),
        dict(id="menufam", group="Dataset-size ladder", status="supported", figs=["menu_families.png"],
             q="Do the same architectures do well on random, motif, evo2 and evoaug data, or does each reservoir need its own search?",
             choice="Menu is reservoir-agnostic: retrain the same pool on each reservoir's data.",
             points=["All " + str(mf.get("n_menu", 33)) + " architectures retrained at 30k on each reservoir, three seeds each.",
                     "Rank agreement with genomic (Spearman of test r): " + (", ".join(f"{k.replace('_', ' ')} {v:.2f}" for k, v in mf.get("spearman_vs_genomic", {}).items()) or "pending") + (f". Retest ceiling (genomic, one retraining vs the mean of the others): {mf['retest_spearman_genomic']:.2f}." if "retest_spearman_genomic" in mf else "."),
                     "Menu ensemble test r (genomic eval): " + ", ".join(f"{r['res'].replace('_', ' ')} {r['ensemble']:.3f}" for r in mf.get("rows", [])) + ". Motif-enriched is best, random worst; the best single model trails each ensemble by 0.02-0.03.",
                     "Rankings agree across reservoirs at about the retest ceiling for motif, evo2 and evoaug. Random is the exception (0.29): what works on random sequences differs more."],
             caveat="30k only; the retest ceiling comes from genomic. Runs on V100 as well as H100 with GPU-scaled time caps.",
             change="If a larger-D check shows random-reservoir rankings diverging further, random needs its own pool."),
        dict(id="menu300", group="Dataset-size ladder", status="supported", figs=["menu_300k.png"],
             q="Can the architecture pool chosen at 100k be reused at 300k, or does 300k need its own retraining?",
             choice="Retrain the whole pool at 300k. Rankings at 100k carry over only partly.",
             points=[f"All {m3.get('done', 0)} architectures retrained at 300k, three seeds each (none hit the 120-minute cap; median 32 minutes)."]
                    + ([f"Rank agreement 100k vs 300k: test {m3['spearman_test_100k_300k']:.2f}, validation {m3['spearman_val_100k_300k']:.2f}; 30k vs 300k test {m3['spearman_test_30k_300k']:.2f}; {m3.get('top5_overlap', '?')} of the top 5 are shared. The within-D retest ceiling at 300k is {m3.get('retest') or float('nan'):.2f}, clearly above this, so rankings keep changing with D."] if "spearman_test_100k_300k" in m3 else ["Rank agreement fills in as architectures finish."])
                    + ([f"Ensemble of the retrained menu at 300k: genomic {m3['ensemble']['genomic']:.3f}, out-of-distribution {m3['ensemble']['ood']:.3f}."] if m3.get("ensemble") else []),
             caveat="Genomic reservoir only at 300k. Both sizes use three-seed means.",
             change="If a search run at 300k found architectures well outside this pool, retraining the old pool would not be enough there."),
        dict(id="indep", group="Search and pooling", status="supported", figs=["independent_pools_30k.png"],
             q="Should every independent search run get its own labelled pool, or should all runs share one?",
             choice="Give each independent run its own pool (a separate labelled draw), nested across sizes inside it. Keep one nested corpus for the final retrain.",
             points=["Menu test at 30k (33 architectures): three seeds on one pool (A) against three independent pools (B), equal model counts (99 each). Pooled ensemble, genomic evaluation, A vs B: " + "; ".join(f"{k.replace('_', ' ')} {v['A']['pooled']['genomic']:.3f} vs {v['B']['pooled']['genomic']:.3f} ({v['B']['pooled']['genomic'] - v['A']['pooled']['genomic']:+.3f})" for k, v in ind.items()) + ".",
                     "Independent pools win in every reservoir on all three evaluation sets (12 of 12 comparisons). Out-of-distribution gain " + ", ".join(f"{v['B']['pooled']['ood'] - v['A']['pooled']['ood']:+.3f}" for v in ind.values()) + "; random-sequence gain " + ", ".join(f"{v['B']['pooled']['random_32k'] - v['A']['pooled']['random_32k']:+.3f}" for v in ind.values()) + ".",
                     "Spread of one replicate's ensemble across the three replicates (SD): seeds on one pool " + ", ".join(f"{__import__('statistics').stdev(v['A']['ens']):.4f}" for v in ind.values()) + "; separate pools " + ", ".join(f"{__import__('statistics').stdev(v['B']['ens']):.4f}" for v in ind.values()) + ". Shared-pool replicates understate run-to-run variance.",
                     "Architecture rankings repeat about equally well across seeds and across pools (Spearman " + ", ".join(f"{v['A']['rank']:.2f}/{v['B']['rank']:.2f}" for v in ind.values()) + " for seeds/pools), so ranking noise is mostly training noise.",
                     (f"AutoResearch at genomic 100k: three runs with their own pools pool to {i100['independent']['pooled']:.3f} (single runs {min(i100['independent']['single']):.3f}-{max(i100['independent']['single']):.3f}) against {i100['shared']['pooled']:.3f} for runs sharing the old 100k pool (single runs {min(i100['shared']['single']):.3f}-{max(i100['shared']['single']):.3f}). This comparison is confounded: the shared pool was the old, biased genomic 100k set." if i100 else "100k comparison pending.")],
             caveat="The clean test is the 30k menu comparison (same code, same pools, only the data draw differs). The 100k search comparison points the same way but cannot separate independence from the old pool's bias. The final retrain at each size should still use one nested corpus.",
             change="If nine seeds on one pool matched three pools of three seeds, the gain would be model count and not data diversity."),
        dict(id="mix", group="Search and pooling", status="provisional", figs=["family_union.png"],
             q="Should the final ensemble use models trained on several reservoirs, or on genomic data only?",
             choice="Use several. Genomic alone is the weakest single choice for genomic held-out sequences; genomic plus motif already captures nearly all of the union's gain.",
             points=["Ensemble test r at 30k (genomic held-out / out-of-distribution / random sequences): " + "; ".join(f"{k} {un[k]['genomic']:.3f} / {un[k]['ood']:.3f} / {un[k]['random_32k']:.3f}" for k in ("genomic only", "motif only", "random only", "genomic + motif", "all four non-genomic", "all five") if k in un) + ".",
                     ("Dropping motif from the union costs the most on genomic sequences (" + f"{un['all five']['genomic']:.3f} to {un['all five minus motif']['genomic']:.3f}" + "); dropping genomic costs nothing (" + f"{un['all five']['genomic']:.4f} to {un['all five minus genomic']['genomic']:.4f}" + ")." if un else "Union results pending."),
                     "On out-of-distribution and random sequences the best single reservoir (motif or random) beats the union, so the union is a compromise: better than genomic alone everywhere, not the best on every evaluation set."],
             caveat="These are ensembles of models that were each trained on one reservoir, not models trained on a mixed corpus. Which sequences to put in the library is a corpus-mixing question that this does not answer. The union also has up to five times as many models.",
             change="A mixed-corpus training run (for example genomic plus motif at equal total size) would show whether mixing at the data level matches mixing at the model level."),
        dict(id="inst", group="Search and pooling", status="supported", figs=["instances_and_topk.png"],
             q="How many models should one AutoResearch instance train, how many should we keep, and how many instances do we need?",
             choice="About 20-30 models per instance, keep the top 8 by validation, pool several independent instances.",
             points=[f"One instance's ensemble saturates around 20-40 models (mean of runs: {', '.join(f3(v) for v in I.get('mean', []))} at n = {I.get('n')}).",
                     f"Pooling the top-k of three instances: k = 1: {f3(top5[0])}, k = 3: {f3(top5[2])}, k = 8: {f3(top5[4])}, all: {f3(top5[6])}. Single instances with all their models: {f3(min(T.get('singles', [0])))}-{f3(max(T.get('singles', [0])))}.",
                     f"Between-instance spread of the final ensemble: SD {f3(I.get('sd_between_runs'), 4)}.",
                     ("At 100k on independent pools (20 models per run, three runs pooled), keeping the top k from each run gives genomic " + ", ".join(f"k={k}: {v:.4f}" for k, v in tk100.get("genomic", {}).items() if k in ("1", "3", "5", "8", "20")) + "; motif, random and evoaug flatten by k = 2-3 as well. Three to five per run is enough at 100k." if tk100.get("genomic") else "100k top-k pending.")],
             caveat="Three Sonnet-5.5 instances at 30k; three per reservoir at 100k."),
        dict(id="rep", group="Search and pooling", status="supported", figs=["replicates.png"],
             q="Is pooling independent runs actually better than a single run, and by how much?",
             choice="Always run several independent instances per reservoir and pool them.",
             points=[f"Three independent runs, 3 reservoirs x 3 D: the pooled ensemble beats the best single run in {wins} of {len(rep)} cells (gain over the best single {sgn(min(gains) if gains else float('nan'))} to {sgn(max(gains) if gains else float('nan'))}; over the average single {sgn(min(meansing) if meansing else float('nan'))} to {sgn(max(meansing) if meansing else float('nan'))}).",
                     "Single-run SD is about 0.009, the same size as many between-strategy gaps: any single-run comparison below that is noise."],
             caveat="HP-only runs in the old loop. The replicate structure (not the absolute r) is what carries over."),
        dict(id="llm", group="Proposer and prompt", status="provisional", figs=["llm_vs_classical.png"],
             q="Is an LLM proposer better than classical samplers, and if so is it because of search or because of its prior?",
             choice="Use the LLM with 2 proposals x 30 rounds, but do not claim it out-searches classical methods.",
             points=[f"Cadence study, genomic: LLM 2x30 {f3(cadg('genomic', 'llm2x30')[0])}, 6x10 {f3(cadg('genomic', 'llm6x10')[0])}, 1x60 {f3(cadg('genomic', 'llm1x60')[0])} (incomplete), classical multi6 {f3(cadg('genomic', 'multi6')[0])}; the LLM led by +0.02 to +0.045, but its proposal quality is flat from round 0, so the edge is its prior.",
                     "Repaired baselines (6 runs each, HP-only, genomic 30k, ensemble of the first 20 models): " + "; ".join(f"{lab} mean {mean(bv.get(k, [])):.3f} (range {min(bv[k]):.3f}-{max(bv[k]):.3f}, n = {len(bv[k])})" for k, lab in (("random_broad", "random in the full box"), ("prior_marginal", "random in the LLM-like prior box"), ("prior_tpe", "TPE in that box"), ("llm_first30", "LLM 2x30 first 20")) if bv.get(k)) + ".",
                     "Sampling the box the LLM tends to propose from, with no LLM at all, matches the LLM in this HP-only setting; plain random sampling in the full box is about 0.004-0.005 behind, inside the single-run SD of 0.009."],
             caveat="HP-only, one reservoir, one size; n = 5-6 runs per baseline and 2 LLM runs with 20 models. The case for the LLM therefore rests on what the baselines cannot do: write new architectures, and propose with knowledge of what has been tried.",
             change="If baselines also matched the LLM on the free-architecture task (which they cannot run), the LLM would be unnecessary."),
        dict(id="kb", group="Proposer and prompt", status="supported", figs=["kb_prompts.png"],
             q="Does the knowledge base or a particular prompt wording make the proposer better?",
             choice="Keep the default prompt; do not spend more budget on prompt wording.",
             points=[f"Knowledge base on vs off, 5 cells: mean {sgn(kb.get('mean'))}, individual differences {', '.join(sgn(x) for x in kb.get('diffs', []) if x is not None)}.",
                     "Six prompt styles x 2 reps at genomic 30k: explore and reflect sit about +0.01 above the default, critic below, but each style's two reps straddle the single-run noise."],
             caveat="HP-only harness. Effects below about 0.01 are not detectable at one run per cell."),
        dict(id="prompt", group="Proposer and prompt", status="supported", figs=["prompt_arms.png"],
             q="Should the free-architecture proposer be given guidance (known-good pieces, a reference LegNet, a build-then-explore schedule) or be left wide open?",
             choice="Give it the known-good guidance and the ensemble note, without a phase schedule (arm W). Leaving it wide open (N) is measurably worse.",
             points=["Arms: P = guidance + reference + build-on-what-works for 6 rounds then explore; W = guidance only; N = nothing. Three independent runs per arm.",
                     f"At equal n = {d.get('prompt_arms', {}).get('n_eq', '?')} models, mean ensemble test r: P {f3(mean(pe.get('P', [])))}, W {f3(mean(pe.get('W', [])))}, N {f3(mean(pe.get('N', [])))}; N is {sgn(mean(pe.get('N', [])) - mean(pe.get('W', [])))} from W, about four times the between-run SD of 0.005.",
                     "Top-5 validation r per run: P 0.823-0.833, W 0.830-0.833, N 0.793-0.817. The build-then-explore phases add nothing over plain guidance."],
             caveat="Sonnet 5.5, genomic 30k only; 3 reps per arm. The wide-open preference is kept as the default for everything the guidance does not cover (architecture families, depth, width).",
             change="If N matched W on another reservoir, we would drop the guidance there too."),
        dict(id="proposer", group="Proposer and prompt", status="supported", figs=["proposer_models.png"],
             q="Which model should write the architectures?",
             choice="Sonnet 5.5. Opus 5.5 is about +0.005 ahead, which is not enough to justify its cost; Sonnet 4.5 is slightly behind.",
             points=["Same prompt (build-on-what-works), genomic 30k, three runs each for Sonnet 5.5, Sonnet 4.5 and Opus 5.5; all nine runs are complete.",
                     f"Ensemble test r at equal n = {d.get('proposer', {}).get('n_eq', '?')} models (mean over runs): " + "; ".join(f"{k} {mean(v.get('ens_at_n', [])):.4f}" for k, v in pr.items() if v.get("ens_at_n")) + ". Run-to-run SD is about 0.002-0.005, so Opus is ahead by under two standard errors.",
                     "Models per run: " + "; ".join(f"{k.replace('|', ' run ')} {v}" for k, v in d.get("proposer", {}).get("cells", {}).items()) + "."],
             caveat="Genomic 30k only. A +0.005 gain would matter if confirmed on another reservoir.",
             change="A gain of 0.01 or more for Opus at equal models on a second reservoir would justify its higher token cost."),
        dict(id="ctx", group="Dataset-size ladder", status="supported", figs=["context_transfer.png"],
             q="When AutoResearch moves from 30k to 100k, should the proposer see what worked at 30k?",
             choice="No. Start every D's search from the same prompt; handing results across sizes does not help.",
             points=["Four arms at genomic 100k, 15 rounds x 2 proposals, four independent runs each: no context, top architectures by validation from 30k (T), T plus a trajectory summary (J), top architectures from 10k as the stale control (X).",
                     "Ensemble test r at " + str(c100.get("n_eq", 30)) + " models per run (mean, runs): " + "; ".join(f"{k} {mean(c100[k]):.4f} (n = {len(c100[k])})" for k in ("none", "T", "J", "X") if c100.get(k)) + ".",
                     "T and J are within +0.002 of no context. The stale 10k context scores highest (about +0.0035, under 2 standard errors), which is the opposite of what useful context would predict. Earlier 30k pilot and the replicate of the one 100k gain also showed nothing."],
             caveat="One J run and one T run are missing from the counts because the exploded-prediction screen removed models. Genomic reservoir only.",
             change="A gain of 0.01 or more for T or J across runs would justify the hand-off."),
        dict(id="fam", group="Search and pooling", status="supported", figs=["families.png", "families_100k.png"],
             q="Does a separate AutoResearch search on each reservoir beat reusing the genomic-derived architecture pool?",
             choice="Reuse the pool: search on genomic data and retrain the pool on each reservoir's data. A fresh per-reservoir search matches it at 30k and at 100k but costs LLM budget.",
             points=["30k: five reservoirs x three independent runs (60 models each, guidance-only prompt). Mean validation r of each run's 5 best models: " + "; ".join(f"{k.replace('_', ' ')} {min(v):.3f}-{max(v):.3f}" for k, v in d.get("family", {}).get("top5", {}).items()) + ". Runs on the same reservoir agree to about 0.003.",
                     "30k pooled ensemble test r (top 8 of 3 runs / all models / retrained menu): " + "; ".join(f"{k.replace('_', ' ')} {v['ar_top8']:.3f} / {v['ar_all']:.3f} / {v['menu']:.3f}" for k, v in fe.items() if v.get("menu") is not None) + ". Search and menu are within 0.006 everywhere.",
                     "100k, each run on its own pool, 20 models per run, top 8 of 3 pooled, against the retrained menu (genomic eval): " + "; ".join(f"{k.replace('_', ' ')} {v['ar_top8']:.3f} vs {v['menu']:.3f}" for k, v in f100.items() if v.get("menu") is not None) + ". Again within 0.004, with the menu a little ahead out of distribution.",
                     "So a search on one reservoir finds nothing that the retrained pool does not already have, on any of the four reservoirs tested, at either size."],
             caveat="Tested on one representative member per family (motif_ct_enriched, evoaug_paper2023, random, genomic); other members of each family were not tested. The menu was distilled from genomic searches at 30k, so this is a transfer result.",
             change="If other members of a family (for example motif_jaspar or motif_grammar) showed a gap between search and menu, the family representatives would be too narrow."),
        dict(id="ens", group="Ensembling", status="carried", figs=[],
             q="How should the pooled candidates be combined?",
             choice="Non-negative ElasticNet on the validation predictions of all models (enet_all_pos), z-scored per model so that one diverged model cannot zero the fit.",
             points=["Beat greedy top-K selection by about +0.01 in the earlier ensemble-construction study; uncertainty weighting and diversity (DPP) selection gave no gain.",
                     "The z-scored version was added after one exploded model (predictions ~1e20) zeroed the raw ElasticNet.",
                     "Audit of 2,040 saved free-form and menu models: 16 (0.8%) output |prediction| > 1e3 on at least one evaluation set while validating normally (the worst reached 1e24). The architecture loop and menu retraining only check the validation predictions, so the ensemble step must screen every evaluation set before fitting; this screen is now in the ledger generator and has to be in the final pipeline."],
             caveat=f"Earlier experiments are in the previous artifact: {OLD_ARTIFACT}"),
        dict(id="sched", group="Dataset-size ladder", status="provisional", figs=["saturation_100k.png"],
             q="How many rounds should each D get as the ladder climbs?",
             choice="Fewer rounds at larger D: about 20 models (10 rounds of 2) at 100k and above, 60 models at 10k-30k.",
             points=(([f"Ensemble gain of one instance from 6 to 30 models: {sat['mean100'][-1] - sat['mean100'][0]:.4f} at 100k ({sat.get('n100', '?')} instances) vs {sat['mean30'][-1] - sat['mean30'][0]:.4f} at 30k ({sat.get('n30', '?')} instances).",
                       f"The last 12 models (18 to 30) add {sat['mean100'][-1] - sat['mean100'][3]:.4f} at 100k vs {sat['mean30'][-1] - sat['mean30'][3]:.4f} at 30k, at or below single-run noise (about 0.003)."]) if sat.get("mean100") else ["Saturation curve pending."])
                    + ["Pooling several instances adds more than extra rounds in one instance (see the pooling card)."],
             caveat="300k not measured; saturation is judged by the ensemble of one instance only. The 30k curve has three instances and the 100k curve fourteen.",
             change="If a 300k instance were still climbing at 20 models, the schedule would need a longer 300k stage."),
        dict(id="remeasure", group="Dataset-size ladder", status="provisional", figs=["menu_scaling.png"],
             q="How does accuracy scale with dataset size when everything is trained in the corrected loop?",
             choice="Report scaling from the retrained menu ensembles (33 architectures, up to three seeds), not from the old single-loop HP-search cells.",
             points=[f"Genomic reservoir, ensemble test r at 10k / 30k / 100k / 300k: {g(10):.3f} / {g(30):.3f} / {g(100):.3f} / {g(300):.3f}. Gain per doubling: {per_dbl(10, 30):.3f}, {per_dbl(30, 100):.3f}, {per_dbl(100, 300):.3f}, so returns diminish but the curve is not flat at 300k.",
                     f"Out-of-distribution sequences, same sizes: {g(10, 'ood'):.3f} / {g(30, 'ood'):.3f} / {g(100, 'ood'):.3f} / {g(300, 'ood'):.3f}; this curve keeps accelerating (gain per doubling {per_dbl(10, 30, 'ood'):.3f}, {per_dbl(30, 100, 'ood'):.3f}, {per_dbl(100, 300, 'ood'):.3f}).",
                     "Other reservoirs at 10k / 30k / 100k (genomic eval): " + "; ".join(f"{k.replace('_', ' ')} {ms[k]['10']['genomic']:.3f} / {ms[k]['30']['genomic']:.3f} / {ms[k]['100']['genomic']:.3f}" for k in ms if k != "genomic" and all(x in ms[k] for x in ("10", "30", "100"))) + ". Motif-enriched is highest on genomic sequences; random is lowest there but highest out of distribution at 100k (" + (f"{ms['random']['100']['ood']:.3f}" if "random" in ms and "100" in ms["random"] else "n/a") + ")."],
             caveat=("The old genomic pools were not nested and differed in label distribution by size. Repeating the ladder on a properly nested genomic pool (one seed) gives " + "; ".join(f"{k}k {nl['ens'][k]['genomic']:.3f}" for k in ("10", "30", "100", "300") if k in nl.get("ens", {})) + ", within about 0.01 of the old ladder at every size, and the out-of-distribution curve agrees to within 0.02. The old scaling conclusions stand. " if nl.get("ens") else "") + "These are menu ensembles, not AutoResearch cells, and non-genomic reservoirs have one seed at 10k and 100k. The old HP-only numbers (0.686 / 0.743 / 0.780 on genomic) are lower at every size.",
             change="A full AutoResearch re-run per size would be needed only if the menu is thought to understate what search finds at 100k and 300k."),
        dict(id="budget", group="Compute and limits", status="supported", figs=["llm_budget.png"],
             q="Can we afford the remaining programme within the LLM usage limit?",
             choice="Dispatcher pauses on the session limit, resumes after the reset, and caps concurrent LLM cells at 14.",
             points=["One usage-limit stop on Oct 21 at 11:03 with 18 concurrent LLM cells; the dispatcher paused, probed after the 12:10 reset and resumed at 12:12 without help. The cap is now 15 cells (about 0.65M tokens an hour), which leaves room for interactive use.",
                     f"HPC cells used about {B.get('raw_tokens', 0) / 1e6:.0f}M raw tokens ({B.get('output_tokens', 0) / 1e6:.1f}M output) over the last {B.get('hours', '?')} active hours ({B.get('calls', '?')} calls).",
                     "The limit was hit at 20:10 on Oct 20 (HPC 3.0M plus laptop 1.7M raw tokens in the window). Remaining programme is roughly 2,100 calls."],
             caveat="Cap is weighted and unpublished; numbers are raw token counts and do not include cache reads."),
    ]
    return cards


CSS = """
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
.choices table{width:100%;border-collapse:collapse;font-size:13.5px;min-width:720px}
.choices th{text-align:left;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);padding:10px 12px;border-bottom:1px solid var(--line)}
.choices td{vertical-align:top;padding:10px 12px;border-bottom:1px solid var(--line);max-width:46ch}
.choices tr:last-child td{border-bottom:0}.choices td:first-child{font-weight:600;white-space:nowrap}
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
    pa = {k.split()[0]: v.get("ens_at_n", []) for k, v in d.get("prompt_arms", {}).get("summary", {}).items()}
    m = lambda v: sum(v) / len(v) if v else float("nan")  # noqa: E731
    gain = [v["B"]["pooled"]["genomic"] - v["A"]["pooled"]["genomic"] for v in ind.values()]
    rows = [
        dict(aspect="Lower D informing higher D", choice="No hand-off. Retrain the whole candidate pool at every D and start each D's search from the same prompt.",
             ev=("Context from 30k or a trajectory summary changes 100k ensembles by at most +0.002 (the useless 10k context scored highest, +0.0035). Picking the top 5 at 10k and retraining costs 0.001-0.006 against retraining everything. Architecture rankings agree across sizes only weakly."), status="supported", card="ctx"),
        dict(aspect="Rounds per D", choice="About 30 rounds x 2 models at 10k-30k, about 10 rounds x 2 at 100k and above.",
             ev=(f"One instance gains {sat['mean30'][-1] - sat['mean30'][0]:.4f} from 6 to 30 models at 30k and {sat['mean100'][-1] - sat['mean100'][0]:.4f} at 100k; the last 12 models add at or below noise at 100k. 300k not measured." if sat.get("mean100") else "pending"), status="provisional", card="sched"),
        dict(aspect="Proposals per round", choice="2 per round.", ev="Cadence study (one run per cell): 2 x 30 beat 6 x 10 and 1 x 60 on genomic and motif, tied on random. Per-round proposal quality is flat, so the edge is the LLM prior, not feedback.", status="provisional", card="llm"),
        dict(aspect="Independent runs per reservoir", choice="Three, pooled.", ev="Pooling three runs beats the best single run in 8 of 9 cells and the average run in 9 of 9; between-run SD about 0.005. More than three has not been tested.", status="provisional", card="rep"),
        dict(aspect="Independent pool per run", choice="Yes: each run gets its own labelled pool, nested across sizes inside it.",
             ev=("At equal model counts, pooling three independent pools beats pooling three seeds on one pool by " + ", ".join(f"{g:+.3f}" for g in gain) + " (genomic, random, motif, evoaug), 12 of 12 comparisons positive across evaluation sets." if ind else "pending"), status="supported", card="indep"),
        dict(aspect="Top models kept per run", choice="Top 5 by validation from each run.",
             ev=("At 30k the top 8 of each of three runs equals the whole pool. At 100k, k = 3 to 5 is within 0.001 of keeping all 20." if tk100.get("genomic") else "At 30k the top 8 equals the whole pool."), status="supported", card="inst"),
        dict(aspect="Reservoir families to search", choice="Search on genomic only, then retrain the pool on each reservoir's data.",
             ev=("Fresh per-reservoir searches match the retrained pool within 0.006 at 30k and within 0.004 at 100k (" + "; ".join(f"{k.replace('_', ' ')} {v['ar_top8'] - v['menu']:+.3f}" for k, v in f100.items() if v.get("menu") is not None) + " at 100k). Tested on one representative per family only." if f100 else "pending"), status="supported", card="fam"),
        dict(aspect="Reservoirs feeding the final ensemble", choice="Several; genomic plus motif captures most of the gain.",
             ev=("Ensemble on genomic held-out sequences: genomic only " + f"{un['genomic only']['genomic']:.3f}, genomic + motif {un['genomic + motif']['genomic']:.3f}, all five {un['all five']['genomic']:.3f}. Out of distribution the best single reservoir beats the union. Ensembles of single-reservoir models, not a mixed corpus." if un else "pending"), status="provisional", card="mix"),
        dict(aspect="Proposer model", choice="Sonnet 5.5.", ev="Equal-n ensemble: " + ", ".join(f"{k} {m(v.get('ens_at_n', [])):.4f}" for k, v in pr.items() if v.get("ens_at_n")) + ". Opus is ahead by under two standard errors.", status="supported", card="proposer"),
        dict(aspect="Prompt", choice="Guidance (known-good pieces, ensemble note), no phase schedule.", ev=f"At 50 models: guidance + phases {m(pa.get('P', [])):.3f}, guidance only {m(pa.get('W', [])):.3f}, nothing {m(pa.get('N', [])):.3f}.", status="supported", card="prompt"),
        dict(aspect="Ensemble method", choice="Non-negative ElasticNet on validation predictions of all models, with a screen for exploded predictions on every evaluation set.", ev="Beat greedy top-K by about +0.01 earlier; 0.8% of free-form models explode on a shifted evaluation set.", status="carried", card="ens"),
        dict(aspect="Training and evaluation loop", choice="One common loop with reverse-complement averaging for every model.", ev="Same network scores 0.69-0.70 in the old loop and 0.73 in the common loop; reverse-complement averaging is +0.014 of that.", status="supported", card="loop"),
        dict(aspect="Pools", choice="Nested canonical pools and independent nested replicates; legacy per-size pools only to reproduce old numbers.", ev=("A nested genomic ladder (one seed) lands within about 0.01 of the old ladder at every size, so the earlier scaling numbers stand." if d.get("nested_ladder", {}).get("ens") else "nested ladder pending"), status="supported", card="remeasure"),
    ]
    return rows


def render(cards, d):
    esc = html.escape
    counts = {k: sum(1 for c in cards if c["status"] == k) for k in STATUS}
    out = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
           "<title>ALBench Evidence Ledger</title>", f"<style>{CSS}</style></head><body><div class=\"wrap\">"]
    out.append('<header><h1>ALBench Evidence Ledger</h1>'
               '<p class="sub">Every choice in the training-corpus pipeline, the experiment that justifies it, and how much weight it can bear. '
               "Figures regenerate from the experiment outputs; running experiments fill in as they finish.</p>"
               f'<div class="stamp">data as of {esc(d.get("generated", "?"))} EDT &middot; {len(cards)} decisions</div></header>')
    out.append('<div class="summary"><button class="filter" data-k="all" aria-pressed="true">All ' + str(len(cards)) + "</button>" +
               "".join(f'<button class="filter" data-k="{k}" aria-pressed="false">{STATUS[k][0]} {n}</button>' for k, n in counts.items() if n) + "</div>")
    out.append('<div class="legend">' + "".join(f'<div><span class="chip s-{k}">{STATUS[k][0]}</span> {esc(STATUS[k][1])}</div>' for k in STATUS) + "</div>")
    out.append('<section class="choices" id="choices"><h2 class="grp">Setup choices at a glance</h2><div class="tblwrap"><table><thead><tr><th>Aspect</th><th>Current choice</th><th>Evidence</th><th>Status</th></tr></thead><tbody>' + "".join(
        f'<tr><td><a href="#{r["card"]}">{esc(r["aspect"])}</a></td><td>{esc(r["choice"])}</td><td>{esc(r["ev"])}</td><td><span class="chip s-{r["status"]}">{STATUS[r["status"]][0]}</span></td></tr>' for r in build_choices(d)) + "</tbody></table></div></section>")
    out.append('<nav class="flow" aria-label="Pipeline">' + "".join(
        f'<a href="#{c["id"]}"><span class="chip s-{c["status"]}">{STATUS[c["status"]][0]}</span><b>{esc(t)}</b><span class="t">{esc(s)}</span></a>'
        for t, s, cid in [("D ladder", "10k, 30k, 100k, 300k", "menu"), ("Proposer", "Sonnet 5.5, 2 x 30 rounds", "proposer"),
                          ("5 reservoir families", "independent instances each", "fam"), ("Top by validation", "8 per instance, pooled", "inst"),
                          ("Retrain at every D", "same pool, each dataset", "menu"), ("Ensemble", "enet_all_pos", "ens")]
        for c in [next(x for x in cards if x["id"] == cid)]) + "</nav>")
    seen = []
    for c in cards:
        if c["group"] not in seen:
            seen.append(c["group"])
    for g in seen:
        out.append(f'<h2 class="grp" data-group="{esc(g)}">{esc(g)}</h2>')
        for c in [x for x in cards if x["group"] == g]:
            st = c["status"]
            out.append(f'<section class="card" id="{c["id"]}" data-status="{st}" data-group="{esc(g)}"><div class="hd"><h3>{esc(c["q"])}</h3><span class="chip s-{st}">{STATUS[st][0]}</span></div>')
            out.append(f'<p class="choice"><b>Choice</b>{esc(c["choice"])}</p>')
            for f in c["figs"]:
                out.append(f'<figure class="figure"><img{" class=narrow" if f in NARROW else ""} src="figs/{f}" alt="{esc(c["q"])}" loading="lazy"></figure>')
            out.append('<div class="body"><ul>' + "".join(f"<li class=\"num\">{esc(p)}</li>" for p in c["points"]) + "</ul>")
            out.append(f'<div class="meta"><b>Caveat.</b> {esc(c["caveat"]).replace(esc(OLD_ARTIFACT), f"<a href={chr(34)}{OLD_ARTIFACT}{chr(34)}>previous artifact</a>")}</div>')
            if c.get("change"):
                out.append(f'<div class="meta"><b>What would change the decision.</b> {esc(c["change"])}</div>')
            if c.get("see"):
                out.append(f'<div class="meta"><b>Figure.</b> <a href="figs/{c["see"]}">{esc(c["see"])}</a></div>')
            out.append("</div></section>")
    out.append('<div class="log"><h2>Update log</h2><ol>' + "".join(f"<li><time>{esc(t)}</time> {esc(m)}</li>" for t, m in reversed(CHANGELOG)) + "</ol></div>")
    out.append(f'<footer>Generated by scripts/analysis/make_justification.py and build_justification_html.py. Earlier HP-search decisions: <a href="{OLD_ARTIFACT}">ALBench Design Decisions</a>.</footer>')
    out.append(f"</div><script>{JS}</script></body></html>")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="outputs/justification")
    a = ap.parse_args()
    src = Path(a.src)
    d = json.load(open(src / "data.json"))
    cards = build(d)
    (src / "index.html").write_text(render(cards, d))
    print("wrote", src / "index.html", len(cards), "cards")


if __name__ == "__main__":
    main()
