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
    n_ctx = sum(sum(v) for v in ctx.values())
    fam = d.get("family", {}).get("progress", {})
    n_fam = {f: sum(v for k, v in fam.items() if k.startswith(f + "|")) for f in ("random", "motif_ct_enriched", "evo2", "evoaug_paper2023")}
    pa = d.get("prompt_arms", {}).get("summary", {})
    pe = {k.split()[0]: v.get("ens_at_n", []) for k, v in pa.items()}
    mean = lambda v: sum(v) / len(v) if v else float("nan")
    bp = d.get("baselines_progress", {})
    bt = {k: sum(v) for k, v in bp.items()}
    B = d.get("budget", {})
    mf = d.get("menu_fam", {})

    cards = [
        dict(id="loop", group="Training and evaluation", status="supported", figs=["loop_decomposition.png"],
             q="Why did earlier scaling runs report r = 0.69-0.70 for a LegNet that scores 0.73 in the architecture loop?",
             choice="Train every model, from every source, in one common loop and report RC-averaged predictions.",
             points=[f"Same network, same HPs, three configs: reported test r {f3(min(nat))}-{f3(max(nat))} in the old loop vs {f3(min(arch))}-{f3(max(arch))} in the arch loop ({sgn(L.get('total_gap'))} on average).",
                     f"Two parts: reverse-complement averaging at test time {sgn(L.get('rc_avg'))}; the loop itself {sgn(L.get('loop_fwd_matched'))} on matched forward-only predictions.",
                     f"Inside the loop, the per-sample crop-shift ({sgn(L['flips'].get('A_flip_shift'))}) and fp16 ({sgn(L['flips'].get('A_flip_amp'))}) cost accuracy; early-stop rule, gradient clip and the time cap do not matter (within noise)."],
             caveat="Every scaling-law number measured before Oct 20 used the old loop and forward-only predictions. The shape of the curves probably survives; the absolute levels do not. Re-measuring them in the common loop is a decision still owed (see 'Scaling law re-measure' below).",
             change="A config where the old loop wins would mean the ingredients interact with the architecture and we need per-architecture loops."),
        dict(id="arch", group="Training and evaluation", status="supported", figs=["arch_vs_legnet.png"],
             q="Does a freely-designed architecture beat a well-tuned LegNet once both are trained the same way?",
             choice="Do not claim an architecture advantage for free-form search. Its value is diversity for the ensemble and a wide, assumption-free search.",
             points=[f"LegNet-eff HP-search winners: {', '.join(f3(v) for v in d['arch_vs_legnet']['LegNet-eff configs'])}. Top-3 free-form architectures: {', '.join(f3(v) for v in d['arch_vs_legnet']['free-form architectures'])}.",
                     "Single seeds scatter by about +-0.005, so these are indistinguishable.",
                     f"Retraining the full 33-model menu puts the reference LegNet at rank {rr.get('10k', '?')} / {rr.get('30k', '?')} / {rr.get('100k', '?')} at 10k / 30k / 100k, so it is a strong baseline once D >= 30k."],
             caveat="Genomic reservoir at 30k only; 3 seeds per architecture."),
        dict(id="menu", group="Dataset-size ladder", status="supported", figs=["menu_across_D.png"],
             q="Can we pick architectures at a small D and reuse them at a larger D?",
             choice="No. Retrain the whole candidate pool at every D and ensemble there; do not pre-select the top few at 10k.",
             points=[f"Rank correlation of the same {M.get('n_entries', 33)} architectures between D: 10k vs 30k, validation {f3(sp['spearman_10k_30k'].get('val'), 2)} / test {f3(sp['spearman_10k_30k'].get('test'), 2)}; 30k vs 100k, validation {f3(sp['spearman_30k_100k'].get('val'), 2)} / test {f3(sp['spearman_30k_100k'].get('test'), 2)}.",
                     f"Ensemble of the whole retrained menu: {', '.join(f3(v) for v in ens.get('all', []))} at 10k / 30k / 100k. Top-5 chosen at 10k and retrained: {', '.join(f3(v) for v in ens.get('top5_at_10k', []))}. HP-only search (old loop): {', '.join(f3(v) for v in ens.get('hp_only', []))}.",
                     "Retrainings per architecture so far: " + ", ".join(f"{k} {v}" for k, v in M.get("seeds_per_D", {}).items()) + " (more are running; the correlations are noisier than they look)."],
             caveat="Retest noise is not yet separated from true rank change: second and third seeds for 10k and 30k are still arriving. Menu is genomic only; family menus are running (see below). The HP-only comparison mixes harnesses.",
             change="If rank stability between two seeds at one D is also near zero, the 10k-30k result is noise, and top-k selection at small D would be fine."),
        dict(id="menufam", group="Dataset-size ladder", status="running", figs=["menu_families.png"],
             q="Do the same architectures do well on random, motif, evo2 and evoaug data, or does each reservoir need its own search?",
             choice="Menu is reservoir-agnostic and retrained per reservoir (decided earlier). This tests that on the new loop.",
             points=[f"Progress so far (of {mf.get('n_menu', 33)}): " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in mf.get("progress", {}).items()) + ".",
                     "Plots fill in once at least 8 menu entries per reservoir finish."],
             caveat="30k only. Runs on V100 as well as H100 with GPU-scaled time caps."),
        dict(id="inst", group="Search and pooling", status="supported", figs=["instances_and_topk.png"],
             q="How many models should one AutoResearch instance train, how many should we keep, and how many instances do we need?",
             choice="About 20-30 models per instance, keep the top 8 by validation, pool several independent instances.",
             points=[f"One instance's ensemble saturates around 20-40 models (mean of runs: {', '.join(f3(v) for v in I.get('mean', []))} at n = {I.get('n')}).",
                     f"Pooling the top-k of three instances: k = 1: {f3(top5[0])}, k = 3: {f3(top5[2])}, k = 8: {f3(top5[4])}, all: {f3(top5[6])}. Single instances with all their models: {f3(min(T.get('singles', [0])))}-{f3(max(T.get('singles', [0])))}.",
                     f"Between-instance spread of the final ensemble: SD {f3(I.get('sd_between_runs'), 4)}."],
             caveat="Three Sonnet-5.5 instances, genomic 30k."),
        dict(id="rep", group="Search and pooling", status="supported", figs=["replicates.png"],
             q="Is pooling independent runs actually better than a single run, and by how much?",
             choice="Always run several independent instances per reservoir and pool them.",
             points=[f"Three independent runs, 3 reservoirs x 3 D: the pooled ensemble beats the best single run in {wins} of {len(rep)} cells (gain over the best single {sgn(min(gains) if gains else float('nan'))} to {sgn(max(gains) if gains else float('nan'))}; over the average single {sgn(min(meansing) if meansing else float('nan'))} to {sgn(max(meansing) if meansing else float('nan'))}).",
                     "Single-run SD is about 0.009, the same size as many between-strategy gaps: any single-run comparison below that is noise."],
             caveat="HP-only runs in the old loop. The replicate structure (not the absolute r) is what carries over."),
        dict(id="llm", group="Proposer and prompt", status="provisional", figs=["llm_vs_classical.png"],
             q="Is an LLM proposer better than classical samplers, and if so is it because of search or because of its prior?",
             choice="Use the LLM with 2 proposals x 30 rounds, but do not claim it out-searches classical methods.",
             points=[f"Cadence study, genomic: LLM 2x30 {f3(cadg('genomic', 'llm2x30')[0])}, 6x10 {f3(cadg('genomic', 'llm6x10')[0])}, 1x60 {f3(cadg('genomic', 'llm1x60')[0])} (incomplete), classical multi6 {f3(cadg('genomic', 'multi6')[0])}; the LLM leads by +0.02 to +0.045.",
                     "The LLM's mean proposal quality is already high in round 0 and does not climb, so its edge is its prior, not search.",
                     f"Repaired baselines (random in the LLM-like prior box; TPE seeded from that prior; 6 runs x 30 models each) are running: random-full-box {bt.get('random_broad', 0)}, random-prior {bt.get('prior_marginal', 0)}, TPE-prior {bt.get('prior_tpe', 0)} of 180 models each."],
             caveat="The classical arm in the cadence study was weak by construction (<= 11 trials per sampler, no prior, four anchor copies). The repaired baselines exist to close that gap. If the prior-box baselines match the LLM, the case for the LLM is diversity and prior knowledge, which still justifies it for the free-architecture search.",
             change="If prior-box TPE matches the LLM at equal n, we say so in the paper and keep the LLM only for the architecture-writing step."),
        dict(id="kb", group="Proposer and prompt", status="supported", figs=["kb_prompts.png"],
             q="Does the knowledge base or a particular prompt wording make the proposer better?",
             choice="Keep the default prompt; do not spend more budget on prompt wording.",
             points=[f"Knowledge base on vs off, 5 cells: mean {sgn(kb.get('mean'))}, individual differences {', '.join(sgn(x) for x in kb.get('diffs', []) if x is not None)}.",
                     "Six prompt styles x 2 reps at genomic 30k: explore and reflect sit about +0.01 above the default, critic below, but each style's two reps straddle the single-run noise."],
             caveat="HP-only harness. Effects below about 0.01 are not detectable at one run per cell."),
        dict(id="prompt", group="Proposer and prompt", status="running", figs=["prompt_arms.png"],
             q="Should the free-architecture proposer be given guidance (known-good pieces, a reference LegNet, a build-then-explore schedule) or be left wide open?",
             choice="Wide open unless the evidence shows a clear benefit.",
             points=["Arms: P = guidance + reference + build-on-what-works for 6 rounds then explore; W = guidance only; N = nothing. Three independent runs per arm.",
                     f"At equal n = {d.get('prompt_arms', {}).get('n_eq', '?')} models, mean ensemble r: P {f3(mean(pe.get('P', [])))}, W {f3(mean(pe.get('W', [])))}, N {f3(mean(pe.get('N', [])))}. N is behind early; P and W are indistinguishable so far.",
                     "Counts per cell are unequal while runs finish; the figure labels each point with its model count."],
             caveat="Unequal model counts: read the right-hand panel only at equal n. P cells are complete; W and N were interrupted by the usage limit and resumed.",
             change="If N catches up by 60 models, the guidance only saves rounds and the wide-open prompt wins on principle."),
        dict(id="proposer", group="Proposer and prompt", status="running", figs=["proposer_models.png"],
             q="Which model should write the architectures?",
             choice="Sonnet 5.5 (cost and rate-limit headroom), unless Opus clearly wins.",
             points=["Same prompt (P), genomic 30k, three runs each for Sonnet 5.5, Sonnet 4.5, Opus 5.5.",
                     "Opus 5.5 cells are queued behind the LLM concurrency cap and have only a few models so far."],
             caveat="Not yet decisive; Opus has too few models to compare."),
        dict(id="ctx", group="Dataset-size ladder", status="running", figs=["context_transfer.png"],
             q="When AutoResearch moves from 30k to 100k, should the proposer see what worked at 30k?",
             choice="Test four arms with four reps each: no context, top architectures by validation from 30k (T), T plus a trajectory summary (J), and top architectures from 10k (X) as the stale-context control.",
             points=[f"Earlier HP-only pilot (n = 1): no robust effect; the one +0.03 gain at 100k did not replicate on a repeat (0.802 vs 0.771).",
                     f"New test: 16 cells x 15 rounds x 2 proposals at 100k; {n_ctx} models trained so far."],
             caveat="Cells are H100-only and sit in the lowest-priority queue, so results arrive slowly. Only validation numbers go into the context text."),
        dict(id="fam", group="Search and pooling", status="running", figs=["families.png"],
             q="Does AutoResearch find equally good models on every reservoir family?",
             choice="Run three independent instances on one member of each family, pool the top few by validation.",
             points=["Families: genomic, random, motif_ct_enriched, evo2, evoaug_paper2023. Prompt W, 30k, 30 rounds x 2.",
                     "Models trained so far: " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in n_fam.items()) + "."],
             caveat="Genomic bars reuse the W cells, which are themselves unfinished."),
        dict(id="ens", group="Ensembling", status="carried", figs=[],
             q="How should the pooled candidates be combined?",
             choice="Non-negative ElasticNet on the validation predictions of all models (enet_all_pos), z-scored per model so that one diverged model cannot zero the fit.",
             points=["Beat greedy top-K selection by about +0.01 in the earlier ensemble-construction study; uncertainty weighting and diversity (DPP) selection gave no gain.",
                     "The z-scored version was added after one exploded model (predictions ~1e20) zeroed the raw ElasticNet."],
             caveat=f"Earlier experiments are in the previous artifact: {OLD_ARTIFACT}"),
        dict(id="sched", group="Dataset-size ladder", status="planned", figs=[],
             q="How many rounds should each D get as the ladder climbs?",
             choice="Fewer rounds at larger D (compute per model grows ~10x per decade of D) and warm-start from the smaller D only if the context test shows a gain.",
             points=["Not designed yet. Needs the context result and the per-instance saturation curve at 100k (the 30k curve saturates at 20-30 models).",
                     "Illustrative schedule only: 30 rounds at 10k, 15 at 30k-100k, 6-10 at 300k."],
             caveat="Cost at 300k is dominated by oracle labelling plus training; rounds are the only free parameter."),
        dict(id="remeasure", group="Training and evaluation", status="open", figs=[],
             q="Do the scaling laws from earlier in the project survive the loop correction?",
             choice="Re-measure the 10k / 30k / 100k scaling cells with the common loop and RC-averaged predictions, 3 seeds per cell.",
             points=["The loop correction moves absolute r by about +0.035 at 30k. The earlier slopes and cross-reservoir ordering are unlikely to flip, but the claim is untested.",
                     "Menu retraining at three D already gives one set of curves in the new loop (see the ladder card)."],
             caveat="User decision pending on whether to spend the compute."),
        dict(id="budget", group="Compute and limits", status="running", figs=["llm_budget.png"],
             q="Can we afford the remaining programme within the LLM usage limit?",
             choice="Dispatcher pauses on the session limit, resumes after the reset, and caps concurrent LLM cells at 14.",
             points=[f"HPC cells used about {B.get('raw_tokens', 0) / 1e6:.0f}M raw tokens ({B.get('output_tokens', 0) / 1e6:.1f}M output) over the last {B.get('hours', '?')} active hours ({B.get('calls', '?')} calls).",
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
