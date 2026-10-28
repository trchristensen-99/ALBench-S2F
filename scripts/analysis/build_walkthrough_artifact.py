# ruff: noqa
#!/usr/bin/env python
"""Build the 'ALBench Results Walkthrough' page (one self-contained HTML file): procedure, evaluation sets, a worked AutoResearch example at 30k, scaling results,
the 30k-genomic top-up experiment, and an appendix with the decisions, AutoResearch v2 and caveats.

    python scripts/analysis/build_walkthrough_artifact.py --in <dir with data_oct30/31/32.json, decisions.json, findings.json, report_*.md> --out walkthrough.html
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_prelim_artifact import (
    CSS as BASE_CSS,
    ar2_build_notes,
    ar2_compare,
    decisions_section,
    esc,
    fmt,
    md_inline,
    parse_report,
    table,
    v1v0,
)  # noqa: E402

SETS3 = ("genomic", "ood", "random_32k")
SETN = {"genomic": "genomic test", "ood": "designed OOD", "random_32k": "random sequences"}
FAM_COL = {"genomic": "#1b6ca8", "perturbation": "#c4571a", "motif": "#2a8f5b", "random": "#7a5195"}
STRATS = [
    ("genomic", "Gosai genomic", "genomic"),
    ("zoonomia_orthologs", "Zoonomia orthologs", "genomic"),
    ("evoaug_paper2023", "EvoAug, published settings", "perturbation"),
    ("mutagenesis_mr10", "Random mutagenesis, 10%", "perturbation"),
    ("motif_ct_enriched", "Motif, CT-enriched", "motif"),
    ("motif_shared_core", "Motif, shared core", "motif"),
    ("random", "Uniform random", "random"),
    ("random_gc_sweep", "Random, GC 25-75% batches", "random"),
]
EXTRA_CSS = """
.step{display:grid;grid-template-columns:44px 1fr;gap:14px;margin:12px 0;align-items:start}
.step .n{width:36px;height:36px;border-radius:50%;background:var(--accent);color:var(--bg);display:grid;place-items:center;font-weight:700;font-variant-numeric:tabular-nums}
.step .b{background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:12px 14px}
.step h3{margin:0 0 4px;font-size:16px}.step p{margin:4px 0;font-size:15px}.step ul{margin:4px 0 4px 18px;padding:0;font-size:14.5px}
.flow{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:14px 0;font-size:14px}
.flow .box{background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:6px 10px}
.flow .arr{color:var(--muted)}
.heat td{padding:5px 8px;text-align:right}.heat td.l{text-align:left;min-width:170px;white-space:normal}
.small{font-size:13.5px;color:var(--muted)}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
.kv{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0}.kv span{background:var(--accent-soft);border-radius:6px;padding:2px 9px;font-size:13.5px}
.bar{height:9px;background:var(--accent);border-radius:3px;display:inline-block;vertical-align:middle}
.tag.gen{color:var(--c-genomic);border-color:var(--c-genomic)}
.tl td:first-child{min-width:130px}
@media print{nav{display:none}}
"""


def fig_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def fig_run(run):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = run["candidates"]
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    xs = [x["round"] + (0.12 if i % 2 else -0.12) for i, x in enumerate(c)]
    ys = [x["val_r"] for x in c]
    best = np.maximum.accumulate(ys)
    ax.scatter(
        xs,
        ys,
        s=[18 + 14 * (x["n_params"] or 0) / 1e6 for x in c],
        color="#1b6ca8",
        alpha=0.85,
        zorder=3,
    )
    ax.step(
        [x["round"] for x in c],
        best,
        where="post",
        color="#c4571a",
        lw=1.6,
        label="best so far",
        zorder=2,
    )
    top = sorted(range(len(c)), key=lambda i: -ys[i])[:3]
    for rank, i in enumerate(top):
        ax.annotate(
            f"#{rank + 1}",
            (xs[i], ys[i]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            color="#c4571a",
        )
    ax.set_xlabel("round (2 proposals per round)")
    ax.set_ylabel("validation Pearson r")
    ax.set_xticks(range(10))
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_title(
        "One search run, 30k genomic pool: every proposal (dot size = parameters)", fontsize=10
    )
    return fig_png(fig)


def fig_weights(w):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colmap = {
        "genomic": "#1b6ca8",
        "random": "#7a5195",
        "motif CT-enriched": "#2a8f5b",
        "evoaug": "#c4571a",
        "evo2": "#a8892a",
        "mixed corpus (genomic + random + motif)": "#6b7a80",
    }
    m = [x for x in w["members"]]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    y = np.arange(len(m))[::-1]
    ax.barh(y, [x["weight"] for x in m], color=[colmap.get(x["family"], "#888") for x in m])
    ax.set_yticks(
        y,
        [f"{x['origin'].replace('_', ' ')} · {(x['n_params'] or 0) / 1e6:.1f}M" for x in m],
        fontsize=7.5,
    )
    ax.set_xlabel(
        "weight in the stack (non-negative ElasticNetCV, fitted on validation predictions)"
    )
    ax.set_title(
        "30k genomic pool: which of the 20 configurations the ensemble uses (colour = reservoir the search ran on)",
        fontsize=10,
    )
    ax.grid(axis="x", alpha=0.25)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[Patch(color=v, label=k.split(" (")[0]) for k, v in colmap.items()],
        fontsize=7.5,
        frameon=False,
        loc="lower right",
    )
    return fig_png(fig)


def fig_scale(d30):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sc = d30["scale31"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    for ax, e in zip(axs, SETS3):
        for st, label, fam in STRATS:
            v = (sc.get(st) or {}).get("sizes", {})
            pts = [
                (int(D), v[D]["first6"]["ensemble"][e])
                for D in ("10000", "30000", "100000")
                if D in v and v[D].get("first6")
            ]
            if pts:
                ax.plot(
                    *zip(*pts),
                    marker="o",
                    color=FAM_COL[fam],
                    ls="-"
                    if st in ("genomic", "motif_ct_enriched", "random", "evoaug_paper2023")
                    else "--",
                    label=label,
                    lw=1.6,
                )
        ax.set_xscale("log")
        ax.set_xticks([10000, 30000, 100000], ["10k", "30k", "100k"])
        ax.set_title(SETN[e], fontsize=10)
        ax.grid(alpha=0.25)
    axs[0].set_ylabel("ensemble Pearson r")
    axs[-1].legend(fontsize=7.5, loc="lower right", frameon=False)
    return fig_png(fig)


def fig_arms(d30):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    al = d30["al31"]
    keys = [k for k, v in al.items() if v.get("first6")]
    kc = {
        "genomic": "#1b6ca8",
        "reservoir": "#c4571a",
        "mix": "#7a7a7a",
        "acquisition": "#2a8f5b",
        "ceiling": "#b0b0b0",
    }
    fig, axs = plt.subplots(1, 3, figsize=(14, 6), sharey=True)
    ctl = al["genomic"]["first6"]["ensemble"]
    for ax, e in zip(axs, SETS3):
        order = sorted(keys, key=lambda k: al[k]["first6"]["ensemble"][e])
        ax.barh(
            [al[k]["label"][:46] for k in order],
            [al[k]["first6"]["ensemble"][e] - ctl[e] for k in order],
            color=[kc[al[k]["kind"]] for k in order],
        )
        ax.axvline(0, color="#555", lw=0.8)
        ax.set_title(f"{SETN[e]}: change vs +70k genomic", fontsize=9.5)
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(axis="y", labelsize=7.5)
    fig.tight_layout()
    return fig_png(fig)


def intro_section():
    return """<section id="procedure"><h2>1 · The procedure</h2>
<p class="lead">We are choosing what to put on a roughly 4M-oligo K562 MPRA library. A <b>reservoir strategy</b> generates candidate sequences, an <b>acquisition strategy</b> decides which to label, an <b>oracle</b> supplies the labels, and small <b>student models</b> trained on the labelled set tell us how useful that set is. The deliverable is a training-corpus recommendation, not a model.</p>
<div class="flow"><span class="box">Reservoir<br><span class="small">candidate sequences</span></span><span class="arr">→</span><span class="box">Acquisition<br><span class="small">which to label</span></span><span class="arr">→</span><span class="box">Oracle<br><span class="small">K562 labels</span></span><span class="arr">→</span><span class="box">Pool of size D<br><span class="small">nested prefixes</span></span><span class="arr">→</span><span class="box">AutoResearch<br><span class="small">20 configurations</span></span><span class="arr">→</span><span class="box">Train + stack<br><span class="small">weighted ensemble</span></span><span class="arr">→</span><span class="box">Evaluate<br><span class="small">on held-out sets</span></span></div>

<div class="step"><div class="n">1</div><div class="b"><h3>Reservoir: where candidate sequences come from</h3>
<p>Strategies are grouped in families; this study scales eight of them.</p>
<ul><li><b>Genomic</b>: Gosai genomic CREs (about 315k exist, so it cannot scale past that); Zoonomia orthologs (real aligned orthologous CREs from the 241-mammal alignment).</li>
<li><b>Perturbation</b>: EvoAug-style structural perturbations at the published settings; random mutagenesis of genomic CREs (here 10% per base).</li>
<li><b>Random and composition</b>: uniform random; random at six GC levels (25 to 75%) pooled in equal batches, as an experimentalist would order several GC batches.</li>
<li><b>Motif</b>: planted transcription-factor motifs from JASPAR and CIS-BP, cell-type enriched (CT) or shared-core (K562 and HepG2).</li>
<li>Also registered, not scaled here: dinucleotide shuffle, GC-matched random, model-guided and generative arms, mixtures.</li></ul></div></div>

<div class="step"><div class="n">2</div><div class="b"><h3>Acquisition: which candidates to pay to label</h3>
<p>Without acquisition, a set is simply taken from the reservoir (the non-active arms). With acquisition, a model trained on existing data scores a large candidate pool and a batch is chosen in one round: ensemble mean or SD, uncertainty in excess of the activity trend, diversity (k-means on sequence composition), uncertainty plus diversity (a BADGE-like k-means++ on member disagreement), stratified and composition-based designs, and the oracle's own error as an unattainable ceiling. Plain SD is largely a function of activity, so the excess-over-trend version is run alongside it.</p></div></div>

<div class="step"><div class="n">3</div><div class="b"><h3>Oracle labelling</h3>
<p>Synthetic sequences have no measurements, so every sequence is labelled by <code>oracle_v3</code>: an AlphaGenome-encoder model with a regression head, fine-tuned on the Gosai K562 data as a 10-fold, chromosome-split ensemble. On chromosomes it never saw, its Pearson r with the experiment is <b>0.931</b> on average (range 0.923 to 0.939). Labelling runs at several hundred sequences per second per GPU (300 to 900 measured), and every pool is stamped with the oracle that labelled it. Test chromosomes (chr7, chr13) and validation chromosomes (chr19, chr21, chrX) are excluded from every reservoir (audited: zero overlap).</p></div></div>

<div class="step"><div class="n">4</div><div class="b"><h3>Datasets of increasing size are nested</h3>
<p>A dataset of size D is the first D sequences of a labelled pool, so 10k lies inside 30k inside 100k and a size comparison changes only the amount of data. The pools used here are the canonical seed-42 nested pools; the top-up experiment uses the replicate-1 pools.</p></div></div>

<div class="step"><div class="n">5</div><div class="b"><h3>AutoResearch: a menu of 20 model configurations</h3>
<p>The student is not one architecture. An LLM (Claude Sonnet 5.5) proposes two complete PyTorch models per round, each with its training recipe; every one is trained and scored on <b>validation only</b>; the history is fed back; ten rounds make one <b>run</b> of 20 models. We ran 18 runs (three on each of five reservoir families and three on a mixed corpus) and took the best model of every run plus the two best of the rest: <b>menu v1, 20 configurations</b>. The same menu is trained on every pool up to 100k (12 configurations from 300k), with early-stopping patience 16. The per-size rule and the evidence for it are in the appendix.</p></div></div>

<div class="step"><div class="n">6</div><div class="b"><h3>Ensembling</h3>
<p>All configurations are trained on the pool. Their predictions on a shared validation set (29,449 real chr19, chr21 and chrX sequences relabelled by the oracle) are z-scored and combined by a <b>non-negative ElasticNetCV</b>; the weights are fitted once and applied to the test predictions. Zero-weight configurations simply drop out. In earlier tests this beat greedy top-K selection by about 0.01 and uncertainty or diversity weighting added nothing; it is the study's standard, and every number below is such an ensemble unless it says single model.</p></div></div>
</section>"""


EVAL_ROWS = [
    (
        "Headline",
        "genomic",
        "Real held-out genomic sequences (chr7 and chr13)",
        "In-distribution accuracy on natural sequence",
        "genomic reservoirs",
    ),
    (
        "Headline",
        "ood",
        "The high-activity <b>designed</b> Gosai sequences (mean oracle activity about 4, against 0.2 to 1.2 for everything we generate)",
        "Extrapolation to the activity range people design for",
        "no reservoir reaches this activity range; motif comes closest",
    ),
    (
        "Headline",
        "random_32k",
        "Uniform random sequences",
        "Behaviour on unconstrained sequence",
        "random",
    ),
    (
        "Headline",
        "gc_sweep",
        "Random sequences spread over GC content",
        "GC dependence, including the high-GC extrapolation regime",
        "random, GC sweep",
    ),
    (
        "Composition twins",
        "dinuc_shuffle",
        "Dinucleotide shuffles of held-out CREs",
        "Composition without motif syntax",
        "dinuc shuffle",
    ),
    (
        "Composition twins",
        "gc_matched",
        "Random sequence matched to the genomic GC histogram",
        "Is genomic value just GC?",
        "GC-matched random",
    ),
    (
        "Composition twins",
        "motif_planted",
        "Motif-planted sequences on test backgrounds with novel combinations",
        "Motif syntax on unseen backgrounds",
        "motif families",
    ),
    (
        "Composition twins",
        "zoonomia",
        "Zoonomia orthologs of test-chromosome windows",
        "Cross-species real sequence",
        "Zoonomia orthologs",
    ),
    (
        "Perturbation",
        "sub_low / sub_med / sub_high",
        "Held-out sequences with 1 to 50% substitutions at three rates",
        "Robustness to point mutation",
        "mutagenesis, EvoAug",
    ),
    (
        "Perturbation",
        "ins_* / del_*",
        "Insertions and deletions (1 to 10%) at low, medium and high rates",
        "Robustness to indels",
        "EvoAug",
    ),
    (
        "Perturbation",
        "inversion / translocation",
        "Held-out sequences with segments inverted or moved",
        "Structural perturbation",
        "EvoAug",
    ),
    (
        "Variants",
        "snv_ref / snv_alt",
        "Reference and alternate alleles of genomic SNVs (Hamming-1 pairs)",
        "Variant effect prediction; effects are tiny",
        "genomic",
    ),
    ("Controls", "ctrl_neg", "Negative-control sequences", "Floor check", "–"),
]


def evalsets_section(d32):
    es = d32.get("evalsets", {})

    KEYS = {
        "sub_low / sub_med / sub_high": ["sub_low", "sub_med", "sub_high"],
        "ins_* / del_*": ["ins_low", "ins_med", "ins_high", "del_low", "del_med", "del_high"],
        "inversion / translocation": ["inversion", "translocation"],
        "snv_ref / snv_alt": ["snv_ref", "snv_alt"],
    }

    def n(name):
        ks = [k for k in KEYS.get(name, [name]) if k in es]
        if not ks:
            return "–"
        return "{:,}".format(es[ks[0]]["n"]) + (f" each (×{len(ks)})" if len(ks) > 1 else "")

    rows = []
    last = None
    out = [
        "<div class='tw'><table><thead><tr><th>Set</th><th>Sequences</th><th>How it is built</th><th>What it probes</th><th>Matches reservoir</th></tr></thead><tbody>"
    ]
    for grp, name, how, probes, match in EVAL_ROWS:
        if grp != last:
            out.append(f"<tr class='grp'><td colspan='5'>{esc(grp)}</td></tr>")
            last = grp
        out.append(
            f"<tr><td><code>{esc(name)}</code></td><td style='text-align:right'>{n(name)}</td><td style='text-align:left;white-space:normal'>{how}</td><td style='text-align:left;white-space:normal'>{esc(probes)}</td><td style='text-align:left;white-space:normal'>{esc(match)}</td></tr>"
        )
    out.append("</tbody></table></div>")
    return f"""<section id="evals"><h2>2 · The evaluation sets</h2>
<p class="lead">Everything is scored as Pearson r against oracle K562 labels on held-out sets. Three are reported in the headline tables: <b>genomic</b> (natural sequence), <b>designed OOD</b> (high-activity designed sequences, the regime a library designer cares about) and <b>random sequences</b>. The others are test-side twins of each reservoir family and perturbation battery, used to see whether a strategy helps beyond the data it resembles. Validation, which drives early stopping and ensemble weights, is always real genomic sequence from chr19, chr21 and chrX, so a reservoir is never tuned to its own evaluation set.</p>
{"".join(out)}
<p class="small">The test sets are kept separate on purpose: where strategies favour some sets over others, that is reported rather than averaged away.</p></section>"""


def run_section(d32):
    run = d32["run_example"]
    c = run["candidates"]
    img = fig_run(run)
    rows = []
    best = max(x["val_r"] for x in c)
    top3 = sorted(c, key=lambda x: -x["val_r"])[:3]
    for x in c:
        rank = next((i + 1 for i, t in enumerate(top3) if t["id"] == x["id"]), None)
        idea = x["idea"] or ""
        idea = idea if len(idea) < 120 else idea[:117].rsplit(" ", 1)[0] + "…"
        rows.append(
            f"<tr><td>{x['round']}</td><td style='text-align:left;white-space:normal'>{esc(idea)}</td><td>{(x['n_params'] or 0) / 1e6:.2f}M</td><td>{esc(x['optimizer'])}</td><td>{x['epochs_trained']}</td><td{' class=best' if rank == 1 else ''}>{x['val_r']:.4f}{' ·#' + str(rank) if rank else ''}</td></tr>"
        )
    tbl = (
        "<div class='tw'><table><thead><tr><th>Round</th><th>Proposal (the model writes the idea and the code)</th><th>Params</th><th>Optimizer</th><th>Epochs</th><th>Val r</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )
    ar = run["args"]
    return f"""<div class="step"><div class="n">a</div><div class="b"><h3>A single search run</h3>
<div class="kv"><span>reservoir: genomic, replicate-1 pool</span><span>D = 30,000</span><span>{ar.get("rounds", 10)} rounds × {ar.get("per_round", 2)} proposals = 20 models</span><span>proposer: {esc(run.get("model"))}</span><span>feedback: validation only</span><span>guidance: known-good blocks + 'these will be ensembled'</span></div>
<p>Each round the proposer sees the history of what it has tried and the validation scores, then writes two new networks with their own optimizer, learning rate, width and depth. They are trained (early stopping, time cap) and scored; nothing from a test set is ever shown. The best candidate of this run reached validation r {best:.3f}; the top three are marked.</p>
<figure><img alt="val r by proposal" src="{img}"><figcaption>Search progress: 20 proposals, best so far in orange. Most proposals land between 0.78 and 0.83 and the best came in the last round, so this run had not yet plateaued.</figcaption></figure>
{tbl}</div></div>"""


def menu_section(d32):
    w = d32["weights30k"]
    rows = []
    fam_order = {}
    for m in w["members"]:
        fam_order.setdefault(m["family"], []).append(m)
    img = fig_weights(w)
    ens = w["ensemble"]
    singles = [m["test"]["genomic"] for m in w["members"]]
    best_single = max(singles)
    mean_single = float(np.mean(singles))
    rows = []
    for m in sorted(w["members"], key=lambda x: -(x["search_val"] or 0)):
        idea = m["idea"] or ""
        idea = idea if len(idea) < 100 else idea[:97].rsplit(" ", 1)[0] + "…"
        rows.append(
            f"<tr><td style='text-align:left;white-space:normal'>{esc(m['origin'])}</td><td style='text-align:left;white-space:normal'>{esc(idea)}</td><td>{(m['n_params'] or 0) / 1e6:.2f}M</td><td>{(m['search_val'] or 0):.3f}</td><td>{m['val_r']:.3f}</td><td>{m['test']['genomic']:.3f}</td><td>{m['weight']:.3f}</td></tr>"
        )
    tbl = (
        "<div class='tw'><table><thead><tr><th>Search run</th><th>Architecture idea</th><th>Params</th><th>Search val r</th><th>Val r at 30k</th><th>Single-model test r</th><th>Stack weight</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )
    return f"""<div class="step"><div class="n">b</div><div class="b"><h3>From 18 runs to a menu of 20</h3>
<p>Eighteen runs like the one above (360 proposals) were run on five reservoir families (genomic, random, motif CT-enriched, EvoAug, Evo 2; three each) and on a mixed corpus (three). The best model of every run, plus the two best of the remainder, make the 20-configuration menu: residual SE towers with different read-outs, U-Nets, Res2Net and DenseNet-style towers, a weight-tied recurrent block, attention-pooled readouts, composition-conditioned towers. They range from 0.29M to 2.5M parameters (median about 1.4M).</p>
<h3>Training at D = 30k and the ensemble</h3>
<p>All 20 are trained on the 30k genomic pool (patience 16, one seed) and their predictions on the shared validation set are stacked. Single models reach <b>{min(singles):.3f} to {best_single:.3f}</b> on the genomic test set (mean {mean_single:.3f}); the ensemble reaches <b>{ens["genomic"]:.3f}</b> genomic, <b>{ens["ood"]:.3f}</b> designed OOD and <b>{ens["random_32k"]:.3f}</b> random sequences. <b>{w["n_positive"]} of {w["n"]}</b> configurations get a positive weight, and the weight is spread across search reservoirs, which is why the menu does not need to be searched on the reservoir it is applied to.</p>
<figure><img alt="stack weights" src="{img}"><figcaption>Ensemble weights at 30k genomic.</figcaption></figure>
{tbl}</div></div>"""


def ar2_example_section(d32, inp):
    a = d32["ar2_30k"]
    calls = {}
    for c in a["calls"]:
        calls.setdefault(c["mode"], {})[c["status"]] = c["n"]
    fam_rows = []
    for f in sorted(a["families"], key=lambda x: -(x["confirmed_mean"] or x["screen_best"] or 0)):
        fam_rows.append(
            (
                esc(f["family"].replace("_", " ")),
                [
                    str(f["n_candidates"]),
                    f["state"],
                    str(f["n_runs"]),
                    f["screen_best"],
                    f["confirmed_mean"],
                ],
                "",
            )
        )
    t = table(
        [
            "Family",
            "Candidates",
            "State",
            "Training runs",
            "Best single run (val r)",
            "Mean of 3 confirmation seeds",
        ],
        fam_rows,
    )
    cand_rows = []
    for c in a["candidates"]:
        s = c["summary"] or ""
        s = s if len(s) < 120 else s[:117].rsplit(" ", 1)[0] + "…"
        origin = (
            "hand-written reference"
            if c["source"] in (None, "reference") or (c["summary"] or "").startswith("reference")
            else ("refinement of " + c["parent"] if c["parent"] else "LLM, new family")
        )
        cand_rows.append(
            f"<tr><td style='text-align:left'>{esc(c['family'].replace('_', ' '))}</td><td style='text-align:left;white-space:normal'>{esc(s)}</td><td>{(c['n_params'] or 0) / 1e6:.2f}M</td><td style='text-align:left'>{esc(origin)}</td></tr>"
        )
    ct = (
        "<div class='tw'><table><thead><tr><th>Family</th><th>What the model is</th><th>Params</th><th>Origin</th></tr></thead><tbody>"
        + "".join(cand_rows)
        + "</tbody></table></div>"
    )
    p = a["purposes"]
    return f"""<div class="step"><div class="n">d</div><div class="b"><h3>The same cell under AutoResearch v2 (the controller-driven design)</h3>
<p>The newer design replaces the open-ended loop with a controller: three hand-written anchor architectures are tuned and confirmed first; the LLM then writes one model at a time to fill empty architecture families; every family gets a fixed recipe-tuning budget before it is judged; results enter the record only after three confirmation seeds. At genomic 30k this ran <b>{a["n_runs"]} training runs</b> ({p.get("first_look", 0)} first looks, {p.get("tune", 0) + p.get("second_chance", 0) + p.get("deep_tune", 0)} recipe-tuning runs, {p.get("screen", 0)} procedure-screen runs, {p.get("confirm", 0)} confirmation runs, {p.get("refine", 0)} for refinements, {p.get("final", 0)} final) and <b>{sum(sum(v.values()) for v in calls.values())} LLM calls</b>: {calls.get("SKETCH", {}).get("ok", 0) + calls.get("SKETCH", {}).get("failed", 0)} sketch calls (ten ideas each), {calls.get("EXPLORE", {}).get("ok", 0)} implementations of a new family and {calls.get("REFINE", {}).get("ok", 0)} single-change refinements. Training used the specification's loop: AdamW, no shift augmentation, up to 40 epochs.</p>
{t}
{ct}
<p>Confirmed and menu-eligible: <b>cnn_residual 0.814, rnn_hybrid 0.806, cnn_dilated 0.799</b> (validation r, mean of three seeds), with the transformer anchor retired at 0.710. No new family beat the best anchor beyond the noise threshold, so the report states that the space is flat at this data size. The four families' best configurations (with their tuned recipes) were then exported as a menu and retrained with the same per-D procedure as menu v1; the ensemble is compared with menu v1 in the appendix.</p></div></div>"""


def rounds_section(inp):
    p = os.path.join(inp, "data_rounds.json")
    if not os.path.exists(p):
        return ""
    d = json.load(open(p))
    per = d["per_run"]
    NS = ("10", "20", "30", "40", "60")
    rows = []
    for n in NS:
        v = [r[n] for r in per.values() if n in r]
        if not v:
            continue
        m = lambda f: float(np.mean([f(x) for x in v]))
        pooled = [f[n]["stack_all"]["genomic"] for f in d["pooled"].values() if n in f]
        rows.append(
            (
                f"first {n} candidates ({int(n) // 2} rounds)",
                [
                    m(lambda x: x["best_val"]),
                    m(lambda x: x["best_test"]),
                    m(lambda x: x["stack_all"]["genomic"]),
                    m(lambda x: x["stack_all"]["ood"]),
                    m(lambda x: x["stack_all"]["random_32k"]),
                    float(np.mean(pooled)) if pooled else None,
                ],
                "",
            )
        )
    t = table(
        [
            "Search length",
            "Best single: val r",
            "Best single: test r",
            "Run ensemble: genomic",
            "Run ensemble: designed",
            "Run ensemble: random",
            "Three runs pooled: genomic",
        ],
        rows,
        best=[0, 1, 2, 3, 4, 5],
    )
    gains = [
        (r["30"]["stack_all"]["genomic"] - r["20"]["stack_all"]["genomic"])
        for r in per.values()
        if "20" in r and "30" in r
    ]
    pooled_gain = [
        f["30"]["stack_all"]["genomic"] - f["20"]["stack_all"]["genomic"]
        for f in d["pooled"].values()
        if "30" in f and "20" in f
    ]
    guided_gain = [
        f["30"]["stack_all"]["genomic"] - f["20"]["stack_all"]["genomic"]
        for k, f in d["pooled"].items()
        if k != "son55_none" and "30" in f and "20" in f
    ]
    _u = d["pooled"].get("son55_none", {})
    unguided_gain = (
        (_u["30"]["stack_all"]["genomic"] - _u["20"]["stack_all"]["genomic"])
        if "30" in _u
        else float("nan")
    )
    return f"""<div class="step"><div class="n">c</div><div class="b"><h3>Why 10 rounds? What more rounds buy</h3>
<p>Fifteen existing runs on the 30k genomic pool were run for 30 rounds (about 60 proposals). Taking the first 10, 20, 30, 40 or 60 proposals of each shows what a longer search adds. Test r is on the genomic test set unless the column says otherwise; the ensemble is the same non-negative stack, built from the search-time models.</p>
{t}
<p>Going from 10 to 15 rounds (20 to 30 proposals) raises the best single model's validation r by about {np.mean([per[k]["30"]["best_val"] - per[k]["20"]["best_val"] for k in per if "30" in per[k]]):+.4f} and the ensemble of that one run by <b>{np.mean(gains):+.4f}</b> ({sum(g > 0 for g in gains)} of {len(gains)} runs improve, none by more than {max(gains):.4f}). When the candidates of the three runs of one family are pooled, the way the menu pools runs, the same step adds <b>{np.mean(pooled_gain):+.4f}</b> on average across the five run families, but only <b>{np.mean(guided_gain):+.4f}</b> across the four searched with the standard guidance ({", ".join(f"{g:+.4f}" for g in guided_gain)}); the family searched without guidance gains {unguided_gain:+.4f}. So there is a real but small boost for a single run's ensemble, and a smaller one once runs are pooled, because other runs already supply much of the diversity that extra rounds would. At the level of the final ensemble it is about the size of one-seed noise (0.002 to 0.003) and costs half again as many trainings per run, so we keep 10 rounds as the default and put extra budget into more runs over different reservoirs; a cell that can afford 15 rounds should expect roughly +0.001 to +0.002.</p>
<p class="small">Limits: search-time models on one reservoir and size (30k genomic), not a retrained menu; the final menu takes only the best model of each run (plus two), where the gain from more rounds is the {np.mean([per[k]["30"]["best_val"] - per[k]["20"]["best_val"] for k in per if "30" in per[k]]):+.4f} in the best single model's validation r.</p></div></div>"""


def alln_fmt(x):
    return f"{x:.3f}"


def heat_table(d32, size):
    cells = d32.get("allsets", {})
    sets = [
        "genomic",
        "ood",
        "random_32k",
        "gc_sweep",
        "dinuc_shuffle",
        "sub_med",
        "ins_med",
        "del_med",
        "inversion",
        "translocation",
        "snv_ref",
    ]
    head = ["Strategy"] + [s.replace("_32k", "").replace("_", " ") for s in sets]
    rows = []
    vals = {s: [] for s in sets}
    for st, label, fam in STRATS:
        c = cells.get(f"{st}|{size}")
        if c:
            for s in sets:
                if s in c["r"]:
                    vals[s].append(c["r"][s])
    rng = {s: (min(v), max(v)) for s, v in vals.items() if v}
    out = [
        "<div class='tw heat'><table><thead><tr>"
        + "".join(f"<th>{esc(h)}</th>" for h in head)
        + "</tr></thead><tbody>"
    ]
    for st, label, fam in STRATS:
        c = cells.get(f"{st}|{size}")
        tds = []
        for s in sets:
            if c and s in c["r"]:
                lo, hi = rng[s]
                a = 0.0 if hi == lo else (c["r"][s] - lo) / (hi - lo)
                tds.append(
                    f"<td style='background:rgba(11,110,110,{0.08 + 0.42 * a:.2f})'>{c['r'][s]:.3f}</td>"
                )
            else:
                tds.append("<td class='na'>–</td>")
        out.append(
            f"<tr><td class='l'><span class='dot' style='background:{FAM_COL[fam]}'></span>{esc(label)}{'' if c else ' <span class=small>(cell not finished)</span>'}</td>{''.join(tds)}</tr>"
        )
    out.append("</tbody></table></div>")
    return "".join(out)


def scaling_section(d30, d32):
    sc = d30["scale31"]
    sizes = ("10000", "30000", "100000")
    head = ["Strategy"] + [f"{int(D) // 1000}k {SETN[s]}" for D in sizes for s in SETS3]
    rows = []
    src100 = []
    for st, label, fam in STRATS:
        v = (sc.get(st) or {}).get("sizes", {})
        vals = []
        for D in sizes:
            x = (v.get(D) or {}).get("first6")
            for s in SETS3:
                vals.append(x["ensemble"][s] if x else None)
            if D == "100000" and (v.get(D) or {}).get("source") == "v1val":
                src100.append(label)
        rows.append(
            (f"<span class='dot' style='background:{FAM_COL[fam]}'></span>{esc(label)}", vals, fam)
        )
    t = table(head, rows, best=list(range(9)))
    foot = (
        f"<p class='small'>100k for {', '.join(src100)} comes from the menu-v1 retrain on the replicate-1 pool (patience 8, 20 configurations); every other cell is patience 16 on the nested pool, compared on the first 6 menu configurations (the comparison every cell shares). A dash means the cell had not finished training; no cell is imputed.</p>"
        if src100
        else ""
    )
    img = fig_scale(d30)
    return f"""<section id="scaling"><h2>4 · Scaling results</h2>
<p class="lead">Each strategy on its own nested pool at 10k, 30k and 100k, one seed per configuration. The best value in each column is shaded.</p>
{t}{foot}
<figure><img alt="scaling curves" src="{img}"><figcaption>Ensemble r against dataset size. Colour is the reservoir family; solid lines mark the four core strategies.</figcaption></figure>
<h3>The same ensembles on the wider evaluation battery</h3>
<p class="lead">30k pools. Shading runs from the lowest to the highest value in each column, so each column shows which reservoir is best for that evaluation set rather than which set is easy.</p>
{heat_table(d32, 30000)}
<h3>At 10k</h3>
{heat_table(d32, 10000)}
</section>"""


def topup_section(d30):
    al = d30["al31"]
    ctl = al["genomic"]["first6"]["ensemble"]
    mixr = (al.get("mix_random") or {}).get("first6", {}).get("ensemble")
    ARMS = [
        ("genomic", "70k more genomic (control)", "control"),
        ("res_random", "70k uniform random", "reservoir"),
        ("res_random_gc_sweep", "70k random, GC 25-75% batches", "reservoir"),
        ("res_motif_ct_enriched", "70k motif, CT-enriched", "reservoir"),
        ("res_motif_shared_core", "70k motif, shared core", "reservoir"),
        ("res_evoaug_paper2023", "70k EvoAug, published settings", "reservoir"),
        ("res_mutagenesis_mr10", "70k random mutagenesis, 10%", "reservoir"),
        ("res_zoonomia_orthologs", "70k Zoonomia orthologs", "reservoir"),
        ("mix_random", "70k at random from the even mix", "mix"),
        ("mix_mean", "highest ensemble mean", "acq"),
        ("mix_std", "highest ensemble SD", "acq"),
        ("mix_resid", "SD excess over the activity trend", "acq"),
        ("mix_binned", "highest SD within activity bins", "acq"),
        ("mix_diverse", "cluster-balanced (diversity only)", "acq"),
        ("mix_stdiv", "SD, then diversity", "acq"),
        ("mix_badge", "BADGE-like k-means++ on disagreement", "acq"),
        ("mix_strat2d", "activity × GC stratified (baseline)", "acq"),
        ("mix_neyman", "Neyman allocation over strata", "acq"),
        ("mix_powersd", "stochastic SD sampling", "acq"),
        ("mix_oracle", "largest true error (ceiling, uses labels)", "ceil"),
    ]
    head = [
        "Arm",
        "Genomic",
        "Designed OOD",
        "Random seqs",
        "Δ genomic",
        "Δ OOD",
        "Δ random",
        "Added seqs: mean label",
        "from random / motif / EvoAug",
    ]
    rows = []
    groups = {
        "control": "Control: 30k + 70k more genomic",
        "reservoir": "Add 70k from one reservoir (non-active)",
        "mix": "Even mix of random, motif and EvoAug (300k candidates)",
        "acq": "Selected from the mix by the 30k-genomic ensemble (active)",
        "ceil": "Ceiling",
    }
    for a, label, kind in ARMS:
        x = (al.get(a) or {}).get("first6")
        if not x:
            continue
        e = x["ensemble"]
        sel = (al.get(a) or {}).get("selection") or {}
        sf = sel.get("source_frac")
        vals = (
            [e[s] for s in SETS3]
            + [e[s] - ctl[s] for s in SETS3]
            + [sel.get("label_mean"), "/".join(f"{100 * v:.0f}" for v in sf) + " %" if sf else "–"]
        )
        rows.append((esc(label), vals, kind))
    t = table(head, rows, best=[0, 1, 2], groups=groups)
    img = fig_arms(d30)
    n_done = len(rows)
    return f"""<section id="topup"><h2>5 · Adding 70k sequences to 30k genomic: non-active against active</h2>
<p class="lead">The registered-report regime: start from a base set, add a batch in one round. Base = 30k genomic (replicate 1). The control adds 70k more genomic. Non-active arms add 70k from one reservoir. Active arms score a 300k candidate pool (100k each of random, motif CT-enriched and EvoAug) with the ensemble trained on the 30k base and keep 70k. Final models are trained on 100k with the first 6 menu configurations; differences under about 0.005 are within one-seed noise.</p>
{t}
<figure><img alt="top-up arms" src="{img}"><figcaption>Change in ensemble r against the +70k-genomic control, per evaluation set. Colour: orange = single reservoir, grey = random draw from the mix, green = selected by the ensemble.</figcaption></figure>
<p class="small">{n_done} of 20 arms had results at build time; arms without enough trained models are omitted rather than estimated.</p>
{rounds3_section(d30)}</section>"""


def rounds3_section(d30):
    r3 = d30.get("rounds3") or {}
    names = {"mean": "highest ensemble mean", "std": "highest ensemble SD"}
    rows = []
    for crit in ("mean", "std"):
        row = r3.get(crit)
        if not row:
            continue
        one, three = row.get("one_round"), row.get("three_round")
        for tag, x, lab in (("one", one, "1 round of 70k"), ("three", three, "3 rounds of 23,333")):
            sel = row["selection"]["one_round" if tag == "one" else "three_round"]
            if x:
                vals = [x["ensemble"][s] for s in SETS3] + [
                    sel["label_mean"],
                    "/".join(f"{100 * v:.0f}" for v in sel["source_frac"]) + " %",
                ]
            else:
                vals = [
                    None,
                    None,
                    None,
                    sel["label_mean"],
                    "/".join(f"{100 * v:.0f}" for v in sel["source_frac"]) + " %",
                ]
            rows.append(
                (
                    f"{names[crit]}: {lab}"
                    + ("" if x else " <span class='small'>(still training)</span>"),
                    vals,
                    crit,
                )
            )
    if rows:
        t = table(
            [
                "Selection",
                "Genomic",
                "Designed OOD",
                "Random seqs",
                "Added seqs: mean label",
                "from random / motif / EvoAug",
            ],
            rows,
            best=[0, 1, 2],
            groups={
                "mean": "Score the 300k candidates by ensemble mean",
                "std": "Score the 300k candidates by ensemble SD",
            },
        )
        jac = ", ".join(f"{names[c]} {r3[c]['selection']['jaccard']:.2f}" for c in r3)
    else:
        t, jac = (
            "<p class='lead'>These cells are still training; the table fills in when they finish.</p>",
            "",
        )
    return f"""<h3>One round of 70k against three rounds of 23,333</h3>
<p class="lead">Same 30k genomic base, same 300k candidate pool, same round-0 student (the 8 best architectures trained on the base), same final training (menu v1 first 6 configurations, patience 16, 100k sequences). In the three-round version the student is retrained on the base plus everything chosen so far (53,333 and 76,666 sequences) before each of rounds 2 and 3, and the already-chosen sequences are removed from the pool.</p>
{t}
{'<p class="small">Overlap (Jaccard) between the sequences chosen in one round and in three: ' + jac + ".</p>" if jac else ""}
<p>An earlier run of the same comparison with the older 8-architecture menu and two seeds per architecture (16 models) gave, on the genomic / designed / random sets: highest mean 0.8431 / 0.5978 / 0.8624 in one round against 0.8420 / 0.5926 / 0.8466 in three; highest SD 0.8466 / 0.5985 / 0.8643 in one round against 0.8451 / 0.5867 / 0.8618 in three. Three rounds were not better than one.</p>"""


def appendix(d31, inp):
    return f"""<section id="appendix"><h2>6 · Appendix</h2>
{decisions_section(inp)}
<h3>AutoResearch v2 menus against menu v1 and v0 (our per-D procedure)</h3>{ar2_compare(d31)}
{ar2_build_notes()}
<h3>Menu v1 against the older v0 menu on the same pools</h3>{v1v0(d31)}
<div class="note">Preliminary: one seed per configuration and one replicate pool throughout; the top-up and 100k cells use the first 6 menu configurations. Cells still training at build time are shown as dashes or omitted, never filled in.</div></section>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L = lambda f: json.load(open(os.path.join(a.inp, f)))
    d30, d31, d32 = L("data_oct30.json"), L("data_oct31.json"), L("data_oct32.json")
    findings = L("findings.json") if os.path.exists(os.path.join(a.inp, "findings.json")) else []
    find_html = "".join(
        f"<div><b>{esc(f['title'])}</b>{md_inline(f['text'])}</div>" for f in findings
    )
    snap = time.strftime("%b %d, %H:%M")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ALBench Results Walkthrough</title><style>{BASE_CSS}{EXTRA_CSS}</style></head><body><div class="wrap">
<header><h1>ALBench Results Walkthrough</h1>
<p class="sub">How we choose training sequences for a K562 MPRA library: the procedure, the evaluation sets, a worked AutoResearch example at 30k, scaling results for eight reservoir strategies, and a top-up experiment comparing non-active and active selection.</p>
<div class="chips"><span class="chip">snapshot <b>{esc(snap)}</b></span><span class="chip">preliminary: one seed, one replicate pool</span></div></header>
<nav><a href="#glance">At a glance</a><a href="#procedure">1 Procedure</a><a href="#evals">2 Eval sets</a><a href="#example">3 AutoResearch at 30k</a><a href="#scaling">4 Scaling</a><a href="#topup">5 Top-up</a><a href="#appendix">Appendix</a></nav>
<section id="glance"><h2>At a glance</h2><div class="find">{find_html}</div></section>
{intro_section()}
{evalsets_section(d32)}
<section id="example"><h2>3 · A worked AutoResearch example at D = 30k</h2>
<p class="lead">What was actually done for one cell, the genomic reservoir at 30k sequences: one search run, how runs become a menu, how the menu becomes a weighted ensemble, and what the newer controller-driven search did on the same cell.</p>
{run_section(d32)}{menu_section(d32)}{rounds_section(a.inp)}{ar2_example_section(d32, a.inp)}</section>
{scaling_section(d30, d32)}
{topup_section(d30)}
{appendix(d31, a.inp)}
</div></body></html>"""
    open(a.out, "w").write(page)
    print("wrote", a.out, len(page) // 1024, "KB")


if __name__ == "__main__":
    main()
