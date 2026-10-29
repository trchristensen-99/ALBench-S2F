# ruff: noqa
#!/usr/bin/env python
"""Add the slide-ready evidence section (finder + figures with plain-language captions) to the top of the Evidence Ledger page.

Reads the published ledger page (index.html as published, which carries the artifact skeleton), keeps its inner document unchanged and inserts after the header: a live-status strip, a "why did you do it this
way?" finder (searchable questions -> one-line answer -> slide), and the slides. New and updated rows in the parameter table point to the slides. Numbers in the captions of the new slides are read from
slide_figures/benchmarks/data_pk35.json, so a rebuild after more cells finish updates them.

    python scripts/analysis/build_ledger_deck.py --page <published index.html> --deck slide_figures/deck --figs slide_figures --out <dir>      (writes <dir>/index.html and <dir>/deck/*.png)
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import time

esc = html.escape

CSS = """
.deck{margin:26px 0 8px}
.dgrp{margin:30px 0 12px;font-size:12px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);font-weight:700}
.live{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0 4px}
.live .pill{border:1px solid var(--line);background:var(--surface);border-radius:8px;padding:8px 12px;font-size:13px;line-height:1.35}
.live .pill b{display:block;font-size:12px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted)}
.callout{border:1px solid var(--line);border-left:4px solid var(--accent);background:var(--surface);border-radius:10px;padding:14px 18px;margin:16px 0;max-width:96ch}
.callout h3{margin:0 0 6px;font-size:16px}.callout p{margin:6px 0}.callout ul{margin:6px 0 0;padding-left:18px}
.finder{margin:18px 0 8px}
.finder input{font:inherit;width:100%;max-width:560px;padding:9px 12px;border:1px solid var(--line);border-radius:8px;background:var(--surface);color:var(--ink)}
.finder input:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.qa{width:100%;border-collapse:collapse;font-size:13.5px}
.qa th{text-align:left;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);padding:8px 10px;border-bottom:1px solid var(--line)}
.qa td{vertical-align:top;padding:8px 10px;border-bottom:1px solid var(--line)}
.qa td:first-child{font-weight:600;min-width:260px;max-width:46ch}.qa td.ans{max-width:62ch}
.qa a.sl{white-space:nowrap;font-weight:600;text-decoration:none;border:1px solid var(--line);border-radius:6px;padding:2px 7px;margin-right:4px;display:inline-block}
.qa tr[hidden]{display:none}
.slide{background:var(--surface);border:1px solid var(--line);border-radius:10px;margin:0 0 20px;overflow:hidden;scroll-margin-top:12px}
.slide .sh{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:12px 16px 6px}
.slide .sn{font:700 12px var(--mono);background:var(--ink);color:var(--bg);border-radius:5px;padding:3px 8px}
.slide .sg{font-size:12px;color:var(--muted);letter-spacing:.04em;text-transform:uppercase}
.slide h3{margin:0;padding:2px 16px 10px;font-size:18px;line-height:1.3;text-wrap:balance}
.slide .fig{background:#fff;border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:8px;overflow-x:auto}
.slide .fig img{display:block;width:100%;height:auto;max-width:1180px;margin:0 auto;min-width:600px}
.slide .cap{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px 22px;padding:12px 16px 14px}
.slide .cap div{font-size:13.5px;line-height:1.5;color:var(--ink)}
.slide .cap b{display:block;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin-bottom:2px}
.slide .ft{display:flex;gap:14px;flex-wrap:wrap;padding:0 16px 12px;font-size:12.5px;color:var(--muted)}
.slide .ft a{color:var(--accent)}
.slide .ft button.dl{font:inherit;font-size:12.5px;border:1px solid var(--line);background:var(--bg);color:var(--ink);border-radius:6px;padding:3px 9px;cursor:pointer}
.slide .ft button.dl:hover{border-color:var(--accent)}
.slide .ft button.dl:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.jump{display:flex;flex-wrap:wrap;gap:8px;margin:6px 0 4px}.jump a{font-size:13px;border:1px solid var(--line);background:var(--surface);color:var(--ink);padding:5px 11px;border-radius:999px;text-decoration:none}
.jump a:hover{border-color:var(--accent)}
"""

FINDER_JS = """
<script>
(function(){var btns=[].slice.call(document.querySelectorAll('button.dl'));if(!btns.length||!window.claude||!claude.use)return;
claude.use('downloads').then(function(d){if(!d)return;btns.forEach(function(b){b.hidden=false;b.addEventListener('click',function(){
fetch(b.dataset.src).then(function(r){return r.blob()}).then(function(bl){return d.save({filename:b.dataset.src.split('/').pop(),data:bl})}).then(function(){b.textContent='Saved'}).catch(function(e){if(e&&e.code==='declined')return;b.textContent='Could not save: right-click the figure instead'})})})}).catch(function(){})})();
</script>
<script>
(function(){var q=document.getElementById('qsearch');if(!q)return;var rows=[].slice.call(document.querySelectorAll('table.qa tbody tr'));
q.addEventListener('input',function(){var t=q.value.trim().toLowerCase();rows.forEach(function(r){r.hidden=!!t&&r.textContent.toLowerCase().indexOf(t)<0})})})();
</script>
"""


def chip(status):
    lab = {
        "supported": "Supported",
        "provisional": "Provisional",
        "running": "Running",
        "open": "Open",
        "carried": "Carried over",
    }[status]
    return f'<span class="chip s-{status}">{lab}</span>'


def pm(d, k, nd=3):
    """'+0.070 (SE 0.004)' from a paired entry."""
    e = d[k]
    se = "" if e["se"] != e["se"] else f" (SE {e['se']:.{nd}f})"
    return f"{e['mean']:+.{nd}f}{se}"


def slides(P):
    """List of slide dicts. P = data_pk35.json."""
    pr, E, n = P["paired"], P["ensembles"], P["n_models"]
    c20 = pr.get("c20s10")
    c20n = n.get("c20s10", 0)
    ctx_done = c20n >= 14
    flank_numbers = (
        f"20 real bases each side: genomic test {pm(c20, 'genomic')}, random sequences {pm(c20, 'random_32k')}, experimental labels {pm(c20, 'genomic_measured')} on {c20['n']} of 15 architectures so far."
        if c20
        else "Arms are training."
    )
    if "c100s10" in pr and pr["c100s10"]["n"]:
        flank_numbers += f" 100 bases: {pm(pr['c100s10'], 'genomic')} on {pr['c100s10']['n']}."
    ema = pr.get("ema")
    ema_ens = E.get("ema"), E.get("ema__base_same_configs")
    ens_txt = ""
    if ema_ens[0] and ema_ens[1]:
        ens_txt = f" In an ensemble of the same {ema_ens[0]['n']} configurations the gain vanishes ({ema_ens[0]['genomic']:.4f} with averaging, {ema_ens[1]['genomic']:.4f} without)."
    forest_numbers = (
        f"Learning rate, batch size and weight decay each move a single model by at most 0.004 (seed noise: SE {P['seed_floor']['genomic']['se']:.4f}); AdamW instead of Muon costs {pm(pr['adamw'], 'genomic')}. "
        f"Shifting the window over real flank: {pm(pr['flank_shift'], 'genomic')}. Weight averaging: {pm(ema, 'genomic')} on {ema['n']} configurations.{ens_txt}"
        if ema
        else ""
    )
    pooled = E.get("pool_all_menu_arms")
    sw = {
        k: pr[k]["genomic"]["mean"]
        for k in (
            "c4s10",
            "c8s10",
            "c12s10",
            "c20s10",
            "c60s10",
            "c100s10",
            "c200s10",
            "c20s0",
            "c20s30",
        )
        if k in pr and pr[k]["n"] >= 2
    }
    sweep_txt = (
        "Students get the plasmid flank the assay and the oracle see, not the bare insert; every student before Oct 29 saw 200 bases with zeros beyond them. "
        + (
            f"The gain is a junction effect that saturates quickly: {sw.get('c4s10', float('nan')):+.3f} with 4 bases per side, {sw.get('c12s10', float('nan')):+.3f} with 12, {sw.get('c20s10', float('nan')):+.3f} with 20, and no more with 60, 100 or 200 ({sw.get('c60s10', float('nan')):+.3f}, {sw.get('c100s10', float('nan')):+.3f}, {sw.get('c200s10', float('nan')):+.3f}). "
            f"Use 20 bases and a +-10 shift over real flank (no shift {sw.get('c20s0', float('nan')):+.3f}, +-30 {sw.get('c20s30', float('nan')):+.3f})."
            if sw
            else ""
        )
    )
    ctx_models = [m for m in P["models"] if m["arm"] == "c20s10"]
    allctx = [m for m in P["models"] if m["arm"][0] == "c" and m["arm"][1].isdigit()]
    best_ctx = max(allctx, key=lambda m: m["val"]) if allctx else None
    single_txt = (
        f"Flank-context singles (20 bases, {len(ctx_models)} of 15 architectures so far): validation {min(m['val'] for m in ctx_models):.3f} to {max(m['val'] for m in ctx_models):.3f}, test {min(m['genomic'] for m in ctx_models):.3f} to {max(m['genomic'] for m in ctx_models):.3f}; best by validation {best_ctx['genomic']:.3f} (val {best_ctx['val']:.3f}). "
        "Best earlier single model of ours: val 0.840, test 0.754 by validation (best test of any single 0.774)."
        if ctx_models and best_ctx
        else ""
    )
    S = []
    S.append(
        dict(
            id="s1",
            grp="A. What the student sees and how it trains",
            status="running" if not ctx_done else "supported",
            img="deck/deck_flank_context.png",
            own_title=True,
            read="Each point is the average change in test r when the same architecture, trained on the same 30k genomic sequences with the same seed, is also shown real plasmid sequence on each side of its 200-bp insert (x). Blue and purple are scored against oracle labels, open orange against the experimental measurements. The grey band is seed noise.",
            so=sweep_txt,
            nums=flank_numbers,
            change="The gain shrinks to the +0.008 of the zero-padded real-flank shift once all 15 architectures finish, or the reservoir ranking flips (slide S5).",
            links=[("#loop", "training loop entry"), ("#student-input", "parameter table")],
            qs="flank context plasmid input collaborator gap why better student sees",
        )
    )
    S.append(
        dict(
            id="s2",
            grp="A. What the student sees and how it trains",
            status="running" if not ctx_done else "provisional",
            img="deck/deck_single_models.png",
            own_title=True,
            read="Every dot is one trained model on the 30k genomic pool: x = validation r, y = test r. The red star is the collaborator's best single model as he reported it. Models with flank context have the same validation-to-test gap (about 0.04) as his; our earlier models have 0.07 to 0.08.",
            so="Treat the collaborator's lead as explained first by input context, not by more search rounds. His files and code are still needed to confirm that his val/test sets and labels match ours.",
            nums=single_txt,
            change="His validation and test files turn out to differ from ours, or his training recipe adds something large on top of context.",
            links=[],
            qs="collaborator peter koo 0.8165 single model replicate pi gap",
        )
    )
    S.append(
        dict(
            id="s3",
            grp="A. What the student sees and how it trains",
            status="supported",
            img="deck/deck_what_moves_a_model.png",
            own_title=True,
            read="One row per change to the unchanged menu (20 architectures, same seed). The dot is the paired change in genomic test r, the bar its standard error over architectures, the grey band twice the seed-to-seed noise. Only rows clear of the band matter.",
            so="Keep one shared recipe (Muon, lr 2e-3, wd 1e-4, batch 256, one-cycle) for every student: moving lr, batch size or weight decay changes nothing measurable. Spend effort on what the student sees and on weight averaging, not on recipe search.",
            nums=forest_numbers,
            change="A recipe change that clears the noise band on at least two reservoirs; or weight averaging that survives in ensembles.",
            links=[("#recipe", "epochs and early stopping")],
            qs="recipe learning rate batch size weight decay adamw muon optimizer shared weight averaging ema batchnorm hyperparameters",
        )
    )
    S.append(
        dict(
            id="s4",
            grp="A. What the student sees and how it trains",
            status="provisional",
            img="deck/deck_benchmark_ladder.png",
            own_title=True,
            read="Genomic test r when 30k genomic sequences are the only training data. Filled dots are scored against the oracle labels, open diamonds against the experimental labels. Grey = published MPRA-LegNet, blue = our earlier menu, orange = students that see the flank, red = collaborator.",
            so="Compare every method at the same 30k genomic training set and the same test sets. AlphaGenome-FT-MPRA (0.96 oracle, 0.92 experimental) is left off the axis because it fine-tunes the oracle's own backbone.",
            nums=(
                f"Every 30k model of ours pooled: {pooled['genomic']:.3f} oracle, {pooled['genomic_measured']:.3f} experimental ({pooled['n']} models). The old menu: 0.782 / 0.743. Published-recipe LegNet ensemble: 0.750."
                if pooled
                else "Arms are training."
            ),
            change="The collaborator's test set is not ours (his 0.8165 and 0.8317 are then not comparable).",
            links=[("#ens", "ensembling entry")],
            qs="benchmark legnet alphagenome compare baseline ensemble oracle experimental labels teacher distillation collaborator",
        )
    )
    S.append(
        dict(
            id="s5",
            grp="D. Fairness across strategies and sizes",
            status="running",
            img="deck/deck_fairness_flank.png",
            own_title=True,
            read="Each point is one reservoir (training-sequence generator) at 30k: x = ensemble r of the same architectures without context, y = with 20 flank bases. Above the diagonal = gains from context; the rank correlation says whether the order of strategies is preserved.",
            so="Adopt the context only if it lifts every strategy about equally, so the comparison between strategies stays fair. If it favours some strategies, the corpus ranking has to be redone with context.",
            nums="Ten reservoirs, 15 architectures each, training now.",
            change="The rank correlation falls below about 0.9 or one family gains much less than the others.",
            links=[("#fam", "reservoir families entry")],
            qs="fairness reservoir ranking strategies flank context same for all unbiased",
        )
    )
    return S


def finder_rows():
    return [
        (
            "Why was our old student so much worse than the collaborator's?",
            "It saw the 200-bp insert alone; the oracle and the assay see it inside the plasmid. Adding 20 real bases of flank each side lifts a single model by about +0.07.",
            ["s1", "s2"],
        ),
        (
            "Why do all students share one training recipe?",
            "Learning rate, batch size and weight decay each change a single model by at most 0.004 (noise 0.001); only AdamW instead of Muon hurts (-0.023). The proposer also converges on it.",
            ["s3", "s6"],
        ),
        (
            "Is weight averaging worth adding?",
            "It adds about +0.01 to a single model and nothing measurable to an ensemble of the same configurations; cheap, so a harness option, not a result to lean on.",
            ["s3"],
        ),
        (
            "Why patience 16?",
            "+0.0044 (SE 0.0006) over patience 8, positive on 20 of 21 replicate pools, for 1.9 times the training time; 24 and 32 are inconsistent.",
            ["s7"],
        ),
        (
            "Why Claude Sonnet 5.5 and not Opus?",
            "At equal model counts Opus is ahead by under two standard errors (0.7788 against 0.7734).",
            ["s9"],
        ),
        (
            "Why a prompt with a guidance block?",
            "Removing it costs 0.016 to 0.020 in ensemble r; the explore, reflect and phase variants add nothing.",
            ["s8"],
        ),
        (
            "Why a separate coder but no evaluator agent?",
            "The separate coder adds +0.005 over the single-call loop; the evaluator and reflector add at most +0.005 with intervals through zero.",
            ["s10"],
        ),
        (
            "Why 10 rounds of 2 proposals?",
            "10 to 15 rounds adds +0.002; 6 proposals per round trails 2 per round at equal model counts.",
            ["s11", "s12"],
        ),
        (
            "Does an explore-then-exploit manager help?",
            "+0.006 (SE 0.003, 4 of 6 runs) at 30 models; no difference at 20. Not enough to call.",
            ["s13"],
        ),
        (
            "One long run or several short ones?",
            "At 60 models: 1 run 0.7740, 3 runs of 20 0.7768, 5 runs of 12 0.7756; several runs mainly cut the spread.",
            ["s14"],
        ),
        (
            "How many architectures and runs go into the ensemble?",
            "30k and 100k are within 0.002 of the full menu from 10 to 11 architectures; 10k still gains at 20. Run pooling is concave.",
            ["s15", "s16"],
        ),
        (
            "Is the menu biased toward the genomic search that built it?",
            "A search on the target reservoir ties the carried menu (menu minus own search +0.002 to +0.023); source of the architectures changes the ensemble by at most 0.007.",
            ["s17"],
        ),
        (
            "Is a tuned LegNet enough on its own?",
            "A tuned LegNet sits at the median of the searched architectures; ensembles of 20 tie (0.782 / 0.782 at 30k).",
            ["s18", "s4"],
        ),
        (
            "Do reservoir rankings depend on what the student sees?",
            "Being measured now: the same architectures on ten reservoirs with and without context.",
            ["s5"],
        ),
        (
            "How does accuracy scale with dataset size?",
            "1 - r and MSE are straight in log-log; exponents 0.18 to 0.24 on the genomic test at 10k to 100k.",
            ["s19", "s20"],
        ),
        (
            "Why nested, independently drawn pools?",
            "Nested prefixes vary only the amount of data; three independent pools beat three seeds on one pool in 12 of 12 comparisons.",
            ["s21"],
        ),
        (
            "Why oracle labels, and what about experimental ones?",
            "Oracle labels let any generated sequence be labelled; held-out r against experiment is 0.931. Students also correlate with experiment (0.74 to 0.80) and the ordering is the same.",
            ["s4"],
        ),
        (
            "Which validation set and why?",
            "V0c: 29,449 real chr19/21/X sequences labelled by the current oracle; no other variant gave a reliable win.",
            [],
        ),
        (
            "Why a non-negative ElasticNet stack?",
            "Ties greedy selection and beats the plain mean by 0.002 to 0.018 across 12 cells.",
            [],
        ),
        (
            "Why one acquisition round?",
            "One round of 70k is at least as good as three of 23k; selection by ensemble mean or SD beats random by about 0.01 (genomic) and 0.12 (designed).",
            [],
        ),
        (
            "What if the search ran for many more rounds (the collaborator's 200 models)?",
            "Not tested yet; long runs at 10k and 30k with the context harness are queued after the context results.",
            [],
        ),
    ]


def build(page, deck_dir, figs_dir, out, P):
    raw = open(page).read()
    i = raw.index("<!doctype html>", 20)  # the inner document starts at its own doctype
    inner = raw[i:]
    j = inner.index(
        "</body></html>"
    )  # the first one closes the page; a second belongs to the skeleton
    inner = inner[: j + len("</body></html>")]
    # CSS
    inner = inner.replace("</style></head>", CSS + "</style></head>", 1)
    S = slides(P)
    # ---- slide registry: existing figures (already published under figs/ or copied from slide_figures)
    ex = EXISTING
    allS = S + ex
    allS = [s for s in allS if s.get("img")]
    for k, s in enumerate(allS, 1):
        s["num"] = f"S{k}"
    idmap = {s["id"]: s["num"] for s in allS}
    # figure files
    os.makedirs(os.path.join(out, "deck"), exist_ok=True)
    for s in allS:
        src = s.get("src")
        if src:
            shutil.copy(os.path.join(figs_dir, src), os.path.join(out, s["img"]))
        elif s["img"].startswith("deck/"):
            shutil.copy(
                os.path.join(deck_dir, os.path.basename(s["img"])), os.path.join(out, s["img"])
            )
    # live status strip
    n = P["n_models"]
    pills = [
        (
            "Flank context arms",
            f"{sum(v for k, v in n.items() if k.startswith('c') and k[1].isdigit())} models of 170 trained",
        ),
        (
            "Weight averaging arms",
            f"{sum(n.get(k, 0) for k in ('ema', 'bn', 'emabn', 'ema999bn'))} of 80",
        ),
        ("LegNet reference arms", f"{sum(v['n'] for v in P['legnet']['arms'].values())} of 63"),
        (
            "Fairness (10 reservoirs)",
            f"{sum(1 for v in P.get('fairness', {}).values() if v['complete'])} of 10 complete",
        ),
        ("Loop with context", "built and under test; long runs (10k, 30k) queued next"),
    ]
    live = "".join(f'<div class="pill"><b>{esc(a)}</b>{esc(b)}</div>' for a, b in pills)
    # finder
    rows = []
    for q, a, sl in finder_rows():
        links = (
            " ".join(f'<a class="sl" href="#{x}">{idmap[x]}</a>' for x in sl if x in idmap)
            or '<span class="note">ledger below</span>'
        )
        rows.append(f"<tr><td>{esc(q)}</td><td class='ans'>{esc(a)}</td><td>{links}</td></tr>")
    finder = (
        '<div class="finder"><input id="qsearch" type="search" placeholder="Type a question or a word: recipe, patience, Opus, flank, LegNet, scaling, fairness ..." aria-label="Search the questions"></div>'
        '<div class="tblwrap"><table class="qa"><thead><tr><th>If someone asks</th><th>The short answer</th><th>Slide</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
    )
    # slide groups in order
    order = []
    for s in allS:
        if s["grp"] not in order:
            order.append(s["grp"])
    jump = "".join(f'<a href="#g{k}">{esc(g)}</a>' for k, g in enumerate(order))
    body = []
    for k, g in enumerate(order):
        body.append(f'<h2 class="dgrp" id="g{k}">{esc(g)}</h2>')
        for s in [x for x in allS if x["grp"] == g]:
            title = "" if s.get("own_title") else f"<h3>{esc(s['title'])}</h3>"
            links = " ".join(f'<a href="{h}">{esc(t)}</a>' for h, t in s.get("links", []))
            body.append(
                f'<section class="slide" id="{s["id"]}"><div class="sh"><span class="sn">{s["num"]}</span><span class="sg">{esc(s["grp"].split(". ", 1)[-1])}</span>{chip(s["status"])}</div>{title}'
                f'<div class="fig"><img src="{s["img"]}" alt="{esc(s.get("alt") or s.get("title") or s["id"])}" loading="lazy"></div>'
                f'<div class="cap"><div><b>How to read it</b>{esc(s["read"])}</div><div><b>What we do because of it</b>{esc(s["so"])}</div>'
                f"<div><b>Numbers</b>{esc(s['nums'])}</div><div><b>What would change our mind</b>{esc(s['change'])}</div></div>"
                f'<div class="ft"><button class="dl" type="button" data-src="{s["img"]}" hidden>Save PNG ({s.get("size", "2667 x 1500, 16:9")})</button>{links}</div></section>'
            )
    stamp = time.strftime("%Y-%m-%d %H:%M %Z")
    deck = (
        '<section class="deck" id="deck"><h2 class="dgrp">What changed on Oct 29</h2>'
        '<div class="callout"><h3>The student was not seeing what the oracle sees.</h3>'
        "<p>Students trained on the 200-bp insert alone, with zeros beyond it, while every label comes from an oracle that reads the insert inside the plasmid. Showing the student 20 real plasmid bases on each side raises a single model's test r by about 0.07 (slides S1 to S3) and brings single models to the range of the collaborator's best search (S2, S4).</p>"
        "<ul><li>The earlier menu, recipe, patience, ensemble and scaling results were all obtained without the context; whether they carry over is being measured (S5 and the live strip below).</li>"
        "<li>The AutoResearch loop now takes the context as an option (built, under test); long runs at 10k and 30k follow.</li></ul></div>"
        f'<div class="live">{live}</div>'
        '<h2 class="dgrp">Why did you do it this way? Find the answer</h2>'
        + finder
        + f'<h2 class="dgrp">The evidence in slides (PowerPoint-ready PNGs)</h2><div class="jump">{jump}</div>'
        + "".join(body)
        + f'<p class="note">Slide figures regenerate from the experiment outputs with scripts/analysis/make_deck_figures.py; captions for S1 to S5 read their numbers from data_pk35.json. Built {stamp}.</p></section>'
        + FINDER_JS
    )
    # header text and stamp
    inner = inner.replace("</header>", "</header>" + deck, 1)
    inner = re.sub(
        r'<div class="stamp">[^<]*</div>',
        f'<div class="stamp">slides and answers as of {stamp} &middot; parameter table and cards as of 2026-10-27</div>',
        inner,
        count=1,
    )
    inner = inner.replace(
        "Every choice in the training-corpus pipeline, the experiment that justifies it, and how much weight it can bear.",
        "Every choice in the training-corpus pipeline, the experiment that justifies it, and how much weight it can bear: start with the slides and the question finder, details below.",
        1,
    )
    # parameter-table rows for the new choices
    new_rows = (
        '<tr><td><a href="#s1" id="student-input">What the student sees (input context)</a></td><td>200-bp insert plus real plasmid flank on each side (20 bases to start; the sweep to 200 is running). Training window shifts over real flank. Architectures must accept any input length divisible by 8.</td>'
        '<td>Paired by architecture and seed: about +0.07 test r at 20 bases per side (slide S1); single models reach the collaborator\'s range (S2).</td><td><span class="chip s-running">Running</span></td>'
        "<td>Which context size is enough; whether strategy rankings are unchanged (S5); five of the 20 menu architectures are hard-wired to 200 positions and need rewriting.</td></tr>"
        '<tr><td><a href="#s3">Weight averaging and BatchNorm recalculation</a></td><td>Off by default; available as harness switches.</td>'
        '<td>+0.01 on a single model, no measurable gain in an ensemble of the same configurations (S3).</td><td><span class="chip s-provisional">Provisional</span></td><td>Arms still training; LegNet-style reference under the same switches.</td></tr>'
    )
    anchor = '<tr class="gr"><td colspan="5">4. The menu and its retraining</td></tr>'
    assert anchor in inner
    inner = inner.replace(anchor, new_rows + anchor, 1)
    # wrap back into the skeleton the artifact service adds itself: publish the inner document only
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "index.html"), "w").write(inner)
    print("wrote", os.path.join(out, "index.html"), len(inner), "bytes;", len(allS), "slides")
    return allS


EXISTING = []


def _ex(**kw):
    EXISTING.append(kw)


# ---- existing figures (copied from slide_figures or already published in figs/) with captions from the ledger and the slide plan
_ex(
    id="s6",
    grp="A. What the student sees and how it trains",
    status="supported",
    src="justification/justify_recipes.png",
    img="deck/justify_recipes.png",
    size="2438 x 989",
    title="The recipe the menu shares is the empirical winner of 1,712 proposed recipes, but the proposer's settings come as a package",
    read="Left: share of all candidates (grey) and of each run's best three (orange) that use the menu's setting. Middle: within each of 84 runs, Muon minus AdamW validation r. Right: how the proposer drifts toward the setting as it sees results.",
    so="Share the recipe across students. Muon beats AdamW within 82 of 84 runs (+0.021), but Muon runs use weight decay 1e-4 and AdamW 0.01 or more, so the optimizer and the decay are confounded (slide S3 separates the recipe knobs).",
    nums="Top 3 of each run: 91% Muon, 96% lr 2e-3, 88% wd 1e-4, 94% batch 256, 98% one-cycle. The recipe was not re-tuned per dataset size except patience and epochs.",
    change="A recipe arm that beats the shared recipe by more than the noise band on two reservoirs.",
    links=[("#recipe", "epochs and early stopping")],
)
_ex(
    id="s7",
    grp="A. What the student sees and how it trains",
    status="supported",
    src="justification/justify_patience.png",
    img="deck/justify_patience.png",
    size="2363 x 894",
    title="Patience 16 beats patience 8 on 20 of 21 replicate pools",
    read="Paired difference in ensemble r (patience 16 minus 8) per replicate pool; the bar is the mean and its standard error; cost shown as training time.",
    so="Stop a model after 16 epochs without a validation gain of 1e-4 (the previous standard was 8).",
    nums="+0.0044 (SE 0.0006) over patience 8 at 1.9 times the training time; patience 24 and 32 are inconsistent.",
    change="A second seed that shows the gain is smaller than the noise floor of 0.004.",
    links=[("#recipe", "epochs and early stopping")],
)
_ex(
    id="s8",
    grp="B. How the search is run",
    status="supported",
    img="figs/prompt_arms.png",
    size="published figure",
    title="A guidance block of known-good pieces in the prompt is worth about 0.02; explore, reflect and phase variants add nothing",
    read="Ensemble r of searches run with different prompts, one bar per prompt arm; the longest bar is the default.",
    so="Use the single default prompt with the guidance block, the ensemble-aware note and the reference LegNet; give the proposer validation feedback only.",
    nums="At 50 models: guidance + phases 0.773, guidance only 0.772, nothing 0.753. Explore style: -0.0005 [-0.0029, +0.0018] pooled over 51 paired runs on five reservoirs.",
    change="A prompt variant that beats the default by more than 0.005 on pooled runs.",
    links=[("#prompt", "prompt entry")],
)
_ex(
    id="s9",
    grp="B. How the search is run",
    status="supported",
    img="figs/proposer_models.png",
    size="published figure",
    title="Sonnet 5.5 is as good a proposer as Opus 5.5 at equal model counts",
    read="Ensemble r of three runs per proposer model, equal numbers of models.",
    so="Use Claude Sonnet 5.5 for proposing and coding: it costs less and Opus is ahead by under two standard errors.",
    nums="Sonnet 5.5 0.7734, Sonnet 4.5 0.7682, Opus 5.5 0.7788.",
    change="A future model that clears the noise band.",
    links=[("#proposer", "proposer entry")],
)
_ex(
    id="s10",
    grp="B. How the search is run",
    status="supported",
    src="justification/justify_evaluator_agent.png",
    img="deck/justify_evaluator_agent.png",
    size="2699 x 1045",
    title="A separate coder helps; an evaluator that filters proposals adds little",
    read="Ensemble r against the default single-call loop for each arm (coder, coder + evaluator, random pick of the same six ideas, reflector), per reservoir, with intervals.",
    so="Run the multi-agent loop with a separate coder and a coordinator that picks 2 of 6 ideas; skip the extra reflection stages.",
    nums="Coder + evaluator beats the default call (+0.005 / +0.019 / +0.014) and a random pick of six; over the coder alone it adds +0.000 / +0.005 / +0.004, intervals include zero.",
    change="An evaluator variant that beats coder-only by more than 0.005 across reservoirs.",
    links=[("#loopels", "loop entry")],
)
_ex(
    id="s11",
    grp="B. How the search is run",
    status="supported",
    src="justification/justify_rounds.png",
    img="deck/justify_rounds.png",
    size="2964 x 807",
    title="Ten rounds are enough: rounds 10 to 15 add less than the noise floor",
    read="Gain of one run's ensemble (and of the pooled menu) as rounds are added; the grey line is the noise floor.",
    so="Ten rounds per search (20 models with 2 per round).",
    nums="10 to 15 rounds adds +0.0023 to one run's ensemble and +0.0018 to the pooled menu, below the 0.004 noise floor.",
    change="Long runs (50 rounds, queued) beat the 10-round ensemble by more than the noise on a fresh pool.",
    links=[("#sched", "rounds entry")],
)
_ex(
    id="s12",
    grp="B. How the search is run",
    status="supported",
    src="justification/justify_proposals_per_round.png",
    img="deck/justify_proposals_per_round.png",
    size="2165 x 934",
    title="Two proposals per round beat six and beat one at an equal number of trained models",
    read="Ensemble r against the number of trained models for 1, 2 and 6 proposals per round; reservoirs are the replicates.",
    so="Two proposals per round.",
    nums="At an equal model count 6 per round trails 2 per round by 0.014 at 6 models and 0.010 at 12, and is within 0.003 from 24 on; serial (1 per round) is about 0.03 behind where it finished.",
    change="A larger n of replicates showing 6 per round ahead at 24+ models.",
    links=[("#sched", "rounds entry")],
)
_ex(
    id="s13",
    grp="B. How the search is run",
    status="provisional",
    img="deck/deck_manager.png",
    own_title=True,
    size="2667 x 1500, 16:9",
    read="Six runs per arm on the same six replicate pools; dots are the run's ensemble r, grey lines join paired runs, the black diamond is the mean. Right: the paired differences with their standard errors; grey band = noise.",
    so="Do not add the explore-then-exploit manager yet: +0.006 with a standard error of 0.003 is inside what a single run can move by. It does change what is proposed (more distinct recipes, many more U-Net-like models) but not the predictions.",
    nums="Schedule 0.7762 vs coordinator-only 0.7700 (paired +0.0062, SE 0.0032, 4 of 6 runs); at 20 models no difference; prediction correlation stays 0.86 to 0.87.",
    change="More replicates with the context harness show the gain exceeding 0.005 consistently.",
    links=[],
)
_ex(
    id="s14",
    grp="B. How the search is run",
    status="provisional",
    src="justification/justify_runs_vs_models.png",
    img="deck/justify_runs_vs_models.png",
    size="2952 x 1360",
    title="At 60 models, one run, three runs of 20 and five runs of 12 differ by 0.002 to 0.003 in the mean; several runs mostly cut the spread",
    read="Ensemble r of random draws of runs at a fixed total of 60 models; the bar is the mean and the whisker the spread across draws.",
    so="Several independent runs of about 20 models, each on its own pool, rather than one long run.",
    nums="1 run 0.7740 +- 0.0055, 3 runs of 20 0.7768 +- 0.0030, 5 runs of 12 0.7756 +- 0.0010 (spread understated for many runs: draws share 8 runs).",
    change="The 200-model lineage search of the collaborator, once reproduced here, beats pooled shorter runs on a fresh pool.",
    links=[("#runcount", "run-count entry")],
)
_ex(
    id="s15",
    grp="C. How the ensemble is built",
    status="supported",
    src="justification/justify_models_plateau.png",
    img="deck/justify_models_plateau.png",
    size="2273 x 961",
    title="30k and 100k are within 0.002 of the full menu from 10 to 11 architectures; 10k still gains at 20",
    read="Ensemble r as the number of architectures K grows, with random-draw bands; the dashed line is the full menu.",
    so="Keep all 20 architectures at 10k and 30k; about 12 from 300k up for cost.",
    nums="K = 12 loses 0.0010 on average (worst 0.0042), K = 8 0.0024 (worst 0.0060) at 30k to 300k.",
    change="Menus built from more runs that plateau later.",
    links=[("#reducedall", "what is retrained entry")],
)
_ex(
    id="s16",
    grp="C. How the ensemble is built",
    status="provisional",
    src="justification/justify_runs_plateau.png",
    img="deck/justify_runs_plateau.png",
    size="2646 x 870",
    title="Pooling independent runs is concave: six to eleven runs reach 90% of the gain",
    read="Ensemble r of pooled runs against the number of runs, at 10k, 30k and 100k.",
    so="Pool several independent runs per reservoir; the menu was distilled from 15.",
    nums="Three runs sit 0.005 to 0.012 below the ensemble of all runs.",
    change="Different curves on other reservoirs.",
    links=[("#runcount", "run-count entry")],
)
_ex(
    id="s17",
    grp="C. How the ensemble is built",
    status="supported",
    img="figs/unseen_search_vs_menu.png",
    size="published figure",
    title="A frozen menu matches a fresh search on every reservoir we tried",
    read="Menu minus the reservoir's own three-run search at equal model counts, per reservoir and size.",
    so="Search once on genomic data and retrain the menu on every reservoir; do not search per reservoir.",
    nums="Menu minus own search: 10k +0.007 to +0.023, 30k +0.002 to +0.015, 100k +0.005 to +0.006; difference of differences -0.008 to +0.009.",
    change="A reservoir family where its own search beats the menu by more than 0.005.",
    links=[("#archsource", "architecture-source entry"), ("#unseen", "unseen families entry")],
)
_ex(
    id="s18",
    grp="C. How the ensemble is built",
    status="supported",
    img="figs/arch_vs_legnet.png",
    size="published figure",
    title="A tuned LegNet sits at the median of the searched architectures",
    read="Rank of the tuned LegNet among the retrained menu entries at each size.",
    so="Free-form search buys diversity for the ensemble, not a better single architecture.",
    nums="Mean rank 10 of 33 at 10k, 15 of 33 at 30k, 14 of 33 at 100k, 16 of 33 at 300k.",
    change="Architectures that beat LegNet across sizes by more than noise.",
    links=[("#legref", "LegNet entry")],
)
_ex(
    id="s19",
    grp="D. Fairness across strategies and sizes",
    status="provisional",
    src="scaling/scaling_oldmenu_three.png",
    img="deck/scaling_oldmenu_three.png",
    size="3000 x 1080",
    title="Accuracy keeps rising to 1M sequences, roughly but not exactly linear in log-log",
    read="Genomic test r against training-set size for uniform random, motif (CT-enriched) and genomic sequences, old menu.",
    so="Measure anchors at 600k and 1M instead of extrapolating; the 10k to 300k power law overshoots them by about 0.01.",
    nums="Genomic is capped at 315k sequences; random and motif anchors measured at 600k and 1M.",
    change="Anchors that fall on the extrapolated line.",
    links=[("#anchor", "scaling beyond 300k entry")],
)
_ex(
    id="s20",
    grp="D. Fairness across strategies and sizes",
    status="provisional",
    src="scaling/scaling_1mr_loglog.png",
    img="deck/scaling_1mr_loglog.png",
    size="3000 x 1040",
    title="Eleven reservoirs at 10k, 30k and 100k in log space",
    read="1 - r against training-set size on log axes, one line per reservoir; steeper = gains faster with data.",
    so="Rank reservoirs by level at the deployed size; the rate differs by evaluation set.",
    nums="Genomic test r at 100k: genomic 0.830, motif CT 0.833, EvoAug 0.832, Evo2 0.832, mutagenesis 0.828, motif core 0.828, GC sweep 0.805, random 0.804, D3-64 0.819, D3-256 0.804, Zoonomia 0.769.",
    change="Context changes the ordering (slide S5).",
    links=[("#fam", "reservoir families entry")],
)
_ex(
    id="s21",
    grp="D. Fairness across strategies and sizes",
    status="supported",
    img="figs/independent_pools_30k.png",
    size="published figure",
    title="Three independent pools beat three seeds on one pool",
    read="Ensemble r at equal model counts for pooled independent pools against pooled seeds of one pool.",
    so="Give each independent run its own labelled pool; nest the sizes inside it.",
    nums="+0.013, +0.004, +0.006, +0.014 (genomic, random, motif, EvoAug); 12 of 12 comparisons positive.",
    change="A fourth reservoir where seeds win.",
    links=[("#indep", "independent pools entry")],
)
_ex(
    id="s22",
    grp="E. How we compare",
    status="provisional",
    src="benchmarks/bench_genomic_ensembles.png",
    img="deck/bench_genomic_ensembles.png",
    size="3209 x 1771",
    title="MPRA-LegNet and AlphaGenome-FT-MPRA against our earlier ensemble at 30k and 10k",
    read="Genomic, designed and random test r for each baseline at 30k (top) and 10k (bottom); error bars are SD over seeds.",
    so="Use the measured-label column when comparing with AlphaGenome: its oracle-label score distils the oracle's own backbone.",
    nums="30k genomic: menu x20 0.782, tuned LegNet x20 0.782, published-recipe LegNet x20 0.750; AlphaGenome-FT S2 about 0.96 against oracle, 0.917 against experiment.",
    change="A second oracle family.",
    links=[("#oracle", "oracle entry")],
)

_ARGS_NOTE = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", required=True)
    ap.add_argument("--deck", default="slide_figures/deck")
    ap.add_argument("--figs", default="slide_figures")
    ap.add_argument("--data", default="slide_figures/benchmarks/data_pk35.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = json.load(open(a.data))
    build(a.page, a.deck, a.figs, a.out, P)


if __name__ == "__main__":
    main()
