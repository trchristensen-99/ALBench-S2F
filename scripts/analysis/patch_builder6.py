# ruff: noqa
"""Sixth patch for build_justification_html.py: Oct 25-26 results, the procedure-specification table, the coverage matrix and the extra assumption rows."""

import sys

p = sys.argv[1]
s = open(p).read()


def rep(a, b, count=1):
    global s
    assert s.count(a) == count, (s.count(a), a[:90])
    s = s.replace(a, b)


# 1) cards
rep(
    "    return oct22_cards(d) + oct23_cards(d) + oct25_cards(d) + oct25b_cards(d) + cards\n",
    "    from ledger_oct26 import oct26_cards\n    return oct22_cards(d) + oct23_cards(d) + oct25_cards(d) + oct25b_cards(d) + oct26_cards(d) + cards\n",
)
# 2) extra assumption rows
rep(
    '    ov = _o22_overrides(d)[0]\n    for r_ in rows:\n        r_.update(ov.get(r_["a"], {}))\n    return rows',
    '    from ledger_oct26 import extra_assumption_rows\n    rows += extra_assumption_rows(d)\n    ov = _o22_overrides(d)[0]\n    for r_ in rows:\n        r_.update(ov.get(r_["a"], {}))\n    return rows',
)
# 3) group order
rep(
    'ORDER = ["Experiments launched Oct 22", "Results from the Oct 22 program",',
    'ORDER = ["Experiments launched Oct 22", "Results from the Oct 22 program", "Results from the Oct 25-26 program",',
)
# 4) spec section before the assumptions table; refreshed recipe paragraph
rep(
    """    rows = build_assumptions(d)
    parts = ['<section class="choices" id="assumptions">""",
    """    from ledger_oct26 import coverage_html, spec_rows

    spec = spec_rows(d)
    sp = ['<section class="choices" id="spec"><h2 class="grp">The procedure, parameter by parameter</h2>'
          '<p class="goal">Every setting of the HP-optimisation and ensembling pipeline in the order it is applied, what the recipe currently uses, the experiment behind it, and what is still open. '
          'The status chip is a judgement; the link goes to the evidence card.</p>'
          '<div class="tblwrap"><table><thead><tr><th>Parameter</th><th>Setting in the recipe</th><th>Evidence</th><th>Status</th><th>Still open</th></tr></thead><tbody>']
    last_ = None
    for r in spec:
        if r["stage"] != last_:
            sp.append(f'<tr class="gr"><td colspan="5">{esc(r["stage"])}</td></tr>')
            last_ = r["stage"]
        link = f'<a href="#{r["card"]}">{esc(r["p"])}</a>'
        sp.append(f'<tr><td>{link}</td><td>{esc(r["s"])}</td><td>{esc(r["ev"])}</td><td><span class="chip s-{r["st"]}">{STATUS[r["st"]][0]}</span></td><td>{esc(r["open"])}</td></tr>')
    sp.append("</tbody></table></div>")
    sp.append('<h3 id="coverage" class="sub3">Which reservoirs at which D</h3>' + coverage_html(d) + "</section>")
    out.append("".join(sp))
    rows = build_assumptions(d)
    parts = ['<section class="choices" id="assumptions">""",
)
rep(
    """One search on genomic data (three independent runs, each on its own pool, about 20 to 30 models per run, top 3 to 5 kept), retrained at every dataset size and reservoir with three seeds, combined by a non-negative ElasticNet ensemble. '
             'Nothing more elaborate has helped beyond noise yet; the running experiments try to overturn it with per-reservoir searches, mixed searches and mixed corpora.""",
    """One search at 30k on genomic data (several independent runs, each on its own pool: three runs fall 0.005 to 0.012 short of the ensemble of 10 to 15 runs, and 6 to 11 reach 90% of that gain; 20 models per run with the default prompt, the top 3 of each kept), the pooled 33-architecture menu retrained at every size and reservoir (about 8 architectures from 300k up, three seeds at 100k and below), combined by a non-negative ElasticNet ensemble fitted on validation predictions. '
             'Nothing more elaborate has beaten it by more than noise: per-reservoir and mixed searches, context hand-off, explore and reflect prompts, adaptive acquisition rounds and a full selection ensemble all tie. The coordinator stage (+0.005 on genomic and random 30k) did not hold up across the grid (pooled difference about zero).""",
)
# 5) css
rep(
    'CSS = """',
    'CSS = """\n.sub3{margin:22px 0 6px;font-size:15px}\ntable.cov{width:auto}\ntable.cov td.cv{text-align:center;min-width:46px}\ntd.cv-x{background:rgba(42,125,79,.16)}\ntd.cv-q{background:rgba(106,79,163,.16)}\ntd.cv-pool{background:rgba(217,130,43,.16)}\ntd.cv-cap{background:rgba(139,151,166,.2)}\n.note{font-size:12.5px;opacity:.8;max-width:78ch}',
)
# 6) changelog
rep(
    "CHANGELOG = [\n",
    'CHANGELOG = [\n    ("2026-10-26 20:00", "Added the Oct 25-26 results: assay noise, 600k and 1M anchors, acquisition (reduced selection ensemble, one round against three), HepG2 and differential students over nine pools, explore and coordinator prompt grids, nine seeds, the seed ladder and the Zoonomia rebuild. The page now opens with the procedure, parameter by parameter (prompt, rounds, runs, epochs, early stopping, top-k, reservoirs per D, seeds, ensembling and the rest) with a reservoir-by-D coverage matrix; the assumptions table gains a targets, acquisition and robustness group."),\n',
)
rep(
    '    cards = build(d)\n    (src / "index.html").write_text(render(cards, d))',
    """    cards = build(d)
    for c_ in cards:
        if c_["id"] in ("program2", "program3", "program4"):
            c_["status"] = "carried"
            c_["choice"] = "Launch record: these cells have finished and their results are in the cards below. " + c_["choice"]
    if d.get("oct22", {}).get("live"):
        d["generated"] = "2026-" + d["oct22"]["live"]["stamp"]
    (src / "index.html").write_text(render(cards, d))""",
)
open(p, "w").write(s)
print("patched", p)
