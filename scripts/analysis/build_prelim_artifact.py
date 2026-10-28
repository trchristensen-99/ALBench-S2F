# ruff: noqa
#!/usr/bin/env python
"""Build the 'ALBench Preliminary Results' page (one self-contained HTML file) from the analysis outputs.

Inputs (a directory): data_oct30.json (scaling and top-up), data_oct31.json (v1 against v0, ar2 menus against v1 / v0), figs_oct30/*.png, figs_oct31/*.png,
report_<cell>.md (the ar2 search reports) and optionally findings.json (hand-written headline statements: list of {"title", "text"}).

    python scripts/analysis/build_prelim_artifact.py --in <dir> --out prelim.html
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
import time

SETS = ("genomic", "ood", "random_32k")
SET_LABEL = {"genomic": "genomic test", "ood": "designed OOD", "random_32k": "random sequences"}
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
ARMS = [
    ("genomic", "30k + 70k more genomic (control)", "control"),
    ("res_random", "70k uniform random", "reservoir"),
    ("res_random_gc_sweep", "70k random, GC 25-75% batches", "reservoir"),
    ("res_motif_ct_enriched", "70k motif, CT-enriched", "reservoir"),
    ("res_motif_shared_core", "70k motif, shared core", "reservoir"),
    ("res_evoaug_paper2023", "70k EvoAug, published settings", "reservoir"),
    ("res_mutagenesis_mr10", "70k random mutagenesis, 10%", "reservoir"),
    ("res_zoonomia_orthologs", "70k Zoonomia orthologs", "reservoir"),
    ("mix_random", "70k at random from the even mix", "mix"),
    ("mix_mean", "mix: highest ensemble mean", "acquisition"),
    ("mix_std", "mix: highest ensemble SD", "acquisition"),
    ("mix_resid", "mix: SD excess over the activity trend", "acquisition"),
    ("mix_binned", "mix: highest SD within activity bins", "acquisition"),
    ("mix_diverse", "mix: cluster-balanced (diversity only)", "acquisition"),
    ("mix_stdiv", "mix: SD, then diversity", "acquisition"),
    ("mix_badge", "mix: BADGE-like k-means++ on disagreement", "acquisition"),
    ("mix_strat2d", "mix: activity x GC stratified (baseline)", "acquisition"),
    ("mix_neyman", "mix: Neyman allocation over strata", "acquisition"),
    ("mix_powersd", "mix: stochastic SD sampling", "acquisition"),
    ("mix_oracle", "mix: largest true error (ceiling, uses labels)", "ceiling"),
]
CSS = """
:root{--bg:#f4f6f5;--surface:#ffffff;--ink:#15201d;--muted:#56655f;--rule:#d3dad7;--accent:#0b6e6e;--accent-soft:#dcefee;--good:#1f7a4d;--bad:#a23b2a;--warn:#946200;
--c-genomic:#1b6ca8;--c-perturbation:#c4571a;--c-motif:#2a8f5b;--c-random:#7a5195;--hl:#e8f3f2}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0f1514;--surface:#172120;--ink:#e6efec;--muted:#98a8a2;--rule:#2a3836;--accent:#4fc3bd;--accent-soft:#16312f;--good:#5fcf94;--bad:#ef8b78;--warn:#e0b050;--hl:#1b2b2a}}
:root[data-theme="dark"]{--bg:#0f1514;--surface:#172120;--ink:#e6efec;--muted:#98a8a2;--rule:#2a3836;--accent:#4fc3bd;--accent-soft:#16312f;--good:#5fcf94;--bad:#ef8b78;--warn:#e0b050;--hl:#1b2b2a}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1040px;margin:0 auto;padding:0 16px 64px}
header{padding:36px 0 18px;border-bottom:1px solid var(--rule)}
h1{font:600 34px/1.15 ui-serif,Georgia,"Times New Roman",serif;margin:0 0 6px;text-wrap:balance}
.sub{color:var(--muted);max-width:68ch;margin:0}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px}
.chip{font-size:13px;padding:3px 10px;border:1px solid var(--rule);border-radius:999px;background:var(--surface);color:var(--muted);font-variant-numeric:tabular-nums}
.chip b{color:var(--ink);font-weight:600}
nav{position:sticky;top:0;background:var(--bg);z-index:5;border-bottom:1px solid var(--rule);padding:8px 0;display:flex;gap:6px;flex-wrap:wrap}
nav a{font-size:14px;color:var(--muted);text-decoration:none;padding:4px 10px;border-radius:6px}
nav a:hover,nav a:focus-visible{background:var(--accent-soft);color:var(--ink);outline:none}
section{padding:30px 0 6px}
h2{font:600 24px/1.2 ui-serif,Georgia,serif;margin:0 0 4px;text-wrap:balance}
h3{font-size:16px;margin:22px 0 6px}
p{max-width:74ch;margin:8px 0}
.lead{color:var(--muted)}
.find{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px;margin:14px 0}
.find div{background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:12px 14px}
.find b{display:block;margin-bottom:3px}
.tw{overflow-x:auto;margin:10px 0;border:1px solid var(--rule);border-radius:8px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
th,td{padding:6px 10px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left;white-space:normal;min-width:210px}
thead th{font-weight:600;color:var(--muted);border-bottom:1px solid var(--rule);font-size:12.5px;text-transform:uppercase;letter-spacing:.04em}
tbody tr+tr td{border-top:1px solid var(--rule)}
tr.grp td{background:var(--accent-soft);font-weight:600;font-size:12.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
td.best{background:var(--hl);font-weight:600}
td.na{color:var(--muted)}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}
.pos{color:var(--good)}.neg{color:var(--bad)}
figure{margin:14px 0}img{max-width:100%;height:auto;border:1px solid var(--rule);border-radius:8px;background:#fff}
figcaption{color:var(--muted);font-size:14px;margin-top:4px;max-width:74ch}
.note{border-left:3px solid var(--warn);padding:2px 12px;margin:12px 0;color:var(--muted);max-width:74ch}
code{font:13px ui-monospace,Menlo,monospace;background:var(--accent-soft);padding:1px 5px;border-radius:4px}
details{margin:10px 0}summary{cursor:pointer;color:var(--accent)}
.dec{display:grid;gap:10px;margin:12px 0}
.dec>div{background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:12px 14px}
.dec h4{margin:0 0 4px;font-size:15px;display:flex;gap:8px;align-items:baseline;flex-wrap:wrap}
.tag{font-size:11.5px;text-transform:uppercase;letter-spacing:.05em;padding:1px 8px;border-radius:999px;border:1px solid var(--rule);color:var(--muted)}
.tag.ok{color:var(--good);border-color:var(--good)}.tag.prov{color:var(--warn);border-color:var(--warn)}
.dec p{margin:4px 0;color:var(--ink)}.dec .why{color:var(--muted);font-size:14px}
@media (max-width:560px){h1{font-size:27px}}
"""


def esc(s) -> str:
    return html.escape(str(s))


def img(path: str, alt: str) -> str:
    if not os.path.exists(path):
        return ""
    b = base64.b64encode(open(path, "rb").read()).decode()
    return f'<img alt="{esc(alt)}" src="data:image/png;base64,{b}">'


def fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def table(head, rows, best=None, groups=None):
    """rows: list of (label, [values or None], extra_class); best: list of column indices (0-based over values) where the max is highlighted among rows."""
    best = best or []
    colmax = {}
    for j in best:
        v = [r[1][j] for r in rows if r[1][j] is not None and isinstance(r[1][j], (int, float))]
        colmax[j] = max(v) if v else None
    out = [
        "<div class='tw'><table><thead><tr>"
        + "".join(f"<th>{esc(h)}</th>" for h in head)
        + "</tr></thead><tbody>"
    ]
    last = None
    for label, vals, grp in rows:
        if groups and grp != last:
            out.append(
                f"<tr class='grp'><td colspan='{len(head)}'>{esc(groups.get(grp, grp))}</td></tr>"
            )
            last = grp
        cells = []
        for j, v in enumerate(vals):
            cls = ""
            if v is None:
                cells.append("<td class='na'>–</td>")
                continue
            if (
                j in colmax
                and isinstance(v, (int, float))
                and colmax[j] is not None
                and abs(v - colmax[j]) < 5e-9
            ):
                cls = " class='best'"
            cells.append(f"<td{cls}>{fmt(v) if isinstance(v, (int, float)) else esc(v)}</td>")
        out.append(f"<tr><td>{label}</td>{''.join(cells)}</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def delta(x):
    if x is None:
        return None
    return x


def scaling_section(d):
    sc = d.get("scale31") or {}
    sizes = ("10000", "30000", "100000")
    head = ["Strategy"] + [f"{int(D) // 1000}k {SET_LABEL[s]}" for D in sizes for s in SETS]
    rows, n_cells = [], 0
    for st, label, fam in STRATS:
        v = (sc.get(st) or {}).get("sizes", {})
        vals = []
        for D in sizes:
            c = v.get(D)
            x = (c or {}).get("first6") if c else None
            for s in SETS:
                vals.append(x["ensemble"][s] if x else None)
            n_cells += 1 if x else 0
        rows.append(
            (f"<span class='dot' style='background:var(--c-{fam})'></span>{esc(label)}", vals, fam)
        )
    t = table(head, rows, best=list(range(9)))
    ns = []
    for st, label, fam in STRATS:
        v = (sc.get(st) or {}).get("sizes", {})
        ns.append(
            ", ".join(
                f"{int(D) // 1000}k: {((v.get(D) or {}).get('all') or {}).get('n_models', 0)}"
                for D in sizes
            )
        )
    return t, n_cells, ns


def arms_section(d):
    al = d.get("al31") or {}
    ctl = ((al.get("genomic") or {}).get("first6") or {}).get("ensemble")
    mixr = ((al.get("mix_random") or {}).get("first6") or {}).get("ensemble")
    head = [
        "Arm (30k genomic + 70k)",
        "Genomic",
        "Designed OOD",
        "Random seqs",
        "Δ genomic vs control",
        "Δ OOD vs control",
        "Δ random vs control",
        "Δ OOD vs even-mix random",
        "Models",
    ]
    rows, done = [], 0
    groups = {
        "control": "Control",
        "reservoir": "Add 70k from one reservoir",
        "mix": "Even mix of random, motif and EvoAug (300k candidates)",
        "acquisition": "Selected from the mix by the 30k-genomic ensemble",
        "ceiling": "Ceiling",
    }
    for a, label, kind in ARMS:
        x = (al.get(a) or {}).get("first6")
        if not x:
            rows.append((esc(label), [None] * 8, kind))
            continue
        e = x["ensemble"]
        done += 1
        rows.append(
            (
                esc(label),
                [e["genomic"], e["ood"], e["random_32k"]]
                + [(e[s] - ctl[s]) if ctl else None for s in SETS]
                + [(e["ood"] - mixr["ood"]) if mixr else None],
                kind,
            )
        )
        rows[-1][1].append(f"{x['n_models']}")
    t = table(head, rows, best=[0, 1, 2], groups=groups)
    return t, done


def parse_report(txt):
    out = {}
    m = re.search(r"## Verdict\s+(.*?)(?:\n## |\Z)", txt, re.S)
    out["verdict"] = m.group(1).strip() if m else ""
    m = re.search(r"## Variance split\s+(.*?)(?:\n## |\Z)", txt, re.S)
    out["variance"] = m.group(1).strip() if m else ""
    m = re.search(r"## Procedure screen\s+(.*?)(?:\n## |\Z)", txt, re.S)
    out["screen"] = m.group(1).strip() if m else ""
    m = re.search(r"LLM calls: (\d+) \(([^)]*)\)", txt)
    out["calls"] = f"{m.group(1)} ({m.group(2)})" if m else ""
    return out


def md_inline(s):
    s = esc(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    return s.replace("\n", "<br>")


def ar2_search_section(inp):
    parts = []
    for cell, label in (
        ("genomic_10k", "Genomic 10k"),
        ("genomic_30k", "Genomic 30k"),
        ("random_30k", "Random 30k"),
        ("motif_30k", "Motif CT-enriched 30k"),
    ):
        p = os.path.join(inp, f"report_{cell}.md")
        if not os.path.exists(p):
            continue
        txt = open(p).read()
        rep = parse_report(txt)
        fam = re.findall(r"\|\s*([a-z_]+)\s*\|\s*(\d+)\s*\|\s*([a-z_]+)\s*\|\s*(\d+)\s*\|", txt)
        rows = [(esc(f), [int(a), st, int(b)], "") for f, a, st, b in fam]
        parts.append(f"<h3>{esc(label)}</h3>")
        parts.append(table(["Family", "Candidates", "State", "Runs"], rows))
        parts.append(f"<p class='lead'>LLM calls {esc(rep['calls'])}.</p>")
        if rep["verdict"]:
            parts.append(f"<p>{md_inline(rep['verdict'])}</p>")
        if rep["screen"] and cell == "genomic_30k":
            parts.append(
                f"<p><b>Procedure screen (anchors, 2 seeds each).</b> {md_inline(rep['screen'])}</p>"
            )
    return "".join(parts)


def ar2_compare(d31):
    a2 = (d31 or {}).get("ar2") or {}
    head = ["Cell", "Method", "Models", "Genomic", "Designed OOD", "Random seqs"]
    rows = []
    for cell, v in a2.items():
        for key, label in (
            ("ar2_all_seeds", "ar2 menu (families' best configs, 3 seeds)"),
            ("v1_first_K", f"menu v1, first {v.get('K')} configs"),
            ("v1", "menu v1 (20 configs)"),
            ("v0", "menu v0 (30 usable configs)"),
        ):
            e = v.get(key)
            if not e:
                continue
            rows.append(
                (
                    f"{esc(cell)} · {esc(label)}",
                    [
                        e["n_models"],
                        e["ensemble"]["genomic"],
                        e["ensemble"]["ood"],
                        e["ensemble"]["random_32k"],
                    ],
                    cell,
                )
            )
    if not rows:
        return "<p class='lead'>Menu training is still running; this table fills in when the shards finish.</p>"
    # integer column (models) should not be bolded: pass best only for score columns; table() handles ints as numbers, so convert to str for the first
    rows = [(r[0], [str(r[1][0])] + r[1][1:], r[2]) for r in rows]
    return table(head, rows, best=[1, 2, 3], groups={c: c.replace("_", " ") for c in a2})


def v1v0(d31):
    v = (d31 or {}).get("v1val") or {}
    head = [
        "Pool",
        "v1 genomic",
        "v0 genomic",
        "Δ",
        "v1 OOD",
        "v0 OOD",
        "Δ",
        "v1 random",
        "v0 random",
        "Δ",
    ]
    rows = []
    for cell, x in v.items():
        a, b = x["v1"]["ensemble"], x["v0"]["ensemble"]
        vals = []
        for s in SETS:
            vals += [a[s], b[s], a[s] - b[s]]
        rows.append((esc(cell.replace("_", " ")), vals, ""))
    return table(head, rows)


def ar2_build_notes():
    return """<h3>How it was built and what was tested</h3>
<p>The system was implemented from the specification in about a night by four parallel builders (compute side; orchestration; controller and simulator; proposer and prompts) against a fixed interface contract, then integrated and run. CPU tests: <b>242 pass, 1 skipped</b> (the real-CLI isolation test T7, which needs the cluster CLI; its adapter was exercised with a fake CLI and with real sketch, implementation and tagging calls). Passing: loader checks (T1), label scaling (T2), strand averaging (T3), the nine broken-model fixtures (T4), crash and resume at 20 random kill points (T5), exact rebuild (T6), the held-back set unreachable from prompts (T8, T9), winner's curse (T10), lesson numbers (T11), template freeze (T12), early termination (T13), stress screen (T14), capability check (T15), live balance changes (T16).</p>
<p><b>Not met as specified (simulator, 20 seeds):</b> the two planted better families were confirmed and menu-eligible in 16 of 20 runs (the specification asks 18); the flat stop fired in 16 of 20; the budget forecast differed from simulated counts by 46% on average (asks 25%). Not built: the LLM-written procedure phase (P) and the legacy-model import. Deviations: stress threshold 25 SD instead of 10 (the 10 SD rule flagged normal models at 10.5 to 11 SD); the flat-stop progress rule; unspent budget rolls forward; most real runs ended without the final reconfirmation phase because families still being finished used its budget, so the held-back chrX comparison is mostly missing.</p>
<p><b>Evaluator parity (the specification's first milestone) is not met.</b> Two menu-v1 architectures trained at genomic 10k reach validation r 0.736 and 0.719 in the ar2 loop against 0.766 and 0.763 in our loop (differences −0.030 and −0.044; the specification asks for 0.005). The ar2 loop has no shift augmentation (excluded by the specification), its own one-cycle schedule and patience 8. In its own loop those two v1 architectures score about what the best ar2-found families do (0.723 at 10k), so the searches did not find better single models than the earlier ones under the same loop; the lower ar2-menu ensembles above partly reflect that its recipes were tuned in a loop that ranks architectures differently from ours.</p>
<details><summary>Six bugs found only by real runs (all fixed; the mock-based tests did not cover them)</summary><p>1) A recipe whose run errored after retries was never counted as finished, so a family waited forever (the transformer anchor stuck in two searches). 2) The screen's correlation option used a loss name the evaluator did not know (18 failed runs). 3) The run loop spun at about one step per second when a step only resubmitted runs. 4) The screen's combination step crashed the orchestrator because its runs belong to different models. 5) The screen combined two loss options, which duplicates a single option's runs, so it never finished. 6) The screen's adoption decision was never read back, so it repeated.</p></details>
<p class="lead">Run settings: budgets about half the specification's (anchors 30, main 50, final 18; main 100 at genomic 10k; the genomic 30k search also ran the 36-run loss and training screen), the three anchor families written by hand (a LegNet-like residual CNN, a CNN-GRU, a CNN-attention hybrid) and new families written by <code>claude-sonnet-5-5</code> through the Claude Code CLI, one search per cell, 2 to 3 GPU workers each.</p>"""


def decisions_section(inp):
    p = os.path.join(inp, "decisions.json")
    if not os.path.exists(p):
        return ""
    items = json.load(open(p))
    out = [
        "<section id='decisions'><h2>Decisions and the evidence behind them</h2><div class='dec'>"
    ]
    for it in items:
        cls = "ok" if it.get("status") == "supported" else "prov"
        out.append(
            f"<div><h4>{esc(it['decision'])}<span class='tag {cls}'>{esc(it.get('status', ''))}</span></h4><p>{md_inline(it['evidence'])}</p><p class='why'>{md_inline(it.get('limits', ''))}</p></div>"
        )
    out.append("</div></section>")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d30 = json.load(open(os.path.join(a.inp, "data_oct30.json")))
    d31 = json.load(open(os.path.join(a.inp, "data_oct31.json")))
    findings = (
        json.load(open(os.path.join(a.inp, "findings.json")))
        if os.path.exists(os.path.join(a.inp, "findings.json"))
        else []
    )
    scale_t, n_scale, ns = scaling_section(d30)
    arms_t, n_arms = arms_section(d30)
    fig = lambda f, alt: img(os.path.join(a.inp, f), alt)
    snap = time.strftime("%b %d %H:%M")
    find_html = "".join(
        f"<div><b>{esc(f['title'])}</b>{md_inline(f['text'])}</div>" for f in findings
    )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ALBench Preliminary Results</title><style>{CSS}</style></head><body><div class="wrap">
<header><h1>ALBench Preliminary Results</h1>
<p class="sub">Reservoir scaling, a 30k-genomic top-up experiment with non-active and active selection, and the AutoResearch v2 search system run on the standard cells. All models are trained with menu v1 (patience 16 unless noted) and scored by the ensemble of the models trained in each cell on three held-out evaluation sets.</p>
<div class="chips"><span class="chip">snapshot <b>{esc(snap)}</b></span><span class="chip">scaling cells with results <b>{n_scale}/24</b></span><span class="chip">top-up arms with results <b>{n_arms}/{len(ARMS)}</b></span><span class="chip">one seed per configuration unless noted</span></div></header>
<nav><a href="#glance">At a glance</a><a href="#decisions">Decisions</a><a href="#scaling">Scaling</a><a href="#topup">Top-up</a><a href="#ar2">AutoResearch v2</a><a href="#menu">Menu evidence</a><a href="#caveats">Caveats</a></nav>
<section id="glance"><h2>At a glance</h2><div class="find">{find_html or "<div><b>Interim snapshot</b>Cells still training fill in below; empty cells show a dash.</div>"}</div></section>
{decisions_section(a.inp)}
<section id="scaling"><h2>Scaling by strategy</h2>
<p class="lead">Each strategy trained on its own nested pool at 10k, 30k and 100k. Values are ensemble Pearson r over the first 6 menu configurations (the comparison every cell shares). Best per column is shaded.</p>
{scale_t}
<p class="lead">Models trained per strategy and size (all configurations): {"; ".join(f"{esc(l)} ({n})" for (_, l, _), n in zip(STRATS, ns))}.</p>
<figure>{fig("figs_oct30/scale31.png", "Scaling curves by strategy")}<figcaption>Ensemble r against dataset size for the three test sets. Colour is the reservoir family; solid lines mark the four core strategies.</figcaption></figure></section>
<section id="topup"><h2>30k genomic plus 70k</h2>
<p class="lead">Start from the 30k genomic replicate-1 set and add 70k sequences by one of the listed rules, then train 100k-sized models. The acquisition arms score a 300k even mix of random, motif and EvoAug candidates with the 30k-genomic ensemble (30 configurations). Differences under about 0.005 are within the noise of one training seed.</p>
{arms_t}
<figure>{fig("figs_oct30/al31.png", "Top-up arms")}<figcaption>Ensemble r of each arm on the three test sets.</figcaption></figure></section>
<section id="ar2"><h2>AutoResearch v2</h2>
<p class="lead">Implemented from the design specification and run as four searches (genomic 10k and 30k, random 30k, motif 30k; the 100k search was left out). Each search found its best configuration per architecture family; those configurations were then retrained with our per-D procedure and stacked.</p>
{ar2_search_section(a.inp)}
<h3>Ensembles from the ar2 menus against menu v1 and v0 (our per-D procedure)</h3>{ar2_compare(d31)}
{ar2_build_notes()}
<figure>{fig("figs_oct31/ar2_vs_v1.png", "ar2 against v1 and v0")}<figcaption>Ensemble r by method for each cell with a trained ar2 menu.</figcaption></figure></section>
<section id="menu"><h2>Menu evidence</h2><p class="lead">Menu v1 (20 configurations from runs on five reservoir families and a mixed corpus) against the v0 menu (30 usable configurations from genomic runs), retrained on the same pools.</p>{v1v0(d31)}</section>
<section id="caveats"><h2>Caveats</h2><div class="note">Preliminary: one seed per configuration, one pool replicate, partial cells shown as dashes. The top-up and 100k scaling cells use the first 6 menu configurations to fit the time; 10k and 30k scaling cells use 20.</div></section>
</div></body></html>"""
    open(a.out, "w").write(page)
    print("wrote", a.out, len(page) // 1024, "KB")


if __name__ == "__main__":
    main()
