#!/usr/bin/env python
# ruff: noqa: E731
"""One dense slide: the AutoResearch procedure at one size and one reservoir, by training size, by reservoir, the evidence for each step and the student ladder, drawn for the right two-thirds of a 16:9 slide (a 4:3 area,
8.9 x 6.67 inches; the left third is left free for background). Native, editable PowerPoint shapes, one table and one chart; opens in Google Slides.

    python3 scripts/analysis/make_one_slide_diagram.py --data DIR_WITH_JSON --out ~/Downloads/autoresearch_lab_meeting
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

_spec = importlib.util.spec_from_file_location(
    "deck", os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_lab_meeting_deck.py")
)
deck = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(deck)
Deck, rgb = deck.Deck, deck.rgb
NAVY, TEAL, TEAL_L, AMBER, AMBER_L, GREY_L, GREY_M, GREY_D = (
    deck.NAVY,
    deck.TEAL,
    deck.TEAL_L,
    deck.AMBER,
    deck.AMBER_L,
    deck.GREY_L,
    deck.GREY_M,
    deck.GREY_D,
)

W, H = 8.9, 6.67  # the canvas: 4:3, the right two-thirds of a 13.333 x 7.5 slide


def draw(d: Deck, s, ox: float, oy: float, base: dict) -> None:
    X = lambda v: ox + v  # noqa: E731
    Y = lambda v: oy + v  # noqa: E731
    box = lambda x, y, w, h, t, **k: d.box(s, X(x), Y(y), w, h, t, **k)  # noqa: E731
    text = lambda x, y, w, h, t, size=9, bold=False, color=NAVY, **k: d.text(
        s, X(x), Y(y), w, h, t, size, bold, color, **k
    )  # noqa: E731
    arrow = lambda x1, y1, x2, y2, c=GREY_D, wd=1.5: d.arrow(s, X(x1), Y(y1), X(x2), Y(y2), c, wd)  # noqa: E731
    chip = lambda x, y, w, h, t: d.box(
        s, X(x), Y(y), w, h, t, fill=AMBER_L, line=AMBER, size=7, color=AMBER, bold=True
    )  # noqa: E731

    text(0, 0, W, 0.38, "AutoResearch: one loop, three training sizes, many reservoirs", 15, True)
    # ---------------------------------------------------------------- 1. one run
    text(0, 0.4, 3.3, 0.22, "1  One run: reservoir R at training size D", 9.5, True, TEAL)
    text(
        3.4,
        0.42,
        5.5,
        0.2,
        "10 rounds x 2 = 20 models per run (30 with manager) | Sonnet 5.5 | agents never see test labels",
        7.5,
        False,
        GREY_D,
        align=PP_ALIGN.RIGHT,
    )
    d.box(s, X(1.55), Y(0.64), 5.7, 1.04, "", fill="F6FBFB", line=TEAL)
    xs = [(0.0, 1.45, "Inputs", "pool of D sequences from R; oracle labels; V0c validation; 200 bp + 20 real flank bases, shift +-10", GREY_L, None),
          (1.65, 1.2, "Propose", "6 ideas: architecture + training recipe, from the validation report", TEAL_L, TEAL),
          (3.05, 1.2, "Coordinate", "picks 2 of 6, ensemble-aware; explore 40% of rounds, then exploit", TEAL_L, TEAL),
          (4.45, 1.2, "Code", "writes class Net; <= 40M params, forward check; up to 2 repairs", TEAL_L, TEAL),
          (5.85, 1.3, "Evaluate", "trains under the fixed harness, time cap per model; validation r only", TEAL_L, TEAL),
          (7.35, 1.5, "Outputs", "20-30 models -> weighted stack on V0c (positive ElasticNet) -> 18 test sets; best configs -> frozen menu", AMBER_L, None)]  # fmt: skip
    for x, w, h, t, fill, line in xs:
        box(x, 0.7, w, 0.92, t, fill=fill, line=line, size=7.5, head=h)
    for x1, x2 in ((1.45, 1.65), (2.85, 3.05), (4.25, 4.45), (5.65, 5.85), (7.15, 7.35)):
        arrow(x1, 1.16, x2, 1.16, TEAL, 1.75)
    chips = [(0.0, 1.45, "20-flank input: +0.07 r"), (1.65, 1.2, "guidance: +0.016 to +0.020"), (3.05, 1.2, "coordinator +0.005 (random pick +0.001)"),
             (4.45, 1.2, "separate coder +0.005"), (5.85, 1.3, "patience 16: +0.001 to +0.011"), (7.35, 1.5, "weighted stack +0.001 to +0.003")]  # fmt: skip
    for x, w, t in chips:
        chip(x, 1.74, w, 0.27, t)
    # ---------------------------------------------------------------- 2. by size
    text(
        0,
        2.08,
        4.5,
        0.22,
        "2  By training size (20-model stacks, genomic r)",
        9.5,
        True,
        TEAL,
    )
    ag = lambda k: (
        f"{base[k]['AlphaGenome S1 (frozen encoder)']['stack_genomic']:.3f} | {base[k]['AlphaGenome S2 (fine-tuned)']['stack_genomic']:.3f}"
    )
    g = lambda k, f: f"{base[k][f]['stack_genomic']:.3f}"  # noqa: E731
    rows = [
        ["", "10k", "30k", "100k"],
        ["loop run / menu retrain", "20 / ~2 H100-h", "20-30 / ~3 H100-h", "20 / ~5 H100-h"],
        ["tuned LegNet recipe", "800 ep, wd 0.4", "500 ep, wd 0.2", "500 ep, wd 0.1"],
        [
            "out-of-the-box LegNet (100 ep)",
            g("10k", "LegNet published"),
            g("30k", "LegNet published"),
            g("100k", "LegNet published"),
        ],
        [
            "tuned LegNet x 20 seeds",
            g("10k", "LegNet tuned"),
            g("30k", "LegNet tuned"),
            g("100k", "LegNet tuned"),
        ],
        ["menu v2, 20 configs", g("10k", "menu v2"), g("30k", "menu v2"), g("100k", "menu v2")],
        [
            "loop run, 20 models",
            g("10k", "AutoResearch run"),
            g("30k", "AutoResearch run"),
            g("100k", "AutoResearch run"),
        ],
        [
            "AlphaGenome-FT x10, S1 | S2 (ceiling)",
            ag("10k"),
            ag("30k"),
            "not run",
        ],
        [
            "ensemble to use",
            "tuned + menu: 0.794",
            "tuned LegNet: 0.891",
            "tuned LegNet: 0.938",
        ],
    ]
    fills = (
        {(4, c): TEAL_L for c in range(4)}
        | {(8, c): AMBER_L for c in range(1, 4)}
        | {(7, c): "E7E3F1" for c in range(4)}
    )
    d.table(
        s,
        X(0),
        Y(2.32),
        4.55,
        rows,
        col_w=[1.75, 0.93, 0.93, 0.94],
        size=7.5,
        row_h=0.2,
        fills=fills,
    )
    chip(
        0,
        4.14,
        4.55,
        0.2,
        "tuned vs published recipe, one LegNet: +0.066 (10k) +0.024 (30k) +0.007 (100k)",
    )
    chip(
        0,
        4.37,
        4.55,
        0.22,
        "AlphaGenome-FT vs measured labels: 0.89|0.91 (10k), 0.90|0.92 (30k)",
    )
    # ---------------------------------------------------------------- 3. by reservoir
    text(4.7, 2.08, 4.2, 0.22, "3  By reservoir (one pool each, same D)", 9.5, True, TEAL)
    box(
        4.7,
        2.32,
        1.3,
        1.17,
        "20 configs: top 3 models of each standard-input 30k search, picked by stack gain on V0c",
        fill=AMBER_L,
        size=7.5,
        head="Frozen menu v2",
    )
    arrow(6.0, 2.9, 6.12, 2.9)
    res = [
        "genomic",
        "random",
        "random+GC sweep",
        "motif CT-enr.",
        "motif shared",
        "motif syntax",
        "evoaug (paper)",
        "evoaug (ours)",
        "evo2",
        "dinuc shuffle",
        "GC matched",
        "mutagenesis",
        "zoonomia",
        "zoonomia orth.",
        "D3 steps",
    ]
    for i, r in enumerate(res):
        box(
            6.15 + (i % 3) * 0.92,
            2.32 + (i // 3) * 0.235,
            0.88,
            0.205,
            r,
            fill=GREY_L,
            line=GREY_M,
            size=6.5,
        )
    box(
        4.7,
        3.58,
        2.15,
        0.52,
        "retrain 20 configs (1 seed, patience 16) -> stack on V0c -> 18 test sets",
        fill=TEAL_L,
        line=TEAL,
        size=7.5,
    )
    arrow(6.85, 3.84, 7.0, 3.84)
    box(
        7.0,
        3.58,
        1.9,
        0.52,
        "compare reservoirs, paired by replicate pool; differences < 0.005 are noise",
        fill=AMBER_L,
        size=7.5,
    )
    chip(
        4.7,
        4.17,
        4.2,
        0.2,
        "fresh search per reservoir vs carried menu: -0.003 to -0.024, none better",
    )
    chip(
        4.7,
        4.4,
        4.2,
        0.2,
        "menu size: ~11 configs enough at 30k; 6-11 runs give 90% of the pooled gain",
    )
    # ---------------------------------------------------------------- 4. ladder chart + conclusions
    text(
        0,
        4.62,
        4.5,
        0.22,
        "4  30k ladder: single model vs ensemble, four evaluation sets",
        9.5,
        True,
        TEAL,
    )
    sets = [
        ("genomic", "genomic"),
        ("ood", "OOD"),
        ("random_32k", "random"),
        ("snv_delta", "SNV effect"),
        ("gc_sweep", "GC sweep"),
        ("dinuc_shuffle", "dinuc shuffle"),
    ]
    fams = [
        ("LegNet published", "out-of-box LegNet"),
        ("LegNet tuned", "tuned LegNet"),
        ("menu v2", "menu v2"),
        ("AutoResearch run", "loop run"),
        ("AlphaGenome S2 (fine-tuned)", "AlphaGenome-FT S2"),
    ]
    sets = sets[:4]
    for j, (kind, title) in enumerate(
        (("single", "single model (mean)"), ("stack", "ensemble: weighted stack"))
    ):
        cd = CategoryChartData()
        cd.categories = [n for _, n in sets]
        for f, lab in fams:
            cd.add_series(lab, [round(base["30k"][f][f"{kind}_{k}"], 3) for k, _ in sets])
        gf = s.shapes.add_chart(
            XL_CHART_TYPE.COLUMN_CLUSTERED,
            Inches(X(0) + j * 2.3),
            Inches(Y(4.82)),
            Inches(2.25),
            Inches(1.85),
            cd,
        )
        ch = gf.chart
        ch.font.size = Pt(6.5)
        ch.has_title = True
        ch.chart_title.text_frame.text = title
        ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(7.5)
        ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
        ch.has_legend, ch.legend.position, ch.legend.include_in_layout = (
            True,
            XL_LEGEND_POSITION.BOTTOM,
            False,
        )
        ch.value_axis.minimum_scale, ch.value_axis.maximum_scale = 0.0, 1.0
        ch.value_axis.major_unit = 0.25
        ch.plots[0].gap_width = 50
        for ser, col in zip(ch.series, ("9AA3B2", "1B7F8C", "B8741A", "B44A3E", "5B4B8A")):
            ser.format.fill.solid()
            ser.format.fill.fore_color.rgb = rgb(col)
    text(4.7, 4.62, 4.2, 0.22, "5  Reading it", 9.5, True, TEAL)
    pts = [
        "Tuned LegNet is the strongest student trained from scratch at 30k and 100k; the menu only adds diversity at 10k.",
        "The loop's own 20 models trail the tuned LegNet at every size (0.842 vs 0.891 at 30k), the menu at all sizes. Redo against the tuned bar is running (rr38).",
        "Doubling or tripling epochs does not help the menu (+0.001 +- 0.004) but helps LegNet a lot: search the recipe per size?",
        "AlphaGenome-FT students (same 30k pool; S1 frozen encoder, S2 fine-tuned) score 0.96 vs oracle labels because the oracle is itself an AlphaGenome model (0.92 vs measured labels): a ceiling only.",
        "Feedback wanted: reference choice, per-size ensemble, epoch budget at 100k.",
    ]
    d.box(
        s,
        X(4.7),
        Y(4.84),
        4.2,
        1.8,
        "\n".join("- " + p for p in pts),
        fill=GREY_L,
        size=7.5,
        align=PP_ALIGN.LEFT,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=os.path.expanduser("~/Downloads/autoresearch_lab_meeting"))
    a = ap.parse_args()
    base = json.load(open(f"{a.data}/data_baselines_allsets.json"))
    for name, (sw, sh, ox, oy) in {
        "autoresearch_one_slide_right_two_thirds.pptx": (13.333, 7.5, 13.333 - W, (7.5 - H) / 2),
        "autoresearch_one_slide_diagram_only.pptx": (W, H, 0.0, 0.0),
    }.items():
        d = Deck()
        d.prs.slide_width, d.prs.slide_height = Inches(sw), Inches(sh)
        s = d.prs.slides.add_slide(d.prs.slide_layouts[6])
        draw(d, s, ox, oy, base)
        s.notes_slide.notes_text_frame.text = "Numbers: analyze_baselines_allsets.py, analyze_legnet_recipes.py, Evidence Ledger, docs/TUNED_LEGNET.md. Oracle labels; stacks fitted on V0c; test sets never used for a choice."
        d.prs.save(f"{a.out}/{name}")
        print("wrote", name)


if __name__ == "__main__":
    main()
