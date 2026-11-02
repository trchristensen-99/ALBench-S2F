#!/usr/bin/env python
"""Lab-meeting deck: how the AutoResearch loop works at each training size and for each reservoir, and the experiments that justify each step.

Writes an editable PowerPoint file (native shapes, tables and charts; opens in Google Slides) plus every figure as a PNG. Numbers come from the analysis JSON files written by
analyze_baselines_allsets.py, analyze_legnet_recipes.py and compare_pi_30k.py and from the Evidence Ledger figures (outputs/justification_oct22/figs); nothing is recomputed here.

    python3 scripts/analysis/make_lab_meeting_deck.py --data DIR_WITH_JSON --figs DIR_WITH_LEDGER_PNGS --out ~/Downloads/autoresearch_lab_meeting

Needs python-pptx and matplotlib (the system python3 has both).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import matplotlib
import matplotlib.ticker  # noqa: F401

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.chart.data import CategoryChartData  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION  # noqa: E402
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE  # noqa: E402
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN  # noqa: E402
from pptx.oxml.ns import qn  # noqa: E402
from pptx.util import Emu, Inches, Pt  # noqa: E402

NAVY, TEAL, TEAL_L = "1F2A44", "1B7F8C", "D9EEF0"
AMBER, AMBER_L = "B8741A", "FBEFD9"
GREY_L, GREY_M, GREY_D = "EEF0F3", "8A93A3", "4A5468"
RED, GREEN = "B44A3E", "2E7D4F"
FAM_COL = {
    "LegNet published": "#9aa3b2",
    "LegNet tuned": "#1b7f8c",
    "menu v2": "#b8741a",
    "AutoResearch run": "#b44a3e",
    "AlphaGenome S2 (fine-tuned)": "#5b4b8a",
}
SETS = [
    ("genomic", "genomic test"), ("ood", "designed / OOD"), ("random_32k", "random sequences"), ("gc_sweep", "GC sweep"),
    ("dinuc_shuffle", "dinucleotide shuffle"), ("snv_delta", "SNV effect (alt - ref)"), ("snv_ref", "SNV reference"), ("snv_alt", "SNV alternate"),
    ("sub_low", "substitutions, low"), ("sub_med", "substitutions, med"), ("sub_high", "substitutions, high"), ("ins_low", "insertions, low"),
    ("ins_med", "insertions, med"), ("ins_high", "insertions, high"), ("del_low", "deletions, low"), ("del_med", "deletions, med"),
    ("del_high", "deletions, high"), ("translocation", "translocation"), ("inversion", "inversion"),
]  # fmt: skip


def rgb(h: str) -> RGBColor:
    return RGBColor.from_string(h)


# ---------------------------------------------------------------- figures
def fig_recipes(rec: dict, out: str) -> str:
    """Genomic single-model r against epochs for weight decay 0.1 / 0.2 / 0.4 at the three sizes (the LegNet recipe sweeps)."""
    arms = {  # name: (epochs, weight decay)
        "e100 (control)": (100, 0.1), "e150": (150, 0.1), "e200": (200, 0.1), "e300": (300, 0.1), "e500": (500, 0.1),
        "e100_wd2": (100, 0.2), "e150_wd2": (150, 0.2), "e300_wd2": (300, 0.2), "e500_wd2": (500, 0.2), "e800_wd2": (800, 0.2),
        "e500_wd4": (500, 0.4), "e800_wd4": (800, 0.4),
    }  # fmt: skip
    best = {"10k": "e800_wd4", "30k": "e500_wd2", "100k": "e500"}
    cols = {0.1: "#9aa3b2", 0.2: "#1b7f8c", 0.4: "#b8741a"}
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
    for ax, size in zip(axs, ("10k", "30k", "100k")):
        d = rec[size]
        gval = lambda a: d[a]["genomic"] if a == "e100 (control)" else d[a]["genomic"]["mean"]  # noqa: E731
        for wd, c in cols.items():
            pts = sorted((e, gval(a)) for a, (e, w) in arms.items() if w == wd and a in d)
            if pts:
                ax.plot(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    "-o",
                    color=c,
                    label=f"weight decay {wd}",
                    lw=2,
                    ms=5,
                )
        b = best[size]
        ax.plot(arms[b][0], gval(b), "*", color="#b44a3e", ms=17, zorder=5, label="chosen recipe")
        ax.set_xscale("log")
        ax.set_xticks([100, 200, 500, 800])
        ax.set_xticklabels(["100", "200", "500", "800"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(f"{size} training sequences", fontsize=13, fontweight="bold")
        ax.set_xlabel("epochs (published recipe: 100)")
        ax.grid(alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].set_ylabel("genomic-test r, mean single LegNet")
    axs[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle(
        "LegNet training recipe: longer and more regularised at small sizes (AdamW, lr 0.01, batch 1024, one-cycle; 5 seeds per point, 20 for the published control)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    p = f"{out}/fig_legnet_recipe_sweep.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_allsets(base: dict, out: str) -> str:
    """Stack r of every student family on all evaluation sets, one panel per training size."""
    fig, axs = plt.subplots(1, 3, figsize=(14, 6.6), sharey=True)
    ys = np.arange(len(SETS))[::-1]
    for ax, size in zip(axs, ("10k", "30k", "100k")):
        for fam, col in FAM_COL.items():
            if fam not in base[size]:
                continue
            x = [base[size][fam][f"stack_{k}"] for k, _ in SETS]
            ax.plot(x, ys, "o", color=col, label=fam if size == "10k" else None, ms=6, alpha=0.95)
        ax.set_title(f"{size} training sequences", fontsize=13, fontweight="bold")
        ax.set_xlabel("stack r (20 models; AlphaGenome: 10)")
        ax.set_yticks(ys)
        ax.set_yticklabels([n for _, n in SETS], fontsize=9)
        ax.grid(axis="x", alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(
        h, lab, loc="lower center", ncol=5, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02)
    )
    fig.suptitle(
        "Student families on all 18 evaluation sets (oracle labels; weighted stack fitted on V0c)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    p = f"{out}/fig_all_eval_sets.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_arms(pi: dict, base: dict, out: str) -> str:
    """AutoResearch arms at 30k (all-model stack, genomic test) against the baselines."""
    order = [
        ("ctx35 (standard loop, 4 runs)", "standard loop"), ("mgr35 coordinator-schedule (3)", "coordinator only, 15 rounds"),
        ("mgr35 explore-then-exploit (3)", "explore, then exploit"), ("mgr38 reference shown, schedule_ens (3)", "+ published LegNet shown as reference"),
        ("mgr38 reference + fallback manager (3)", "+ fallback manager"), ("op38 open prompt, no reference (3)", "open prompt (no architecture hint)"),
        ("op38 open prompt, late reference + fallback (3)", "open prompt, late reference"), ("op38 open prompt + target (3)", "open prompt + target sentence"),
    ]  # fmt: skip
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ys = np.arange(len(order))[::-1]
    for y, (key, label) in zip(ys, order):
        vals = [
            v["all_stack_oracle"]
            for v in pi["groups"][key].values()
            if isinstance(v, dict) and "all_stack_oracle" in v
        ]
        ax.plot(vals, [y] * len(vals), "o", color="#b44a3e", alpha=0.45, ms=7)
        ax.plot([np.mean(vals)], [y], "D", color="#b44a3e", ms=9)
    for name, col, ls in (
        ("LegNet published", "#9aa3b2", "--"),
        ("menu v2", "#b8741a", "--"),
        ("LegNet tuned", "#1b7f8c", "-"),
    ):
        v = base["30k"][name]["stack_genomic"]
        ax.axvline(v, color=col, ls=ls, lw=2, label=f"{name} (20 models): {v:.3f}")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for _, lab in order], fontsize=9.5)
    ax.set_xlabel(
        "genomic-test r, stack of all models of one run (30k; dots: replicate pools; diamond: mean)"
    )
    ax.set_xlim(0.825, 0.9)
    ax.grid(axis="x", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(
        "AutoResearch arms at 30k against the baselines (20-seed LegNet banks and the 20-config menu)",
        fontsize=11,
    )
    fig.tight_layout()
    p = f"{out}/fig_autoresearch_arms_30k.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


# ---------------------------------------------------------------- pptx helpers
class Deck:
    def __init__(self) -> None:
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(13.333), Inches(7.5)

    def slide(self, title: str, subtitle: str = "", notes: str = ""):
        s = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self.text(s, 0.45, 0.28, 12.4, 0.6, title, 26, True, NAVY)
        if subtitle:
            self.text(s, 0.45, 0.88, 12.4, 0.58, subtitle, 13, False, GREY_D)
        if notes:
            s.notes_slide.notes_text_frame.text = notes
        return s

    @staticmethod
    def text(
        s,
        x,
        y,
        w,
        h,
        txt,
        size=12,
        bold=False,
        color=NAVY,
        align=PP_ALIGN.LEFT,
        anchor=MSO_ANCHOR.TOP,
    ):
        tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = anchor
        tf.margin_left = tf.margin_right = Inches(0.04)
        tf.margin_top = tf.margin_bottom = Inches(0.02)
        for i, line in enumerate(txt.split("\n")):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align
            r = p.add_run()
            r.text = line
            r.font.size, r.font.bold, r.font.name = Pt(size), bold, "Calibri"
            r.font.color.rgb = rgb(color)
        return tb

    @staticmethod
    def box(
        s,
        x,
        y,
        w,
        h,
        txt,
        fill=GREY_L,
        line=None,
        size=12,
        bold=False,
        color=NAVY,
        align=PP_ALIGN.CENTER,
        rounded=True,
        head=None,
    ):
        shp = s.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
            Inches(x),
            Inches(y),
            Inches(w),
            Inches(h),
        )
        if rounded:
            shp.adjustments[0] = 0.08
        shp.fill.solid()
        shp.fill.fore_color.rgb = rgb(fill)
        if line:
            shp.line.color.rgb = rgb(line)
            shp.line.width = Pt(1.25)
        else:
            shp.line.fill.background()
        shp.shadow.inherit = False
        tf = shp.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = tf.margin_right = Inches(0.08)
        tf.margin_top = tf.margin_bottom = Inches(0.04)
        lines = ([(head, True)] if head else []) + (
            [(t, bold) for t in txt.split("\n")] if txt else []
        )
        for i, (line_txt, b) in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align
            r = p.add_run()
            r.text = line_txt
            r.font.size = Pt(size)
            r.font.bold = b
            r.font.name = "Calibri"
            r.font.color.rgb = rgb(color)
        return shp

    @staticmethod
    def arrow(s, x1, y1, x2, y2, color=GREY_D, width=2.0):
        c = s.shapes.add_connector(
            MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
        )
        c.line.color.rgb = rgb(color)
        c.line.width = Pt(width)
        ln = c.line._get_or_add_ln()
        tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
        ln.append(tail)
        return c

    @staticmethod
    def picture(s, path, x, y, w, h):
        from PIL import Image

        iw, ih = Image.open(path).size
        sc = min(w / iw, h / ih)
        pw, ph = iw * sc, ih * sc
        return s.shapes.add_picture(
            path, Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph)
        )

    def table(
        self,
        s,
        x,
        y,
        w,
        rows,
        col_w=None,
        size=11,
        header_fill=NAVY,
        row_h=0.34,
        first_col_bold=True,
        fills=None,
    ):
        nr, nc = len(rows), len(rows[0])
        gt = s.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
        t = gt.table
        if col_w:
            for i, cw in enumerate(col_w):
                t.columns[i].width = Inches(cw)
        for r in range(nr):
            t.rows[r].height = Inches(row_h)
            for c in range(nc):
                cell = t.cell(r, c)
                cell.text = ""
                p = cell.text_frame.paragraphs[0]
                run = p.add_run()
                run.text = str(rows[r][c])
                run.font.size, run.font.name = Pt(size), "Calibri"
                run.font.bold = r == 0 or (c == 0 and first_col_bold)
                run.font.color.rgb = rgb("FFFFFF" if r == 0 else NAVY)
                cell.fill.solid()
                f = (
                    header_fill
                    if r == 0
                    else (fills or {}).get((r, c), "FFFFFF" if r % 2 else GREY_L)
                )
                cell.fill.fore_color.rgb = rgb(f)
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                cell.margin_left = cell.margin_right = Inches(0.06)
                cell.margin_top = cell.margin_bottom = Inches(0.012 if size <= 9 else 0.04)
        return t

    def tag(self, s, x, y, txt, w=1.6):
        return self.box(
            s, x, y, w, 0.3, txt, fill=AMBER_L, line=AMBER, size=10, color=AMBER, bold=True
        )


def evidence_slide(
    d: Deck, title: str, what: str, found: str, img: str, source: str, notes: str = ""
):
    s = d.slide(title, "", notes)
    d.box(
        s,
        0.45,
        1.05,
        3.55,
        1.7,
        what,
        fill=GREY_L,
        size=12,
        align=PP_ALIGN.LEFT,
        head="What we tried",
    )
    d.box(
        s,
        0.45,
        2.9,
        3.55,
        3.4,
        found,
        fill=TEAL_L,
        size=12,
        align=PP_ALIGN.LEFT,
        head="What it showed",
    )
    d.picture(s, img, 4.2, 1.05, 8.7, 5.5)
    d.text(s, 0.45, 6.85, 12.4, 0.5, source, 9, False, GREY_M)
    return s


def build(args) -> None:
    os.makedirs(f"{args.out}/figures", exist_ok=True)
    jd = lambda n: json.load(open(f"{args.data}/{n}"))  # noqa: E731
    base, rec, pi = (
        jd("data_baselines_allsets.json"),
        jd("data_legnet_recipes.json"),
        jd("data_pi30k.json"),
    )
    for k in base:  # the SNV effect as its own column for the plots
        for fam, v in base[k].items():
            v.setdefault("stack_snv_delta", v.get("stack_snv_delta", float("nan")))
    figs = {
        "recipes": fig_recipes(rec, f"{args.out}/figures"),
        "allsets": fig_allsets(base, f"{args.out}/figures"),
        "arms": fig_arms(pi, base, f"{args.out}/figures"),
    }
    led = {}
    for n in (
        "loop_elements",
        "prompt_grid",
        "run_count_curves",
        "kcurve",
        "unseen_search_vs_menu",
        "recipe_sensitivity",
        "legnet_reference",
        "arch_source",
        "seeds_nine",
    ):
        src = f"{args.figs}/{n}.png"
        if os.path.exists(src):
            shutil.copy(src, f"{args.out}/figures/ledger_{n}.png")
            led[n] = f"{args.out}/figures/ledger_{n}.png"

    d = Deck()
    # ------------------------------------------------ 1. the loop, one run
    s = d.slide(
        "AutoResearch at one training size and one reservoir",
        "One run of the loop: an LLM proposes, codes and tests architectures with their training recipes; the trained models are combined into one weighted ensemble.",
        "Talking points: the loop never sees test labels, only validation (V0c) metrics. The ensemble weights are fitted on V0c as well. Amber tags point to the evidence slides.",
    )
    d.text(s, 0.45, 1.45, 2.8, 0.3, "INPUTS (fixed for every run)", 11, True, GREY_D)
    ins = [
        (
            "Reservoir pool",
            f"D sequences (10k / 30k / 100k), nested subsets of one pool; chr7 and chr13 held out as test, chr19, chr21, chrX as validation",
        ),
        ("Labels", "oracle ensemble (AlphaGenome-based, oracle_v3) labels every sequence"),
        (
            "Student input standard",
            "200-bp insert + 20 real flank bases per side (240 bp); shift +-10 over real flank; reverse-complement augmentation and averaging",
        ),
        (
            "Validation set V0c",
            "29,449 real chr19/21/X sequences, oracle labels; early stopping, feedback to the agents and ensemble weights",
        ),
    ]
    for i, (h, t) in enumerate(ins):
        d.box(s, 0.45, 1.8 + i * 1.12, 2.9, 1.02, t, fill=GREY_L, size=10, head=h)
    d.arrow(s, 3.4, 3.7, 3.75, 3.7)
    d.box(s, 3.8, 1.45, 6.6, 4.55, "", fill="F6FBFB", line=TEAL, rounded=True)
    d.text(
        s,
        3.95,
        1.5,
        6.3,
        0.35,
        "THE LOOP: 10 rounds x 2 models = 20 per run (30 in manager runs)",
        11,
        True,
        TEAL,
    )
    steps = [
        (
            "1  Propose",
            "6 ideas per round: architecture + training recipe, from the validation report and the guidance (known-good settings)",
        ),
        (
            "2  Coordinate",
            "picks the 2 ideas that help most; ensemble-aware; manager: explore the first 40% of rounds, then exploit",
        ),
        (
            "3  Code",
            "writes `class Net`; checked (<= 40M params, CPU forward test); up to 2 repair attempts",
        ),
        (
            "4  Evaluate",
            "trains under the fixed harness (patience 16, per-model time cap); validation r only; exploded predictions screened",
        ),
    ]
    pos = [
        (3.95, 1.95),
        (7.3, 1.95),
        (7.3, 3.85),
        (3.95, 3.85),
    ]  # clockwise: propose, coordinate, code, evaluate
    for (h, t), (x, y) in zip(steps, pos):
        d.box(s, x, y, 2.85, 1.4, t, fill=TEAL_L, line=TEAL, size=10.5, head=h)
    d.arrow(s, 6.8, 2.65, 7.3, 2.65, TEAL)
    d.arrow(s, 8.72, 3.35, 8.72, 3.85, TEAL)
    d.arrow(s, 7.3, 4.55, 6.8, 4.55, TEAL)
    d.arrow(s, 5.37, 3.85, 5.37, 3.35, TEAL)
    d.box(
        s,
        3.95,
        5.4,
        6.2,
        0.5,
        "validation report + lessons go back to the proposer (a separate reflector stage was tested and dropped)",
        fill="FFFFFF",
        size=10,
        color=GREY_D,
    )
    d.arrow(s, 10.45, 3.7, 10.8, 3.7)
    d.text(s, 10.85, 1.45, 2.2, 0.3, "OUTPUTS", 11, True, GREY_D)
    outs = [
        (
            "Run ensemble",
            "all trained models, positive ElasticNet stack of z-scored V0c predictions (zero weight = unused)",
        ),
        (
            "Evaluation",
            "18 test sets, oracle labels: genomic, designed / OOD, random, GC sweep, SNV effect, structural variants ...",
        ),
        (
            "Frozen menu",
            "the best configurations of many runs become the menu that is retrained on other pools (next slides)",
        ),
    ]
    for i, (h, t) in enumerate(outs):
        d.box(s, 10.85, 1.8 + i * 1.5, 2.1, 1.38, t, fill=AMBER_L, size=10, head=h)
    d.box(
        s,
        0.45,
        6.3,
        12.5,
        0.62,
        "Nothing from the test sets reaches an agent (checked by a unit test). The same fixed input standard, validation set, reference LegNet and ensemble rule are used at every size and reservoir.",
        fill="FFFFFF",
        line=GREY_M,
        size=11,
        color=GREY_D,
    )
    d.tag(s, 3.95, 6.0, "evidence: slides 6 to 11", 2.2)
    d.tag(s, 6.3, 6.0, "reference: slide 14", 1.8)

    # ------------------------------------------------ 2. by training size
    s = d.slide(
        "What changes with the training size",
        "The procedure is the same; the budget, the number of configurations and the reference recipe change. 20-model stacks, genomic test r (oracle labels).",
        "The recommended per-size ensemble is the new result of the last days: the tuned LegNet recipe is the strongest student at 30k and 100k.",
    )
    gs = lambda k, f, m="stack_genomic": f"{base[k][f][m]:.3f}"  # noqa: E731
    rows = [
        ["", "10k", "30k", "100k"],
        ["Loop run", "20 models (10 rounds x 2)", "20 models; 30 with a manager", "20 models"],
        ["Per-model time cap in the loop (H100-min)", "25", "25", "45"],
        [
            "Frozen menu v2: 20 configs, one seed each",
            "about 2 H100-hours",
            "about 3 H100-hours",
            "about 5 H100-hours",
        ],
        [
            "Out-of-the-box LegNet (published recipe, 100 epochs)",
            gs("10k", "LegNet published"),
            gs("30k", "LegNet published"),
            gs("100k", "LegNet published"),
        ],
        [
            "Tuned LegNet (per-size recipe, 20 seeds)",
            gs("10k", "LegNet tuned"),
            gs("30k", "LegNet tuned"),
            gs("100k", "LegNet tuned"),
        ],
        ["Tuned recipe", "800 epochs, wd 0.4", "500 epochs, wd 0.2", "500 epochs, wd 0.1"],
        [
            "Menu v2 (20 LLM-found configs)",
            gs("10k", "menu v2"),
            gs("30k", "menu v2"),
            gs("100k", "menu v2"),
        ],
        [
            "One AutoResearch run (20 models, mean of runs)",
            gs("10k", "AutoResearch run"),
            gs("30k", "AutoResearch run"),
            gs("100k", "AutoResearch run"),
        ],
        [
            "Best combined ensemble",
            "tuned LegNet + menu: 0.794",
            "tuned LegNet seeds: 0.891",
            "tuned LegNet seeds: 0.938",
        ],
    ]
    d.table(
        s,
        0.45,
        1.55,
        12.45,
        rows,
        col_w=[4.6, 2.5, 2.65, 2.7],
        size=11.5,
        row_h=0.46,
        fills={
            (5, 1): TEAL_L,
            (5, 2): TEAL_L,
            (5, 3): TEAL_L,
            (9, 1): AMBER_L,
            (9, 2): AMBER_L,
            (9, 3): AMBER_L,
        },
    )
    d.text(
        s,
        0.45,
        6.3,
        12.4,
        0.8,
        "Seed noise of one LegNet is 0.002 (100k) to 0.01 (10k); a stack of 20 is stable to about 0.003. Out-of-the-box and tuned LegNet are both kept as baselines to show what the loop adds when the architecture is held fixed. AlphaGenome-FT students (not shown here) are baselines only: they share the pretrained encoder with the oracle.",
        10.5,
        False,
        GREY_D,
    )

    # ------------------------------------------------ 3. by reservoir
    s = d.slide(
        "What changes with the reservoir (at one training size)",
        "A reservoir strategy generates the candidate sequences; the question is which one gives the best student at a fixed budget. The student recipe is held fixed.",
        "The frozen menu keeps the comparison between reservoirs about the data and not about a lucky search; a fresh search on every reservoir was tested and did not beat the carried menu (evidence slide 12).",
    )
    d.box(
        s,
        0.45,
        1.5,
        2.75,
        2.0,
        "20 configurations: the top 3 models of each standard-input search on the 30k genomic pool, picked greedily by what they add to the stack on V0c",
        fill=AMBER_L,
        size=10.5,
        head="Frozen menu v2",
    )
    d.arrow(s, 3.25, 2.5, 3.6, 2.5)
    d.text(s, 3.65, 1.45, 5.4, 0.3, "ONE POOL PER RESERVOIR, SAME SIZE D", 11, True, GREY_D)
    chips = [
        "genomic",
        "random",
        "random + GC sweep",
        "motif: CT-enriched",
        "motif: shared core",
        "motif: syntax core",
        "evoaug (paper)",
        "evoaug (ours)",
        "evo2",
        "dinucleotide shuffle",
        "GC matched",
        "mutagenesis",
        "zoonomia",
        "zoonomia orthologs",
        "D3 steps",
    ]
    for i, c in enumerate(chips):
        d.box(
            s,
            3.65 + (i % 3) * 1.68,
            1.8 + (i // 3) * 0.55,
            1.6,
            0.46,
            c,
            fill=GREY_L,
            line=GREY_M,
            size=10,
        )
    d.arrow(s, 9.05, 2.6, 9.4, 2.6)
    d.box(
        s,
        9.45,
        1.5,
        3.45,
        2.2,
        "Retrain the 20 configurations on the pool (one seed each, patience 16, standard input)\n-> weighted stack on V0c\n-> score on the 18 evaluation sets",
        fill=TEAL_L,
        line=TEAL,
        size=11,
        head="Per reservoir",
    )
    d.arrow(s, 11.15, 3.75, 11.15, 4.15)
    d.box(
        s,
        9.45,
        4.2,
        3.45,
        1.1,
        "paired by replicate pool; differences below ~0.005 are noise",
        fill=AMBER_L,
        size=10.5,
        head="Compare reservoirs",
    )
    d.box(
        s,
        0.45,
        3.75,
        2.75,
        2.35,
        "Fresh AutoResearch search on the reservoir itself, instead of the carried menu: not better (slide 12). Reservoir ranking is about the data, so the student is held fixed.",
        fill=GREY_L,
        size=10.5,
        head="Why not search per reservoir",
    )
    d.box(
        s,
        3.65,
        4.75,
        5.4,
        0.95,
        "Deliverable: a training-corpus recommendation for the ~3.2M-oligo library, not a model. Each reservoir is scored at 10k, 30k and 100k (where a pool exists) on all 18 evaluation sets, headline: genomic, designed / OOD and random.",
        fill="FFFFFF",
        line=GREY_M,
        size=11,
        color=GREY_D,
    )
    d.tag(s, 0.45, 6.4, "evidence: slides 11 and 12", 2.2)

    # ------------------------------------------------ 4. baseline ladder
    s = d.slide(
        "Where each student family stands (20-model ensembles, all sizes)",
        "Weighted stacks on V0c, oracle labels, mean of the runs. AlphaGenome-FT is a ceiling / baseline only (10 seeds, 10k and 30k).",
        "Numbers from analyze_baselines_allsets.py. The AutoResearch run is the plain loop (no steering), 20 models, averaged over 2 to 3 pools. The SNV effect is alt minus ref predictions against the oracle's alt minus ref (all pairs).",
    )
    fams = [
        "LegNet published",
        "LegNet tuned",
        "menu v2",
        "AutoResearch run",
        "AlphaGenome S2 (fine-tuned)",
    ]
    sets4 = [
        ("genomic", "genomic"),
        ("ood", "OOD"),
        ("random_32k", "random"),
        ("snv_delta", "SNV effect"),
    ]
    rows = [["size", "student"] + [n for _, n in sets4]]
    fills = {}
    for si, k in enumerate(("10k", "30k", "100k")):
        for fi, fam in enumerate(fams):
            if fam not in base[k]:
                continue
            r = [k if fi == 0 else "", fam.replace(" (fine-tuned)", " (fine-tuned)")] + [
                f"{base[k][fam][f'stack_{m}']:.3f}" for m, _ in sets4
            ]
            rows.append(r)
            if fam == "LegNet tuned":
                for c in range(2, 6):
                    fills[(len(rows) - 1, c)] = TEAL_L
            if fam.startswith("AlphaGenome"):
                for c in range(0, 6):
                    fills[(len(rows) - 1, c)] = "E7E3F1"
    d.table(
        s,
        0.45,
        1.5,
        12.45,
        rows,
        col_w=[1.0, 4.45, 1.75, 1.75, 1.75, 1.75],
        size=12,
        row_h=0.325,
        fills=fills,
    )
    d.text(
        s,
        0.45,
        6.5,
        12.4,
        0.8,
        "At 30k and 100k the tuned LegNet beats the menu, which beats the loop's own 20-model run; at 10k the menu and the tuned LegNet are within 0.005 on genomic (the menu is higher on random sequences). The loop's single models are weak (explore phase) and its stack recovers. All 18 sets on the next slide.",
        10.5,
        False,
        GREY_D,
    )

    # ------------------------------------------------ 5. all eval sets figure
    s = d.slide(
        "Performance on all evaluation sets",
        "Stack r per set; oracle labels; every family is read on the same 18 test sets.",
        "",
    )
    d.picture(s, figs["allsets"], 0.45, 1.3, 12.45, 5.6)
    d.text(
        s,
        0.45,
        6.95,
        12.4,
        0.4,
        "Source: analyze_baselines_allsets.py (data_baselines_allsets.json). AlphaGenome-FT shares the pretrained encoder with the oracle: reported as a ceiling, never as a member.",
        9,
        False,
        GREY_M,
    )

    # ------------------------------------------------ evidence slides
    notes_src = "Source: Evidence Ledger (outputs/justification_oct22/figs); paired by replicate pool, 95% bootstrap intervals."
    e = evidence_slide
    # input standard (native chart)
    s = d.slide(
        "Evidence 1: the student input is 200 bp + 20 real flank bases",
        "Sequence-context sweep on the 30k genomic pool (sw38_ / fx35_ cells); numbers from docs/MENU.md.",
        "Context length: 12 to 20 bases give the whole gain; 60 or 200 add nothing measurable. The shift is done over real flank bases, never zeros.",
    )
    cd = CategoryChartData()
    cd.categories = ["single model", "20-config ensemble"]
    cd.add_series("bare 200-bp insert", (0.745, 0.78))
    cd.add_series("insert + 20 real flank bases per side", (0.82, 0.85))
    gf = s.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(4.4), Inches(1.3), Inches(8.3), Inches(5.2), cd
    )
    ch = gf.chart
    ch.has_legend, ch.legend.position, ch.legend.include_in_layout = (
        True,
        XL_LEGEND_POSITION.BOTTOM,
        False,
    )
    ch.value_axis.minimum_scale, ch.value_axis.maximum_scale = 0.7, 0.9
    ch.value_axis.has_title, ch.value_axis.axis_title.text_frame.text = True, "genomic-test r"
    ch.plots[0].has_data_labels = True
    ch.plots[0].data_labels.font.size = Pt(12)
    ch.plots[0].data_labels.number_format, ch.plots[0].data_labels.number_format_is_linked = (
        "0.00",
        False,
    )
    ch.series[0].format.fill.solid()
    ch.series[0].format.fill.fore_color.rgb = rgb("9AA3B2")
    ch.series[1].format.fill.solid()
    ch.series[1].format.fill.fore_color.rgb = rgb(TEAL)
    d.box(
        s,
        0.45,
        1.05,
        3.7,
        1.6,
        "Students on the bare 200-bp insert versus the insert inside its real plasmid context, with a shift over the real flank.",
        fill=GREY_L,
        size=12,
        align=PP_ALIGN.LEFT,
        head="What we tried",
    )
    d.box(
        s,
        0.45,
        2.8,
        3.7,
        3.4,
        "About +0.07 test r for single models and ensembles. This is how the oracle that labels the data reads a sequence. 12 to 20 bases give the whole gain; 60 or 200 add nothing, so 20 is the fixed standard.",
        fill=TEAL_L,
        size=12,
        align=PP_ALIGN.LEFT,
        head="What it showed",
    )
    d.text(
        s,
        0.45,
        6.85,
        12.4,
        0.4,
        "Source: docs/MENU.md (input standard); sw38_ and fx35_ cells.",
        9,
        False,
        GREY_M,
    )
    e(
        d,
        "Evidence 2: which parts of the loop pay for themselves",
        "Each loop element switched on or off with everything else fixed, paired by replicate pool (6 pools per contrast).",
        "A separate coder and a coordinator that picks 2 of 6 ideas each add about +0.005; a random pick of the same 6 adds nothing. A reflector adds nothing beyond the coordinator. The guidance block (known-good settings) is worth +0.016 to +0.020.",
        led["loop_elements"],
        notes_src + " Loop-elements card.",
    )
    e(
        d,
        "Evidence 3: prompt variants hold up across reservoirs and sizes",
        "Explore-style prompt and 'coordinator without reflector' against the default prompt on five reservoirs at 10k and 30k.",
        "The explore prompt does nothing once pooled (n = 51); the coordinator without a reflector gains +0.003 pooled (n = 48). Single cells vary a lot, which is why only pooled contrasts are used.",
        led["prompt_grid"],
        notes_src,
    )
    e(
        d,
        "Evidence 4: showing the reference LegNet and the manager (30k)",
        "Arms of the loop on three 30k genomic replicate pools: coordinator only, explore-then-exploit manager, published LegNet shown as reference, fallback manager, and an open prompt with no architecture hint.",
        "Showing the reference LegNet from the start is the best arm (+0.010 over coordinator only, SE 0.003). A fallback manager, an open prompt and a stated target do not help. No arm reaches the 20-config menu or a 20-seed tuned LegNet bank.",
        figs["arms"],
        "Source: compare_pi_30k.py (data_pi30k.json), 3 pools per arm.",
    )
    e(
        d,
        "Evidence 5: how many independent runs are worth pooling",
        "Ensembles of the top 3 models of k independent runs, subsets of 10 to 15 runs at 10k, 30k, 100k, random and motif pools.",
        "Run-to-run noise is as large as the differences between strategies (single-run sd about 0.008). 90% of the pooled gain arrives at 6 to 11 runs, so the menu is built from 18 runs rather than trusting one.",
        led["run_count_curves"],
        notes_src,
    )
    e(
        d,
        "Evidence 6: how many configurations to train at each size",
        "Retrain the menu at 10k, 30k and 100k; take the first K configurations in a given order; compare the best-8 ensemble with the full menu.",
        "Configurations needed for the best-8 ensemble to stay within tolerance: 17 to 30 at 10k, about 11 at 30k, 9 to 16 at 100k. Taking the best two of every run first beats a random order by 3 to 6 configurations. The final ensemble is a weighted stack over everything trained.",
        led["kcurve"],
        notes_src + " The weighted stack is 0.001 to 0.003 above the 8-member mean.",
    )
    e(
        d,
        "Evidence 7: a fresh search per reservoir does not beat the menu",
        "For each reservoir, a search on that reservoir against the menu built from genomic searches (3 runs against 3 pools).",
        "The carried menu is as good or better on every reservoir and size. The student can therefore be held fixed when comparing reservoirs.",
        led["unseen_search_vs_menu"],
        notes_src,
    )
    e(
        d,
        "Evidence 8: training recipe of the menu (patience and epochs)",
        "Menu retrained with patience 4 or 16 and with half or double the epochs. Newer test: 3x the epochs with early stopping off at 10k and 30k with the standard input (a 100k run at 2.5x is in progress).",
        "Patience 16 is best (+0.001 at 100k to +0.011 at 10k); doubling the epochs changes the ensemble by -0.005 to +0.002, mostly negative. Tripling the epochs with the standard input changes members by +0.001 +- 0.004 (10k) and +0.001 +- 0.003 (30k): the menu is not under-trained, so 200 epochs would cost time without gain.",
        led["recipe_sensitivity"],
        notes_src + " ep38 cells for the 3x-epoch test (docs/TUNED_LEGNET.md).",
    )
    e(
        d,
        "Evidence 9: the LegNet reference needs its own recipe",
        "One-knob changes of the published recipe, then more epochs and more weight decay, 5 seeds per arm, 20 for the control; the best arm per size then run to 20 seeds.",
        "The published recipe is under-trained: genomic r of one network rises by +0.066 (10k), +0.024 (30k) and +0.007 (100k). Smaller data wants more epochs and more weight decay; too much weight decay hurts at 30k and 100k.",
        figs["recipes"],
        "Source: analyze_legnet_recipes.py (data_legnet_recipes.json); docs/TUNED_LEGNET.md.",
    )
    # ------------------------------------------------ feedback
    s = d.slide("Where we would like feedback", "", "Questions for the room.")
    qs = [
        (
            "Reference",
            "Should the loop's reference and in-run baseline be the tuned LegNet (and the out-of-the-box LegNet kept only as a baseline)?",
        ),
        (
            "Per-size ensemble",
            "Tuned LegNet seeds alone at 30k and above, plus the menu only at 10k: does that match your intuition about diversity?",
        ),
        (
            "What the loop adds",
            "Held to one architecture the loop should add a boost over the tuned recipe; today the loop's own 20 models are below the tuned LegNet. Is architecture search the right lever at 30k and above?",
        ),
        (
            "Budget",
            "Epochs: the menu does not gain from 3x epochs, LegNet gains a lot. Worth searching the recipe per size instead of the architecture?",
        ),
        (
            "Baselines",
            "AlphaGenome-FT is a ceiling only (shares the encoder with the oracle). Anything else you would want on the ladder?",
        ),
    ]
    for i, (h, t) in enumerate(qs):
        d.box(
            s,
            0.45,
            1.3 + i * 1.1,
            12.45,
            0.98,
            t,
            fill=GREY_L if i % 2 else TEAL_L,
            size=13,
            align=PP_ALIGN.LEFT,
            head=h,
        )
    d.prs.save(f"{args.out}/autoresearch_loop_lab_meeting.pptx")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--figs", required=True)
    ap.add_argument("--out", default=os.path.expanduser("~/Downloads/autoresearch_lab_meeting"))
    build(ap.parse_args())


if __name__ == "__main__":
    main()
