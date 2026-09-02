"""Sequence-budget workbook -- Slides-sized margins table + full candidate menu.

MAIN TABLE is sized to paste into a 16:9 Google Slides slide with no resizing:
  20 rows x 10 columns, 9.2 in wide x 4.3 in tall, 8 pt.
  (Slide 10 x 5.625 in; title ~0.8 in, footnote ~0.3 in -> 4.3 in of table height;
   20 rows x 0.215 in = 4.3 in. 8 pt text needs ~15.5 pt row height, which is 0.215 in.)
Exceeding 20 rows or 10 columns forces a shrink-to-fit and the numbers stop being legible, so the
funded-arm table is capped there and the FULL candidate list lives on the Menu sheet instead.

Two axes are decided IN SILICO rather than by spending arms: perturbation rates (mutagenesis rate,
indel/translocation/inversion mix) and motif-design parameters (count, order, spacing, background).
Each family orders ONE tuned configuration plus one untuned control, so the value of in-silico tuning
is itself measured (H5) rather than assumed.
"""

import os

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

OUT = os.path.expanduser("~/Downloads/joint_PI_meeting_aug20/sequence_budget.xlsx")

# (family, reservoir, cell-type role, tier)  -- tier A = single-R arm, B = main pool, C = sub-pool
RESERVOIRS = [
    ("Random", "Null control (uniform ACGT + dinuc-shuffle)", "agn", "A"),
    ("Genomic", "ENCODE accessible — open in BOTH", "joint", "A"),
    ("Genomic", "ENCODE accessible — CT-differential", "diff", "A"),
    ("Genomic", "Zoonomia ortholog CREs (phylogenetic)", "agn", "A"),
    ("Genomic", "Gosai alt alleles (Δ-supervision)", "agn", "A"),
    ("Genomic pert.", "Mutagenesis — tuned rate (SNV pairs)", "agn", "A"),
    ("Genomic pert.", "Structural — indel/transloc/inv, tuned", "agn", "A"),
    ("Genomic pert.", "EvoAug combined — untuned control", "agn", "A"),
    ("Motif-based", "Grammar — full motif DB (CT-agnostic)", "agn", "A"),
    ("Motif-based", "Grammar — shared-core motifs", "joint", "A"),
    ("Motif-based", "Grammar — CT-enriched, mixed factorial", "diff", "A"),
    ("Model-gen.", "DNA-LM (HyenaDNA) — unconditioned", "agn", "A"),
    ("Model-gen.", "D3 diffusion — activity-conditioned", "diff", "A"),
    ("Model-gen.", "D3 diffusion — genomic-conditioned", "agn", "A"),
    ("Model-gen.", "D3 diffusion — uncertainty-conditioned", "agn", "A"),
    ("POOLED", "All reservoirs, equal parts", "agn", "B"),
    ("POOLED", "Genomic-derived only", "agn", "C"),
    ("POOLED", "Synthetic / generated only", "agn", "C"),
]
ACQ = [
    "Random",
    "Uncertainty (joint)",
    "Uncertainty (per CT)",
    "Diversity",
    "Uncert. × Divers.",
    "Activity-strat. (2D joint)",
]

ON = set()
for i, (_f, _n, _ct, tier) in enumerate(RESERVOIRS):
    if tier == "A":
        ON.add((i, 0))
    if tier == "B":
        ON.update((i, j) for j in range(len(ACQ)))
    if tier == "C":
        ON.update({(i, 1), (i, 3)})

HDR = PatternFill("solid", fgColor="1E293B")
CTRLH = PatternFill("solid", fgColor="92400E")
EDIT = PatternFill("solid", fgColor="FEF3C7")
OFFF = PatternFill("solid", fgColor="F1F5F9")
CALC = PatternFill("solid", fgColor="F8FAFC")
SUMF = PatternFill("solid", fgColor="DBEAFE")
POOL = PatternFill("solid", fgColor="FDE68A")
FAMF = {
    "Random": "EDE9FE",
    "Genomic": "DBEAFE",
    "Genomic pert.": "CCFBF1",
    "Motif-based": "FEF9C3",
    "Model-gen.": "FCE7F3",
    "POOLED": "FDE68A",
}
CTF = {
    "joint": PatternFill("solid", fgColor="E0E7FF"),
    "diff": PatternFill("solid", fgColor="FEE2E2"),
    "agn": None,
}
thin = Side(style="thin", color="BFBFBF")
BOX = Border(left=thin, right=thin, top=thin, bottom=thin)

nR, nA = len(RESERVOIRS), len(ACQ)
POOL_I = [i for i, r in enumerate(RESERVOIRS) if r[3] != "A"]
C0 = 4  # D = first acquisition column
TOTC = C0 + nA  # J = row total  -> 10 columns total (A..J)
AH, A0 = 8, 9  # table header row / first data row
ALAST, SROW = A0 + nR - 1, A0 + nR  # last data row / sigma row  (rows 8..27 = 20 rows)
CH, CT0 = SROW + 4, SROW + 5
CLAST = CT0 + nR - 1
LC = get_column_letter(C0 + nA - 1)
CRNG = f"$D${CT0}:${LC}${CLAST}"
PRNG = f"$D${CT0 + POOL_I[0]}:${LC}${CT0 + POOL_I[-1]}"

wb = Workbook()
ws = wb.active
ws.title = "Budget"
ws["A1"] = "Sequence budget — funded arms"
ws["A1"].font = Font(bold=True, size=14)
ws["A2"] = (
    f"Paste rows {AH}–{SROW} (A:J) into Slides — exactly 20 × 10, 9.2ʺ × 4.3ʺ at 8 pt, fits 16:9 "
    "with no shrink.   Edit the CONTROL grid below: 0 = drop (budget redistributes) · 1 = active · >1 = pin."
)
ws["A2"].font = Font(italic=True, size=9, color="666666")


def lbl(cell, t, **kw):
    ws[cell] = t
    ws[cell].font = Font(size=9, **kw)


lbl("A3", "Synthesis budget", bold=True)
b = ws["C3"]
b.value = 3_200_000
b.fill, b.number_format, b.border, b.font = EDIT, "#,##0", BOX, Font(bold=True, size=9)
lbl("D3", "sequences ORDERED (pooled arms share them)", italic=True, color="666666")
lbl("A4", "Pooled-arm overlap", bold=True)
o = ws["C4"]
o.value = 0.0
o.fill, o.number_format, o.border, o.font = EDIT, "0%", BOX, Font(bold=True, size=9)
lbl(
    "D4",
    "mean pairwise Jaccard between selections — MEASURE IN SILICO before ordering",
    italic=True,
    color="B45309",
)

n_pool = f"(COUNTBLANK({PRNG})+COUNTIF({PRNG},1))"
n_all = f"(COUNTBLANK({CRNG})+COUNTIF({CRNG},1))"
eff = f"({n_all}-{n_pool}+1+MAX(0,{n_pool}-1)*(1-$C$4))"
lbl("A5", "Arms funded", bold=True)
ws["C5"] = f"={n_all}"
ws["C5"].font = Font(bold=True, size=9)
lbl("D5", "Effective cells", bold=True)
ws["F5"] = f"=ROUND({eff},1)"
ws["F5"].font = Font(bold=True, size=9)
lbl("G5", "Sequences / arm", bold=True)
sc = ws["I5"]
sc.value = f'=ROUND(($C$3-SUMIF({CRNG},">1"))/MAX(1,{eff}),-2)'
sc.number_format, sc.font, sc.fill, sc.border = "#,##0", Font(bold=True, size=11), SUMF, BOX
ws["A6"] = (
    '=IF(I5>=100000,"✓ "&TEXT(I5,"#,##0")&"/arm — from-scratch curve 1k→"&TEXT(I5,"#,##0")&'
    '" (~2 OOMs) with 3 replicates at each of the 3 smallest points",'
    'IF(I5>=30000,"1k→30k curve (~1.5 OOMs)","⚠ too small for a per-arm curve"))'
)
ws["A6"].font = Font(bold=True, size=9, color="15803D")


def header(row, fill, first):
    for c, t in ((1, "Family"), (2, first), (3, "CT")):
        h = ws.cell(row, c, t)
        h.fill, h.font = fill, Font(bold=True, color="FFFFFF", size=8)
        h.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for j, a in enumerate(ACQ):
        h = ws.cell(row, C0 + j, a)
        h.fill, h.font = fill, Font(bold=True, color="FFFFFF", size=7.5)
        h.alignment = Alignment(wrap_text=True, horizontal="center", vertical="center")


header(AH, HDR, "Reservoir strategy")
h = ws.cell(AH, TOTC, "Σ arm")
h.fill, h.font = HDR, Font(bold=True, color="FFFFFF", size=7.5)
h.alignment = Alignment(wrap_text=True, horizontal="center", vertical="center")

for i, (fam, name, ct, tier) in enumerate(RESERVOIRS):
    r, cr = A0 + i, CT0 + i
    f = ws.cell(r, 1, fam if (i == 0 or RESERVOIRS[i - 1][0] != fam) else "")
    f.font = Font(size=7.5, bold=True)
    f.fill = PatternFill("solid", fgColor=FAMF[fam])
    f.alignment = Alignment(vertical="center", wrap_text=True)
    n = ws.cell(r, 2, name)
    n.font = Font(size=8, bold=tier != "A")
    n.alignment = Alignment(vertical="center")
    if tier != "A":
        n.fill = POOL
    c = ws.cell(r, 3, ct)
    c.font, c.alignment = Font(size=7), Alignment(horizontal="center", vertical="center")
    if CTF[ct]:
        c.fill = CTF[ct]
    for j in range(nA):
        col = get_column_letter(C0 + j)
        cell = ws.cell(r, C0 + j, f'=IF({col}{cr}=0,"—",IF({col}{cr}>1,{col}{cr},$I$5))')
        cell.number_format, cell.border, cell.font = "#,##0", BOX, Font(size=8)
        cell.alignment = Alignment(horizontal="right", vertical="center")
        if (i, j) not in ON:
            cell.fill = OFFF
        elif i % 2:
            cell.fill = CALC
    t = ws.cell(r, TOTC, f'=SUMIF({get_column_letter(C0)}{r}:{LC}{r},">0")')
    t.number_format, t.fill, t.border = "#,##0", SUMF, BOX
    t.font, t.alignment = Font(size=8, bold=True), Alignment(horizontal="right", vertical="center")

ws.cell(SROW, 2, "Σ arm sizes").font = Font(size=8, bold=True)
for j in range(nA):
    col = get_column_letter(C0 + j)
    c = ws.cell(SROW, C0 + j, f'=SUMIF({col}{A0}:{col}{ALAST},">0")')
    c.number_format, c.fill, c.border = "#,##0", SUMF, BOX
    c.font, c.alignment = Font(size=8, bold=True), Alignment(horizontal="right")
g = ws.cell(SROW, TOTC, f"=SUM({get_column_letter(C0)}{SROW}:{LC}{SROW})")
g.number_format, g.font, g.border = "#,##0", Font(size=8, bold=True), BOX
g.fill = PatternFill("solid", fgColor="BBF7D0")
g.alignment = Alignment(horizontal="right")
ws.cell(
    SROW + 1,
    1,
    f'="Σ arm sizes "&TEXT({get_column_letter(TOTC)}{SROW},"#,##0")&'
    f'"   ·   sequences ORDERED "&TEXT(ROUND($I$5*{eff},-2),"#,##0")&"   ·   "&'
    f'IF(ABS(ROUND($I$5*{eff},-2)-$C$3)<15000,"✓ within budget","⚠ MISMATCH")&'
    '"   ·   CT: agn = cell-type agnostic · joint = shared/both · diff = differential"',
).font = Font(bold=True, size=8)

ws.cell(CH - 1, 1, "CONTROL GRID — edit here (0 = drop · 1 = active · >1 = pin)").font = Font(
    bold=True, size=11, color="92400E"
)
header(CH, CTRLH, "Reservoir strategy")
for i, (fam, name, ct, tier) in enumerate(RESERVOIRS):
    r = CT0 + i
    ws.cell(r, 1, fam).font = Font(size=7.5, bold=True)
    ws.cell(r, 1).fill = PatternFill("solid", fgColor=FAMF[fam])
    nn = ws.cell(r, 2, name)
    nn.font = Font(size=8, bold=tier != "A")
    if tier != "A":
        nn.fill = POOL
    c = ws.cell(r, 3, ct)
    c.font = Font(size=7)
    c.alignment = Alignment(horizontal="center")
    if CTF[ct]:
        c.fill = CTF[ct]
    for j in range(nA):
        cell = ws.cell(r, C0 + j, 1 if (i, j) in ON else 0)
        cell.fill, cell.border = (EDIT if (i, j) in ON else OFFF), BOX
        cell.alignment, cell.font = Alignment(horizontal="center"), Font(size=8)

# widths chosen so the pasted block is 9.2 in wide  (px ~= 7*w + 5, 96 dpi)
for col, w in (("A", 11.0), ("B", 25.5), ("C", 5.5)):
    ws.column_dimensions[col].width = w
for j in range(nA):
    ws.column_dimensions[get_column_letter(C0 + j)].width = 11.0
ws.column_dimensions[get_column_letter(TOTC)].width = 10.0
for r in range(AH, SROW + 1):
    ws.row_dimensions[r].height = 15.5  # 0.215 in -> 20 rows = 4.3 in
ws.freeze_panes = "D9"

# ================= MENU: every candidate, for pruning =================
wsm = wb.create_sheet("Menu")
wsm["A1"] = "Reservoir candidate menu — prune here, then set the CONTROL grid"
wsm["A1"].font = Font(bold=True, size=13)
MENU = [
    (
        "Random",
        "Uniform ACGT",
        "funded (null)",
        "experiment",
        "Null control. Pooled with dinuc-shuffle into one arm.",
    ),
    (
        "Random",
        "Non-uniform ACGT (GC/mononuc-matched)",
        "PRUNE",
        "in-silico",
        "Nearly redundant with dinuc-shuffle, which already preserves mono- AND dinucleotide composition. "
        "Composition matching is a within-arm design choice, not a separate arm.",
    ),
    (
        "Random",
        "Dinuc shuffle of genomic",
        "funded (null)",
        "experiment",
        "Preserves composition, destroys grammar — the informative null.",
    ),
    (
        "Genomic",
        "Gosai et al. (UKBB+GTEx, 400k ref/alt)",
        "ref ALREADY IN TRAINING",
        "—",
        "Ref sequences are the 300k base, so not a reservoir. ALT alleles funded as Δ-supervision: value is "
        "the paired contrast, not sequence novelty.",
    ),
    (
        "Genomic",
        "ENCODE accessible ~2M — open in BOTH",
        "funded",
        "experiment",
        "Tests the joint/shared-accessibility strategy.",
    ),
    (
        "Genomic",
        "ENCODE accessible — CT-differential",
        "funded",
        "experiment",
        "K562-only ∪ HepG2-only in ONE arm, factorial within it. Assayed in both cell types, so specificity "
        "is measured bidirectionally at half the arm cost of separate rows.",
    ),
    (
        "Genomic",
        "Phylogenetic — Zoonomia ortholog CREs",
        "funded",
        "in-silico (clade cutoff)",
        "Evolutionary constraint as the sampling prior; the only reservoir carrying cross-species signal. "
        "OPEN: clade/identity cutoff — ask Anirban.",
    ),
    (
        "Genomic pert.",
        "Mutagenesis, rate sweep (esp. SNV pairs)",
        "funded — TUNED",
        "in-silico",
        "Screen the rate in silico on the oracle, order ONE tuned rate.",
    ),
    (
        "Genomic pert.",
        "Insertions / deletions",
        "folded into 'Structural'",
        "in-silico",
        "Separate arms per operator would cost 3 arms to answer a mix question the oracle can screen.",
    ),
    ("Genomic pert.", "Translocations", "folded into 'Structural'", "in-silico", "As above."),
    ("Genomic pert.", "Inversions", "folded into 'Structural'", "in-silico", "As above."),
    (
        "Genomic pert.",
        "EvoAug combined",
        "funded — UNTUNED CONTROL",
        "experiment",
        "Off-the-shelf default. Pairing it against the two tuned arms measures what in-silico tuning bought (H5).",
    ),
    (
        "Motif-based",
        "Grammar — full motif DB, CT-agnostic",
        "funded",
        "in-silico (params)",
        "The genuinely cell-type-agnostic motif arm: use the whole known motif DB with NO expression "
        "filtering. Teaches grammar, not cell-type identity.",
    ),
    (
        "Motif-based",
        "Grammar — shared-core motifs",
        "funded",
        "in-silico (params)",
        "TFs expressed in BOTH lines (Carl's MYC/AP1 core). Tests whether the shared core suffices.",
    ),
    (
        "Motif-based",
        "Grammar — CT-enriched, mixed factorial",
        "funded",
        "in-silico (params)",
        "Within-arm 2×2: K562-enriched only / HepG2-enriched only / BOTH in one sequence / neither. "
        "The BOTH cell is what teaches differential activity, and attribution survives because the "
        "factorial is inside the arm.",
    ),
    (
        "Motif-based",
        "background · motif count · order · spacing",
        "IN-SILICO AXES",
        "in-silico",
        "Do NOT spend arms enumerating these. Screen on the oracle, order the tuned configuration per "
        "motif-set arm.",
    ),
    (
        "Model-gen.",
        "DNA-LM (HyenaDNA) — unconditioned",
        "funded",
        "experiment",
        "Generative prior, no activity signal.",
    ),
    (
        "Model-gen.",
        "D3 diffusion — unconditioned",
        "PRUNE",
        "in-silico",
        "Redundant with HyenaDNA-unconditioned; the conditioning axis is the interesting one.",
    ),
    (
        "Model-gen.",
        "D3 diffusion — activity-conditioned",
        "funded",
        "experiment",
        "Carl: conditional generation will likely dominate by a large margin. This is that test.",
    ),
    (
        "Model-gen.",
        "D3 diffusion — genomic-conditioned",
        "funded",
        "experiment",
        "Genomic realism without activity targeting.",
    ),
    (
        "Model-gen.",
        "D3 diffusion — uncertainty-conditioned",
        "funded",
        "experiment",
        "GENERATION as active learning — the only arm where acquisition is fused into the generator.",
    ),
]
hdr = ["Family", "Variant", "Status", "Decide by", "Rationale"]
for j, t in enumerate(hdr, 1):
    c = wsm.cell(3, j, t)
    c.fill, c.font = HDR, Font(bold=True, color="FFFFFF", size=9)
for i, row in enumerate(MENU, start=4):
    for j, v in enumerate(row, 1):
        c = wsm.cell(i, j, v)
        c.font = Font(
            size=8.5, bold=(j == 3 and ("PRUNE" in v or "IN-SILICO" in v or "ALREADY" in v))
        )
        c.alignment = Alignment(wrap_text=True, vertical="top")
        if j == 1:
            c.fill = PatternFill("solid", fgColor=FAMF.get(row[0], "FFFFFF"))
for w, col in ((14, "A"), (34, "B"), (24, "C"), (17, "D"), (78, "E")):
    wsm.column_dimensions[col].width = w

# ================= acquisition design =================
wsa = wb.create_sheet("Acquisition")
L = [
    "MATCHED N IS NON-NEGOTIABLE — this is how 'per cell type' avoids doubling",
    "  A per-CT arm does NOT get 2N. It gets N/2 selected by K562 uncertainty + N/2 by HepG2 uncertainty,",
    "  TOTAL N, directly comparable to the joint arm at N. Giving it 2N would confound N with strategy and",
    "  destroy the main-effect test — the extra sequences alone would make it win.",
    "",
    "TEST joint-vs-perCT ONCE, NOT PER FAMILY",
    "  Crossing {joint, per-CT} with 5 acquisition families is 10 arms and there is no power for that.",
    "  Run the contrast in the ONE family where the mechanism is unambiguous — uncertainty is a per-model,",
    "  per-task quantity, so it is CT-dependent by construction. Everything else runs joint. A positive",
    "  result licenses the follow-up; a null one retires the question cheaply.",
    "",
    "WHICH FAMILIES ARE EVEN CELL-TYPE DEPENDENT",
    "  Diversity        NO  — a sequence-space quantity (k-mer / embedding coverage), computed without any",
    "                          model output, so it has no per-CT variant. Principled, not a budget dodge.",
    "  Uncertainty      YES — model-predictive variance, one value per cell type. Gets the joint/per-CT contrast.",
    "  Uncert x Divers  YES, partially — inherits CT-dependence through its uncertainty term only. Run JOINT;",
    "                          its per-CT variant is a second-order effect on an interaction term.",
    "  Activity-strat.  YES — uniform over the K562 range is NOT uniform over the HepG2 range.",
    "",
    "ACTIVITY-STRATIFIED ON THE 2D JOINT GRID — one arm that also delivers the contrast",
    "  Stratify uniformly over bins of (predicted K562, predicted HepG2) rather than over one axis.",
    "  The OFF-DIAGONAL bins (high K562 / low HepG2 and the reverse) ARE contrast-selected sequences, so a",
    "  separate |K562 − HepG2| acquisition column is not needed — it is a stratum inside this arm, and the",
    "  differential contrast is recovered by comparing off-diagonal to diagonal strata.",
    "  This is what frees the 6th column and keeps the table at 10 columns wide.",
    "  Peter wanted an even spread over a wide dynamic range; the 2D version does that AND covers specificity.",
    "  COST (Carl): uniform coverage oversamples LOW-activity sequences — the ones needing ~100k reads per",
    "  read of the high end. This arm drives sequencing cost, and is the candidate for a high/low split assay.",
    "",
    "REDUNDANCY IS MEASURED, NOT ASSUMED",
    "  Peter expects strong correlation WITHIN the uncertainty family and WITHIN the diversity family, and",
    "  uncertainty in regression tracks activity because larger values carry larger absolute error.",
    "  Activity-stratified is NOT a high-activity prior — it samples UNIFORMLY, the opposite of uncertainty,",
    "  which concentrates where error is largest. Keep both, but confirm with the pre-screen:",
    "  ACTION (free, existing 300k model): score the whole pool under every candidate acquisition, take",
    "  pairwise rank correlation AND Jaccard overlap of the top-N selections, cluster, fund one per cluster.",
    "  Same run returns the exact UNION size, which sets N so the order lands on budget.",
]
for k, t in enumerate(L, start=2):
    c = wsa.cell(k, 1, t)
    c.font = Font(size=9, bold=(t == t.upper() and t.strip() != "" and not t.startswith(" ")))
wsa.column_dimensions["A"].width = 106

# ================= schedule =================
ws3 = wb.create_sheet("Schedule")
ws3["A1"] = "Three analyses from the same ordered sequences"
ws3["A1"].font = Font(bold=True, size=13)
rows = [
    ("", "Init", "Training-set schedule", "Span", "Answers", "Tier"),
    (
        "A1  From-scratch per-arm",
        "random",
        "nested 1k · 3k · 10k · 30k · N within each arm",
        "~2 OOMs",
        "which strategy scales better, unconfounded by the 300k base",
        "CONFIRMATORY (level)",
    ),
    (
        "A2  Marginal over base",
        "300k Gosai",
        "base + nested 10k · 30k · N",
        "~0.5 OOM",
        "we already have Gosai — what do we order next?",
        "CONFIRMATORY",
    ),
    (
        "A3  Union corpus",
        "random",
        "nested 10k … 300k … ~3.5M over Gosai + all ordered",
        "~2.5 OOMs",
        "how far does this go; the extrapolation for the funding case",
        "HEADLINE",
    ),
]
for i, r in enumerate(rows, start=3):
    for j, v in enumerate(r, start=1):
        c = ws3.cell(i, j, v)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        if i == 3:
            c.fill, c.font = HDR, Font(bold=True, color="FFFFFF", size=9)
        else:
            c.font = Font(size=9, bold=(j == 1))
for w, col in ((26, "A"), (11, "B"), (38, "C"), (10, "D"), (44, "E"), (21, "F")):
    ws3.column_dimensions[col].width = w
for k, t in enumerate(
    [
        "",
        "REPLICATES.  At each of the 3 smallest D points, train 3 models on DISJOINT subsets of the arm.",
        "Carl: a model trained on one sequence set measures THAT DATASET; to measure the STRATEGY you must train",
        "on different sequence sets drawn from it. Nesting (Rafi: grow 10k→30k, do not resample) and replication",
        "(Carl) come out of the same ordered sequences.",
        "",
        "LEVEL IS CONFIRMATORY, SLOPE IS NOT.  Our slope-variance analysis found reservoir informativeness shows",
        "up as LEVEL, not RATE — slopes were statistically indistinguishable at 3 D points. A1 gives 5 points with",
        "replicates, which makes slope testable, but it is pre-registered EXPLORATORY. Offsets are the powered test.",
    ],
    start=8,
):
    ws3.cell(k, 1, t).font = Font(size=9, italic=True, color="444444")

# ================= hypotheses =================
ws2 = wb.create_sheet("Hypotheses")
H = [
    "CONFIRMATORY — powered, fixed before ordering",
    "  H1  RESERVOIR main effect.  From-scratch (A1), matched N, acquisition = random. Reservoirs differ in",
    "      the LEVEL of the scaling curve. Measures a reservoir as raw sampling, which is what it is.",
    "  H2  ACQUISITION main effect.  Pooled arms, matched N: every acquisition selects from the SAME union",
    "      pool, so acquisition is the only thing varying — a cleaner contrast than the full grid gave.",
    "  H3  MARGINAL informativeness over the 300k Gosai base (A2) differs by reservoir.",
    "  H4  H1 and H3 RANK STRATEGIES DIFFERENTLY, and the divergence tracks distance from the existing",
    "      corpus: strategies duplicating Gosai coverage (ENCODE, EP-PCR) scale fine from scratch yet add",
    "      little at the margin; random and generated sequences do the reverse. Pre-registering this turns",
    "      the corpus-overlap confound into a result.",
    "  H5  IN-SILICO TUNING TRANSFERS.  Tuned mutagenesis and tuned structural arms beat the untuned EvoAug",
    "      control. This is the arm that justifies deciding perturbation and motif parameters in silico",
    "      rather than experimentally — without it, the tuning is an untested assumption.",
    "  H6  JOINT vs PER-CELL-TYPE acquisition (uncertainty family only, matched N).",
    "",
    "EXPLORATORY — reported as such, may lack power",
    "  E1  Scaling EXPONENT differs by strategy (see Schedule for why this is not confirmatory).",
    "  E2  COMPOSITION / enrichment in the pooled arms: which reservoir does each acquisition draw from?",
    "      Enrichment_i = picked_i / expected_i; requires the union pool to offer EQUAL numbers per reservoir.",
    "      Preference is NOT value — a reservoir can be over-picked and not help. Descriptive; H1 licenses",
    "      the informativeness claim.",
    "  E3  SUB-POOL arms (genomic-only / synthetic-only): does acquisition behaviour depend on what is",
    "      available to select from?",
    "  E4  MOTIF-SET provenance: agnostic full-DB vs shared-core vs CT-enriched-mixed, and within the mixed",
    "      arm, the 2×2 factorial (K562-enriched / HepG2-enriched / both / neither).",
    "  E5  A3 extrapolation beyond the measured range.",
    "",
    "WHY THE MOTIF AXIS IS PROVENANCE, NOT 'GENERIC vs SPECIFIC'",
    "  Every motif method looks cell-type specific because you must pick a motif SET. The escape is that the",
    "  set need not be filtered by expression at all: the full known motif DB is genuinely CT-agnostic, and",
    "  teaches grammar rather than cell-type identity. So the axis is WHERE THE MOTIFS COME FROM, at 3 levels:",
    "     full DB (agnostic)  ·  expressed in BOTH (shared core)  ·  CT-enriched (differential)",
    "  'Both' therefore is NOT the union of CT-specific sets. Union/mixing is a DIFFERENT question and lives",
    "  inside the CT-enriched arm as a within-arm factorial, where a sequence carrying a K562-enriched AND a",
    "  HepG2-enriched motif is the cell that actually teaches differential activity — with attribution intact,",
    "  because the single-motif cells are in the same arm. Treating single-CT as 'the generic case' would lose",
    "  the agnostic level, which is the only one that isolates grammar from cell-type identity.",
    "",
    "LEAKAGE — what needs controlling, and what does not",
    "  Motif IDENTITY should NOT be held out. Include every motif we know: that is what a practitioner would",
    "  do, and a model that never saw a motif cannot be expected to generalise to it. The object of study is",
    "  generalisation to novel COMBINATIONS and PERTURBATIONS of known motifs, plus discovery of unknown ones.",
    "  DO control:  (a) generator leakage — hold out motif COMBINATIONS / grammars / spacings, not identities,",
    "  else performance measures memorising the planting procedure;  (b) chromosome holdout including anything",
    "  DERIVED from val/test chromosomes — EP-PCR, EvoAug and Zoonomia are all genome-seeded and the Gosai base",
    "  is genomic, so seeds need a partition disjoint from BOTH (a three-way split, not two);  (c) distributional",
    "  inflation — our motif val A/B measured ~+0.14 that chromosome-based validation did NOT fix, since motif",
    "  sequences are intrinsically easier; mitigation already adopted is to select motif-trained models on a",
    "  genomic-distribution val set.",
    "",
    "ASSAY CALIBRATION",
    "  Re-measure test sets AND the val set in full, else models are selected in old-assay units and scored in",
    "  new ones. ~10k stratified anchors from the 300k train set, strata chosen from ONE HALF of Gosai's",
    "  replicates and calibrated on the other half — selecting on the same noisy values you then regress on",
    "  biases the slope. Integrate with a per-assay readout head on a shared trunk; fit an explicit transform",
    "  first as a diagnostic for whether a global correction is adequate at all.",
]
for k, t in enumerate(H, start=2):
    c = ws2.cell(k, 1, t)
    c.font = Font(size=9, bold=(t == t.upper() and t.strip() != "" and not t.startswith(" ")))
ws2.column_dimensions["A"].width = 108

wb.save(OUT)
print("wrote", OUT)
print(
    f"  MAIN TABLE  rows {AH}-{SROW} x cols A-{get_column_letter(TOTC)}  = "
    f"{SROW - AH + 1} rows x {TOTC} cols"
)
w_in = (11.0 + 25.5 + 5.5 + 11.0 * nA + 10.0) * 7 / 96
print(
    f"  width {w_in:.2f} in   height {(SROW - AH + 1) * 15.5 / 72:.2f} in   (16:9 slide = 10.0 x 5.625 in)"
)
print(f"  arms funded = {len(ON)}  ->  {3_200_000 / len(ON):,.0f} seq/arm at 0% overlap")
print(f"  menu rows = {len(MENU)}   sheets = {wb.sheetnames}")
