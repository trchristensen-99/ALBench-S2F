"""Sequence-budget workbook -- live-editable on Zoom, Slides-exact paste block.

LAYOUT.  Columns A:J rows 8..27 are the paste block: exactly 20 x 10 at 8 pt, 8.6 x 4.3 in, which
fits a 16:9 slide with no shrink-to-fit. The on/off toggles sit in columns L:Q on the SAME ROWS as
the table, not in a separate block below it. That is deliberate: sharing rows means inserting or
deleting a strategy updates the table and its controls in one action, and the numbers visibly move
next to the switch you just flipped -- which is what you want when screen-sharing.

BLANK IS INERT.  A control cell counts as an arm only if it holds exactly 1 (or >1 to pin). If blank
counted as active, inserting an empty row would silently add nA arms and shrink every other arm.

Allocation is EVEN across active arms; synthesis-cost differences are handled by adjusting the
scaling SLOPE per unit cost afterwards, never by ordering unequal numbers of sequences.
"""

import os

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

OUT = os.path.expanduser("~/Downloads/joint_PI_meeting_aug20/sequence_budget.xlsx")

# (family, reservoir, cell-type role, tier)  tier A = single-R arm, B = main pool, C = sub-pool
RESERVOIRS = [
    ("Random", "Null control (uniform ACGT + dinuc-shuffle)", "agn", "A"),
    ("Genomic", "ENCODE accessible - open in BOTH", "joint", "A"),
    ("Genomic", "ENCODE accessible - CT-differential", "diff", "A"),
    ("Genomic", "Zoonomia ortholog CREs (phylogenetic)", "agn", "A"),
    ("Genomic", "Gosai alt alleles (delta-supervision)", "agn", "A"),
    ("Genomic pert.", "Mutagenesis - tuned rate (SNV pairs)", "agn", "A"),
    ("Genomic pert.", "Structural - indel/transloc/inv, tuned", "agn", "A"),
    ("Genomic pert.", "EvoAug combined - untuned control", "agn", "A"),
    ("Motif-based", "Grammar - full motif DB (CT-agnostic)", "agn", "A"),
    ("Motif-based", "Grammar - shared-core motifs", "joint", "A"),
    ("Motif-based", "Grammar - CT-enriched, mixed factorial", "diff", "A"),
    ("Model-gen.", "DNA-LM (HyenaDNA) - unconditioned", "agn", "A"),
    ("Model-gen.", "D3 diffusion - activity-conditioned", "diff", "A"),
    ("Model-gen.", "D3 diffusion - genomic-conditioned", "agn", "A"),
    ("Model-gen.", "D3 diffusion - uncertainty-conditioned", "agn", "A"),
    ("POOLED", "All reservoirs, equal parts", "agn", "B"),
    ("POOLED", "Genomic-derived only", "agn", "C"),
    ("POOLED", "Synthetic / generated only", "agn", "C"),
]
ACQ = [
    "Random",
    "Uncertainty (joint)",
    "Uncertainty (per CT)",
    "Diversity",
    "Uncert. x Divers.",
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
C0 = 4
TOTC = C0 + nA
AH, A0 = 8, 9
ALAST, SROW = A0 + nR - 1, A0 + nR
K0 = TOTC + 2
LC = get_column_letter(C0 + nA - 1)
KC0, KCL = get_column_letter(K0), get_column_letter(K0 + nA - 1)
CRNG = f"${KC0}${A0}:${KCL}${ALAST}"
PRNG = f"${KC0}${A0 + POOL_I[0]}:${KCL}${A0 + POOL_I[-1]}"

wb = Workbook()
ws = wb.active
ws.title = "Budget"
ws["A1"] = "Sequence budget - funded arms"
ws["A1"].font = Font(bold=True, size=14)
ws["A2"] = (
    f"Paste A{AH}:{get_column_letter(TOTC)}{SROW} into Slides - exactly 20 x 10 at 8 pt, fits 16:9 "
    "with no shrink.   Toggles are in L:Q on the SAME ROWS: 1 = on, 0/blank = off, >1 = pin."
)
ws["A2"].font = Font(italic=True, size=9, color="666666")


def lbl(cell, t, **kw):
    ws[cell] = t
    ws[cell].font = Font(size=9, **kw)


lbl("A3", "Synthesis budget", bold=True)
ws["C3"] = 3_200_000
ws["C3"].fill, ws["C3"].number_format = EDIT, "#,##0"
ws["C3"].border, ws["C3"].font = BOX, Font(bold=True, size=9)
lbl("D3", "sequences ORDERED (pooled arms share them)", italic=True, color="666666")
lbl("A4", "Pooled-arm overlap", bold=True)
ws["C4"] = 0.0
ws["C4"].fill, ws["C4"].number_format = EDIT, "0%"
ws["C4"].border, ws["C4"].font = BOX, Font(bold=True, size=9)
lbl(
    "D4",
    "mean pairwise Jaccard of selections - MEASURE IN SILICO before ordering",
    italic=True,
    color="B45309",
)

n_pool = f"COUNTIF({PRNG},1)"
n_all = f"COUNTIF({CRNG},1)"
n_arms = f'({n_all}+COUNTIF({CRNG},">1"))'
eff = f"({n_all}-{n_pool}+IF({n_pool}=0,0,1+MAX(0,{n_pool}-1)*(1-$C$4)))"

lbl("A5", "Arms funded", bold=True)
ws["C5"] = f"={n_arms}"
ws["C5"].font = Font(bold=True, size=9)
lbl("D5", "Effective cells", bold=True)
ws["F5"] = f"=ROUND({eff},1)"
ws["F5"].font = Font(bold=True, size=9)
lbl("G5", "Sequences / arm", bold=True)
ws["I5"] = f'=ROUND(($C$3-SUMIF({CRNG},">1"))/MAX(1,{eff}),-2)'
ws["I5"].number_format, ws["I5"].font = "#,##0", Font(bold=True, size=11)
ws["I5"].fill, ws["I5"].border = SUMF, BOX
ws["A6"] = (
    '=IF(I5>=100000,"OK "&TEXT(I5,"#,##0")&"/arm - from-scratch curve 1k to "&TEXT(I5,"#,##0")'
    '&" (~2 OOMs), 3 replicates at each of the 3 smallest points",'
    'IF(I5>=30000,"1k to 30k curve only (~1.5 OOMs)","TOO SMALL for a per-arm curve"))'
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
h = ws.cell(AH, TOTC, "Sum arm")
h.fill, h.font = HDR, Font(bold=True, color="FFFFFF", size=7.5)
h.alignment = Alignment(wrap_text=True, horizontal="center", vertical="center")

for i, (fam, name, ct, tier) in enumerate(RESERVOIRS):
    r = A0 + i
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
    c.font = Font(size=7)
    c.alignment = Alignment(horizontal="center", vertical="center")
    if CTF[ct]:
        c.fill = CTF[ct]
    for j in range(nA):
        kcol = get_column_letter(K0 + j)
        cell = ws.cell(r, C0 + j, f'=IF({kcol}{r}=1,$I$5,IF({kcol}{r}>1,{kcol}{r},"-"))')
        cell.number_format, cell.border, cell.font = "#,##0", BOX, Font(size=8)
        cell.alignment = Alignment(horizontal="right", vertical="center")
        if (i, j) not in ON:
            cell.fill = OFFF
        elif i % 2:
            cell.fill = CALC
    t = ws.cell(r, TOTC, f'=SUMIF({get_column_letter(C0)}{r}:{LC}{r},">0")')
    t.number_format, t.fill, t.border = "#,##0", SUMF, BOX
    t.font = Font(size=8, bold=True)
    t.alignment = Alignment(horizontal="right", vertical="center")

ws.cell(SROW, 2, "Sum of arm sizes").font = Font(size=8, bold=True)
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
    f'="Sum of arms "&TEXT({get_column_letter(TOTC)}{SROW},"#,##0")&'
    f'"   |   sequences ORDERED "&TEXT(ROUND($I$5*{eff},-2),"#,##0")&"   |   "&'
    f'IF(ABS(ROUND($I$5*{eff},-2)-$C$3)<15000,"within budget","MISMATCH")&'
    '"   |   CT: agn = agnostic, joint = shared/both, diff = differential"',
).font = Font(bold=True, size=8)

# ---- control toggles: SAME ROWS, columns L:Q
ws.cell(AH - 1, K0, "CONTROL - 1 = on, 0/blank = off, >1 = pin").font = Font(
    bold=True, size=9, color="92400E"
)
for j, a in enumerate(ACQ):
    h = ws.cell(AH, K0 + j, a.split(" (")[0][:11])
    h.fill, h.font = CTRLH, Font(bold=True, color="FFFFFF", size=7)
    h.alignment = Alignment(wrap_text=True, horizontal="center", vertical="center")
for i in range(nR):
    for j in range(nA):
        cell = ws.cell(A0 + i, K0 + j, 1 if (i, j) in ON else 0)
        cell.fill, cell.border = (EDIT if (i, j) in ON else OFFF), BOX
        cell.alignment, cell.font = Alignment(horizontal="center"), Font(size=8)

for col, w in (("A", 11.0), ("B", 25.5), ("C", 5.5)):
    ws.column_dimensions[col].width = w
for j in range(nA):
    ws.column_dimensions[get_column_letter(C0 + j)].width = 11.0
ws.column_dimensions[get_column_letter(TOTC)].width = 10.0
ws.column_dimensions[get_column_letter(TOTC + 1)].width = 2.0
for j in range(nA):
    ws.column_dimensions[get_column_letter(K0 + j)].width = 8.5
for r in range(AH - 1, SROW + 1):
    ws.row_dimensions[r].height = 15.5
ws.freeze_panes = "D9"

# ================== DECISIONS: the at-a-glance sheet ==================
wsd = wb.create_sheet("Decisions")
wsd["A1"] = "Every design choice, and why"
wsd["A1"].font = Font(bold=True, size=13)
DEC = [
    (
        "BUDGET",
        "Even N across all funded arms; weights removed",
        "Unequal N confounds synthesis cost with informativeness. Cost enters later as a per-unit-cost "
        "adjustment to the scaling slope, testing the same N per strategy.",
    ),
    (
        "BUDGET",
        "Order the UNION; overlap is a discount, not a surcharge",
        "Pooled arms that pick the same sequence synthesise it once, so overlap BUYS larger arms. "
        "0% -> 128k/arm, 50% -> 160k/arm. Budget never expands.",
    ),
    (
        "GRID",
        "Fund the MARGINS (a plus-shape), not the interior",
        "ANOVA main effects need the margins. The full 18x6 grid is 108 cells at 30k each - one point "
        "per cell, no replicates, no curve. 25 arms at 128k each buys curves WITH error bars.",
    ),
    (
        "GRID",
        "Table capped at 20 rows x 10 columns",
        "16:9 slide at 8 pt. Beyond that Slides shrinks to fit and the numbers stop being legible. "
        "The full candidate list lives on the Menu sheet instead.",
    ),
    (
        "R EFFECT",
        "Single-reservoir arms run at acquisition = random",
        "Peter: reservoirs are just sampling. Random keeps the reservoir measurement unconfounded by "
        "selection, which is what a reservoir actually is.",
    ),
    (
        "A EFFECT",
        "Pooled arms: all acquisitions select from ONE union pool",
        "Pool held fixed means acquisition is the only varying factor - a cleaner contrast than the "
        "full grid ever gave, since there the reservoir varied too.",
    ),
    (
        "R x A",
        "Interaction recovered as composition, not funded as cells",
        "In pooled arms, record which reservoir each pick came from: enrichment = picked/expected. "
        "Gives reservoir preference for free. But preference is NOT value - descriptive only.",
    ),
    (
        "ANALYSIS",
        "Three curves from the SAME ordered sequences",
        "A1 from-scratch per-arm (~2 OOMs, strategy comparison) | A2 marginal over the 300k Gosai base "
        "(~0.5 OOM, the practical question) | A3 union corpus 10k-3.5M (~2.5 OOMs, the headline).",
    ),
    (
        "ANALYSIS",
        "Level is confirmatory, slope is exploratory",
        "Our own slope-variance analysis found reservoir informativeness shows up as LEVEL not RATE - "
        "slopes were indistinguishable at 3 D points. Having the prior null makes this defensible.",
    ),
    (
        "REPLICATES",
        "3 models on DISJOINT subsets at each of the 3 smallest D points",
        "Carl: a model trained on one sequence set measures THAT DATASET, not the method. Peter: "
        "disjoint subsamples serve as replicates. Costs no extra sequences.",
    ),
    (
        "REPLICATES",
        "Nest 10k -> 30k -> N, never resample from scratch",
        "Rafi. The same ordered sequences then yield both the scaling curve and the replicate spread.",
    ),
    (
        "MOTIF",
        "The axis is motif PROVENANCE, not generic-vs-specific",
        "Full known DB unfiltered (agnostic) | expressed in BOTH (shared core) | CT-enriched. Only the "
        "unfiltered level isolates grammar from cell-type identity, which is why we keep it.",
    ),
    (
        "MOTIF",
        "'Both' is NOT the union of CT-specific sets",
        "Mixing is a different question and lives INSIDE the CT-enriched arm as a 2x2 factorial "
        "(K562-only / HepG2-only / both / neither). The both cell teaches differential activity and "
        "attribution survives because the single-motif cells are in the same arm.",
    ),
    (
        "MOTIF",
        "Count, order, spacing, background decided IN SILICO",
        "Enumerating them experimentally would cost ~10 arms to answer what the oracle can screen. "
        "Order one tuned configuration per motif-set arm.",
    ),
    (
        "PERTURB",
        "Indel + translocation + inversion folded into ONE tuned Structural arm",
        "Three separate arms would spend the budget answering a mix question the oracle can screen.",
    ),
    (
        "PERTURB",
        "Keep an UNTUNED EvoAug control arm (H5)",
        "Without it, in-silico tuning is an untested assumption. This one arm measures what the "
        "tuning actually bought and defends the whole in-silico shortcut.",
    ),
    (
        "ACQ",
        "One representative per family, 6 columns",
        "Rafi: one acquisition per family, leave the variance to reservoirs.",
    ),
    (
        "ACQ",
        "Per-CT arm gets N/2 + N/2 = N, never 2N",
        "Matched N is required for the main-effect test. At 2N the extra sequences alone would make "
        "it win, confounding N with strategy.",
    ),
    (
        "ACQ",
        "Joint-vs-per-CT tested in the uncertainty family ONLY (H6)",
        "Crossing {joint, per-CT} with 5 families is 10 arms and there is no power for it. Uncertainty "
        "is per-model per-task, so unambiguously CT-dependent. A null result retires the question cheaply.",
    ),
    (
        "ACQ",
        "Diversity has NO per-CT variant",
        "It is a sequence-space quantity (k-mer / embedding coverage) computed without model output, "
        "so it is CT-independent by construction - a principled exclusion, not a budget dodge.",
    ),
    (
        "ACQ",
        "Contrast |K562-HepG2| folded into activity-stratified 2D",
        "Stratify over bins of (predicted K562, predicted HepG2). The OFF-DIAGONAL bins ARE contrast "
        "selections, so the differential is a stratum inside this arm. Delivers Peter's even spread "
        "over dynamic range and Carl's specificity angle in one arm, and frees the 10th column.",
    ),
    (
        "GOSAI",
        "Ref is not a reservoir; ALT alleles funded as delta-supervision",
        "Ref sequences ARE the 300k base. Alts sit 1 bp from sequences the model already fit, so they "
        "add no coverage - their value is the paired contrast.",
    ),
    (
        "LEAKAGE",
        "Include ALL known motifs; hold out COMBINATIONS, not identities",
        "A model that never saw a motif cannot be expected to generalise to it, and including "
        "everything known is what a practitioner would do. The real risk is memorising the planting "
        "procedure, so hold out grammars/spacings and use a three-way chromosome split (Gosai base is "
        "genomic, and EP-PCR / EvoAug / Zoonomia are all genome-seeded).",
    ),
    (
        "EVAL",
        "Battery of test sets, never a single scalar",
        "Peter: do not trust one eval; different libraries may be optimal for different purposes.",
    ),
    (
        "ASSAY",
        "Re-measure test AND val sets in full; ~10k train anchors",
        "Val in old units with test in new units means selecting models in one space and scoring in "
        "another. Anchor strata picked from ONE HALF of Gosai's replicates and calibrated on the "
        "other half, else regression-to-the-mean biases the slope.",
    ),
    (
        "ASSAY",
        "Integrate old + new via a per-assay head on a shared trunk",
        "Handles sequence-dependent batch effects. Fit an explicit global transform first, purely as a "
        "diagnostic for whether a global correction is adequate at all.",
    ),
]
for j, t in enumerate(["Area", "Decision", "Why"], 1):
    c = wsd.cell(3, j, t)
    c.fill, c.font = HDR, Font(bold=True, color="FFFFFF", size=9)
for i, (a, d, w) in enumerate(DEC, start=4):
    for j, v in enumerate((a, d, w), 1):
        c = wsd.cell(i, j, v)
        c.font = Font(size=8.5, bold=(j <= 2))
        c.alignment = Alignment(wrap_text=True, vertical="top")
    wsd.cell(i, 1).fill = PatternFill("solid", fgColor="F1F5F9")
for w, col in ((13, "A"), (46, "B"), (86, "C")):
    wsd.column_dimensions[col].width = w

# ================== HOW TO EDIT LIVE ==================
wsh = wb.create_sheet("HowToEdit")
HOW = [
    "PRESENTING ON ZOOM",
    "  Share the Excel WINDOW (not the whole screen), set zoom to ~140% so the block fills the frame.",
    "  Toggles sit in L:Q on the same rows as the table, so the audience sees the numbers move next to",
    "  the switch you flip. Keep a static pasted copy on a backup slide in case live editing gets messy.",
    "  If the PIs want to edit it themselves afterwards, upload to Google Sheets - all formulas here are",
    "  plain COUNTIF / SUMIF / IF and survive the import.",
    "",
    "ADD A STRATEGY (row)",
    "  1. Right-click a row INSIDE the block (rows 10-26) and Insert. Never insert at the very first or",
    "     very last row of the block - Excel only auto-expands ranges for interior inserts.",
    "  2. Click the row above, copy it, paste into the new row. This brings both the D:J formulas and the",
    "     L:Q toggles across in one action, because they share rows.",
    "  3. Type the family, name and CT tag in A:C.",
    "  Blank toggles are INERT, so the new row contributes nothing until you type 1 into L:Q.",
    "  NOTE: 20 rows is the Slides ceiling. Past 18 strategies, turn one off rather than adding a row.",
    "",
    "REMOVE A STRATEGY",
    "  Preferred: set its toggles to 0. Nothing breaks, the budget redistributes, and the row stays",
    "  visible as an explicit 'considered and rejected' - useful in front of the PIs.",
    "  Only delete the row if you need the space back for a new strategy.",
    "",
    "ADD AN ACQUISITION (column)",
    "  1. Insert a column inside D:I (e.g. right-click column I -> Insert).",
    "  2. Insert a matching column inside L:Q.",
    "  3. Copy an adjacent table cell across the new column, and an adjacent toggle across the new one.",
    "  4. Type the header. Row and column sums, and the per-arm share, all auto-expand.",
    "  NOTE: 10 columns is the Slides ceiling. A 7th acquisition means dropping the Sum column or the",
    "  CT column from the paste block.",
    "",
    "WHAT RECALCULATES AUTOMATICALLY",
    "  Sequences/arm (I5) - the single driver, = (budget - pinned) / effective cells.",
    "  Effective cells (F5) - applies the pooled-overlap discount.",
    "  Every table cell reads its own row's toggle, so nothing is hard-coded per row.",
    "  Row sums (J), column sums (row 27), the ordered-vs-budget check (row 28).",
    "",
    "THINGS THAT WILL BREAK IT",
    "  Typing a number directly into the TABLE (D:J) - that overwrites the formula. Type into L:Q instead.",
    "  Inserting at the exact top or bottom edge of the block, which leaves ranges unexpanded.",
    "  Sorting the table - the toggles ride along with the rows, but a sort that moves only part of the",
    "  block will desynchronise them. Sort A:Q together or not at all.",
]
for k, t in enumerate(HOW, start=2):
    c = wsh.cell(k, 1, t)
    c.font = Font(size=9, bold=(t.strip() != "" and not t.startswith("  ")))
wsh.column_dimensions["A"].width = 104

# ================== MENU ==================
wsm = wb.create_sheet("Menu")
wsm["A1"] = "Reservoir candidate menu - prune here, then set the toggles"
wsm["A1"].font = Font(bold=True, size=13)
MENU = [
    (
        "Random",
        "Uniform ACGT",
        "funded (null)",
        "experiment",
        "Null control. Pooled with dinuc-shuffle into a single arm.",
    ),
    (
        "Random",
        "Non-uniform ACGT (GC/mononuc-matched)",
        "PRUNE",
        "in-silico",
        "Nearly redundant with dinuc-shuffle, which already preserves mono- AND dinucleotide "
        "composition. Composition matching is a within-arm design choice, not an arm.",
    ),
    (
        "Random",
        "Dinuc shuffle of genomic",
        "funded (null)",
        "experiment",
        "Preserves composition, destroys grammar - the informative null.",
    ),
    (
        "Genomic",
        "Gosai et al. (UKBB+GTEx, 400k ref/alt)",
        "ref ALREADY IN TRAINING",
        "-",
        "Ref sequences are the 300k base, so not a reservoir. ALT alleles funded as delta-supervision.",
    ),
    (
        "Genomic",
        "ENCODE accessible ~2M - open in BOTH",
        "funded",
        "experiment",
        "Tests the joint / shared-accessibility strategy.",
    ),
    (
        "Genomic",
        "ENCODE accessible - CT-differential",
        "funded",
        "experiment",
        "K562-only and HepG2-only in ONE arm, factorial within it. Assayed in both cell types, so "
        "specificity is measured bidirectionally at half the arm cost of separate rows.",
    ),
    (
        "Genomic",
        "Phylogenetic - Zoonomia ortholog CREs",
        "funded",
        "in-silico (clade cutoff)",
        "Evolutionary constraint as the sampling prior; the only reservoir carrying cross-species "
        "signal. OPEN: clade / identity cutoff - ask Anirban.",
    ),
    (
        "Genomic pert.",
        "Mutagenesis, rate sweep (esp. SNV pairs)",
        "funded - TUNED",
        "in-silico",
        "Screen the rate on the oracle, order ONE tuned rate.",
    ),
    (
        "Genomic pert.",
        "Insertions / deletions",
        "folded into Structural",
        "in-silico",
        "Separate arms per operator would cost 3 arms to answer a mix question the oracle can screen.",
    ),
    ("Genomic pert.", "Translocations", "folded into Structural", "in-silico", "As above."),
    ("Genomic pert.", "Inversions", "folded into Structural", "in-silico", "As above."),
    (
        "Genomic pert.",
        "EvoAug combined",
        "funded - UNTUNED CONTROL",
        "experiment",
        "Off-the-shelf defaults. Pairing it against the two tuned arms measures what in-silico tuning "
        "bought (H5).",
    ),
    (
        "Motif-based",
        "Grammar - full motif DB, CT-agnostic",
        "funded",
        "in-silico (params)",
        "The genuinely cell-type-agnostic motif arm: the whole known motif DB with NO expression "
        "filtering. Teaches grammar, not cell-type identity.",
    ),
    (
        "Motif-based",
        "Grammar - shared-core motifs",
        "funded",
        "in-silico (params)",
        "TFs expressed in BOTH lines (Carl's MYC / AP1 core). Tests whether the shared core suffices.",
    ),
    (
        "Motif-based",
        "Grammar - CT-enriched, mixed factorial",
        "funded",
        "in-silico (params)",
        "Within-arm 2x2: K562-enriched only / HepG2-enriched only / BOTH in one sequence / neither.",
    ),
    (
        "Motif-based",
        "background, motif count, order, spacing",
        "IN-SILICO AXES",
        "in-silico",
        "Do NOT spend arms enumerating these. Screen on the oracle, order the tuned configuration.",
    ),
    (
        "Model-gen.",
        "DNA-LM (HyenaDNA) - unconditioned",
        "funded",
        "experiment",
        "Generative prior with no activity signal.",
    ),
    (
        "Model-gen.",
        "D3 diffusion - unconditioned",
        "PRUNE",
        "in-silico",
        "Redundant with HyenaDNA-unconditioned; conditioning is the interesting axis.",
    ),
    (
        "Model-gen.",
        "D3 diffusion - activity-conditioned",
        "funded",
        "experiment",
        "Carl: conditional generation will likely dominate by a large margin. This is that test.",
    ),
    (
        "Model-gen.",
        "D3 diffusion - genomic-conditioned",
        "funded",
        "experiment",
        "Genomic realism without activity targeting.",
    ),
    (
        "Model-gen.",
        "D3 diffusion - uncertainty-conditioned",
        "funded",
        "experiment",
        "GENERATION as active learning - the only arm fusing acquisition into the generator.",
    ),
]
for j, t in enumerate(["Family", "Variant", "Status", "Decide by", "Rationale"], 1):
    c = wsm.cell(3, j, t)
    c.fill, c.font = HDR, Font(bold=True, color="FFFFFF", size=9)
for i, row in enumerate(MENU, start=4):
    for j, v in enumerate(row, 1):
        c = wsm.cell(i, j, v)
        c.font = Font(size=8.5, bold=(j == 3 and v.upper() == v))
        c.alignment = Alignment(wrap_text=True, vertical="top")
        if j == 1:
            c.fill = PatternFill("solid", fgColor=FAMF.get(row[0], "FFFFFF"))
for w, col in ((14, "A"), (34, "B"), (24, "C"), (17, "D"), (78, "E")):
    wsm.column_dimensions[col].width = w

# ================== SCHEDULE ==================
ws3 = wb.create_sheet("Schedule")
ws3["A1"] = "Three analyses from the same ordered sequences"
ws3["A1"].font = Font(bold=True, size=13)
rows = [
    ("", "Init", "Training-set schedule", "Span", "Answers", "Tier"),
    (
        "A1  From-scratch per-arm",
        "random",
        "nested 1k, 3k, 10k, 30k, N within each arm",
        "~2 OOMs",
        "which strategy scales better, unconfounded by the 300k base",
        "CONFIRMATORY (level)",
    ),
    (
        "A2  Marginal over base",
        "300k Gosai",
        "base + nested 10k, 30k, N",
        "~0.5 OOM",
        "we already have Gosai - what do we order next?",
        "CONFIRMATORY",
    ),
    (
        "A3  Union corpus",
        "random",
        "nested 10k ... 300k ... ~3.5M over Gosai + all ordered",
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
for w, col in ((26, "A"), (11, "B"), (40, "C"), (10, "D"), (44, "E"), (21, "F")):
    ws3.column_dimensions[col].width = w
for k, t in enumerate(
    [
        "",
        "REPLICATES. At each of the 3 smallest D points, train 3 models on DISJOINT subsets of the arm.",
        "Carl: a model trained on one sequence set measures THAT DATASET; to measure the STRATEGY you must",
        "train on different sequence sets drawn from it. Nesting (Rafi: grow 10k->30k, do not resample) and",
        "replication (Carl) both come out of the same ordered sequences.",
        "",
        "LEVEL IS CONFIRMATORY, SLOPE IS NOT. Our slope-variance analysis found reservoir informativeness",
        "shows up as LEVEL, not RATE - slopes were statistically indistinguishable at 3 D points. A1 gives 5",
        "points with replicates, making slope testable, but it stays pre-registered as EXPLORATORY.",
    ],
    start=8,
):
    ws3.cell(k, 1, t).font = Font(size=9, italic=True, color="444444")

# ================== HYPOTHESES ==================
ws2 = wb.create_sheet("Hypotheses")
H = [
    "CONFIRMATORY - powered, fixed before ordering",
    "  H1  RESERVOIR main effect. From-scratch (A1), matched N, acquisition = random. Reservoirs differ",
    "      in the LEVEL of the scaling curve.",
    "  H2  ACQUISITION main effect. Pooled arms, matched N: every acquisition selects from the SAME union",
    "      pool, so acquisition is the only thing varying.",
    "  H3  MARGINAL informativeness over the 300k Gosai base (A2) differs by reservoir.",
    "  H4  H1 and H3 RANK STRATEGIES DIFFERENTLY, and the divergence tracks distance from the existing",
    "      corpus: strategies duplicating Gosai coverage scale fine from scratch yet add little at the",
    "      margin. Pre-registering this turns the corpus-overlap confound into a result.",
    "  H5  IN-SILICO TUNING TRANSFERS. Tuned mutagenesis and tuned structural arms beat the untuned",
    "      EvoAug control - the arm that justifies deciding perturbation and motif parameters in silico.",
    "  H6  JOINT vs PER-CELL-TYPE acquisition, uncertainty family only, matched N.",
    "",
    "EXPLORATORY - reported as such, may lack power",
    "  E1  Scaling EXPONENT differs by strategy (see Schedule for why this is not confirmatory).",
    "  E2  COMPOSITION / enrichment in the pooled arms: which reservoir does each acquisition draw from?",
    "      Enrichment = picked/expected; requires the union pool to offer EQUAL numbers per reservoir.",
    "      Preference is NOT value - descriptive; H1 licenses the informativeness claim.",
    "  E3  SUB-POOL arms (genomic-only / synthetic-only): does acquisition behaviour depend on what is",
    "      available to select from?",
    "  E4  MOTIF-SET provenance, and within the CT-enriched arm the 2x2 motif factorial.",
    "  E5  A3 extrapolation beyond the measured range.",
    "",
    "PRE-SCREEN BEFORE ORDERING - free, uses the existing 300k model",
    "  Score the whole pool under every candidate acquisition; take pairwise rank correlation AND Jaccard",
    "  overlap of the top-N selections; cluster; fund one representative per cluster. The same run returns",
    "  the exact UNION size, which sets N so the order lands on budget.",
    "  Peter expects strong correlation within the uncertainty family and within the diversity family, and",
    "  uncertainty in regression tracks activity because larger values carry larger absolute error.",
    "  Activity-stratified is NOT a high-activity prior - it samples UNIFORMLY, the opposite of",
    "  uncertainty. Keep both, but confirm with the screen.",
]
for k, t in enumerate(H, start=2):
    c = ws2.cell(k, 1, t)
    c.font = Font(size=9, bold=(t.strip() != "" and not t.startswith("  ")))
ws2.column_dimensions["A"].width = 108

wb.save(OUT)
w_in = (11.0 + 25.5 + 5.5 + 11.0 * nA + 10.0) * 7 / 96
print("wrote", OUT)
print(
    f"  PASTE BLOCK A{AH}:{get_column_letter(TOTC)}{SROW} = {SROW - AH + 1} rows x {TOTC} cols, "
    f"{w_in:.2f} x {(SROW - AH + 1) * 15.5 / 72:.2f} in"
)
print(
    f"  toggles {KC0}{A0}:{KCL}{ALAST} (same rows)   arms = {len(ON)} -> "
    f"{3_200_000 / len(ON):,.0f} seq/arm"
)
print("  sheets:", wb.sheetnames)
