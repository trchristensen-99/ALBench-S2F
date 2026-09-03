"""Editable inventory tables v2 — adds the Shendure/JB rearrangement family, makes every motif-based
strategy explicit about HOW each step is done, and splits long tables for Google Slides.

Why the explicitness matters: the current motif_planted strategies plant HARD-CODED consensus
hexamers (e.g. 'AGATAA' for GATA1). That is the specific thing likely to draw pushback, so each row
now states the motif SOURCE, the BACKGROUND construction, and the PLACEMENT rule separately, and the
proposed data-derived replacement is listed alongside rather than silently swapped in.
"""

import os

import pandas as pd

OUT = os.path.expanduser("~/Downloads/pi_meeting_figs/editable_tables")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- RESERVOIR
R_COLS = [
    "Family",
    "Variant",
    "Motif source (how identified)",
    "Background (how built)",
    "Placement rule",
    "Key params",
    "Status",
    "Notes",
]
reservoir = [
    # --- natural / control
    [
        "genomic",
        "chr-split real",
        "n/a",
        "real K562 MPRA sequences (chr7/13 held out)",
        "n/a — unmodified",
        "D only",
        "KEEP",
        "anchor + control",
    ],
    [
        "random",
        "uniform",
        "n/a",
        "i.i.d. uniform ACGT, 200 bp",
        "n/a",
        "length",
        "KEEP",
        "base control; best-OOD of the naive controls",
    ],
    [
        "dinuc_shuffle",
        "—",
        "n/a",
        "dinucleotide-preserving shuffle of real genomic (Altschul-Erickson)",
        "n/a — destroys order, keeps composition",
        "—",
        "KEEP",
        "supersedes GC-matched (composition control already implied)",
    ],
    # --- augmentation / evolutionary
    [
        "evoaug",
        "evoaug_heavy, evoaug_structural",
        "n/a",
        "real genomic + stochastic augmentations",
        "insert / delete / translocate / invert / mutate",
        "PARAMETERISE: #augs per seq, aug length range, per-op probs",
        "PARAMETERISE",
        "best in-distribution at 300k (0.914); params currently implicit",
    ],
    [
        "phylogenetic_zoonomia",
        "cutoff variants",
        "n/a",
        "orthologous CRE sequences from Zoonomia whole-genome alignments",
        "n/a — real orthologues",
        "CUTOFF: primates / mammals / all — ASK ANIRBAN",
        "PARAMETERISE",
        "2nd best in-distribution (0.912); which clade cutoff is unresolved",
    ],
    # --- our motif strategies (explicit)
    [
        "motif_planted",
        "v1, v2",
        "HARD-CODED consensus hexamers (9 K562 motifs, e.g. AGATAA=GATA1) + RC",
        "v1 uniform random; v2 real genomic from chr_train pool",
        "3-7 motifs/seq at random positions; v2 can preserve native sites",
        "n_motifs, preserve_native_motifs",
        "REVISE",
        "PUSHBACK RISK: motifs are literal consensus strings, not data-derived",
    ],
    [
        "motif_finemo",
        "genomic-context / random placement",
        "PROPOSED: FiNeMo (or TF-MoDISco) hits on "
        "attribution maps -> real motif instances with coordinates",
        "excise motif spans, dinuc-shuffle the remaining background",
        "re-insert motifs at original coordinates OR at random positions",
        "motif caller, hit threshold, placement mode",
        "PROPOSED",
        "PI-requested replacement for motif_planted; scaffolded, needs FiNeMo hits",
    ],
    [
        "motif_grammar",
        "—",
        "curated K562 motif library",
        "random background",
        "controlled spacing / orientation / block position",
        "spacing, orientation, block position",
        "KEEP",
        "systematic grammar sweep; records full config per sequence",
    ],
    [
        "motif_shuffled",
        "—",
        "motif-support scoring",
        "real genomic",
        "prefer high motif support",
        "motif set, score key",
        "KEEP",
        "",
    ],
    # --- Shendure / JB rearrangement family (NEW)
    [
        "JB_reconstitution",
        "activity-stratified",
        "DMS + ProBound biophysical model on top CREs "
        "(alt: sequence-function model attributions); motif + position recorded with 15 bp flanks",
        "400 endogenous genomic backgrounds, stratified high/med/low activity",
        "motif identity, ORDER and POSITION held FIXED; background VARIED",
        "n backgrounds, activity strata",
        "ADD (JB)",
        "isolates background effect; sequence = background + motifs + positions",
    ],
    [
        "JB_synthetic_thrypsis",
        "5 / 10 / 20 breaks",
        "same as reconstitution (motifs protected)",
        "WT sequence itself (composition preserved exactly)",
        "cut at N breakpoints AVOIDING motifs, then rearrange the fragments",
        "n_breaks in {5,10,20}",
        "ADD (JB)",
        "keeps WT composition + much of the order; isolates arrangement",
    ],
    [
        "JB_random_deposition",
        "—",
        "same as reconstitution",
        "varied background",
        "motif ORDER kept; background AND motif POSITIONS varied",
        "position sampling rule",
        "ADD (JB)",
        "isolates positional effect",
    ],
    # --- variant / mixture
    [
        "prm (partial mutagenesis)",
        "1/5/10/20 pct",
        "n/a",
        "real genomic",
        "random point mutations at rate p",
        "mutation rate",
        "KEEP",
        "attribution/uncertainty-guided variants also exist",
    ],
    [
        "snv_pairs",
        "ref/alt",
        "n/a",
        "real genomic loci",
        "single-base substitution (ref vs alt oligo)",
        "mono vs multi-context",
        "ADD",
        "441k train pairs built + oracle-labelled; enables delta supervision",
    ],
    [
        "mixed_combination",
        "mix3 / mix5 / mix6",
        "inherits",
        "equal parts of component reservoirs",
        "inherits",
        "component set + ratios",
        "KEEP",
        "PI hypothesis: a single strategy may not sustain a power law",
    ],
    # --- removed / deprioritised
    [
        "gc_matched",
        "—",
        "n/a",
        "GC-matched genomic",
        "n/a",
        "—",
        "REMOVE",
        "per feedback: dinuc_shuffle already controls composition",
    ],
    [
        "in_silico_evolution",
        "ISE variants",
        "n/a",
        "oracle-guided optimisation",
        "n/a",
        "—",
        "REMOVE",
        "per feedback: let acquisition select high-activity instead",
    ],
    [
        "uncertainty_guided",
        "—",
        "n/a",
        "oracle-uncertainty sampling",
        "n/a",
        "—",
        "DEPRIORITISE",
        "per feedback",
    ],
]
res_df = pd.DataFrame(reservoir, columns=R_COLS)

# what-varies matrix — the clean way to show the whole design space on one slide
VARY = [
    ["genomic (real)", "fixed (real)", "real", "real", "real"],
    ["dinuc_shuffle", "shuffled", "destroyed", "destroyed", "destroyed"],
    ["random", "synthetic", "none", "n/a", "n/a"],
    ["motif_planted_v2", "real genomic", "consensus (hard-coded)", "random", "random"],
    [
        "motif_finemo (proposed)",
        "dinuc-shuffled",
        "data-derived (FiNeMo)",
        "preserved or random",
        "genomic or random",
    ],
    ["JB reconstitution", "VARIED (400 strat.)", "fixed", "FIXED", "FIXED"],
    [
        "JB synthetic thrypsis",
        "fixed (WT comp.)",
        "fixed",
        "mostly preserved",
        "VARIED (rearranged)",
    ],
    ["JB random deposition", "VARIED", "fixed", "FIXED", "VARIED"],
]
vary_df = pd.DataFrame(
    VARY, columns=["Strategy", "Background", "Motif identity", "Motif order", "Motif position"]
)

# ---------------------------------------------------------------- ACQUISITION
acq = [
    ["random", "uniform sample from the reservoir", "IMPLEMENTED", "baseline"],
    [
        "uncertainty",
        "highest oracle/ensemble disagreement",
        "IMPLEMENTED",
        "DEPRIORITISE per feedback",
    ],
    ["prior_knowledge", "motif/annotation-informed preference", "IMPLEMENTED", ""],
    ["ensemble_acq", "ensemble-derived acquisition score", "IMPLEMENTED", ""],
    [
        "diversity_guided",
        "k-mer farthest-first / k-centre coverage",
        "IMPLEMENTED",
        "added for Step-2",
    ],
    ["badge", "gradient-embedding + k-means++", "NOT IMPLEMENTED", "proposed"],
    ["batchbald", "batch-aware mutual information", "NOT IMPLEMENTED", "proposed"],
    ["bait", "Fisher-information based", "NOT IMPLEMENTED", "proposed"],
    ["combined", "hybrid uncertainty x diversity", "NOT IMPLEMENTED", "proposed"],
    [
        "activity_stratified",
        "sample across the activity spectrum",
        "IMPLEMENTED",
        "relevant to JB reconstitution strata",
    ],
]
acq_df = pd.DataFrame(acq, columns=["Strategy", "Rule", "Status", "Notes"])

# ---------------------------------------------------------------- EVAL SETS
ev = [
    [
        "genomic (in-dist)",
        "held-out chr7/13 real MPRA",
        "31,435",
        "KEEP",
        "primary in-distribution",
    ],
    [
        "OOD designed",
        "designed sequences unlike training",
        "~22k",
        "KEEP",
        "capability probe; ranking flips here",
    ],
    [
        "SNV delta (strict mono)",
        "ref/alt pairs, 1 oligo context per variant",
        "29,383",
        "KEEP",
        "canonical; stamped snv_mono_chrsplit_v1",
    ],
    [
        "SNV delta (incl. multi-context)",
        "same + variants in several oligo contexts",
        "43,104",
        "ADD",
        "report alongside mono: multi-context inflates r (0.91 vs 0.76)",
    ],
    [
        "eQTL / GTEx",
        "measured expression QTLs",
        "TBD",
        "ADD — DATA NOT YET ACQUIRED",
        "the application reviewers will ask about",
    ],
    [
        "substitution / indel / structural",
        "sub, ins, del, inversion, translocation",
        "~32k each",
        "KEEP",
        "",
    ],
    ["dinuc_shuffle / random controls", "negative controls", "10-32k", "KEEP", ""],
]
ev_df = pd.DataFrame(ev, columns=["Eval set", "Definition", "n", "Status", "Notes"])

# ---------------------------------------------------------------- HP AXES
hp = [
    ["FM: branch point", "conv / res_tower / unet1 / full", "full", "where the MPRA head attaches"],
    ["FM: head arch", "linear / mlp / attn / conv", "mlp", "readout"],
    ["FM: freeze transformer", "on / off", "ON", "free preservation: 512bp ~ 8 tokens"],
    [
        "FM: BatchNorm mode",
        "frozen / dual / train",
        "frozen",
        "MPRA inputs corrupt genomic BN stats",
    ],
    [
        "FM: CL mode",
        "none / distill / replay_real / distill_replay",
        "distill",
        "replay uses real targets",
    ],
    ["FM: replay lambda", "1 / 10 / 50", "10", "constraint strength"],
    ["FM: replay rate", "every 1 / 4 / 16 steps", "1", "frequency, independent of lambda"],
    ["FM: anchor length", "512 / 8192 / 524288", "524288", "native geometry is affordable"],
    ["FM: encoder LR mult", "0.01-0.3", "0.1", "drift control"],
    [
        "LegNet (from-scratch)",
        "width, kernel, depth, dropout, wd, lr",
        "PI-feedback broad ranges",
        "width to 1024, ks to 25",
    ],
]
hp_df = pd.DataFrame(hp, columns=["Axis", "Values swept", "Current default", "Notes"])

# ---------------------------------------------------------------- WRITE
res_a = res_df.iloc[:9]  # slide 1: natural/augmentation/our-motif
res_b = res_df.iloc[9:]  # slide 2: JB family, variant, mixture, removed
files = {
    "reservoir_part1_of_2.csv": res_a,
    "reservoir_part2_of_2.csv": res_b,
    "reservoir_FULL.csv": res_df,
    "what_varies_matrix.csv": vary_df,
    "acquisition.csv": acq_df,
    "eval_sets.csv": ev_df,
    "hp_axes.csv": hp_df,
}
for name, df in files.items():
    df.to_csv(os.path.join(OUT, name), index=False)
with pd.ExcelWriter(os.path.join(OUT, "inventory_tables_v2.xlsx")) as xl:
    res_df.to_excel(xl, sheet_name="Reservoir", index=False)
    vary_df.to_excel(xl, sheet_name="WhatVaries", index=False)
    acq_df.to_excel(xl, sheet_name="Acquisition", index=False)
    ev_df.to_excel(xl, sheet_name="EvalSets", index=False)
    hp_df.to_excel(xl, sheet_name="HP_axes", index=False)
print("wrote", len(files) + 1, "files ->", OUT)
for n, d in files.items():
    print(f"  {n:32s} {d.shape[0]} rows x {d.shape[1]} cols")
