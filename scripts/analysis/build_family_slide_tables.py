"""Slide-ready strategy tables: one CSV per reservoir family.

DESCRIPTIVE ONLY -- no eval numbers. These slides answer "what is each strategy
and why are we running it"; results live in the figures. Mixing the two invites
reading a ranking off a design table whose numbers go stale the moment another
cell finishes.

Sized to fill a 16:9 slide as-pasted: 4 columns of substantive text, 4-9 rows.
Google Slides sizes a pasted table from its content, so few wide columns fill the
frame while many narrow ones leave a strip that needs manual resizing.

Families follow the GENERATIVE MECHANISM, not the code module:
  genomic  - real sequence (human or orthologous mammal)
  evoaug   - perturbations applied TO real sequence (incl. mutagenesis)
  random   - synthetic sequence from a composition model
  motif    - synthetic background with deliberate TF-motif content
  guided   - model-in-the-loop generation toward an activity objective
"""

from __future__ import annotations

import csv
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "outputs/figures/slide_tables"
OUT.mkdir(parents=True, exist_ok=True)

HEADER = ["Strategy", "How sequences are made", "What it tests", "Key parameters"]

FAMILIES = {
    "1_genomic": [
        ("genomic", "Real human CREs from held-out training chromosomes",
         "The bar every synthetic strategy must beat; real regulatory syntax at real frequencies",
         "chr-split pool; no transformation"),
        ("zoonomia_relaxed", "Orthologous sequence from all 241 mammals in the Zoonomia alignment",
         "Whether evolutionary variation supplies useful diversity that human sequence alone lacks",
         "min_called_frac=0.80; N-fill=closest; excludes chr7/13"),
        ("zoonomia_far", "Orthologs from the least human-like third by observed identity",
         "Real sequence approaching novel composition - how far from human is still useful",
         "distance=far; identity-ranked tertile"),
        ("zoonomia_near", "Orthologs from the most human-like third",
         "Near-duplicate control: does close orthology add anything over human sequence",
         "distance=near"),
        ("zoonomia_mid", "Orthologs from the middle identity tertile",
         "Interpolates between near and far to locate any optimum",
         "distance=mid"),
        ("zoonomia_primates", "Primate clade only (43 of 241 species)",
         "Clade-restricted vs distance-restricted: which axis of relatedness matters",
         "clade=primates; min_called_frac=0.80"),
    ],
    "2_evoaug": [
        ("evoaug_E7", "Mutation + insertion + deletion applied to real genomic sequence",
         "Best-performing operator subset; isolates local edits from large rearrangements",
         "no translocation/inversion; published rates"),
        ("evoaug_E6", "All published EvoAug operators, 3 augmentations per sequence",
         "Augmentation depth: how many stacked edits before sequence stops being realistic",
         "3 augs/seq; paper operator set"),
        ("evoaug_E4", "All published operators, 2 augmentations per sequence",
         "Lower-depth arm of the same sweep",
         "2 augs/seq"),
        ("evoaug_E5", "All published operators, 4 augmentations per sequence",
         "Upper-depth arm of the same sweep",
         "4 augs/seq"),
        ("evoaug_E2", "Point mutation (5%) plus insertion",
         "Minimal two-operator combination",
         "mutation 5% + insertion"),
        ("evoaug_E3", "Point mutation only, 10% of positions",
         "Single-operator baseline for the family",
         "mutation 10%"),
        ("evoaug_paper2023", "Exact DeepSTARR settings from the EvoAug paper",
         "Reference point: reproduces published configuration for comparability",
         "paper defaults, unmodified"),
        ("mutagenesis", "Uniform random point substitution of genomic sequence",
         "Simplest perturbation; separates 'any edit' from EvoAug's structured operators",
         "random substitution, no indels"),
    ],
    "3_random": [
        ("random", "Uniform random nucleotides",
         "Absolute floor - performance attributable to sequence length and model capacity alone",
         "i.i.d. uniform over ACGT"),
        ("gc_matched", "Random sequence with GC content matched to the genomic distribution",
         "Isolates GC composition from all other genomic structure",
         "per-sequence GC sampled from genomic pool"),
        ("dinuc_shuffle", "Dinucleotide-preserving shuffle of real genomic sequence",
         "Preserves local composition while destroying motif order - the standard negative control",
         "Altschul-Erikson dinucleotide shuffle"),
    ],
    "4_motif": [
        ("motif_ct_dense", "TF motifs planted densely into GC-matched genomic backgrounds",
         "Whether motif DENSITY drives learning beyond motif presence",
         "8-15 motifs/seq; K562+HepG2 accessible TFs; CIS-BP"),
        ("motif_ct_enriched", "Same vocabulary and backgrounds, sparser planting",
         "Motif presence at realistic density; the family reference arm",
         "2-6 motifs/seq; cluster_at=0.99; crispness-weighted"),
        ("motif_jaspar", "Identical procedure, JASPAR2024 vocabulary instead of CIS-BP",
         "Database robustness: does the ranking depend on the motif catalogue itself",
         "JASPAR2024 CORE vertebrates, non-redundant"),
        ("motif_grammar", "Motifs planted with controlled order and spacing",
         "Whether regulatory SYNTAX matters beyond motif content",
         "ordered placement; fixed inter-motif spacing"),
    ],
    "5_model_guided": [
        ("ise_guided", "In-silico evolution of genomic seeds toward predicted high activity",
         "Directly targets the high-activity regime where OOD sequences live (oracle mean 4.0 vs genomic 0.5)",
         "iterative mutate-and-select on oracle score"),
        ("d3_conditioned", "D3 discrete diffusion conditioned on high activity",
         "Generative coverage of high-activity sequence space rather than local hill-climbing",
         "activity-conditioned sampling"),
        ("gpn_star", "GPN-STAR likelihood-guided generation",
         "Sequence-model plausibility as the generative prior instead of an activity oracle",
         "likelihood-guided sampling"),
    ],
}

for fam, rows in FAMILIES.items():
    p = OUT / f"{fam}.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for r in rows:
            w.writerow(list(r))
    print(f"  wrote {p.name}  ({len(rows)} rows x {len(HEADER)} cols)")
print(f"\n  {OUT}")
