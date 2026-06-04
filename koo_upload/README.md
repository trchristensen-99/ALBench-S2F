# Malinois / lentiMPRA Dataset — Koo Lab Shared Copy

> **Placement:** this bundle is the **Gosai et al. 2024** lentiMPRA (the data
> used to train Malinois). It is a **different** dataset from the
> `lentimpra/agarwal_2025/` already in koo shared storage, so it lives in a
> sibling directory: `/grid/koo/home/shared/data/lentimpra/gosai_2024/`.

## Overview

Large-scale **lentiMPRA** dataset from the Malinois / Boda2 work
(Gosai et al., *Nature* 2024 — "Machine-guided design of
cell-type-targeting cis-regulatory elements"). It measures cis-regulatory
activity of **798,064** 200-bp sequences across three human cell lines:

- **K562** — chronic myelogenous leukemia / erythroleukemia
- **HepG2** — hepatocellular carcinoma
- **SK-N-SH** — neuroblastoma

Activity is reported as **log2 fold-change (RNA / DNA)** per cell type, each with
a standard error. This is the dataset used to train the **Malinois** oracle.

## Files

### 1. Core MPRA dataset — `DATA-Table_S2__MPRA_dataset.txt`
268 MB, 798,064 data rows (+ 1 header), tab-separated. One file holds **all three
cell types** (in the source repo the `hepg2/` and `sknsh/` copies are just
symlinks back to this file).

| column | description |
|--------|-------------|
| `IDs` | element / variant ID, e.g. `7:70038969:G:T:A:wC` (chr:pos:ref:alt:allele:tag) |
| `chr` | chromosome (used for chromosome-split train/val/test) |
| `data_project` | source sub-library (e.g. `UKBB` = UK Biobank GWAS-variant elements) |
| `OL` | oligo / library design index |
| `class` | trait / annotation labels for the element (e.g. `BMI,BFP`, `Depression_GP`) |
| `K562_log2FC`, `HepG2_log2FC`, `SKNSH_log2FC` | activity (log2 RNA/DNA) per cell type |
| `K562_lfcSE`, `HepG2_lfcSE`, `SKNSH_lfcSE` | standard error of each log2FC |
| `sequence` | **200-bp** element sequence (ACGT) |

### 2. Controls & SNV pairs (`controls/`)

**2a. Negative/shuffled controls — Agarwal et al. 2025.** These three files come
from the **Agarwal 2025** K562 lentiMPRA (sourced from our repo's
`data/agarwal_2025/`), a *different* paper from the core Gosai dataset. They are
bundled here only because the Malinois pipeline uses them as negative/calibration
controls:
- `k562_all_controls_200bp.tsv` — 500 control elements (`name`, `category`, `sequence`).
- `k562_shuffled_controls_200bp.tsv` — 250 shuffled-sequence negative controls (`name`, `sequence`).
- `k562_dinucleotide_shuffled_controls.csv` — 250 dinucleotide-preserving shuffled
  controls **with measured activity** (`sequence_230nt`, `element_200nt`,
  `log2_rep1..3`, `log2_mean`).

**2b. SNV variant-effect pairs — derived from the Gosai dataset itself.**
- `train_snv_pairs_clean.tsv` — 14,356 ref/alt SNV pairs for variant-effect (SNV
  delta) evaluation (`variant_key`, `ref_idx`, `alt_idx`, `ref_log2fc`,
  `alt_log2fc`, `delta_log2fc`). These are built **from `DATA-Table_S2`**, not
  from Agarwal: `scripts/create_k562_test_sets.py` parses the `IDs` column
  (`chr:pos:ref:alt:allele:tag`), keeps reference (`allele R`) + alt (`allele A`)
  rows that are true SNVs, drops duplicate `variant_key`s (poly-/multi-allelic
  loci), and merges ref↔alt on `variant_key`. Same provenance as the core
  dataset — Gosai et al. 2024.

### 3. Trained Malinois model (`model/`)
`malinois_artifacts__20211113_021200__287348.tar.gz` — 51 MB. Contains:
- `artifacts/torch_checkpoint.pt` — PyTorch checkpoint (Malinois, Basset-branched architecture).
- `artifacts/lightning_logs/version_0/checkpoints/epoch=97-step=175517.ckpt` — Lightning checkpoint.
- `artifacts/lightning_logs/version_0/hparams.yaml` — training hyperparameters.

## Pipeline: flanking sequences & padding

MPRA elements are **200 bp** inserts. Models with a larger receptive field
(AlphaGenome S2F = **600 bp**; the "compact" variant = **384 bp**) pad each
element with the **real MPRA construct flanks** (not N's), taken verbatim from
`boda2-main/boda/common/constants.py`. **Each flank constant is 300 bp:**

**5′ upstream flank (300 bp):**
```
ACGAAAATGTTGGATGCTCATACTCGTCCTTTTTCAATATTATTGAAGCATTTATCAGGGTTACTAGTACGTCTCTCAAGGATAAGTAAGTAATATTAAGGTACGGGAGGTATTGGACAGGCCGCAATAAAATATCTTTATTTTCATTACATCTGTGTGTTGGTTTTTTGTGTGAATCGATAGTACTAACATACGCTCTCCATCAAAACAAAACGAAACAAAACAAACTAGCAAAATAGGCTGTCCCCAGTGCAAGTGCAGGTGCCAGAACATTTCTCTGGCCTAACTGGCCGCTTGACG
```

**3′ downstream flank (300 bp):**
```
CACTGCGGCTCCTGCGATCTAACTGGCCGGTACCTGAGCTCGCTAGCCTCGAGGATATCAAGATCTGGCCTCGGCGGCCAAGCTTAGACACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGTTGGTAAAGCCACCATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCT
```

**Padding rule** (`data/k562_full.py` → `apply_boda_padding`):
- Target window **W = 600 bp** (AlphaGenome). For a 200-bp element, `pad_needed =
  400`, split symmetrically: `left = UPSTREAM[-len_up:]`, `right =
  DOWNSTREAM[:len_down]`, with `len_up + len_down = W - len(seq)`. If the 300-bp
  flanks are shorter than needed, the remainder is N-padded to W.
- Compact window **W = 384 bp** (or `min_var_len + 2*flank_bp`): built as
  `left_flank + variable + right_flank` with the full variable retained and flanks
  sliced to fit; an optional `shift` redistributes flank between the two sides
  (used for shift-augmentation).

So a deployed model input is `UPSTREAM_slice + <200-bp element> + DOWNSTREAM_slice`
centered to the model's window length.

## Provenance

- Source repo: `ALBench-S2F` (CSHL) — core dataset + SNV pairs from `data/k562/`,
  Agarwal negative/shuffled controls from `data/agarwal_2025/`, model from `data/`.
- Core dataset + SNV pairs: Gosai et al., *Nature* 2024 (Boda2 / Malinois lentiMPRA).
- Negative/shuffled controls (`controls/` 2a only): Agarwal et al. 2025 K562 lentiMPRA.
- Flank constants: `boda2-main/boda/common/constants.py`.
