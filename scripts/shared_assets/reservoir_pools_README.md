# Reservoir pools (labelled candidate pools for acquisition experiments)

Contact: Trevor (christen@cshl.edu). Written 2026-11-07, zoonomia_orthologs added 2026-11-08.

One `.npz` per reservoir strategy: the canonical draw (generation seed 42), oracle_v3-labelled, at the largest size
the study uses. **Every smaller size is a prefix**, so `sequences[:30000]` is exactly the 30k pool our 30k numbers were
produced on. Do not shuffle before slicing — that turns a size comparison into a comparison of different samples.

`MANIFEST.json` has the same table machine-readably, plus a checksum and the source path of every file.

## What is here

| file | n | GC | label mean ± sd | what the reservoir is |
|---|---|---|---|---|
| `random_d300000.npz` | 300,000 | 0.500 | +0.64 ± 0.95 | uniform random 200-mers; the null every other reservoir has to beat |
| `genomic_d300000.npz` | 300,000 | 0.459 | +0.47 ± 1.09 | real human CREs from the Gosai library, holdout chromosomes removed |
| `evoaug_paper2023_d300000.npz` | 300,000 | 0.462 | +0.45 ± 1.01 | EvoAug-style structural edits (insert/delete/translocate/invert) of real CREs |
| `motif_ct_enriched_d300000.npz` | 300,000 | 0.476 | +0.78 ± 1.23 | synthetic sequences planted with K562-enriched CIS-BP motifs |
| `motif_shared_core_d300000.npz` | 300,000 | 0.464 | +1.21 ± 1.47 | synthetic sequences built from the motif core shared across cell types |
| `mutagenesis_d300000.npz` | 300,000 | 0.462 | +0.45 ± 1.00 | point mutations of real CREs at the study's default rate |
| `zoonomia_orthologs_d300000.npz` | 300,000 | 0.420 | +0.26 ± 0.78 | real orthologous CRE windows from the 241-mammal Zoonomia alignment |
| `gc_matched_d300000.npz` | 300,000 | 0.449 | +0.50 ± 0.91 | random sequences matched to the genomic pool's GC distribution |
| `motif_syntax_core_d300000.npz` | 300,000 | 0.450 | +0.56 ± 1.06 | shared-core motifs placed with learned spacing/orientation syntax |
| `hg38_random_d100000.npz` | 100,000 | 0.408 | +0.29 ± 0.79 | random hg38 windows (not CRE-selected), holdouts removed by sequence |
| `mutagenesis_mr10_d100000.npz` | 100,000 | 0.464 | +0.45 ± 0.94 | mutagenesis at a 10% per-base rate |
| `random_gc_sweep_d100000.npz` | 100,000 | 0.499 | +0.81 ± 1.17 | random sequences swept across GC content |

The first six are the core set the reservoir comparison reports; the rest are controls and variants.

**A note on `zoonomia_orthologs`** (real orthologous CRE windows from the 241-mammal Zoonomia alignment). Every
draw of it before 2026-11-07 was **15.4% exact duplicate sequences** — one window emits one sequence per callable
species, and in a conserved window many species carry the identical 200-mer. The sampler now deduplicates, and the
pools here are the rebuilt ones: 300,000 rows, 300,000 distinct, published 2026-11-08. If you pulled this file
before that date, replace it. The older `zoonomia` arm (human sequence mutated under conservation-derived *rates*)
is deliberately not published: the study dropped it in favour of the real alignment sequences.

**Independent replicate draws** (three per reservoir, used for our error bars) exist but are not published here —
ask if you want them.

## File format

```
sequences      (n,) object   200-bp strings, ACGT
oracle_labels  (n,) float32  oracle_v3 predicted K562 activity
oracle_id      ()   str      "oracle_v3"
strategy, seed ()            provenance, stamped at generation
```

The labels are **oracle predictions, not measurements**. That is the design: the oracle stands in for the assay so a
corpus can be scored before anything is synthesised, and every student in the study is trained on exactly these
labels. The oracle's own accuracy ceiling is r ≈ 0.95 on held-out real sequences and ≈ 0.84 on designed ones.

## The input standard — please use it

Every student in the study sees the 200-bp insert **inside its real plasmid context: 20 real bases each side, a
240-bp input**, with the window shifted by up to ±10 real bases during training, the centred window at evaluation,
and predictions averaged over both strands. This is worth about **+0.07 test r** over the bare insert, so a model
trained on bare 200-mers is not comparable with any number in the study. The plasmid constants are in
`../data/plasmid_flank.txt`; `albench_pools.wrap()` applies them.

## Quick start

```python
import sys; sys.path.insert(0, "/grid/koo/home/shared/albench/reservoir_pools")
import albench_pools as ap

seqs, y = ap.load_pool("random", 30_000)   # candidate pool + its oracle labels
x = ap.wrap(seqs)                          # 240-bp model input (20 real flank bases each side)

val_s, val_y = ap.load_val()               # V0c: early stopping and stack weights. NEVER train on it.
test_s, test_y = ap.load_eval("genomic")   # held-out chr7/13; also "ood", "random_32k", and 16 more
print(ap.pearson(pred, test_y))
```

`python albench_pools.py` prints the inventory. For an acquisition experiment the usual shape is: take a pool, select
a subset by whatever rule you are testing, train on `(subset, labels)`, and report on the evaluation sets — the labels
are already there, so no oracle call is needed unless you generate new sequences.

## Evaluation sets

`../data/eval_sets/` holds all 19, including the three in `../data/test_sets/` (same files). The headline three are
`genomic` (31,435 held-out chr7/chr13 sequences), `ood` (22,962 designed sequences, the hardest set) and `random_32k`.
The rest are the perturbation battery: `sub|ins|del_{low,med,high}`, `translocation`, `inversion`, `dinuc_shuffle`,
`gc_sweep`, `gc_matched`, `motif_planted`, `zoonomia`, `random_10k`, and `snv` (29,383 ref/alt pairs, stored as
`ref_sequences`/`alt_sequences` with `delta_mean`). Note `zoonomia_oracle.npz` carries 9.9% duplicate sequences for
the same reason as the reservoir above; it is an evaluation set, so this inflates how correlated its points are
rather than shrinking a training set, but it is worth knowing.

`genomic_oracle.npz` and `val_V0c_measured.npz` also carry `true_label`: the real measured MPRA activity, for
checking the oracle rather than for training.
