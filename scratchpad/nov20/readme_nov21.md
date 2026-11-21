
---

# Added 2026-11-21 — 1M-scale reservoirs and the mixture pools

For mixed-reservoir scaling experiments. Everything below follows the same conventions as
the rest of this folder: canonical generation seed 42, oracle_v3 labels, 200-bp sequences,
and **every size is a prefix of the next**, so `sequences[:30000]` is exactly the 30k pool.
`MANIFEST_nov21.json` has the machine-readable table, with each file's row count, distinct
count, GC, label mean/sd, provenance stamps and source path.

Nesting was re-verified file by file for everything published here, not assumed: each
strategy's full size chain was loaded and checked to be a true prefix chain. Duplicate
counts were measured too and are in the manifest.

## New single-reservoir pools

| file | n | sizes available by slicing | what it is |
|---|---|---|---|
| `motif_shared_core_d1000000.npz` | 1,000,000 | 10k 30k 55k 80k 100k 105k 300k 400k 500k 600k 1M | synthetic sequences from the motif core shared across cell types |
| `motif_ct_enriched_d1000000.npz` | 1,000,000 | 10k 30k 100k 300k 1M | synthetic sequences planted with K562-enriched CIS-BP motifs |
| `dinuc_shuffle_d300000.npz` | 300,000 | 10k 30k 100k 300k | dinucleotide-preserving shuffles of real CREs |
| `evoaug_ours_default_d300000.npz` | 300,000 | 10k 30k 100k 300k | EvoAug-style structural edits at our default parameters (the `evoaug_paper2023` file already here uses the published 2023 settings instead) |
| `evo2_d100000.npz` | 100,000 | 10k 30k 100k | model-generated sequences (Evo2) |
| `d3_steps64_d100000.npz` | 100,000 | 10k 30k 100k | discrete diffusion, 64 sampling steps |
| `d3_steps256_d100000.npz` | 100,000 | 10k 30k 100k | discrete diffusion, 256 sampling steps |
| `accessible_hepg2_only_d100000.npz` | 100,000 | 10k 30k 100k | CREs accessible in HepG2 but not K562 |
| `accessible_k562_only_d30000.npz` | 30,000 | 10k 30k | CREs accessible in K562 but not HepG2 |
| `accessible_tiles_k562_d100000.npz` | **60,689** | 10k 30k (full) | tiles across K562-accessible regions — **the pool exhausts at 60,689**, so the file is short of its name; slice at or below that |

The two 1M files are the only reservoirs that reach 1M. The genomic pool cannot: real
sequence caps near 315k (300k replicate draws are ~95% identical), which is why the
3.2M library cannot be mostly real CREs and why mixtures matter at all.

## Mixture pools — `mixtures/`

Sixteen mixture families, each a fixed recipe over the single reservoirs, published at the
largest size with smaller sizes available by slicing. All sixteen were verified nested.

| family | sizes | recipe |
|---|---|---|
| `mix_ms_mc_c` | 10k 30k 100k 300k **1M** | shared-core + cell-type-enriched motifs, canonical draw |
| `mix_ms_mc_r1/r2/r3` | 10k 30k 100k 300k | the same recipe, three independent draws (use these for error bars) |
| `mix_ms1_mc2_r1/r2/r3` | 10k 30k 100k 300k | 1:2 shared-core to cell-type-enriched |
| `mix_ms2_mc1_r1/r2/r3` | 10k 30k 100k 300k | 2:1 shared-core to cell-type-enriched |
| `mix_ms_gen_r1/r2/r3` | 100k 300k | shared-core motifs + genomic |
| `mix_ms_mc_gen_r1/r2/r3` | 100k 300k | both motif reservoirs + genomic |

A warning that cost us real time: **a re-run on the same pool is nearly as variable as a
new pool**, so a single run pair proves nothing about a mixture ratio. Use the `r1/r2/r3`
families, and analyse within a replicate set rather than across.

## Two things deliberately NOT published here

`zoonomia_hx` exists in our pools but is held back: it carries no `params` provenance
stamp, has no registry entry we can point at, and 423 of its 100,000 rows are exact
duplicates. We have been burnt before by artefacts whose provenance could not be
reconstructed. Ask if you want it and we will regenerate it with a stamp.

The older `zoonomia` arm (human sequence mutated under conservation-derived *rates*, as
opposed to the real orthologous windows in `zoonomia_orthologs`) remains unpublished for
the reason given in the Nov-8 section above.

## Reminders that still apply

- `val_v0c` is the validation set. **Never train on it.** It is for early stopping and
  stack weights only.
- Labels are oracle_v3 predictions, not measurements — by design, so a corpus can be scored
  before anything is synthesised.
- Use the 240-bp input standard (20 real plasmid bases each side, ±10 shift in training,
  centred window at evaluation, both strands averaged). It is worth about +0.07 test r, so
  a model trained on bare 200-mers is not comparable with any number in the study.
  `albench_pools.wrap()` applies it.
