# Where the data lives

`albench` code is versioned in git. Data is not. This document says where each kind
of data lives and how to obtain it.

## Why data is not in git

Until 2026-09-15 roughly 800 MB of derived experiment intermediates were tracked,
which made a fresh clone download ~300 MB of files that no current code path reads.
They are now ignored (see `.gitignore`) and kept on disk / archived.

**Nothing was deleted.** `git rm --cached` removes a file from the index while
leaving it in the working tree, so every one of these directories is still present on
both the laptop and the cluster checkout.

## The three kinds of data

### 1. Assets `albench` needs to run

Declared in `albench/paths.py`, each with an environment override, a default
location, and instructions. Run:

```bash
albench doctor      # what you have, what is missing, and how to get each one
```

These are downloadable or buildable from scratch: hg38, JASPAR/CIS-BP PFMs, ENCODE
peaks and TF sets, Zoonomia rates, the oracle ensemble.

### 2. Primary datasets

`data/k562/`, `data/yeast/` — ignored by git from the start. These are the MPRA
measurements everything else derives from. Obtain from the lab share or the original
publications.

### 3. Derived experiment intermediates (untracked as of 2026-09-15)

```
data/neg_aug_final_push/            data/synthetic_negatives_corrected/
data/neg_aug_aggressive/            data/synthetic_negatives_cpg_aware/
data/neg_aug_combos/                data/synthetic_negatives_gosai_native/
data/cpg_augmentation/              data/cpg_counterfactual/
data/agarwal_2025/
```

**Reproducibility status, stated honestly:** only
`data/synthetic_negatives_corrected/` has a generating script in the repo
(`scripts/generate_corrected_negatives.py`). The `neg_aug_*`, `cpg_*` and remaining
`synthetic_negatives_*` directories were produced by exploratory work whose scripts
were not kept, so they are **not reproducible from this repo** and exist only as
files. Treat them as archive material.

`data/agarwal_2025/` is different: it is still referenced by ~15 scripts and is
partly re-obtainable —
- `ENCFF252GNM.tsv`, `ENCFF857LYJ.tsv` — ENCODE, by accession
- `Table_S3/S6/S7*.xlsx` — supplementary files of Agarwal et al. 2025
- `k562_*_controls*.tsv/csv` — derived locally; not re-obtainable from a public source

## If you need one of these directories

Ask Trevor. Because several are not reproducible, they should be copied to durable
storage rather than regenerated, and any purge of git history must happen only after
that copy exists and has been verified.
