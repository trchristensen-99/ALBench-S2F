# K562 baseline comparison: AlphaGenome adapter vs Malinois (1-pager for colleagues)

## What we did

- Trained **AlphaGenome adapter heads** (frozen AlphaGenome encoder + small Boda-style head) on the full K562 MPRA dataset using the **same chromosome splits** as the Malinois paper (val: chr 19, 21, X; test: chr 7, 13).
- Compared to the **Malinois** baseline (published K562 model from the boda2/CODA pipeline).

## Results (validation, chr 19, 21, X)

| Model | Val Pearson R |
|-------|----------------|
| **Malinois** (paper/notebook) | 0.913 |
| **AlphaGenome** (best head) | **0.928** (boda-mean) |
| **AlphaGenome** (range across heads) | 0.923 – 0.928 |

AlphaGenome adapter heads (boda-sum, boda-mean, boda-max, boda-center) all reach **~0.92–0.93** val Pearson on the same validation set.

## Results (test, chr 7, 13)

- **Malinois** (boda2 tutorial): test Pearson **~0.88–0.89** (reported in notebook).
- **AlphaGenome** chrom-test: not yet in our eval pipeline (current eval reports HashFrag splits only); val performance suggests test would be competitive.

## Takeaway

- On the **original K562 dataset** (chromosome splits, not HashFrag), the **AlphaGenome adapter** matches or exceeds Malinois validation performance (0.92–0.93 vs 0.91).
- **Note:** The Malinois eval job on the HPC failed because the Malinois artifact was not present (`$HOME/malinois_artifacts__...tar.gz`). Download with `gsutil cp gs://tewhey-public-data/CODA_resources/malinois_artifacts__20211113_021200__287348.tar.gz .` on the HPC, then resubmit `scripts/slurm/eval_malinois_boda2_tutorial.sh` to get actual test Pearson (expected 0.88–0.89).
- All AlphaGenome numbers are from **verified completed training runs** (see `Current_status_of_project.md` and `scripts/analysis/alphagenome_baseline_val_pearson.json`).
