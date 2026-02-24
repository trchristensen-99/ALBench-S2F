# Current status of project

Summary of work done and remaining steps to efficiently train adapter heads on lentiMPRA (K562) and compare to Malinois on the same dataset.

---

## 1. Architecture & Training Setup

### 1.1 Model

- **Architecture:** Frozen AlphaGenome encoder (JAX/Haiku) + trainable adapter head. Encoder outputs (B, T=5, D=1536) for 600 bp input; head predicts K562 log2FC (regression).
- **Data splits:** Full K562 Malinois-style chromosome-based splits (val = chr 19,21,X; test = chr 7,13). N≈627k train / 58k val / 62k test.
- **Training script:** `experiments/train_oracle_alphagenome_full.py`
- **Config:** `configs/experiment/oracle_alphagenome_k562_full.yaml` — lr=0.001, weight_decay=1e-6, epochs=50, early stopping patience=5 on val Pearson.

### 1.2 Head architectures (v4)

Five Boda-style heads, all registered as `alphagenome_k562_head_{arch_slug}_v4`:

| Arch slug | Pooling | Hidden layers |
|-----------|---------|---------------|
| `boda-flatten-512-512` | Flatten T×D → Linear | norm → flatten → 512 → 512 → 1 |
| `boda-sum-512-512` | Sum over T | norm → 512 → 512 → 1 → sum |
| `boda-mean-512-512` | Mean over T | norm → 512 → 512 → 1 → mean |
| `boda-max-512-512` | Max over T | norm → 512 → 512 → 1 → max |
| `boda-center-512-512` | Center token | norm → 512 → 512 → 1 → center |

Layer names: `hidden_0`, `hidden_1`, `output`, `norm` (avoids stale checkpoint collisions).

### 1.3 Embedding cache (no_shift mode)

- **Mode:** `aug_mode=no_shift` — encoder runs once at startup to build canonical + RC caches; head-only training uses **all** precomputed embeddings every epoch.
- **Training protocol:** For each batch, two gradient steps are taken — one on canonical embeddings and one on RC embeddings with the same labels. This fully utilises both cached files (~2× gradient updates per epoch vs random 50% RC sampling) at zero additional compute cost.
- **Cache location:** `outputs/ag_flatten/embedding_cache/` (shared by all 5 head runs)
- **Cache contents:** N=627k canonical embeddings (N, T=5, D=1536) float16 + N RC embeddings with same labels = ~21.5 GB total. Already built Feb 23.
- **Speed:** ~4–8 min/epoch on H100 (bamgpu01) with doubled updates. Expect early stop at ~6–10 epochs → 30–80 min total.
- **Production plan:** Once no_shift runs converge, rerun with `aug_mode=full` (shift ±15 bp + RC) for final production results.

### 1.4 Evaluation

- **Script:** `eval_ag.py` — evaluates on K562 HashFrag test sets under `data/k562/test_sets/`:
  - **ID:** `test_in_distribution_hashfrag.tsv` → Pearson R
  - **SNV:** `test_snv_pairs_hashfrag.tsv` → absolute and delta Pearson R
  - **OOD:** `test_ood_cre.tsv` → Pearson R
- **Batch eval:** `scripts/analysis/eval_boda_k562.py` — loops over all `outputs/ag_*` checkpoints.
- **Malinois baseline:** `scripts/analysis/eval_malinois_baseline.py` — same HashFrag + chrom-test sets.
- **Unified comparison:** `scripts/analysis/compare_malinois_ag.py`.

---

## 2. Current State (Feb 23, 2026)

### Running jobs (all 5 heads, doubled-cache protocol)

| Job | Head | Status |
|-----|------|--------|
| 652947 | boda-flatten | PENDING → bamgpu01 |
| 652948 | boda-sum | PENDING → bamgpu01 |
| 652949 | boda-mean | PENDING → bamgpu01 |
| 652950 | boda-max | PENDING → bamgpu01 |
| 652951 | boda-center | PENDING → bamgpu01 |

All using `aug_mode=no_shift` + shared cache at `outputs/ag_flatten/embedding_cache/`. No cache rebuild needed.

### Previous runs (old protocol — 50% random RC sampling)

These completed with good results and are useful as a baseline:

| Head | Val Pearson (best) | Epochs | Time |
|------|--------------------|--------|------|
| boda-sum | 0.9262 | 11 | 34 min |
| boda-mean | 0.9277 | 8 | 27 min |
| boda-max | 0.9258 | 9 | 38 min |
| boda-center | 0.9233 | 12 | 40 min |

### Key fixes applied (Feb 23)

1. **reinit_head_params Layout3:** `fresh_params` from Haiku's `init()` uses a semi-flat key format (`"head/.../hidden_0" → {"w": tensor}`). Fixed by direct dict lookup instead of complex navigation.
2. **All-cache training:** no_shift loop now processes both canonical and RC embeddings per batch (same labels for RC), fully utilising both cache files per epoch.

---

## 3. Remaining Steps

### 3.1 Immediate

- Wait for jobs 652947–652951 to complete. Expect ~6–10 epochs early stop.
- Check `outputs/ag_*/best_model/` checkpoints exist for all 5 heads.

### 3.2 Evaluation

Once all 5 heads have `best_model/` checkpoints:
1. Run `scripts/analysis/eval_boda_k562.py` on HPC for HashFrag ID/SNV/OOD results.
2. Run `scripts/analysis/compare_malinois_ag.py` for side-by-side Malinois comparison.

### 3.3 Production runs (shift augmentation)

After confirming no_shift results:
1. Resubmit with `++aug_mode=full` for final production runs (encoder runs every step, full shift ±15 bp + RC).
2. Use `kooq/koolab` partition for longer time limits.

### 3.4 Optional

- `encoder-1024-dropout` head: reference-style single hidden layer (1024 units + dropout).
- Compact-window run: `use_compact_window: true` for adaptive-W approach.

---

## 4. Quick Reference

| Item | Location |
|------|----------|
| Train AlphaGenome head | `experiments/train_oracle_alphagenome_full.py` |
| Config | `configs/experiment/oracle_alphagenome_k562_full.yaml` |
| Boda head run scripts | `scripts/slurm/train_oracle_alphagenome_full_{flatten,sum,mean,max,center}.sh` |
| Eval one AG checkpoint | `python eval_ag.py <ckpt_dir> <head_name> [arch]` |
| Batch eval Boda heads | `python scripts/analysis/eval_boda_k562.py` |
| Eval Malinois | `python scripts/analysis/eval_malinois_baseline.py` |
| Compare all models | `python scripts/analysis/compare_malinois_ag.py` |
| Shared embedding cache | `outputs/ag_flatten/embedding_cache/` (already built, ~21.5 GB) |
| HPC SSH | `ssh -i ~/.ssh/id_ed25519_citra christen@bamdev4.cshl.edu` |
| Repo on HPC | `/grid/wsbs/home_norepl/christen/ALBench-S2F` |
