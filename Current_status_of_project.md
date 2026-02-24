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

Layer names: `hidden_0`, `hidden_1`, `output`, `norm` (avoids stale checkpoint collisions with old `hidden1`/`hidden2` names).

### 1.3 Embedding cache (no_shift mode)

- **Mode in use:** `aug_mode=no_shift` — encoder runs once at startup to build canonical + RC caches; head-only training with 50% RC augmentation per sequence. ~20–50× faster per epoch than full mode.
- **Cache location:** `outputs/ag_flatten/embedding_cache/` (shared by all 5 head runs)
- **Cache size:** ~21.5 GB float16 (train + val). Already built Feb 23.
- **Speed:** ~2–4 min/epoch on H100 (bamgpu01). Typical run: 8–12 epochs → 20–50 min total.
- **Production plan:** Once no_shift runs give good performance, rerun with `aug_mode=full` (shift ±15 bp + RC) for final production results.

### 1.4 Evaluation

- **Script:** `eval_ag.py` — loads checkpoint, evaluates on K562 HashFrag test sets under `data/k562/test_sets/`:
  - **ID:** `test_in_distribution_hashfrag.tsv` → Pearson R
  - **SNV:** `test_snv_pairs_hashfrag.tsv` → absolute and delta Pearson R
  - **OOD:** `test_ood_cre.tsv` → Pearson R
- **Batch eval:** `scripts/analysis/eval_boda_k562.py` — loops over all `outputs/ag_*` checkpoints, prints TSV.
- **Malinois baseline:** `scripts/analysis/eval_malinois_baseline.py` — evaluates same HashFrag + chrom-test sets for apples-to-apples comparison.
- **Unified comparison:** `scripts/analysis/compare_malinois_ag.py` — runs both models, prints one TSV.

---

## 2. Current State (Feb 23, 2026)

### Completed runs

| Head | Job | Val Pearson (best) | Epochs | Time |
|------|-----|--------------------|--------|------|
| boda-sum | 652607 | **0.9262** | 11 (early stop) | 34 min |
| boda-mean | 652608 | **0.9277** | 8 (early stop) | 27 min |
| boda-max | 652609 | **0.9258** | 9 (early stop) | 38 min |
| boda-center | 652610 | **0.9233** | 12 (early stop) | 40 min |

### Running

| Head | Job | Status |
|------|-----|--------|
| boda-flatten | **652943** | RUNNING on bamgpu01 — reinit now confirmed working (replaced 4 head param entries) |

### Key bug fixed (Feb 23)

The `reinit_head_params` function in `albench/models/embedding_cache.py` silently failed to replace stale `hidden_0/w` weights (shape 196608×512, from T=128 dummy init by `create_model_with_heads`) with correct ones (7680×512, for T=5). Root cause: the function assumed Haiku's `fresh_params` was a nested dict but it's actually a semi-flat dict (keys like `"head/.../hidden_0"` → value `{"w": tensor}`). Fixed by detecting the flat key format and using direct dict lookup to replace matching keys. Took multiple iterations to diagnose the exact `fresh_params` structure.

---

## 3. Remaining Steps

### 3.1 Immediate

1. **Wait for flatten (652943) to complete** — expect ~8–12 epochs, ~30–50 min. Check for val Pearson ~0.92–0.93 (consistent with other heads).
2. **Confirm `best_model/` checkpoints** exist for all 5 heads under `outputs/ag_*/`.

### 3.2 Evaluation

Once all 5 heads have `best_model/` checkpoints:
1. Run `scripts/analysis/eval_boda_k562.py` (or `compare_malinois_ag.py`) on HPC.
2. Compare all heads + Malinois on HashFrag ID/SNV/OOD test sets.

### 3.3 Production runs (shift augmentation)

After confirming no_shift results are good:
1. Resubmit all 5 heads with `++aug_mode=full` for final production quality runs.
2. Use longer walltime (kooq/koolab partition recommended for 30-day limit).

### 3.4 Optional

- **encoder-1024-dropout:** Reference-style head (`encoder-1024-dropout` arch), submitted as a 6th variant.
- **Compact-window run:** `use_compact_window: true` to compare 600 bp vs adaptive-W approach.

---

## 4. Quick Reference

| Item | Location |
|------|----------|
| Train AlphaGenome head | `experiments/train_oracle_alphagenome_full.py` |
| Config | `configs/experiment/oracle_alphagenome_k562_full.yaml` |
| Boda head runs | `scripts/slurm/train_oracle_alphagenome_full_{flatten,sum,mean,max,center}.sh` |
| Eval one AG checkpoint | `python eval_ag.py <ckpt_dir> <head_name> [arch]` |
| Batch eval Boda heads | `python scripts/analysis/eval_boda_k562.py` |
| Eval Malinois | `python scripts/analysis/eval_malinois_baseline.py` |
| Compare all models | `python scripts/analysis/compare_malinois_ag.py` |
| Shared embedding cache | `outputs/ag_flatten/embedding_cache/` (already built) |
| HPC SSH | `ssh -i ~/.ssh/id_ed25519_citra christen@bamdev4.cshl.edu` |
| Repo on HPC | `/grid/wsbs/home_norepl/christen/ALBench-S2F` |
