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

---

## 5. Current focus (Feb 24, 2026)

### 5.1 Goals

- **AlphaGenome vs Malinois on original K562 test (chr 7, 13)**: Evaluate the four AlphaGenome Boda heads (sum, mean, max, center) on the same chromosome-based K562 test split as Malinois and include them in a unified comparison report.
- **Hybrid (full-shift) training runs**: Train hybrid versions of the four heads that mix cached no_shift embeddings with full encoder+shift augmentation, for a closer match to Malinois’ training distribution but with reduced compute.
- **Robust HPC setup**: Make Slurm jobs and environments robust to missing deps and VPN issues (no assumptions that Cursor can reach the cluster; all cluster-side commands are meant to be run from your local machine on VPN).

### 5.2 New code and scripts

- **Chrom-test evaluation for AlphaGenome**
  - **`eval_ag.py`**: Added `evaluate_chrom_test(ckpt_dir, head_name, data_path="data/k562", arch=None)` which:
    - Loads the same checkpoint/config used for HashFrag eval.
    - Uses `K562FullDataset(..., split="test")` (chr 7, 13) with canonical/RC handling aligned to Malinois.
    - Returns `pearson_r`, `spearman_r`, `mse`, `n` for direct comparison to Malinois’ chrom-test.
  - **`scripts/analysis/eval_ag_chrom_test.py`**:
    - Loops over `boda-sum-512-512`, `boda-mean-512-512`, `boda-max-512-512`, `boda-center-512-512`.
    - For each head, loads `outputs/ag_*/best_model` on HPC and calls `evaluate_chrom_test`.
    - Writes `outputs/ag_chrom_test_results.json` with a dict like `{ "boda_sum": {pearson_r, spearman_r, mse, n}, ... }`.
  - **Slurm wrapper**: `scripts/slurm/eval_ag_chrom_test.sh` (partition `gpuq`, 1 GPU) runs:
    - `uv run python scripts/analysis/eval_ag_chrom_test.py --data_path data/k562 --output outputs/ag_chrom_test_results.json`

- **Unified comparison report**
  - **`scripts/analysis/compare_malinois_alphagenome_results.py`**:
    - Reads:
      - Malinois results: `outputs/malinois_eval_boda2_tutorial/result.json` (chrom-test + HashFrag).
      - AlphaGenome validation baseline: `scripts/analysis/alphagenome_baseline_val_pearson.json`.
      - AlphaGenome HashFrag results: `outputs/ag_hashfrag_results.json` (optional).
      - AlphaGenome chrom-test: `outputs/ag_chrom_test_results.json` (new).
    - CLI arg `--ag_chrom_test_json` (default `outputs/ag_chrom_test_results.json`) feeds a **Section 2** table “Original K562 test set (chr 7, 13)” with:
      - Malinois chrom-test row.
      - One row per AlphaGenome head (boda_sum/mean/max/center) using metrics from `eval_ag_chrom_test.py`.
    - Can be run locally or on HPC via:
      - `uv run python scripts/analysis/compare_malinois_alphagenome_results.py --output outputs/malinois_ag_comparison.md`
    - Supports `--allow_missing_malinois` for draft reports when the Malinois result JSON is not yet present.

- **Hybrid (aug_mode="hybrid") Slurm jobs**
  - Hybrid jobs reuse the no_shift embedding cache (`outputs/ag_flatten/embedding_cache`) and randomly alternate:
    - **Cache path**: canonical/RC from cache, no shift.
    - **Encoder path**: live AlphaGenome encoder with full shift ±15 bp + RC.
  - Updated/added Slurm scripts:
    - `scripts/slurm/train_oracle_alphagenome_full_sum_hybrid.sh`
    - `scripts/slurm/train_oracle_alphagenome_full_mean_hybrid.sh`
    - `scripts/slurm/train_oracle_alphagenome_full_max_hybrid.sh`
    - `scripts/slurm/train_oracle_alphagenome_full_center_hybrid.sh`
    - `scripts/slurm/train_oracle_alphagenome_full_flatten_hybrid.sh`
  - Common settings (as of Feb 24):
    - `#SBATCH --partition=gpuq`
    - `#SBATCH --time=12:00:00`, `--mem=96G`, `--cpus-per-task=8`, `--gpus=1`
    - `uv run python experiments/train_oracle_alphagenome_full.py \`
      - `++head_arch="boda-*_512-512"`
      - `++aug_mode="hybrid"`
      - `++batch_size=64"`
      - `++output_dir=outputs/ag_*_hybrid"`
      - `++cache_dir=outputs/ag_flatten/embedding_cache"`

- **Helper scripts**
  - `scripts/slurm/submit_hybrid_jobs.sh`:
    - Intended to run on HPC from repo root (`/grid/wsbs/home_norepl/christen/ALBench-S2F`).
    - Runs `uv sync` to install Python deps, then `sbatch`’s all five hybrid Slurm scripts.
  - `scripts/sync_and_eval_ag_chrom.sh`:
    - Intended to be run on your local machine (with VPN) from repo root.
    - `rsync`s the repo to HPC and submits `eval_ag_chrom_test.sh`.

### 5.3 Dependency and environment updates

- **`pyproject.toml`**:
  - Added core JAX + optimizer deps so `uv sync` on HPC installs them:
    - `jax>=0.4.0`
    - `jaxlib>=0.4.0`
    - `optax>=0.2.0`
  - This fixes the previous `ModuleNotFoundError: jax` in `experiments/train_oracle_alphagenome_full.py`.
- **External packages**:
  - `experiments/train_oracle_alphagenome_full.py` imports `alphagenome_ft` (external/original AlphaGenome code).
  - On HPC, this still needs to be available in the environment (e.g. installed via pip from a private wheel/repo or added to `PYTHONPATH`); otherwise jobs will fail with `ModuleNotFoundError: alphagenome_ft` after JAX is installed.

### 5.4 HPC job and evaluation status

- **Hybrid training jobs (gpuq)**
  - Recent job IDs (e.g. `653426–653429` on `gpuq`) failed quickly with:
    - `ModuleNotFoundError: No module named 'jax'`.
  - After syncing the updated `pyproject.toml` and running `uv sync` on HPC, these hybrid jobs should progress past that error.
  - Partition `kooq` / QoS `koolab` H100 variants were prototyped but rolled back in the public scripts due to access/queue constraints; current recommended config is `gpuq` with 1 GPU.

- **AlphaGenome chrom-test (chr 7, 13)**
  - Code and Slurm wrapper are in place (`eval_ag_chrom_test.py`, `scripts/slurm/eval_ag_chrom_test.sh`).
  - The evaluation must run **on HPC**, where:
    - `outputs/ag_*/best_model` checkpoints exist.
    - `data/k562` (full K562 MPRA table) is present.
  - From HPC repo root:
    - `sbatch scripts/slurm/eval_ag_chrom_test.sh`
  - Expected output: `outputs/ag_chrom_test_results.json`, used by `compare_malinois_alphagenome_results.py`.

- **Malinois boda2 evaluation**
  - Recent HPC jobs named `malinois_boda2` on `gpuq` (e.g. `653430–653433`) failed early; log inspection and artifact download are still pending on-cluster.
  - Locally, a `outputs/malinois_eval_boda2_tutorial/result.json` exists and is currently used for the comparison report; you may still want to regenerate/verify this on HPC for consistency.

- **VPN / connectivity constraints**
  - From the Cursor environment, `bamdev4.cshl.edu` is **not reachable** (DNS resolution fails or times out), even when your laptop is on VPN.
  - All ssh/rsync/Slurm commands in this document are therefore **instructions for you to run in a local terminal** (where VPN is active), not commands that can be executed by Cursor itself.

### 5.5 Next actions for you

- **On your laptop (with VPN active):**
  - Sync repo to HPC and ensure deps are installed:
    - `rsync -avz --exclude='.git' --exclude='*.tar.gz' --exclude='__pycache__' ./ christen@bamdev4.cshl.edu:/grid/wsbs/home_norepl/christen/ALBench-S2F/`
    - `ssh christen@bamdev4.cshl.edu "cd /grid/wsbs/home_norepl/christen/ALBench-S2F && uv sync"`
  - Submit hybrid runs and chrom-test eval:
    - `ssh christen@bamdev4.cshl.edu "cd /grid/wsbs/home_norepl/christen/ALBench-S2F && bash scripts/slurm/submit_hybrid_jobs.sh && sbatch scripts/slurm/eval_ag_chrom_test.sh"`
  - Monitor:
    - `ssh christen@bamdev4.cshl.edu "squeue -u christen"`

- **After `outputs/ag_chrom_test_results.json` exists (HPC or local copy):**
  - Regenerate comparison report with Malinois + AlphaGenome on chr 7, 13:
    - `uv run python scripts/analysis/compare_malinois_alphagenome_results.py --output outputs/malinois_ag_comparison.md`

