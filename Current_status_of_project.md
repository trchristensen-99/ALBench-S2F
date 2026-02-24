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

## 2. Current State (Feb 24, 2026)

### Completed no_shift runs (doubled-cache protocol, jobs 652947–652951)

All 5 heads completed. Checkpoints at `outputs/ag_*/best_model/`:

| Head | Val Pearson (best) | Status |
|------|--------------------|--------|
| boda-sum | 0.9262 | ✓ DONE |
| boda-mean | 0.9268 | ✓ DONE |
| boda-max | 0.9224 | ✓ DONE |
| boda-center | 0.9195 | ✓ DONE |
| boda-flatten | TBD (check log) | ✓ DONE |

### Malinois baseline (COMPLETED Feb 24)

Job 654751 (`malinois_boda2`) completed successfully. Boda2 tutorial protocol (FlankBuilder + load_model), evaluated on chr 7, 13:

| Metric | Value |
|--------|-------|
| Pearson R | **0.8688** |
| Spearman R | 0.7928 |
| MSE | 0.3259 |

Result saved at `outputs/malinois_eval_boda2_tutorial/result.json` on HPC.

### Running evaluation and hybrid training jobs (Feb 24, jobs 655164–655169)

| Job | Name | Status |
|-----|------|--------|
| 655164 | ag_chrom_test | RUNNING — evaluating boda heads on chr 7,13 → `outputs/ag_chrom_test_results.json` |
| 655165 | ag_sum_hybrid | RUNNING — hybrid training (50% cache + 50% encoder+shift) |
| 655166 | ag_mean_hybrid | RUNNING |
| 655167 | ag_max_hybrid | RUNNING |
| 655168 | ag_center_hybrid | RUNNING |
| 655169 | ag_flatten_hybrid | RUNNING |

All jobs on `gpuq` partition. Hybrid jobs: `aug_mode=hybrid`, `batch_size=64`, output to `outputs/ag_*_hybrid/`.

**Note**: Earlier job batches (654750–654756, 654843–654848) failed due to missing Python packages (`haiku`, `jaxtyping`, `alphagenome` base package). These are now resolved in `setup_hpc_deps.sh`.

---

## 3. Remaining Steps

### 3.1 Immediate (in progress — Feb 24)

- **Wait for job 655164** (ag_chrom_test) to finish. Check: `squeue --me`
- **Retrieve results** once complete:
  ```bash
  # On HPC:
  cat outputs/ag_chrom_test_results.json
  ```
- **Generate comparison report locally** after copying results:
  ```bash
  uv run python scripts/analysis/compare_malinois_alphagenome_results.py \
    --output outputs/malinois_ag_comparison.md
  ```

### 3.2 Hybrid runs (RUNNING, jobs 655165–655169)

Hybrid runs: `aug_mode=hybrid`, 50% cache + 50% live encoder (±15 bp shift + RC). Time limit: 12h on `gpuq`.

If they hit the limit, resubmit on `kooq --qos=koolab`:
```bash
# In each hybrid slurm script, change:
# #SBATCH --partition=gpuq  →  --partition=kooq
# #SBATCH --time=12:00:00   →  --time=24:00:00
# and add: #SBATCH --qos=koolab
```

### 3.3 After hybrid runs complete

1. Eval hybrid heads on chr 7, 13 (update `eval_ag_chrom_test.py` to include `outputs/ag_*_hybrid/best_model`)
2. Rerun comparison report with all heads (no_shift + hybrid)
3. Consider `full` shift aug runs if hybrid results are promising

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
    - `jax>=0.4.0`, `jaxlib>=0.4.0`, `optax>=0.2.0`

- **`scripts/slurm/setup_hpc_deps.sh`** (KEY FILE — sourced by all Slurm scripts):
  - Idempotent script that installs HPC-specific packages missing from `pyproject.toml` into the uv venv.
  - Packages managed (in install order):
    1. `alphagenome==0.6.0` — base DeepMind SDK (not declared by alphagenome_ft)
    2. `alphagenome_ft` — from `~/alphagenome_ft-main` (local, `--no-deps`)
    3. `alphagenome-research` — from GitHub rev `35ea7aa5` (**with deps** — pulls pyfaidx, pyranges, tensorflow, etc.)
    4. `jmp==0.0.4` — mixed-precision for JAX
    5. `jaxtyping` — type annotations for JAX (used throughout alphagenome_ft)
    6. `dm-haiku` — Haiku neural network library
    7. `chex` — JAX test/assertion utilities
    8. `orbax-checkpoint` — JAX checkpoint I/O
  - Each check uses `uv run python -c "import <pkg>"` so subsequent job runs skip already-installed packages.
  - **All Slurm scripts source this before `uv run python`.**

### 5.4 HPC job and evaluation status (Feb 24)

- **Malinois boda2**: COMPLETE. Pearson R = 0.8688, Spearman R = 0.7928 on chr 7,13.
- **AlphaGenome chrom-test (655164)**: RUNNING on bamgpu01. Expected ~30–60 min.
- **Hybrid training (655165–655169)**: RUNNING on bamgpu01/02. Expected ~12h (50 epochs or early stop).

- **SSH note**: Use `ssh -i ~/.ssh/id_ed25519_citra christen@143.48.80.155` (IP instead of hostname when DNS is flaky). The SSH agent often needs re-keying after system sleep: `ssh-add ~/.ssh/id_ed25519_citra`.
- **sbatch on login node**: Use full path `/cm/shared/apps/slurm/current/bin/sbatch` on bamdev4; `sbatch` is available without path inside Slurm job scripts.

### 5.5 Next actions

1. **Check ag_chrom_test results** once job 655164 completes:
   ```bash
   ssh -i ~/.ssh/id_ed25519_citra christen@143.48.80.155 \
     "cat /grid/wsbs/home_norepl/christen/ALBench-S2F/outputs/ag_chrom_test_results.json"
   ```
2. **Run comparison report** locally:
   ```bash
   uv run python scripts/analysis/compare_malinois_alphagenome_results.py \
     --output outputs/malinois_ag_comparison.md
   ```
3. **Monitor hybrid jobs** (~12h): check logs at `logs/ag_*_hybrid-655165*.{out,err}`
4. After hybrid runs: eval them on chr 7,13 (add `outputs/ag_*_hybrid/best_model` to `eval_ag_chrom_test.py`)

