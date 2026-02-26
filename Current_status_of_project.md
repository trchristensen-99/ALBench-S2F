# Current status of project

Summary of work done and remaining steps to efficiently train adapter heads on lentiMPRA (K562) and compare to Malinois on the same dataset.

---

## 0. Codebase Restructure (Feb 25, 2026 — IN PROGRESS)

Per PI feedback, the repository is being restructured so that `albench/` becomes a **lightweight, model-agnostic active-learning engine** anyone can use with their own models and tasks. Application-specific code moves outside of it.

### New top-level layout

| Directory | Contents | Status |
|-----------|---------|--------|
| `albench/` | AL engine only: `SequenceModel` ABC, `ALLoop`, reservoir/acquisition strategies | Slimmed down |
| `data/` | Dataset loaders (K562, Yeast), `TaskConfig`, hash-frag splits, utils | Moved from `albench/data/` + `albench/task.py` |
| `models/` | DREAM-RNN, AlphaGenome heads/wrappers, oracle & student wrappers, training utils | Merged from `albench/models/` + `albench/oracle/` + `albench/student/` |
| `evaluation/` | Scaling-curve utilities, yeast test-set helpers | Moved from `albench/evaluation.py` + `albench/evaluation_utils/` |

### Key design decisions
- **Oracle and Student merged into `SequenceModel`**: both roles implement the `albench.model.SequenceModel` ABC — the oracle/student split was a belt-and-suspenders distinction that added no real polymorphism. The old `Oracle`/`Student` aliases have been fully removed; all code now uses `SequenceModel` directly.
- **`loop.py` stays in `albench/`** but is refactored into an `ALLoop` megaclass that tracks state across rounds, replacing the imperative `run_al_loop` function.
- **`albench/utils.py`** re-exports the sequence utilities (`one_hot_encode`, `reverse_complement`) so `albench` users never need to know about the `data/` package.
- **`pyproject.toml`** updated to export `data`, `models`, `evaluation` as top-level importable packages.

### Implementation status
- [ ] Phase 1: Move `albench/data/` → `data/`, `albench/models/`+`oracle/`+`student/` → `models/`, evaluation → `evaluation/`
- [ ] Phase 2: Slim `albench/` (delete moved dirs, add shims, update `loop.py` imports)
- [ ] Phase 3: Update all import sites (experiments, scripts, tests, root files)
- [ ] Phase 4: Update `pyproject.toml`
- [ ] Phase 5: Reorganise `tests/` into `tests/test_albench/` and `tests/test_experiments/`

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

### AlphaGenome + Malinois test set results — full comparison (COMPLETED Feb 24)

Eval on chr 7, 13 (N=62,582). All AlphaGenome evals use FW+RC averaging. Malinois eval
updated to also use FW+RC averaging (job 656461; result: Pearson R = 0.8834).

| Model | aug_mode | Pearson R | Spearman R | MSE |
|-------|----------|-----------|------------|-----|
| **boda-flatten** | no_shift | **0.9061** | 0.8245 | 0.2368 |
| **boda-sum** | no_shift | **0.9051** | 0.8205 | 0.2379 |
| **boda-mean** | no_shift | **0.9046** | 0.8216 | 0.2397 |
| boda-mean | hybrid | 0.9047 | 0.8204 | 0.2436 |
| boda-sum | hybrid | 0.9038 | 0.8202 | 0.2466 |
| **boda-max** | no_shift | **0.9027** | 0.8142 | 0.2507 |
| boda-flatten | hybrid | 0.9035 | 0.8200 | 0.2440 |
| boda-max | hybrid | 0.9035 | 0.8148 | 0.2443 |
| boda-center | no_shift | 0.8966 | 0.8090 | 0.2618 |
| boda-center | hybrid | 0.8957 | 0.8031 | 0.2660 |
| **Malinois (RC-ensemble)** | — | **0.8834** | 0.8086 | 0.2958 |
| Malinois (no RC) | — | 0.8688 | 0.7928 | 0.3259 |

**Key findings:**
- AlphaGenome adapter heads outperform Malinois by **~2–2.3% Pearson R** (RC-to-RC comparison).
- `boda-flatten` no_shift is the best single model (0.9061 Pearson R).
- Hybrid augmentation (±15 bp shift) does **not** improve over no_shift on this test set — scores are essentially equal or slightly lower. The no_shift cached-embedding protocol is both faster and matches quality.
- The Malinois RC-ensemble result (0.8834) now matches the published value (~0.88–0.89), confirming the earlier gap was entirely due to missing RC averaging.

Result file: `outputs/ag_chrom_test_results.json` on HPC.

**Bug fixed**: The initial eval (job 655490) used 200bp center-crop + 384bp padding (wrong), giving 0.70x Pearson. Fixed in `eval_ag.py:evaluate_chrom_test` to use the full 600bp sequences.

**N-padding placement**: Ns are added at the **outer ends** of the sequence (before upstream flank / after downstream flank), not between the variable region and flanks. The Addgene flanks are each 300bp; Ns only appear for sequences shorter than ~600bp where the combined flank slices + insert fall short of 600bp.

### Why Malinois scores slightly below the published result (~0.88–0.89)

Our Malinois eval gives Pearson R = 0.869, while the original Gosai et al. (2023) paper reported ~0.88–0.89 on the **same chr 7, 13 test set**. Two factors likely explain the gap:

1. **No reverse-complement averaging in our Malinois eval** (likely primary reason): `eval_malinois_boda2_tutorial.py` evaluates only one orientation (forward strand). Our AlphaGenome eval averages FW + RC predictions. RC averaging is a standard ensembling trick that consistently improves Pearson R by ~0.5–1%. The original paper may have done RC averaging or trained with RC augmentation that implicitly makes the model more strand-symmetric.

2. **Data filtering or preprocessing differences**: Minor differences in how the K562 MPRA dataset is filtered (duplicate handling, quality thresholds, sequence length boundaries) could shift the test-set composition slightly relative to the original paper's internal dataset. We use the boda2 tutorial's `FlankBuilder` and `load_model()` which are the official artifact, so this effect is likely small.

**Bottom line**: The 0.869 vs. 0.88–0.89 gap is most likely due to the absence of RC averaging/ensembling in our Malinois eval. The meaningful comparison is AlphaGenome vs. Malinois **on the same chr 7,13 split** (~0.90 vs. 0.87), which shows a clear ~3–4% advantage for AlphaGenome.

### Hybrid training jobs (Feb 24, jobs 655491–655495, COMPLETED)

| Job | Name | Val Pearson | Test Pearson | Status |
|-----|------|-------------|--------------|--------|
| 655491 | ag_sum_hybrid | — | 0.9038 | ✓ DONE |
| 655492 | ag_mean_hybrid | 0.9233 | 0.9047 | ✓ DONE |
| 655493 | ag_max_hybrid | 0.9229 | 0.9035 | ✓ DONE |
| 655494 | ag_center_hybrid | 0.9217 | 0.8957 | ✓ DONE |
| 655495 | ag_flatten_hybrid | — | 0.9035 | ✓ DONE |

All jobs on `gpuq` partition, 12h limit. Checkpoints at `outputs/ag_*_hybrid/best_model/`.

---

## 3. Remaining Steps

### 3.1 Immediate (Feb 24)

- **Chrom-test eval DONE** — results in `outputs/ag_chrom_test_results.json` on HPC (see table above).
- **Generate comparison report** (can do now):
  ```bash
  # Copy results from HPC:
  scp christen@bamdev4.cshl.edu:/grid/wsbs/home_norepl/christen/ALBench-S2F/outputs/ag_chrom_test_results.json outputs/
  scp christen@bamdev4.cshl.edu:/grid/wsbs/home_norepl/christen/ALBench-S2F/outputs/malinois_eval_boda2_tutorial/result.json outputs/malinois_eval_boda2_tutorial/
  # Generate report:
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

### 3.3 Evaluating hybrid runs once they complete

Follow these steps after jobs 655491–655495 finish (or after any future hybrid job batch).

**Step 1 — Confirm jobs completed**
```bash
# From local machine (on VPN):
ssh -i ~/.ssh/id_ed25519_citra christen@143.48.80.155 \
  "sacct -u christen --jobs=655491,655492,655493,655494,655495 \
    --format=JobID,JobName,State,ExitCode,Elapsed --noheader"
```
All five should show `COMPLETED` (or `TIMEOUT`). If a job timed out, the best checkpoint is still saved — see Step 2.

**Step 2 — Verify checkpoints exist**
```bash
ssh -i ~/.ssh/id_ed25519_citra christen@143.48.80.155 \
  "ls /grid/wsbs/home_norepl/christen/ALBench-S2F/outputs/ag_*_hybrid/best_model/checkpoint 2>/dev/null || echo MISSING"
```
Expected: five lines, one per head (sum/mean/max/center/flatten). If any are missing, check the job log:
```bash
ssh -i ~/.ssh/id_ed25519_citra christen@143.48.80.155 \
  "tail -50 /grid/wsbs/home_norepl/christen/ALBench-S2F/logs/ag_*hybrid*655491*.err"
```

**Step 3 — Push any local changes, then pull on HPC**
```bash
# On local machine:
git push

# On HPC:
ssh -i ~/.ssh/id_ed25519_citra christen@143.48.80.155 \
  "cd /grid/wsbs/home_norepl/christen/ALBench-S2F && git pull"
```
This ensures `eval_ag_chrom_test.py` (now updated to include hybrid CONFIGS) is present on HPC.

**Step 4 — Submit eval job**
```bash
ssh -i ~/.ssh/id_ed25519_citra christen@143.48.80.155 \
  "cd /grid/wsbs/home_norepl/christen/ALBench-S2F && \
   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/eval_ag_chrom_test.sh"
```
The eval script (`scripts/slurm/eval_ag_chrom_test.sh`) runs on `gpuq` with a 2h time limit and writes `outputs/ag_chrom_test_results.json`. It automatically skips any head whose checkpoint is missing.

Monitor progress:
```bash
ssh -i ~/.ssh/id_ed25519_citra christen@143.48.80.155 \
  "squeue -u christen --format='%i %j %T %M' --noheader"
```

**Step 5 — Copy results back**
```bash
scp -i ~/.ssh/id_ed25519_citra \
  christen@143.48.80.155:/grid/wsbs/home_norepl/christen/ALBench-S2F/outputs/ag_chrom_test_results.json \
  outputs/ag_chrom_test_results.json
```

**Step 6 — Generate comparison report**
```bash
uv run python scripts/analysis/compare_malinois_alphagenome_results.py \
  --output outputs/malinois_ag_comparison.md
```
This reads `outputs/ag_chrom_test_results.json` (all heads: no_shift + hybrid) and `outputs/malinois_eval_boda2_tutorial/result.json`, then writes a Markdown table.

**Head names in `eval_ag_chrom_test.py`**: As of the last update, CONFIGS includes:
- no_shift: `boda_sum`, `boda_mean`, `boda_max`, `boda_center`, `boda_flatten`
- hybrid: `boda_sum_hybrid`, `boda_mean_hybrid`, `boda_max_hybrid`, `boda_center_hybrid`, `boda_flatten_hybrid`

Missing checkpoints are automatically skipped with a stderr message.

**If hybrid jobs timed out before completing**: Resubmit with a longer time limit on `kooq`:
```bash
# Edit each hybrid Slurm script locally, then push+pull+sbatch as above.
# Change in each script:
#   #SBATCH --partition=gpuq   →  #SBATCH --partition=kooq
#   #SBATCH --time=12:00:00    →  #SBATCH --time=24:00:00
# Add:
#   #SBATCH --qos=koolab
# Then resume training from the latest checkpoint (set ++resume_from= in the script).
```

### 3.4 Optional

- `encoder-1024-dropout` head: reference-style single hidden layer (1024 units + dropout).

#### 384 bp compact-window training (alternate preprocessing approach)

**Rationale**: Instead of padding each variable-region sequence to 600 bp with real flanks (and N-padding the remainder), build a fixed 384 bp window (3 × 128 tokens — exact AlphaGenome stride alignment) around the variable region using only real Addgene flank sequence. This has several advantages:
- **No N-padding at all**: even the shortest variable region (≥73 bp) + both 200 bp flanks = ≥473 bp > 384 bp, so there is always enough real sequence to fill a 384 bp window.
- **Clean shift augmentation**: shifting ±N bp simply redistributes N bases between the upstream and downstream flank slices — no edge effects or N-sequence contamination.
- **Example**: 73 bp insert → window has (384−73)/2 ≈ 155 bp flank on each side; can shift ±(155) bp and still have real sequence on both sides.

**Existing infrastructure** (already in `albench/data/k562_full.py`):
- `K562FullDataset(store_raw=True)` — stores variable-length raw sequences instead of 600 bp padded.
- `dataset.set_compact_window(min_var_len=73, window_bp=384, flank_bp=200)` — sets W=384.
- `dataset._build_compact_sequence(seq, shift=0)` — builds `left_flank + seq + right_flank` of exactly W bp, with `shift` redistributing flank length.

**What would need to be added** (without touching the current 600 bp pipeline):
1. New `aug_mode="compact_shift"` branch in `experiments/train_oracle_alphagenome_full.py` collate function that calls `_build_compact_sequence(seq, shift=np.random.randint(-max_shift, max_shift+1))` per sample.
2. New Slurm scripts for compact-window runs (e.g., `train_oracle_alphagenome_full_sum_compact.sh`) with `++aug_mode=compact_shift ++output_dir=outputs/ag_sum_compact`.
3. The compact eval in `eval_ag.py`/`eval_ag_chrom_test.py` must also use `store_raw=True` + `_build_compact_sequence(seq, shift=0)` for the test set (or just use `_center_pad(seq, target_len=384)`).

**Token count change**: 384 bp → T=3 tokens (vs. T=5 for 600 bp). This affects the `boda_flatten` head (input dim 3×1536=4608 vs. 7680) — it would need retraining. Sum/mean/max/center heads are unaffected.

**Priority**: Lower — do after hybrid runs are evaluated and results indicate a clear benefit from shift augmentation.

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

---

## 6. Yeast AlphaGenome Run (Feb 25-26, 2026)

### 6.1 Initial Sweep & Flank Discovery
- Initial validation scores for the head sweep plateaued around 0.54 Pearson R.
- Identified that the 100GB sequence cache was built using generic plasmid flanks rather than the specific `pTpT/TEF1` reference flanks, leading to poor model interpretation of the core 80bp sequence.
- The winning architecture was identified as a 1-layer, 512-unit MLP (`flatten_1x512`).

### 6.2 Cache Rebuilding & Bug Fixes
- **Job 685314** (24 hours on H100) was submitted to rebuild the 100GB cache using the exact `pTpT/TEF1` flanking nucleotides and the `flatten_1x512` head.
- **Stage 2 Bug Fix:** Discovered that Stage 2 (unfrozen encoder end-to-end fine-tuning) would crash because `AlphaGenomeModel` didn't implement `.load_checkpoint()`. Fixed `experiments/train_oracle_alphagenome_yeast.py` to manually merge the Orbax `._params` dictionary so Stage 1 head parameters correctly transfer to Stage 2.

### 6.3 Stage 2 Fine-Tuning Preparation
- Investigated the reference `alphagenome_FT_MPRA` repository for K562 Stage 2 best practices:
  - **Batch Size:** 32 (frozen encoder uses 4096, but back-propping through the transformer requires small batches to avoid OOM).
  - **Learning Rate:** 1e-5.
  - **Epochs:** 50.
- Upgraded the python training script to support a separate `second_stage_batch_size` and a `pretrained_head_dir` argument to bypass Stage 1 entirely pointing directly to orbax checkpoints.

Created two ready-to-run Stage 2 SLURM scripts for when Job 685314 finishes:
1. `scripts/slurm/train_ag_yeast_stage2_1x512_k562.sh`: Runs the exact K562 best practices (LR 1e-5, Batch 32, static validation padding).
2. `scripts/slurm/train_ag_yeast_stage2_1x512_full_aug.sh`: Applies stringent `aug_mode="full"` (live sequence padding shifts) and uses a slightly lower `5e-6` LR for maximum generalization.
