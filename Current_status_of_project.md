# ALBench-S2F — Current Project Status

**Last updated:** 2026-03-03 (Tue, EST)  
**Updated by:** Codex (GPT-5)  
**Scope of this update:** Yeast (*S. cerevisiae*) Exp 0 status, active jobs, and what remains

---

## Snapshot

- We are actively executing yeast Exp 0 on both HPC (H100/V100 via SLURM) and Citra (A6000).
- DREAM-RNN yeast scaling already has complete coverage for all 10 fractions with at least 4 completed runs/fraction on HPC.
- DREAM-RNN yeast oracle 10-fold CV training is complete (`outputs/oracle_dream_rnn_yeast_kfold/`), and pseudolabel generation is submitted but waiting for scheduler quota.
- AlphaGenome yeast scaling/oracle arrays are submitted and partially running; no completed `result.json` yet in `outputs/exp0_yeast_scaling_alphagenome/`.
- Additional AG yeast stage-2 finetune mode sweeps are running to evaluate unfreezing strategies.

---

## Experiment 0 Requirements (Yeast) — Status

| Requirement | Status | Notes |
|---|---|---|
| DREAM-RNN scaling: 0.1%, 0.2%, 0.5%, 1%, 2%, 5%, 10%, 20%, 50%, 100% | **In hand** | HPC has 4 completed runs per fraction in `outputs/exp0_yeast_scaling/` |
| AG scaling at same 10 fractions | **Running** | SLURM array `744715` running tasks `0-2`, `3-9` pending (QOS limit) |
| DREAM-RNN oracle ensemble (10-fold CV) on all available train data | **Done** | `outputs/oracle_dream_rnn_yeast_kfold/oracle_{0..9}/summary.json` present |
| Oracle pseudolabel generation for train/val/test | **Queued** | Job `744720` pending (QOS limit) |
| AG oracle ensemble (10-fold CV) | **Queued** | SLURM array `744716` pending (QOS limit) |

---

## Running / Queued Jobs

### HPC (`bamdev4.cshl.edu`) — checked 2026-03-03 10:46 EST

Running now:
- `744711_[2-5]` (`ag_yeast_ft_modes`) — AG finetune mode sweep tasks
- `744715_[0-2]` (`exp0_ag_yeast`) — AG yeast scaling fractions (first 3 tasks)
- `731304_[0-1]` (`ag_yeast_ft_modes`) — earlier yeast finetune sweep tasks still active

Pending now (scheduler/QOS throttling):
- `744716_[0-9]` (`ag_yeast_oracle`) — AG yeast oracle CV array
- `744715_[3-9]` (`exp0_ag_yeast`) — remaining AG scaling tasks
- `744720` (`oracle_pseudolabels_yeast`) — DREAM oracle pseudolabel generation

### Citra (`143.48.59.3`) — checked 2026-03-03

Running now:
- `experiments/exp0_yeast_scaling.py` at fractions `0.5` and `1.0` with `seed=null` (random init/downsampling path)

Completed currently on Citra:
- 8 result files for fractions `0.001` through `0.2` (`outputs/exp0_yeast_scaling/`)

---

## Completed Results Inventory (Yeast)

### DREAM-RNN scaling (`outputs/exp0_yeast_scaling/`)

- Total completed `result.json`: **40** (HPC)
- Coverage: **4 completed runs at each fraction**  
  `0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0`
- Seed makeup currently mixed:
  - Historical fixed seeds (`42, 43, 44`) dominate older runs
  - Additional random-seed runs exist (from `seed=null` launches)

### DREAM-RNN oracle 10-fold CV (`outputs/oracle_dream_rnn_yeast_kfold/`)

- Completed fold summaries: **10 / 10**
- Per-fold files: `oracle_{0..9}/summary.json`
- Aggregate means from existing summaries:
  - `val_pearson_mean ~= 0.6062`
  - `test_random_pearson_mean ~= 0.8001` (ID)
  - `test_genomic_pearson_mean ~= 0.6139` (OOD)
  - `test_snv_abs_pearson_mean ~= 0.8729` (SNV-abs)

### AG yeast artifacts found (historical + current)

Existing yeast AG output directories on HPC:
- `outputs/ag_yeast/` (includes embedding cache)
- `outputs/exp0_yeast_scaling_alphagenome/` (current scaling target; no completed `result.json` yet)
- `outputs/ag_yeast_oracle_finetune/` (head-only + stage2 experiment outputs/checkpoints)
- `outputs/ag_yeast_stage2/`
- `outputs/ag_yeast_sweep/`
- `outputs/oracle_alphagenome_yeast*` (older oracle attempts/checkpoints)

Note: many historical AG directories contain checkpoints/configs but not standardized `result.json`/`summary.json`, so best-head comparison requires log parsing or explicit eval backfill.

---

## Recent Code/Workflow Changes in Flight

- Fixed AG yeast stage-2 training window/collation behavior in  
  `experiments/train_oracle_alphagenome_yeast.py`.
- Added DREAM yeast oracle pseudolabel pipeline:
  - `experiments/generate_oracle_pseudolabels_yeast_dream.py`
  - `configs/experiment/generate_oracle_pseudolabels_yeast_dream.yaml`
  - `scripts/slurm/generate_oracle_pseudolabels_yeast_dream.sh`
- Relaunched yeast AG jobs after dependency blockage/timeouts (`744715`, `744716`, `744720`).

---

## Remaining Work to Close Yeast Exp 0

1. Finish AG scaling array `744715` and collect per-fraction metrics.
2. Run AG oracle CV array `744716` to completion.
3. Run pseudolabel generation job `744720` after scheduler release.
4. Consolidate and deduplicate DREAM scaling points (keep one unique metric tuple per run profile).
5. Backfill standardized AG result summaries for historical yeast AG runs so head/stage2 variants are directly comparable.
6. Produce final yeast scaling plots (DREAM vs AG) with the required 10 fractions and test metrics.

---

## Next 24h Triage

### Done

- DREAM-RNN yeast scaling coverage exists for all 10 fractions with 4 completed runs/fraction on HPC.
- DREAM-RNN yeast 10-fold oracle training completed and fold summaries are available.
- Yeast AG stage-2 finetune fixes and DREAM pseudolabel generation script/config/SLURM were added.

### Running

- HPC: `744711_[2-5]` and `731304_[0-1]` (AG yeast finetune sweeps).
- HPC: `744715_[0-2]` (AG yeast scaling, first tasks).
- Citra: yeast DREAM scaling at fractions `0.5` and `1.0` with `seed=null`.

### Blocked / Queued

- HPC queue throttling (`QOSMaxJobsPerUserLimit`) is delaying:
  - `744715_[3-9]` (remaining AG scaling fractions)
  - `744716_[0-9]` (AG yeast oracle CV array)
  - `744720` (yeast DREAM oracle pseudolabel generation)
- Historical AG yeast runs are not uniformly summarized (`result.json`/`summary.json` missing in many dirs), which blocks clean best-head ranking without backfill.

### Next 24h

1. Watch AG scaling tasks to first completion and verify per-fraction outputs in `outputs/exp0_yeast_scaling_alphagenome/`.
2. As queue frees, confirm AG oracle array launch (`744716`) and monitor first fold logs.
3. Run queued pseudolabel job (`744720`) and validate output files/columns.
4. Backfill AG yeast summary metrics from historical runs to enable apples-to-apples head/stage2 comparison.
5. Refresh consolidated yeast plots/tables once new AG results land.

---

## Latest Developments (2026-03-03, afternoon EST)

- **AG yeast evaluation patch applied (both Citra + HPC repos):**
  - Validation for yeast AlphaGenome now RC-averages forward and reverse-complement predictions in:
    - `experiments/train_oracle_alphagenome_yeast.py`
    - `experiments/exp0_yeast_scaling_alphagenome.py`
  - Defaulted on via config:
    - `configs/experiment/oracle_alphagenome_yeast.yaml`
    - `configs/experiment/oracle_alphagenome_yeast_finetune_sweep.yaml`
    - `configs/experiment/exp0_yeast_scaling_alphagenome.yaml`
  - New key: `eval_use_reverse_complement: true`

- **DREAM-RNN v256 status check (HPC):**
  - `748234` (`oracle_dream_yeast_v256`): tasks `0-3` running, tasks `4-9` pending (`JobArrayTaskLimit`).
  - `748235` (`exp0_yeast_v256` scaling): no longer present in `squeue` (likely completed/finished; verify from outputs/sacct in next check).

- **DREAM-RNN v256 status check (Citra):**
  - Active long runs: oracle folds `0/1/2` plus scaling fractions `1.0` and `0.02`.
  - Completed fraction results currently detected in `outputs/exp0_yeast_scaling_v256_citra/`:
    - `0.001, 0.002, 0.005` (1 completed run each)
  - Fractions `0.01, 0.05, 0.1, 0.2, 0.5, 1.0` still in-progress or incomplete.

- **OOM root cause + mitigation for Citra scaling relaunches:**
  - Prior failures for `0.01/0.05/0.1/0.2/0.5` were CUDA OOM from overlapping launches on already occupied GPUs.
  - Relaunched `0.01/0.05/0.1` on dedicated GPUs; rerun logs show no traceback/OOM at startup and CUDA training active.
  - Added stricter “one-at-a-time” relaunch helpers for remaining `0.2` and `0.5` to avoid concurrent GPU collisions.
