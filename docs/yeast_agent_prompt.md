# Yeast Experiments — Agent Kickoff Prompt

> Copy-paste this prompt to start a new Claude Code session focused on running the yeast
> experiments in ALBench-S2F. The human K562 experiments are being handled separately.

---

## Prompt (copy everything below this line)

---

You are continuing the ALBench-S2F project. Your job is to get the **yeast S. cerevisiae**
experiments running on the CSHL HPC cluster. A separate Claude instance is handling the human
K562 experiments; your focus is exclusively on yeast.

## What the project is

ALBench-S2F (repo: `~/Downloads/ALBench-S2F` on the Mac, `/grid/wsbs/home_norepl/christen/ALBench-S2F` on HPC) benchmarks sequence-to-function models for active learning on MPRA datasets.

The central question is: **how does model performance scale with training-set size?** We compare:
- **DREAM-RNN** — CNN-RNN trained from scratch (baseline)
- **AlphaGenome (AG)** — frozen pre-trained encoder + fine-tuned small head (expected to be far more data-efficient)

For K562 human data, AlphaGenome reaches DREAM-RNN's full-data ceiling (Pearson 0.82) with only ~2% of the data. We expect a similar advantage for yeast. Your job is to reproduce this experiment for yeast.

## Key reference files to read first

Read these files to orient yourself before doing anything:

1. `Current_status_of_project.md` — full results table, current job status, important technical details
2. `REMOTE_ACCESS.md` — SSH credentials, access instructions, HPC/Citra specifics
3. `~/Downloads/cshl_hpc_cheatsheet.md` — HPC partition/QoS reference
4. `~/Downloads/ALBench_plan_v2.md` — full experiment plan and registered-report structure

## What has already been done for yeast

- **DREAM-RNN yeast scaling**: 1–4 seeds per fraction (fractions 0.001–1.0), val Pearson only
  (no test metrics). Incomplete but results exist in `outputs/exp0_yeast_scaling/` and
  `outputs/analysis/synced/exp0_yeast_scaling/` on HPC.
  Val Pearson at full data: ~0.62 (much lower ceiling than K562's 0.82 with DREAM-RNN).
- **Yeast oracle script exists**: `experiments/train_oracle_alphagenome_yeast.py` — supports
  full/no_shift/hybrid aug modes, 2-stage training (frozen head then full fine-tune).
- **Yeast DREAM-RNN scaling script**: `experiments/exp0_yeast_scaling.py` (DREAM-RNN);
  SLURM: `scripts/slurm/exp0_yeast_scaling.sh` (old — uses `uv run` not `uv run --no-sync`)

## What does NOT yet exist for yeast

1. **Yeast embedding cache** — needs to be built (saves encoder outputs for all sequences)
2. **Yeast AG oracle** (10 seeds) — not started
3. **Yeast AG scaling experiment** — `exp0_yeast_scaling_alphagenome.py` does not exist
4. **Complete DREAM-RNN scaling** — needs 3 seeds per fraction with test metrics

## Yeast-specific technical details

### Data
- Dataset: `data/yeast/` on HPC (Seq2Fun yeast MPRA)
- ~960K training+pool sequences, 150bp cores padded to 384bp
- Splits: train/pool/val/test (hashFrag-based)
- Dataset class: `data.yeast.YeastDataset`
- Read `data/yeast.py` to understand the data format

### AlphaGenome configuration for yeast
- **Input length**: 384bp (54bp 5' plasmid flank + 150bp core + right padding)
- **Encoder tokens**: T=3 (384 / 128 = 3 tokens)
- **Objective**: 18-bin cross-entropy (KL divergence) on discretised expression levels
- **Metric**: Pearson r between predicted expected bin and ground-truth expression
- The plasmid flanks are defined in `experiments/train_oracle_alphagenome_yeast.py` —
  search for `_FLANK_5` and `_FLANK_3` near the top of the file
- `register_s2f_head(..., task_mode="yeast")` (not "human") — important!

### Cache size (yeast)
~8.4 GB per canonical + RC pair (960K seqs × T=3 × D=1536 × float16). Fits in 200G RAM/VRAM.

### Two-stage option
The yeast oracle script supports optional stage 2 (full encoder unfreeze):
- `second_stage_lr: 1e-5`, `second_stage_epochs: 50`
- Set `second_stage_lr: null` to skip (recommended for initial experiments)

## What to do, in order

### Step 1: Check current state of HPC and existing yeast results
- SSH to HPC and list existing outputs in `outputs/ag_yeast_oracle*/`, `outputs/exp0_yeast_scaling*/`
- Check if any jobs are currently running for yeast
- Read the yeast oracle script to understand the config

### Step 2: Build the yeast embedding cache
The oracle script (`train_oracle_alphagenome_yeast.py`) with `aug_mode=no_shift` will
auto-build the cache if it doesn't exist. BUT it's better to build it once in a dedicated job
first (so all oracle seeds can share it).

Analog to the K562 cache builder is `scripts/slurm/build_hashfrag_embedding_cache.sh`.
You'll need to create a yeast equivalent:
- Script: `scripts/slurm/build_yeast_embedding_cache.sh`
- Python: `scripts/analysis/build_yeast_embedding_cache.py`
- Cache dir: `outputs/ag_yeast/embedding_cache/`
- Splits to cache: train, pool, val (same pattern as K562)
- Yeast sequences are 384bp (T=3 encoder tokens), not 600bp (T=5)
- See `scripts/analysis/build_hashfrag_embedding_cache.py` as the K562 template

SLURM spec: `gpuq/slow_nice`, H100, 3h walltime, 14 CPUs, 200G RAM.

### Step 3: Train yeast AG oracle (10 seeds, cached approach)
The K562 analog is `experiments/train_oracle_alphagenome_hashfrag_cached.py`.
You'll need a yeast equivalent that:
- Loads the yeast embedding cache (train + val splits)
- Uses `build_head_only_train_fn` + canonical/RC passes per epoch
- Evaluates on yeast test sets after training
- Outputs to `outputs/ag_yeast_oracle_cached/oracle_{0-9}/`

OR you can use `train_oracle_alphagenome_yeast.py` with `aug_mode=no_shift` directly
(it already supports cached training) — create a config YAML and SLURM array job.

Config to use: `configs/experiment/oracle_alphagenome_yeast.yaml` (may need to be created).
SLURM: `scripts/slurm/train_oracle_alphagenome_yeast_cached_array.sh`

### Step 4: Write yeast AG scaling experiment
Create `experiments/exp0_yeast_scaling_alphagenome.py` modelled on `exp0_k562_scaling_alphagenome.py`.
Key differences vs K562:
- Use `YeastDataset` instead of `K562Dataset`
- 384bp context (yeast plasmid flanks, not MPRA flanks)
- `task_mode="yeast"` for head registration
- T=3 encoder tokens (for `reinit_head_params(num_tokens=3, dim=1536)`)
- Fractions of the yeast train+pool combined (~960K sequences)
- Fractions: [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00] (10 fractions, same as DREAM-RNN)
- Yeast embedding cache at `outputs/ag_yeast/embedding_cache/`
- Config: `configs/experiment/exp0_yeast_scaling_alphagenome.yaml`
- SLURM: `scripts/slurm/exp0_yeast_scaling_alphagenome.sh` (array 0–9 for 10 fractions)

### Step 5: Run everything on HPC
1. Commit all new scripts and push to GitHub
2. SSH to HPC, `git pull`
3. Submit cache-build job first (prerequisites for oracle + scaling)
4. After cache completes, submit oracle array + scaling array
5. Monitor progress

### Step 6: Complete yeast DREAM-RNN scaling
The existing runs lack test metrics. Check `experiments/exp0_yeast_scaling.py` to understand
why test metrics are -1 (likely the test set paths aren't being found or the evaluation
function isn't called). Fix and rerun with 3 seeds per fraction.

The SLURM script `scripts/slurm/exp0_yeast_scaling.sh` uses `uv run` not `uv run --no-sync`
— update it to use `uv run --no-sync`.

## Code style / environment notes
- Pre-commit hooks: `ruff` (format + import sort). Run hooks will auto-fix; re-stage and recommit.
- All SLURM scripts: use `uv run --no-sync python` (not bare `uv run python`)
- XLA flags for H100: `export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"`
- `source scripts/slurm/setup_hpc_deps.sh` at start of every SLURM job (CHECK-ONLY)
- Never run `uv pip install` from concurrent SLURM jobs (NFS flock = venv corruption)
- Head names must include `_v4` suffix to avoid stale checkpoint parameter matching
- After `create_model_with_heads()`, always call `reinit_head_params()` before `freeze_except_head()`

## Infrastructure reminder
- HPC: `ssh -i ~/.ssh/id_ed25519_citra christen@bamdev4.cshl.edu`
- Citra (PyTorch only): `ssh -i ~/.ssh/id_ed25519_citra trevor@143.48.59.3`
- sbatch: `/cm/shared/apps/slurm/current/bin/sbatch`
- AlphaGenome weights: `/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1`
- For long jobs: `--partition=gpuq --qos=slow_nice --time=48:00:00`
- For eval/debug: `--partition=gpuq --qos=fast --time=4:00:00`
- **No koolab access** — don't request `--qos=koolab`
- VPN required when off-campus

## Expected yeast results
Based on K562 data and the AlphaGenome paper, we expect:
- AG at ~1% of yeast data should already outperform DREAM-RNN at full data (0.62 val Pearson)
- AG oracle at full yeast data: likely >0.80 Pearson (DREAM-RNN only reaches 0.62)
- Yeast is harder than K562 (smaller effect sizes, noisy labels) but the relative advantage of AG should still be large

If the yeast oracle achieves <0.70, consider whether the architecture/hyperparameters need tuning (check `experiments/train_oracle_alphagenome_yeast.py` for available options).
