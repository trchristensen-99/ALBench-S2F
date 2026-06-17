# ALBench-S2F — Onboarding & Orientation

> **Read this first.** It explains what the project does, how the repo is laid
> out, and how to run the current pipeline. For deep design of the underlying
> active-learning engine (the `SequenceModel` / `ReservoirSampler` /
> `AcquisitionFunction` abstractions) see [`ARCHITECTURE.md`](ARCHITECTURE.md).
> For the active methodology plan see [`docs/phase1_bakeoff_plan.md`](docs/phase1_bakeoff_plan.md).

## 1. What this project is

ALBench-S2F (**A**ctive-**L**earning **Bench**mark, **S**equence-**to**-**F**unction)
studies how to train DNA sequence-to-function models efficiently under a limited
labeling budget. The core experimental loop:

1. A **frozen AlphaGenome-derived oracle** assigns in-silico activity labels to
   DNA sequences (a cheap stand-in for an expensive real MPRA assay).
2. **Student models** (LegNet) are trained on oracle-labeled sequences. Their
   hyperparameters/architecture are chosen by an **HP-search strategy**.
3. Student predictions are **ensembled** and scored against the oracle
   (`ensemble-oracle-r`, Pearson r vs the oracle on held-out sequences).

The current research question is **which HP-search strategy is most
compute-efficient**, measured on cumulative GPU-seconds — see §4.

## 2. Repo map (what lives where)

| Path | What it is |
|---|---|
| `albench/` | Core AL engine package: `model.py` (SequenceModel ABC), `loop.py` (AL driver), `reservoir/` (candidate generators), `acquisition/` (selection fns). Model-agnostic; see ARCHITECTURE.md. |
| `models/` | Concrete models: LegNet student, AlphaGenome oracle wrappers. |
| `data/` | Dataset loaders + `TaskConfig` (K562 lentiMPRA, yeast). Cached test/oracle batteries with `PROVENANCE.json`. |
| `experiments/` | **Main entry points.** `scaling_hp_search.py` (HP-search driver) and `hp_strategies.py` (strategy registry) are the two files you'll touch most. |
| `scripts/` | SLURM submitters (`submit_step1_bakeoff.py`, …), data prep, `install_hpc_packages.sh`. |
| `scripts/analysis/` | Post-hoc analysis: efficiency curves, ElasticNet ensemble/recipe search, persona-combo ranking. |
| `configs/` | Hydra YAML for the older Exp0–5 framework. |
| `docs/` | Methodology plans + prompts (see §6). |
| `outputs/` | Run artifacts (training metas, reservoir caches, oracle scores). **Not** source. |
| `results/`, `evaluation/`, `figures_schematics/` | Aggregated results, eval scripts, schematic figures. |
| `tests/` | Unit/integration tests. |
| `*.md` at root | Status logs + this guide (see §6). |

## 3. The oracle

- The oracle is a frozen AlphaGenome encoder + a trained probing head; it labels
  sequences so we never need real assays during the benchmark.
- **Canonical oracle = `full856k_clean`.** All labels used for decisions must be
  stamped with this version. Caches/batteries carry a `PROVENANCE.json` recording
  `oracle_id` + test-set version; the pipeline hard-fails on unstamped caches.
- AlphaGenome weights on HPC:
  `/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1`.

## 4. HP-search strategies

Registered in `experiments/hp_strategies.py` (`STRATEGY_REGISTRY`);
`get_strategy(name, seed)` is the dispatch. Each strategy proposes student HP
configs; `scaling_hp_search.py` trains them **sequentially on one GPU** and feeds
results back.

| Family | Strategies | One-liner |
|---|---|---|
| baseline | `random` | uniform random sampling (the no-model floor) |
| Optuna BO | `optuna_tpe`, `optuna_parallel`, `optuna_cmaes`, `optuna_gp`, `optuna_qmc` | TPE (single/batch), CMA-ES (continuous dims), GP/BoTorch (sample-efficient), scrambled-Sobol QMC (space-filling floor) |
| evolutionary | `evo_single`, `evo_batch`, `evo_explore`, `evo_exploit`, `evo_massive`, `evo_adaptive`, `evo_knowledgeable` | perturb top history; differ in batch size, explore/exploit temperature, per-dim adaptivity, KB-bias |
| multi-fidelity | `ray_asha`, `ray_bohb` | async successive halving / Bayesian-Hyperband (own trial loop; special-cased, needs `ray[tune]`) |
| LLM AutoResearch | `llm_autoresearch` | Claude proposes configs from run history + a domain knowledge-base + a persona prompt; never falls back to random |

Deprecated `autoresearch_*` names alias to `evo_*` via `canonical_strategy()`.

## 5. The workflow (Phase 0 → 1 → 2)

This is a **linear** procedure; the frozen output of each phase feeds the next.

- **Phase 0 — LLM prompt screen.** Pick the best LLM personas/prompt styles
  before they enter the bake-off. (`persona_combos.py` ranks persona subsets by
  ensemble oracle-r at a fixed combo size.)
- **Phase 1 — strategy bake-off (consolidated).** One deep run per
  (reservoir × D × strategy × seed). The **knee** of each strategy's
  oracle-r-vs-GPU-seconds curve gives optimal compute; the **height/ranking** of
  those curves picks the strategy. Design = **anchor + OFAT + corner**
  (perturb one axis at a time off a single anchor cell, never random combos —
  random combos confound the axes). See `docs/phase1_bakeoff_plan.md`.
- **Phase 2 — deploy.** Reuse the frozen `{strategies, rounds-or-GPU-seconds}`
  recipe at every dataset size D.

### Budget axis
Primary x-axis is **cumulative GPU-seconds** (absorbs the "expensive strategy
proposes slow models" confound automatically). Model-count is a secondary lens.
All strategies train the same number of models (200/cell); they differ in
GPU-seconds, which is exactly what we measure.

## 6. Existing docs (don't duplicate these)

| File | Purpose |
|---|---|
| `README.md` | Quick-start: install + run. |
| `ARCHITECTURE.md` | Deep design of the AL engine + the Exp0–5 framework. |
| `docs/phase1_bakeoff_plan.md` | **Active** methodology plan for the HP-strategy bake-off. |
| `docs/yeast_agent_prompt.md` | LLM agent system prompt for HP optimization. |
| `docs/presentation_summary_malinois_vs_alphagenome.md` | Oracle comparison writeup. |
| `EXPERIMENT_TRACKER.md`, `Current_status_of_Experiment_{0,1}.md` | Progress/status logs. |
| `REMOTE_ACCESS.md` | Lab-internal HPC access notes (not git-tracked). |

## 7. Running on HPC (quick reference)

- Repo root on HPC: `/grid/wsbs/home_norepl/christen/ALBench-S2F`.
- One-time venv setup from the **login node only**:
  `scripts/install_hpc_packages.sh` (uv sync + GPU JAX + AlphaGenome + ray, etc.).
- SLURM jobs use `uv run --no-sync python …` (never `uv pip install` from a job —
  it corrupts the shared `.venv` over NFS). `sbatch` lives at
  `/cm/shared/apps/slurm/current/bin`.
- GPUs: partition `gpuq`, QoS tiers `fast` (4h) / `default` (12h) / `slow_nice`
  (2-day, preemptible, checkpoint-resumable).
- **Launch the Phase-1 anchor bake-off:**
  ```bash
  STEP1_RESERVOIRS=genomic STEP1_DS=30000 \
  STEP1_STRATS="random,optuna_tpe,optuna_cmaes,optuna_gp,optuna_qmc,\
evo_single,evo_batch,evo_explore,evo_exploit,evo_massive,evo_adaptive,evo_knowledgeable" \
  STEP1_SEEDS="42:0,43:1,44:2" \
  uv run --no-sync python scripts/submit_step1_bakeoff.py
  ```
  One SLURM job per (strategy × seed); resumes from on-disk history on requeue.
- **Analyze:** `scripts/analysis/plot_hp_rounds_and_ensemble.py` (efficiency
  curves + Kneedle knee), then
  `scripts/analysis/strategy_combination_ablation.py --all_subsets` (frozen recipe).
