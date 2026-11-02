# ALBench-S2F: codebase outline (as of 2026-11-02, read-only audit)

Scope: the local checkout (706 tracked files at the time of the audit) and the cluster checkout `bamdev4:/grid/wsbs/home_norepl/christen/ALBench-S2F` (HEAD `9841f8b` of 2026-10-27, **82 commits behind**, 4 tracked files modified, 547 untracked entries, including the directory `scratchpad/` that holds the code that actually runs; a verbatim snapshot of the live files is now in `cluster_live/`).
Written on 2026-11-02 from a read-only audit: counts come from `git ls-files`, `git log -1 --format=%cs -- <path>`, an AST import graph of all tracked `.py` files, `grep`/`git grep`, and read-only `ssh` listings. File-level statements are true as of that date; `docs/CLEANUP_PLAN.md` says what to change, `docs/AUDIT_NOV2.md` lists the bugs found.

---------------------------------------------------------------------------------------------------

## 1. The current setup in one page

| item | current value | where it is defined |
|---|---|---|
| question | which *training corpus* (reservoir strategy x acquisition) to put on a ~3.2M-oligo MPRA library; deliverable is a recommendation, not a model | `AGENTS.md` |
| main data setting | K562 genomic pools at 10k / 30k / 100k / 300k, canonical nested (seed 42) and independent replicates r1-r3 | `outputs/grid_pools/k562_nested/`, `.../k562_reps/` (cluster, untracked) |
| labels | `oracle_v3`: 10 chromosome-fold AlphaGenome-MPRA models (held-out r 0.931), ~300 seq/s on one H100 | `albench.paths` asset `oracle`; code in `experiments/exp1_1_scaling.py::_load_oracle` |
| validation | `V0c` = 29,449 real chr19/21/X genomic sequences, oracle_v3 labels; drives early stopping, stack weights and loop feedback; never train on it | asset `val_v0c` |
| test sets | `genomic` (chr7/chr13, 31,435), `ood` (designed, 22,962), `random_32k`; 18-19 sets in `scaling_hp_search.load_all_test_sets()` | asset `k562_test_sets`; `data/k562/test_sets_ag_s2_chrsplit` |
| student input standard | 200-bp insert + 20 real plasmid bases each side = 240 bp; training window shifted +-10 over real flank; RC-averaged prediction | `albench/arch_search/harness_standard.py` (`FLANK_CTX=20`, `SHIFT_MAX=10`, `INPUT_LEN=240`) |
| reference student | published MPRA-LegNet with a per-size tuned recipe: 10k e800 wd0.4, 30k e500 wd0.2, 100k e500; 20 seeds per size; stack genomic r 0.780 / 0.891 / 0.938 | `docs/TUNED_LEGNET.md`; `scripts/program_oct22/bench33_legnet.py` |
| frozen menu | menu v1c = 20 model-written configurations (PyTorch source + recipe), all trained per size, one positive-ElasticNetCV stack on z-scored V0c predictions; best only at 10k (+ combination) | `albench/ensemble/menu.py`, `albench/ensemble/menus/*.json`, `docs/MENU.md` |
| LLM agent loop (AutoResearch) | propose -> coordinate -> code -> evaluate (-> reflect), optional manager (`schedule_ens`, `schedule_fb`, `variants`); Claude Sonnet 5.5 via the CLI token path; 15 rounds x 2 models; does *not* beat the tuned LegNet bank at 30k | `albench/arch_search/agent_loop.py` (library) + the launcher `scratchpad/arch_autoresearch.py` (cluster only) |
| baselines | AlphaGenome-FT students are baselines only, never ensemble members | `AGENTS.md`, `scripts/program_oct22/bench33_ag.py` |
| scheduler | `scratchpad/dispatcher.py` (cluster only), run by cron every 10 min in a tmux session, reads `scratchpad/dispatch_registry.json` (2,881 cells, 40 unfinished) and spreads cells over Slurm `fast` / `default` / `slow_nice` / MIG | cluster |

---------------------------------------------------------------------------------------------------

## 2. Flow of a run: pool -> labels -> student -> stack -> evaluation

```
 registry strategy ──generate──> candidate sequences ──oracle_v3──> labelled raw pool (npz)
        │                                                               │ slice / permute (seed)
        │                                                               v
        │                               nested pools: k562_nested/{strategy}_d{10k,30k,100k,300k}.npz  (seed 42)
        │                                             k562_reps/{strategy}_r{1,2,3}_d{D}.npz          (independent draws)
        v                                                               │ every size is a prefix of the next
 (acquisition: select from a candidate pool using a trained ensemble)   v
                                                          train students on the pool (V0c for early stopping)
                                  ┌───────────────────────┬───────────────────────┬──────────────────────────┐
                                  │ tuned MPRA-LegNet bank │ frozen menu v1c       │ LLM loop (AutoResearch)   │  (+ AlphaGenome-FT baselines)
                                  │ bench33_legnet.py      │ menu_retrain.py /     │ arch_autoresearch.py      │
                                  │ --cfg e500_wd2 ...     │ `albench menu train`  │ --loop agent ...          │
                                  └───────────┬───────────┴───────────┬───────────┴──────────────┬───────────┘
                                              v same on-disk layout: labels.npz + r{seed}_menu_{idx}.npz / _meta.json (val_pred, test_pred_<set>)
                          stack: albench/ensemble/stack.py::fit_zscored_stack (positive ElasticNetCV, z-scored V0c predictions)
                                              v
                          evaluate: Pearson r on genomic / ood / random_32k (+ 15-16 other sets, + measured Gosai labels)
                                              v
                          scripts/analysis/analyze_*.py -> outputs/justification_oct22/data_*.json -> figures / decks / Evidence Ledger
```

Step by step, with the command a person would run:

1. **Generate candidates.** `albench list` shows 17 reservoir strategies and 12 acquisition methods. `albench generate --strategy random --n 1000 --out x.npz` needs no data (verified, runs on a laptop). Heavy strategies need assets (`albench doctor`).
2. **Build + label a pool.** `python scripts/build_pool.py --strategy S --size N --generation-seed K --set k=v --out-dir DIR` generates and labels in one go (calls `experiments.exp1_1_scaling._load_oracle`, reads `AG_S2_ORACLE_DIR`, stamps `oracle_id`). `scripts/label_pool.py` labels an existing file; `scripts/label_pool_multicell.py` adds HepG2 and K562-HepG2 labels.
3. **Nest.** `scripts/program_oct22/slice_nested_new.py`, `slice_pool.py`, `scripts/slice_replicate_pools.py`, `scripts/build_nested_pools.py`, helper functions in `stage2_lib.py` (`slice_unseen`, `make_genomic_reps`). `albench.core.pools.nested_subsets(n, sizes, seed)` = one seeded PCG64 permutation, sorted prefixes. `HP_REQUIRE_NESTED=1` makes the loaders refuse the legacy `k562_cumulative`.
4. **Train students** (all write the same layout so one scorer reads every arm):
   - tuned LegNet bank: `python scripts/program_oct22/bench33_legnet.py --D 30000 --cfg e500_wd2 --seeds 0=<seed>,1=<seed>,... --ctx 20` (loads `external/human_legnet/model.py`, which is **gitignored**; output `outputs/menu_retrain_v0/bench38_ln_<cfg>_<D>k_c20/`);
   - menu: library `albench menu train --size 30000 --pool pool.npz --out runs/x --shard 0/4` then `albench menu stack --out runs/x`; on the cluster the cells run `scratchpad/menu_retrain.py --menu M.json --reservoir_cache POOL --D D --shard i/n --val_cache V0c --flank_ctx 20 --shift_max 10` (which has 8 more options than the tracked copy, see section 8);
   - loop: `scratchpad/arch_autoresearch.py --reservoir_cache POOL --D D --out_dir DIR --rounds 15 --per_round 2 --guidance known_good --ensemble_aware --loop agent --agent_cfg '{...}' --flank_ctx 20 --shift_max 10 --reference_legnet --val_cache V0c --resume` with `LLM_MODEL=claude-sonnet-5-5` in the environment (token from `~/.config/albench/llm.env`);
   - AlphaGenome-FT baseline: `scripts/program_oct22/bench33_ag.py` (stage 1 head then stage 2 full fine-tune).
5. **Stack.** `fit_zscored_stack` (positive ElasticNetCV; members with weight zero unused). Guards: V0c only, oracle_v3 stamp checked (`experiments/test_set_guards.py::CANONICAL_ORACLE_ID`), exploded-prediction screen, Muon-missing refusal.
6. **Evaluate.** `albench menu evaluate --ensemble DIR` (3 headline sets) or the analysis scripts (`analyze_legnet_recipes.py`, `analyze_bench33.py`, `analyze_combined_ensembles.py`, `compare_legnet_vs_loop.py`, ...) which re-fit stacks on V0c and compare arms paired by pool/seed; measured (experimental) labels come from `data/k562/DATA-Table_S2__MPRA_dataset.txt`.
7. **Schedule.** `python scripts/program_oct22/register_<x>.py [--dry]` appends cells to `scratchpad/dispatch_registry.json` under `flock`; the dispatcher turns each cell into an sbatch file in `scratchpad/dispatch_jobs/` and resubmits it until its `done` marker (a `COMPLETE` file or logged keys) exists.

---------------------------------------------------------------------------------------------------

## 3. Directory outline

Legend: **LIVE** = needed by the current setup; **REPRO** = reproduces older numbers; **DEAD** = nothing references it (details and verification in `cleanup_plan.md`).

### 3.0 Repository root (tracked)

| file | what | state |
|---|---|---|
| `AGENTS.md` | orientation for coding agents and newcomers; the only up-to-date front door | LIVE, 3 stale statements (see plan) |
| `README.md` | quick start; still the Hydra Exp0-5 framework | STALE |
| `ARCHITECTURE.md` | Sep-17 design guide for the Exp0-5 benchmark, DREAM-RNN student, Citra hosts | STALE |
| `ONBOARDING.md` | Oct-13 guide to the HP-search bake-off (phases 0-2) | STALE |
| `pyproject.toml`, `uv.lock` | package `albench-s2f`, console script `albench = albench.cli:main`; 19 dependencies of which 4 are imported nowhere (hydra-core, python-dotenv, biopython, imageio) and omegaconf only by stale tests | LIVE (needs trimming) |
| `requirements/{menu,oracle,lock_full}.txt` | the pinned two-level environment recipe the study ran with | LIVE |
| `.pre-commit-config.yaml`, `pyrightconfig.json` | ruff v0.11.0 hook; pyright config naming 3 files that do not exist | LIVE / stale entries |
| `.gitignore` | ignores `outputs/ data/k562 data/yeast results/ external/ boda2-main/ wandb/ attic/ .claude/ REMOTE_ACCESS.md` and 12 derived-data dirs | LIVE |

Untracked/ignored on the laptop: `outputs/` (1.8 GB), `data/` (5.9 GB of data next to code), `external/` (366 MB: LegNet, borzoi-pytorch, nucleotide-transformer), `results/` (134 MB), `attic/` (11 MB, the Sep-15 removal), `boda2-main/` (0.9 MB), `wandb/`, `.claude/`, `.env` (holds a `WANDB_API_KEY`; not tracked), `REMOTE_ACCESS.md` (private host/user notes, Mar 18, gitignored).

### 3.1 `albench/` (87 tracked files, 16.2k LOC) - the package, LIVE

| path | what it does / how the current setup uses it |
|---|---|
| `cli.py` | the `albench` command: `doctor`, `list [--kind]`, `generate`, `sweep`, `screen`, `menu {list,export,train,stack,predict,evaluate}` |
| `core/paths.py` | `ASSETS` table (13 assets, each with env var, default relpath, `how_to_get`, optional site fallbacks) and `resolve(key)`; `REPO_ROOT` found by walking up to `pyproject.toml` |
| `core/registry.py` | the strategy table: `register(Spec(name, factory, adapter, params, assets, group))`; 17 reservoirs + 12 acquisition methods (BADGE/BatchBALD with their control variants) |
| `core/pools.py` | `PoolSpec` (hash-named pool files), `write_pool`, `nested_subsets`, `load_pool_subset` |
| `core/run.py`, `core/loop.py` | `build_run_config`, `ALLoop` / `run_al_loop` (the classic reservoir -> acquisition -> oracle -> student loop; used by `albench screen` and by the library `MenuEnsemble` as a `SequenceModel`) |
| `core/protocols.py` | five Protocols for a future "domain" split (documentation only) |
| `paths.py`, `pools.py`, `registry.py`, `run.py`, `loop.py` (top level) | 16-line **shims** that alias the `core/` modules (`sys.modules[__name__] = module`); `AGENTS.md` still tells people to edit `albench/registry.py` |
| `reservoir/` (27 files) | 17 registered samplers (random, genomic, dinuc_shuffle, gc_matched, gc_random, mutagenesis=`partial_mutagenesis`, evoaug*, motif_planted_v2 (motif_* arms and the Zoonomia phylogenetic sampler), zoonomia_orthologs, encode_accessibility (descoped), ise) + 12 modules that are **not registered** (activity_stratified, adaptive_selection, curriculum, diversity_guided, motif_clustering, motif_density, motif_grammar, recombination, uncertainty_guided, tf_motif_shuffle, fixed_pool, mixed_pool) |
| `acquisition/` (13) | uncertainty variants, `badge.py` (expected-gradient-outer-product version + `epistemic_only` / `kmeanspp_only` controls), `batchbald.py` (records `n_informative_`), `binned.py`, `diversity.py`; `ensemble_acq.py` (works on `MenuEnsemble`); unregistered: `bait`, `combined`, `prior_knowledge` |
| `ensemble/` | `menu.py` (1,245 lines: `Menu`, `train_member`, `train_menu`, `MenuEnsemble` (a `SequenceModel`), `stack_dir`, `load_labelled`, `evaluate`), `stack.py` (the single stacking rule, imported by 34 analysis scripts), `menus/{menu_v0,v1,v1c}.json` + `per_size*.json` (the data: 30 / 20 / 20 configurations and K per size) |
| `arch_search/` | the AutoResearch machinery: `agent_loop.py` (multi-agent roles and managers, 1.7k lines), `arch_autoresearch.py` (**repo copy of the launcher; stale, see section 8**), `propose.py` (prompt + CLI call), `validate.py` (AST whitelist, <=40M params, CPU forward check), `train_generated.py`, `flank_context.py` + `plasmid_flank.py` + `harness_standard.py` (the input standard), `multitask.py` (aux target), `mpra_legnet_ref.py` (published MPRA-LegNet source shown to the proposer), `val_cache.py` (V0c loader with oracle stamp), `bridge.py` + `gh_bridge.py` (obsolete laptop/GitHub LLM transport) |
| `motifs/vocabulary.py` | JASPAR/CIS-BP motif vocabulary used by `motif_planted_v2` |
| `composition.py`, `evaluation.py`, `task.py`, `model.py`, `utils.py` | GC/CpG split analysis, scaling-curve helper, `TaskConfig`, `SequenceModel` ABC, one-hot utilities |
| `README.md`, `pyproject.toml` | describe a standalone `pip install albench` distribution (placeholder `yourusername` URL) - not used |

### 3.2 `scripts/` (328 tracked files, 56k LOC)

| subdir | files | purpose / current role | state |
|---|---|---|---|
| `scripts/` (root) | 34 | pool builders and labellers (`build_pool.py`, `label_pool*.py`, `slice_replicate_pools.py`, `build_nested_pools.py`, `generate_labeled_pools.py`, `verify_pools.py`), yeast oracle prep, `make_shared_release.py`, `setup_env.sh`, `ar2_*` | `build_pool.py` + `label_pool.py` + `setup_env.sh` + `make_shared_release.py` LIVE; yeast/FM/curves/CIS-BP builders REPRO |
| `scripts/program_oct22/` | 92 (90 tracked) | the **experiment program of Oct 22 - Nov 2**: 51 `register_*.py` (cell registrations), 18 `patch_*.py` (in-place edits of the cluster launcher and menu_retrain), runtime scripts called by cells (`bench33_legnet.py`, `bench33_ag.py`, `bench33_common.py`, `menu_retrain.py`, `scale_configs.py`, `label_nested_pool.py`, `slice_*.py`, `acq_*.py`, `build_hepg2_pools.py`, `publish_menu_ensembles.py`), ops (`stage2.py`, `stage2_lib.py`, `watchdog.py`, `status_oct26.py`, `refresh_analyses.sh`) | LIVE (name is misleading: it holds Nov 2 work) |
| `scripts/analysis/` | 177 (173 `.py`, 4 `.bak*`) | one-off analyses, figure and deck builders: 72 belong to the current study (cited by docs, touched since Oct 28, or imported/mentioned by such a script), 7 form the Evidence-Ledger builder, 2 are called by Sep pipelines, 92 are Sep to Oct 26 analyses that nothing references (GC/CpG, Sahu, Agarwal, ISM, HepG2, hp_* protocol study, scaling_v2); full lists in `cleanup_plan.md` appendices | 72 + 7 LIVE-REPRO, 92 REPRO-low-value, 4 `.bak` DEAD |
| `scripts/val_variants/` | 13 | validation-set variant study (V1-V3 builders, kmer similarity guard); `build_val_composition.py` feeds the gated `vc39_` cells | REPRO (1 file LIVE) |
| `scripts/pipelines/` | 11 | Sep-era shell pipelines (`human_screen.sh` 210-cell screen, `yeast_oracle.sh`, curves, `scaling_canonical_oct13_*`) built on `scripts/cluster/submit.sh`; the current work does not use them; `yeast_oracle.sh` calls a missing `experiments/train_yeast_oracle_v2.py` | REPRO |
| `scripts/cluster/` | 2 | `submit.sh` (portable sbatch wrapper with `--tiered`), `site.env.example` | REPRO (superseded by the dispatcher) |
| `scripts/slurm/` | 1 | `relabel_battery_oraclev3.sbatch` | REPRO |

### 3.3 `experiments/` (12 files, 9.8k LOC) - legacy drivers that are still imported

| file | role now |
|---|---|
| `exp1_1_scaling.py` (3,508 lines) | the old scaling driver, but it holds `_load_oracle` / `_load_k562_ag_s2_oracle` (the oracle loader) used by `build_pool.py`, `label_pool.py`, `label_nested_pool.py` and 14 analysis scripts: **LIVE for labelling**, the rest is legacy |
| `scaling_hp_search.py` | data/test-set loaders (`load_all_test_sets`, `load_chr_*`, `_atomic_savez`) imported by the live launcher; also the old HP bake-off driver (17 registered `bl_*` cells) |
| `llm_autoresearch.py`, `hp_strategies.py`, `hp_knowledge_base.py`, `ray_tune_search.py` | HP-search bake-off (Phase 0-2 of ONBOARDING); `albench/arch_search/propose.py` still imports `_call_claude_cli` from `llm_autoresearch.py` |
| `test_set_guards.py` | `CANONICAL_ORACLE_ID` and unstamped-cache refusal; LIVE |
| `train_oracle_s2_v2.py` | how oracle_v3 was trained (needed to rebuild the oracle) |
| `oracle_oof_eval.py`, `deploy_train.py`, `train_dream_rnn_oracle.py` | no importer, no mention anywhere |

### 3.4 `models/`, `data/`, `evaluation/`, `configs/`

- `models/` (17 files): AlphaGenome heads/embedding cache (needed by the oracle loader), `legnet*.py` + `training*.py` (older student path used by the HP bake-off), DREAM-RNN/CNN/Attn (yeast). The study's LegNet bank does **not** use these; it loads `external/human_legnet/model.py`.
- `data/` (8 `.py` files, 54 KB) shares its name with the 5.9 GB git-ignored data tree: `k562_full.py` holds the plasmid constants `MPRA_UPSTREAM/DOWNSTREAM`; `yeast.py`, `k562.py`, `loaders.py` are legacy dataset classes.
- `evaluation/exp1_eval.py`: used only by `exp1_1_scaling.py`.
- `configs/`: only `curves/human_additive.yaml` and `screen/human_30k_300k.yaml` remain (Sep-era pipelines). The Hydra tree that README/ARCHITECTURE describe no longer exists in git.

### 3.5 `ar2/` (65 files, 13k LOC) + `tests/ar2/` - "AutoResearch v2"

A self-contained controller-driven architecture search (data manifests, evaluator, gatekeeper, memory db, job queue, proposer with Claude Code CLI adapter, tuner, simulator). Built Oct 28; run as 9 worker cells (`ar2w_*`) and 24 per-D ensembling cells (`ar2m_*`), all finished. Not mentioned in `AGENTS.md` or any doc; `albench` does not import it (it imports `albench.ensemble.stack`). 2 of its tests fail locally (`test_evaluator::test_stress_set_and_rule`, `test_gatekeeper::...[batch_leak]`).

### 3.6 `tests/` (83 files; 486 tests collected outside the 6 broken modules: 481 pass, 3 fail, 2 skipped)

Passing and meaningful: `test_menu.py` (525 lines, 30 s), `test_flank_context.py`, `test_registry.py`, `test_pools.py`, `test_run.py`, `test_acquisition.py`, `test_agent_*.py` (5 files), `test_composition.py`, `test_holdout_defaults.py`, `test_label_pool_multicell.py`, `test_compile_*.py`, `tests/test_albench/*`. Smoke subset (registry, pools, run, albench, holdout, flank_context, acquisition): **138 tests in 5 s** on a laptop.
Broken: 6 modules fail at collection (they import `experiments.exp0_scaling`, `exp1_benchmark`, `exp2_rounds`, `exp3_pool_size`, `exp4_cost`, `exp5_best_student` or `scripts/preflight/run_single.py`, all moved to `attic/`), so `pytest tests/ -q` (the command in `AGENTS.md`) aborts with "6 errors during collection"; plus 3 failures.

### 3.7 `docs/` (14 files)

Current: `MENU.md` (recipe, K per size, shared folder), `TUNED_LEGNET.md` (per-size recipe), `STUDY_LOG_OCT29_NOV2.md`, `LEGNET_VS_LOOP_AND_NEXT_ARMS.md`, `LOOP_DIVERSITY_AND_VALIDATION.md`, `per_d_procedure_audit_oct30.md`, `ENVIRONMENT.md`, `figures/` (2 PNG). Design/background: `STRUCTURE.md` (harness-vs-study seam; steps 1-2 done), `ATTIC.md`, `DATA_ARCHIVE.md`, `REPRODUCE.md` (Sep-15 screen and yeast arms: stale), `frontier_session_brief.md` (74 KB prompt written for an outside model on Oct 22: a snapshot, not documentation).

### 3.8 `slide_figures/` (76 files, 17 MB) and `docs/figures/`

Generated PNGs plus the `data_*.json` snapshots the deck builders read (`slide_figures/benchmarks/data_*.json`, `scaling/scaling_law_data.json`, `reservoirs/data_*.json`). Produced by `scripts/analysis/make_*_figures.py`, `make_deck_figures.py`, `make_slide_figures.py`, `make_architecture_diagrams.py`. Regenerable; the PNG history alone is ~22 MB of repository blobs.

---------------------------------------------------------------------------------------------------

## 4. Entry points a newcomer runs

| goal | command | needs |
|---|---|---|
| set up (laptop) | `uv sync --extra dev` (pyproject) **or** `scripts/setup_env.sh [--oracle]` (pinned `requirements/menu.txt`; copies Muon from the lab share) | `uv`; Muon is not on PyPI |
| what do I have / what is missing | `albench doctor` | - |
| list everything tunable | `albench list [--kind reservoir\|acquisition]` | - |
| generate sequences | `albench generate --strategy random --n 1000 --out x.npz`; `albench sweep --strategy zoonomia --n 100000 --set a=1,2 --write jobs.sh` | assets per strategy |
| expand the Sep screen | `albench screen ...` | legacy |
| menu: what to train | `albench menu list --size 30000`; `albench menu export --size 100000 --out cfg.json` | - |
| menu: train/stack/predict/evaluate | `albench menu train --pool P.npz --out DIR --shard 0/4`; `... stack`, `... predict`, `... evaluate` | GPU, Muon, `val_v0c` asset |
| build + label a pool | `python scripts/build_pool.py --strategy S --size N --generation-seed K --out-dir D` | oracle (`ALBENCH_ORACLE_DIR`/`AG_S2_ORACLE_DIR`), `ag_weights`, JAX env |
| tuned LegNet bank | `python scripts/program_oct22/bench33_legnet.py --D 30000 --cfg e500_wd2 --seeds 0=11,1=12 --ctx 20` | `external/human_legnet/model.py` (not in git) |
| register experiment cells | `python scripts/program_oct22/register_nov2b.py --dry` | cluster paths hard-coded |
| run the tests | `pytest tests/ -q` fails; `pytest tests/test_registry.py tests/test_pools.py tests/test_run.py tests/test_albench tests/test_flank_context.py tests/test_acquisition.py -q` passes in 5 s | - |
| refresh the collaborator snapshot | `python scripts/make_shared_release.py --dest /grid/koo/home/shared/albench/ALBench-S2F --verify` | - |

---------------------------------------------------------------------------------------------------

## 5. Data assets and where they live

### 5.1 `albench/core/paths.py` asset table (`albench doctor` output on the laptop: 12 of 13 missing, only `k562_test_sets` found)

Resolution order: explicit path > env var > `paths.local.yaml` > `<ALBENCH_DATA>/<default_relpath>` > `repo_relpaths` > `fallbacks`.

| asset key | env var | default relpath | what | CSHL location (fallback / known) |
|---|---|---|---|---|
| `oracle` | `ALBENCH_ORACLE_DIR` (+ `AG_S2_ORACLE_DIR`) | `oracle/oracle_v3_s2` | oracle_v3, 10 fold models | `/grid/koo/home/shared/albench/oracle_v3_s2`; repo: `outputs/oracle_v3/s2` |
| `ag_weights` | `ALPHAGENOME_WEIGHTS` | `alphagenome/alphagenome-jax-all_folds-v1` | AlphaGenome backbone | `/grid/wsbs/home_norepl/christen/alphagenome_weights/...`, `/grid/koo/home/shared/albench/alphagenome_weights/...` |
| `val_v0c` | `ALBENCH_VAL_V0C` | `val_variants/val_V0c.npz` | validation set V0c | `/grid/koo/home/shared/albench/data/val_V0c.npz`; repo: `outputs/val_variants/val_V0.npz` (same sequences; naming V0 vs V0c is confusing) |
| `k562_test_sets` | `ALBENCH_K562_TEST_SETS` | `k562/test_sets` | genomic / ood / random_32k oracle labels (8 MB) | `/grid/koo/home/shared/albench/data/test_sets`; repo: `data/k562/test_sets_ag_s2_chrsplit` |
| `menu_ensembles` | `ALBENCH_MENU_ENSEMBLES` | `menu_ensembles` | trained genomic 10k/30k ensembles with weights (bare-insert era) | `/grid/koo/home/shared/albench/ensembles` |
| `hg38` | `ALBENCH_HG38` | `reference/hg38.fa` | reference genome | 3 fallbacks in other people's homes (`dalin`, `toobian`, `Elzar_Oxford`) |
| `jaspar_meme`, `cisbp_meme`, `tf_sets` | `ALBENCH_*` | `motifs/...` | motif vocabularies / TF sets for the `motif_*` arms | shared `cl_procap/annotations/` for JASPAR |
| `zoonomia_rates`, `zoonomia_alignment` | `ALBENCH_ZOONOMIA_*` | `zoonomia/...` | per-position rates (190 MB), 241-mammal alignment h5 | `/grid/koo/home/shared/d3/data/zoonomia/zoonomia_241.h5` |
| `encode_peaks`, `bg_cache` | `ALBENCH_ENCODE_PEAKS`, `ALBENCH_BG_CACHE` | `encode_accessibility`, `chr_split_cache/chr_train_ref_only.npz` | peaks (descoped arm); train-chromosome background pool | repo: `outputs/chr_split_cache/` |

Five `how_to_get` strings point at scripts that are not tracked (`scripts/build_chr_split_cache.py`, `build_zoonomia_rates.py`, `build_encode_partition.py`, `generate_corrected_negatives.py`) or at a CLI sub-command that does not exist (`albench setup-data`).

### 5.2 Shared lab folders (CSHL, `/grid/koo/home/shared/`)

| path | content |
|---|---|
| `albench/README.md` | the collaborator README (layout, input standard, V0c, quick start) |
| `albench/ALBench-S2F/` | curated code snapshot built by `scripts/make_shared_release.py` (RELEASE.md: commit `fa7b7d2`, Oct 29; contains `albench/`, 9 tests, `docs/{MENU,STRUCTURE,ENVIRONMENT}.md`, `AGENTS.md`, `build_pool.py`, `label_pool.py`, `generate_labeled_pools.py`, and an import closure of `data/ models/ experiments/exp1_1_scaling.py ...`) |
| `albench/menu/` | `menu_v1c.json`, `per_size_v1c.json`, `menu_{10k,30k,100k}.json`, `v0/` |
| `albench/ensembles/` | `genomic_{10k,30k}_patience{8,16}` (90 members each, v0 menu, bare insert) |
| `albench/data/` | `val_V0c.npz`, `val_V0c_measured.npz`, `test_sets/` (3 npz), `genomic_start/` (10k, 30k training sets), `plasmid_flank.txt` |
| `albench/oracle_v3_s2/` | `fold_0..fold_9`, README |
| `albench/alphagenome_weights/`, `albench/vendor/` | backbone weights; `muon.py`, `alphagenome_ft/` |
| `albench/scaling_oct13_14/`, `albench/oct28_results/` | result packages (figures, tables, `data_oct3x.json`, `ar2/`) |
| `data/episomal_mpra/gosai_2024_k562_30k_autoresearch/` | the 30k training package handed to an outside PI (train_30k, V0c with measured labels, test sets) |
| `data/` (other lab members) | `lentimpra`, `reference`, `chrombpnet`, ...; not ours |
| `d3/data/zoonomia/zoonomia_241.h5` | Zoonomia alignment (collaborator) |

### 5.3 What `docs/DATA_ARCHIVE.md` says is not reproducible

`data/neg_aug_*`, `data/cpg_*`, `data/synthetic_negatives_*` (git-ignored since Sep 15); only `synthetic_negatives_corrected` had a generator, and that script is no longer tracked.

---------------------------------------------------------------------------------------------------

## 6. Experiment-registration scripts by cell prefix

Each `scripts/program_oct22/register_*.py` appends cells (`name`, `gpus`, `out_dir`, `env`, `needs`, `cmd`, `done`) to `scratchpad/dispatch_registry.json` under `flock`, writes a `.bak_<tag>` copy first, and supports `--dry` / `--only`. Counts are registered cells (all done unless noted) from the cluster registry of 2026-11-02 (2,881 cells; 40 not finished at 17:20: hab2 x9, rr38 x3, pa38 x4, lv39 x6, rr39 x3, ef39 x9, vc39 x3, lr39 x4).
Common design: canonical nested genomic pool unless a replicate pool (r1..r3) is named; V0c validation; patience 16 for retrains; one seed per configuration; paired controls.

### 6.1 Nov 1-2 (latest; mostly unfinished, gated by `scratchpad/gates/<name>_go`)

| prefix (cells) | script | what it tests |
|---|---|---|
| `lr38` (40) + `lr38s` (9) | `register_lr38, 38b, 38c, 38d, 38s` | per-size recipe of the published LegNet: one-knob neighbours of e100, epochs 150-800, weight decay 0.05-0.4; `lr38s` = 20-seed bank at the per-size recipe. **Main result** (`docs/TUNED_LEGNET.md`) |
| `lr39` (6; 4 pending at last look) | `register_lr39.py` (untracked) | schedule length gaps (e200/e300) and early stopping inside the one-cycle schedule |
| `ep38` (9) | `register_ep38` | menu v2 with 3x epochs and no early stopping: menu is not under-trained |
| `pa38` (6; 2 done) | `register_nov2b` | harness parity: published LegNet through the *menu library harness* (patience off, fp16) vs the native bank |
| `rr38` (8; 5 done), `rr39` (3) | `register_rr38`, `register_nov2b` | the standard managed loop (coordinator, 40% explore, reference LegNet, bank bar) at 10k/30k/100k; `rr39` repeats it after the launcher patience fix (`AL_PATIENCE`, `AL_ONECYCLE_FULL`) |
| `lv39` (6) | `register_nov2b` | loop restricted to LegNet variants (manager `variants`); 10k x2, 30k x3, 100k x1; `lv39b_go` gate stages 10k/100k |
| `ef39` (9, gated `ef39_go`) | `register_nov2b` | explore fraction 0.2 / 0.6 / 1.0 with the rr39 settings |
| `vc39` (3, gated `vc39_go`) | `register_nov2b` | loop with the mixed validation set `val_B1` (`scripts/val_variants/build_val_composition.py`) |
| `hab2` (9, **0 done, stale**) | scratchpad-only `harness_ab.py` | Oct-20 harness A/B; still registered, never finished |

### 6.2 Oct 29-31 (input standard, recipes, loop controls, collaborator architectures)

| prefix | script | what |
|---|---|---|
| `sw38` (102) | `register_sw38` | context length (c12..c60) and shift (s0..s30, oracle schedule `orc`, `rca`) sweeps around the standard; 5 standard cells x 17 arms. Result: 20 bases / shift 10 is the standard |
| `fx35` (40), `fr35` (40), `fe35` (4), `lg35c` (9) | `register_fx35b`, `register_fx36`, `register_pk35` | real flank context (4..200 bases), EMA + BN recalculation, fairness across 10 reservoirs with context |
| `fx36` (12) | `register_fx36` | augmentation closer to the oracle's (per-sample shift, p=0.5, shift TTA) |
| `pk35` (16), `lg35` (12) | `register_pk35` | where a collaborator's 0.8165 single model comes from (weight averaging, BN recalculation, dilation) |
| `ctx35` (4), `lng35` (4), `mgr35` (6) | `register_lr35` | loop with flank-context harness; long (200-model) runs; explore/exploit manager |
| `st38` (31) + `st100` | `register_st38`, `register_st100` | gentle steering arms a0..a4 (reference shown, harness-assigned recipe, quota manager, minimal guidance); st100 at 100k |
| `mgr34` (18), `mgr38` (6), `op38` (9) | `register_mgr34`, `register_mgr38`, `register_op38` | manager arms; reference fallback; open prompt without the LegNet hint |
| `bench38` (65), `bench33` (34) | `register_bl38`, `register_ln100(b)`, `register_bench33` | published MPRA-LegNet (e100 / b256 / pub) 20-seed banks at 10k/30k/100k; AlphaGenome-FT baselines (`bench33_ag_*`) |
| `lg38` (12), `lg34` (8) | `register_lg38`, `register_div34c` | collaborator-style LegNet reference (EMA/BN) at other sizes |
| `pi38` (55), `tr38` (29) | `register_pi38` | five collaborator architectures re-implemented (`build_menu_pi.py`), two recipes each; recipe transplant (menu v2 / PI archs under the LegNet recipe) |
| `hp38` (88), `rc38` (34), `v2m38` (42), `v1c38` (16), `mv1c` (4), `mt38` (12), `scl30` (33) | `register_nx38`, `register_ln100`, `register_mt38`, `register_mv1c`, `register_oct29/30` | menu recipe sweeps at 10k/100k; single-pass RC augmentation; menu v2; menu v1c; auxiliary HepG2 target; width-scaled configurations |
| `hp34` (36), `fl34` (4), `ds34` (8), `hp34s1`/`fl34s1` (8) | `register_div34`, `register_div34b` | recipe diversity of menu v1 at 30k; flank shift; menus searched at 10k/100k; second seed for the noise floor |

### 6.3 Oct 22-28 (program on pools, reservoirs, acquisition, scaling)

| prefix | script | what |
|---|---|---|
| `g30/rnd30/mot30/g10/g100x` (15/15/15/10/7), `u/u10/u100` (18/18/9), `un30/un10/un100` (63/84/72), `mix` (63), `ind30` (45), `mx30` (15) | `register_big`, `register_mix`, `register_unseen_sizes`, `register_oct23/26c/28`, `stage2.py` | independent-pool AutoResearch runs and menu retrains for genomic/random/motif/evoaug/evo2 and "unseen" reservoirs (motif_shared_core, motif_syntax_core, zoonomia_orthologs, mutagenesis, gc_matched, dinuc_shuffle); mixed corpora (genomic + random + motif) |
| `ac30/an30/ae30/ag30/xd30/xr30/ex30/rf30` | `register_agent_pilot`, `register_oct25`, `register_oct26b`, `register_oct27`, `register_oct28` | loop-element controls: coordinator, reflector, separate coder, random picks, prompt content (`--guidance`) |
| `ind30e` (110), `ind30v` (105), `ind30h` (96), `ind30s/l/q/r/n1/n2/m/x/w/t` | `register_oct26/27/28/29`, `register_val_variants` | menu recipe sensitivity (patience 4..32, epochs x0.5/x2), validation-set variants (V0x/V0c/V1-V3), HepG2 and differential students, noise, seeds, ladders |
| `sc31` (96), `sc32` (36), `al31` (120), `as` (90), `al3r` (24), `a600/a1m` (34/34), `mrr` (60) | `register_oct26`, `register_oct30`, `register_oct30b`, `register_yash`, `register_div34` | **scaling** (11 reservoirs x 10k/30k/100k, 300k, 600k/1M anchors) and **acquisition** (30k genomic + 70k selected by SD, mean, diversity, BADGE-like k-means++ ... from a 300k candidate pool; `acq_dryrun.py`, `acq_arms2.py`, `acq_rounds.py`) |
| `ar2w` (9), `ar2m` (24), `pool31`, `pool*` | `register_ar2`, `register_ar2menu`, `register_oct26/30` | ar2 search workers and per-D ensembling; raw pool builds (`build_pool.py` loops) |
| `bl` (17), `mr/mr2/mrf/mrf2` (16/29/40/44), `xe10/xe30/xd10`, `fam` (12), `af` (15), `r100` (12), `ctx` (16), `nest` (15; `scale_configs.py`) | cluster-only `scratchpad/build_registry.py`, `add_menu_fam*.py`, `register_reps.py`, `register_x.py`, `seed_cells.py` (not in git) | Oct 13-21: HP-search baselines through `scaling_hp_search.py` (`bl`), menu retrains at 10k/30k/300k (`mr*`, `nest`), 30-round AutoResearch per architecture family (`fam`, `af`), AutoResearch at 100k (`r100`), cross-D context hand-off ladder (`ctx`, not flank context) |

(Registrations whose generator is **not** in git: everything in the last row; `register_genomic_menu_sizes.py` exists only on the cluster.)

---------------------------------------------------------------------------------------------------

## 7. What is where on the cluster that is not in git

### 7.1 `scratchpad/` (159 MB, 387 entries, untracked) - the live operations directory

Essential to run the current experiments:

| file | purpose | repo copy |
|---|---|---|
| `dispatcher.py` (539 lines, Oct 30) | cross-queue dispatcher: per-cell chain of short Slurm jobs, hops between QoS tiers and GPU types (`fast` 2 H100 + 2 V100, `default` <=15 submitted / 8 running, `slow_nice` 20 + 20, MIG slices on `migq` for small retrain cells), dependency gates (`needs`), `defer_prefixes.txt`, `mig_cap.txt`, `dispatch_STOP`, LLM rate-limit handling (exit 75, probe, cooldown), stall flags, 3-short-failure pause | **none** (not in git) |
| `dispatch_registry.json` (2.4 MB, 2,881 cells) + `.registry.lock` | the cell list the dispatcher re-reads every cycle | none; ~60 `.bak_<tag>` copies (about 100 MB) |
| `arch_autoresearch.py` (828 lines, Nov 2) | **the launcher every loop cell runs** (patched in place by 12 `patch_loop_*.py` / `patch_harness_*.py` scripts) | `albench/arch_search/arch_autoresearch.py` is different: HEAD copy lacks the input standard, EMA/BN, patience env, reference LegNet, val cache wiring (708 lines only in cluster, 126 only in HEAD); the dirty local worktree is a half port (357 lines still only in cluster) |
| `menu_retrain.py` (319 long lines = 605 formatted, Oct 29) | the retraining script every menu cell runs | `scripts/program_oct22/menu_retrain.py` lacks 197 lines of the live one (`--flank_ctx`, `--shift_max`, `--shift_per_sample`, `--shift_prob`, `--tta`, `--aux_labels`, `--optimizer`, `--lr_abs`, `--wd_abs`, `--no_context`, standard defaults) |
| `agent_loop.py` | identical to `albench/arch_search/agent_loop.py` | in git |
| `stage2_lib.py`, `stage2.py` | pool slicing helpers (`slice_unseen`, `make_genomic_reps`), cron follow-up that slices pools and registers dependent cells | identical copies in `scripts/program_oct22/` |
| `ens_util.py` | scale-robust `enet_all_pos` (imported by `scripts/val_variants/analyze_val_variants.py`) | none |
| `gates/` (empty), `defer_prefixes.txt` (empty), `mig_cap.txt` (=20) | operator switches | - |
| `dispatch_jobs/` (4,855 sbatch files) | the generated job scripts | - |

One-off or historical (about 95 `.py`, 51 `.sh`, 45 `.sbatch`, 36 `.log`, ~60 registry backups, ~70 `*.bak_*` of dispatcher/launcher/menu_retrain, 16 `*_cells/` directories of pre-dispatcher sbatch files):
- Oct 14-21 analyses and generators: `gen_*_oct1x.py`, `morning_aggregate*.py`, `compare_arms.py`, `boot_arms.py`, `confound.py`, `leak_probe.py`, `prior_vs_search.py`, `harness_ab.py` (+ `ab_decomp.py`), `weight_transfer_oct15.py`, `usage_scan.py`, `route_qos.py`, `dispatcher_v2.py`, `build_registry.py`, `add_menu_fam*.py`, `register_reps.py/x.py`, `seed_cells.py`, `finish_reps.py`;
- one-shot patch scripts (`patch_arch.py`, `patch_final.py`, `patch_reset.py`, `patch_ratelimit.py`, `patch_slowmig*.py`, `patch_eff4/5.py`, `fix_wd.py`, `fix_free.py`, ...);
- `an_*.sh`/`.log`, `cmp_*.sh`, `smoke_*.sh`, `launch_*.sh`: wrappers that submit analysis scripts to `cpuq`.

### 7.2 Run machinery outside git

- **Cron** (one entry): every 10 min, if no `dispatcher.py` process and no `dispatch_STOP` file, start `tmux new-session -d -s dispatch "... python3 scratchpad/dispatcher.py >> outputs/dispatch/stdout.log"` with the system Anaconda python (no venv).
- **Credentials** (locations only): `~/.config/albench/llm.env` (`CLAUDE_CODE_OAUTH_TOKEN`, sourced by every sbatch), `~/.config/albench/gh_token`.
- **Venv**: `ALBench-S2F/.venv` (Python 3.11.7, alphagenome 0.6.0, alphagenome_ft 0.1.8, jax 0.9.0.1, vendored Muon); cells run `uv run --no-sync python ...` after `source .venv/bin/activate`.
- **Account**: SSH key login works only while `/grid/koo/home/christen` is not group-writable (memory note); login nodes bamdev1-4 share Slurm and filesystem.

### 7.3 Data/outputs on the cluster checkout (all untracked/ignored)

- `outputs/` (544 entries): `grid_pools/` (20 dirs: `k562_nested` 68 files, `k562_reps` 192, `k562_reps_raw*`, `k562_hepg2`, `k562_acq`, legacy `k562_cumulative`, `README_NESTING.md`), `oracle_v3/{s1,s2,s2_screen}`, `val_variants/` (24 entries; `val_V0.npz` is V0c), `menu_retrain_v0/` (700 run directories: every student result), `archsearch_reps_oct22/` (loop runs), `bench33/`, `menus_oct28..nov2/` (built menus), `ar2/`, `dispatch/` (state.json, dispatch.log, 4,111 job logs), `chr_split_cache/`, `justification_oct22/` (analysis data), `gosai_k562_lfcse.npy`, many Oct 13-22 experiment dirs.
- `data/k562/` (Gosai table, `test_sets_ag_s2_chrsplit`, `oracle_folds_v2.json`, ...), 14 more derived-data dirs.
- `external/` (2 GB): `human_legnet` (clone of the published repo; **required by `bench33_legnet.py`**), `borzoi-pytorch`, `nucleotide-transformer`, `PARM`, `sahu_geo`.
- Clutter (none referenced by the registry or dispatcher; newest file is Oct 15): **147 loose files at the repo root** (108 `.sbatch`, 23 `.py`, 10 `.sh`, 4 `slurm-*.out`...), **129 untracked files in `scripts/slurm/`** (103 from May), 85 loose scripts in `scripts/`, dirs `analysis/`, `ar2/` (stale copy), `logs/` (947 MB), `tmp_sync_bundle/`, `deprecated_hashfrag_archive/`, `checkpoints/`, `wandb/` (505 MB), `experiments/archive/`, `scripts/{colab,preflight,scaling_ensemble,scaling_hp_search,_inference_tmp}/`.
- **116 untracked files collide with tracked files in `origin/main`** (copies scp'd instead of pulled), so a plain `git pull` on the cluster fails until they are moved away.

---------------------------------------------------------------------------------------------------

## 8. Divergences you need to know about before trusting the repo

1. The launcher (`scratchpad/arch_autoresearch.py`) and the retraining script (`scratchpad/menu_retrain.py`) that the registry runs are **ahead of git**. `AGENTS.md` says the retraining script and the loop take the input standard from `harness_standard.py`; in git neither does (0 references), only `albench.ensemble.menu` does.
2. `bench33_legnet.py` depends on `external/human_legnet/model.py`, which is git-ignored, so the tuned-LegNet bank cannot be rebuilt from a fresh clone. (`albench/arch_search/mpra_legnet_ref.py::SOURCE` embeds the same architecture and could replace it; equivalence of outputs not measured here.)
3. 108 tracked files hard-code `/grid/wsbs/home_norepl/christen/ALBench-S2F` (149 occurrences), including `albench/arch_search/arch_autoresearch.py` and `bridge.py`, which `AGENTS.md` says must contain none.
4. The oracle loader lives inside a 3.5k-line legacy driver (`experiments/exp1_1_scaling.py`).
5. Docs cite `scratchpad/oct28/PLAN.md` ("a local working log"): it exists only in the laptop session scratch space.

---------------------------------------------------------------------------------------------------

## 9. How liveness was determined (so the plan's claims can be re-checked)

- AST import graph of all 535 tracked `.py` files (function-level imports included; `sys.path` bare-module imports resolved by unique basename): `the audit's scratch folder (not kept)`.
- Roots: `albench/cli.py`, tests, all `register_*.py`, and the scripts that registry cells actually execute (counted from the cluster registry: `menu_retrain.py` 2,170 cells, `arch_autoresearch.py` 493, `bench33_legnet.py` 102, `bench33_ag.py` 46, `build_pool.py` 27, `scaling_hp_search.py` 17, `harness_ab.py` 9, `ar2 worker` 9, `albench.cli menu train` 1, `scale_configs.py` 1).
- Reachable from those roots: albench 64/78 modules, models 16/17, experiments 8/12, scripts/analysis only 3/173 (analysis scripts are leaves, so they are classified by documentation, recency and mention instead).
- Textual mentions (basename or path in any tracked text file), last-touch dates, doc citations, `git grep` of class and module names, cluster md5 comparison of 112 scratchpad `.py` against tracked files (16 identical, 2 same name different content, 94 not in git).
- Commands run: `pytest tests -q` (6 collection errors; the rest 481 passed / 3 failed / 2 skipped in 208 s), `ruff check .` (1,146 errors in 91 files), `ruff format --check .` (107 files would change), `albench doctor|list|generate`.
