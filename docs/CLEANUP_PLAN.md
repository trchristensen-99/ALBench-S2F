# ALBench-S2F: cleanup plan for onboarding collaborators (read-only audit, 2026-11-02)

Companion to `docs/CODEBASE_OUTLINE.md`. Nothing here has been executed: every command is a proposal for review. Risk labels: **none** (cannot change behaviour), **low** (behaviour of a live run cannot change; a tool or doc may), **needs-check** (a human must confirm something first). "After" lists what to run to prove nothing broke.

---------------------------------------------------------------------------------------------------

## 0. Summary

**Five most valuable actions (in order)**

1. **Put the code that actually runs into git.** The launcher every loop cell executes (`scratchpad/arch_autoresearch.py`, 828 lines), the retraining script every menu cell executes (`scratchpad/menu_retrain.py`, 197 lines ahead of the tracked copy) and the dispatcher itself (`scratchpad/dispatcher.py`, 539 lines, not tracked at all) exist only on the cluster. The tracked launcher and retraining script contain none of the input standard that `AGENTS.md` says they take from `harness_standard.py`. The local worktree holds a half port (357 lines still missing). Sync into git, pull on the cluster (adds 338 files, modifies 11, deletes 0; 116 untracked copies must be moved aside first), then rewrite the registry commands of the 40 unfinished cells. Steps P1a to P1c.
2. **Replace the three front-door documents and fix the test command.** `README.md`, `ARCHITECTURE.md`, `ONBOARDING.md` describe a Hydra Exp0-5 benchmark and an HP bake-off that no longer exist in git (26 documented files missing: 16 scripts and 10 experiment/config/model files; 6 documented docs missing); `pytest tests/ -q` (the command in `AGENTS.md`) dies with 6 collection errors; `AGENTS.md` tells people to edit shim files. Steps P2, P6.
3. **Prune about 115 dead or superseded files** (92 old analysis scripts, 4 `.bak`, 6 stale tests, 9 unregistered samplers + BAIT, the filesystem LLM bridge, a second `pyproject.toml` and README) and move the 79 current analysis scripts and the 90 `program_oct22` scripts into one clearly named `study/` tree, with 34 Sep-era tools under `study/legacy/`. Steps P3, P5.
4. **Remove machine-specific paths and the hidden dependency.** 108 tracked files hard-code `/grid/wsbs/home_norepl/christen/ALBench-S2F` (149 occurrences), two of them in `albench/` (forbidden by `AGENTS.md`); the tuned LegNet bank needs `external/human_legnet/model.py`, which is git-ignored and exists only on the cluster. Step P4.
5. **Tidy the cluster and the local `.git`.** 147 loose files at the cluster repo root (none referenced; newest Oct 15), 129 untracked files in `scripts/slurm/`, ~130 `.bak` files and 16 `*_cells/` directories in `scratchpad/`; locally `.git` is 268 MB though everything reachable from `main` is 37.7 MB (reflog + 3 old stashes). Steps P8, P9.

**Items I am NOT sure are dead** are listed in section 5.

**Ordering constraint.** The dispatcher is live (10-12 jobs running, 40 cells unfinished: `hab2`, `rr38`, `pa38`, `lv39`, `rr39`, `ef39`, `vc39`, `lr39`). Those cells run `scratchpad/arch_autoresearch.py`, `scratchpad/menu_retrain.py` and `scripts/program_oct22/bench33_legnet.py`. Do not move or rename those paths on the cluster until their `cmd` strings have been rewritten (P1c) or the cells have finished. The dispatcher re-reads the registry every 180 s, so a rewrite needs no restart. Other people/agents are editing the local worktree right now: do the work on a branch.

---------------------------------------------------------------------------------------------------

## 1. Classification

### 1.1 Top-level entries

| path | class | evidence / how verified |
|---|---|---|
| `albench/` | **LIVE** | `albench` CLI entry point; imported by every script and the loop; 64 of 78 modules import-reachable from the live roots, the rest listed in 1.3 |
| `tests/` | **LIVE** (9 stale files) | 481 tests pass; 6 modules import files that moved to `attic/`; see 1.4 |
| `docs/` | **LIVE**, partly stale | 8 current notes; `REPRODUCE.md`, `ATTIC.md` stale; `frontier_session_brief.md` is a snapshot |
| `requirements/` | **LIVE** | consumed by `scripts/setup_env.sh`, `docs/ENVIRONMENT.md` |
| `scripts/` | mixed | per-subdirectory table in 1.2 |
| `experiments/` | mixed: 5 LIVE-library, 5 REPRO, 2 unreferenced | `exp1_1_scaling.py` is the oracle loader (imported by `build_pool.py`, `label_pool.py`, `label_nested_pool.py`, 14 analysis scripts); `scaling_hp_search.py` is imported by the live launcher; `llm_autoresearch.py::_call_claude_cli` is imported by `propose.py`; REPRO: `hp_*`, `ray_tune_search.py`, `train_oracle_s2_v2.py` (how oracle_v3 was trained), `oracle_oof_eval.py` (out-of-fold oracle ceiling; mentioned only in `docs/ATTIC.md`); unreferenced (0 importers, 0 mentions, no registry cell): `deploy_train.py`, `train_dream_rnn_oracle.py` (yeast) |
| `models/` | **REPRO** with a LIVE core | `alphagenome_heads.py`, `embedding_cache.py`, `loss_utils.py`, `training*.py`, `legnet*.py` are imported (through `exp1_1_scaling.py`, `bench33_ag.py`, analysis); DREAM-RNN/CNN/Attn (7 files) only by the yeast pipeline and `exp1_1_scaling.py`. The study's LegNet does **not** come from here |
| `data/` (python) | **LIVE** for `k562_full.py` (plasmid constants), **REPRO** for the rest | 8 `.py` files imported by 7-9 modules each; the same directory also holds 5.9 GB of ignored data, which is confusing (rename is **needs-check**: ~25 import sites) |
| `evaluation/` | **REPRO** | one file, imported only by `exp1_1_scaling.py` |
| `configs/` | **REPRO** | 2 YAML files, read only by `scripts/pipelines/*` and `scripts/build_curve_plan.py` (Sep-era) |
| `ar2/` | **REPRO / needs-check** | finished experiment (9 + 24 cells done), no doc mentions it, `albench` does not import it, 2 of its tests fail; 65 files, 13k LOC |
| `slide_figures/` | **REPRO** (generated) | 76 files, 17 MB; produced by `scripts/analysis/make_*figures.py`; PNGs are regenerable, `data_*.json` are deck-builder inputs |
| `ONBOARDING.md`, `ARCHITECTURE.md`, `README.md` | **STALE** | section 2.5 |
| `pyproject.toml`, `uv.lock`, `.pre-commit-config.yaml`, `.gitignore` | **LIVE** | section 2.6 |
| `pyrightconfig.json` | **stale entries** | includes `eval_ag.py`, `test_head.py`, `custom_model.py`: none exists |
| `attic/` (ignored, 11 MB) | **on-disk archive** | `docs/ATTIC.md`; nothing may import it (verified: no tracked file mentions `attic/` outside docs) |
| `boda2-main/` (ignored, 0.9 MB) | **DEAD** | `git grep boda2` hits 3 comments/help strings; no import; untracked; delete locally |
| `results/` (ignored, 134 MB), `wandb/` (16 KB) | **DEAD (local)** | Mar-Jun outputs; only 4 Sep analysis scripts write to `results/debias_variants/` |
| `external/` (ignored; local 366 MB LegNet/borzoi-pytorch/nucleotide-transformer; cluster 2 GB) | **LIVE on the cluster, DEAD locally** | cluster needs `external/human_legnet/model.py` (bench33_legnet.py:83); the local copies are not referenced by tracked code |
| `outputs/` (ignored, 1.8 GB local, huge on cluster) | **data** | by design not in git |
| `.env` (ignored), `.claude/`, `REMOTE_ACCESS.md` (ignored, Mar 18) | local | `.env` holds a W&B key; `REMOTE_ACCESS.md` holds host/user/key-path notes and outdated queue info: keep untracked |

### 1.2 `scripts/` subdirectories

| subdir (files) | class | decision |
|---|---|---|
| `scripts/` root (34) | split | **LIVE (keep in `scripts/`)**: `build_pool.py`, `label_pool.py`, `label_pool_multicell.py`, `slice_replicate_pools.py`, `verify_pools.py`, `build_nested_pools.py`, `build_mixture_pools.py`, `label_d3_pool.py`, `make_shared_release.py`, `setup_env.sh`, `generate_labeled_pools.py` (ships in the collaborator release), asset builders `build_tf_sets.py`, `fetch_cisbp.py` (cited by `paths.py`). **REPRO -> `study/legacy/`**: yeast (`build_yeast_folds`, `build_yeast_eval_classes`, `predict_yeast_eval_classes`, `yeast_oracle_sanity_preview`), Sep curves (`build_curve_plan`, `preflight_curves`, `link_screen_pools`, `pool_params_match`, `emit_pool_commands`, `build_delta_pairs`, `build_genomic_partition`, `build_grid_pools`), `fm_scaling_driver`, `build_full_oracle_cache`, `relabel_battery_oraclev3_ensemble`, `verify_oracle_v3_repro`, and the 5 `ar2_*` files (move with `ar2/`) |
| `scripts/program_oct22/` (90) | **LIVE** | the whole Oct 22 - Nov 2 program; rename to `study/program/` once the dispatcher is idle (P5c). 50 `register_*` are the record of what each cell tested; 18 `patch_*` are provenance of launcher edits (REPRO after P1); runtime scripts are executed by cells |
| `scripts/analysis/` (177) | **72 + 7 LIVE-REPRO, 92 REPRO-low-value, 4 DEAD, 2 pipeline-bound** | Appendix B/C/E; method: cited by a doc, touched since Oct 28, or imported/mentioned by such a script = current (72); `ledger_*`/`build_justification_html`/`patch_builder*` form the Evidence Ledger builder (7); the remaining 92 are Sep to Oct 26 analyses (GC/CpG, Sahu, Agarwal, ISM, HepG2, hp_* protocol study, scaling_v2, `pi_coords*`) with 0 references outside themselves |
| `scripts/val_variants/` (13) | **REPRO** (1 LIVE) | `build_val_composition.py` feeds the gated `vc39_` cells; the rest is the Oct 23 V1-V3 study (`register_val_variants.py` says "NOT launched"); move all to `study/val_variants/` |
| `scripts/pipelines/` (11) | **REPRO** | Sep-era (`human_screen.sh`, `yeast_oracle.sh`, curves, canonical Oct-13 watchdogs); the current work does not use them; `yeast_oracle.sh` calls missing `experiments/train_yeast_oracle_v2.py` |
| `scripts/cluster/` (2) | **REPRO** | portable `submit.sh`/`site.env.example`, superseded by the dispatcher; `docs/REPRODUCE.md` still documents it |
| `scripts/slurm/` (1) | **REPRO** | `relabel_battery_oraclev3.sbatch` |

### 1.3 Library-level candidates (inside `albench/`, `experiments/`, `models/`)

| item | class | evidence |
|---|---|---|
| `albench/arch_search/bridge.py` | **DEAD** | 0 importers, 0 mentions; filesystem LLM bridge replaced by the CLI token path on Oct 16 |
| `albench/arch_search/gh_bridge.py` | **REPRO / needs-check** | imported only under `LLM_USE_BRIDGE=1` in `experiments/llm_autoresearch.py:920` and by an Oct-13 watchdog script; 0 of 2,881 registry cells set `LLM_USE_BRIDGE` |
| `albench/pyproject.toml`, `albench/README.md` | **DEAD** | nested second package definition (placeholder `yourusername` URL, `testpaths=tests/test_albench`); `paths._find_repo_root` already skips it. README describes `pip install albench`, `albench[analysis]` |
| 9 unregistered samplers: `activity_stratified`, `adaptive_selection`, `curriculum`, `diversity_guided`, `motif_clustering`, `motif_density`, `motif_grammar`, `recombination`, `uncertainty_guided` | **DEAD (needs-check on intent)** | never registered, importers: only `reservoir/__init__.py`; classes named nowhere else; `exp1_1_scaling.py` reaches them only by string name through Hydra configs that are no longer tracked; no pool in `outputs/grid_pools` was built from them |
| `albench/acquisition/bait.py` | **DEAD** | only `acquisition/__init__.py` imports it |
| `albench/reservoir/{tf_motif_shuffle,fixed_pool,mixed_pool}.py`, `acquisition/{combined,prior_knowledge}.py` | keep (REPRO) | referenced by tests / `exp1_1_scaling.py` |
| `albench/{paths,pools,registry,run,loop}.py` | keep until imports are updated (P7) | 16-line shims of `albench/core/*` (`docs/STRUCTURE.md` step 2); importers: `albench.paths` 18 files, `albench.pools` 8, `albench.registry` 7, `albench.run` 3, `albench.loop` 3 |
| `albench/arch_search/train_generated.py::main` | **DEAD entry** | imports `scaling_hp_search_ms`, which exists nowhere |
| `experiments/ray_tune_search.py`, `hp_strategies.py`, `hp_knowledge_base.py` | **REPRO** | imported by `scaling_hp_search.py`, which 17 `bl_*` cells and the launcher still use |
| `models/loss_utils.py`, `legnet*.py`, `training*.py` | **REPRO** | used by the HP bake-off path |

### 1.4 Tests

| file | state |
|---|---|
| `tests/test_experiments/test_exp1_utils.py`, `test_exp2_utils.py`, `test_exp3_utils.py`, `test_exp5_utils.py`, `test_experiment_utils.py`, `tests/test_shift_aug_adapters.py` | **DEAD**: collection errors; import `experiments.exp0_scaling`, `exp1_benchmark`, `exp2_rounds`, `exp3_pool_size`, `exp4_cost`, `exp5_best_student`, or read `scripts/preflight/run_single.py` (all in `attic/`) |
| `tests/test_experiments/test_new_strategies.py::test_partial_mutagenesis_sampler_prefers_filtered_metadata` | stale (constructor arguments changed) |
| `tests/ar2/test_evaluator.py::test_stress_set_and_rule`, `tests/ar2/test_gatekeeper.py::test_t4_broken_models_fail_their_named_check[batch_leak]` | fail on this laptop (torch 2.10, Python 3.12, CPU): needs-check whether environment or code |
| everything else (481 tests) | keep; subset `test_registry test_pools test_run test_albench test_holdout_defaults test_flank_context test_acquisition` = 138 tests in 5 s is the proposed smoke test |

---------------------------------------------------------------------------------------------------

## 2. Findings that drive the steps

### 2.1 Live code that is not in git (details: outline section 7 and 8)

| file | cluster (live) | git |
|---|---|---|
| `dispatcher.py` | `scratchpad/dispatcher.py`, 539 lines, patched in place about 30 times since Oct 22 (backups kept) | absent |
| launcher | `scratchpad/arch_autoresearch.py`, 828 lines (13 references to the input standard) | `albench/arch_search/arch_autoresearch.py`: HEAD 593 lines, 0 references; dirty worktree 837 lines, 0 references, 357 lines still only on the cluster; the cluster's own repo copy of that file differs from HEAD by 110 lines |
| `menu_retrain.py` | `scratchpad/menu_retrain.py`, standard defaults + 9 flags | `scripts/program_oct22/menu_retrain.py`: lacks 197 lines, has 13 the cluster no longer has |
| `agent_loop.py`, `stage2.py`, `stage2_lib.py`, 7 `register_*` | `scratchpad/` | identical copies tracked (md5) |
| `ens_util.py`, `harness_ab.py`, `build_registry.py`, `register_genomic_menu_sizes.py`, ~90 more | `scratchpad/` | absent |
| `external/human_legnet/model.py` | cluster `external/` | ignored (needed by `bench33_legnet.py`) |
| albench library on the cluster | `menu.py`, `flank_context.py`, `bench33_ag.py` differ from HEAD | HEAD newer |

### 2.2 Absolute paths and user names in tracked files

| pattern | files | notes |
|---|---|---|
| `/grid/wsbs/home_norepl/christen/ALBench-S2F` | 108 files / 149 lines (160 incl. other `.../christen/...` paths) | `scripts/program_oct22` 53, `scripts/analysis` 44, `scripts/` 8, `scripts/pipelines` 6, `val_variants` 4, docs 3, **`albench/arch_search/arch_autoresearch.py:25`**, **`albench/arch_search/bridge.py:27`**, `ONBOARDING.md`, `ARCHITECTURE.md`, `requirements/oracle.txt` |
| `/grid/koo/home/shared` | 20 files / 36 lines | `core/paths.py` fallbacks (allowed), `setup_env.sh`, `docs/ENVIRONMENT.md`, 8 analysis, 6 registrations |
| other people's homes | `core/paths.py` fallbacks: `/grid/koo/home/dalin/...`, `/grid/vakoc/home/toobian/...`, `/grid/ngs/data/Elzar_Oxford/...`; `/grid/koo/home/mundewadi` (2) | allowed by the house rule but exposes colleagues' paths and breaks when they leave; replace by one `ALBENCH_HG38` instruction |
| `christen` | 112 files | mostly the path above |
| `trevor@143.48.59.3` (Citra), `bamdev4`, `bamgpu101` | `ARCHITECTURE.md` 4 lines; `bamdev4` also in `docs/STUDY_LOG_OCT29_NOV2.md` | remove from tracked docs |
| `/Users/...` | 0 files | clean |
| secrets | none found (`sk-ant-`, `ghp_`, `github_pat_`, `AKIA`, `xox*`, OPENSSH/RSA blocks absent from tree and from `git log -G` on `main`) | `.env` is ignored; tokens live in `~/.config/albench/` |

### 2.3 Large tracked binaries and history

- No model checkpoints or data files are tracked (no `.pt/.ckpt/.npz/.h5/.pkl`). Largest tracked blobs: `uv.lock` 369 KB, `slide_figures/deck/deck_context_effect_allsets.png` 573 KB, 76 PNGs in `slide_figures/` (18 MB total) and 2 in `docs/figures/` (0.5 MB).
- Clone size is small: `git rev-list --objects --disk-usage main` = 37.7 MB. The local `.git` is 268 MB because the reflog (242 entries) and 3 stashes keep the pre-Sep-15 history alive (`--all --reflog` = 263 MB). Largest reachable history blobs: 5 versions of `figures_schematics/schematic_hp_workflow.png` (0.7-0.86 MB each), PNG regenerations in `slide_figures/` (22 MB total).
- Do **not** rewrite history (another purge happened Sep 15 and cost a data-loss scare, see `docs/DATA_ARCHIVE.md`); only tidy the local repo (P9).

### 2.4 Duplicated modules

| what | copies | keep |
|---|---|---|
| launcher | tracked `albench/arch_search/arch_autoresearch.py`; cluster `scratchpad/arch_autoresearch.py`; cluster repo copy | one, in `albench/arch_search/` |
| retraining script | tracked `scripts/program_oct22/menu_retrain.py`; cluster `scratchpad/menu_retrain.py`; library `albench.ensemble.menu.train_member` (documented to reproduce it) | one script + the library |
| published LegNet | `external/human_legnet/model.py`; `albench/arch_search/mpra_legnet_ref.py::SOURCE`; `models/legnet.py` (repo variant, memory: "differs from canonical") | `mpra_legnet_ref.py` (after the equivalence check in P4e) |
| stack | `albench/ensemble/stack.py` (34 importers); scratchpad `ens_util.enet_all_pos`; own ElasticNetCV code in 48 files (43 in `scripts/analysis`) | `stack.py`; fold `ens_util` into it |
| one-hot encoders | 9 (`albench/utils`, `data/utils`, `ensemble/menu`, `arch_search/train_generated`, `flank_context.wide_onehot`, `bench33_common`, `fm_scaling_driver`, `motif_vocab_diag`) | `albench.utils.one_hot_encode` + `flank_context` |
| package definitions | root `pyproject.toml` and `albench/pyproject.toml` | root |
| env recipes | `pyproject.toml` (+`uv.lock`, torch>=2, jax, hydra...) vs `requirements/menu.txt` (pinned) | document both roles or unify |
| schedulers | `scripts/cluster/submit.sh` (+pipelines) vs the dispatcher | dispatcher |
| registry shims | `albench/registry.py` <-> `albench/core/registry.py` (also paths, pools, run, loop) | `core/` |
| reservoirs | `motif_planted.py` and `motif_planted_v2.py` (v1 registered through `motif_density` only); cluster-only `albench/motif_planted_v2.py`, `albench/vocabulary.py` (stray copies) | v2 |

### 2.5 Stale statements, by document

**`README.md`**
1. L8: "`REMOTE_ACCESS.md` (not tracked in git)": file is git-ignored, holds hostnames/user/key paths and queue information from March.
2. "Local Validation": `uv run pytest tests/ -v` fails at collection (6 errors).
3. "Running Experiments" 1-9: `experiments/exp0_scaling.py`, `exp0_yeast_scaling.py`, `exp1_benchmark.py`, `exp2_rounds.py`, `exp3_pool_size.py`, `exp4_cost.py`, `exp5_best_student.py`, `train_oracle_alphagenome.py`, root config `configs/config.yaml`, `+task=k562 +student=dream_rnn`: none exists in git (checked file by file).
4. "Data Download": `scripts/download_data.py` missing. "Standardized Result Aggregation": `scripts/analysis/{sync_remote_results,analyze_experiment_results,aggregate_exp0_results}.py` missing.
5. "Project Layout": omits `ar2/`, `slide_figures/`, `requirements/`, `albench/{core,ensemble,arch_search}`; says `configs/` = Hydra YAML (2 non-Hydra files remain).
6. "New here?" and "Student input standard" paragraphs are current and should be kept.

**`ARCHITECTURE.md`** (Sep 17)
1. S1: "six registered-report experiments (Exp 0-5)", student = DREAM-RNN, oracle = "AlphaGenome frozen encoder + probing head". Now: one study (corpus recommendation), oracle_v3 = 10 fine-tuned AlphaGenome-S2 folds, students = tuned MPRA-LegNet / menu / loop; DREAM-RNN is the yeast oracle only.
2. S2 tree lists `models/alphagenome_oracle.py`, `models/perfect_oracle.py`, `experiments/exp1_benchmark.py`, "exp2-5 stubs", `configs/{config.yaml,task,student,acquisition,reservoir}`, `scripts/{download_data.py,setup_runtime.sh,slurm/}`: absent; omits `albench/{core,ensemble,arch_search,motifs}`; shows `albench/loop.py` as real (it is a shim).
3. S4.1/4.3: HashFrag 80:10:10 splits, "5 channels (RC flag)". Now: chromosome holdout (chr7/chr13 test, chr19/21/X validation) and a 240-bp 4-channel input.
4. S7: Citra `trevor@143.48.59.3`; Koo HPC partition `kooq`, QoS `koolab`, node `bamgpu101`, `/grid/koo/data/christen/...` (the account has no `koolab` QoS); commands `scripts/setup_runtime.sh`, `run_with_runtime.sh`, `scripts/pipelines/human_screen.sh train`.
5. S9-S10: "Hydra only (no argparse anywhere)", "Conventional Commits", "Google-style docstrings". No Hydra entry point remains; every script uses argparse; 1,600 commits are free-form; `AGENTS.md` asks for docstrings that say why.
6. S11 "Not yet started": AlphaGenome oracle, BADGE, BatchBALD, EvoAug/motif/mutagenesis reservoirs, Exp 1-5: all implemented or retired.

**`ONBOARDING.md`** (Oct 13)
1. S1 "The current research question is which HP-search strategy is most compute-efficient" (superseded by `AGENTS.md`: a training corpus recommendation).
2. S2 "`experiments/` ... `scaling_hp_search.py` and `hp_strategies.py` are the two files you'll touch most": the live work is in `scripts/program_oct22/` plus cluster `scratchpad/`.
3. S4-S5 HP-search strategy families, Phase 0-1-2 bake-off, "200 models per cell", `STEP1_*` launch recipe via `scripts/submit_step1_bakeoff.py` (missing; the untracked copy is `experiments/submit_step1_bakeoff.py` on the cluster).
4. S6 lists 6 documents that do not exist: `docs/phase1_bakeoff_plan.md`, `docs/yeast_agent_prompt.md`, `docs/presentation_summary_malinois_vs_alphagenome.md`, `EXPERIMENT_TRACKER.md`, `Current_status_of_Experiment_{0,1}.md`.
5. S7 `scripts/analysis/plot_hp_rounds_and_ensemble.py`, `strategy_combination_ablation.py`: missing.

**`REMOTE_ACCESS.md`** (local, ignored): `scripts/slurm/setup_hpc_deps.sh` (removed), Citra, `kooq`/`koolab`, Exp0 output paths, "Key Experiment Paths"; replace by a tracked, secret-free `docs/CLUSTER.md` (P6).

**`AGENTS.md`**
1. Testing: `python scripts/test_samplers_fast.py` does not exist in git (the cluster has untracked `scripts/test_samplers.py`, `test_fast.py`).
2. Testing: `pytest tests/ -q` fails; "`ruff format . && ruff check .` (pre-commit runs both)": 107 files would be reformatted and 1,146 lint errors exist (939 in `scripts/analysis`); pre-commit only checks changed files and pins ruff v0.11.0 while the venv has 0.15.1.
3. "Where things live": `albench/registry.py` ("the strategy table") and `albench/paths.py` are 16-line shims; the real files are `albench/core/registry.py` and `albench/core/paths.py`. "Adding a reservoir strategy, step 2: one `register(Spec(...))` call in `albench/registry.py`" would edit a shim that nothing reads.
4. "No absolute paths in `albench/`": violated by `arch_search/arch_autoresearch.py:25` and `arch_search/bridge.py:27`.
5. "The student input standard ... the library, the retraining script and the AutoResearch loop take them from there": the tracked retraining script and the tracked launcher have zero references to `harness_standard`.
6. "K = 21 / 11 / 12 at 10k / 30k / 100k" vs `docs/MENU.md` (menu v1c: all 20 at 10k/30k/100k, 12 from 300k) and `per_size_v1c.json`.
7. Missing from the file: the dispatcher and `scratchpad/`, the tuned-LegNet bank script (only in `docs/TUNED_LEGNET.md`), the loop launcher, the shared lab folder, the `study` layout.

**`docs/REPRODUCE.md`**: "Two arms are live" (the 210-cell human screen, the yeast DREAM-RNN oracle); `yeast_oracle.sh` calls missing `experiments/train_yeast_oracle_v2.py`. **`docs/ATTIC.md`**: file counts (57 / 8 / 28) and "root `run_*.sbatch`, watchdogs" are no longer true. **`docs/STUDY_LOG_OCT29_NOV2.md`**: cites `scratchpad/oct28/PLAN.md` (exists only in a laptop session directory) and `scratchpad/dispatch_registry.json` (not in git). **`docs/DATA_ARCHIVE.md`**: cites `scripts/generate_corrected_negatives.py` (untracked). **`docs/STRUCTURE.md`**: proposal; status paragraph is accurate (steps 1-2 only) - label it "design note, not implemented". **`albench/README.md`**: standalone-package instructions, placeholder URL.

### 2.6 `pyproject.toml`, entry points, dependencies, tooling

- Console script `albench = albench.cli:main` is the only entry point and works. `python -m ar2` is a second (undeclared) entry point.
- Dead dependencies (never imported in tracked code): `hydra-core`, `python-dotenv`, `biopython`, `imageio`; `omegaconf` only in the 3 stale tests. Imported but undeclared: `optuna`, `ray` (HP bake-off), `h5py`, `tqdm` (declared), `matplotlib` is declared and used only by analysis. `scripts/setup_env.sh`/`requirements/menu.txt` pin a different set (torch 2.10, numpy 2.4, sklearn 1.8) - say which one a newcomer should use.
- `[tool.hatch.build.targets.wheel] packages = ["albench","data","models","evaluation","experiments","scripts"]` would package the 5.9 GB `data/` tree and `scripts` (no `__init__.py`).
- `[tool.ruff.lint.per-file-ignores] "custom_model.py"`, `pyrightconfig.json` entries `eval_ag.py`, `test_head.py`, `custom_model.py`: files do not exist.
- Local venv is Python 3.12.7; `requirements/menu.txt` says 3.11; pyproject `>=3.11`; the cluster venv is 3.11.7.
- `ruff`: `albench` 9 errors (all in the dirty launcher), `ar2`/`data`/`evaluation`/`val_variants`/`pipelines` clean, `scripts/analysis` 939, `scripts/program_oct22` 177 (E701/E702 one-liners, I001, F405 star imports).

### 2.7 Documentation a newcomer needs and cannot find

| missing | content (what to write) |
|---|---|
| environment on laptop vs cluster | one page: laptop = `uv sync --extra dev` or `setup_env.sh` (Python 3.11, Muon from GitHub), what works without data (`albench list/generate/menu list`, smoke tests), what needs a GPU, the oracle, the lab share; cluster = `/grid/.../ALBench-S2F/.venv`, `uv run --no-sync`, never `uv pip install` from jobs, `module load`, login nodes bamdev1-4 |
| how to get data | generated table from `paths.ASSETS` + the shared-folder map (`cp -r /grid/koo/home/shared/albench/data ...`), V0 vs V0c naming, `k562_nested` vs `k562_reps` vs legacy `k562_cumulative`, how to build a pool from scratch |
| smoke test | 3 commands: `albench doctor`; `albench generate --strategy random --n 1000 --out /tmp/x.npz`; the 5-second pytest subset (proposed `albench smoke` or a `make smoke`) |
| add a student / architecture | menu JSON schema (`source` = `class Net(nn.Module)`, `train_config`), `build_and_check` limits (imports, <=40M params, CPU forward), input must be `INPUT_LEN` = 240 and RC-averaged, how to add an entry to `menus/`, `albench menu train/stack`, how to compare with the 20-seed LegNet bank (`analyze_legnet_recipes.py`), rule that AlphaGenome students are baselines |
| launch a cell through the dispatcher | template `register_*.py` (fields `name, gpus, ext, out_dir, env, needs, cmd, done`), `--dry`, flock + `.bak`, gates (`scratchpad/gates/<n>_go`), `defer_prefixes.txt`, `dispatch_STOP`, status (`outputs/dispatch/status.txt`, `status_oct26.py`), LLM cells need `LLM_MODEL` and `~/.config/albench/llm.env`, exit code 75 = rate limit, never edit the registry by hand, never rewrite finished cells |
| experiment index | the prefix table of outline section 6, kept next to `docs/STUDY_LOG` and generated from the `register_*` docstrings |
| layout map | what `study/` is, what is frozen, which scripts are safe to run |

### 2.8 Cluster clutter (all verified untracked; none referenced by the registry or dispatcher)

Repo root: 147 loose files (108 `.sbatch`, 23 `.py`, 10 `.sh`, 4 `slurm-*.out`, 1 `.yaml`, 1 `.bak`; mtimes: Feb 17, Mar 1, Jun 33, Jul 10, Aug 8, Sep 54, Oct 37, newest Oct 15). `scripts/slurm/` 129 untracked (103 from May). `scripts/` 85 loose legacy scripts. Directories: `analysis/`, `ar2/` (stale), `logs/` 947 MB, `tmp_sync_bundle/`, `deprecated_hashfrag_archive/`, `checkpoints/`, `wandb/` 505 MB, `experiments/archive/`, `scripts/{colab,preflight,scaling_ensemble,scaling_hp_search,_inference_tmp}/`, 12 `data/*` derived dirs. `scratchpad/`: 60 `dispatch_registry*.bak*` (about 100 MB), 18 `dispatcher*.bak*`, 22 `arch_autoresearch.py.bak_*` / `menu_retrain.py.bak_*`, 16 `*_cells/` directories of pre-dispatcher sbatch files, 4,855 generated `dispatch_jobs/*.sbatch`, 36 logs.

---------------------------------------------------------------------------------------------------

## 3. Steps, in priority order (all commands are proposals; run on a branch)

### P0. Restore point and branch (risk: none)

```bash
cd /Users/christen/Downloads/ALBench-S2F
git status --short                                   # commit or stash other people's WIP first (launcher, TUNED_LEGNET.md, register_lr39.py)
git tag archive/pre-cleanup-2026-11-02 HEAD
git push origin archive/pre-cleanup-2026-11-02        # any deleted file: git checkout archive/pre-cleanup-2026-11-02 -- <path>
git switch -c cleanup/onboarding
```
After: `git tag -l 'archive/*'`.

### P1. Make git match what runs (risk: needs-check; do this first)

**P1a. Bring the live files in** (read-only `scp` from the cluster; then format and de-path them)
```bash
C=bamdev4:/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad
mkdir -p albench/dispatch && touch albench/dispatch/__init__.py
scp $C/dispatcher.py       albench/dispatch/dispatcher.py
scp $C/menu_retrain.py     scripts/program_oct22/menu_retrain.py            # replaces the stale tracked copy
git diff --stat albench/arch_search/arch_autoresearch.py                    # inspect the dirty local port first: do not overwrite it blindly
scp $C/arch_autoresearch.py /tmp/arch_autoresearch.cluster.py               # then merge by hand: cluster-only = 357 lines vs the dirty worktree
.venv/bin/ruff format albench/dispatch scripts/program_oct22/menu_retrain.py
```
Then edit `albench/dispatch/dispatcher.py`: `R = Path(os.environ.get("ALBENCH_ROOT", Path(__file__).resolve().parents[2]))`, `USER = os.environ.get("USER")`, `SB = os.environ.get("SLURM_BIN", "/cm/shared/apps/slurm/current/bin")`, `CLAUDE_LOGS = Path.home()/".claude/projects"`. Add `tests/test_dispatcher.py` for the pure functions (`parse_reset`, `choose`, `wall_and_budget`, `is_done`). In the launcher replace `REPO = "/grid/..."` by `from albench.core.paths import REPO_ROOT`.
After: `pytest tests/test_agent_loop.py tests/test_agent_coordinator.py tests/test_flank_context.py tests/test_menu.py -q`; on the cluster `python scratchpad/dispatcher.py --once --dry` still prints a cycle line.

**P1b. Cluster cutover in place** (P1a must be merged into `origin/main` first: the cluster pulls `origin/main`) (the path `/grid/wsbs/home_norepl/christen/ALBench-S2F` is baked into 2,881 `cmd` strings, so keep it; clean in place)
```bash
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
git fetch origin
A=../ALBench-S2F_archive_2026-11; mkdir -p $A/collisions
git diff > $A/cluster_local_modifications.diff                            # 4 modified tracked files
git ls-tree -r --name-only origin/main | while read f; do                  # 116 untracked copies that would block the merge
  [ -e "$f" ] && ! git ls-files --error-unmatch "$f" >/dev/null 2>&1 && { mkdir -p "$A/collisions/$(dirname "$f")"; mv "$f" "$A/collisions/$f"; }
done
git checkout -- albench/arch_search/agent_loop.py albench/arch_search/arch_autoresearch.py scripts/analysis/analyze_oct27.py scripts/program_oct22/status_oct26.py
git merge --ff-only origin/main        # adds 338 files, modifies 11, deletes 0 (computed from 9841f8b..c0eb697)
```
Risk: **needs-check**: new jobs import the newer `albench/` (`menu.py`, `flank_context.py`, `bench33_ag.py` differ today); confirm the pending cells were designed against it (they were registered by origin/main scripts). `scratchpad/` stays (add it to `.gitignore`, P4f). After: `squeue -u christen`, `tail outputs/dispatch/dispatch.log`, one pending cell's job log.

**P1c. Point the unfinished cells at tracked files** (cluster, under the registry lock; dispatcher re-reads each cycle; touches only cells that are not done, finished cells keep their historical `cmd`)
```python
import fcntl, json, os, shutil
os.chdir("/grid/wsbs/home_norepl/christen/ALBench-S2F")
with open("scratchpad/.registry.lock", "w") as lk:
    fcntl.flock(lk, fcntl.LOCK_EX)
    P = "scratchpad/dispatch_registry.json"; shutil.copy(P, P + ".bak_cleanup")
    reg = json.load(open(P)); n = 0
    for c in reg:
        d = c["done"]
        if d["type"] == "file" and os.path.exists(d["path"]): continue
        new = (c["cmd"].replace("scratchpad/arch_autoresearch.py", "albench/arch_search/arch_autoresearch.py")
                       .replace("scratchpad/menu_retrain.py", "scripts/program_oct22/menu_retrain.py"))
        n += new != c["cmd"]; c["cmd"] = new
    json.dump(reg, open(P + ".tmp", "w")); os.replace(P + ".tmp", P); print("rewrote", n)
```
Only run after the new launcher is on the cluster (P1b) and one cell has been started by hand with it (`--smoke` for the launcher, `--shard 0/1 --seeds 1` on a tiny pool for `menu_retrain.py`).

### P2. Fix the test command and delete dead tests (risk: none)
```bash
git rm tests/test_experiments/test_exp1_utils.py tests/test_experiments/test_exp2_utils.py \
       tests/test_experiments/test_exp3_utils.py tests/test_experiments/test_exp5_utils.py \
       tests/test_experiments/test_experiment_utils.py tests/test_shift_aug_adapters.py
# test_new_strategies.py::test_partial_mutagenesis...: update to the current PartialMutagenesisSampler signature or delete the test
```
After: `pytest tests -q -p no:cacheprovider` must collect everything; expected 0 collection errors, remaining failures only the 2 `tests/ar2` items (needs-check).

### P3. Delete dead code (risk: none unless stated; all verified unreferenced)
```bash
git rm scripts/analysis/greedy_deploy_select.py.bak scripts/analysis/loso_menu_fairness.py.bak140 \
       scripts/analysis/selection_regret.py.bak140 scripts/analysis/slope_variance.py.bak140   # 0 refs; first is byte-identical to its live file
git rm albench/pyproject.toml albench/README.md albench/arch_search/bridge.py
```
Needs-check group (decide first, see section 5):
```bash
git rm albench/reservoir/{activity_stratified,adaptive_selection,curriculum,diversity_guided,motif_clustering,motif_density,motif_grammar,recombination,uncertainty_guided}.py albench/acquisition/bait.py
# then edit albench/reservoir/__init__.py and albench/acquisition/__init__.py (drop the 9 + 1 imports and __all__ entries)
```
After: `.venv/bin/ruff check albench tests && pytest tests/test_registry.py tests/test_albench tests/test_experiments tests/test_acquisition.py -q && albench list | head`.

### P4. Remove machine-specific paths and the hidden dependency (risk: low)

a. **List what is left**: `git grep -lF '/grid/wsbs/home_norepl/christen/ALBench-S2F' | wc -l` (108 today).
b. **One root constant**: in `scripts/program_oct22/stage2_lib.py` replace `R = "/grid/wsbs/home_norepl/christen/ALBench-S2F"` by `R = os.environ.get("ALBENCH_ROOT", str(Path(__file__).resolve().parents[2]))`; the registrations that do `from stage2_lib import R` inherit it. Find the ~20 that define their own root: `git grep -nE '^R *= *["'"'"']/grid|sys\.path\.insert\(0, *["'"'"']/grid' scripts/program_oct22` and replace each by `from stage2_lib import R` (registrations insert `scratchpad` and the repo root into `sys.path`; after P5c the helper lives next to them, so the `sys.path.insert` lines can go).
c. **Analysis scripts** (44 files): replace `REPO = Path("/grid/...")` by `REPO = Path(os.environ.get("ALBENCH_ROOT", Path(__file__).resolve().parents[2]))`; many use `os.getcwd()` as root, which is fine if documented as "run from repo root".
d. **`scripts/setup_env.sh`**: `SHARED=${ALBENCH_SHARED:-/grid/koo/home/shared/albench}`.
e. **`external/human_legnet`**: switch `bench33_legnet.py::load_legnet` to `albench.arch_search.mpra_legnet_ref` after proving equality (same `state_dict` keys/shapes and identical forward on random input with copied weights; not measured in this audit) and record the upstream commit in a docstring; until then add `external/human_legnet/model.py` (187 lines, MIT, 6.4 KB) to git with its LICENSE under `third_party/human_legnet/`.
f. **`.gitignore`**: add `/scratchpad/`; keep the rest.
g. **`core/paths.py` fallbacks**: delete the three colleague-home `hg38` fallbacks and `/grid/koo/home/mundewadi`; keep `/grid/koo/home/shared/...`.
After: `git grep -c '/grid/wsbs' | wc -l` must be 0 outside `docs/CLUSTER.md`; `pytest` smoke subset; on the cluster `ALBENCH_ROOT` unset must still resolve (via `parents[2]`).

### P5. Reorganise what stays (risk: low, mechanical; do after P1 and after the dispatcher is idle or P1c has rewritten pending cells)

**P5a. Analysis: keep the 72 + 7, archive 92** (lists in Appendix B, C, D; `names_*.txt` = those lists, one per line)
```bash
mkdir -p study/analysis study/legacy/scripts attic/2026-11/scripts/analysis
# names_*.txt: one file name per line, copied from Appendix B, C and D
for n in $(cat names_current.txt names_ledger.txt); do git mv scripts/analysis/$n study/analysis/$n; done
for n in $(cat names_old.txt);      do cp -p scripts/analysis/$n attic/2026-11/scripts/analysis/$n && git rm -q scripts/analysis/$n; done   # on-disk copy (ignored) + history + tag
git mv scripts/analysis/eval_yeast_oracle_by_class.py scripts/analysis/verify_labels.py study/legacy/scripts/   # called by scripts/pipelines
```
**P5b. Sep-era tools to `study/legacy/`** (yeast, curves, FM, pipelines, cluster, configs, val_variants, ar2 helper scripts)
```bash
mkdir -p study/legacy/{scripts,pipelines,cluster,configs}
git mv scripts/build_yeast_folds.py scripts/build_yeast_eval_classes.py scripts/predict_yeast_eval_classes.py scripts/yeast_oracle_sanity_preview.py \
       scripts/build_curve_plan.py scripts/preflight_curves.py scripts/link_screen_pools.py scripts/pool_params_match.py scripts/emit_pool_commands.py \
       scripts/build_delta_pairs.py scripts/build_genomic_partition.py scripts/build_grid_pools.py scripts/fm_scaling_driver.py \
       scripts/build_full_oracle_cache.py scripts/relabel_battery_oraclev3_ensemble.py scripts/verify_oracle_v3_repro.py \
       experiments/train_dream_rnn_oracle.py study/legacy/scripts/
git mv scripts/pipelines/* study/legacy/pipelines/ ; git mv scripts/cluster/* study/legacy/cluster/ ; git mv scripts/slurm study/legacy/slurm
git mv configs/curves configs/screen study/legacy/configs/
git mv scripts/val_variants study/val_variants
mkdir -p ar2/scripts && git mv scripts/ar2_finish.sh scripts/ar2_launch.sh scripts/ar2_status_all.sh scripts/ar2_parity.py scripts/ar2_prepare_data.py ar2/scripts/
git grep -lE 'scripts/(val_variants|ar2_|pipelines|cluster)' | xargs sed -i '' -E 's#scripts/(val_variants)#study/\1#g; s#scripts/(ar2_)#ar2/scripts/\1#g; s#scripts/(pipelines|cluster)#study/legacy/\1#g'   # review the diff
```
**P5c. Rename the program directory** (gated: dispatcher idle or P1c done; 88 files and 120 occurrences mention the old path)
```bash
git mv scripts/program_oct22 study/program
git grep -lF 'scripts/program_oct22' | xargs sed -i '' 's#scripts/program_oct22#study/program#g'
# unfinished cells: apply the same replace to their cmd in the registry (P1c snippet)
```
**P5d. Slide figures and docs**
```bash
git mv slide_figures study/slide_figures      # (optional second step: git rm -r --cached 'study/slide_figures/**/*.png'; READMEs and data_*.json stay)
mkdir -p docs/archive docs/notes
git mv docs/frontier_session_brief.md docs/archive/
git mv docs/STUDY_LOG_OCT29_NOV2.md docs/LEGNET_VS_LOOP_AND_NEXT_ARMS.md docs/LOOP_DIVERSITY_AND_VALIDATION.md docs/per_d_procedure_audit_oct30.md docs/notes/
git grep -lE 'slide_figures|docs/(STUDY_LOG|LEGNET_VS_LOOP|LOOP_DIVERSITY|per_d_procedure)' | xargs sed -i '' -E 's#(^|[^/])slide_figures#\1study/slide_figures#g'   # review
```
After each: `git grep -nE 'scripts/analysis|scripts/program_oct22|slide_figures' -- ':!study'` should only show intended text; `.venv/bin/ruff check . && pytest <smoke subset> -q`; `python study/analysis/analyze_legnet_recipes.py --help`.

### P6. Documentation (risk: none)

1. `README.md`: 60 lines max: what the project is (3 sentences from `AGENTS.md`), `albench doctor/list/generate`, smoke test, links. Delete sections "Running Experiments", "Data Download", "Standardized Result Aggregation", "W&B", old "Project Layout".
2. `git rm ONBOARDING.md` after folding two facts into `AGENTS.md` (oracle_v3 provenance stamp; cluster QoS rules). `git mv ARCHITECTURE.md docs/archive/ARCHITECTURE_2026-09.md` (or replace with a one-page layout map).
3. `AGENTS.md`: fix the 7 statements of 2.5; add a "study/ and the cluster" section.
4. New: `docs/SETUP.md`, `docs/DATA.md`, `docs/ADD_A_STUDENT.md`, `docs/RUN_ON_CLUSTER.md`, `docs/EXPERIMENTS.md` (prefix table), `docs/CLUSTER.md` (secret-free replacement for `REMOTE_ACCESS.md`); content in 2.7. Add `docs/README.md` index; label `STRUCTURE.md` "design note, not implemented"; update `REPRODUCE.md`/`ATTIC.md` or move to `docs/archive/`.
5. Optional generator: `scripts/make_experiment_index.py` that prints the prefix table from the docstrings of `register_*.py`.

### P7. Tooling and metadata (risk: low)

- `pyproject.toml`: drop `hydra-core`, `python-dotenv`, `biopython`, `imageio` (and `omegaconf` after P2); add `optuna`/`ray` as an `hpsearch` extra or delete the HP bake-off; replace the wheel `packages` list by `["albench"]` (**needs-check**: confirm `uv sync` still makes `experiments`, `models`, `data` importable from the repo root, otherwise keep them but not `scripts`/`data` trees).
- `pyrightconfig.json`: remove `eval_ag.py`, `test_head.py`, `custom_model.py`; drop the matching ruff per-file-ignore.
- Ruff: pin one version in both `.pre-commit-config.yaml` and `requirements` (venv 0.15.1 vs hook 0.11.0); `[tool.ruff] extend-exclude = ["study", "attic"]` (or per-file-ignores E701/E702/E401 for `study/**`) and one style-only commit running `ruff format` on the 18 remaining non-study files (albench 2, models 1, experiments 4, tests 2, ...). `ruff check albench tests experiments models` then passes.
- Retire the shims: `git grep -lE 'albench\.(paths|pools|registry|run|loop)\b' -- '*.py' | xargs perl -pi -e 's/albench\.(paths|pools|registry|run|loop)\b/albench.core.$1/g'`, keep `tests/test_albench/test_migration_shims.py` until the shims are deleted (then delete both). Needs-check: `albench/__init__.py` re-exports from `albench.loop`.
- Add a Makefile or `albench smoke` printing the 3 smoke commands.
After: `uv sync --extra dev && albench doctor && pytest -q` in a fresh clone (`git clone --no-local . /tmp/fresh`).

### P8. Cluster tidy (risk: none for moves into an archive directory; nothing is deleted). Run after P1b.
```bash
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
A=../ALBench-S2F_archive_2026-11; mkdir -p $A/root $A/scripts_untracked $A/dirs $A/scratchpad/{registry_backups,code_backups,cells}
# 1. root clutter: every untracked FILE at the repo root (147)
git ls-files --others --exclude-standard | grep -v '/' > /tmp/root_untracked.txt
for f in $(cat /tmp/root_untracked.txt); do grep -qF "$f" scratchpad/dispatch_registry.json scratchpad/dispatcher.py && echo "REFERENCED: $f"; done   # expect no output
xargs -a /tmp/root_untracked.txt -I{} mv {} $A/root/
# 2. untracked files below scripts/ (129 in slurm/, 85 loose, ...); program_oct22 is handled by P1b
git ls-files --others --exclude-standard scripts | grep -v '^scripts/program_oct22/' | while read f; do
  mkdir -p "$A/scripts_untracked/$(dirname "${f#scripts/}")"; mv "$f" "$A/scripts_untracked/${f#scripts/}"; done
# 3. untracked directories (checkpoints/ is a symlink to the AlphaGenome weights: leave it)
mv analysis tmp_sync_bundle deprecated_hashfrag_archive experiments/archive $A/dirs/
# 4. scratchpad: keep dispatcher.py dispatch_registry.json .registry.lock stage2*.py ens_util.py menu_retrain.py arch_autoresearch.py agent_loop.py gates/ defer_prefixes.txt mig_cap.txt dispatch_jobs/ outputs/
mv scratchpad/dispatch_registry.*bak* scratchpad/dispatch_registry.json.empty_nov1_2310 $A/scratchpad/registry_backups/   # keep dispatch_registry.json.bak_cleanup and the 3 newest until the cutover is verified
mv scratchpad/*.py.bak* scratchpad/*.bak_* scratchpad/dispatcher.bak_*.py scratchpad/stage2*.bak_*.py $A/scratchpad/code_backups/
mv scratchpad/*_cells scratchpad/*_cells_* $A/scratchpad/cells/
```
After: `python3 scratchpad/dispatcher.py --once --dry | tail -3`; `squeue -u christen | wc -l` unchanged; the next dispatcher cycle logs `cycle: {...}` without `cycle error`. `logs/` (947 MB) and `wandb/` (505 MB): archive to the lab share or delete after the owner agrees.

### P9. Local git hygiene (risk: none for the remote; local reflog loses history)
```bash
git stash list                       # 3 Aug-era stashes (fm-*, pre-reconcile WIP): inspect, then drop or archive as patches
git stash show -p stash@{2} > /tmp/stash2.patch
git reflog expire --expire=now --all && git gc --prune=now      # .git 268 MB -> about 40 MB
```
Also delete local-only `boda2-main/`, `results/`, `wandb/`, the 366 MB `external/*` copies if the owner confirms they are scratch.

---------------------------------------------------------------------------------------------------

## 4. Risk and verification matrix

| step | risk | after, run |
|---|---|---|
| P0 | none | `git tag -l` |
| P1a | needs-check | `pytest tests/test_agent_loop.py tests/test_agent_coordinator.py tests/test_flank_context.py tests/test_menu.py -q`; dispatcher `--once --dry` |
| P1b | needs-check | `albench doctor` on the cluster; `squeue`; first resumed job log |
| P1c | needs-check | dispatcher log one cycle; a pending cell's sbatch `cmd` points at an existing file (`test -f`) |
| P2 | none | `pytest tests -q` (0 collection errors) |
| P3 (first group) | none | `ruff check`, smoke subset, `albench list` |
| P3 (orphan samplers) | needs-check | `pytest tests/test_experiments tests/test_registry.py -q`; `albench list` unchanged (17 + 12 strategies) |
| P4 | low | `git grep '/grid/wsbs'` = 0 outside docs; smoke subset; one `register_*.py --dry` on the cluster |
| P5 | low | `git grep` for old paths; `ruff check .`; `--help` of 3 moved scripts; compare `git diff --stat -M` shows renames only |
| P6 | none | read-through; every command in README runs on a clean clone |
| P7 | low | fresh-clone `uv sync`, `albench doctor`, `pytest` smoke |
| P8 | none | dispatcher status; no change in `squeue` |
| P9 | none (local) | `git count-objects -vH`; `git log --oneline | wc -l` unchanged |

---------------------------------------------------------------------------------------------------

## 5. Not sure these are dead (need a human decision)

1. **`ar2/` + `tests/ar2/` + 5 `scripts/ar2_*`** (13k LOC): finished, undocumented; keep as `ar2/` marked experimental or archive? Two failing tests.
2. **The 9 unregistered samplers and `bait.py`**: dead as code paths, but they are the ideas catalogue of the frontier brief (e.g. `recombination`); archive in the tag rather than delete if in doubt.
3. **`albench/arch_search/gh_bridge.py`** and the `LLM_USE_BRIDGE` branch in `experiments/llm_autoresearch.py`: no cell uses them, but the HPC CLI path was org-blocked at one point (memory note); keep until the token path is guaranteed.
4. **Yeast stack** (`models/dream_*`, `data/yeast.py`, `scripts/build_yeast_*`, `experiments/train_dream_rnn_oracle.py`, yeast pipeline): the yeast arm is part of the registered-report plan (brief) but nothing runs now and the pipeline is already broken.
5. **`experiments/exp1_1_scaling.py` (3.5k lines)**: legacy driver but the only home of the oracle loader; extract `_load_oracle`/`_load_k562_ag_s2_oracle` (about 600 lines) into `albench/oracle.py` before archiving the rest.
6. **`experiments/{hp_strategies,hp_knowledge_base,ray_tune_search,llm_autoresearch,scaling_hp_search}.py`**: HP bake-off, but `scaling_hp_search` is imported by the live launcher and `propose.py` imports `_call_claude_cli` from `llm_autoresearch`; extract those first.
7. **`scripts/program_oct22/patch_*.py` (18)**: their effect is baked into the cluster launcher; after P1 they are pure provenance. Keep in `study/` for one release.
8. **Stale registry cells**: `hab2_*` (9, never finished, wait on `harness_ab.py`), `rr38_30k_r3`, `rr38_100k_*` (superseded by `rr39`?), `pa38_*_s1`: close or delete only with the owner.
9. **`slide_figures/`**: PNGs are regenerable but the JSON beside them is the only stored output of several analyses; untrack PNGs only.
10. **`scripts/make_shared_release.py` KEEP lists** encode the author's own "what a collaborator needs"; after P5 its paths (`scripts/build_pool.py`, tests list) must still match, and `docs/STRUCTURE.md` must be added/removed consistently.
11. **Local `external/`, `results/`, `boda2-main/`** are untracked scratch; confirm nothing off-repo depends on them.
12. **`data/` naming** (code and 5.9 GB of data share a directory): rename to `datasets/` touches ~25 imports; do it only with a compat shim.
13. **Old menus**: `menu_v0.json` (needed to load the saved 10k/30k ensembles in the shared folder) and `menu_v1.json` (identical to v1c at 200 bp, cited by cells) must stay.
14. **Cluster `data/*` derived directories** listed in `docs/DATA_ARCHIVE.md` as non-reproducible: archive, never delete.
15. **`/grid/koo/home/shared/albench/ALBench-S2F`** (collaborator snapshot of Oct 29, commit `fa7b7d2`): will be stale after P5; rebuild with `make_shared_release.py` once the cleanup lands.

---------------------------------------------------------------------------------------------------

## 6. Verification log (what was actually run)

- Local: `git ls-files`, `git ls-tree -r -l HEAD`, `git log -1 --format=%cs -- <path>`, `git grep` for paths/names/secrets, `git rev-list --objects --disk-usage`, AST import graph (`audit/tmp/refgraph.py` -> `refgraph.json`), `albench doctor`, `albench list`, `albench generate --strategy random --n 1000` (works), `pytest tests -q` (6 collection errors; with those ignored: 481 passed, 3 failed, 2 skipped in 208 s), smoke subset (138 tests, 5 s), `ruff check .` (1,146 errors), `ruff format --check .` (107 files).
- Cluster (read-only `ssh -o BatchMode=yes bamdev4`): `git status`, `git rev-list`, `ls`/`find`/`du` of scratchpad and root, md5 of 112 scratchpad scripts vs tracked, md5 of 21 key repo files vs HEAD, registry JSON statistics (2,881 cells, entry points, prefix counts, unfinished cells), `crontab -l`, `ps`, listing of `/grid/koo/home/shared/albench` and `data/`, tail of `outputs/dispatch/dispatch.log`.
- Not measured: numerical equivalence of the three LegNet implementations; whether the 2 failing ar2 tests are environmental; whether pending cells run identically under the newer `albench/` library.

---------------------------------------------------------------------------------------------------

## Appendix A. Files proposed for `git rm` (verified unreferenced unless marked)

```
scripts/analysis/greedy_deploy_select.py.bak            (byte-identical to the live script)
scripts/analysis/loso_menu_fairness.py.bak140
scripts/analysis/selection_regret.py.bak140
scripts/analysis/slope_variance.py.bak140
tests/test_experiments/test_exp1_utils.py
tests/test_experiments/test_exp2_utils.py
tests/test_experiments/test_exp3_utils.py
tests/test_experiments/test_exp5_utils.py
tests/test_experiments/test_experiment_utils.py
tests/test_shift_aug_adapters.py
albench/pyproject.toml
albench/README.md                                        (rewrite as 10 lines if a package README is wanted)
albench/arch_search/bridge.py
needs-check: albench/reservoir/{activity_stratified,adaptive_selection,curriculum,diversity_guided,motif_clustering,motif_density,motif_grammar,recombination,uncertainty_guided}.py
needs-check: albench/acquisition/bait.py
needs-check: experiments/deploy_train.py                 (0 importers, 0 mentions; yeast `train_dream_rnn_oracle.py` goes to study/legacy with the yeast stack)
```

## Appendix B. `scripts/analysis` kept as current-study (72) -> `study/analysis/`

```
ag_check.py
analyze_baselines_allsets.py
analyze_bench33.py
analyze_best30k.py
analyze_best30k_pools.py
analyze_cadence_budget.py
analyze_combined_ensembles.py
analyze_context_allsets.py
analyze_div34.py
analyze_extras38.py
analyze_kcurve_spread.py
analyze_legnet_recipes.py
analyze_loop_proposals.py
analyze_menu_vs_legnet.py
analyze_mgr34.py
analyze_oct22.py
analyze_oct25.py
analyze_oct26.py
analyze_oct27.py
analyze_oct30.py
analyze_oct31.py
analyze_oct32.py
analyze_patience.py
analyze_pi38.py
analyze_pk35.py
analyze_pool_variants.py
analyze_rankD_dedup.py
analyze_recipe_choices.py
analyze_reflect.py
analyze_reservoir_distributions.py
analyze_rounds.py
analyze_runs_budget.py
analyze_scaling_law.py
analyze_scaling_trend.py
analyze_seed_scaling.py
analyze_st38.py
analyze_sw38.py
analyze_val_compositions.py
build_ledger_deck.py
build_menu_perD.py
build_menu_pi.py
build_menu_v1.py
build_menu_v2.py
build_prelim_artifact.py
build_walkthrough_artifact.py
cadence_results.py
compare_legnet_vs_loop.py
compare_pi_30k.py
compile_nested_genomic_error_metrics.py
compile_oct14_error_metrics.py
compile_oct14_scaling.py
eval_matrix.py
extract_loop_models.py
figs_oct28.py
hp_protocol_study.py
make_architecture_diagrams.py
make_bench33_figures.py
make_deck_figures.py
make_justification.py
make_lab_meeting_deck.py
make_oct14_scaling_package.py
make_one_slide_diagram.py
make_per_size.py
make_reservoir_figures.py
make_slide_figures.py
measured_singles.py
plot_scaling_error_metrics.py
plot_scaling_latest_nested.py
precision_speed.py
scaling_exponents.py
transplant_recipe.py
zoonomia_n_thresholds.py
```

## Appendix C. `scripts/analysis` Evidence-Ledger builder kept (7) -> `study/analysis/`

```
build_justification_html.py
ledger_oct26.py
ledger_oct27.py
make_ladder_context.py
patch_builder6.py
patch_builder7.py
program_status.py
```

## Appendix D. `scripts/analysis` archived (92): Sep to Oct 26 analyses, 0 references outside themselves

```
a3_ensemble_compare.py
accessibility_subset.py
ag_head_provenance.py
agarwal_gosai_mapping.py
agarwal_gosai_v2.py
aggregate_debias_screen.py
analyze_anchor_effect.py
analyze_oct23.py
analyze_oct25c.py
analyze_oct28.py
baseline_six_panel_eval.py
build_family_slide_tables.py
characterize_strategies.py
determine_deploy_recipe.py
ensemble_predictions.py
eval_composition_split.py
extended_overlap.py
figs_oct26.py
figs_oct27.py
find_overlap_agarwal_gosai.py
fold_summary.py
full_debias_scoreboard.py
gc_bias_ag_hepg2.py
gc_bias_ag_sknsh.py
gc_bias_oracle.py
gc_bias_sknsh_torch.py
gc_mechanism_hepg2.py
genomic_shuffle_cpg.py
genomic_three_metrics.py
greedy_deploy_select.py
greedy_forward_curve.py
hp_audit_a1_a3.py
hp_audit_figs_v2.py
hp_axis_variance.py
hp_budget_curve.py
hp_convergence.py
hp_coords_combined.py
hp_coords_subsets.py
hp_coords_v3.py
hp_process_figure.py
ism_highgc_vs_genomic.py
ism_localization_control.py
jaspar_clusters.py
loso_menu_fairness.py
make_meeting_figs.py
make_oracle_table_figure.py
morning_followups.py
motif_sampling_audit.py
motif_vocab_diag.py
multicell_label_diagnostics.py
multiseed_init_analysis.py
new_axes_effect.py
new_vs_old_d30k.py
ood_by_reservoir.py
oracle_fold_disagreement.py
oracle_oof_foldmap.py
overnight_cpu_pack.py
per_sequence_eval.py
pi_coords_v2.py
pi_coords_variants.py
pi_dsweep_compare.py
pi_dxr_compare.py
pi_update_coords.py
plot_bar_4panel_v2.py
plot_reservoir_curves.py
plot_scaling_results.py
plot_stage4_curves.py
poolcap_bias_seed.py
rank_cells.py
rank_stability.py
replicate_ceiling.py
report_oracle_v2.py
rounds_to_converge.py
run_twe_300k.py
sahu_curve_analysis.py
sahu_hepg2_motif_compare.py
sahu_motif_analysis.py
saturation_curve.py
scaling_v2_curves.py
scaling_v2_figures.py
selection_regret.py
six_panel_distributions.py
slope_variance.py
snv_ceiling_v3.py
snv_pairing_fix.py
stage4_curves.py
strategy_by_reservoir.py
time_adjusted_composition.py
top_models_hp_dump.py
train_legnet_percell.py
wd_ablation.py
yeast_overlap.py
```

## Appendix E. Pipeline-bound (2) -> `study/legacy/scripts/`

`eval_yeast_oracle_by_class.py` (called by `yeast_oracle.sh`), `verify_labels.py` (called by `curves_overnight.sh`).
