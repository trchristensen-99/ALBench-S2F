# Silent loss, routing walls and a silent-failure review (2026-11-10)

All numbers below were measured on bamdev4 on 2026-11-10 by read-only scans of `outputs/`. The scripts
were run with `nice` on the login node. The per-directory data is in
[`silent_loss_triage_2026-11-10.json`](silent_loss_triage_2026-11-10.json).

## 1. Fix: a menu shard is complete only when nothing is left to try

`orchestration/menu_retrain.py` used to write `COMPLETE_{k}of{n}` whenever a shard exited. Each failure
is now classified as one of three kinds:

| kind | examples | what happens |
|---|---|---|
| filtered | `forward too slow` (`MENU_MAX_FWD_MS`), `too many parameters` | deliberate gate: not retried, not a loss |
| final | the source does not build, aux head failure, exploded seeded predictions | a loss at once (a retry repeats it) |
| retry | CUDA OOM, non-finite loss, other training errors | no marker, so the dispatcher resubmits, up to `--max_attempts` (default 2) |

The attempt count comes from `failed.jsonl`, so the retry bound holds across jobs and cannot loop. A
finished shard with losses writes `DEGRADED_{k}of{n}.json` (each lost model with its reason and attempt
count), then `COMPLETE_{k}of{n}`. The dispatcher's `is_done()` is unchanged, so every existing marker
still reads as done. `is_degraded()` adds a `degraded` count to the cycle line and `DEGRADED` to
`status.txt`.

To re-run after fixing an entry, add `--retry_failed TAG` to the command of a new cell with the same
`out_dir`, and set that cell's `done` path to `RETRY_{TAG}_{k}of{n}`. The reset is logged once per tag,
so a resubmitted re-run continues its count instead of starting over.

The resume tests also found a second bug. menu_retrain's legacy-directory test read `A or B and C`,
which Python evaluates as `A or (B and C)`. Once any model existed, a resumed job of a cell using the
default flank context took the no-context path and died on the override guard. That happened in 893 job
logs on 2026-11-08, across 10 cells. Retries depend on resumed jobs working, so this is fixed too.

The 10 cells were `sc31_zoonomia_orthologs_100k_s{2,3,5}`, `sc31_zoonomia_orthologs_30k_s3`,
`un10_zoonomia_orthologs_r{1,2}_s2`, `un30_zoonomia_orthologs_r3_s{1,2}` and `un30_zoov2_r1_s{0,1}`.
Each burned 84 to 97 jobs between 13:06 and 18:54; every job FAILED in 17 to 24 s (exit 1). The loop
ended when they were paused by hand (`paused_why` in `state.json`). **No progress was lost in current
results**: all 10 are now done, and every model in their shards was written on Nov 9, 11:11 to 12:47,
after the directories were restarted, in single jobs with no resume.

The dispatcher's automatic pause (3 consecutive jobs that end in under 5 min) **did not fire** for any
of them: `short_fails` is 0 and no short-failure line was logged, although it fired for other cells
in the same hours. `check_finished()` is the same as on Nov 3. I did not find the cause. Candidates are
a second writer of `state.json`, or `checked`/`last_job` being overwritten. Until it is explained, the
pause should not be relied on as the backstop for a resubmit loop.

## 2. Triage of directories marked complete with models missing

The audit's 288 counted every directory with a `failed.jsonl`. Counting only models that are **still
missing** (no `_meta.json`) leaves **226 directories**, missing 534 models:

| cause | models | nature |
|---|---|---|
| shape mismatch at build | 303 | the architecture does not run at the 240-bp context input (`Sizes of tensors must match`, `mat1 and mat2 shapes cannot be multiplied`) |
| forward too slow / too many params | 73 | deliberate gate |
| exploded predictions | 66 | deliberate rejection (one would poison the ensemble) |
| other build failure | 43 | e.g. `banned name: getattr` |
| CUDA OOM | 27 | slice memory |
| non-finite loss and other training errors | 22 | |

The "forward too slow" gate is **not** most of it. The largest cause is length-dependent menu entries
that cannot run at 240 bp. That is deterministic and systematic, and it drops the same entries from
every context run of `sw38_`, `un10k_`, `un30k_`, `fr35_`, `fx35_`, `fx36_`, `mt38_`, `sc31_`, `fe35_`
and `al31_`.

**Current results: real losses (4 directories, 44 models)**

| directory | trained | missing | cause |
|---|---|---|---|
| `refattn32_30k_genomic_can` | 0 | 18 | OOM (every model) |
| `refbank_30k_genomic_can` | 33 | 21 | 18 `banned name: getattr` (Malinois), 3 non-finite loss |
| `archinvar_30k_motif_shared_core` | 6 | 3 | OOM |
| `archinvar_30k_accessible_tiles_k562` | 7 | 2 | OOM |

**Historical: real losses (131 directories)**, by family and missing models: `sw38_` 80 dirs (80
shape); `un30k_` 6 (54 shape); `un10k_` 6 (54 shape); `fr35_` 10 (40 shape, 10 build); `fx35_` 9 (36
shape, 6 build, 1 OOM); `mrr_` 4 (16 training errors); `fx36_` 3 (12 shape, 3 build); `mt38_` 3 (12
shape, 3 build); `sc31_` 3 (9 shape, 2 build); `fe35_` 1; `tr38_` 2 (3 OOM); `al31_` 1; `m3_` 2;
`prec30k_` 1.

**Historical: designed filters only (91 directories)**: 71 forward-too-slow or parameter-cap models and
66 exploded-prediction models. Nothing to re-run.

No `refrnn*`, `refmal*`, `m3b_`, `m3c_` or `bench38_ln_*` directory is missing a model.

## 3. RETRAIN_PFX: why the 42 unmatched families were not added

> **Correction.** The first version of this section used a slice's ×8 cap multiplier, but MIG jobs never
> get these walls: `sbatch_text` calls `wall_and_budget(c, "mig", ...)`, which returns `WALL["mig"]`
> (36 h). On `slow_nice` the slowest class is a V100 (×5). The conclusion holds when measured per cell
> and per GPU class. The fix, a wall derived from each cell's per-model cap, is on branch
> `wall-from-cap`: models that do not fit their wall go from 22 to 0 of 3,168 (cell, GPU) pairs.

Each retrain family whose name matches no rule gets the 2-day `slow_nice` wall, and adding it to
`RETRAIN_PFX` would give it 3:30. For each such family I compared the longest single model measured in
its `*_meta.json` files, and the per-model cap on its slowest allowed GPU (`--base_cap_min` × 5 on a
V100, × 8 on a MIG slice), with a 3:30 wall less 15 minutes for loading:

* 13 families already have a measured model longer than that, for example `hp38_` 490 min, `pi38_` 459
  and `al31_` 367 (all on slices), and `refrnnrecipe_` 341 on a V100.
* 28 more have a per-model cap above it.
* Only `yash300_` fits.

So the 2-day default is what lets these families' long models finish, and adding them would make each
such model restart at every job. The same measurement on the 36 families that **are** in `RETRAIN_PFX`
finds 12 with a measured model longer than their 3:30 wall: `sw38_` 427 min, `ind30m_` 284, `nest_` 276,
`mr2_` 249 and others. Those models finished on `default` (12 h) or `fast`. On `slow_nice` they are
killed at 3:30 and start over on the next job.

**Proposed instead (not applied):** derive the `slow_nice` wall from the cell, not from its name:
`wall = per-model cap on the assigned GPU + load margin`, rounded up. This needs `wall_and_budget()` to
read `--base_cap_min` from `cmd` and to know the GPU class. It changes where live work lands, so it
should be its own change.

## 4. Review: failures that produce a plausible result

A separate reviewer read `albench/` and `orchestration/` for this one class of bug. I checked the items
below against the code or the data myself. None is fixed here: each changes acquisition, sampler or
prompt behaviour, so the call belongs to whoever owns those experiments.

1. **Binned BatchBALD fills the batch in index order once it is exhausted** (`albench/acquisition/binned.py:221-225`).
   The gain is the joint entropy `H(conf, y_j)`. Ranking by it is fine, because `H(conf)` is the same
   for every candidate, but the exhaustion test `gains <= 1e-12` can never fire once `H(conf) > 0`.
   When all T members are in distinct configurations, every gain equals `log T`, and `argmax` returns
   the lowest remaining index: about 375 of 384 picks with 10 members, with no warning. Fix: test
   `gains - H(conf)`.
2. **One exploding member makes ensemble-variance acquisition return the last k candidates.**
   `MenuEnsemble._predict_member` returns all-NaN for an exploded member (`menu.py:966`).
   `EnsembleAcquisition` then takes `np.var` over members, which is NaN everywhere, under
   `errstate(all="ignore")`, and `argsort(...)[-k:]` returns the last k indices. BatchBALD warns in the
   same case but still returns a full batch. Fix: drop non-finite members before stacking.
3. **`--guidance none` drops `--ensemble_aware`** (`arch_autoresearch.py:648`, which returns before line
   690). 18 registry cells are affected (`xd30_gn_{rnd,mot,gen10}_*`). The no-guidance ablation
   therefore also removed ensemble-awareness.
4. **AutoResearch writes `COMPLETE` when its rounds run out short of target**
   (`arch_autoresearch.py:1209`, condition `not budget_hit and midx > 0`). This is the same class as the
   menu bug, but `midx >= target` would be the naive fix (resubmitting forever). It needs the same
   terminal-marker-plus-record treatment.
5. **Muon falls back to AdamW with only a print** (`arch_autoresearch.py:273`). `menu_retrain` and the
   loop call `train_one` directly, so they bypass the library's `_require_optimizer` guard. *Downgraded
   after review:* `build_opt` returns the optimizer actually used, and `meta["optimizer_used"]` records
   it, so every past run can be checked from its artefact. This is hygiene, not a results problem. The
   one gap is an unrecognised name (not `muon`/`adam`/`adamw`): it trains AdamW but is returned, and
   recorded, under its original name.
6. **The Zoonomia reservoir breaks the chromosome holdout rule.** The rate file excludes only chr7 and
   chr13, so 14,702 of its 199,373 regions (7.4%) are on chr19, chr21 or chrX, and
   `PhylogeneticZoonomiaSampler` applies no filter by default. Measured on the current nested
   `zoonomia_d300000` pool: 6.9% of 20,000 sampled sequences share ≥20 exact 40-mers with a chr19/21/X
   region, against 1.7% for `genomic` (background from repeats). Against `val_V0c` itself, however,
   `zoonomia` overlaps *less* than `genomic` (1.2% vs 5.7% sharing ≥20 40-mers; 0.09% vs 0.37% near
   copies). So the current validation metrics are not measurably contaminated, but any chr19/21/X-based
   evaluation would be. Separately, `mut_rate or DEFAULT_RATE` turns `mut_rate=0` into 0.02.
7. **Declared parameters that are never read, or are read under another name.** `binned.py`
   `n_joint_samples` is stored and never read. With `mutation_rate_distribution="normal"`,
   `mutation_rate` is ignored in favour of the undeclared `mean_rate`/`std_rate`. The EvoAug `p_*` knobs
   are normalised before sampling, so they act as relative weights, not the probabilities their help
   strings promise.

Not verified here (reported by the reviewer, worth a look): `core/loop.py` does not de-duplicate
candidates against already-labelled sequences; `menu export` omits `per_size`, so an exported menu
trains with patience 8; `_load_pool` ignores `--task`; `menu_retrain`'s provenance and override guard
omit `AL_PRECISION`, `AL_RC_MODE`, `AL_ONECYCLE_FULL` and `AL_NONFINITE_SKIP`.
