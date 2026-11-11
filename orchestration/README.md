# orchestration: the code that schedules and runs every cell on the CSHL cluster

This is the **source of truth** for the cluster's dispatcher and launchers. Until 2026-11-10 they existed
only as untracked, hand-patched files in the cluster's `scratchpad/`, and this folder (then
`cluster_live/`) held read-only snapshots. Registry cells invoke `scratchpad/<file>.py` **by that path**,
in thousands of `cmd` strings, so the cluster keeps those paths as symlinks into this folder:

```
scratchpad/dispatcher.py         -> ../orchestration/dispatcher.py
scratchpad/menu_retrain.py       -> ../orchestration/menu_retrain.py
scratchpad/arch_autoresearch.py  -> ../orchestration/arch_autoresearch.py
```

**The cluster is not a place to author code.** Edit here, commit, push, and `git pull` on the cluster.
Every file written only on the cluster has become a fork: the hand-patched `scratchpad/` copies, the
`albench/` working-tree edits that blocked `git pull`, and 336 untracked files sitting at paths that main
also tracks (archived 2026-11-10 in `../ALBench-S2F_archive_2026-11-10/`). `scratchpad/harness_ab.py` is
the one launcher still in that state.

**Never move or rename these files, and never replace a symlink with a copy.** A copy is exactly how the
three-way fork of `arch_autoresearch.py` happened. An edit made on the cluster through
`scratchpad/<file>.py` is an edit to the tracked file: commit it, or the next `git pull` aborts.

| file | role |
|---|---|
| `dispatcher.py` | cross-queue Slurm dispatcher, below. Runs under the system Anaconda `python3`, not the venv, so it must stay **stdlib-only**. |
| `menu_retrain.py` | trains menu entries: every reference, menu and scaling retrain cell (about 2,400). Imports `train_one` / `predict_rc` from `albench/arch_search/arch_autoresearch.py`. |
| `arch_autoresearch.py` | a launcher only. The AutoResearch loop lives in `albench/arch_search/arch_autoresearch.py` (one copy, shared with `menu_retrain.py`). Do not add code here. |
| `ens_util.py` | scale-robust `enet_all_pos`, imported by `scripts/val_variants/analyze_val_variants.py`. |
| `validate_registry.py` | checks a registry against `registry_schema.md`, plus routing and unread knobs. |
| `registry_schema.md` | the cell schema: `name`, `gpus`, `ext`, `out_dir`, `env`, `needs`, `cmd`, `done`. |
| `dispatch_registry_2026-11-02.json.gz` | dated record of the registry (2,881 cells). The live registry is not tracked (gitignored). |

Not here: `scratchpad/harness_ab.py`, which 9 `hab2_` cells invoked (all done). It is still untracked on
the cluster.

## What the dispatcher does

Every `CYCLE_S = 180` s it re-reads `scratchpad/dispatch_registry.json` and `outputs/dispatch/state.json`,
then for each cell:

1. **done?** `is_done()` checks the cell's `done` spec on disk. Done cells are skipped forever.
2. **operator switches**: names starting with a line of `scratchpad/defer_prefixes.txt` are skipped,
   and `scratchpad/mig_cap.txt` overrides `MIG_CAP`. Both are re-read every cycle with no restart.
   `scratchpad/dispatch_STOP` stops the loop and keeps cron from restarting it.
3. **last job**: `check_finished()` asks `sacct`. Exit code **75** means the LLM usage limit, not a
   failure: all LLM cells pause until the reset time parsed from the CLI's log, plus 2 min, and then a
   probe call must succeed. Any other end in < 300 s that is not `COMPLETED` counts as a short failure,
   and 3 in a row pause the cell (`state.json` `paused`; clear it by hand).
4. **running?** If a job `d_{name}` (or a name in `ext`) is queued or running, the cell is active. A
   running job with no new result for `STALL_S` (3 h H100, 6 h V100) is logged as `STALL?` and never
   killed. A job pending on `default` for longer than `PD_MIGRATE_S` (20 min) is re-routed through `choose()`
   when another queue has room.
5. **gates**: every path in `needs` must exist.
6. **submit**: `choose()` picks QOS and GPU from free GPUs and the caps below, `wall_and_budget()` picks
   the walltime and `{BUDGET}`, and `sbatch_text()` writes `scratchpad/dispatch_jobs/d_{name}_{qos}_{gpu}.sbatch`
   (venv, `PYTHONPATH=$PWD:$PWD/experiments`, the cell's `env`, then `cmd`). At most `MAX_SUBS = 24`
   submissions per cycle.

Each cycle ends with one `cycle: {...}` line in `outputs/dispatch/dispatch.log` and a refreshed
`outputs/dispatch/status.txt`. **Liveness check**: the last log line should be at most about 3 minutes
older than the cluster's `date`.

`--once` runs one cycle. `--dry` does not write `state.json` or `status.txt` and does not call `sbatch`,
but it still writes `.sbatch` files, calls `squeue` and `sacct`, and appends a `cycle:` line to
`dispatch.log`. So a dry run is not invisible, and its line is indistinguishable from a real cycle's.

## Restarting it

Cron, every 10 minutes (`crontab -l` on bamdev4):

```
*/10 * * * * [ ! -e /grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad/dispatch_STOP ] && ! pgrep -f "[s]cratchpad/dispatcher.py" >/dev/null && /usr/bin/tmux new-session -d -s dispatch "cd /grid/wsbs/home_norepl/christen/ALBench-S2F && exec /grid/it/data/elzar/easybuild/software/Anaconda3/2024.02-1/bin/python3 scratchpad/dispatcher.py >> outputs/dispatch/stdout.log 2>&1"
```

To restart by hand: `tmux kill-session -t dispatch`, then kill the dispatcher's PID **explicitly**
(`pgrep -af "[s]cratchpad/dispatcher.py"`, check the command line, `kill <pid>`), then run the tmux
command above or wait for cron. **Never `pkill -f "scratchpad/dispatcher.py"`**: the pattern also matches
the command line of the ssh or shell running the pkill. The `[s]` in cron's own test avoids the same
trap. The old unbracketed pattern matched cron's shell and kept the dispatcher down for 13 h on Nov 1-2.

## QOS caps it encodes

From `sacctmgr` and probing, and only correct for this account. Update them when the QOS changes.

| constant | value | QOS limit it mirrors |
|---|---|---|
| `FAST_CAP` | `{"h100": 2, "v100": 2}` | `fast` (4 h): 2 H100 + 2 V100 per user |
| `DEFAULT_SUBMIT_CAP` | `15` | `default` (12 h): 16 submitted per user, dependency-held jobs included; 1 kept spare. At most 8 of ours run on `default` at once. |
| `SLOW_CAP` | `{"h100": 20, "v100": 20}` | `slow_nice`: 20 + 20 GPUs per user, pending included |
| `SLOW_JOBS_PU` | `40` | `slow_nice` MaxJobsPU. **A MIG slice uses one of these 40 slots**, the same as a full GPU. |
| `MIG_CAP` / `MIGQ_CAP` | `28` / `0` | `migq` has 28 slices (1g.12gb, 10.8 GB, 3-6x slower than an H100) and accepts QOS `slow_nice`, not `mig`, so `MIGQ_CAP = 0` keeps the dead `mig` QOS branch off |
| `MAX_LLM_ACTIVE`, `LLM_COOLDOWN_S` | `40`, `1800` | all LLM cells share one account usage limit |

## The prefix tuples: read this before naming a new arm family

```python
MIG_NO      = ("ind30z_", "a600_", "a1m_", "mrr_", "pool", ..., "archinvar_", "refattn32_")   # never on a MIG slice
RETRAIN_PFX = ("mix_", "un30_", "un10_", "un100_", "as_", "ind30", ..., "mv1c_")          # slow_nice wall 3:30, fast V100 allowed
LLM_SMALL   = ("g30_", "rnd30_", "u_", "mot30_", ..., "xd10_")                            # slow_nice wall 8:00, budget 7 h
```

> **Menu-retrain walls no longer depend on these names.** Since 2026-11-10 a `menu_retrain` cell on
> `slow_nice` gets `max(3:30, per-model cap on that GPU + 45 min)`, rounded up to 30 min and capped at
> 2 days (`menu_wall()`). The per-model cap is `--base_cap_min` × 5 on a V100 and × 1 on an H100. The
> 45 min is measured: over 17,800 models, training plus evaluation overran the cap by at most 23 min,
> and job start-up has a p95 of 1.7 min. Under the name-based rule, 22 (cell, GPU) pairs had a model
> longer than their wall (it restarted at every `slow_nice` job); under the new rule there are none.
> MIG jobs (`WALL["mig"]`) and LLM cells are unchanged. `RETRAIN_PFX` still decides fast-V100
> eligibility, and `MIG_NO` still decides slices.
>
> **Warning: they match on literal cell-name prefixes (`str.startswith`).** A new arm family whose name
> appears in none of them gets the defaults, and nothing complains: a 2-day `slow_nice` wall instead of
> 3.5 h (it rarely backfills, so it waits), or, for a `menu_retrain` cell with `--D` <= 100k, a 10.8 GB
> MIG slice. The second is how DREAM-Attn cells ran out of memory until `archinvar_` and `refattn32_` were
> added to `MIG_NO` on 2026-11-10. This has bitten twice. When you register a new family, decide its row
> in each tuple, then run
> `$PY orchestration/validate_registry.py scratchpad/dispatch_registry.json --root .`, which lists
> every pending family that falls through to a default. It uses the dispatcher's own `wall_and_budget()`
> and `mig_ok()`, so it cannot disagree with them.

`wall_and_budget()` also has explicit rules for `sw38_100k` (10 h), `a1m_acq53_` (48 h), `a1m_acq5`
(20 h) and `g100x_` / `u100_` / `g300s_` / `a600_` / `a1m_` (14 h). `mig_ok()` also refuses LLM cells,
shards listed in `MIG_BAD`, and cells whose `failed.jsonl` records an out-of-memory error.

## Other operations facts

* **`python3` on the login node is 3.6** (`/usr/bin/python3`) and cannot even parse this folder. The
  dispatcher and the validator use the Anaconda 3.11 that cron uses: `PY=/grid/it/data/elzar/easybuild/software/Anaconda3/2024.02-1/bin/python3`.

* LLM cells need `LLM_MODEL` in their `env` and the token in `~/.config/albench/llm.env` (not in git).
  The probe call uses `claude -p` with `~/.local/bin` prepended to `PATH`, because cron's `PATH` lacks it.
* Gates: a cell whose `needs` lists `scratchpad/gates/<name>_go` waits until that file exists.
* Never `uv pip install` while jobs run: use `scripts/install_hpc_packages.sh` from the login node.
* No heavy Python on the login node: `OMP_NUM_THREADS=2 nice -n 19 .venv/bin/python ...` (environment
  assignment first), or sbatch it to `cpuq`.
