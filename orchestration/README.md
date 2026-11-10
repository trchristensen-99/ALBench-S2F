# cluster_live: the code that actually runs on the CSHL cluster (snapshot of 2026-11-02)

These files were not under version control. The cluster checkout was 82 commits behind `origin/main` and ran from `scratchpad/`, a directory of untracked, hand-patched copies; 93% of the 2,881 registered cells run one of
the first three files below. They are copied here **unchanged** (no rename, no refactor) so that the running system is recorded and recoverable; the source of truth on the cluster is still `scratchpad/` until the sync
described in `docs/CLEANUP_PLAN.md` (step P1) has been done.

| file | role | repo counterpart (differs) |
|---|---|---|
| `dispatcher.py` | cross-queue Slurm dispatcher: reads `scratchpad/dispatch_registry.json` every cycle, submits cells as short jobs across `fast` / `default` / `slow_nice` and GPU types, handles `needs` gates, the LLM usage-limit pause (exit code 75, probe call), stalls | none |
| `arch_autoresearch.py` | the launcher every loop cell runs (own `train_one`, input standard, reference LegNet text, val cache, `AL_*` harness switches); patched in place by `scripts/program_oct22/patch_loop_*.py`, `patch_harness_*.py` | `albench/arch_search/arch_autoresearch.py` (older, other `train_one`) |
| `menu_retrain.py` | the retraining script every menu cell runs (`--flank_ctx`, `--shift_max`, `--tta`, `--aux_labels`, `--optimizer`, ...) | `scripts/program_oct22/menu_retrain.py` (197 lines behind) |
| `ens_util.py` | scale-robust `enet_all_pos` imported by `scripts/val_variants/analyze_val_variants.py` | none |
| `dispatch_registry_2026-11-02.json.gz` | the cell registry (2,881 cells: name, gpus, env, needs, cmd, done); the only record of per-cell environment | generators in `scripts/program_oct22/register_*.py` (about 200 cells come from generators that are not in git) |

Operations facts that are not in any file here:
* started by cron every 10 minutes if no dispatcher process exists and `scratchpad/dispatch_STOP` is absent, in tmux session `dispatch` with the system Anaconda python; the cron test is now `pgrep -f "[s]cratchpad/dispatcher.py"` (the old pattern matched cron's own shell and could never restart it: the dispatcher was down for 13 hours on Nov 1-2);
* gates: a cell whose `needs` lists `scratchpad/gates/<name>_go` waits until that file exists; `defer_prefixes.txt`, `mig_cap.txt`, `dispatch_STOP` are operator switches;
* LLM cells need `LLM_MODEL` in their `env` and the token in `~/.config/albench/llm.env` (not in this repository); all LLM cells share one account usage limit;
* MIG slices (`migq`, node bamgpu18) accept QOS `slow_nice`, not `mig`; the dispatcher's `mig` branch is dead (audit finding), small jobs were put there by hand.
