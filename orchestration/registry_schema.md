# Dispatch registry schema

`scratchpad/dispatch_registry.json` on the cluster is a JSON **list of cells**. It is not tracked: about
5,200 cells, 4.3 MB, rewritten many times a day by the `scripts/program_oct22/register_*.py` generators.
`dispatch_registry_2026-11-02.json.gz` in this folder is a frozen, dated record (2,881 cells), not the
live file. Check the live file with `python3 orchestration/validate_registry.py` (see below) before the
dispatcher reads it: the dispatcher re-reads it every 180 s and a malformed cell makes the whole cycle
log `cycle error` and submit nothing.

## A cell

| key | type | meaning |
|---|---|---|
| `name` | str, unique, `[A-Za-z0-9_.+-]+` | Slurm job name is `d_{name}`; `state.json` is keyed by it. **The leading characters decide routing** (see the README: `MIG_NO`, `RETRAIN_PFX`, `LLM_SMALL`). |
| `gpus` | non-empty list from `h100`, `v100` | GPU types the cell may run on. A MIG slice is decided separately (`mig_ok()`), not here. |
| `ext` | list of str | Names of jobs submitted outside the dispatcher that count as this cell running. The dispatcher only waits for them and never cancels them. Usually `[]`. |
| `out_dir` | str, relative to the repo root | Created before the command runs. `progress()` counts `r*_meta.json` here, and `mig_ok()` reads `failed.jsonl` here for earlier out-of-memory failures. |
| `env` | object str → str | Exported as `export K=V ...` before `cmd`. The only record of per-cell harness settings (`AL_*`, `LLM_MODEL`, `MENU_MAX_*`). **Values must be strings.** An `AL_*` knob that no code reads is silently ignored: the validator reports it. |
| `needs` | list of str, or `null` | Repo-relative files that must all exist before the cell is submitted (pools, upstream runs, `scratchpad/gates/<x>_go` operator gates). |
| `cmd` | str | Shell command run from the repo root inside the venv, with `PYTHONPATH="$PWD:$PWD/experiments"`. `{BUDGET}` is replaced by the per-job hour budget (`wall_and_budget()`); no other placeholder is substituted. Most cells run `scratchpad/menu_retrain.py` or `scratchpad/arch_autoresearch.py` **by that path**, so those paths must keep working. |
| `done` | object | `{"type": "file", "path": P}`: done when `P` (repo-relative) exists. `{"type": "hab_keys", "file": F, "cfg": C, "seeds": [...], "arms": [...]}`: done when JSONL file `F` has a row for every (cfg, seed, arm). Any other type is **never** done, so the cell is resubmitted forever. |

The dispatcher ignores any other key. The validator flags such keys, since they are usually typos.

## Validating

```bash
# on bamdev4, /usr/bin/python3 is 3.6 and too old; use the dispatcher's: PY=/grid/it/data/elzar/easybuild/software/Anaconda3/2024.02-1/bin/python3
# anywhere (stdlib only; checks the schema, then routing on every cell):
python3 orchestration/validate_registry.py path/to/dispatch_registry.json
# on the cluster, from the repo root (routing on pending cells only; invoked scripts must exist):
$PY orchestration/validate_registry.py scratchpad/dispatch_registry.json --root .
```

Exit status 1 means schema errors. Warnings exit 0, because some are deliberate:

* **default slow_nice wall**: a retrain cell whose name matches no rule in `wall_and_budget()` gets
  2 days. That is not wrong, but the scheduler can rarely backfill a 2-day request, so the cell waits.
* **MIG-eligible**: `mig_ok()` would put the cell on a 10.8 GB slice. That is right for small retrains
  and wrong for anything that out-of-memories there (DREAM-Attn did, which is why `archinvar_` and
  `refattn32_` are in `MIG_NO`).
* **read by no .py file**: an `AL_*` knob in some `env` that nothing reads.

Routing is judged by importing `dispatcher.py` and calling its own `wall_and_budget()` and `mig_ok()`.
The validator does not keep a copy of the rules, so it cannot drift from them.
