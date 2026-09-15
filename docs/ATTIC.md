# The attic: what was removed from the tracked tree, and how to get it back

On 2026-09-15 the tracked tree went from **1365 files to ~285**. Nothing was deleted.
Everything removed sits in `attic/` on disk (git-ignored) and remains in git history,
so any file can be restored with a one-line command.

## Why

The repo had accumulated ~1000 one-off scripts across five years of experiments:
522 hand-written SLURM scripts, 156 preflight probes, 190 analysis scripts, many
superseded by later rounds. A collaborator opening the repo could not tell which of
40 files named `train_oracle_*` was the live one. That is a reproducibility problem,
not a tidiness problem.

## What was kept

The keep set was computed, not guessed: a transitive first-party **import closure**
from the entry points of the current round, then widened where static analysis is
unsafe.

| Kept | Why |
|---|---|
| `albench/` (all 57) | the deliverable package; registry factories and lazy imports make static tracing unreliable, so it is kept whole |
| `configs/` (all 112) | hydra composes by scanning group directories, so pruning members silently breaks overrides |
| `models/`, `data/`, `evaluation/`, `tests/` | reachable from the closure, or whole where tracing is unsafe |
| `experiments/` (8) | `exp1_1_scaling`, `train_oracle_s2_v2`, `train_yeast_oracle_v2`, `oracle_oof_eval`, `test_set_guards`, `scaling_hp_search` + its LLM-search deps |
| `scripts/` (28) | pool/yeast/CIS-BP builders, the tiered launcher, and the analyses behind numbers we cite |
| root `run_*.sbatch`, watchdogs | the runners for the live yeast and human experiments |

Verified before committing: no kept file imports an attic file, all shell and YAML
parse, and the 83 runnable unit tests pass.

## What moved

1091 files: `scripts/slurm` (522), `scripts/preflight` (156), most of
`scripts/analysis` (161 kept 15), `experiments/` archive + superseded drivers (53),
unreferenced model wrappers (8), stale status docs, and `results/`, `koo_upload/`,
`figures_schematics/` build artefacts.

## Restoring

```bash
git log --oneline --all -- path/to/file      # find it
git checkout <commit> -- path/to/file        # bring it back
# or, if you still have the working copy:
mv attic/path/to/file path/to/file
```

To browse what is in the attic without restoring: `ls attic/scripts/analysis/`.

## Caution for existing checkouts

Pulling this commit **removes those 1091 files from your working tree** (they become
untracked history). If a checkout has local-only edits to any of them, stash or copy
them out first. This matters most on the HPC checkout, which carries local drift and
should have files fetched individually (`git checkout origin/main -- <file>`) rather
than pulled wholesale.
