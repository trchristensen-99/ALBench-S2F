# Reproducing the current experiments

Two arms are live. Both label with an oracle ensemble, then train students on the
labelled pools. Nothing below names a partition, a queue, or an absolute path: asset
locations come from `albench.paths`, and scheduling comes from
`scripts/cluster/site.env`.

## Setup

```bash
uv sync
albench doctor                                  # which assets you have, how to get the rest
cp scripts/cluster/site.env.example scripts/cluster/site.env
$EDITOR scripts/cluster/site.env                # partitions, QoS tiers, account
```

`site.env` is optional. With no scheduler on PATH the pipelines run each stage in the
foreground instead, which is slow but correct — useful on a workstation or inside an
interactive allocation.

Environment knobs: `ALBENCH_DATA` (data root), `ALBENCH_REPO` (repo path on compute
nodes), `ALPHAGENOME_WEIGHTS` (oracle backbone), `ALBENCH_PYTHON` (how to invoke
Python, e.g. `"module load cuda/12.4 && python"`).

## Human arm: the 30k parameter screen

```bash
./scripts/pipelines/human_screen.sh enumerate   # 210 cells from the screen config
./scripts/pipelines/human_screen.sh generate    # sequences per cell
./scripts/pipelines/human_screen.sh label       # oracle labels (needs the oracle)
./scripts/pipelines/human_screen.sh link        # expose cells as pools
./scripts/pipelines/human_screen.sh train       # one LegNet student per cell
./scripts/pipelines/human_screen.sh status
```

Every stage is idempotent — finished work is skipped — so **re-running a stage is the
recovery procedure** after preemption or a walltime kill. No watchdog required.

Subsets are nested by construction: `load_pool_subset` permutes the pool once under
the seed and takes a prefix, so n=5k is a strict subset of n=10k. Resampling
independently per size is what makes scaling curves jagged.

## Yeast arm: the DREAM-RNN oracle

```bash
./scripts/pipelines/yeast_oracle.sh prepare     # folds + pair-safe eval-class map
./scripts/pipelines/yeast_oracle.sh train A     # bulk data only
./scripts/pipelines/yeast_oracle.sh train B     # + 80% of each splittable eval class
./scripts/pipelines/yeast_oracle.sh predict     # score eval classes, all 20 folds
./scripts/pipelines/yeast_oracle.sh compare
```

Variant B measures what including an eval class in oracle training buys on that class.
The comparison is restricted to sequences **both** variants held out — otherwise B is
flattered by exactly what it memorised.

**Label scales must match before you add any external label source.** The bulk yeast
file is binned expression on [0, 17]; the DREAM eval file is MAUDE expression on
[-1.40, 1.66]. Mixing them raw taught the first variant B to predict ~0.16 for eval
sequences and ~11 for everything else, which showed up as r=0.17 on SNVs it had
trained on against 0.87 for a model that never saw them.
`match_eval_labels_to_train_scale` fits the affine correction on the `random` class.

## Scheduling

`scripts/cluster/submit.sh` is the only thing that talks to a scheduler:

```bash
scripts/cluster/submit.sh --name train --gpus 1 --cpus 8 --mem 64G \
    --time 11:30:00 --array 1-105%20 --tiered -- python my_script.py
```

`--tiered` spreads work across the QoS tiers in `ALBENCH_TIERS`, asking the scheduler
via `sbatch --test-only` where each job would actually start rather than tracking
capacity itself, and refusing placements that would preempt your own running jobs.

**Constrain recurrent training to your fastest accelerator** (`ALBENCH_GPU_CONSTRAINT`).
Measured here, DREAM-RNN epochs took ~5 min on an H100 against ~38 min on a V100 — a
7.6x penalty, where a CNN of similar size paid only 2.4x. Leave convolutional training
unconstrained so it can use whatever is idle.
