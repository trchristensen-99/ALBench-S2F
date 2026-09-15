# Reproducing the current experiments

Two arms are live. Both label with an oracle ensemble, then train students on the
labelled pools. Paths come from `albench.paths` (see `albench doctor`); no script
contains a machine-specific location outside a documented fallback.

## Setup

```bash
uv sync
albench doctor          # lists every asset, whether you have it, and how to get it
```

`ALBENCH_DATA` sets the data root, `ALBENCH_REPO` the repo root used by the SLURM
runners, `ALPHAGENOME_WEIGHTS` the AlphaGenome checkpoint.

## Human arm: the 30k parameter screen

```bash
# 1. enumerate the screen cells (9 strategies x parameters x seeds = 210)
albench screen --config configs/screen/human_30k_300k.yaml --write outputs/screen/jobs.sh

# 2. generate sequences            -> outputs/screen/cache/<cell>.npz
sbatch run_screengen.sbatch

# 3. label with the AG oracle      -> <cell>__labeled.npz  (D=30k half)
sbatch run_label30k.sbatch

# 4. expose the flat cache in the layout the scaling driver expects
python scripts/link_screen_pools.py

# 5. train one LegNet student per cell
sbatch run_screentrain.sbatch
sbatch run_screentrain_watchdog.sbatch     # resubmits if slow_nice preempts
```

Subsets are nested by construction: `load_pool_subset` permutes the pool once under
the seed and takes a prefix, so n=5k is a strict subset of n=10k. This matters for
scaling curves — resampling independently per size makes them jagged.

## Yeast arm: the DREAM-RNN oracle

```bash
python scripts/build_yeast_folds.py             # random 10-fold, duplicates grouped
python scripts/build_yeast_eval_classes.py      # class map + pair-safe stratified folds

sbatch run_yoracle.sbatch                       # variant A: bulk data only
sbatch run_yoracleB.sbatch                      # variant B: + 80% of each eval class
sbatch run_yoracle_watchdog.sbatch

sbatch run_yopredict.sbatch                     # score eval classes with all 20 folds
python scripts/analysis/eval_yeast_oracle_by_class.py
```

Variant B exists to measure what including an eval class in oracle training buys on
that class. The A-vs-B comparison is restricted to sequences **both** held out —
otherwise B is flattered by exactly what it memorised.

## GPU placement

LSTM work (the yeast oracle) must be H100-constrained: measured ~5 min/epoch on H100
against ~38 min on V100, a 7.6x penalty. LegNet is a CNN and pays only 2.4x, so it is
deliberately left unconstrained to use the plentiful idle V100s.

```bash
scripts/slurm/launch_tiered.sh run_yofold_one.sbatch "--tag _B" -- 0 1 2 3
```
spreads jobs across the fast/default/slow_nice tiers, asking Slurm via
`sbatch --test-only` where each would actually start, and refusing placements that
would preempt our own running jobs.
