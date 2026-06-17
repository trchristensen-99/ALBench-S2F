# Phase 1 — Consolidated HP-Strategy Bake-off

**Status:** active (anchor launching 2026-06-17). Supersedes the earlier two-step
plan that calibrated optimal rounds (old "Phase 1") and *then* selected the strategy
(old "Phase 2") as separate passes.

## What changed and why

The old plan ran two passes: first fix the number of HP-search rounds per strategy,
then compare strategies at that fixed budget. Those two questions are coupled — a
strategy's value *is* the shape of its performance-vs-compute curve — so we now answer
both from **one deep run per cell**:

- **Optimal compute** = the knee of the per-strategy efficiency curve (Kneedle on
  ensemble-oracle-r vs cumulative GPU-seconds).
- **Strategy choice** = the height/ranking of those same curves at/after the knee.

The frozen output is a single recipe `{strategies, rounds-or-GPU-seconds}` reused at
every deploy cell.

## Budget axis (the fair currency)

**Primary:** ensemble-oracle-r vs **cumulative GPU-seconds**. This is the honest
bang-for-buck axis and it absorbs the "expensive-models" confound automatically — a
strategy that proposes large/slow configs simply advances along the x-axis faster and
must justify the cost with height. (The live proof: the LLM `explore` persona proposing
huge configs is our multi-hour long-pole at equal *model* count — equal-model budgets
over-charge expensive-proposal strategies in wall-clock while flattering their
per-model numbers.)

**Secondary lens:** ensemble-oracle-r vs **model count** — kept to disentangle
proposal *quality* (per model) from proposal *cost*.

**Cap:** walltime (slow_nice, 2 days). Slow strategies truncate gracefully via
checkpoint/resume; the GPU-seconds curve up to the cap is still valid, and the knee
is typically well before the cap. The old equal-`MODEL_BUDGET=200` setting existed to
keep Phase-2 pool sizes comparable; under consolidation we read curves vs GPU-seconds,
so unequal effective depth across strategies is fine and expected.

### Note on within-cell parallelism

Within a single job/cell, proposed models train **sequentially on one GPU**
(`scaling_hp_search.py`: "Train each config sequentially (single-GPU for now)").
Parallelism is **across cells** — every (reservoir × D × strategy × seed) is its own
SLURM job on its own H100. So per-round proposals are *not* trained concurrently; the
LLM/evo batch size (`per_strategy_per_round`) only controls how often the strategy sees
feedback, not throughput. All strategies in the bake-off train the **same number of
models** (no strategy is starved); they differ in GPU-seconds, which is exactly what we
measure. Multi-GPU per-round fan-out is possible future work but is deliberately traded
away for simple, resumable checkpointing.

## Experimental design — anchor + OFAT + corner (not random combos)

Random combinations of {D × reservoir × acquisition} *confound the three axes* and hide
selection bias behind an n=3 average. Instead we measure **transfer** directly: pick one
anchor, perturb one axis at a time, and confirm the strategy ranking is preserved.

### Anchor cell — the primary decision
`reservoir=genomic, D=30k, acquisition=random`. **All** strategies, **3 seeds**
(`42:0, 43:1, 44:2`). Primary ranking + per-strategy knee come from here.

### OFAT transfer probes — does the choice generalize?
Perturb exactly **one** axis off the anchor; run only the **top-K strategies + 2
baselines (`random`, `optuna_tpe`)**, **2 seeds**, and check rank-correlation vs the
anchor ranking:
- **D axis:** 30k → 100k, 30k → 300k
- **Reservoir axis:** genomic → random, genomic → tf_planting (`motif_planted_v2`)
- **Acquisition axis:** random → uncertainty, random → diversity

Ranking preserved on an axis ⇒ transfer holds, reuse the anchor recipe. Ranking breaks
⇒ that axis needs per-cell treatment.

### One corner — interaction check
`{D=300k, reservoir=tf_planting, acquisition=diversity}`, top-K strategies only. If it
matches what the single-axis main effects predict ⇒ no meaningful interactions. If not
⇒ investigate before trusting transfer.

### Seed allocation
3 seeds where the decision is made (anchor); 2 seeds at the probe/corner cells (only
checking rank preservation, coarser estimate acceptable). Cross-seed SD (~0.013–0.018)
swamps small gaps, so decisions are made on rank stability across seeds, not single-seed
point estimates.

## Strategy roster

| family | strategies |
|---|---|
| baselines | `random`, `optuna_qmc` (scrambled-Sobol), `optuna_tpe` (TPE BO) |
| Bayesian opt | `optuna_cmaes` (CMA-ES, continuous dims), `optuna_gp` (GP/BoTorch) |
| evolutionary | `evo_single`, `evo_batch`, `evo_explore`, `evo_exploit`, `evo_massive`, `evo_adaptive`, `evo_knowledgeable` |
| multi-fidelity | `ray_asha` (async successive halving), `ray_bohb` (Bayesian + Hyperband) |
| LLM AutoResearch | the Phase-0 deploy-3 personas (pending finalization) |

`optuna_cmaes/gp/qmc` were added 2026-06-17. CMA-ES optimizes the continuous HPs
(lr × weight_decay × dropouts) where the one-HP-at-a-time `evo_*` family is weakest;
GP is more sample-efficient at small budgets; QMC is a fairer "no-model" floor than
i.i.d. random.

## Acquisition strategies (do not block strategy selection on these)

Acquisition reshapes the training *distribution*, a second-order effect on optimal HPs,
so the bake-off runs the default **random** acquisition; acquisition-sensitivity is the
**lowest-priority** OFAT axis, tested only after the anchor recipe is set.

- **Uncertainty:** implement via a cheap **proxy LegNet** (trained on the initial
  genomic seed) with MC-dropout / small-ensemble variance. Do **not** use the oracle's
  own uncertainty — the oracle is the label generator, so oracle-uncertainty AL is
  leaky and would flatter the result; oracle-margin is at most a throwaway sanity check,
  flagged as leaky.
- **Diversity:** start with kmer-spectrum or embedding-kNN distance (cheap); **LCMD**
  (largest-cluster-max-distance, the principled batch-diversity method) needs embeddings
  and is deferred.

## Analysis pipeline

1. Per-strategy efficiency curve + Kneedle knee
   (`scripts/analysis/plot_hp_rounds_and_ensemble.py`).
2. Exhaustive all-subsets ElasticNetCV recipe search over pooled strategies →
   diminishing-returns knee → frozen `{strategies, rounds}`
   (`scripts/analysis/strategy_combination_ablation.py --all_subsets`). Handles variable
   atom counts, so unequal per-strategy depth is fine.
3. Rank-correlation of strategy ordering across the OFAT probe cells → transfer verdict.

## Launch ledger

- 2026-06-17: anchor algo+evo strategies (12) × genomic × 30k × 3 seeds launched.
  Deferred: `ray_asha`/`ray_bohb` (need `ray[tune]`+`hpbandster`+`ConfigSpace` in the
  venv) and the LLM arms (pending the Phase-0 deploy-3 persona result).
