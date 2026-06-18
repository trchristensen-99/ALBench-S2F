"""Launch STEP 1 of the scaling-laws redo — the one-time methodology bake-off.

Goal (memory project_scaling_redo_design.md, AMENDED 2026-06-05): run EVERY HP-search
strategy family DEEP, at a small grid (D=30k first; D=300k deferred) × 3 reservoirs
{genomic, motif_planted_v2, dinuc_shuffle} × multiple seeds, so we can:
  (1) build a per-strategy efficiency curve on CUMULATIVE GPU-SECONDS (the fair
      bang-for-buck axis) → Kneedle + marginal-gain knee → optimal #rounds/strategy
      (scripts/analysis/plot_hp_rounds_and_ensemble.py).
  (2) run the EXHAUSTIVE all-subsets ElasticNetCV recipe search over the pooled
      strategies → diminishing-returns knee → FROZEN recipe {strategies, rounds}
      (scripts/analysis/strategy_combination_ablation.py --all_subsets).

The frozen recipe is then reused at every D in Step 2 (the deploy phase).

GRANULARITY: one job per (reservoir × D × strategy × seed) so each strategy gets a
DEEP independent run (own out_dir), runs in parallel, and is walltime-safe. All
runs for a (reservoir × D) cell pool downstream for the ablation.

STRATEGIES (every family):
  algo / evo (one job each):  random, optuna_tpe, optuna_cmaes, optuna_gp,
    optuna_qmc, evo_single, evo_batch, evo_explore, evo_exploit, evo_massive,
    evo_adaptive, evo_knowledgeable
  Ray Tune schedulers (own engine, own job):  ray_asha, ray_bohb
  LLM AutoResearch (own job each, one model/style per process):
    llm_default (opus default), llm_diverse (sonnet diverse), llm_exploit (sonnet exploit)

BUDGET — EQUALIZE THE MODEL BUDGET (Phase-1 calibration, 2026-06-10): every strategy
gets MODEL_BUDGET=200 models per cell (NOT a common round count), so Phase-2 pool sizes
and attribution stay fair. Each strategy uses its NATURAL per-round step and rounds is
derived (rounds = MODEL_BUDGET // per_round): per_round=1 for sequential/adaptive
(random, optuna_tpe, evo_single/explore/exploit/adaptive/knowledgeable → 200 rounds, so
they update after every eval); native batch for evo_batch=4 / evo_massive=10; ray engines
get num_samples = rounds*per_round = 200 trials; LLM per_round=2 (1 Claude call → 2
configs → 200 models in 100 calls). Strategies are compared on GPU-SECONDS so the unequal
per_round is apples-to-apples. Multi-seed: SEEDS pairs (data_seed, hp_seed).

PREREQUISITE: ray[tune]+hpbandster+ConfigSpace must be installed in the shared venv
(one-time, login node: scripts/install_hpc_packages.sh). Ray variants are SKIPPED with
a warning if `import ray` fails, so the rest of the bake-off still launches.

Env overrides: STEP1_RESERVOIRS, STEP1_DS, STEP1_MODEL_BUDGET (default 200),
STEP1_SEEDS ("42:0,43:1,44:2"), STEP1_STRATS (comma subset), SMOKE_ONLY=1.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
BIN = "/cm/shared/apps/slurm/current/bin"
SBATCH = f"{BIN}/sbatch"
SQUEUE = f"{BIN}/squeue"

# Fresh root for the epochs=100 regime (no-mixing guard); override via STEP1_OUT_ROOT.
OUT_ROOT = os.environ.get("STEP1_OUT_ROOT", f"{REPO}/outputs/hp_step1_bakeoff_e100")

# 3 reservoirs for the methodology search (design: strategy-TYPE coverage, not 10).
# "genomic" = the chr genomic train pool (no reservoir_cache); the others use caches.
DEFAULT_RESERVOIRS = "genomic,motif_planted_v2,dinuc_shuffle"
RESERVOIRS = os.environ.get("STEP1_RESERVOIRS", DEFAULT_RESERVOIRS).split(",")
DS = [int(x) for x in os.environ.get("STEP1_DS", "30000,300000").split(",")]  # plan: {30k,300k}

# Phase-1 calibration (2026-06-10): equalize MODELS per strategy, not rounds. Each
# strategy gets MODEL_BUDGET models; rounds = MODEL_BUDGET // per_round, where per_round
# is the strategy's natural proposal step (sequential update each eval; batched evo and
# LLM amortize). Keeps Phase-2 pool sizes fair; compare strategies on GPU-seconds.
MODEL_BUDGET = int(os.environ.get("STEP1_MODEL_BUDGET", "200"))
DEFAULT_PER_ROUND = {
    # sequential / adaptive — update after every eval
    "random": 1,
    "optuna_tpe": 1,
    "optuna_cmaes": 1,
    "optuna_gp": 1,
    "optuna_qmc": 1,
    "evo_single": 1,
    "evo_explore": 1,
    "evo_exploit": 1,
    "evo_adaptive": 1,
    "evo_knowledgeable": 1,
    # batched evolutionary — native batch (design intent: batch vs massively-parallel)
    "evo_batch": 4,
    "evo_massive": 10,
    # ray schedulers own the trial loop — num_samples = rounds*per_round = MODEL_BUDGET
    "ray_asha": 1,
    "ray_bohb": 1,
    # LLM — one Claude call returns per_round configs (200 models in 100 calls)
    "llm_autoresearch": 2,
}
EPOCHS = 100
PATIENCE = 15
MIN_DELTA = 1e-3
POOL_D = 1_000_000
DATA_SEED_REF = 42  # reservoir-cache seed in the cache filename

# (data_seed, hp_seed) pairs — multiple downsample + HP-init seeds for robust knees.
SEEDS = [
    tuple(int(x) for x in pair.split(":"))
    for pair in os.environ.get("STEP1_SEEDS", "42:0,43:1,44:2").split(",")
]

# Round-based single-strategy variants (one strategy per job → deep & parallel).
EVO_STRATS = [
    "random",
    "optuna_tpe",
    "optuna_cmaes",
    "optuna_gp",
    "optuna_qmc",
    "evo_single",
    "evo_batch",
    "evo_explore",
    "evo_exploit",
    "evo_massive",
    "evo_adaptive",
    "evo_knowledgeable",
]
RAY_STRATS = ["ray_asha", "ray_bohb"]
# (variant_label, strategy, llm_model, llm_style, llm_context, novel_axes)
# FROZEN from the Phase-0 confirm bundle (outputs/hp_llm_ablation_confirm_e100, Jun 17): the four
# complementary personas, each at the novel-axis setting where it was validated (exploit/diverse/
# explore won with novel axes ON; critic won at nv0). All at ctxnone (no cross-experiment KB priors)
# so the LLM is judged on intrinsic in-context optimization — fair vs the algo/evo strategies. The
# final ensemble subset is chosen downstream by the all-subsets ElasticNet ablation over the pooled
# strategies; the bake-off just gives every persona a slot. diverse_nv0 dropped (nv1 dominates +0.016).
LLM_VARIANTS = [
    ("llm_exploit_nv1", "llm_autoresearch", "claude-sonnet-4-6", "exploit", "none", "1"),
    ("llm_critic_nv0", "llm_autoresearch", "claude-sonnet-4-6", "critic", "none", "0"),
    ("llm_diverse_nv1", "llm_autoresearch", "claude-sonnet-4-6", "diverse", "none", "1"),
    ("llm_explore_nv1", "llm_autoresearch", "claude-sonnet-4-6", "explore", "none", "1"),
]


def _ray_available() -> bool:
    # Jobs run via `uv run` (venv has ray); the login node where this submitter
    # runs may not. STEP1_FORCE_RAY=1 bypasses the login-node import probe.
    if os.environ.get("STEP1_FORCE_RAY") == "1":
        return True
    try:
        import ray  # noqa: F401

        return True
    except Exception:
        return False


def all_variants() -> list[tuple[str, str, str, str, str, str]]:
    """(label, strategy, llm_model, llm_style, llm_context, novel). One strategy per variant.
    Non-LLM variants leave model/style/context empty and novel="0" (unused)."""
    subset = os.environ.get("STEP1_STRATS", "").strip()
    allowed = set(subset.split(",")) if subset else None
    out: list[tuple[str, str, str, str, str, str]] = []
    for s in EVO_STRATS:
        if allowed is None or s in allowed:
            out.append((s, s, "", "", "", "0"))
    if _ray_available():
        for s in RAY_STRATS:
            if allowed is None or s in allowed:
                out.append((s, s, "", "", "", "0"))
    else:
        print(
            "  WARN: `import ray` failed — SKIPPING ray_asha/ray_bohb. "
            "Run scripts/install_hpc_packages.sh on the login node first."
        )
    for label, strat, model, style, ctx, novel in LLM_VARIANTS:
        if allowed is None or strat in allowed or label in allowed:
            out.append((label, strat, model, style, ctx, novel))
    return out


def pool_cache(reservoir: str) -> str | None:
    if reservoir == "genomic":
        return None  # use the chr genomic train pool (no reservoir_cache)
    return f"{REPO}/outputs/reservoir_cache/k562_{reservoir}_d{POOL_D}_seed{DATA_SEED_REF}.npz"


def val_cache(reservoir: str) -> str | None:
    """Held-out, transform-matched val cache (same transform on chr19/21/X backgrounds).
    None for genomic (uses real --chr_val). Returns the path only if it exists on disk;
    otherwise the run falls back to the per-combo 10%% holdout (with a launcher warning)."""
    if reservoir == "genomic":
        return None
    p = f"{REPO}/outputs/reservoir_val_cache/k562_{reservoir}_val_seed{DATA_SEED_REF}.npz"
    return p if Path(p).exists() else None


def strat_budget(strategy: str) -> tuple[int, int]:
    """(rounds, per_round) such that rounds*per_round == MODEL_BUDGET (200 models/cell).

    per_round is the strategy's natural proposal step; rounds is derived so every
    strategy trains the same number of models (fair Phase-2 attribution).
    """
    per_round = DEFAULT_PER_ROUND.get(strategy, 1)
    rounds = max(1, MODEL_BUDGET // per_round)
    return rounds, per_round


def expected_models(strategy: str) -> int:
    rounds, per_round = strat_budget(strategy)
    return rounds * per_round  # one strategy/engine per job


def out_dir(reservoir: str, D: int, variant: str, ds: int, hs: int) -> Path:
    return Path(f"{OUT_ROOT}/k562_{reservoir}_d{D}/seed{ds}_{hs}/{variant}")


def is_complete(od: Path, strategy: str) -> bool:
    if not od.exists():
        return False
    # A cell that ran all rounds and exited rc=0 is complete even if a few
    # trainings NaN-failed (so it sits a model or two below the target count).
    if (od / ".bakeoff_done").exists():
        return True
    return len(list(od.glob("*_meta.json"))) >= expected_models(strategy)


def qos_walltime(D: int, is_llm: bool) -> tuple[str, str]:
    """Deep one-time runs → generous walltime. LLM never uses fast (throttle waits)."""
    if is_llm:
        return ("default", "12:00:00") if D <= 100_000 else ("slow_nice", "2-00:00:00")
    if D <= 30_000:
        return "default", "12:00:00"
    return "slow_nice", "2-00:00:00"


def qos_chain(D: int, is_llm: bool) -> list[tuple[str, str]]:
    pref_qos, pref_wt = qos_walltime(D, is_llm)
    chain = [(pref_qos, pref_wt)]
    if pref_qos != "slow_nice":
        chain.append(("slow_nice", "2-00:00:00"))
    return chain


SUBMIT_CAP = {"fast": 16, "default": 16, "slow_nice": 1000}


def job_script(
    reservoir, D, variant, strategy, model, style, context, novel, cache_path, ds, hs, qos, wt
):
    is_llm = bool(model)
    rounds, per_round = strat_budget(strategy)
    label = f"s1_{reservoir}_d{D}_{variant}_s{ds}_{hs}"
    od = out_dir(reservoir, D, variant, ds, hs)
    # genomic non-random reservoirs validate on chr holdout to match production;
    # cache-based reservoirs use per-combo 10% holdout (no --chr_val).
    chr_val_arg = "--chr_val" if (reservoir == "genomic" and strategy != "random") else ""
    cache_arg = f"--reservoir_cache {cache_path}" if cache_path else ""
    # Non-genomic reservoirs: held-out transform-matched val if its cache exists,
    # else fall back to the per-combo 10% holdout (chr_val_arg stays empty).
    vcache = val_cache(reservoir)
    val_cache_arg = f"--reservoir_val_cache {vcache}" if vcache else ""
    llm_env = ""
    if is_llm:
        llm_env = (
            f'export LLM_MODEL="{model}"\n'
            f'export LLM_PROMPT_STYLE="{style}"\n'
            f'export LLM_CONTEXT="{context}"\n'
            f'export LLM_ALLOW_NOVEL_AXES="{novel}"'
        )
    script = f"""#!/bin/bash
#SBATCH --job-name={label}
#SBATCH --output={REPO}/logs/%x-%A.out
#SBATCH --partition=gpuq
#SBATCH --qos={qos}
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time={wt}
#SBATCH --mem=100G
#SBATCH --exclude=bamgpu15,bamgpu25
#SBATCH --export=ALL
#SBATCH --requeue
set -uo pipefail
set +u; source /etc/profile.d/modules.sh; [ -f ~/.bash_profile ] && source ~/.bash_profile; set -u
module load EB5 2>/dev/null || true
cd {REPO}
echo "=== job start (SLURM_RESTART_COUNT=${{SLURM_RESTART_COUNT:-0}}; resume preloads on-disk history) ==="
source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/experiments"
export HP_FAST=1
export HP_CACHE_DIR="$PWD/outputs/tensor_cache"
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1
{llm_env}
ATTEMPT=0
while true; do
  uv run --no-sync python experiments/scaling_hp_search.py \\
    --strategies {strategy} --rounds {rounds} --per_strategy_per_round {per_round} \\
    --D {D} --ref_only {chr_val_arg} {cache_arg} {val_cache_arg} \\
    --data_seed {ds} --hp_seed {hs} \\
    --epochs {EPOCHS} --early_stop_patience {PATIENCE} --min_delta {MIN_DELTA} \\
    --out_dir {od}
  rc=$?
  if [ $rc -eq 0 ]; then touch {od}/.bakeoff_done; echo "=== DONE rc=0 ==="; break; fi
  ATTEMPT=$((ATTEMPT+1))
  if [ $rc -eq 42 ]; then
    if [ $ATTEMPT -ge 12 ]; then echo "=== too many rate-limit stops; giving up ==="; exit 42; fi
    echo "=== rate-limit budget exhausted (exit 42); sleep 1800s then resume (attempt $ATTEMPT) ==="
    sleep 1800; continue
  fi
  echo "=== FAILED rc=$rc ==="; exit $rc
done
"""
    return label, script


def sbatch(label: str, script: str) -> tuple[str | None, str]:
    p = Path(f"/tmp/_s1_{label}.sh")
    p.write_text(script)
    r = subprocess.run([SBATCH, "--parsable", str(p)], capture_output=True, text=True, timeout=30)
    if r.returncode == 0:
        return r.stdout.strip().split(";")[0], ""
    return None, r.stderr.strip()


def main():
    smoke = os.environ.get("SMOKE_ONLY") == "1"
    reservoirs = ["genomic"] if smoke else RESERVOIRS
    ds_list = [30000] if smoke else DS
    seeds = [(42, 0)] if smoke else SEEDS
    variants = all_variants()
    if smoke:
        variants = [v for v in variants if v[0] in ("random", "ray_asha")]

    r = subprocess.run(
        [SQUEUE, "--me", "-h", "--format=%j|%q", "--states=PD,R,CF"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    inflight = set()
    qcount = {"fast": 0, "default": 0, "slow_nice": 0}
    for ln in r.stdout.strip().split("\n"):
        if not ln.strip():
            continue
        name, _, q = ln.partition("|")
        inflight.add(name.strip())
        if q.strip() in qcount:
            qcount[q.strip()] += 1

    n_sub = n_skip = n_done = n_full = 0
    for D in ds_list:
        # Phase-2 assumes the 30k reservoir ranking transfers across D, so D=300k
        # runs genomic ONLY (the cross-D cell); the cache reservoirs stay at 30k.
        reservoirs_D = reservoirs if D <= 30_000 else [r for r in reservoirs if r == "genomic"]
        for reservoir in reservoirs_D:
            cache_path = pool_cache(reservoir)
            if cache_path is not None and not Path(cache_path).exists():
                print(f"  SKIP {reservoir}: reservoir cache missing ({cache_path})")
                continue
            for ds, hs in seeds:
                for variant, strategy, model, style, context, novel in variants:
                    label = f"s1_{reservoir}_d{D}_{variant}_s{ds}_{hs}"
                    od = out_dir(reservoir, D, variant, ds, hs)
                    if is_complete(od, strategy):
                        n_done += 1
                        continue
                    if label in inflight:
                        n_skip += 1
                        continue
                    submitted = False
                    for qos, wt in qos_chain(D, bool(model)):
                        if qcount.get(qos, 0) >= SUBMIT_CAP.get(qos, 0):
                            continue
                        lbl, sh = job_script(
                            reservoir,
                            D,
                            variant,
                            strategy,
                            model,
                            style,
                            context,
                            novel,
                            cache_path,
                            ds,
                            hs,
                            qos,
                            wt,
                        )
                        jid, err = sbatch(lbl, sh)
                        if jid:
                            n_sub += 1
                            qcount[qos] = qcount.get(qos, 0) + 1
                            print(f"  {lbl} -> {jid} [{qos}]")
                            submitted = True
                            break
                        if "QOSMaxSubmitJobPerUser" in err:
                            qcount[qos] = SUBMIT_CAP.get(qos, 0)
                            continue
                        print(f"  ERR {lbl}: {err[:160]}")
                        break
                    if not submitted:
                        n_full += 1

    print(
        f"\nSubmitted: {n_sub}  Skipped(inflight): {n_skip}  "
        f"AlreadyDone: {n_done}  AllTiersFull: {n_full}"
    )
    print(f"Queue counts after run: {qcount}")
    if smoke:
        print("SMOKE_ONLY: genomic/d30000/seed42_0, variants {random, ray_asha} only.")


if __name__ == "__main__":
    main()
