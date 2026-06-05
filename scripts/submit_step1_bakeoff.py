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
  algo / evo (one job each):  random, optuna_tpe, evo_single, evo_batch,
    evo_explore, evo_exploit, evo_massive, evo_adaptive, evo_knowledgeable
  Ray Tune schedulers (own engine, own job):  ray_asha, ray_bohb
  LLM AutoResearch (own job each, one model/style per process):
    llm_default (opus default), llm_diverse (sonnet diverse), llm_exploit (sonnet exploit)

BUDGET — deep & one-time (override via env). ROUNDS=50 × PER_ROUND=2 → 100 models per
strategy per cell; Ray engines get num_samples = ROUNDS*PER_ROUND = 100 trials.
Multi-seed: SEEDS pairs (data_seed, hp_seed) for robust knees.

PREREQUISITE: ray[tune]+hpbandster+ConfigSpace must be installed in the shared venv
(one-time, login node: scripts/install_hpc_packages.sh). Ray variants are SKIPPED with
a warning if `import ray` fails, so the rest of the bake-off still launches.

Env overrides: STEP1_RESERVOIRS, STEP1_DS, STEP1_ROUNDS, STEP1_PER_ROUND,
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

OUT_ROOT = f"{REPO}/outputs/hp_step1_bakeoff"

# 3 reservoirs for the methodology search (design: strategy-TYPE coverage, not 10).
# "genomic" = the chr genomic train pool (no reservoir_cache); the others use caches.
DEFAULT_RESERVOIRS = "genomic,motif_planted_v2,dinuc_shuffle"
RESERVOIRS = os.environ.get("STEP1_RESERVOIRS", DEFAULT_RESERVOIRS).split(",")
DS = [int(x) for x in os.environ.get("STEP1_DS", "30000").split(",")]  # 300k deferred
ROUNDS = int(os.environ.get("STEP1_ROUNDS", "50"))
PER_ROUND = int(os.environ.get("STEP1_PER_ROUND", "2"))
EPOCHS = 15
PATIENCE = 5
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
    "evo_single",
    "evo_batch",
    "evo_explore",
    "evo_exploit",
    "evo_massive",
    "evo_adaptive",
    "evo_knowledgeable",
]
RAY_STRATS = ["ray_asha", "ray_bohb"]
# (variant_label, strategy, llm_model, llm_style)
LLM_VARIANTS = [
    ("llm_default", "llm_autoresearch", "claude-opus-4-7", "default"),
    ("llm_diverse", "llm_autoresearch", "claude-sonnet-4-6", "diverse"),
    ("llm_exploit", "llm_autoresearch", "claude-sonnet-4-6", "exploit"),
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


def all_variants() -> list[tuple[str, str, str, str]]:
    """(label, strategies, llm_model, llm_style). One strategy per variant."""
    subset = os.environ.get("STEP1_STRATS", "").strip()
    allowed = set(subset.split(",")) if subset else None
    out: list[tuple[str, str, str, str]] = []
    for s in EVO_STRATS:
        if allowed is None or s in allowed:
            out.append((s, s, "", ""))
    if _ray_available():
        for s in RAY_STRATS:
            if allowed is None or s in allowed:
                out.append((s, s, "", ""))
    else:
        print(
            "  WARN: `import ray` failed — SKIPPING ray_asha/ray_bohb. "
            "Run scripts/install_hpc_packages.sh on the login node first."
        )
    for label, strat, model, style in LLM_VARIANTS:
        if allowed is None or strat in allowed or label in allowed:
            out.append((label, strat, model, style))
    return out


def pool_cache(reservoir: str) -> str | None:
    if reservoir == "genomic":
        return None  # use the chr genomic train pool (no reservoir_cache)
    return f"{REPO}/outputs/reservoir_cache/k562_{reservoir}_d{POOL_D}_seed{DATA_SEED_REF}.npz"


def expected_models(strategy: str) -> int:
    return ROUNDS * PER_ROUND  # one strategy/engine per job


def out_dir(reservoir: str, D: int, variant: str, ds: int, hs: int) -> Path:
    return Path(f"{OUT_ROOT}/k562_{reservoir}_d{D}/seed{ds}_{hs}/{variant}")


def is_complete(od: Path, strategy: str) -> bool:
    if not od.exists():
        return False
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


def job_script(reservoir, D, variant, strategy, model, style, cache_path, ds, hs, qos, wt):
    is_llm = bool(model)
    label = f"s1_{reservoir}_d{D}_{variant}_s{ds}_{hs}"
    od = out_dir(reservoir, D, variant, ds, hs)
    # genomic non-random reservoirs validate on chr holdout to match production;
    # cache-based reservoirs use per-combo 10% holdout (no --chr_val).
    chr_val_arg = "--chr_val" if (reservoir == "genomic" and strategy != "random") else ""
    cache_arg = f"--reservoir_cache {cache_path}" if cache_path else ""
    llm_env = ""
    if is_llm:
        llm_env = f'export LLM_MODEL="{model}"\nexport LLM_PROMPT_STYLE="{style}"'
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
set -uo pipefail
set +u; source /etc/profile.d/modules.sh; [ -f ~/.bash_profile ] && source ~/.bash_profile; set -u
module load EB5 2>/dev/null || true
cd {REPO}
source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/experiments"
export HP_FAST=1
export HP_CACHE_DIR="$PWD/outputs/tensor_cache"
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1
{llm_env}
ATTEMPT=0
while true; do
  uv run --no-sync python experiments/scaling_hp_search.py \\
    --strategies {strategy} --rounds {ROUNDS} --per_strategy_per_round {PER_ROUND} \\
    --D {D} --ref_only {chr_val_arg} {cache_arg} \\
    --data_seed {ds} --hp_seed {hs} \\
    --epochs {EPOCHS} --early_stop_patience {PATIENCE} \\
    --out_dir {od}
  rc=$?
  if [ $rc -eq 0 ]; then echo "=== DONE rc=0 ==="; break; fi
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
    for reservoir in reservoirs:
        cache_path = pool_cache(reservoir)
        if cache_path is not None and not Path(cache_path).exists():
            print(f"  SKIP {reservoir}: reservoir cache missing ({cache_path})")
            continue
        for D in ds_list:
            for ds, hs in seeds:
                for variant, strategy, model, style in variants:
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
