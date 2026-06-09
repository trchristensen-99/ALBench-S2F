"""Launch the 10-reservoir × multi-D AutoResearch HP SEARCH (mixed6, constant budget).

One-time HP-search phase of the scaling-laws redo (memory project_scaling_redo_design.md).
Searches HP over the locked 10 reservoir strategies under random-downsample acquisition,
to be distilled into the N=5 fixed DEPLOY pool.

COMPOSITION (validated mixed6, see project_optimal_pipeline.md) — 6 strategies/cell:
  LLM-driven (3 separate jobs, run_search uses one model/style per process):
    - llm_default : LLM_MODEL=claude-opus-4-7   LLM_PROMPT_STYLE=default
    - llm_diverse : LLM_MODEL=claude-sonnet-4-6  LLM_PROMPT_STYLE=diverse
    - llm_exploit : LLM_MODEL=claude-sonnet-4-6  LLM_PROMPT_STYLE=exploit
  Algorithmic (1 job runs all 3 together, no LLM / no rate-limit):
    - autoresearch_batch, autoresearch_massive, random

BUDGET — CONSTANT across all D for fairness (user, 2026-06-02): rounds=4 ×
per_strategy_per_round=2 → 8 models per LLM job + 24 per algo job = 48 models/cell.
NOT D-scaled (the old mixed6 gave small D more models; that's the unfairness we fix).

SUBSAMPLE-FROM-1M: every D points at the (reservoir) d1M seed42 reservoir cache;
load_chr_train_pool subsamples deterministically (same draw as the deploy phase).

RATE-LIMIT: llm_autoresearch pauses-and-waits (6h budget) then retries the SAME
prompt; never falls back to random. If the budget is blown it exits 42 and is
resumable — the per-job exit-42 loop below re-runs it, so it self-heals.

Each variant writes its OWN out_dir (model_ids collide across LLM variants
otherwise); all 4 are pooled downstream for ElasticNetCV stacking.

Env overrides: HP_RESERVOIRS, HP_DS, HP_ROUNDS, HP_PER_ROUND, SMOKE_ONLY=1.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
BIN = "/cm/shared/apps/slurm/current/bin"
SBATCH = f"{BIN}/sbatch"
SQUEUE = f"{BIN}/squeue"

DEFAULT_RESERVOIRS = (
    "random,prm_1pct,prm_20pct,prm_attribution_1pct,prm_uncertainty_1pct,"
    "evoaug_heavy,motif_planted_v2,dinuc_shuffle,phylogenetic_zoonomia,"
    "mixed_genomic_random"
)
RESERVOIRS = os.environ.get("HP_RESERVOIRS", DEFAULT_RESERVOIRS).split(",")
DS = [int(x) for x in os.environ.get("HP_DS", "3000,10000,30000,100000,300000,1000000").split(",")]
ROUNDS = int(os.environ.get("HP_ROUNDS", "4"))
PER_ROUND = int(os.environ.get("HP_PER_ROUND", "2"))
# Definitive-run budget (2026-06-09): the prior epochs=15/patience=5 study measured
# "fastest learner in 15 epochs" (median optimal best_epoch≈36), not best model.
EPOCHS = 100
PATIENCE = 15
MIN_DELTA = 1e-3
DATA_SEED = 42
HP_SEED = 0
POOL_D = 1_000_000
TASK = "k562"

# (variant_label, strategies, llm_model, llm_style). llm_model/style empty for algo.
LLM_VARIANTS = [
    ("llm_default", "llm_autoresearch", "claude-opus-4-7", "default"),
    ("llm_diverse", "llm_autoresearch", "claude-sonnet-4-6", "diverse"),
    ("llm_exploit", "llm_autoresearch", "claude-sonnet-4-6", "exploit"),
]
ALGO_VARIANT = ("algo", "autoresearch_batch,autoresearch_massive,random", "", "")
ALL_VARIANTS = LLM_VARIANTS + [ALGO_VARIANT]

# Caches still generating — dependent cells get afterok so they self-start.
PENDING_CACHE_JOBS = {
    "prm_uncertainty_1pct": "2441326_6",
    "mixed_genomic_random": "2441547",
}


def pool_cache(reservoir: str) -> str:
    return f"{REPO}/outputs/reservoir_cache/k562_{reservoir}_d{POOL_D}_seed{DATA_SEED}.npz"


def n_strategies(strategies: str) -> int:
    return len(strategies.split(","))


def expected_models(strategies: str) -> int:
    return n_strategies(strategies) * ROUNDS * PER_ROUND


def qos_walltime(D: int, is_llm: bool) -> tuple[str, str]:
    """LLM jobs get generous walltime (rate-limit waits hold the slot); algo jobs
    are compute-bound and predictable so they can use the cheaper fast tier."""
    if is_llm:
        # never fast (4h too tight once throttling/pausing kicks in)
        if D <= 100_000:
            return "default", "12:00:00"
        return "slow_nice", "2-00:00:00"
    # algo (no LLM)
    if D <= 10_000:
        return "fast", "04:00:00"
    if D <= 100_000:
        return "default", "12:00:00"
    return "slow_nice", "2-00:00:00"


# Fresh root for the epochs=100 regime so it never mixes with the epochs=15 results
# (the regime no-mixing guard would otherwise hard-fail). Override via HP_OUT_ROOT.
OUT_ROOT = os.environ.get("HP_OUT_ROOT", f"{REPO}/outputs/hp_search_e100")


def val_protocol() -> str:
    """Validation methodology for this study's jobs.

    'holdout' (default, 2026-06-03 methodology): no --chr_val → scaling_hp_search
    carves a per-combo 10% random holdout from each (reservoir × strategy) train
    pool, so every run validates on a constant ratio of its own training
    distribution.
    'chr_val' (legacy): genomic chr19/21/X holdout (--chr_val for non-random
    reservoirs).

    The in-flight main search is PINNED to its original protocol by a
    `<OUT_ROOT>/.val_protocol` marker so a watchdog resubmit never changes val
    methodology mid-study. New studies (new out_root, no marker) get 'holdout'.
    Env HP_VAL_PROTOCOL overrides for ad-hoc launches.
    """
    env = os.environ.get("HP_VAL_PROTOCOL", "").strip()
    if env:
        return env
    marker = Path(OUT_ROOT) / ".val_protocol"
    if marker.exists():
        tok = marker.read_text().strip().split()
        if tok:
            return tok[0]
    return "holdout"


def out_dir(reservoir: str, D: int, variant: str) -> Path:
    return Path(f"{OUT_ROOT}/k562_{reservoir}_d{D}/{variant}")


def is_complete(od: Path, strategies: str) -> bool:
    if not od.exists():
        return False
    n_meta = len(list(od.glob("*_meta.json")))
    return n_meta >= expected_models(strategies)


def job_script(reservoir, D, variant, strategies, model, style, cache_path, dep, qos, wt):
    is_llm = bool(model)
    label = f"hp_{reservoir}_d{D}_{variant}"
    od = out_dir(reservoir, D, variant)
    dep_line = f"#SBATCH --dependency=afterok:{dep}" if dep else ""
    chr_val_arg = "--chr_val" if (val_protocol() == "chr_val" and reservoir != "random") else ""
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
{dep_line}
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
export TQDM_DISABLE=1
{llm_env}
ATTEMPT=0
while true; do
  uv run --no-sync python experiments/scaling_hp_search.py \\
    --strategies {strategies} --rounds {ROUNDS} --per_strategy_per_round {PER_ROUND} \\
    --D {D} --ref_only {chr_val_arg} \\
    --reservoir_cache {cache_path} \\
    --data_seed {DATA_SEED} --hp_seed {HP_SEED} \\
    --epochs {EPOCHS} --early_stop_patience {PATIENCE} --min_delta {MIN_DELTA} \\
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
    p = Path(f"/tmp/_hp_{label}.sh")
    p.write_text(script)
    r = subprocess.run([SBATCH, "--parsable", str(p)], capture_output=True, text=True, timeout=30)
    if r.returncode == 0:
        return r.stdout.strip().split(";")[0], ""
    return None, r.stderr.strip()


# Per-QOS submit caps (sacctmgr MaxSubmitPU). slow_nice is the deep overflow tier.
SUBMIT_CAP = {"fast": 16, "default": 16, "slow_nice": 1000}


def qos_chain(D: int, is_llm: bool) -> list[tuple[str, str]]:
    """Preferred (qos, walltime), then slow_nice overflow so small-D jobs that
    blow the fast/default MaxSubmitPU=16 cap still land instead of failing."""
    pref_qos, pref_wt = qos_walltime(D, is_llm)
    chain = [(pref_qos, pref_wt)]
    if pref_qos != "slow_nice":
        chain.append(("slow_nice", "2-00:00:00"))
    return chain


def main():
    smoke = os.environ.get("SMOKE_ONLY") == "1"
    reservoirs = ["random"] if smoke else RESERVOIRS
    ds = [3000] if smoke else DS

    # In-flight job names (dedup) + per-qos counts (respect MaxSubmitPU caps).
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
        cache_exists = Path(cache_path).exists()
        dep = PENDING_CACHE_JOBS.get(reservoir)
        if not cache_exists and not dep:
            print(f"  SKIP {reservoir}: cache missing and no pending cache job — generate first")
            continue
        for D in ds:
            for variant, strategies, model, style in ALL_VARIANTS:
                label = f"hp_{reservoir}_d{D}_{variant}"
                od = out_dir(reservoir, D, variant)
                if is_complete(od, strategies):
                    n_done += 1
                    continue
                if label in inflight:
                    n_skip += 1
                    continue
                submitted = False
                for qos, wt in qos_chain(D, bool(model)):
                    if qcount.get(qos, 0) >= SUBMIT_CAP.get(qos, 0):
                        continue  # this tier is at MaxSubmitPU — try overflow
                    lbl, sh = job_script(
                        reservoir,
                        D,
                        variant,
                        strategies,
                        model,
                        style,
                        cache_path,
                        dep if not cache_exists else None,
                        qos,
                        wt,
                    )
                    jid, err = sbatch(lbl, sh)
                    if jid:
                        n_sub += 1
                        qcount[qos] = qcount.get(qos, 0) + 1
                        depnote = f" dep=afterok:{dep}" if (not cache_exists and dep) else ""
                        print(f"  {lbl} -> {jid} [{qos}]{depnote}")
                        submitted = True
                        break
                    if "QOSMaxSubmitJobPerUser" in err:
                        qcount[qos] = SUBMIT_CAP.get(qos, 0)  # mark full, overflow
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
        print("SMOKE_ONLY: launched random/d3000 (4 variants) only.")


if __name__ == "__main__":
    main()
