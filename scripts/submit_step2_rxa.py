"""STEP 2 of the scaling-laws redo — NATIVE HP searches across the R×A grid at D>=100k.

WHY a separate launcher from submit_step1_bakeoff.py:
  Step 1 ran the methodology bake-off on the GENOMIC reservoir at D=30k (every HP
  strategy, 3 seeds, 200 models) to pick the recipe + N* menu. Step 2 asks the
  scientific ALBench question: does an HP menu transfer ACROSS reservoir (R) and
  acquisition (A), or do you need a NATIVE search per (R,A)? To answer it we must
  run a REAL HP search on each (R,A) cell — transfer is of the architecture/HP
  config, NEVER trained weights (always fresh-init retrain), so "retrain only" is
  not a substitute for the native baseline (memory project_deploy_persearch_design).

  Three ensembles get compared per (D,R,A) cell, ALL locked then retrained on the
  SAME fresh, disjoint train+val draw and scored on the SAME held-out test:
    (1) DEPLOYED menu      — the fixed per-D N* menu (built on genomic D=30k/anchor)
    (2) GENOMIC-transferred — the menu from a native genomic search at THIS D
    (3) NATIVE (this script)— the menu from a native search on THIS (R,A) data
  Fairness refinement (user, 2026-06-18): the native arm must NOT keep a home-field
  advantage. Its final menu is retrained on a fresh train+val it never saw during
  the search — identical treatment to arms (1) and (2). That disjointness is
  enforced at the data layer by HP_POOL_RESERVE_EVAL_FRAC (see scaling_hp_search.
  pool_partition): the search samples ONLY the searchable universe; the reserved
  tail is the fresh retrain/eval draw. data_seed draws merely OVERLAP — they are
  not disjoint — so a reserved partition is required, not just a new seed.

ECONOMY (memory project_deploy_ensemble_construction_jun18):
  - MODEL_BUDGET=100/cell: the single-val ensemble oracle-r plateaus by ~100 models
    at D=30k (50->0.764, 100->0.767, 150->0.768, all->0.767); half the GPU of 200.
  - 2 seeds at D>=100k: downsampling variance shrinks with D, so 2 deployment
    simulations suffice (3 only needed at the 30k anchor).
  - checkpoint-resumable: scaling_hp_search preloads on-disk history on requeue.

GRID (all env-overridable):
  STEP2_DS=100000,300000   STEP2_RESERVOIRS=genomic,motif_planted_v2
  STEP2_ACQS=,uncertainty  (empty token = the default/random acquisition)
  STEP2_SEEDS=42:0,43:1     STEP2_MODEL_BUDGET=100  STEP2_RESERVE_EVAL_FRAC=0.34
  STEP2_STRATS=<comma subset>  (default: the bake-off strategy families)
  DRY_RUN=1 to print the plan without submitting.

PREREQUISITE: each non-genomic (R,A) cell needs its acquisition-selected, canonical-
oracle-stamped reservoir cache on disk at
  outputs/reservoir_cache/k562_{R}_{A}_d{POOL_D}_seed{REF}.npz
(+ matching outputs/reservoir_val_cache/k562_{R}_{A}_val_seed{REF}.npz). genomic
uses the chr-train pool + real --chr_val (no cache). Cells whose cache is missing
are SKIPPED with a warning — generate them first (acquisition selection -> cache).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from submit_step1_bakeoff import (  # noqa: E402  reuse budget + strategy families
    EVO_STRATS,
    LLM_VARIANTS,
    RAY_STRATS,
    _ray_available,
    strat_budget,
)

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
BIN = "/cm/shared/apps/slurm/current/bin"
SBATCH = f"{BIN}/sbatch"
SQUEUE = f"{BIN}/squeue"

OUT_ROOT = os.environ.get("STEP2_OUT_ROOT", f"{REPO}/outputs/hp_step2_rxa_e100")

DS = [int(x) for x in os.environ.get("STEP2_DS", "100000,300000").split(",") if x]
RESERVOIRS = [
    r for r in os.environ.get("STEP2_RESERVOIRS", "genomic,motif_planted_v2").split(",") if r
]
# "" = the default acquisition (random downsample); named tags fold into the cache
# path + cell dir (k562_{R}_{A}_...), matching submit_deploy.py's convention.
ACQS = os.environ.get("STEP2_ACQS", ",uncertainty").split(",")
SEEDS = [
    tuple(int(x) for x in pair.split(":"))
    for pair in os.environ.get("STEP2_SEEDS", "42:0,43:1").split(",")
]

# Half the 30k budget — the single-val ensemble plateaus by ~100 models/cell.
MODEL_BUDGET = int(os.environ.get("STEP2_MODEL_BUDGET", "100"))
# Fraction of each R×A pool reserved (canonically, data_seed-independent) for the
# fair retrain/eval draw — never searched. Must leave room for both a D-sized
# searchable universe AND a D-sized fresh draw; 0.34 of a 1M pool = 340k reserved
# (enough for D=300k retrain) / 660k searchable (enough for D=300k search).
RESERVE_EVAL_FRAC = float(os.environ.get("STEP2_RESERVE_EVAL_FRAC", "0.34"))

EPOCHS = 100
PATIENCE = 15
MIN_DELTA = 1e-3
POOL_D = 1_000_000
DATA_SEED_REF = 42  # reservoir-cache seed in the cache filename

SUBMIT_CAP = {"fast": 16, "default": 16, "slow_nice": 1000}


def variants() -> list[tuple[str, str, str, str, str, str]]:
    """(label, strategy, llm_model, llm_style, llm_context, novel). Same families as
    the bake-off so the native menu is built by the identical procedure. Narrow with
    STEP2_STRATS once the Phase-2 recipe is frozen to the chosen strategy subset."""
    subset = os.environ.get("STEP2_STRATS", "").strip()
    allowed = set(subset.split(",")) if subset else None
    out: list[tuple[str, str, str, str, str, str]] = []
    for s in EVO_STRATS:
        if allowed is None or s in allowed:
            out.append((s, s, "", "", "", "0"))
    if _ray_available():
        for s in RAY_STRATS:
            if allowed is None or s in allowed:
                out.append((s, s, "", "", "", "0"))
    for label, strat, model, style, ctx, novel in LLM_VARIANTS:
        if allowed is None or strat in allowed or label in allowed:
            out.append((label, strat, model, style, ctx, novel))
    return out


def cell_tag(reservoir: str, acq: str) -> str:
    """k562_{R} for the default acquisition, k562_{R}_{A} for a named one."""
    return f"k562_{reservoir}" if not acq else f"k562_{reservoir}_{acq}"


def pool_cache(reservoir: str, acq: str) -> str | None:
    if reservoir == "genomic":
        return None  # chr genomic train pool, no cache
    return f"{REPO}/outputs/reservoir_cache/{cell_tag(reservoir, acq)}_d{POOL_D}_seed{DATA_SEED_REF}.npz"


def val_cache(reservoir: str, acq: str) -> str | None:
    if reservoir == "genomic":
        return None
    p = f"{REPO}/outputs/reservoir_val_cache/{cell_tag(reservoir, acq)}_val_seed{DATA_SEED_REF}.npz"
    return p if Path(p).exists() else None


def out_dir(reservoir: str, acq: str, D: int, variant: str, ds: int, hs: int) -> Path:
    return Path(f"{OUT_ROOT}/{cell_tag(reservoir, acq)}_d{D}/seed{ds}_{hs}/{variant}")


def expected_models(strategy: str) -> int:
    rounds, per_round = strat_budget(strategy)
    return rounds * per_round


def is_complete(od: Path, strategy: str) -> bool:
    if not od.exists():
        return False
    if (od / ".bakeoff_done").exists():
        return True
    return len(list(od.glob("*_meta.json"))) >= expected_models(strategy)


def qos_walltime(D: int, is_llm: bool) -> tuple[str, str]:
    # D>=100k native searches are long; LLM never uses fast (throttle waits).
    return ("slow_nice", "2-00:00:00")


def job_script(
    reservoir, acq, D, variant, strategy, model, style, context, novel, cache_path, ds, hs, qos, wt
):
    is_llm = bool(model)
    rounds, per_round = strat_budget(strategy)
    label = f"s2_{cell_tag(reservoir, acq)}_d{D}_{variant}_s{ds}_{hs}"
    od = out_dir(reservoir, acq, D, variant, ds, hs)
    chr_val_arg = "--chr_val" if (reservoir == "genomic" and strategy != "random") else ""
    cache_arg = f"--reservoir_cache {cache_path}" if cache_path else ""
    vcache = val_cache(reservoir, acq)
    val_cache_arg = f"--reservoir_val_cache {vcache}" if vcache else ""
    llm_env = ""
    if is_llm:
        llm_env = (
            f'export LLM_MODEL="{model}"\n'
            f'export LLM_PROMPT_STYLE="{style}"\n'
            f'export LLM_CONTEXT="{context}"\n'
            f'export LLM_ALLOW_NOVEL_AXES="{novel}"'
        )
    # Reserve the fair-eval partition only for cache pools (the R×A study). genomic
    # uses real chr-val and the raw pool caps ~618k (no room for a disjoint 300k
    # retrain) — its transfer baseline reuses the existing genomic search.
    reserve_frac = RESERVE_EVAL_FRAC if cache_path else 0.0
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
export HP_POOL_RESERVE_EVAL_FRAC={reserve_frac}
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
    p = Path(f"/tmp/_s2_{label}.sh")
    p.write_text(script)
    r = subprocess.run([SBATCH, "--parsable", str(p)], capture_output=True, text=True, timeout=30)
    if r.returncode == 0:
        return r.stdout.strip().split(";")[0], ""
    return None, r.stderr.strip()


def main():
    dry = os.environ.get("DRY_RUN") == "1"
    vs = variants()

    inflight: set[str] = set()
    qcount = {"fast": 0, "default": 0, "slow_nice": 0}
    if not dry:
        r = subprocess.run(
            [SQUEUE, "--me", "-h", "--format=%j|%q", "--states=PD,R,CF"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        for ln in r.stdout.strip().split("\n"):
            if not ln.strip():
                continue
            name, _, q = ln.partition("|")
            inflight.add(name.strip())
            if q.strip() in qcount:
                qcount[q.strip()] += 1

    n_sub = n_skip = n_done = n_full = n_nocache = 0
    for D in DS:
        for reservoir in RESERVOIRS:
            for acq in ACQS:
                # genomic has no acquisition variants (it is the chr pool, real val);
                # only emit its default cell once.
                if reservoir == "genomic" and acq:
                    continue
                cache_path = pool_cache(reservoir, acq)
                if cache_path is not None and not Path(cache_path).exists():
                    print(
                        f"  SKIP {cell_tag(reservoir, acq)}: reservoir cache missing ({cache_path})"
                    )
                    n_nocache += 1
                    continue
                for ds, hs in SEEDS:
                    for variant, strategy, model, style, context, novel in vs:
                        label = f"s2_{cell_tag(reservoir, acq)}_d{D}_{variant}_s{ds}_{hs}"
                        od = out_dir(reservoir, acq, D, variant, ds, hs)
                        if is_complete(od, strategy):
                            n_done += 1
                            continue
                        if label in inflight:
                            n_skip += 1
                            continue
                        qos, wt = qos_walltime(D, bool(model))
                        if dry:
                            print(
                                f"  [DRY] {label} [{qos} {wt}] reserve={RESERVE_EVAL_FRAC if cache_path else 0.0}"
                            )
                            n_sub += 1
                            continue
                        if qcount.get(qos, 0) >= SUBMIT_CAP.get(qos, 0):
                            n_full += 1
                            continue
                        lbl, sh = job_script(
                            reservoir,
                            acq,
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
                        else:
                            print(f"  ERR {lbl}: {err[:160]}")

    verb = "Planned" if dry else "Submitted"
    print(
        f"\n{verb}: {n_sub}  Skipped(inflight): {n_skip}  AlreadyDone: {n_done}  "
        f"TierFull: {n_full}  NoCache: {n_nocache}"
    )
    if not dry:
        print(f"Queue counts after run: {qcount}")


if __name__ == "__main__":
    main()
