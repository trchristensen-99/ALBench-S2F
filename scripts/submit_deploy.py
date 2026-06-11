"""PHASE-4 DEPLOY launcher — fan the FROZEN N* recipe across every
reservoir x acquisition x D x seed cell, NO search.

Each cell runs experiments/deploy_train.py: train the N* configs once, then ElasticNet
the trained members on THAT cell's own val preds (weights refit per cell, never frozen
— see albench/ensemble/stack.py). Pure GPU, no Claude calls, so safe to run wide.

The recipe is per-D: RECIPE_DIR/deploy_recipe_d{D}.json holds the full HP dicts locked
by Phase 3. Each combo's reservoir_cache (the acquisition-selected, oracle-labeled pool)
+ reservoir_val_cache (transform-matched chr19/21/X holdout) follow the same path
convention as submit_step1_bakeoff.py. genomic validates on real --chr_val instead.

Acquisition axis: a cell is "reservoir" or "reservoir:acquisition". When an acquisition
tag is given it is folded into the cache filename + cell dir (k562_{res}_{acq}_...), so
this is forward-compatible with per-acquisition pools without inventing them here.

Idempotent: skips a cell whose .deploy_done exists or whose job is in-flight.

Usage:
  DRY_RUN=1 python scripts/submit_deploy.py
  DEPLOY_RESERVOIRS=genomic,motif_planted_v2 DEPLOY_DS=30000 python scripts/submit_deploy.py
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
BIN = "/cm/shared/apps/slurm/current/bin"
SBATCH = f"{BIN}/sbatch"
SQUEUE = f"{BIN}/squeue"

OUT_ROOT = os.environ.get("DEPLOY_OUT_ROOT", f"{REPO}/outputs/deploy_e100")
RECIPE_DIR = os.environ.get("DEPLOY_RECIPE_DIR", f"{REPO}/configs/deploy_recipes")

# "genomic" uses the chr genomic train pool (no reservoir_cache, real --chr_val).
DEFAULT_RESERVOIRS = "genomic,motif_planted_v2,dinuc_shuffle"
RESERVOIRS = [r for r in os.environ.get("DEPLOY_RESERVOIRS", DEFAULT_RESERVOIRS).split(",") if r]
# Optional acquisition tags; "" = the single default (none) acquisition.
ACQUISITIONS = [a for a in os.environ.get("DEPLOY_ACQUISITIONS", "").split(",") if a] or [""]
DS = [int(d) for d in os.environ.get("DEPLOY_DS", "30000,300000").split(",") if d]
# data_seeds for deploy reps (deploy has no hp_seed — there is no search).
DATA_SEEDS = [int(s) for s in os.environ.get("DEPLOY_DATA_SEEDS", "42,43,44").split(",") if s]

POOL_D = int(os.environ.get("DEPLOY_POOL_D", "1000000"))
DATA_SEED_REF = 42  # reservoir-cache seed in the cache filename
EPOCHS = int(os.environ.get("DEPLOY_EPOCHS", "100"))
PATIENCE = int(os.environ.get("DEPLOY_PATIENCE", "15"))
MIN_DELTA = 1e-3

SUBMIT_CAP = {"fast": 16, "default": 16, "slow_nice": 1000}


def cell_tag(reservoir: str, acquisition: str) -> str:
    return reservoir if not acquisition else f"{reservoir}_{acquisition}"


def recipe_path(D: int) -> str:
    return f"{RECIPE_DIR}/deploy_recipe_d{D}.json"


def pool_cache(reservoir: str, acquisition: str) -> str | None:
    if reservoir == "genomic" and not acquisition:
        return None
    return f"{REPO}/outputs/reservoir_cache/k562_{cell_tag(reservoir, acquisition)}_d{POOL_D}_seed{DATA_SEED_REF}.npz"


def val_cache(reservoir: str, acquisition: str) -> str | None:
    """Transform-matched held-out val cache; None for genomic (uses --chr_val).
    Returns the path only if it exists; otherwise the run falls back to the 10% holdout."""
    if reservoir == "genomic" and not acquisition:
        return None
    p = f"{REPO}/outputs/reservoir_val_cache/k562_{cell_tag(reservoir, acquisition)}_val_seed{DATA_SEED_REF}.npz"
    return p if Path(p).exists() else None


def out_dir(reservoir: str, acquisition: str, D: int, ds: int) -> Path:
    return Path(f"{OUT_ROOT}/k562_{cell_tag(reservoir, acquisition)}_d{D}/seed{ds}")


def is_complete(od: Path) -> bool:
    return od.exists() and (od / ".deploy_done").exists()


def qos_walltime(D: int) -> tuple[str, str]:
    return ("default", "12:00:00") if D <= 30_000 else ("slow_nice", "2-00:00:00")


def job_script(reservoir, acquisition, D, ds, qos, wt) -> tuple[str, str]:
    tag = cell_tag(reservoir, acquisition)
    label = f"dep_{tag}_d{D}_s{ds}"
    od = out_dir(reservoir, acquisition, D, ds)
    recipe = recipe_path(D)
    chr_val_arg = "--chr_val" if (reservoir == "genomic" and not acquisition) else ""
    cache = pool_cache(reservoir, acquisition)
    cache_arg = f"--reservoir_cache {cache}" if cache else ""
    vcache = val_cache(reservoir, acquisition)
    val_cache_arg = f"--reservoir_val_cache {vcache}" if vcache else ""
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
echo "=== deploy job start (SLURM_RESTART_COUNT=${{SLURM_RESTART_COUNT:-0}}; resume skips done members) ==="
source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/experiments"
export HP_FAST=1
export HP_CACHE_DIR="$PWD/outputs/tensor_cache"
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1
uv run --no-sync python experiments/deploy_train.py \\
  --recipe {recipe} --out_dir {od} \\
  --D {D} --data_seed {ds} --ref_only {chr_val_arg} {cache_arg} {val_cache_arg} \\
  --epochs {EPOCHS} --early_stop_patience {PATIENCE} --min_delta {MIN_DELTA}
rc=$?
if [ $rc -eq 0 ]; then echo "=== DONE rc=0 ==="; else echo "=== FAILED rc=$rc ==="; fi
exit $rc
"""
    return label, script


def sbatch(label: str, script: str) -> tuple[str | None, str]:
    p = Path(f"/tmp/_dep_{label}.sh")
    p.write_text(script)
    r = subprocess.run([SBATCH, "--parsable", str(p)], capture_output=True, text=True, timeout=30)
    if r.returncode == 0:
        return r.stdout.strip().split(";")[0], ""
    return None, r.stderr.strip()


def main() -> None:
    dry = os.environ.get("DRY_RUN") == "1"
    cells = [
        (res, acq, D, ds)
        for res in RESERVOIRS
        for acq in ACQUISITIONS
        for D in DS
        for ds in DATA_SEEDS
    ]
    print(f"=== deploy: {len(cells)} cells (no search; GPU-only) ===")
    print(f"    reservoirs={RESERVOIRS} acq={ACQUISITIONS} D={DS} seeds={DATA_SEEDS}")
    print(f"    recipes={RECIPE_DIR}/deploy_recipe_d<D>.json  out={OUT_ROOT}/")

    for D in sorted({c[2] for c in cells}):
        if not Path(recipe_path(D)).exists():
            print(
                f"  WARN: missing recipe {recipe_path(D)} — D={D} cells will fail until it exists"
            )
    if dry:
        for res, acq, D, ds in cells:
            print(f"  {cell_tag(res, acq):28s} D={D:<7d} seed={ds}")
        print("=== DRY_RUN=1: nothing submitted ===")
        return

    sq = subprocess.run(
        [SQUEUE, "--me", "-h", "--format=%j|%q", "--states=PD,R,CF"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    inflight = set()
    qcount = {"fast": 0, "default": 0, "slow_nice": 0}
    for ln in sq.stdout.strip().split("\n"):
        if not ln.strip():
            continue
        name, _, q = ln.partition("|")
        inflight.add(name.strip())
        if q.strip() in qcount:
            qcount[q.strip()] += 1

    n_sub = n_skip = n_done = 0
    for res, acq, D, ds in cells:
        label = f"dep_{cell_tag(res, acq)}_d{D}_s{ds}"
        od = out_dir(res, acq, D, ds)
        if is_complete(od):
            n_done += 1
            continue
        if label in inflight:
            n_skip += 1
            continue
        pref_qos, pref_wt = qos_walltime(D)
        chain = [(pref_qos, pref_wt)]
        if pref_qos != "slow_nice":
            chain.append(("slow_nice", "2-00:00:00"))
        submitted = False
        for qos, wt in chain:
            if qcount.get(qos, 0) >= SUBMIT_CAP.get(qos, 0):
                continue
            lbl, sh = job_script(res, acq, D, ds, qos, wt)
            jid, err = sbatch(lbl, sh)
            if jid:
                n_sub += 1
                qcount[qos] = qcount.get(qos, 0) + 1
                print(f"  SUB {label} -> {jid} [{qos}]")
                submitted = True
                break
            print(f"  ERR {label} [{qos}]: {err[:160]}")
        if not submitted:
            print(f"  HOLD {label}: all qos at cap")
    print(f"=== submitted={n_sub} skip_inflight={n_skip} already_done={n_done} ===")


if __name__ == "__main__":
    main()
