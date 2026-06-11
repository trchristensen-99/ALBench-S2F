"""Generate the per-reservoir, transform-matched HELD-OUT val caches the bake-off
consumes via `--reservoir_val_cache` (scripts/submit_step1_bakeoff.py:val_cache).

WHY: a genomic-derived reservoir (motif_planted_v2, dinuc_shuffle, ...) must be
validated on a val set built with the SAME transform as its train pool — e.g. motifs
planted onto HELD-OUT (chr19/21/X) backgrounds, NOT the plain genomic val set. We get
that by running generate_reservoir_cache.py with `--background_cache chr_val_ref_only.npz`
(RESERVOIR_BG_CACHE so self-loading samplers like MotifPlantedV2Sampler honor it too),
oracle-labeling with the CANONICAL oracle, and stamping `oracle_id=full856k_clean` so the
search's contamination guard accepts it. Sized to ~11%% of the genomic train pool
(VAL_D ≈ 0.11 × 314,981 ≈ 34,648) so any run can subsample val_frac×D from it.

Oracle labeling is GPU-only (NO Claude/Max usage cap) → safe to run overnight.

One GPU job per non-genomic bake-off reservoir. genomic is skipped (it validates on the
real chr holdout via --chr_val, no transform-matched cache needed). Idempotent: skips a
reservoir whose val cache already exists or whose job is in-flight.

Usage:
  DRY_RUN=1 python scripts/submit_reservoir_val_sets.py     # print plan, submit nothing
  python scripts/submit_reservoir_val_sets.py               # submit
  RESERVOIR_VAL_RESERVOIRS=motif_planted_v2 python scripts/submit_reservoir_val_sets.py
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
BIN = "/cm/shared/apps/slurm/current/bin"
SBATCH = f"{BIN}/sbatch"
SQUEUE = f"{BIN}/squeue"

# Mirror submit_step1_bakeoff's non-genomic reservoirs (genomic excluded — uses chr_val).
DEFAULT_RESERVOIRS = "motif_planted_v2,dinuc_shuffle"
RESERVOIRS = [
    r for r in os.environ.get("RESERVOIR_VAL_RESERVOIRS", DEFAULT_RESERVOIRS).split(",") if r
]

DATA_SEED_REF = 42  # matches val_cache() filename in submit_step1_bakeoff.py
# ~11% of the 314,981-seq genomic train pool; subsample-able down to val_frac×D per run.
VAL_D = int(os.environ.get("RESERVOIR_VAL_D", "34648"))
ORACLE = os.environ.get("RESERVOIR_VAL_ORACLE", "ag_s2")
ORACLE_STAMP = "full856k_clean"  # canonical; lets the search contamination guard accept it
BG_CACHE = f"{REPO}/outputs/chr_split_cache/chr_val_ref_only.npz"

OUT_DIR = f"{REPO}/outputs/reservoir_val_cache"

SUBMIT_CAP = {"fast": 16, "default": 16, "slow_nice": 1000}


def out_path(reservoir: str) -> str:
    return f"{OUT_DIR}/k562_{reservoir}_val_seed{DATA_SEED_REF}.npz"


def job_script(reservoir: str, qos: str, wt: str) -> tuple[str, str]:
    label = f"rvc_{reservoir}_s{DATA_SEED_REF}"
    out = out_path(reservoir)
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
echo "=== job start (SLURM_RESTART_COUNT=${{SLURM_RESTART_COUNT:-0}}) ==="
source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/experiments"
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1
uv run --no-sync python scripts/generate_reservoir_cache.py \\
  --task k562 --reservoir {reservoir} --D {VAL_D} --seed {DATA_SEED_REF} \\
  --oracle {ORACLE} --oracle_id_stamp {ORACLE_STAMP} \\
  --background_cache {BG_CACHE} \\
  --out {out}
rc=$?
if [ $rc -eq 0 ]; then echo "=== DONE rc=0 ==="; else echo "=== FAILED rc=$rc ==="; fi
exit $rc
"""
    return label, script


def sbatch(label: str, script: str) -> tuple[str | None, str]:
    p = Path(f"/tmp/_rvc_{label}.sh")
    p.write_text(script)
    r = subprocess.run([SBATCH, "--parsable", str(p)], capture_output=True, text=True, timeout=30)
    if r.returncode == 0:
        return r.stdout.strip().split(";")[0], ""
    return None, r.stderr.strip()


def main() -> None:
    dry = os.environ.get("DRY_RUN") == "1"
    print(f"=== reservoir val-set generation: {len(RESERVOIRS)} reservoir(s), VAL_D={VAL_D:,} ===")
    print(f"    background={BG_CACHE}")
    print(f"    oracle={ORACLE} stamp={ORACLE_STAMP}  out={OUT_DIR}/")
    for r in RESERVOIRS:
        print(f"  {r:20s} -> {out_path(r)}")
    if dry:
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
    for reservoir in RESERVOIRS:
        label = f"rvc_{reservoir}_s{DATA_SEED_REF}"
        if Path(out_path(reservoir)).exists():
            print(f"  DONE {reservoir}: val cache exists")
            n_done += 1
            continue
        if label in inflight:
            print(f"  SKIP {reservoir}: in-flight")
            n_skip += 1
            continue
        # GPU labeling job; prefer default (12h) then fall back to slow_nice.
        submitted = False
        for qos, wt in [("default", "12:00:00"), ("slow_nice", "2-00:00:00")]:
            if qcount.get(qos, 0) >= SUBMIT_CAP.get(qos, 0):
                continue
            lbl, sh = job_script(reservoir, qos, wt)
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
