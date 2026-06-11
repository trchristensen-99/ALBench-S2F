"""LLM AutoResearch PROMPT SCREEN — a focused, thorough comparison of LLM-proposer
variants BEFORE the main STEP-1 strategy bake-off.

Why separate from submit_step1_bakeoff.py: the main bake-off compares STRATEGY FAMILIES
(random / optuna / evo / ray / LLM) on an equal GPU-seconds footing. Flooding it with
dozens of LLM variants would give the LLM family dozens of "lottery tickets" and bias the
STEP-1B all-subsets ElasticNet toward LLM atoms. Instead we screen the LLM proposer here,
across a WIDE matrix of {prompt style × model × novel-axes}, at a SMALL set of cells, then:
  (1) promote the most GPU-seconds-efficient style as the LLM family's single representative
      in the main bake-off, and
  (2) optionally feed configs from the best 2-3 styles into the STEP-1B ensemble pool.

Matrix (all env-overridable):
  styles   : default, explore, exploit, critic, diverse, neutral   (LLM_SCREEN_STYLES)
  models   : claude-opus-4-7, claude-sonnet-4-6                     (LLM_SCREEN_MODELS)
             (add claude-haiku-4-5-20251001 for a cheap-model arm)
  novel    : 0, 1  (LLM_ALLOW_NOVEL_AXES off/on)                    (LLM_SCREEN_NOVEL)
  cells    : genomic × D=30k × seeds {42:0,43:1}                    (LLM_SCREEN_RESERVOIRS/_DS/_SEEDS)
  depth    : ROUNDS=20 × PER_ROUND=2 = 40 models/variant/cell       (LLM_SCREEN_ROUNDS/_PER_ROUND)

COST WARNING: every LLM variant calls Claude via the CLI on HPC — it CONSUMES the Claude
usage cap. n_calls ≈ n_variants × n_cells × ROUNDS. Print the estimate, smoke first.

Usage:
  SMOKE_ONLY=1 python scripts/submit_llm_prompt_screen.py     # 1 variant, 2 rounds, genomic
  python scripts/submit_llm_prompt_screen.py                  # full matrix
  LLM_SCREEN_STYLES=default,neutral LLM_SCREEN_MODELS=claude-sonnet-4-6 python scripts/submit_llm_prompt_screen.py
  DRY_RUN=1 python scripts/submit_llm_prompt_screen.py        # print plan + call estimate, submit nothing
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
BIN = "/cm/shared/apps/slurm/current/bin"
SBATCH = f"{BIN}/sbatch"
SQUEUE = f"{BIN}/squeue"

OUT_ROOT = os.environ.get("LLM_SCREEN_OUT_ROOT", f"{REPO}/outputs/hp_llm_screen_e100")

STYLES = os.environ.get(
    "LLM_SCREEN_STYLES", "default,explore,exploit,critic,diverse,neutral"
).split(",")
MODELS = os.environ.get("LLM_SCREEN_MODELS", "claude-opus-4-7,claude-sonnet-4-6").split(",")
NOVEL = [x.strip() for x in os.environ.get("LLM_SCREEN_NOVEL", "0,1").split(",")]
RESERVOIRS = os.environ.get("LLM_SCREEN_RESERVOIRS", "genomic").split(",")
DS = [int(x) for x in os.environ.get("LLM_SCREEN_DS", "30000").split(",")]
SEEDS = [
    tuple(int(x) for x in pair.split(":"))
    for pair in os.environ.get("LLM_SCREEN_SEEDS", "42:0,43:1").split(",")
]
ROUNDS = int(os.environ.get("LLM_SCREEN_ROUNDS", "20"))
PER_ROUND = int(os.environ.get("LLM_SCREEN_PER_ROUND", "2"))
TEMPERATURE = os.environ.get("LLM_SCREEN_TEMPERATURE", "")  # "" = engine default

EPOCHS = 100
PATIENCE = 15
MIN_DELTA = 1e-3
POOL_D = 1_000_000
DATA_SEED_REF = 42

SUBMIT_CAP = {"fast": 16, "default": 16, "slow_nice": 1000}


def model_short(model: str) -> str:
    for tag in ("opus", "sonnet", "haiku"):
        if tag in model:
            return tag
    return model.replace("/", "_")


def variants() -> list[tuple[str, str, str, str]]:
    """(label, model, style, novel) over the full screen matrix."""
    out = []
    for style in STYLES:
        for model in MODELS:
            for nv in NOVEL:
                label = f"llm_{style}_{model_short(model)}_nv{nv}"
                out.append((label, model, style, nv))
    return out


def pool_cache(reservoir: str) -> str | None:
    if reservoir == "genomic":
        return None
    return f"{REPO}/outputs/reservoir_cache/k562_{reservoir}_d{POOL_D}_seed{DATA_SEED_REF}.npz"


def out_dir(reservoir: str, D: int, variant: str, ds: int, hs: int) -> Path:
    return Path(f"{OUT_ROOT}/k562_{reservoir}_d{D}/seed{ds}_{hs}/{variant}")


def is_complete(od: Path) -> bool:
    if not od.exists():
        return False
    if (od / ".screen_done").exists():
        return True
    return len(list(od.glob("*_meta.json"))) >= ROUNDS * PER_ROUND


def qos_chain(D: int) -> list[tuple[str, str]]:
    # LLM never uses `fast` (rate-limit waits would burn the 4h cap). Prefer default
    # at small D, else slow_nice; always fall back to slow_nice.
    pref = ("default", "12:00:00") if D <= 100_000 else ("slow_nice", "2-00:00:00")
    chain = [pref]
    if pref[0] != "slow_nice":
        chain.append(("slow_nice", "2-00:00:00"))
    return chain


def job_script(reservoir, D, variant, model, style, novel, cache_path, ds, hs, qos, wt):
    label = f"lps_{reservoir}_d{D}_{variant}_s{ds}_{hs}"
    od = out_dir(reservoir, D, variant, ds, hs)
    chr_val_arg = "--chr_val" if reservoir == "genomic" else ""
    cache_arg = f"--reservoir_cache {cache_path}" if cache_path else ""
    temp_env = f'export LLM_TEMPERATURE="{TEMPERATURE}"\n' if TEMPERATURE else ""
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
export LLM_MODEL="{model}"
export LLM_PROMPT_STYLE="{style}"
export LLM_ALLOW_NOVEL_AXES="{novel}"
{temp_env}ATTEMPT=0
while true; do
  uv run --no-sync python experiments/scaling_hp_search.py \\
    --strategies llm_autoresearch --rounds {ROUNDS} --per_strategy_per_round {PER_ROUND} \\
    --D {D} --ref_only {chr_val_arg} {cache_arg} \\
    --data_seed {ds} --hp_seed {hs} \\
    --epochs {EPOCHS} --early_stop_patience {PATIENCE} --min_delta {MIN_DELTA} \\
    --out_dir {od}
  rc=$?
  if [ $rc -eq 0 ]; then touch {od}/.screen_done; echo "=== DONE rc=0 ==="; break; fi
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
    p = Path(f"/tmp/_lps_{label}.sh")
    p.write_text(script)
    r = subprocess.run([SBATCH, "--parsable", str(p)], capture_output=True, text=True, timeout=30)
    if r.returncode == 0:
        return r.stdout.strip().split(";")[0], ""
    return None, r.stderr.strip()


def main():
    smoke = os.environ.get("SMOKE_ONLY") == "1"
    dry = os.environ.get("DRY_RUN") == "1"
    reservoirs = ["genomic"] if smoke else RESERVOIRS
    ds_list = [30000] if smoke else DS
    seeds = [(42, 0)] if smoke else SEEDS
    rounds = 2 if smoke else ROUNDS
    vlist = variants()
    if smoke:
        vlist = vlist[:1]

    n_cells = len(reservoirs) * len(ds_list) * len(seeds)
    n_jobs = len(vlist) * n_cells
    est_calls = n_jobs * rounds
    print(f"=== LLM prompt screen: {len(vlist)} variants × {n_cells} cells = {n_jobs} jobs ===")
    print(f"=== ROUNDS={rounds} → ~{est_calls} Claude CLI calls (CONSUMES the usage cap) ===")
    for label, model, style, nv in vlist:
        print(f"  {label:32s} model={model} style={style} novel={nv}")
    if dry:
        print("=== DRY_RUN=1: nothing submitted ===")
        return

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

    n_sub = n_skip = n_done = 0
    for reservoir in reservoirs:
        cache_path = pool_cache(reservoir)
        if cache_path is not None and not Path(cache_path).exists():
            print(f"  SKIP {reservoir}: reservoir cache missing ({cache_path})")
            continue
        for D in ds_list:
            for ds, hs in seeds:
                for variant, model, style, nv in vlist:
                    label = f"lps_{reservoir}_d{D}_{variant}_s{ds}_{hs}"
                    od = out_dir(reservoir, D, variant, ds, hs)
                    if is_complete(od):
                        n_done += 1
                        continue
                    if label in inflight:
                        n_skip += 1
                        continue
                    submitted = False
                    for qos, wt in qos_chain(D):
                        if qcount.get(qos, 0) >= SUBMIT_CAP.get(qos, 0):
                            continue
                        lbl, sh = job_script(
                            reservoir, D, variant, model, style, nv, cache_path, ds, hs, qos, wt
                        )
                        jid, err = sbatch(lbl, sh)
                        if jid:
                            n_sub += 1
                            qcount[qos] = qcount.get(qos, 0) + 1
                            submitted = True
                            print(f"  SUB {label} -> {jid} ({qos})")
                            break
                        print(f"  ERR {label} ({qos}): {err}")
                    if not submitted:
                        print(f"  HOLD {label}: all qos at cap")
    print(f"=== submitted={n_sub} skip_inflight={n_skip} already_done={n_done} ===")


if __name__ == "__main__":
    main()
