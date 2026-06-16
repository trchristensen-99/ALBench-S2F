"""LLM PROMPT ABLATION — a focused follow-up to the Phase-0 prompt screen that
isolates the factors that confound the LLM family's bake-off comparison.

Two families of single-axis probes, bundled so they RANK TOGETHER on one GPU-seconds
axis (same model/seed/reservoir/depth as the screen):

(A) CONTEXT SCAFFOLDING — how much domain steering the prompt carries
    (ADVANCED_GUIDANCE, cross-experiment KB priors, editorial closers). The well-
    sampled screen winners are `default` and `exploit`, so we vary CONTEXT on THOSE
    only (full → nokb → none) plus two controls — `blank` (black-box: no domain
    framing) and `misguided` (deliberately wrong guidance).
      Interpretation: if `default/full` beats `blank`, the SCAFFOLDING is doing the
      work — the LLM's edge over evo/random is partly human-injected domain knowledge
      + prior-run KB those strategies don't get (a fairness confound). If `blank`
      matches, the edge is intrinsic in-context optimization. `misguided` vs `blank`
      tests whether the model anchors on bad guidance or overrides it from observations.

(B) PROPOSER MECHANICS — does the model actually USE the feedback loop, and how does
    the way we present/consume it matter. Held at (default, none) so the run's OWN
    observations are the only signal the probe manipulates:
      - shuffle  : permute the score<->config pairing shown to the model. If best-
                   single still climbs, the apparent optimization is regression-to-
                   better-sampling, not feedback use. (LLM_SHUFFLE_FEEDBACK)
      - hist5/histfull : show top-5 vs ALL observations instead of top-30 — does
                   history DEPTH matter? (LLM_HISTORY_MAX)
      - chrono   : present history as-observed instead of best-first — does RANKING/
                   recency framing matter? (LLM_HISTORY_ORDER)
      - n1 / n5  : 1 vs 5 proposals per call (baseline is n=2) — does batch width
                   trade off against rounds of feedback?

COST: every cell calls Claude (CONSUMES the usage cap). n_calls ≈ Σ rounds × per_round.
Print the estimate; DRY_RUN=1 to plan only; SMOKE_ONLY=1 for 1 cell × 2 rounds.

Usage:
  DRY_RUN=1   python scripts/submit_llm_prompt_ablation.py
  SMOKE_ONLY=1 python scripts/submit_llm_prompt_ablation.py
  python scripts/submit_llm_prompt_ablation.py
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
BIN = "/cm/shared/apps/slurm/current/bin"
SBATCH = f"{BIN}/sbatch"
SQUEUE = f"{BIN}/squeue"

# Phase read at import so the follow-up writes to its OWN root by default — it runs an
# EQUAL-pool (24-config) grid and must not reuse the main run's 20-config cells.
PHASE = os.environ.get("LLM_ABL_PHASE", "main").strip().lower()
if PHASE == "confirm":
    _DEFAULT_ROOT = f"{REPO}/outputs/hp_llm_ablation_confirm_e100"
elif PHASE == "followup":
    _DEFAULT_ROOT = f"{REPO}/outputs/hp_llm_ablation_followup_e100"
else:
    _DEFAULT_ROOT = f"{REPO}/outputs/hp_llm_ablation_e100"
OUT_ROOT = os.environ.get("LLM_ABL_OUT_ROOT", _DEFAULT_ROOT)

MODEL = os.environ.get("LLM_ABL_MODEL", "claude-sonnet-4-6")
NOVEL = os.environ.get("LLM_ABL_NOVEL", "0")  # hold off so CONTEXT alone controls background
RESERVOIR = "genomic"
D = int(os.environ.get("LLM_ABL_D", "30000"))
DATA_SEED, HP_SEED = 42, 0
ROUNDS = int(os.environ.get("LLM_ABL_ROUNDS", "10"))  # plateau hits by ~r8-10
PER_ROUND = int(os.environ.get("LLM_ABL_PER_ROUND", "2"))
TEMPERATURE = os.environ.get("LLM_ABL_TEMPERATURE", "")

EPOCHS = 100
PATIENCE = 15
MIN_DELTA = 1e-3

SUBMIT_CAP = {"fast": 16, "default": 16, "slow_nice": 1000}


@dataclass(frozen=True)
class Cell:
    """One ablation cell. ``label`` is the variant subdir (unique per cell).
    ``per_round`` defaults to the global PER_ROUND; ``env`` holds extra probe
    exports (LLM_SHUFFLE_FEEDBACK / LLM_HISTORY_MAX / LLM_HISTORY_ORDER / ...).
    Total proposals shown to the model per cell = rounds × per_round."""

    label: str
    style: str
    context: str
    novel: str = NOVEL
    per_round: int = PER_ROUND
    env: dict[str, str] = field(default_factory=dict)
    data_seed: int = DATA_SEED
    hp_seed: int = HP_SEED


# (A) CONTEXT SCAFFOLDING. No-prior styles (blank/misguided) force context="none".
CONTEXT_CELLS = [
    Cell("llm_default_ctxfull", "default", "full"),
    Cell("llm_default_ctxnokb", "default", "nokb"),
    Cell("llm_default_ctxnone", "default", "none"),
    Cell("llm_exploit_ctxfull", "exploit", "full"),
    Cell("llm_exploit_ctxnokb", "exploit", "nokb"),
    Cell("llm_exploit_ctxnone", "exploit", "none"),
    Cell("llm_blank_ctxnone", "blank", "none"),
    Cell("llm_misguided_ctxnone", "misguided", "none"),
]

# (B) PROPOSER MECHANICS. Baseline = llm_default_ctxnone above (default style, context
# none, n=2): each probe flips exactly ONE axis off that baseline so it ranks against it.
PROBE_CELLS = [
    # #1 shuffled-feedback control (uninformative pairing).
    Cell("llm_default_ctxnone_shuffle", "default", "none", env={"LLM_SHUFFLE_FEEDBACK": "1"}),
    # #2 history depth + ordering.
    Cell("llm_default_ctxnone_hist5", "default", "none", env={"LLM_HISTORY_MAX": "5"}),
    Cell("llm_default_ctxnone_histfull", "default", "none", env={"LLM_HISTORY_MAX": "full"}),
    Cell("llm_default_ctxnone_chrono", "default", "none", env={"LLM_HISTORY_ORDER": "chrono"}),
    # #3 proposals-per-call (1 and 5 vs the n=2 baseline). Rounds rescaled so total
    # proposals (rounds × per_round) ~matches the baseline budget (20).
    Cell("llm_default_ctxnone_n1", "default", "none", per_round=1),
    Cell("llm_default_ctxnone_n5", "default", "none", per_round=5),
    # #4 no-history control — strict bookend to shuffle: removes the feedback channel
    # entirely (pure prior sampling). If it matches baseline, feedback isn't being used.
    Cell("llm_default_ctxnone_nohist", "default", "none", env={"LLM_HIDE_HISTORY": "1"}),
    # #5 worst-first history — anti-elitist: failures up top. Does it learn from bad configs?
    Cell("llm_default_ctxnone_worst", "default", "none", env={"LLM_HISTORY_ORDER": "worst"}),
    # #6 determinism replicate — baseline at a fresh proposer seed (same data) to measure
    # the LLM proposer's run-to-run noise floor (calibrates how big a probe gap must be).
    Cell("llm_default_ctxnone_rep", "default", "none", hp_seed=1),
]

CELLS = CONTEXT_CELLS + PROBE_CELLS

# (C) PHASE-0 FOLLOW-UP. The single-seed ablation surfaced two strongest levers:
# proposals-per-call (n1<n2<n5) and — for the ENSEMBLE metric — the "diverse" style
# (ensemble-decorrelation proposer). Their single-seed deltas (~0.005-0.03) sit near the
# ~0.008 run-to-run noise floor, so re-test the winners across 3 covaried seeds to clear
# it. n=8 extends the proposals-per-call trend; diverse×{n2,n8} isolates the ensemble-aware
# proposer and the combined lever. Includes a multi-seed default baseline (the 42:0 cell is
# reused from the main ablation). Gated: LLM_ABL_PHASE=followup.
# FOLLOWUP_BUDGET sets the per-cell pool size (24, divisible by both batch widths 2 and 8)
# so every cell ensembles an EQUAL number of configs.
FOLLOWUP_BUDGET = 24
FOLLOWUP_SEEDS = [(42, 0), (43, 1), (44, 2)]
FOLLOWUP_BASE = [
    ("llm_default_ctxnone", "default", "none", 2, {}),  # multi-seed baseline
    ("llm_default_ctxnone_n8", "default", "none", 8, {}),  # wider batch
    ("llm_diverse_ctxnone_n2", "diverse", "none", 2, {}),  # ensemble-aware proposer
    ("llm_diverse_ctxnone_n8", "diverse", "none", 8, {}),  # ensemble-aware + wide batch
]
FOLLOWUP_CELLS = [
    Cell(label, style, ctx, per_round=n, env=env, data_seed=ds, hp_seed=hs)
    for (label, style, ctx, n, env) in FOLLOWUP_BASE
    for (ds, hs) in FOLLOWUP_SEEDS
]

# (D) CONFIRMATORY DEPLOY BUNDLE + ROUNDS-PLATEAU (Jun 16). The THREE chosen deploy proposers,
# each held at the novel-axis setting where it was actually validated (exploit/diverse won with
# novel axes ON; critic won at nv0), all at ctxnone (fair bakeoff) on Sonnet. Goals: (1) confirm
# the diverse/decorrelation win + the two top screen personas across 3 covaried seeds at a
# DEPLOY-realistic batch width, and (2) run to a LONG round horizon to locate the true ensemble-
# oracle plateau vs #rounds (the followup only reached 12). Gated: LLM_ABL_PHASE=confirm. Unlike
# the followup, rounds are NOT rescaled to a fixed pool — every cell runs CONFIRM_ROUNDS rounds
# so the per-round curve has a long, equal x-axis. Pool/cell = CONFIRM_ROUNDS × CONFIRM_PER_ROUND.
CONFIRM_ROUNDS = int(os.environ.get("LLM_CONFIRM_ROUNDS", "25"))
CONFIRM_PER_ROUND = int(os.environ.get("LLM_CONFIRM_PER_ROUND", "3"))
CONFIRM_SEEDS = [(42, 0), (43, 1), (44, 2)]
CONFIRM_BASE = [
    ("llm_exploit_nv1", "exploit", "none", "1"),  # strongest screen persona (oracle@B 0.747)
    ("llm_critic_nv0", "critic", "none", "0"),  # 2nd, distinct style (0.740), decorrelates
    ("llm_diverse_nv0", "diverse", "none", "0"),  # ensemble-decorrelation win (+0.016, 3 seeds)
    ("llm_diverse_nv1", "diverse", "none", "1"),  # diverse @ nv1 (screen: +0.005 vs nv0, in-noise)
    (
        "llm_explore_nv1",
        "explore",
        "none",
        "1",
    ),  # 3rd-slot contender; HP-space coverage + novel axes ON
]
CONFIRM_CELLS = [
    Cell(label, style, ctx, novel=nv, per_round=CONFIRM_PER_ROUND, data_seed=ds, hp_seed=hs)
    for (label, style, ctx, nv) in CONFIRM_BASE
    for (ds, hs) in CONFIRM_SEEDS
]


def out_dir(cell: Cell) -> Path:
    return Path(
        f"{OUT_ROOT}/k562_{RESERVOIR}_d{D}/seed{cell.data_seed}_{cell.hp_seed}/{cell.label}"
    )


def job_label(cell: Cell) -> str:
    return f"lpa_{RESERVOIR}_d{D}_{cell.label}_s{cell.data_seed}_{cell.hp_seed}"


def cell_rounds(cell: Cell, rounds: int) -> int:
    """Hold total proposals (rounds × per_round) ~constant across per_round so the
    probe varies batch WIDTH at a fixed evaluation budget, not total compute.
    EXCEPTION: the confirm phase fixes #rounds (not the pool) so the per-round plateau
    curve has a long, equal x-axis — return rounds verbatim."""
    if PHASE == "confirm":
        return rounds
    budget = rounds * PER_ROUND
    return max(1, round(budget / cell.per_round))


def is_complete(cell: Cell, rounds: int) -> bool:
    od = out_dir(cell)
    if not od.exists():
        return False
    if (od / ".ablation_done").exists():
        return True
    return len(list(od.glob("*_meta.json"))) >= cell_rounds(cell, rounds) * cell.per_round


def job_script(cell: Cell, rounds: int, qos: str, wt: str) -> tuple[str, str]:
    label = job_label(cell)
    od = out_dir(cell)
    rnds = cell_rounds(cell, rounds)
    temp_env = f'export LLM_TEMPERATURE="{TEMPERATURE}"\n' if TEMPERATURE else ""
    probe_env = "".join(f'export {k}="{v}"\n' for k, v in sorted(cell.env.items()))
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
export LLM_MODEL="{MODEL}"
export LLM_PROMPT_STYLE="{cell.style}"
export LLM_CONTEXT="{cell.context}"
export LLM_ALLOW_NOVEL_AXES="{cell.novel}"
{probe_env}{temp_env}ATTEMPT=0
while true; do
  uv run --no-sync python experiments/scaling_hp_search.py \\
    --strategies llm_autoresearch --rounds {rnds} --per_strategy_per_round {cell.per_round} \\
    --D {D} --ref_only --chr_val \\
    --data_seed {cell.data_seed} --hp_seed {cell.hp_seed} \\
    --epochs {EPOCHS} --early_stop_patience {PATIENCE} --min_delta {MIN_DELTA} \\
    --out_dir {od}
  rc=$?
  if [ $rc -eq 0 ]; then touch {od}/.ablation_done; echo "=== DONE rc=0 ==="; break; fi
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
    p = Path(f"/tmp/_lpa_{label}.sh")
    p.write_text(script)
    r = subprocess.run([SBATCH, "--parsable", str(p)], capture_output=True, text=True, timeout=30)
    if r.returncode == 0:
        return r.stdout.strip().split(";")[0], ""
    return None, r.stderr.strip()


def main() -> None:
    smoke = os.environ.get("SMOKE_ONLY") == "1"
    dry = os.environ.get("DRY_RUN") == "1"
    if PHASE == "confirm":
        base_cells = CONFIRM_CELLS
    elif PHASE == "followup":
        base_cells = FOLLOWUP_CELLS
    else:
        base_cells = CELLS
    cells = base_cells[:1] if smoke else base_cells
    # Follow-up holds the pool at FOLLOWUP_BUDGET (24) configs so n=2 and n=8 yield EQUAL
    # pools (24 = 12×2 = 3×8); 24 is divisible by both batch widths, unlike the main run's
    # budget of 20. cell_rounds derives budget = rounds × PER_ROUND, so rounds = 12 → 24.
    if smoke:
        rounds = 2
    elif PHASE == "confirm":
        rounds = CONFIRM_ROUNDS
    elif PHASE == "followup":
        rounds = FOLLOWUP_BUDGET // PER_ROUND
    else:
        rounds = ROUNDS

    est_calls = sum(cell_rounds(c, rounds) for c in cells)
    print(
        f"=== LLM prompt ablation: {len(cells)} cells (base rounds={rounds}, per_round={PER_ROUND}) ==="
    )
    print(
        f"=== model={MODEL} novel={NOVEL} reservoir={RESERVOIR} D={D} seed={DATA_SEED}:{HP_SEED} ==="
    )
    print(f"=== ~{est_calls} Claude CLI calls (CONSUMES the usage cap) ===")
    for c in cells:
        rnds = cell_rounds(c, rounds)
        extra = " ".join(f"{k}={v}" for k, v in sorted(c.env.items())) or "-"
        print(
            f"  {c.label:32s} style={c.style:9s} ctx={c.context:5s} "
            f"n={c.per_round} rounds={rnds:2d} calls={rnds}  {extra}"
        )
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
    for cell in cells:
        label = job_label(cell)
        if is_complete(cell, rounds):
            n_done += 1
            continue
        if label in inflight:
            n_skip += 1
            continue
        submitted = False
        # Small D → default (12h) first, then slow_nice. LLM never uses fast.
        for qos, wt in [("default", "12:00:00"), ("slow_nice", "2-00:00:00")]:
            if qcount.get(qos, 0) >= SUBMIT_CAP.get(qos, 0):
                continue
            lbl, sh = job_script(cell, rounds, qos, wt)
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
