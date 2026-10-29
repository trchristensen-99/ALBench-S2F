"""Oct-29 (evening): how to steer the AutoResearch loop gently while it keeps sampling many different architectures (cells st38_).

Every arm uses the standard harness (20 real flank bases each side, shift +-10; the proposer is told the input length is fixed and position-specific designs are allowed), the agent loop with a coordinator
(claude-sonnet-5-5, 10 rounds x 2 models = 20 models), V0c validation feedback, ensemble-aware prompt. The arms differ in the guidance and in how exploration is organised:

  a0  control: the current guidance block (known good settings), recipes proposed by the proposer, coordinator without a manager
  a1  a0 + the published MPRA-LegNet shown as the reference, with its validation r measured in the run's own harness as the bar to beat
  a2  a1 + the harness assigns every model's recipe from a safe set (Muon, learning rate 1e-3 to 4e-3, weight decay 1e-4 to 1e-3, batch 128 to 512, one-cycle or cosine): the proposer writes architectures only
  a3  a2 + quota manager: each round one EXPLOIT and one EXPLORE model, the coordinator sees a coverage table of architectural features (and the ones nobody has tried) and the validation stack
  a4  a3 with the minimal guidance (facts and negative results only, no positive prescriptions of widths, kernels or dropout)

Stage 1: genomic 10k, replicate pools r1 to r3, all five arms (15 runs). Stage 2 (starts when outputs/st38/STAGE2_GO exists): a0 against a4 on random 10k, CT-enriched motif 10k and genomic 30k, pools r1 and r2 (12 runs).
Analysis: scripts/analysis/analyze_st38.py (stack r on three test sets, member prediction correlation, feature coverage, recipes, tokens per model).

    python scripts/program_oct22/register_st38.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad")
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
from stage2_lib import REPS, R  # noqa: E402

os.chdir(R)
DRY = "--dry" in sys.argv
S55 = "claude-sonnet-5-5"
OUT22 = "outputs/archsearch_reps_oct22"
V0C = "/grid/koo/home/shared/albench/data/val_V0c.npz"
GATE = "outputs/st38/STAGE2_GO"
rel = lambda p: os.path.relpath(p, R)  # noqa: E731
ARMS = {
    "a0": dict(guidance="known_good", ref=False, jitter=False, manager="none"),
    "a1": dict(guidance="known_good", ref=True, jitter=False, manager="none"),
    "a2": dict(guidance="known_good", ref=True, jitter=True, manager="none"),
    "a3": dict(guidance="known_good", ref=True, jitter=True, manager="quota"),
    "a4": dict(guidance="minimal", ref=True, jitter=True, manager="quota"),
}


def cell(arm, res, D, r, gated=False):
    a = ARMS[arm]
    pool = f"{REPS}/{res}_r{r}_d{D}.npz"
    out = f"{OUT22}/st38_{arm}_{res}_{D // 1000}k/rep{r}"
    cfg = {"stages": ["propose", "coordinate", "code", "evaluate"], "ideas_per_round": 6}
    if a["manager"] != "none":
        cfg["manager"] = a["manager"]
    if a["jitter"]:
        cfg["recipe_source"] = "harness"
    return dict(
        name=f"st38_{arm}_{res}_{D // 1000}k_r{r}",
        gpus=["h100", "v100"],
        ext=[],
        out_dir=out,
        env={"LLM_MODEL": S55},
        needs=[rel(pool)] + ([GATE] if gated else []),
        cmd=f"uv run --no-sync python scratchpad/arch_autoresearch.py --reservoir_cache {pool} --D {D} --out_dir {R}/{out} --rounds 10 --per_round 2 --max_params 40000000 --base_cap_min 25 "
        f"--budget_hours {{BUDGET}} --guidance {a['guidance']} --ensemble_aware {'--reference_legnet ' if a['ref'] else ''}--loop agent --agent_cfg '{json.dumps(cfg)}' --val_cache {V0C} --resume",
        done=dict(type="file", path=f"{out}/COMPLETE"),
    )


def all_cells():
    out = [cell(arm, "genomic", 10000, r) for r in (1, 2, 3) for arm in ARMS]
    for res, D in (("random", 10000), ("motif_ct_enriched", 10000), ("genomic", 30000)):
        out += [cell(arm, res, D, r, gated=True) for r in (1, 2) for arm in ("a0", "a4")]
    return out


def main():
    path = "scratchpad/dispatch_registry.json"
    os.makedirs("outputs/st38", exist_ok=True)
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        have = {c["name"] for c in reg}
        new = [c for c in all_cells() if c["name"] not in have]
        i = max(i for i, c in enumerate(reg) if "LLM_MODEL" in c.get("env", {}))
        reg2 = reg[: i + 1] + new + reg[i + 1 :]
        assert len({c["name"] for c in reg2}) == len(reg2)
        print("add", len(new), "cells after index", i, "| registry", len(reg), "->", len(reg2))
        missing = sorted(
            {n for c in new for n in c["needs"] if n != GATE and not os.path.exists(n)}
        )
        print("missing pools:", missing)
        if DRY:
            print(new[0]["cmd"][-380:])
            return
        assert not missing
        shutil.copy(path, path + ".bak_st38")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
