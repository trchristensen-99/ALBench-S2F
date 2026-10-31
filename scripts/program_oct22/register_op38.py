"""Oct-31: does an OPEN prompt (no LegNet, no 'squeeze-excitation convolutions are best' hint) find better architectures than the anchored loop? The collaborator's loop, with LegNet removed from the prompt, found
grouped-convolution, ResNeXt and second-order-convolution models that match a well-tuned LegNet and beat everything our loop found. Paired by pool with mgr35_c15 (known-good guidance, coordinator only) and
mgr35_e15 (known-good guidance, schedule_ens manager): genomic 30k replicate pools r1..r3, 15 rounds x 2 models, proposer-chosen recipes, standard input, V0c validation, Sonnet 5.5.

  op38_open_r<k>   --guidance open (no architecture-family hint), no reference code, manager schedule_ens (explore 40%, then exploit)
  op38_target_r<k> --guidance open, no reference code, plus a stated known-achievable target: the in-run LegNet baseline's validation r + 0.003 (the collaborator's open-world models reached about that
                   much above a tuned LegNet at 30k; the target follows the training-set size because the baseline is measured in the run)
  op38_fb_r<k>     --guidance open, the reference LegNet and its in-run baseline are revealed only from the exploit phase (round 6), manager schedule_fb: explore without an anchor, then refine the best
                   explored design or fall back to small modifications of the reference if nothing beats it by 0.005

Replaces the not-yet-started mgr38_ cells (their reference LegNet was shown from round 0, the anchoring the open arms remove); started mgr38 cells are left alone.

    python scripts/program_oct22/register_op38.py [--reps 3] [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad")
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_lr35 as lr  # noqa: E402

DRY = "--dry" in sys.argv
REPS = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 3
COORD = {"stages": ["propose", "coordinate", "code", "evaluate"], "ideas_per_round": 6}
ROUNDS, EXPLORE = 15, 0.4
REVEAL = round(ROUNDS * EXPLORE)  # first exploit round


def all_cells():
    out = []
    for arm, extra, flags in (
        ("open", {"manager": "schedule_ens", "explore_frac": EXPLORE}, ""),
        (
            "fb",
            {"manager": "schedule_fb", "explore_frac": EXPLORE},
            f" --reference_legnet --reference_from_round {REVEAL}",
        ),
        (
            "target",
            {"manager": "schedule_ens", "explore_frac": EXPLORE},
            " --reference_legnet --reference_from_round 999 --target_delta 0.003",
        ),
    ):
        for r in range(1, REPS + 1):
            c = lr.cell(
                f"op38_{arm}_r{r}",
                f"genomic_30k_op38_{arm}",
                r,
                30000,
                ROUNDS,
                2,
                {**COORD, **extra},
            )
            assert " --guidance known_good " in c["cmd"] and " --val_cache " in c["cmd"]
            c["cmd"] = (
                c["cmd"]
                .replace(" --guidance known_good ", " --guidance open ", 1)
                .replace(" --val_cache ", flags + " --val_cache ", 1)
            )
            out.append(c)
    return out


def main():
    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        drop = [
            c["name"]
            for c in reg
            if c["name"].startswith("mgr38_")
            and not os.path.exists(os.path.join(lr.R, c["out_dir"], "labels.npz"))
        ]
        reg = [c for c in reg if c["name"] not in drop]
        have = {c["name"] for c in reg}
        new = [c for c in all_cells() if c["name"] not in have]
        i = max(i for i, c in enumerate(reg) if "LLM_MODEL" in c.get("env", {}))
        reg2 = reg[: i + 1] + new + reg[i + 1 :]
        assert len({c["name"] for c in reg2}) == len(reg2)
        print(
            "drop", len(drop), "unstarted mgr38 cells; add", len(new), "op38 cells after index", i
        )
        if DRY:
            print(new[-1]["cmd"][:900])
            return
        shutil.copy(path, path + ".bak_op38")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
