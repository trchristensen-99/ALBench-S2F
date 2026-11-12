#!/usr/bin/env python3
"""The guidance-prior control, and replicates for the two n=1 seed arms.

Peter's objection to the slide claim ("LegNet-style architectures dominate the loop's
winners") is that it is conditioned on what the search was shown. Reading the generated
CODE of all 12,677 proposals says he is partly right, but not in the way either of us
assumed:

  * the REFERENCE ARCHITECTURE handed to the loop barely matters. Budget-matched at 30k:
    LegNet source 0.8811 (pe49, n=3), nothing at all 0.8772 (nr45, n=3), DREAM-RNN
    0.8757 (dr46, n=1). A spread of 0.0054, inside ar41's own replicate spread (0.0071).
  * the GUIDANCE TEXT may matter a great deal, and it is the thing that actually names
    the motifs: `known_good` states outright that "LegNet-style inverted-residual blocks
    with squeeze-excitation had the highest mean validation Pearson". 126 of 141 arms
    carry it. The measured 4.5x enrichment of inverted-residual blocks among winners is
    therefore not independent evidence that they are best -- the loop was told so.

We cannot currently separate the two, because every `open`-guidance arm we have was run
on a smaller budget (op38_open 3.0 h and op38_fb 10.5 h against 16.0 h here, base_cap_min
25 against 40). That is a compute confound, not a guidance result, and it is why this
script exists.

  A. og51 -- nr45 with ONE flag changed: --guidance known_good -> open. Same 16 h, same
     cap, same pool, same env. 3 reps. This is the control that makes the motif claim
     either defensible or dead.
  B. dr46 / dr47 reps 2 and 3 -- the DREAM-RNN-seeded arms are the only non-LegNet seeds
     we have and both are n=1, so neither the "seed does not matter" nor the
     "dropping the prior costs 0.015" reading currently has error bars.

7 cells, 16 h each, ~112 GPU-h.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import sys

R = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
REG = "scratchpad/dispatch_registry.json"
BACKUP_SUFFIX = ".bak_nov11h"
DRY = "--dry" in sys.argv

TAG = "og51"
NEW_REPS = (2, 3)


def by_name(reg, name):
    for c in reg:
        if c["name"] == name:
            return c
    raise SystemExit(f"{name} is not in the registry")


def guidance_control(reg, rep):
    """nr45 rep N with the architectural prior removed, and nothing else touched."""
    src = by_name(reg, f"nr45_30k_r{rep}")
    cmd = src["cmd"]

    assert "--guidance known_good" in cmd, "nr45's guidance moved; re-read before trusting this"
    assert "--reference_legnet" not in cmd, "nr45 is supposed to carry no reference source"
    assert "--baseline_file" not in cmd, "nr45 is supposed to carry no baseline file"
    cmd = cmd.replace("--guidance known_good", "--guidance open")

    odir = f"outputs/archsearch_reps_oct22/genomic_30k_{TAG}/rep{rep}"
    cmd = cmd.replace(f"genomic_30k_nr45/rep{rep}", f"genomic_30k_{TAG}/rep{rep}")
    assert "nr45" not in cmd, "output directory still points at nr45"
    assert "--guidance open" in cmd and "known_good" not in cmd

    # one factor, or it is not a control
    for flag in ("--rounds", "--per_round", "--budget_hours", "--base_cap_min",
                 "--max_params", "--D", "--flank_ctx", "--shift_max"):
        a = cmd.split(flag)[1].split()[0]
        b = src["cmd"].split(flag)[1].split()[0]
        assert a == b, f"{flag}: {a} against nr45's {b}"
    assert cmd.split("--budget_hours")[1].split()[0] == "16.0", "must match the 16 h arms"
    assert cmd.split("--base_cap_min")[1].split()[0] == "40", "must match the cap-40 arms"

    return dict(
        name=f"{TAG}_30k_r{rep}", gpus=src.get("gpus", ["v100"]), ext=src.get("ext", []),
        env=dict(src.get("env", {})), out_dir=odir, needs=list(src.get("needs", [])),
        cmd=cmd, done=dict(type="file", path=f"{odir}/DONE"),
    )


def seed_replicate(reg, arm, rep):
    """dr46 / dr47 rep 1 re-pointed at an independent labelled draw."""
    src = by_name(reg, f"{arm}_30k_r1")
    cmd = src["cmd"]
    assert "genomic_r1_d30000.npz" in cmd, f"{arm} rep1 does not read the r1 pool"
    cmd = cmd.replace("genomic_r1_d30000.npz", f"genomic_r{rep}_d30000.npz")
    cmd = cmd.replace(f"{arm}_30k/rep1", f"{arm}_30k/rep{rep}")
    assert "/rep1" not in cmd and f"/rep{rep}" in cmd, "output directory still points at rep1"
    assert "--baseline_file" in cmd, "the seed arm must keep its DREAM-RNN baseline"

    odir = f"outputs/archsearch_reps_oct22/genomic_30k_{arm}_30k/rep{rep}"
    needs = [n.replace("genomic_r1_d30000.npz", f"genomic_r{rep}_d30000.npz")
             for n in src.get("needs", [])]
    return dict(
        name=f"{arm}_30k_r{rep}", gpus=src.get("gpus", ["v100"]), ext=src.get("ext", []),
        env=dict(src.get("env", {})), out_dir=odir, needs=needs, cmd=cmd,
        done=dict(type="file", path=f"{odir}/DONE"),
    )


def main() -> None:
    os.chdir(R)
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(REG))

        cells = [guidance_control(reg, r) for r in (1, 2, 3)]
        for arm in ("dr46", "dr47"):
            cells += [seed_replicate(reg, arm, r) for r in NEW_REPS]

        have = {c["name"] for c in reg}
        new = [c for c in cells if c["name"] not in have]
        print(f"{len(cells)} built, {len(cells) - len(new)} already registered, {len(new)} new")
        for c in new:
            od = os.path.join(R, c["out_dir"])
            assert not os.path.isdir(od) or not os.listdir(od), f"{c['out_dir']} is not empty"
            for n in c["needs"]:
                assert os.path.exists(os.path.join(R, n)), f"{c['name']}: missing {n}"
        if not new:
            return

        reg2 = reg + new
        assert len({c["name"] for c in reg2}) == len(reg2), "duplicate cell name"
        print(f"registry {len(reg)} -> {len(reg2)}  (~{len(new) * 16} GPU-h)")
        for c in new:
            print("  ", c["name"], "->", c["out_dir"])
        if DRY:
            print("\n--- og51 r1 cmd tail ---\n", new[0]["cmd"][-300:])
            return
        shutil.copy(REG, REG + BACKUP_SUFFIX)
        json.dump(reg2, open(REG + ".tmp", "w"), indent=1)
        os.replace(REG + ".tmp", REG)
        print("registered")


if __name__ == "__main__":
    main()
