"""Nov-10: the three things the user approved, plus a repair.

**1. Parallel seeds for the 20-seed DREAM-RNN bank.** `register_nov10b.py` asked for 17 seeds of a
ONE-entry menu across 6 shards. menu_retrain shards by ENTRY, so shards 1-5 had nothing to do and
wrote COMPLETE immediately, and shard 0 took all 17 seeds serially -- on a V100, at roughly an hour
per e1600 model, that is ~16 h against a 12 h default wall. It is self-healing (menu_retrain skips
a model whose `_meta.json` exists, so a walltime kill resumes) but it is wasteful. Two helper cells
take the upper seed ranges in parallel; shard 0 skips whatever they finish first. Their done-files
are specific model metas rather than COMPLETE markers, so they cannot collide with each other.

**2. The rounds-matched 100k loop.** The 100k arms run 6 rounds and the 10k/30k arms run 15, because
the 36 h budget binds at 100k and not by design. So "the loop contributes less at 100k" is confounded
with "the loop searched less at 100k". This runs 15 rounds at 100k with every other flag identical to
`ar41_100k_r1`, which separates them. The budget has to rise with the rounds or the cell would simply
be killed, so `--budget_hours` is the one flag that deliberately differs and the guard checks that
nothing else does.

**3. The loop-component ablations.** Each element of the loop is justified by Delta-union -- the gain
of (bank + loop) over the bank alone -- not by the loop stack's own score. Four elements have never
been ablated, and one has never even been switched on:

    proposer_ens   AL_PROPOSER_ENSEMBLE=1    show the proposer which members carry stack weight, so
                                             it proposes complements instead of a better single model
    no_coord_ens   drop --ensemble_aware     the coordinator stops seeing the validation stack
    no_ref         drop --reference_legnet   the proposer stops being shown the published network
    explore0       explore_frac 0.0          no explore phase: refinement only
    explore8       explore_frac 0.8          mostly explore

All at 30k against `ar41_30k_r1`, which is the cheap size and the one with 15 rounds already.

    python scripts/program_oct22/register_nov10h.py [--dry]
"""

import fcntl
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_nov9m as m  # noqa: E402

R, OUT, DRY = m.R, m.OUT, m.DRY
REG_PATH = "scratchpad/dispatch_registry.json"
# (name suffix, seed_offset, how many seeds, the model whose meta marks it done)
HELPERS = (("hi", 10, 5, "r14_menu_000_meta.json"), ("hj", 15, 5, "r19_menu_000_meta.json"))
RNN20_DIR = f"{OUT}/refrnn20_30k_genomic_can"
# (tag, how to change ar41_30k_r1) -- one element each, everything else identical
ABLATIONS = (
    ("abl_prop", dict(env={"AL_PROPOSER_ENSEMBLE": "1"})),
    ("abl_nocoord", dict(drop_flag="--ensemble_aware")),
    ("abl_noref", dict(drop_flag="--reference_legnet")),
    ("abl_expl0", dict(explore=0.0)),
    ("abl_expl8", dict(explore=0.8)),
)


def base(reg, name):
    c = next((x for x in reg if x["name"] == name), None)
    assert c is not None, f"{name} is not in the registry"
    return c


def rnn20_helpers(reg):
    src = base(reg, "refrnn20_30k_genomic_can_s0")
    out = []
    for suffix, off, n, marker in HELPERS:
        cmd = re.sub(
            r"--seeds \d+ --shard \d+/\d+ --seed_offset \d+",
            f"--seeds {n} --shard 0/1 --seed_offset {off}",
            src["cmd"],
        )
        assert f"--seed_offset {off}" in cmd and f"--seeds {n} " in cmd, cmd
        out.append(
            dict(
                name=f"refrnn20_30k_{suffix}",
                gpus=["h100", "v100"],
                ext=[],
                out_dir=src["out_dir"],
                env=dict(src.get("env", {})),
                needs=list(src.get("needs", [])),
                cmd=cmd,
                done=dict(type="file", path=f"{src['out_dir']}/{marker}"),
            )
        )
    return out


def rounds_matched(reg):
    src = base(reg, "ar41_100k_r1")
    ref30 = base(reg, "ar41_30k_r1")
    want = re.search(r"--rounds (\d+)", ref30["cmd"]).group(1)
    assert want == "15", f"the 30k arm runs {want} rounds, not 15"
    cmd = re.sub(r"--rounds \d+", f"--rounds {want}", src["cmd"])
    # the budget must rise with the rounds or the cell is just killed; it is the ONLY other change
    cmd = re.sub(r"--budget_hours [\d.]+", "--budget_hours 80.0", cmd)
    cmd = cmd.replace("genomic_100k_ar41/rep1", "genomic_100k_ar45/rep1")
    for flag in ("--per_round", "--base_cap_min", "--max_params", "--guidance"):
        a = cmd.split(flag)[1].split()[0]
        b = src["cmd"].split(flag)[1].split()[0]
        assert a == b, f"{flag}: {a} against {b}"
    assert '"manager": "schedule_ens"' in cmd, "must stay ar41's manager"
    env = dict(src.get("env", {}))
    env.pop(
        "AL_NONFINITE_SKIP", None
    )  # read by no code; its comment claims a protection that does not exist
    return [
        dict(
            name="ar45_100k_r1",
            gpus=src.get("gpus", ["h100"]),
            ext=src.get("ext", []),
            env=env,
            out_dir="outputs/archsearch_reps_oct22/genomic_100k_ar45/rep1",
            needs=list(src.get("needs", [])),
            cmd=cmd,
            done=dict(
                type="file", path="outputs/archsearch_reps_oct22/genomic_100k_ar45/rep1/DONE"
            ),
        )
    ]


def ablations(reg):
    src = base(reg, "ar41_30k_r1")
    out = []
    for tag, how in ABLATIONS:
        cmd, env = src["cmd"], dict(src.get("env", {}))
        if "env" in how:
            env.update(how["env"])
        if "drop_flag" in how:
            assert how["drop_flag"] in cmd, f"{how['drop_flag']} is not in the base arm"
            cmd = cmd.replace(" " + how["drop_flag"], "")
            assert how["drop_flag"] not in cmd, "flag survived removal"
        if "explore" in how:
            assert '"explore_frac": 0.4' in cmd, "the base arm's explore_frac moved"
            cmd = cmd.replace('"explore_frac": 0.4', f'"explore_frac": {how["explore"]}')
        cmd = cmd.replace("genomic_30k_ar41/rep1", f"genomic_30k_{tag}/rep1")
        assert tag in cmd and "ar41/rep1" not in cmd, "output directory still points at ar41"
        # exactly ONE element may differ, or the ablation attributes nothing
        diffs = sum([bool(how.get("env")), bool(how.get("drop_flag")), "explore" in how])
        assert diffs == 1, f"{tag} changes {diffs} things"
        # AL_NONFINITE_SKIP is NOT a dead knob: scratchpad/arch_autoresearch.py:344 reads it, and that
        # is the file the loop runs. An earlier audit checked albench/arch_search/arch_autoresearch.py,
        # a stale copy with zero occurrences, and wrongly concluded it was dead. It must be carried over
        # unchanged, or the arm differs from its baseline in fp16 non-finite handling as well.
        assert env.get("AL_NONFINITE_SKIP") == "1", "the baseline sets it; carry it over"
        out.append(
            dict(
                name=f"{tag}_30k_r1",
                gpus=src.get("gpus", ["h100"]),
                ext=src.get("ext", []),
                env=env,
                out_dir=f"outputs/archsearch_reps_oct22/genomic_30k_{tag}/rep1",
                needs=list(src.get("needs", [])),
                cmd=cmd,
                done=dict(
                    type="file", path=f"outputs/archsearch_reps_oct22/genomic_30k_{tag}/rep1/DONE"
                ),
            )
        )
    return out


def main():
    reg = json.load(open(REG_PATH))
    cells = rnn20_helpers(reg) + rounds_matched(reg) + ablations(reg)
    for c in cells:
        od = os.path.join(R, c["out_dir"])
        if c["name"].startswith("refrnn20_30k_h"):
            continue  # these share the existing bank directory on purpose
        assert not os.path.isdir(od) or not os.listdir(od), f"{c['out_dir']} is not empty"
    assert len({c["name"] for c in cells}) == len(cells)
    print(f"guard: {len(cells)} cells -> " + ", ".join(c["name"] for c in cells))

    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(REG_PATH))
        have = {c["name"] for c in reg}
        new = [c for c in cells if c["name"] not in have]
        reg2 = reg + new
        assert len({c["name"] for c in reg2}) == len(reg2)
        print(f"add {len(new)} cells | registry {len(reg)} -> {len(reg2)}")
        if DRY:
            print([c["name"] for c in new])
            return
        shutil.copy(REG_PATH, REG_PATH + ".bak_nov10h")
        json.dump(reg2, open(REG_PATH + ".tmp", "w"), indent=1)
        os.replace(REG_PATH + ".tmp", REG_PATH)
        print("registered")


if __name__ == "__main__":
    main()
