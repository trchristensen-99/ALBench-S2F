"""Nov-10: run the `lanes` manager at 100k, the one loop variant aimed at the defect every measurement keeps finding.

Every loop arm measured so far has the same shape of failure, and it is not that the stack is bad:
it is that the stack is LegNet. At 30k, `is40`'s LegNet-derived members stack to 0.8873 and its
other members to 0.8668; at 100k, `rr40`'s split 0.9331 against 0.8946 and `ar41`'s -- the best so
far -- 0.9364 against 0.9293. Effective membership tells the same story: 6.1 to 12.3 for loop
stacks against 15.9 for 20 LegNet seeds. The loop finds LegNet variants, ensembles them, and the
non-LegNet half contributes little. That is exactly the architecture confound the ensembles are
supposed to marginalise out.

`ar41` narrowed it by enforcing the tuned recipe (non-LegNet members went from -0.039 to -0.0071
behind at 100k), which showed the recipe was most of the handicap rather than the architecture.
What it does not do is guarantee the ensemble CONTAINS non-LegNet models -- the coordinator may
still spend every slot on the family that scores best, which at 100k is LegNet.

`lanes` is the manager written for precisely this and never run: every round trains one model in
lane A (built on the reference LegNet's blocks) and one in lane B (built WITHOUT them); the
proposer writes half its ideas for each lane and the coordinator must pick one per lane. With
`AL_RECIPE_MODE=tuned` both lanes get the long recipe, so lane B is not handicapped the way the
pre-ar41 arms were. Structural diversity becomes a constraint rather than a hope.

100k, because that is where the problem is worst and where the loop's union gain has decayed to
+0.0029 genomic. One replicate against `ar41_100k`, matched on every budget flag so the only
difference is the manager -- the guard asserts that rather than trusting it.

Open risk worth stating: forcing a lane B pick every round spends half the budget on the weaker
family, so the stack may well come out BELOW ar41's. That would still be the useful answer, because
the question is whether enforced structural diversity buys ensemble value, not whether it buys
single-model accuracy. The reading is the union with the bank and the effective-member count, not
the loop stack alone.

    python scripts/program_oct22/register_nov10e.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_ln100 as ln  # noqa: E402

R, OUT, DRY = ln.R, ln.OUT, ln.DRY
D = 100_000
REF = "ar41_100k_r1"  # the arm this is matched against
TAG = f"ln44_{D // 1000}k"
MATCH = ("--rounds", "--per_round", "--budget_hours", "--base_cap_min", "--max_params")


def main():
    reg_path = "scratchpad/dispatch_registry.json"
    reg = json.load(open(reg_path))
    ref = next((c for c in reg if c["name"] == REF), None)
    assert ref is not None, f"{REF} is not in the registry; nothing to match against"

    # build this arm by editing the reference arm's command, so the budget cannot drift
    cmd = ref["cmd"]
    assert '"manager": "schedule_ens"' in cmd, f"{REF} is not the schedule_ens arm any more"
    cmd = cmd.replace('"manager": "schedule_ens"', '"manager": "lanes"')
    cmd = cmd.replace(f"genomic_{D // 1000}k_ar41/rep1", f"genomic_{D // 1000}k_ln44/rep1")
    assert "ln44" in cmd and "ar41" not in cmd, "the output directory still points at ar41"

    env = dict(ref.get("env", {}))
    assert env.get("AL_RECIPE_MODE") == "tuned", "lane B must get the long recipe, as ar41 does"
    # AL_NONFINITE_SKIP is NOT a dead knob: scratchpad/arch_autoresearch.py:344 reads it, and that
    # is the file the loop runs. An earlier audit checked albench/arch_search/arch_autoresearch.py,
    # a stale copy with zero occurrences, and wrongly concluded it was dead. It must be carried over
    # unchanged, or the arm differs from its baseline in fp16 non-finite handling as well.
    assert env.get("AL_NONFINITE_SKIP") == "1", "the baseline sets it; carry it over"

    cell = dict(
        name=f"{TAG}_r1",
        gpus=ref.get("gpus", ["h100"]),
        ext=ref.get("ext", []),
        out_dir=f"outputs/archsearch_reps_oct22/genomic_{D // 1000}k_ln44/rep1",
        env=env,
        needs=list(ref.get("needs", [])),
        cmd=cmd,
        done=dict(
            type="file", path=f"outputs/archsearch_reps_oct22/genomic_{D // 1000}k_ln44/rep1/DONE"
        ),
    )

    # every budget flag must carry across, or the comparison is not a manager comparison
    for flag in MATCH:
        i = cmd.split(flag)[1].split()[0]
        j = ref["cmd"].split(flag)[1].split()[0]
        assert i == j, f"{flag}: {i} against {REF}'s {j}"
    od = os.path.join(R, cell["out_dir"])
    assert not os.path.isdir(od) or not os.listdir(od), f"{cell['out_dir']} is not empty"
    print(f"guard: manager lanes, every flag in {list(MATCH)} matches {REF}")

    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(reg_path))
        if any(c["name"] == cell["name"] for c in reg):
            print(f"{cell['name']} already registered")
            return
        reg2 = reg + [cell]
        assert len({c["name"] for c in reg2}) == len(reg2)
        print(f"add 1 cell | registry {len(reg)} -> {len(reg2)}")
        if DRY:
            print(cell["name"])
            print()
            print(cell["cmd"][:400])
            return
        shutil.copy(reg_path, reg_path + ".bak_nov10e")
        json.dump(reg2, open(reg_path + ".tmp", "w"), indent=1)
        os.replace(reg_path + ".tmp", reg_path)
        print("registered")


if __name__ == "__main__":
    main()
