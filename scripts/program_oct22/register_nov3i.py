"""Nov-3 night: gated finishing cells for Yash's ensembles. Each waits (dispatcher `needs`) for every shard marker of its size, then merges the shards, fits the stack and scores the three test sets
(`scripts/yash_finish_ensemble.py`): yash100_finish needs the 90 shards of register_nov3h.py, yash300_finish the 36.

    python scripts/program_oct22/register_nov3i.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad")
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_nov2b as nb  # noqa: E402

R, DRY = nb.R, nb.DRY
W = "/grid/koo/home/christen/ALBench-S2F-yash"
PY = "/grid/wsbs/home_norepl/christen/ALBench-S2F/.venv/bin/python"
PLAN = {100000: ("100k", 90), 300000: ("300k", 36)}


def cells():
    out = []
    for D, (tag, n) in PLAN.items():
        od = f"outputs/yash_ens/genomic_{tag}_patience16"
        shards = f"outputs/yash_ens/genomic_{tag}_patience16_shards"
        out.append(
            dict(
                name=f"yash{tag[:-1]}_finish",
                gpus=["h100", "v100"],
                ext=[],
                out_dir=od,
                env={},
                needs=[f"{shards}/s{i:02d}/COMPLETE" for i in range(n)],
                cmd=f"cd {W} && PYTHONPATH={W} {PY} scripts/yash_finish_ensemble.py --D {D} --shards {R}/{shards} --out {R}/{od} --menu v0",
                done=dict(type="file", path=f"{od}/RESULTS.json"),
            )
        )
    return out


def main():
    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        have = {c["name"] for c in reg}
        new = [c for c in cells() if c["name"] not in have]
        print("add", [c["name"] for c in new], "| registry", len(reg), "->", len(reg) + len(new))
        if DRY:
            return
        shutil.copy(path, path + ".bak_nov3o")
        json.dump(reg + new, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
