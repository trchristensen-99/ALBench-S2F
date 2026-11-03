"""Nov-3 night: Yash's 100k and 300k genomic ensembles in EXACTLY the style of the shared 10k and 30k ones, replacing the first (v1c, standard-input, 1 seed) attempt of register_nov3g.py.

He compares sizes, so the recipe must be the one the shared `genomic_{10k,30k}_patience16` used: menu v0 (30 usable configurations), 3 seeds, patience 16, bare 200-bp insert, V0c validation, the first N sequences of the canonical nested genomic
pool. `scripts/yash_menu_retrain.py` runs one (configuration, seed) job per shard: 100k = all 30 configurations x 3 seeds = 90 shards (the 10k and 30k ensembles used all 30; the per-size table says 20 at 100k to save compute, which is not needed now);
300k = the first 12 configurations of the priority order (per_size.json) x 3 seeds = 36 shards. Shards are ordered seed-major, so the first 30 (12) shards already make a complete one-seed ensemble. Each shard has its own output directory
(an out-of-memory failure on a MIG slice then only reruns that shard on a full GPU); merge them afterwards. The driver's name contains `menu_retrain.py` and the cells pass `--D`, which is what makes 100k shards eligible for the idle MIG slices.

Also: `scratchpad/mig_cap.txt` 20 -> 28 (the whole migq partition; we are its only user), the lower-priority PI-architecture cells (pi41_) are deferred and their pending jobs cancelled so that these shards get the slices first.

    python scripts/program_oct22/register_nov3h.py [--dry]
"""

import fcntl
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad")
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_nov2b as nb  # noqa: E402

R, NEST, DRY = nb.R, nb.NEST, nb.DRY
W = "/grid/koo/home/christen/ALBench-S2F-yash"
PY = "/grid/wsbs/home_norepl/christen/ALBench-S2F/.venv/bin/python"
SB = "/cm/shared/apps/slurm/current/bin"
RETIRE = ("yash_g100k_", "yash_g300k_")
PLAN = {
    100000: ("100k", 30, 90),
    300000: ("300k", 12, 36),
}  # size -> (tag, configurations, shards = configurations x 3 seeds)


def cells():
    out = []
    for D, (tag, k, n) in PLAN.items():
        pool = f"{NEST}/genomic_d{D}.npz"
        for i in range(n):
            od = f"outputs/yash_ens/genomic_{tag}_patience16_shards/s{i:02d}"
            out.append(
                dict(
                    name=f"yash{tag[:-1]}_s{i:02d}",
                    gpus=["h100", "v100"],
                    ext=[],
                    out_dir=od,
                    env={},
                    needs=[os.path.relpath(pool, R)],
                    cmd=f"cd {W} && PYTHONPATH={W} {PY} scripts/yash_menu_retrain.py --D {D} --pool {pool} --out {R}/{od} --shard {i}/{n} --k {k} --seeds 3 --patience 16 --menu v0 && touch {R}/{od}/COMPLETE",
                    done=dict(type="file", path=f"{od}/COMPLETE"),
                )
            )
    return out


def main():
    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        keep = [c for c in reg if not c["name"].startswith(RETIRE)]
        have = {c["name"] for c in keep}
        new = [c for c in cells() if c["name"] not in have]
        reg2 = keep + new
        assert len({c["name"] for c in reg2}) == len(reg2)
        print(
            "retire",
            len(reg) - len(keep),
            "| add",
            len(new),
            "| registry",
            len(reg),
            "->",
            len(reg2),
        )
        if DRY:
            print(new[0]["cmd"][:330])
            return
        shutil.copy(path, path + ".bak_nov3m")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
    open("scratchpad/mig_cap.txt", "w").write("28\n")
    open("scratchpad/defer_prefixes.txt", "w").write("pi41_\n")
    for c in (
        reg
    ):  # stop the first (v1c) attempt; its partial members stay in outputs/yash_ens/genomic_*_v1c
        if c["name"].startswith(RETIRE):
            subprocess.call([f"{SB}/scancel", "-u", "christen", "-n", f"d_{c['name']}"])
    q = subprocess.Popen(
        [f"{SB}/squeue", "-u", "christen", "-h", "-t", "PD", "-o", "%i %j"],
        stdout=subprocess.PIPE,
        universal_newlines=True,
    ).communicate()[0]
    for line in q.splitlines():
        jid, name = line.split()
        if name.startswith("d_pi41_"):
            subprocess.call([f"{SB}/scancel", jid])
    print("registered; mig_cap 28; pi41_ deferred, pending pi41 jobs cancelled")


if __name__ == "__main__":
    main()
