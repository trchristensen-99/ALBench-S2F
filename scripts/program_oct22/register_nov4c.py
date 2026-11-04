"""Nov-4: same-pool tuned LegNet banks. The loop runs train on the replicate pools (`k562_reps` r1-r3) while the 20-seed bank trains on the canonical nested pool, and the two share only 3% (10k),
9-10% (30k) and 32% (100k) of their sequences (`scripts/analysis/pool_overlap.py`). So "loop + bank" has so far stacked models trained on two different training sets, and "loop against bank" compared
different pools. This trains the bank (native trainer, per-size tuned recipe, standard input, V0c selection, 20 seeds) on each replicate pool, so that both comparisons can be made on one training set,
and gives the bank's pool-to-pool spread that every reservoir comparison needs. About 37 GPU-hours (median 6.8 / 8.3 / 22.1 min per seed at 10k / 30k / 100k).

  spb{10,30,100}_r{1,2,3}_s{0..19}   outputs/menu_retrain_v0/bench38_ln_<tuned cfg>_<D>k_c20_genomic_r<r>

    python scripts/program_oct22/register_nov4c.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_ln100 as ln  # noqa: E402

R, OUT, F, DRY = ln.R, ln.OUT, ln.F, ln.DRY
REPS = "outputs/grid_pools/k562_reps"
TUNED = {10000: "e800_wd4", 30000: "e500_wd2", 100000: "e500"}
SEED_FILE = f"{R}/outputs/bench33/seeds_same_pool_nov4.json"


def seeds() -> dict:
    if os.path.exists(SEED_FILE):
        return json.load(open(SEED_FILE))
    s = {
        f"{D}_r{r}": [int.from_bytes(os.urandom(3), "big") for _ in range(20)]
        for D in TUNED
        for r in (1, 2, 3)
    }
    if not DRY:
        json.dump(s, open(SEED_FILE, "w"))
    return s


def all_cells():
    sd, out = seeds(), []
    for D, cfg in TUNED.items():
        for r in (1, 2, 3):
            pool = f"{REPS}/genomic_r{r}_d{D}.npz"
            odir = f"{OUT}/bench38_ln_{cfg}_{D // 1000}k_c{F}_genomic_r{r}"
            for j in range(20):
                out.append(
                    dict(
                        name=f"spb{D // 1000}_r{r}_s{j}",
                        gpus=["h100"] if D >= 100000 else ["h100", "v100"],
                        ext=[],
                        out_dir=odir,
                        env={},
                        needs=[pool],
                        cmd=f"uv run --no-sync python scripts/program_oct22/bench33_legnet.py --D {D} --cfg {cfg} --pool {R}/{pool} --seeds {j}={sd[f'{D}_r{r}'][j]} --ctx {F} && touch {R}/{odir}/COMPLETE_s{j}",
                        done=dict(type="file", path=f"{odir}/COMPLETE_s{j}"),
                    )
                )
    return out


def main():
    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        have = {c["name"] for c in reg}
        new = [c for c in all_cells() if c["name"] not in have]
        reg2 = reg + new
        assert len({c["name"] for c in reg2}) == len(reg2)
        print("add", len(new), "cells at the end | registry", len(reg), "->", len(reg2))
        if DRY:
            print(new[0]["cmd"][:400] if new else "nothing new")
            return
        shutil.copy(path, path + ".bak_nov4c")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
