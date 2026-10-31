"""Oct-31: top the published MPRA-LegNet (e100) up to 20 independent seeds at D = 100k (register_ln100.py trained 10). At 100k the e100 recipe is the strongest ensemble we have (0.931 genomic test as a
10-seed stack against 0.921 for menu v2), so its seed count decides how well the per-size comparison is anchored; b256 is not topped up (it breaks at 100k: single 0.893). Same pool, input, recipe and
output directory as the first ten seeds; new seeds are drawn once and stored in outputs/bench33/seeds100k_b.json.

    python scripts/program_oct22/register_ln100b.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_ln100 as ln  # noqa: E402

R, OUT, NEST, D, F, DRY = ln.R, ln.OUT, ln.NEST, ln.D, ln.F, ln.DRY
SEED_FILE = f"{R}/outputs/bench33/seeds100k_b.json"


def seeds():
    if os.path.exists(SEED_FILE):
        return json.load(open(SEED_FILE))
    s = [int.from_bytes(os.urandom(3), "big") for _ in range(10)]
    if not DRY:
        json.dump(s, open(SEED_FILE, "w"))
    return s


def all_cells():
    sd, odir, out = seeds(), f"{OUT}/bench38_ln_e100_100k_c{F}", []
    for k in range(5):
        ids = [10 + 2 * k, 11 + 2 * k]
        pairs = ",".join(f"{i}={sd[i - 10]}" for i in ids)
        out.append(
            dict(
                name=f"bench38_ln_e100_100k_c{F}_s{5 + k}",
                gpus=["h100", "v100"],
                ext=[],
                out_dir=odir,
                env={},
                needs=[f"{NEST}/genomic_d{D}.npz"],
                cmd=f"uv run --no-sync python scripts/program_oct22/bench33_legnet.py --D {D} --cfg e100 --seeds {pairs} --ctx {F} && touch {R}/{odir}/COMPLETE_{5 + k}of10",
                done=dict(type="file", path=f"{odir}/COMPLETE_{5 + k}of10"),
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
            print(new[0]["cmd"][:300] if new else "nothing new")
            return
        shutil.copy(path, path + ".bak_ln100b")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
