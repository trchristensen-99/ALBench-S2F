"""Nov-3 evening: the 100k and 300k genomic ensembles Yash asked for, in the same style as the shared 10k and 30k ones (`/grid/koo/home/shared/albench/ensembles/genomic_{10k,30k}_patience16`).

`albench menu train` of the current default menu (v1c: 20 configurations, the reference LegNet among them) with the study's input standard (240 bp: 20 real flank bases each side, shift +-10, RC averaging), early-stopping patience 16,
one seed per configuration, on the canonical nested genomic pool (the first 100,000 / 300,000 sequences, seed 42, oracle_v3 labels), validation V0c. Per `per_size_v1c.json`: all 20 configurations at 100k, the first 12 at 300k.
Sharded (`--shard k/n`, resumable) so the shards spread over every queue; the code is a worktree of origin/main (`/grid/koo/home/christen/ALBench-S2F-yash`) because the cluster checkout is behind. After all shards: `menu stack`, then
`menu evaluate` on the three test sets, then copy to the shared folder (not automated here).

  yash_g100k_s0..s9    10 shards x 2 configurations      outputs/yash_ens/genomic_100k_v1c
  yash_g300k_s0..s11   12 shards x 1 configuration       outputs/yash_ens/genomic_300k_v1c

    python scripts/program_oct22/register_nov3g.py [--dry]
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

R, NEST, DRY = nb.R, nb.NEST, nb.DRY
W = "/grid/koo/home/christen/ALBench-S2F-yash"
PY = f"/grid/wsbs/home_norepl/christen/ALBench-S2F/.venv/bin/python -c 'import sys; from albench.cli import main; sys.exit(main())'"
SIZES = {100000: ("100k", 10), 300000: ("300k", 12)}


def cells():
    out = []
    for D, (tag, n) in SIZES.items():
        pool = f"{NEST}/genomic_d{D}.npz"
        out_dir = f"outputs/yash_ens/genomic_{tag}_v1c"
        for k in range(n):
            out.append(
                dict(
                    name=f"yash_g{tag}_s{k}",
                    gpus=["h100", "v100"],
                    ext=[],
                    out_dir=out_dir,
                    env={},
                    needs=[os.path.relpath(pool, R)],
                    cmd=f"cd {W} && PYTHONPATH={W} {PY} menu train --pool {pool} --out {R}/{out_dir} --size {D} --shard {k}/{n} && touch {R}/{out_dir}/COMPLETE_{k}of{n}",
                    done=dict(type="file", path=f"{out_dir}/COMPLETE_{k}of{n}"),
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
        reg2 = reg + new
        assert len({c["name"] for c in reg2}) == len(reg2)
        print("add", len(new), "cells | registry", len(reg), "->", len(reg2))
        if DRY:
            print(new[0]["cmd"][:400])
            return
        shutil.copy(path, path + ".bak_nov3l")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
