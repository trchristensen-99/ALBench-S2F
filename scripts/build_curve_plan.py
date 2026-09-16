"""Expand the additive-curve config into concrete generate/train commands.

Kept separate from the shell pipeline so the plan can be inspected, diffed and
cost-estimated before anything is submitted. `plan` prints what would run and what it
would cost; nothing here touches a scheduler.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
POOLS = REPO / "outputs" / "curves" / "pools"
TRAIN = REPO / "outputs" / "curves" / "train"

# Measured on this project's oracle: ~300 sequences/s for the ensemble. An earlier
# 29.7 seq/s figure was a 256-sequence probe that was ~90% JIT compilation.
ORACLE_SEQ_PER_S = 300


def pool_path(res: str, size: int, seed: int, labelled: bool = False) -> Path:
    return POOLS / f"{res}__n{size}__seed{seed}{'__labeled' if labelled else ''}.npz"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/curves/human_additive.yaml")
    ap.add_argument("--out", default="outputs/curves/jobs.txt")
    ap.add_argument("--stage", choices=["plan", "generate", "train", "status"], default="plan")
    args = ap.parse_args()

    cfg = yaml.safe_load((REPO / args.config).read_text())
    res_list = cfg["reservoirs"]
    bases = cfg["baselines"]
    incs = cfg["increments"]
    size, pseed = cfg["pool_size"], cfg["pool_seed"]
    sseeds = cfg.get("subset_seeds", [0])
    base_res, base_seed = cfg["baseline_pool"], cfg["baseline_seed"]

    POOLS.mkdir(parents=True, exist_ok=True)

    gen = [
        f"albench generate --strategy {r} --n {size} --seed {pseed} "
        f"--out {pool_path(r, size, pseed).relative_to(REPO)}"
        for r in res_list
    ]
    # The baseline is its own draw so it is not the same sequences as the genomic
    # ARM's added data; without this the genomic curve would re-add its own baseline.
    gen.append(
        f"albench generate --strategy {base_res} --n {max(bases)} --seed {base_seed} "
        f"--out {pool_path(base_res + '_baseline', max(bases), base_seed).relative_to(REPO)}"
    )

    base_pool = pool_path(base_res + "_baseline", max(bases), base_seed, labelled=True)
    train = []
    for r in res_list:
        for b in bases:
            for inc in incs:
                for ss in sseeds:
                    tag = f"{r}__base{b}__add{inc}__s{ss}"
                    cmd = (
                        f"experiments/exp1_1_scaling.py --task k562 --student legnet "
                        f"--oracle ag_s2 --reservoir {r} "
                        f"--pool-base-dir outputs/curves/pools_linked "
                        f"--training-sizes {inc} --seed {ss} "
                        f"--n-replicates 1 --no-hp-sweep --chr-split --save-predictions "
                        f"--output-dir {(TRAIN / tag).relative_to(REPO)}"
                    )
                    if b > 0:
                        cmd += (
                            f" --base-pool {base_pool.relative_to(REPO)} "
                            f"--base-n {b} --base-seed {base_seed}"
                        )
                    train.append(cmd)

    outp = REPO / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)

    if args.stage == "generate":
        (outp.parent / "generate.txt").write_text("\n".join(gen) + "\n")
        print(f"wrote {len(gen)} generate commands")
        return 0
    if args.stage == "train":
        outp.write_text("\n".join(train) + "\n")
        print(f"wrote {len(train)} train commands")
        return 0
    if args.stage == "status":
        done = sum(
            1
            for c in train
            if any((REPO / c.split("--output-dir ")[1].split()[0]).rglob("result.json"))
        )
        gp = sum(1 for r in res_list if pool_path(r, size, pseed).exists())
        lp = sum(1 for r in res_list if pool_path(r, size, pseed, labelled=True).exists())
        print(f"  pools generated : {gp}/{len(res_list)}")
        print(f"  pools labelled  : {lp}/{len(res_list)}")
        print(f"  curve points    : {done}/{len(train)}")
        return 0

    n_seq = size * (len(res_list) + 1)
    print(f"reservoirs         : {len(res_list)}  {res_list}")
    print(f"baselines          : {bases}   (0 = from-scratch)")
    print(f"increments         : {incs}")
    print(f"subset-order seeds : {sseeds}")
    print(f"\npools to generate+label : {len(gen)} x {size:,} = {n_seq:,} sequences")
    print(f"  estimated oracle time : {n_seq / ORACLE_SEQ_PER_S / 3600:.1f} GPU-hours "
          f"at ~{ORACLE_SEQ_PER_S} seq/s")
    print(f"curve points to train   : {len(train)} "
          f"({len(res_list)} reservoirs x {len(bases)} baselines x {len(incs)} "
          f"increments x {len(sseeds)} seeds)")
    print("\nEvery increment is a nested prefix of one 300k pool, so adding a subset")
    print("seed costs training time only -- never another round of oracle labelling.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
