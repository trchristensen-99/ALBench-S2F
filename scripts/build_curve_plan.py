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


def find_pool(res: str, labelled: bool = False) -> Path | None:
    """Locate a reservoir's pool whatever its size/seed suffix.

    The genomic arm is not generated like the others: it is partitioned out of the
    finite real-CRE set by scripts/build_genomic_partition.py, so its pool carries a
    different n and seed. Globbing keeps the planner from hardcoding that exception.
    """
    suffix = "__labeled.npz" if labelled else ".npz"
    hits = [
        f
        for f in sorted(POOLS.glob(f"{res}__n*__seed*{suffix}"))
        if labelled or not f.name.endswith("__labeled.npz")
    ]
    return hits[0] if hits else None


def pool_capacity(res: str) -> int | None:
    """How many sequences this reservoir can actually supply, from its pool filename."""
    f = find_pool(res)
    if f is None:
        return None
    import re

    m = re.search(r"__n(\d+)__", f.name)
    return int(m.group(1)) if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/curves/human_additive.yaml")
    ap.add_argument("--out", default="outputs/curves/jobs.txt")
    ap.add_argument("--stage", choices=["plan", "generate", "train", "status"], default="plan")
    args = ap.parse_args()

    cfg = yaml.safe_load((REPO / args.config).read_text())
    # A reservoir entry is either a bare name or {name, params, alias}. The alias is
    # what names the pool and the output dir, so two parameterisations of the same
    # strategy stay distinguishable everywhere downstream.
    res_entries = []
    for r in cfg["reservoirs"]:
        if isinstance(r, str):
            res_entries.append({"name": r, "params": {}, "alias": r})
        else:
            res_entries.append(
                {"name": r["name"], "params": r.get("params", {}), "alias": r.get("alias", r["name"])}
            )
    res_list = [e["alias"] for e in res_entries]
    matched = {int(k): int(v) for k, v in (cfg.get("matched_top_increment") or {}).items()}
    bases = cfg["baselines"]
    incs = cfg["increments"]
    size, pseed = cfg["pool_size"], cfg["pool_seed"]
    sseeds = cfg.get("subset_seeds", [0])
    base_res, base_seed = cfg["baseline_pool"], cfg["baseline_seed"]

    POOLS.mkdir(parents=True, exist_ok=True)

    # The genomic arm and the baseline are produced together by
    # scripts/build_genomic_partition.py, as disjoint halves of the finite real-CRE
    # set. Drawing them independently made 95% of the baseline reappear in the
    # "added" genomic data, which would have flattened the genomic curve for a
    # reason unrelated to genomic data being uninformative.
    gen = []
    for e in res_entries:
        if e["alias"] == base_res:
            continue
        sets = "".join(f" --set {k}={v}" for k, v in e["params"].items())
        gen.append(
            f"albench generate --strategy {e['name']} --n {size} --seed {pseed}{sets} "
            f"--out {pool_path(e['alias'], size, pseed).relative_to(REPO)}"
        )

    base_pool_file = find_pool(base_res + "_baseline", labelled=True)
    base_pool = base_pool_file or pool_path(base_res + "_baseline", max(bases), base_seed, True)
    train = []
    skipped: list[str] = []
    for r in res_list:
        cap = pool_capacity(r)
        for b in bases:
            for inc0 in incs:
                # At a matched baseline every arm uses the same budget, so the largest
                # point compares arms rather than comparing budgets.
                inc = matched[b] if (b in matched and inc0 == max(incs)) else inc0
                # A point whose increment exceeds what the reservoir can supply is not
                # a smaller point -- it does not exist. Dropping it explicitly keeps a
                # capacity limit from masquerading as a data point.
                if cap is not None and inc > cap:
                    skipped.append(f"{r} +{inc:,} (capacity {cap:,})")
                    continue
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

    generated_seq = size * len(gen)
    partitioned_seq = sum(
        c for c in (pool_capacity(base_res), pool_capacity(base_res + "_baseline")) if c
    )
    n_seq = generated_seq + partitioned_seq
    print(f"reservoirs         : {len(res_list)}  {res_list}")
    print(f"baselines          : {bases}   (0 = from-scratch)")
    print(f"increments         : {incs}")
    print(f"subset-order seeds : {sseeds}")
    print(
        f"\npools to generate+label : {len(gen)} generated x {size:,} = "
        f"{generated_seq:,}, plus {partitioned_seq:,} partitioned from real CREs "
        f"= {n_seq:,} sequences"
    )
    print(f"  estimated oracle time : {n_seq / ORACLE_SEQ_PER_S / 3600:.1f} GPU-hours "
          f"at ~{ORACLE_SEQ_PER_S} seq/s")
    if skipped:
        uniq = sorted(set(skipped))
        print(
            f"\nSKIPPED {len(skipped)} (reservoir, baseline, increment) combinations "
            f"= {len(skipped) * len(sseeds)} runs, beyond reservoir capacity:"
        )
        for k in uniq:
            print(f"  {k}")
        print("  (real CREs are finite -- this asymmetry is a result, report it)")
    print(f"\ncurve points to train   : {len(train)} "
          f"({len(res_list)} reservoirs x {len(bases)} baselines x {len(incs)} "
          f"increments x {len(sseeds)} seeds)")
    print("\nEvery increment is a nested prefix of one 300k pool, so adding a subset")
    print("seed costs training time only -- never another round of oracle labelling.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
