"""Copy the finished Yash 100k/300k genomic ensembles (outputs/yash_ens/genomic_{100k,300k}_patience16) and their training sets to the shared lab folder, write provenance.json in the style of the 10k/30k ones and update the shared README. Run once, Nov 4 2026."""

import json
import os
import shutil
import subprocess

R = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
SH = "/grid/koo/home/shared/albench"
RECIPE = "early stopping patience 16, best-weight restore, reverse-complement averaged MSE, shift augmentation up to 10 bp (zero-padded), bare 200-bp insert (trained before the 240-bp student input standard, same as the 10k and 30k ensembles)"
info = {}
for tag, D, k in (("100k", 100000, 30), ("300k", 300000, 12)):
    src = f"{R}/outputs/yash_ens/genomic_{tag}_patience16"
    dst = f"{SH}/ensembles/genomic_{tag}_patience16"
    res = json.load(open(f"{src}/RESULTS.json"))
    ens = json.load(open(f"{src}/ensemble.json"))
    assert res["members"] == len(ens["members"]) == 3 * k, (res["members"], k)
    assert os.path.exists(src + "/members"), src
    if os.path.exists(dst):
        raise SystemExit("exists: " + dst)
    os.makedirs(dst)
    shutil.copytree(src + "/members", dst + "/members")
    shutil.copy2(src + "/ensemble.json", dst + "/ensemble.json")
    prov = {
        "source_dir": [
            f"outputs/yash_ens/genomic_{tag}_patience16_shards/s{i:02d}" for i in range(3 * k)
        ][:1]
        + ["... one directory per (configuration, seed): " + str(3 * k) + " shards, merged"],
        "pool": f"outputs/grid_pools/k562_nested/genomic_d{D}.npz",
        "size": D,
        "seeds": 3,
        "configurations": k,
        "n_members": res["members"],
        "n_members_positive_weight": res["positive_weight"],
        "menu": "v0 (30 usable configurations)"
        if k == 30
        else "v0, the first 12 configurations of the priority order (per_size.json, size row 300000)",
        "train_config_recipe": RECIPE,
        "val_cache": {
            "path": "/grid/koo/home/shared/albench/data/val_V0c.npz",
            "variant": "V0c",
            "n": 29449,
            "note": "the 10k and 30k ensembles used the earlier validation set V0; V0c moves the genomic-test ensemble by about +0.002 with the same ordering",
        },
        "oracle_id": "oracle_v3",
        "test_r_ensemble": res["test_r"],
        "validation_r_ensemble": res["val_r_ensemble"],
        "code": "scripts/yash_menu_retrain.py (albench.ensemble.menu.train_menu with flank_ctx=0) and scripts/yash_finish_ensemble.py, ALBench-S2F commits ea1d431 and 711bfbd; trained Nov 3-4 2026",
        "note": "test sets: genomic = held-out chr7/chr13 (oracle labels), ood = designed out-of-distribution set, random_32k = 32,000 random sequences",
    }
    json.dump(prov, open(dst + "/provenance.json", "w"), indent=1)
    shutil.copy2(
        f"{R}/outputs/grid_pools/k562_nested/genomic_d{D}.npz",
        f"{SH}/data/genomic_start/genomic_d{D}.npz",
    )
    info[tag] = res["test_r"]
# README
p = SH + "/README.md"
s = open(p).read()
bak = p + ".bak_nov4"
shutil.copy2(p, bak)


def rep(a, b):
    global s
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)


rep(
    "Use the patience-16 pair.",
    "Use the patience-16 pair. **Added 2026-11-04:** `ensembles/genomic_100k_patience16` and `ensembles/genomic_300k_patience16`, in the same style as the 10k and 30k ones (menu v0, bare 200-bp insert, patience 16, 3 seeds, same genomic pool, same format with `provenance.json`): 100k uses all 30 usable configurations (90 members, 28 with weight), 300k the first 12 configurations of the per-size table (36 members, 16 with weight; all 30 would cost about 54 H100-hours). They were fitted and early-stopped on V0c, the current validation set (the 10k and 30k used the earlier V0; V0c moves the genomic-test ensemble by about +0.002). Test r genomic / designed / random: 100k %.3f / %.3f / %.3f, 300k %.3f / %.3f / %.3f. Across the four sizes (genomic): 0.711, 0.789, %.3f, %.3f."
    % (
        info["100k"]["genomic"],
        info["100k"]["ood"],
        info["100k"]["random_32k"],
        info["300k"]["genomic"],
        info["300k"]["ood"],
        info["300k"]["random_32k"],
        info["100k"]["genomic"],
        info["300k"]["genomic"],
    ),
)
rep("genomic 10k and 30k. |", "genomic 10k, 30k, 100k and 300k. |")
rep(
    "(the 10k and 30k genomic training sets: the first 10,000 / 30,000 sequences",
    "(the 10k, 30k, 100k and 300k genomic training sets: the first 10,000 / 30,000 / 100,000 / 300,000 sequences",
)
rep(
    "Weights exist only for the genomic 10k and 30k ensembles;",
    "Weights exist only for the genomic 10k, 30k, 100k and 300k ensembles;",
)
open(p, "w").write(s)
subprocess.call(
    [
        "chmod",
        "-R",
        "g+rX,o-rwx",
        SH + "/ensembles/genomic_100k_patience16",
        SH + "/ensembles/genomic_300k_patience16",
        SH + "/data/genomic_start",
        p,
    ]
)
print("shipped", info)
