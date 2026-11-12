#!/usr/bin/env python3
"""Five more seeds of each architecture's MEASURED winner at 10k / 30k / 100k.

The S1 scaling-from-zero figure puts one dot per architecture per size, and each dot is a
best-of-3-within-recipe over a handful of recipes. That is both noisy and biased upward,
and at 100k dream_attn's dot rests on a single model. LegNet is the only architecture on
that figure with a real spread (n=20 per size), which is exactly the asymmetry Peter would
be right to object to.

This takes the winner recipe per (architecture, size) as MEASURED from the bracket metas
(scratchpad/nov11/arch_winners.json, written by reading val_pearson, not from notes) and
trains 5 fresh seeds of each. Seed offset 3 keeps them disjoint from the existing s=0..2,
and a separate out_dir means nothing already measured can be overwritten.

Each entry is copied VERBATIM from the menu that produced the measured number -- same
source code, same train_config -- and every cell inherits base_cap_min, gpus and env from
the registry cell that actually ran it, so the new seeds are comparable with the old ones
rather than merely similar.

9 entries over 3 sizes x 5 seeds = 45 models, ~1.8 GPU-h median each, ~81 GPU-h.
"""

from __future__ import annotations

import fcntl
import glob
import json
import os
import shutil
import sys

R = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
REG = "scratchpad/dispatch_registry.json"
BACKUP_SUFFIX = ".bak_nov11i"
DRY = "--dry" in sys.argv

WINNERS = "scratchpad/nov11/arch_winners.json"
SEEDS, SEED_OFFSET, NSHARD = 5, 3, 5
POOL = "outputs/grid_pools/k562_nested/genomic_d%d.npz"


def find_entry(entry_id):
    """The menu file holding this entry, and the entry itself."""
    for path in sorted(glob.glob(os.path.join(R, "outputs/menus_nov*/*.json"))):
        try:
            menu = json.load(open(path))
        except Exception:
            continue
        if not isinstance(menu, list):
            continue
        for e in menu:
            if e.get("id") == entry_id:
                return path, e
    raise SystemExit(f"no menu contains entry {entry_id}")


def source_cell(reg, menu_path):
    """The registry cell that ran this menu, so cap/gpus/env carry over."""
    base = os.path.basename(menu_path)
    for c in reg:
        if base in c.get("cmd", ""):
            return c
    raise SystemExit(f"no registry cell runs {base}")


def main() -> None:
    os.chdir(R)
    winners = json.load(open(WINNERS))
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(REG))
        have = {c["name"] for c in reg}
        new, menus_to_write = [], {}

        for Ds, fams in sorted(winners.items(), key=lambda kv: int(kv[0])):
            D = int(Ds)
            entries, caps, gpus, envs = [], [], [], []
            for fam in sorted(fams):
                path, e = find_entry(fams[fam])
                src = source_cell(reg, path)
                assert "source" in e and "train_config" in e, f"{fams[fam]} is not a full entry"
                entries.append(e)
                caps.append(int(src["cmd"].split("--base_cap_min")[1].split()[0]))
                gpus.append(tuple(src.get("gpus", ["h100", "v100"])))
                envs.append(json.dumps(src.get("env", {}), sort_keys=True))
                print(f"  D={D:>6} {fam:<11} {fams[fam]:<32} from {os.path.basename(path)}")

            # the winners for one size came from different sweeps; take the most generous
            # wall and the narrowest GPU set, or a long recipe gets killed on a short cap
            cap = max(caps)
            gpu = list(sorted(set.intersection(*[set(g) for g in gpus])) or ["h100"])
            assert len(set(envs)) == 1, f"D={D}: source cells disagree on env: {set(envs)}"
            env = json.loads(envs[0])

            mpath = f"outputs/menus_nov11/ARCHREP_{D // 1000}k.json"
            menus_to_write[mpath] = entries
            odir = f"outputs/menu_retrain_v0/archrep{D // 1000}k_genomic_can"
            pool = POOL % D
            assert os.path.exists(os.path.join(R, pool)), f"missing pool {pool}"

            for k in range(NSHARD):
                name = f"archrep{D // 1000}k_s{k}"
                if name in have:
                    continue
                cmd = (
                    f"uv run --no-sync python scratchpad/menu_retrain.py"
                    f" --menu {R}/{mpath}"
                    f" --reservoir_cache {R}/{pool} --D {D}"
                    f" --out_dir {R}/{odir}"
                    f" --seeds {SEEDS} --shard {k}/{NSHARD} --seed_offset {SEED_OFFSET}"
                    f" --base_cap_min {cap}"
                    f" --val_cache /grid/koo/home/shared/albench/data/val_V0c.npz"
                    f" --flank_ctx 20 --shift_max 10"
                )
                new.append(dict(
                    name=name, gpus=gpu, ext=[], env=env, out_dir=odir,
                    needs=[pool, mpath], cmd=cmd,
                    done=dict(type="file", path=f"{odir}/COMPLETE_{k}of{NSHARD}"),
                ))

        n_models = len(winners) * 3 * SEEDS
        print(f"\n{len(new)} new cells, {n_models} models, ~{n_models * 1.8:.0f} GPU-h")
        for c in new:
            print("  ", c["name"], c["gpus"], "cap", c["cmd"].split("--base_cap_min")[1].split()[0])
        if DRY:
            return
        if not new:
            print("nothing to add")
            return

        os.makedirs(os.path.join(R, "outputs/menus_nov11"), exist_ok=True)
        for mpath, entries in menus_to_write.items():
            json.dump(entries, open(os.path.join(R, mpath), "w"), indent=1)
            print("wrote", mpath, f"({len(entries)} entries)")

        reg2 = reg + new
        assert len({c["name"] for c in reg2}) == len(reg2), "duplicate cell name"
        shutil.copy(REG, REG + BACKUP_SUFFIX)
        json.dump(reg2, open(REG + ".tmp", "w"), indent=1)
        os.replace(REG + ".tmp", REG)
        print(f"registered | registry {len(reg)} -> {len(reg2)}")


if __name__ == "__main__":
    main()
