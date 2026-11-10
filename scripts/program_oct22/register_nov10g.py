"""Nov-10: widen the architecture-invariance check from two corpora to six.

`register_nov10f.py` asks whether four architecture families agree that the motif_shared_core
pilot beats the genomic one. Two corpora can only ever test one ranking, and a reviewer's real
question is broader: is the reservoir ORDERING -- the thing every scaling curve in this study is
built on -- a property of the data or of LegNet's inductive bias?

So the same four architectures, the same menu file and therefore the exact same recipes, on four
more 30k corpora spanning the kinds of reservoir the program actually compares:

    random                 unconditioned synthetic, the floor
    dinuc_shuffle          composition-matched to genomic, so it isolates syntax from composition
    motif_ct_enriched      a second motif design, which says whether agreement on
                           motif_shared_core was about motifs or about that one corpus
    accessible_tiles_k562  real sequence selected by K562 accessibility

With nov10f's two, that is six corpora ranked independently by four families. The claim worth
making in a paper is a rank correlation across families, which six points can support and two
cannot.

Deliberately excluded: `zoonomia_orthologs`, whose 30k pool is 15.4% exact duplicates (every other
reservoir is at 0.0%). Its ranking is confounded by that and it would weaken, not strengthen, the
invariance statement.

The menu is nov10f's, not a copy, so the architectures and recipes cannot drift between the two
halves of the experiment. About 12 GPU-hours, and it is V100 work, which is what is idle.

    python scripts/program_oct22/register_nov10g.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_nov9m as m  # noqa: E402
import register_nov10f as f  # noqa: E402

R, OUT, DRY = m.R, m.OUT, m.DRY
D, SEEDS, SHARDS = f.D, f.SEEDS, f.SHARDS
CORPORA = ("random", "dinuc_shuffle", "motif_ct_enriched", "accessible_tiles_k562")
EXCLUDED = {"zoonomia_orthologs": "15.4% exact duplicates in its 30k pool"}


def main():
    # the menu nov10f wrote IS the experiment's definition; this must not write its own
    menu_abs = os.path.join(R, f.MENU_PATH)
    assert os.path.exists(menu_abs), f"run register_nov10f.py first; {f.MENU_PATH} is missing"
    entries = json.load(open(menu_abs))
    assert len(entries) == 4, f"{len(entries)} architectures in the menu, expected 4"
    assert len({e["n_params"] for e in entries}) == 4, "two architectures have the same size"
    print("guard: reusing " + f.MENU_PATH + " -- " + ", ".join(e["id"] for e in entries))

    for bad, why in EXCLUDED.items():
        assert bad not in CORPORA, f"{bad} is excluded: {why}"
    assert not set(CORPORA) & set(f.CORPORA), f"nov10f already ran {set(CORPORA) & set(f.CORPORA)}"

    cells = []
    for corpus in CORPORA:
        pool = f"{m.NEST}/{corpus}_d{D}.npz"
        assert os.path.exists(os.path.join(R, pool)), f"missing {pool}"
        odir = f"{OUT}/archinvar_{D // 1000}k_{corpus}"
        for k in range(SHARDS):
            cells.append(
                dict(
                    name=f"archinvar_{D // 1000}k_{corpus}_s{k}",
                    gpus=["h100", "v100"],
                    ext=[],
                    out_dir=odir,
                    env=dict(m.ENV),
                    needs=[pool, f.MENU_PATH],
                    cmd=(
                        f"uv run --no-sync python scratchpad/menu_retrain.py "
                        f"--menu {R}/{f.MENU_PATH} --reservoir_cache {R}/{pool} --D {D} "
                        f"--out_dir {R}/{odir} --seeds {SEEDS} --shard {k}/{SHARDS} "
                        f"--base_cap_min 600 --val_cache {m.VAL} --flank_ctx 20 --shift_max 10"
                    ),
                    done=dict(type="file", path=f"{odir}/COMPLETE_{k}of{SHARDS}"),
                )
            )

    # every arm must differ from nov10f's genomic arm ONLY in the pool, or the six are not one table
    # compared WITHIN a shard: cells of different shards legitimately differ in --shard k/n
    ref = {
        c["name"].rsplit("_s", 1)[1]: c["cmd"] for c in cells if c["out_dir"].endswith(CORPORA[0])
    }
    for c in cells:
        corpus = c["out_dir"].rsplit(f"_{D // 1000}k_", 1)[1]
        shard = c["name"].rsplit("_s", 1)[1]
        assert ref[shard].replace(CORPORA[0], corpus) == c["cmd"], (
            f"{c['name']} differs from its siblings by more than the corpus"
        )
        assert " --shift_max 10" in c["cmd"] and " --flank_ctx 20" in c["cmd"], (
            "the 240-bp standard"
        )
        od = os.path.join(R, c["out_dir"])
        assert not os.path.isdir(od) or not os.listdir(od), f"{c['out_dir']} is not empty"
    print(
        f"guard: {len(cells)} cells, {len(CORPORA)} new corpora x 4 architectures x {SEEDS} seeds "
        f"({len(CORPORA) + len(f.CORPORA)} corpora in the table once nov10f lands)"
    )

    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        have = {c["name"] for c in reg}
        new = [c for c in cells if c["name"] not in have]
        reg2 = reg + new
        assert len({c["name"] for c in reg2}) == len(reg2)
        print(f"add {len(new)} cells | registry {len(reg)} -> {len(reg2)}")
        if DRY:
            print([c["name"] for c in new])
            return
        shutil.copy(path, path + ".bak_nov10g")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
