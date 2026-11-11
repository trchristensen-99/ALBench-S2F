"""Nov-11: the batch-size x optimizer axis for DREAM-Attn and Malinois, which only DREAM-RNN got.

`register_nov10d.py` swept batch size and optimizer around DREAM-RNN's measured 30k winner and it
was worth +0.015 genomic test -- more than the epochs sweep and the 20-seed replication put
together. That axis has never been run for the other two published architectures. Every model in
`refbank_30k` and `refbankb_30k` was trained at batch 1024 with AdamW; only the epochs, weight decay
and learning rate were varied.

That is a fairness problem before it is a performance one. The deck's headline compares a tuned
LegNet against his three architectures, and the LegNet bank spans e100-e800, two batch sizes, seven
weight decays and early-stop variants, while DREAM-Attn and Malinois have each seen six recipes on
one batch size and one optimizer. A reader does not have to be hostile to ask whether the gap is the
architecture or the search budget, and at the moment we cannot answer. Neither has seen Muon either,
which is half of what helped DREAM-RNN.

This is `register_nov10d` with the architecture and its winner parameterised: the same four points,
the same three seeds, the same pinned (1024, adamw) point left out because it is already measured.
Same design, same N, so the comparison is like for like.

    dream_attn  e500,  wd 0.2, lr 0.003   beat 0.8241 genomic to matter
    malinois    e800,  wd 0.2, lr 0.01    beat 0.8241 genomic to matter

24 models, about 36 GPU-hours.

    python scripts/program_oct22/register_nov11d.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_nov9m as m  # noqa: E402
import register_nov10d as d  # noqa: E402

R, OUT, DRY = m.R, m.OUT, m.DRY
D, SEEDS, SHARDS = d.D, d.SEEDS, d.SHARDS
POINTS, PINNED = d.POINTS, d.PINNED

# architecture -> (epochs, weight_decay, lr, the genomic test r its 30k winner already scores).
# The recipes are each family's best in refbank_30k / refbankb_30k, read from the saved _meta.json.
SWEEP = {
    "dream_attn": (500, 0.2, 0.003, 0.8241),
    "malinois": (800, 0.2, 0.01, 0.8241),
}


def menu_path(arch: str) -> str:
    return f"outputs/menus_nov11/REF{arch.replace('_', '').upper()}RECIPE_{D // 1000}K.json"


def build(arch: str) -> tuple[list, list, str]:
    import torch

    from albench.arch_search.harness_standard import INPUT_LEN
    from albench.arch_search.validate import static_check

    ep, wd, lr, _ = SWEEP[arch]
    tag, mod, n_params = next(a for a in m.ARCHS if a[0] == arch)
    bad = static_check(mod.SOURCE)
    assert bad is None, f"{tag}: the harness validator rejects this source -- {bad}"
    ns = {}
    exec(compile(mod.SOURCE, f"<{tag}>", "exec"), ns)
    net = ns["Net"]()
    assert sum(p.numel() for p in net.parameters()) == n_params, tag
    net.eval()
    with torch.no_grad():
        y = net(torch.randn(2, 4, INPUT_LEN))
    assert y.shape == (2,) and torch.isfinite(y).all(), f"{tag}: output {tuple(y.shape)}"

    entries = [
        {
            "id": f"{tag}_e{ep}_bs{bs}_{opt}",
            "cell": f"{tag}_ref",
            "idea": (
                f"published {tag} at its {D // 1000}k winner (e{ep}, wd {wd}, lr {lr}) "
                f"with batch {bs} and {opt}"
            ),
            "orig_val": 0.0,
            "n_params": n_params,
            "source": mod.SOURCE,
            "train_config": {
                "optimizer": opt,
                "lr": lr,
                "batch_size": bs,
                "lr_schedule": "onecycle",
                "weight_decay": wd,
                "epochs": ep,
            },
        }
        for bs, opt in POINTS
    ]
    assert len({e["id"] for e in entries}) == len(entries), "two points collapsed to one id"

    mp = menu_path(arch)
    short = arch.replace("_", "")
    label = f"ref{short}recipe_{D // 1000}k"
    odir = f"{OUT}/{label}_genomic_can"
    pool = f"{m.NEST}/genomic_d{D}.npz"
    cells = [
        dict(
            name=f"{label}_genomic_can_s{k}",
            gpus=["h100", "v100"],
            ext=[],
            out_dir=odir,
            env=dict(m.ENV),
            needs=[pool, mp],
            cmd=(
                f"uv run --no-sync python scratchpad/menu_retrain.py "
                f"--menu {R}/{mp} --reservoir_cache {R}/{pool} --D {D} "
                f"--out_dir {R}/{odir} --seeds {SEEDS} --shard {k}/{SHARDS} "
                f"--base_cap_min 600 --val_cache {m.VAL} --flank_ctx 20 --shift_max 10"
            ),
            done=dict(type="file", path=f"{odir}/COMPLETE_{k}of{SHARDS}"),
        )
        for k in range(SHARDS)
    ]
    return entries, cells, mp


def main():
    import importlib.util

    # a Muon point that silently trains under AdamW would answer the wrong question
    assert importlib.util.find_spec("muon") is not None, "muon is not installed; drop that point"
    assert PINNED not in POINTS, f"{PINNED} is the measured point; re-running it wastes a cell"

    all_cells, menus = [], {}
    for arch in SWEEP:
        entries, cells, mp = build(arch)
        menus[mp] = entries
        all_cells += cells
        ep, wd, lr, base = SWEEP[arch]
        print(
            f"{arch:<11} e{ep} wd{wd} lr{lr} | {len(entries)} points x {SEEDS} seeds "
            f"= {len(entries) * SEEDS} models in {len(cells)} cells; beat {base} genomic"
        )

    for c in all_cells:
        assert " --shift_max 10" in c["cmd"] and " --flank_ctx 20" in c["cmd"], (
            "the 240-bp standard"
        )
        # --optimizer would override every entry; these points differ BY optimizer
        assert " --optimizer " not in c["cmd"], "the per-entry optimizer must stand"
        od = os.path.join(R, c["out_dir"])
        assert not os.path.isdir(od) or not os.listdir(od), f"{c['out_dir']} is not empty"
    assert len({c["name"] for c in all_cells}) == len(all_cells), "two architectures collided"
    print(f"guard: {len(all_cells)} cells across {len(SWEEP)} architectures")

    if not DRY:
        for mp, entries in menus.items():
            os.makedirs(os.path.join(R, os.path.dirname(mp)), exist_ok=True)
            json.dump(entries, open(os.path.join(R, mp), "w"), indent=1)

    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        have = {c["name"] for c in reg}
        new = [c for c in all_cells if c["name"] not in have]
        reg2 = reg + new
        assert len({c["name"] for c in reg2}) == len(reg2)
        print(f"add {len(new)} cells | registry {len(reg)} -> {len(reg2)}")
        if DRY:
            print([c["name"] for c in new])
            return
        shutil.copy(path, path + ".bak_nov11d")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
