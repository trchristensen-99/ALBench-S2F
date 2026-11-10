"""Nov-10: finish tuning DREAM-RNN on the two axes the bracket never touched, so the negative result is not an artefact of a half-search.

The claim we want to be able to defend is strong: that even the best-tuned published non-LegNet
architecture adds nothing to a tuned MPRA-LegNet ensemble. The 30k union measurement supports it
(bank 0.8907/0.4221; bank + all three references 0.8909/0.4169), but the search behind it was a
coordinate bracket over three axes -- epochs, weight decay, peak learning rate -- with
`batch_size` pinned at 1024 and `optimizer` pinned at AdamW for every point. Those were LegNet's
values, not DREAM-RNN's, which is the exact mistake the per-architecture tuning was created to
avoid. A reviewer would be right to say the negative result tested a quarter of the recipe space.

So the two pinned axes get measured, at DREAM-RNN's own winner (e1600, wd 0.2, lr 0.01):

    batch 256 / 512 / 2048 at AdamW     the DREAM human pipeline trains at well under 1024, and
                                        batch size interacts with a 1600-epoch one-cycle schedule
    Muon at batch 1024                  a different optimiser family entirely; the package is
                                        installed, and the harness refuses to silently fall back
                                        to AdamW when it is missing, so a Muon result is a Muon
                                        result

Four points x 3 seeds. If none beats e1600/1024/AdamW's 0.8511 genomic by more than the seed band,
the recipe space is closed for this architecture on every axis the harness exposes, and the
negative result stands on a full search rather than a partial one. If one of them does win, it
feeds the 20-seed bank in `register_nov10b.py` instead -- which is the point of running this now,
while that bank is still training.

Malinois and DREAM-Attn do not get this treatment: both take essentially zero stacker weight in
the union (bank + Malinois is identical to the bank to four decimals), so there is no gap for a
better recipe to close.

    python scripts/program_oct22/register_nov10d.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_nov9m as m  # noqa: E402

R, OUT, DRY = m.R, m.OUT, m.DRY
D, SEEDS, SHARDS = 30_000, 3, 4
ARCH = "dream_rnn"
EP, WD, LR = 1600, 0.2, 0.01  # DREAM-RNN's measured 30k winner
BASELINE_GENOMIC = 0.8511  # what that winner scores, for the report to beat
# (batch size, optimizer); the first is the measured point and is NOT re-run
PINNED = (1024, "adamw")
POINTS = ((256, "adamw"), (512, "adamw"), (2048, "adamw"), (1024, "muon"))
TAG = f"refrnnrecipe_{D // 1000}k"
MENU_PATH = f"outputs/menus_nov10/{TAG.upper()}.json"


def main():
    import importlib.util

    import torch

    from albench.arch_search.harness_standard import INPUT_LEN
    from albench.arch_search.validate import static_check

    tag, mod, n_params = next(a for a in m.ARCHS if a[0] == ARCH)
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

    # a Muon point that silently trains under AdamW would answer the wrong question
    assert importlib.util.find_spec("muon") is not None, "muon is not installed; drop that point"
    assert PINNED not in POINTS, f"{PINNED} is the measured point; re-running it wastes a cell"
    assert len({p for p in POINTS}) == len(POINTS), "duplicate point"

    entries = [
        {
            "id": f"{tag}_e{EP}_bs{bs}_{opt}",
            "cell": f"{tag}_ref",
            "idea": f"published {tag} at its 30k winner (e{EP}, wd {WD}, lr {LR}) with batch {bs} and {opt}",
            "orig_val": 0.0,
            "n_params": n_params,
            "source": mod.SOURCE,
            "train_config": {
                "optimizer": opt,
                "lr": LR,
                "batch_size": bs,
                "lr_schedule": "onecycle",
                "weight_decay": WD,
                "epochs": EP,
            },
        }
        for bs, opt in POINTS
    ]
    assert len({e["id"] for e in entries}) == len(entries), "two points collapsed to one id"
    if not DRY:
        os.makedirs(os.path.join(R, os.path.dirname(MENU_PATH)), exist_ok=True)
        json.dump(entries, open(os.path.join(R, MENU_PATH), "w"), indent=1)

    odir = f"{OUT}/{TAG}_genomic_can"
    pool = f"{m.NEST}/genomic_d{D}.npz"
    cells = [
        dict(
            name=f"{TAG}_genomic_can_s{k}",
            gpus=["h100", "v100"],
            ext=[],
            out_dir=odir,
            env=dict(m.ENV),
            needs=[pool, MENU_PATH],
            cmd=(
                f"uv run --no-sync python scratchpad/menu_retrain.py "
                f"--menu {R}/{MENU_PATH} --reservoir_cache {R}/{pool} --D {D} "
                f"--out_dir {R}/{odir} --seeds {SEEDS} --shard {k}/{SHARDS} "
                f"--base_cap_min 600 --val_cache {m.VAL} --flank_ctx 20 --shift_max 10"
            ),
            done=dict(type="file", path=f"{odir}/COMPLETE_{k}of{SHARDS}"),
        )
        for k in range(SHARDS)
    ]
    for c in cells:
        assert " --shift_max 10" in c["cmd"] and " --flank_ctx 20" in c["cmd"], (
            "the 240-bp standard"
        )
        # --optimizer would override every entry; these points differ BY optimizer
        assert " --optimizer " not in c["cmd"], "the per-entry optimizer must stand"
        od = os.path.join(R, c["out_dir"])
        assert not os.path.isdir(od) or not os.listdir(od), f"{c['out_dir']} is not empty"
    print(
        f"guard: {len(cells)} cells, {len(entries)} points x {SEEDS} seeds at e{EP}/wd{WD}/lr{LR}; "
        f"beat {BASELINE_GENOMIC} genomic to matter"
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
        shutil.copy(path, path + ".bak_nov10d")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
