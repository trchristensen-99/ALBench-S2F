"""Nov-10: close the two reference rows whose brackets are still open at the top, in the corner each one actually wants.

The 30k bracket came back and it says three things, two of which change what to run next.

**DREAM-RNN is finally bounded.** e500 .8724, e800 .8781, e1200 .8839, e1600 .8859 -- the last step
is +0.0020, inside the seed band, so the knee is around e1200-1600 and the row is done. It needs
roughly three times LegNet's measured 30k optimum of 500 epochs, which is the clearest evidence in
this study that handing a published model LegNet's recipe understates it.

**DREAM-Attn wanted a lower peak learning rate, not more precision.** At lr 0.01 its best point is
.8156 and two of its six recipes lost models to non-finite fp16 losses; at lr 0.003 it reaches
.8511 with all three seeds intact. The instability was the learning rate, and the fp32 re-run I
registered for it was the wrong remedy -- it also OOMed on the 10.75 GiB cards, so it answered
nothing. What is missing is the winning corner: lr 0.003 was only ever run at e500, while the
e500 -> e800 step at lr 0.01 was worth +0.037. This row stays fp16, because lr 0.003 is stable there.

**Malinois is still unbounded above**, exactly where DREAM-RNN was yesterday: e300 .8276, e500 .8503,
e800 .8588, monotone to the top of its bracket. Two more points settle it.

Both rows keep wd 0.2 and their own winning learning rate, so each remains a single-axis extension
of the bracket already measured rather than a new search.

    python scripts/program_oct22/register_nov10a.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_nov9m as m  # noqa: E402

R, OUT, DRY = m.R, m.OUT, m.DRY
D, SEEDS = m.D, m.SEEDS
# (architecture, slug, shards, recipes as (epochs, weight decay, peak lr), the top epoch ALREADY
# measured at that row's learning rate). The bound is per row because it is per learning rate:
# DREAM-Attn's winning lr 0.003 was only ever run at e500, so e800 extends it, while e800 is
# already the top of every lr 0.01 row.
PLAN = (
    (
        "dream_attn",
        "refattnlong",
        3,
        ((800, 0.2, 0.003), (1200, 0.2, 0.003), (1600, 0.2, 0.003)),
        500,
    ),
    ("malinois", "refmallong", 2, ((1200, 0.2, 0.01), (1600, 0.2, 0.01)), 800),
)


def entries_for(arch: str, recipes):
    tag, mod, n_params = next(a for a in m.ARCHS if a[0] == arch)
    return [
        {
            "id": f"{tag}_e{ep}_wd{str(wd).replace('.', '')}_lr{str(lr).replace('.', '')}",
            "cell": f"{tag}_ref",
            "idea": f"published {tag} at epochs {ep}, weight decay {wd}, peak lr {lr}",
            "orig_val": 0.0,
            "n_params": n_params,
            "source": mod.SOURCE,
            "train_config": {
                "optimizer": "adamw",
                "lr": lr,
                "batch_size": 1024,
                "lr_schedule": "onecycle",
                "weight_decay": wd,
                "epochs": ep,
            },
        }
        for ep, wd, lr in recipes
    ]


def main():
    import torch

    from albench.arch_search.harness_standard import INPUT_LEN
    from albench.arch_search.validate import static_check

    cells, summary = [], []
    for arch, slug, shards, recipes, measured_top in PLAN:
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

        # an extension that does not sit above the measured top answers nothing
        assert all(ep > measured_top for ep, _, _ in recipes), (
            f"{tag}: e{measured_top} is already measured at lr {recipes[0][2]}"
        )
        assert len({lr for _, _, lr in recipes}) == 1, f"{tag}: one learning rate per extension"

        entries = entries_for(arch, recipes)
        menu_path = f"outputs/menus_nov10/{slug.upper()}_{D // 1000}k.json"
        odir = f"{OUT}/{slug}_{D // 1000}k_genomic_can"
        if not DRY:
            os.makedirs(os.path.join(R, os.path.dirname(menu_path)), exist_ok=True)
            json.dump(entries, open(os.path.join(R, menu_path), "w"), indent=1)
        pool = f"{m.NEST}/genomic_d{D}.npz"
        for k in range(shards):
            cells.append(
                dict(
                    name=f"{slug}_{D // 1000}k_genomic_can_s{k}",
                    gpus=["h100", "v100"],
                    ext=[],
                    out_dir=odir,
                    env=dict(m.ENV),
                    needs=[pool, menu_path],
                    cmd=(
                        f"uv run --no-sync python scratchpad/menu_retrain.py "
                        f"--menu {R}/{menu_path} --reservoir_cache {R}/{pool} --D {D} "
                        f"--out_dir {R}/{odir} --seeds {SEEDS} --shard {k}/{shards} "
                        f"--base_cap_min 600 --val_cache {m.VAL} --flank_ctx 20 --shift_max 10"
                    ),
                    done=dict(type="file", path=f"{odir}/COMPLETE_{k}of{shards}"),
                )
            )
        summary.append(f"{tag} {[e for e, _, _ in recipes]} @ lr {recipes[0][2]}")

    for c in cells:
        assert " --shift_max 10" in c["cmd"] and " --flank_ctx 20" in c["cmd"], (
            "the 240-bp standard"
        )
        assert c["env"]["AL_PRECISION"] == "fp16", "lr 0.003 is stable in fp16; fp32 OOMed"
        od = os.path.join(R, c["out_dir"])
        assert not os.path.isdir(od) or not os.listdir(od), f"{c['out_dir']} is not empty"
    print(f"guard: {len(cells)} cells | " + " | ".join(summary) + f" | x {SEEDS} seeds")

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
        shutil.copy(path, path + ".bak_nov10a")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
