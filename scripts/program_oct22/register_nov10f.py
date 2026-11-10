"""Nov-10: does a corpus conclusion survive a change of architecture? The invariance check the scaling curves actually need.

Every scaling and acquisition number in this study is measured with an ensemble of tuned MPRA-LegNet
models, and today's pooling measurement showed the loop does not change that: 12 of 12 best-of-run
models at 30k and 6 of 6 at 100k are LegNet-derived. So a reviewer can fairly ask whether the
curves measure "what this corpus is worth" or "what this corpus is worth TO LEGNET".

The usual answer -- find architectures as good as LegNet and ensemble them -- has now failed three
times over: the published references are 0.02 to 0.05 behind at matched ensemble size and add
nothing in union. But that answer was never the one the question needs. A corpus comparison is a
RANKING claim. It does not require alternative architectures to be as accurate; it requires them to
AGREE about which corpus is better. An architecture that is 0.04 worse everywhere and still ranks
the corpora the same way is strong evidence that the ranking is a property of the data rather than
of LegNet's inductive bias -- and that is exactly what the scaling curves claim.

So: the same two 30k corpora, trained with all four architecture families, each under its OWN
measured 30k recipe, 3 seeds each.

    genomic_d30000              the al52 pilot corpus
    motif_shared_core_d30000    the al54 pilot corpus

These two were chosen because the comparison is already a headline result and a large one: the
motif_shared_core pilot is worth +0.044 designed over the genomic pilot, more than three rounds of
acquisition buy. If DREAM-RNN, DREAM-Attn and Malinois all reproduce the sign and roughly the
magnitude of that gap, the result is architecture-invariant and can be stated as such. If they
disagree, we have found a confound that matters more than anything else on the board, and it is
much better to find it here than in review.

    mpra_legnet  e500  wd 0.2  lr 0.01     the per-size tuned entry
    dream_rnn    e1600 wd 0.2  lr 0.01     measured Nov 10, ~3x LegNet's epochs
    malinois     e800  wd 0.2  lr 0.01     measured Nov 10
    dream_attn   e500  wd 0.2  lr 0.003    measured Nov 10, the low-LR point

About 6 GPU-hours, which is the cheapest defensibility this program can buy.

    python scripts/program_oct22/register_nov10f.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_nov9m as m  # noqa: E402

sys.path.insert(0, m.R)
from albench.arch_search import mpra_legnet_ref  # noqa: E402

R, OUT, DRY = m.R, m.OUT, m.DRY
D, SEEDS, SHARDS = 30_000, 3, 4
CORPORA = ("genomic", "motif_shared_core")
MOD = {a[0]: a[1] for a in m.ARCHS}
# (tag, module, epochs, weight decay, peak lr) -- each architecture's OWN measured 30k winner
ARCHS = (
    ("mpra_legnet", mpra_legnet_ref, 500, 0.2, 0.01),
    ("dream_rnn", MOD["dream_rnn"], 1600, 0.2, 0.01),
    ("dream_attn", MOD["dream_attn"], 500, 0.2, 0.003),
    ("malinois", MOD["malinois"], 800, 0.2, 0.01),
)
KNOWN = {"dream_rnn": 4_199_553, "dream_attn": 4_620_289, "malinois": 2_971_901}
MENU_PATH = f"outputs/menus_nov10/ARCHINVAR_{D // 1000}k.json"


def main():
    import torch

    from albench.arch_search.harness_standard import INPUT_LEN
    from albench.arch_search.validate import static_check

    entries, sizes = [], {}
    for tag, mod, ep, wd, lr in ARCHS:
        bad = static_check(mod.SOURCE)
        assert bad is None, f"{tag}: the harness validator rejects this source -- {bad}"
        ns = {}
        exec(compile(mod.SOURCE, f"<{tag}>", "exec"), ns)
        net = ns["Net"]()
        n_params = sum(p.numel() for p in net.parameters())
        if tag in KNOWN:
            assert n_params == KNOWN[tag], f"{tag}: {n_params:,}, expected {KNOWN[tag]:,}"
        net.eval()
        with torch.no_grad():
            y = net(torch.randn(2, 4, INPUT_LEN))
        assert y.shape == (2,) and torch.isfinite(y).all(), f"{tag}: output {tuple(y.shape)}"
        sizes[tag] = n_params
        entries.append(
            {
                "id": f"{tag}_e{ep}_wd{str(wd).replace('.', '')}_lr{str(lr).replace('.', '')}",
                "cell": f"{tag}_invar",
                "idea": f"{tag} at its own measured 30k winner (e{ep}, wd {wd}, lr {lr})",
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
        )
    # four DIFFERENT families, or this measures nothing about invariance
    assert len({e["n_params"] for e in entries}) == len(entries), (
        f"two architectures match: {sizes}"
    )
    assert len(entries) == 4, len(entries)
    print("guard: " + ", ".join(f"{t} {sizes[t]:,}" for t, *_ in ARCHS))

    if not DRY:
        os.makedirs(os.path.join(R, os.path.dirname(MENU_PATH)), exist_ok=True)
        json.dump(entries, open(os.path.join(R, MENU_PATH), "w"), indent=1)

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
                    needs=[pool, MENU_PATH],
                    cmd=(
                        f"uv run --no-sync python scratchpad/menu_retrain.py "
                        f"--menu {R}/{MENU_PATH} --reservoir_cache {R}/{pool} --D {D} "
                        f"--out_dir {R}/{odir} --seeds {SEEDS} --shard {k}/{SHARDS} "
                        f"--base_cap_min 600 --val_cache {m.VAL} --flank_ctx 20 --shift_max 10"
                    ),
                    done=dict(type="file", path=f"{odir}/COMPLETE_{k}of{SHARDS}"),
                )
            )
    # the two corpora must differ ONLY in the pool, or the comparison is not a corpus comparison
    a = [c["cmd"] for c in cells if CORPORA[0] in c["out_dir"]][0]
    b = [c["cmd"] for c in cells if CORPORA[1] in c["out_dir"]][0]
    assert a.replace(CORPORA[0], CORPORA[1]) == b, "the two arms differ by more than the corpus"
    for c in cells:
        assert " --shift_max 10" in c["cmd"] and " --flank_ctx 20" in c["cmd"], (
            "the 240-bp standard"
        )
        od = os.path.join(R, c["out_dir"])
        assert not os.path.isdir(od) or not os.listdir(od), f"{c['out_dir']} is not empty"
    print(f"guard: {len(cells)} cells, 4 architectures x {len(CORPORA)} corpora x {SEEDS} seeds")

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
        shutil.copy(path, path + ".bak_nov10f")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
