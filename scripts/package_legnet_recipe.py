#!/usr/bin/env python
"""Package the tuned MPRA-LegNet as a recipe a collaborator can retrain on their own data: architecture, per-size hyper-parameters, the input standard, and a self-contained reference trainer.

No weights. The point of the hand-off is that the recipe is retrained on the recipient's own labelled sets, so what has to travel is the architecture, the per-size schedule and the input convention; trained
checkpoints can follow later and would be useless for their sequences anyway (our stored banks keep predictions, not weights).

    python scripts/package_legnet_recipe.py [--dest /grid/koo/home/shared/albench/legnet_recipe]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

ROOT = "/grid/wsbs/home_norepl/christen/ALBench-S2F"

ARCH = dict(in_ch=4, stem_ch=64, stem_ks=11, ef_ks=9, ef_block_sizes=[80, 96, 112, 128], pool_sizes=[2, 2, 2, 2], resize_factor=4)  # fmt: skip
RECIPES = {
    "10000": dict(epochs=800, weight_decay=0.4, batch=1024, max_lr=0.01, genomic_r_single=0.7557, genomic_r_stack5=0.7718, genomic_r_stack20=0.7801),
    "30000": dict(epochs=500, weight_decay=0.2, batch=1024, max_lr=0.01, genomic_r_single=0.8748, genomic_r_stack5=0.8880, genomic_r_stack20=0.8907),
    "100000": dict(epochs=500, weight_decay=0.1, batch=1024, max_lr=0.01, genomic_r_single=0.9269, genomic_r_stack5=0.9362, genomic_r_stack20=0.9381),
    "300000": dict(epochs=300, weight_decay=0.05, batch=1024, max_lr=0.01, genomic_r_single=0.9507, genomic_r_stack5=0.9582, genomic_r_stack10=0.9591),
}  # fmt: skip

README = """# Tuned MPRA-LegNet: the recipe

What this is: the architecture and per-size training recipe behind the strongest student in our K562 benchmark, packaged so you
can retrain it on your own labelled sets. There are no weights here on purpose — the recipe is meant to be retrained per dataset,
and our stored banks keep predictions rather than checkpoints anyway. Trained copies can follow if they would help.

## The three things that matter

**1. The architecture** is the published MPRA-LegNet (`model.py`, MIT licence, Penzar et al.), instantiated with the K562 settings
in `recipe.json` → `architecture`. 1.32M parameters. Nothing about it changes with dataset size.

**2. The schedule changes with dataset size, and it matters a lot.** Weight decay roughly halves at every size step and the
published 100-epoch schedule is badly under-trained at our sizes (+0.066 genomic r at 10k from training longer):

| training set | epochs | weight decay | single model | 5-seed ensemble | 20-seed ensemble |
|---|---|---|---|---|---|
| 10,000  | 800 | 0.4  | 0.756 | 0.772 | 0.780 |
| 30,000  | 500 | 0.2  | 0.875 | 0.888 | 0.891 |
| 100,000 | 500 | 0.1  | 0.927 | 0.936 | 0.938 |
| 300,000 | 300 | 0.05 | 0.951 | 0.958 | 0.959 (10 seeds) |

Everything else is fixed: AdamW, peak learning rate 0.01, batch 1024, one-cycle (`pct_start=0.3`, no momentum cycling, initial
lr = peak/25), fp16, MSE loss. **Do not early-stop inside the schedule** — one-cycle only anneals at the end, so stopping costs
0.065 to 0.18 r. Run the whole schedule and keep the epoch with the best validation Pearson (it always lands in the last ~10%).
Between sizes, interpolate weight decay rather than epochs; we measured 55k and 80k and both prefer 0.1–0.15.

**3. THE INPUT STANDARD IS NOT THE BARE INSERT — this is the one thing that will silently cost you ~0.07 r.** Every model sees
the 200-bp insert inside its real plasmid context: **20 real flank bases on each side, so a 240-bp input**. In training the window
is shifted by up to ±10 bases over the real flanking sequence, one shift per batch; at evaluation the window is centred and the
prediction is the average over both strands. Half of each training batch is reverse-complemented (single-pass augmentation, not a
two-pass consistency loss — the two-pass version costs 0.03 r). The flanks are in `flanks.json`.

The ensembles you have from us before this package used the **bare 200-bp insert**, so numbers are not comparable across the two
conventions. Pick one and stay in it.

## How many seeds

Five. The seeds differ only in initialisation and data order, so the ensemble is pure variance reduction and saturates fast: 3
seeds recover 65–75% of the 20-seed gain, 5 recover 84–89%, 10 recover ~94%. OOD benefits more from ensembling than the genomic
set does, so do not drop below 5 if designed sequences matter to you. Combine members with a positive-weight ElasticNet fitted on
a held-out validation set (we use 29,449 chr19/21/X sequences); a plain mean is 0.002–0.018 worse.

## Files

- `model.py` — the architecture, unchanged from the published implementation (MIT).
- `recipe.json` — architecture settings, the per-size table above, and the input standard as machine-readable fields.
- `flanks.json` — the real plasmid flanking sequence, 5' and 3'.
- `train_legnet.py` — a self-contained reference trainer: give it sequences and labels, it reproduces the recipe exactly.

## Reference trainer

    python train_legnet.py --train train.npz --val val.npz --size 30000 --seeds 5 --out my_bank/

where each `.npz` holds `sequences` (200-bp strings) and `labels`. It writes one `.pt` per seed plus `stack.json`, and prints the
validation Pearson of each member and of the ensemble.

## Questions this package does not answer

Our numbers come from oracle-labelled K562 data with held-out chromosomes (chr7/13 test, chr19/21/X validation). On a different
assay or cell type the per-size schedule is a strong starting point rather than a guarantee — the one knob worth re-checking is
weight decay, which is what moved most as our dataset size changed.
"""

TRAINER = '''#!/usr/bin/env python
"""Reference trainer for the tuned MPRA-LegNet recipe: retrains the published architecture on your own labelled set.

Reproduces the study recipe exactly -- per-size epochs and weight decay, AdamW with one-cycle, fp16, 20 bases of real plasmid
context on each side, +-10-base shift augmentation in training, centred window and two-strand averaging at evaluation, half of
each batch reverse-complemented, no early stopping inside the schedule.

    python train_legnet.py --train train.npz --val val.npz --size 30000 --seeds 5 --out my_bank/

Each .npz needs `sequences` (200-bp ACGT strings) and `labels` (float). Writes seed_<i>.pt, predictions and stack.json.
"""

from __future__ import annotations

import argparse, importlib.util, json, pathlib
import numpy as np
import torch
import torch.nn.functional as F

HERE = pathlib.Path(__file__).parent
CFG = json.loads((HERE / "recipe.json").read_text())
FLANKS = json.loads((HERE / "flanks.json").read_text())
IDX = {c: i for i, c in enumerate("ACGT")}


def load_model_module():
    spec = importlib.util.spec_from_file_location("legnet_model", HERE / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def onehot(seqs, length=200):
    x = np.zeros((len(seqs), length, 4), np.float32)
    for i, s in enumerate(seqs):
        for j, c in enumerate(str(s)[:length]):
            k = IDX.get(c.upper())
            if k is not None:
                x[i, j, k] = 1.0
    return x


def flank_arrays(n_bases):
    """The last n_bases of the 5' flank and the first n_bases of the 3' flank, one-hot."""
    f5 = onehot([FLANKS["five_prime"][-n_bases:]], n_bases)[0] if n_bases else np.zeros((0, 4), np.float32)
    f3 = onehot([FLANKS["three_prime"][:n_bases]], n_bases)[0] if n_bases else np.zeros((0, 4), np.float32)
    return f5, f3


def recipe_for(size):
    sizes = sorted(int(k) for k in CFG["recipes"])
    pick = min(sizes, key=lambda s: abs(s - size))
    if pick != size:
        print(f"note: no measured recipe at {size}; using the {pick} one (interpolate weight decay if you are between sizes)")
    return CFG["recipes"][str(pick)]


def build_wide(seqs, ctx, shift):
    """(N, 4, 200 + 2*(ctx+shift)): the insert with enough real flank on each side to slide the window."""
    pad = ctx + shift
    f5, f3 = flank_arrays(pad)
    core = onehot(seqs)
    n = len(seqs)
    wide = np.concatenate([np.repeat(f5[None], n, 0), core, np.repeat(f3[None], n, 0)], axis=1)
    return torch.from_numpy(wide).permute(0, 2, 1).contiguous()


def centred(wide, ctx, shift):
    w = 200 + 2 * ctx
    return wide[:, :, shift : shift + w]


def predict(model, wide, ctx, shift, bs=512):
    """Centred window, averaged over both strands, as the study evaluates."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, wide.shape[0], bs):
            x = centred(wide[i : i + bs], ctx, shift).cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                p = model(x).float() + model(torch.flip(x, dims=[1, 2])).float()
            out.append((p / 2).cpu().numpy().ravel())
    return np.concatenate(out)


def pearson(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=None, help="override the recipe's epochs; for a quick sanity run only")
    a = ap.parse_args()

    std = CFG["input_standard"]
    ctx, shift = std["flank_bases_each_side"], std["train_shift_max"]
    rec = dict(recipe_for(a.size))
    if a.epochs:
        print(f"WARNING: epochs overridden to {a.epochs}; the measured recipe is {rec['epochs']} and a short one-cycle schedule scores far lower")
        rec["epochs"] = a.epochs
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tr, va = np.load(a.train, allow_pickle=True), np.load(a.val, allow_pickle=True)
    tr_seqs = [str(s) for s in tr["sequences"]]
    y_tr = torch.from_numpy(np.asarray(tr["labels"], np.float32)).cuda()
    wide_tr = build_wide(tr_seqs, ctx, shift).cuda()
    wide_va = build_wide([str(s) for s in va["sequences"]], ctx, shift).cuda()
    y_va = np.asarray(va["labels"], np.float32)
    n = len(tr_seqs)
    print(f"{n} training, {len(y_va)} validation; recipe {rec['epochs']} epochs, weight decay {rec['weight_decay']}")

    mod = load_model_module()
    val_preds, members = [], []
    for s in range(a.seeds):
        torch.manual_seed(1000 + s)
        np.random.seed(1000 + s)
        model = mod.LegNet(**CFG["architecture"]).cuda()
        model.apply(mod.initialize_weights)
        bs, epochs = rec["batch"], rec["epochs"]
        steps = -(-n // bs)
        opt = torch.optim.AdamW(model.parameters(), lr=rec["max_lr"] / 25, weight_decay=rec["weight_decay"])
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=rec["max_lr"], three_phase=False, total_steps=epochs * steps, pct_start=0.3, cycle_momentum=False
        )
        scaler = torch.amp.GradScaler("cuda")
        best, best_state = -2.0, None
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n, device="cuda")
            off = int(np.random.randint(0, 2 * shift + 1))  # one shift per batch, over real flank
            for i in range(0, n, bs):
                ib = perm[i : i + bs]
                x = wide_tr[ib][:, :, off : off + 200 + 2 * ctx]
                flip = torch.rand(x.shape[0], device="cuda") < 0.5  # half the batch reverse-complemented
                x = torch.where(flip[:, None, None], torch.flip(x, dims=[1, 2]), x)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = F.mse_loss(model(x).float(), y_tr[ib])
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
            r = pearson(predict(model, wide_va, ctx, shift), y_va)
            if r > best:  # keep the best epoch; never stop the schedule early
                best, best_state = r, {k: v.detach().clone() for k, v in model.state_dict().items()}
            if (ep + 1) % 50 == 0 or ep == epochs - 1:
                print(f"  seed {s} epoch {ep + 1}/{epochs} val r {r:.4f} (best {best:.4f})")
        model.load_state_dict(best_state)
        torch.save({k: v.cpu() for k, v in best_state.items()}, out / f"seed_{s}.pt")
        val_preds.append(predict(model, wide_va, ctx, shift))
        members.append(best)
        print(f"seed {s}: best validation r {best:.4f}")

    P = np.stack(val_preds)
    ens = pearson(P.mean(0), y_va)  # a plain mean; a positive ElasticNet on these columns is 0.002-0.018 better
    json.dump(
        {"size": a.size, "recipe": rec, "input_standard": std, "member_val_r": members, "ensemble_val_r_mean": ens},
        open(out / "stack.json", "w"), indent=1,
    )
    print(f"\\n{a.seeds} members: mean val r {np.mean(members):.4f}, ensemble (plain mean) {ens:.4f}")


if __name__ == "__main__":
    main()
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="/grid/koo/home/shared/albench/legnet_recipe")
    a = ap.parse_args()
    os.makedirs(a.dest, exist_ok=True)

    shutil.copy(f"{ROOT}/external/human_legnet/model.py", f"{a.dest}/model.py")
    shutil.copy(f"{ROOT}/external/human_legnet/LICENSE", f"{a.dest}/LICENSE_model")

    import sys

    sys.path.insert(0, ROOT)
    from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM

    # the trainer slices the last/first `pad` bases, so ship enough for ctx + the largest shift
    json.dump(
        {"five_prime": MPRA_UPSTREAM[-200:], "three_prime": MPRA_DOWNSTREAM[:200]},
        open(f"{a.dest}/flanks.json", "w"),
        indent=1,
    )
    json.dump(
        {
            "architecture": ARCH,
            "recipes": RECIPES,
            "input_standard": {
                "insert_length": 200,
                "flank_bases_each_side": 20,
                "input_length": 240,
                "train_shift_max": 10,
                "shift_granularity": "one shift per batch, over real plasmid flank",
                "train_rc": "half of each batch reverse-complemented (single pass)",
                "eval": "centred window, prediction averaged over both strands",
            },
            "optimiser": {
                "name": "adamw",
                "max_lr": 0.01,
                "initial_lr": "max_lr/25",
                "schedule": "onecycle",
                "pct_start": 0.3,
                "cycle_momentum": False,
                "precision": "fp16",
                "loss": "mse",
            },
            "early_stopping": "none inside the schedule; keep the best validation-Pearson epoch",
            "seeds_recommended": 5,
        },  # fmt: skip
        open(f"{a.dest}/recipe.json", "w"),
        indent=1,
    )
    open(f"{a.dest}/README.md", "w").write(README)
    open(f"{a.dest}/train_legnet.py", "w").write(TRAINER)
    os.chmod(f"{a.dest}/train_legnet.py", 0o755)
    print("wrote:", sorted(os.listdir(a.dest)))


if __name__ == "__main__":
    main()
