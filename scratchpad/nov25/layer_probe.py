"""Which layer's geometry is worth running a diversity rule in?

The concern is that the penultimate layer is essentially a scaled copy of predicted
activity, in which case "k-means++ for diversity" is really "spread out along the activity
axis" wearing a different name -- and all the diversity arms in the study have been running
in that space, because LegNetBank.embed() hooks the input to the final Linear.

This measures it rather than arguing about it. For every layer it reports:

  act_r2        how much of the model's OWN predicted activity a linear readout of this
                embedding recovers. Near 1.0 at a layer means the embedding is, to a linear
                map, the prediction -- which is the claim under test.
  pc1_var       share of embedding variance in the first principal component. A layer that
                is nearly rank-1 has one direction and no geometry for a distance-based
                rule to work with.
  pc1_act_corr  |correlation| between PC1 and predicted activity. High TOGETHER WITH a high
                pc1_var is the specific failure: not just one direction, but that direction.
  eff_dim       participation ratio (sum(l)^2 / sum(l^2)) -- how many directions actually
                carry variance, which is the number a farthest-point rule has to play with.

Then it runs the same farthest-point selection in each layer's space and reports what the
selections look like: how much they overlap, and what activity range they cover. Two layers
that pick the same sequences are not two choices.

    python scratchpad/nov25/layer_probe.py --bank <dir> --pool <npz> --out probe.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("ALBENCH_REPO", Path(__file__).resolve().parents[2]))

# Named from the module tree: stem, four main blocks, the mapper, and the two points inside
# the head. `head.2` is what embed() currently returns -- it is the input to the final
# Linear, so it is the penultimate representation the existing arms have been using.
LAYERS = ("stem", "main.0", "main.1", "main.2", "main.3", "mapper", "head.0", "head.2")


def capture(bank, seqs, layers, device, chunk=512):
    """{layer: (N, C)} activations, mean-pooled over position where the layer is spatial.

    Mean pooling is a real choice and it costs something: a convolutional block's
    positional structure is exactly where motif SYNTAX would live, and pooling discards
    it. It is used here because a distance in (C, L) space is dominated by shifts rather
    than content, and because every candidate layer has to be compared on the same footing.
    """
    import torch

    model = bank.models[0]
    feats: dict[str, list[np.ndarray]] = {ln: [] for ln in layers}
    mods = dict(model.named_modules())
    missing = [ln for ln in layers if ln not in mods]
    if missing:
        raise SystemExit(f"layers not in this model: {missing}")

    def hook(name):
        def fn(_m, _i, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            if o.dim() == 3:  # (B, C, L) -> mean over position
                o = o.mean(dim=2)
            feats[name].append(o.detach().float().cpu().numpy())

        return fn

    # The bank's own encoder (_to_x) and its batch size, so the hooks observe exactly the
    # forward pass the acquisition rules see. Forward strand only: the bank averages the
    # two strands for its predictions, but an embedding is not strand-symmetric and
    # averaging two different points in feature space is not a point in feature space.
    handles = [mods[ln].register_forward_hook(hook(ln)) for ln in layers]
    preds: list[np.ndarray] = []
    try:
        with torch.no_grad():
            for i in range(0, len(seqs), chunk):
                x = bank._to_x(seqs[i : i + chunk])
                preds.append(model(x).float().reshape(-1).cpu().numpy())
    finally:
        for h in handles:
            h.remove()
    return {ln: np.concatenate(v) for ln, v in feats.items()}, np.concatenate(preds)


def describe_layer(z: np.ndarray, act: np.ndarray) -> dict:
    z = z - z.mean(axis=0, keepdims=True)
    # Drop dead units: a column that never varies contributes nothing and would deflate
    # the participation ratio for reasons that have nothing to do with geometry.
    keep = z.std(axis=0) > 1e-8
    z = z[:, keep]
    if z.shape[1] == 0:
        return {"dims": 0}
    u, s, _ = np.linalg.svd(z, full_matrices=False)
    lam = s**2
    pc1 = u[:, 0] * s[0]
    # Linear readout of the model's own prediction, via least squares on the components.
    k = min(z.shape[1], 64)
    A = np.c_[u[:, :k] * s[:k], np.ones(len(z))]
    coef, *_ = np.linalg.lstsq(A, act, rcond=None)
    resid = act - A @ coef
    r2 = 1.0 - float(resid.var() / max(1e-12, act.var()))
    return {
        "dims": int(z.shape[1]),
        "act_r2": round(r2, 4),
        "pc1_var": round(float(lam[0] / lam.sum()), 4),
        "pc1_act_corr": round(float(abs(np.corrcoef(pc1, act)[0, 1])), 4),
        "eff_dim": round(float(lam.sum() ** 2 / (lam**2).sum()), 2),
    }


def farthest_point(z: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """The same greedy rule DiversityAcquisition uses, on whatever space it is handed."""
    rng = np.random.default_rng(seed)
    n = len(z)
    first = int(rng.integers(n))
    sel = [first]
    d = ((z - z[first]) ** 2).sum(axis=1)
    while len(sel) < k:
        d[sel[-1]] = -np.inf
        nxt = int(np.argmax(d))
        sel.append(nxt)
        d = np.minimum(d, ((z - z[nxt]) ** 2).sum(axis=1))
    return np.asarray(sel)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="outputs/wed41/bench38_ln_e500_wd2_30k_c20")
    ap.add_argument("--pool", default="outputs/grid_pools/k562_nested/motif_shared_core_d30000.npz")
    ap.add_argument("--n", type=int, default=6000, help="candidates probed")
    ap.add_argument("--select", type=int, default=600)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch

    from albench.acquisition.legnet_bank import LegNetBank

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bank = LegNetBank(str(REPO / a.bank) if not Path(a.bank).is_absolute() else a.bank)
    model = bank.models[0]
    model.eval().to(device)
    print(f"device {device} | bank {len(bank)} members | layers {len(LAYERS)}")

    z = np.load(REPO / a.pool if not Path(a.pool).is_absolute() else a.pool, allow_pickle=True)
    seqs = [str(s) for s in z["sequences"][: a.n]]
    true_y = np.asarray(z["oracle_labels"][: a.n], dtype=float)
    print(f"{len(seqs):,} candidates from {Path(a.pool).name}")

    feats, act = capture(bank, seqs, LAYERS, device)
    rec: dict = {"bank": a.bank, "pool": Path(a.pool).name, "n": len(seqs), "layers": {}}

    print(f"\n{'layer':<10}{'dims':>6}{'act_r2':>9}{'pc1_var':>9}{'pc1~act':>9}{'eff_dim':>9}")
    for ln in LAYERS:
        d = describe_layer(feats[ln], act)
        rec["layers"][ln] = d
        if d.get("dims"):
            print(
                f"{ln:<10}{d['dims']:>6}{d['act_r2']:>9.3f}{d['pc1_var']:>9.3f}"
                f"{d['pc1_act_corr']:>9.3f}{d['eff_dim']:>9.2f}"
            )

    # Selections: do different layers actually choose differently?
    edges = np.quantile(true_y, np.linspace(0, 1, 11))
    sels = {}
    for ln in LAYERS:
        if not rec["layers"][ln].get("dims"):
            continue
        zz = feats[ln]
        zz = (zz - zz.mean(0)) / np.maximum(zz.std(0), 1e-8)
        sel = farthest_point(zz, a.select)
        sels[ln] = sel
        b = np.clip(np.digitize(true_y[sel], edges[1:-1]), 0, 9)
        rec["layers"][ln]["sel_decile_coverage"] = round(len(set(b.tolist())) / 10, 2)
        rec["layers"][ln]["sel_activity_mean"] = round(float(true_y[sel].mean()), 4)
        rec["layers"][ln]["sel_activity_sd"] = round(float(true_y[sel].std()), 4)

    names = list(sels)
    rec["overlap"] = {}
    print(f"\npairwise selection overlap ({a.select} picked of {len(seqs):,}):")
    print("          " + "".join(f"{n:>10}" for n in names))
    for i, ni in enumerate(names):
        row = []
        for nj in names:
            ov = len(set(sels[ni].tolist()) & set(sels[nj].tolist())) / a.select
            row.append(ov)
            rec["overlap"][f"{ni}|{nj}"] = round(ov, 3)
        print(f"{ni:<10}" + "".join(f"{v:>10.2f}" for v in row))
    # A random selection is the scale to read those overlaps against: two independent
    # draws of 600 from 6,000 share about 10% by chance, so an overlap near 0.1 means the
    # two layers agree no more than chance.
    rec["chance_overlap"] = round(a.select / len(seqs), 3)
    print(f"\nchance overlap for two independent draws: {rec['chance_overlap']:.2f}")

    Path(a.out).write_text(json.dumps(rec, indent=1))
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
