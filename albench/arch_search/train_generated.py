"""Train an LLM-written architecture in the SAME harness as a hand-specified config.

Comparability is the whole point: these models must share the training data, the
validation split, the eval battery and the prediction format with the LegNet
configs, or "did the generated architecture win?" is unanswerable. So this reuses
the existing loaders and writes the same {id}.npz / {id}_meta.json pair, which
means generated models drop straight into greedy selection and the eval matrix
alongside everything else.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments"))


def one_hot(seqs, L=200):
    idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    X = np.zeros((len(seqs), 4, L), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, ch in enumerate(s[:L].upper()):
            k = idx.get(ch)
            if k is not None:
                X[i, k, j] = 1.0
    return X


def train_one(model, Xtr, ytr, Xva, yva, *, epochs=60, bs=128, lr=2e-3, wd=1e-5,
              patience=8, device="cuda", log=print):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    lossf = torch.nn.MSELoss()
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                    batch_size=bs, shuffle=True, drop_last=True)
    Xva_t = torch.from_numpy(Xva).to(dev)
    yva_t = torch.from_numpy(yva).to(dev)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            out = model(xb).squeeze()
            loss = lossf(out, yb)
            if not torch.isfinite(loss):
                return None, {"error": f"non-finite loss at epoch {ep}"}
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vp = torch.cat([model(Xva_t[i:i+512]).squeeze()
                            for i in range(0, len(Xva_t), 512)])
            vmse = float(((vp - yva_t) ** 2).mean())
        sched.step(vmse)
        if vmse < best - 1e-4:
            best, bad = vmse, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_mse": best, "epochs_trained": ep + 1}


def predict(model, X, device="cuda", bs=512):
    import torch
    dev = next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(model(torch.from_numpy(X[i:i+bs]).to(dev)).squeeze().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", default="outputs/arch_search/round0_proposals.json")
    ap.add_argument("--reservoir_cache", required=True)
    ap.add_argument("--D", type=int, default=30000)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--variants", type=int, default=3,
                    help="seeds per architecture; init variance is separate from architecture")
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    import torch
    from scaling_hp_search_ms import (_atomic_savez, _atomic_write_text,
                                      load_all_test_sets, load_chr_test_genomic,
                                      load_chr_train_pool)
    from albench.arch_search.validate import build_and_check

    props = [p for p in json.load(open(args.proposals)) if p.get("ok")]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"  {len(props)} accepted architectures x {args.variants} seeds")

    tr_s, tr_y, va_s, va_y = load_chr_train_pool(
        args.D, ref_only=True, val_frac=0.1, seed=0,
        reservoir_cache=args.reservoir_cache, chr_val=True)
    te_s, te_oracle, te_true = load_chr_test_genomic()
    tests = load_all_test_sets()
    print(f"  train={len(tr_s):,} val={len(va_s):,} test={len(te_s):,} eval_sets={len(tests)}")

    Xtr, ytr = one_hot(tr_s), np.asarray(tr_y, np.float32)
    Xva, yva = one_hot(va_s), np.asarray(va_y, np.float32)
    Xte = one_hot(te_s)
    Xsets = {k: one_hot(v[0]) for k, v in tests.items()}

    labels = {"val_labels": yva, "test_oracle": te_oracle, "test_true": te_true}
    for k, v in tests.items():
        labels[f"oracle_{k}"] = v[1]
    _atomic_savez(Path(args.out_dir) / "labels.npz", **labels)

    for pi, p in enumerate(props):
        for v in range(args.variants):
            mid = f"arch{pi:02d}_s{v}"
            if os.path.exists(os.path.join(args.out_dir, f"{mid}_meta.json")):
                print(f"  SKIP {mid}"); continue
            torch.manual_seed(1000 + v)
            model, verdict = build_and_check(p["source"])
            if model is None:
                print(f"  SKIP {mid}: {verdict.reason}"); continue
            t0 = time.time()
            model, info = train_one(model, Xtr, ytr, Xva, yva, epochs=args.epochs)
            if model is None:
                print(f"  FAIL {mid}: {info.get('error')}"); continue
            res = {"val_pred": predict(model, Xva), "test_pred": predict(model, Xte)}
            for k, Xk in Xsets.items():
                res[f"test_pred_{k}"] = predict(model, Xk)
            r = float(np.corrcoef(res["test_pred"], te_oracle)[0, 1])
            meta = {"model_id": mid, "strategy": "arch_search", "round": 0,
                    "arch_index": pi, "idea": p["idea"], "why": p.get("why", ""),
                    "layers": p.get("layers", {}), "n_params": p.get("n_params", 0),
                    "init_seed": 1000 + v, "source": p["source"],
                    "train_time_sec": time.time() - t0,
                    "val_mse": info.get("best_val_mse"),
                    "epochs_trained": info.get("epochs_trained"),
                    "test_oracle_pearson": r}
            _atomic_savez(Path(args.out_dir) / f"{mid}.npz",
                          **{k: np.asarray(x, np.float32) for k, x in res.items()})
            _atomic_write_text(Path(args.out_dir) / f"{mid}_meta.json",
                               json.dumps(meta, indent=2, default=str))
            print(f"  {mid}  r={r:.4f}  {info.get('epochs_trained')}ep  "
                  f"{time.time()-t0:.0f}s  {p['idea'][:44]}")


if __name__ == "__main__":
    main()
