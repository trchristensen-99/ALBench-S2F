"""Iterative free-architecture AutoResearch: each round the model WRITES new Net(nn.Module)
classes (no HP schema, no family names), they are validated + trained in the same harness as the
LegNet configs, and their test scores are fed back so the next round improves. Writes the same
val_pred/test_pred_* npz + meta as the HP-menu driver, so free-arch models drop straight into the
same greedy/enet_all_pos deploy scoring → directly comparable to the HP-menu grid.

Usage: python arch_autoresearch.py --reservoir_cache <npz> --D <int> --out_dir <dir>
       --rounds 12 --per_round 4 --epochs 80
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments"))

from albench.arch_search.propose import build_arch_prompt, call_model, parse_proposals
from albench.arch_search.train_generated import one_hot, predict, train_one
from albench.arch_search.validate import build_and_check


def pear(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a - a.mean(); b = b - b.mean()
    d = (a * a).sum() ** 0.5 * (b * b).sum() ** 0.5
    return float((a * b).sum() / d) if d > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reservoir_cache", required=True)
    ap.add_argument("--D", type=int, default=30000)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--per_round", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seq_len", type=int, default=200)
    args = ap.parse_args()

    from scaling_hp_search import (
        _atomic_savez, _atomic_write_text,
        load_all_test_sets, load_chr_test_genomic, load_chr_train_pool,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    tr_s, tr_y, va_s, va_y = load_chr_train_pool(
        args.D, ref_only=True, val_frac=0.1, seed=0,
        reservoir_cache=args.reservoir_cache, chr_val=True)
    te_s, te_oracle, te_true = load_chr_test_genomic()
    tests = load_all_test_sets()
    print(f"  train={len(tr_s):,} val={len(va_s):,} test={len(te_s):,} eval_sets={len(tests)}", flush=True)

    Xtr, ytr = one_hot(tr_s, args.seq_len), np.asarray(tr_y, np.float32)
    Xva, yva = one_hot(va_s, args.seq_len), np.asarray(va_y, np.float32)
    Xte = one_hot(te_s, args.seq_len)
    Xsets = {k: one_hot(v[0], args.seq_len) for k, v in tests.items()}

    labels = {"val_labels": yva, "test_oracle": te_oracle, "test_true": te_true}
    for k, v in tests.items():
        labels[f"oracle_{k}"] = v[1]
    _atomic_savez(Path(args.out_dir) / "labels.npz", **labels)

    history = []
    midx = 0
    for rd in range(args.rounds):
        try:
            sys_p, user = build_arch_prompt(args.per_round, history, seq_len=args.seq_len, D=args.D)
            text = call_model(sys_p, user)
            props = parse_proposals(text)
        except Exception as e:
            print(f"  [round {rd}] propose failed: {type(e).__name__}: {e}", flush=True)
            continue
        print(f"  [round {rd}] {len(props)} proposals", flush=True)
        for p in props:
            model, verdict = build_and_check(p["source"], seq_len=args.seq_len)
            if model is None:
                history.append({"idea": p["idea"], "ok": False, "reason": verdict.reason})
                print(f"    discard: {verdict.reason[:70]}", flush=True)
                continue
            import torch
            torch.manual_seed(1000 + midx)
            t0 = time.time()
            model, info = train_one(model, Xtr, ytr, Xva, yva, epochs=args.epochs)
            if model is None:
                history.append({"idea": p["idea"], "ok": False, "reason": info.get("error", "train fail")})
                continue
            res = {"val_pred": predict(model, Xva), "test_pred": predict(model, Xte)}
            for k, Xk in Xsets.items():
                res[f"test_pred_{k}"] = predict(model, Xk)
            r = pear(res["test_pred"], te_oracle); vr = pear(res["val_pred"], yva)
            mid = f"r{rd:02d}_arch_{midx:02d}"
            meta = {"model_id": mid, "strategy": "arch_search", "round": rd,
                    "idea": p["idea"], "why": p.get("why", ""), "n_params": verdict.n_params,
                    "val_pearson": vr, "best_val_pearson": vr, "test_oracle_pearson": r,
                    "val_mse": info.get("best_val_mse"), "epochs_trained": info.get("epochs_trained"),
                    "train_time_sec": time.time() - t0, "source": p["source"]}
            _atomic_savez(Path(args.out_dir) / f"{mid}.npz",
                          **{k: np.asarray(x, np.float32) for k, x in res.items()})
            _atomic_write_text(Path(args.out_dir) / f"{mid}_meta.json",
                               json.dumps(meta, indent=2, default=str))
            history.append({"idea": p["idea"], "ok": True, "score": r,
                            "n_params": verdict.n_params, "val": vr})
            midx += 1
            print(f"    {mid} r={r:.4f} val={vr:.4f} {verdict.n_params:,}p {info.get('epochs_trained')}ep "
                  f"{time.time()-t0:.0f}s  {p['idea'][:50]}", flush=True)
    print(f"DONE_ARCH trained={midx}", flush=True)


if __name__ == "__main__":
    main()
