"""Score a trained student checkpoint against the expanded eval-set panel.

For one ``best.pt`` checkpoint (output of ``run_single.py``), runs RC-
averaged inference over each parquet under ``outputs/eval_sets_expanded/``
(PRM 1/5/10/20%, dinuc_shuffle, EvoAug light/medium/heavy, GC quartiles,
random_uniform), and writes per-eval-set MSE / Pearson R / Spearman to
a single JSON next to the checkpoint.

Outputs:
    <ckpt_dir>/eval_sets_panel.json  — keys: panel name → {pearson_r,
        spearman_r, mse, n_seqs, label_source}

Usage:
    uv run --no-sync python scripts/preflight/score_eval_sets.py \\
        --ckpt results/exp1_1/d_init0/legnet/random/d600000/seed42/best.pt \\
        --arch legnet
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[2]


def _load_run_single():
    spec = importlib.util.spec_from_file_location(
        "run_single", REPO / "scripts" / "preflight" / "run_single.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _evaluate_eval_set(
    model, parquet_path: Path, in_channels: int, device: torch.device, rs
) -> dict:
    df = pd.read_parquet(parquet_path)
    seqs = df["sequence"].astype(str).tolist()
    label_col = (
        "K562_log2FC" if "K562_log2FC" in df.columns else "label" if "label" in df.columns else None
    )
    labels = df[label_col].to_numpy(dtype=np.float32) if label_col else None
    payload_len = 200
    X = rs.one_hot(seqs, seq_len=payload_len, in_channels=in_channels, pad_with_adapters=False)
    Xt = torch.from_numpy(X).float().to(device)
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xt), 1024):
            xb = Xt[i : i + 1024]
            yhat = model(xb).reshape(-1)
            yhat_rc = model(rs._rc_flip(xb)).reshape(-1)
            preds.append(0.5 * (yhat + yhat_rc).cpu().numpy())
    preds = np.concatenate(preds)
    out = {
        "n_seqs": int(len(seqs)),
        "label_source": label_col,
    }
    if labels is not None and len(labels) > 1 and labels.std() > 0 and preds.std() > 0:
        out["pearson_r"] = float(pearsonr(preds, labels)[0])
        out["spearman_r"] = float(spearmanr(preds, labels)[0])
        out["mse"] = float(np.mean((preds - labels) ** 2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to best.pt from run_single.py.")
    ap.add_argument("--arch", required=True, choices=["legnet", "dream_rnn", "dream_attn"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--eval_sets_dir", default=str(REPO / "outputs" / "eval_sets_expanded"))
    args = ap.parse_args()

    rs = _load_run_single()
    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hp = ckpt["hp"]
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = rs.build_model(args.arch, hp, device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    eval_sets_dir = Path(args.eval_sets_dir)
    panels = sorted(eval_sets_dir.glob("*.parquet"))
    if not panels:
        raise SystemExit(f"no panels found in {eval_sets_dir}")
    print(f"Scoring against {len(panels)} eval-set panels …")

    results = {}
    for p in panels:
        name = p.stem
        try:
            res = _evaluate_eval_set(model, p, hp["in_channels"], device, rs)
            print(
                f"  {name}: n={res['n_seqs']:>6}  R={res.get('pearson_r', 'n/a'):.4f}  "
                f"ρ={res.get('spearman_r', 'n/a'):.4f}  MSE={res.get('mse', 'n/a')}"
            )
        except Exception as e:
            res = {"error": str(e)}
            print(f"  {name}: ERROR — {e}")
        results[name] = res

    out_path = ckpt_path.parent / "eval_sets_panel.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
