"""Quick preview of the ensemble oracle's behavior on the yeast test split.

Uses whatever DRNN + DCNN ckpts are currently available (5+3 if the DCNN
extras haven't landed yet, 5+5 once they have). Predicts on the full
71K MAUDE test split, separates predictions for the in-dist random
subset vs the OOD native-genomic subset, and reports:

  - Distribution stats of ensemble pseudolabels (mean, std, range)
  - Per-model std (uncertainty proxy) for in-dist vs OOD
  - Pearson R of ensemble pseudolabel against MAUDE labels (the cross-assay
    ceiling that any student trained on these pseudolabels can reach)

Run:
  uv run --no-sync python scripts/yeast_oracle_sanity_preview.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]


def _reverse_complement(x: torch.Tensor) -> torch.Tensor:
    out = x.flip(dims=[2]).clone()
    out[:, [0, 1, 2, 3]] = out[:, [3, 2, 1, 0]]
    return out


@torch.no_grad()
def _predict_drnn(model, loader, device):
    model.eval()
    preds = []
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        xb_rc = _reverse_complement(xb)
        y_fwd = model(xb).reshape(-1)
        y_rc = model(xb_rc).reshape(-1)
        preds.append(((y_fwd + y_rc) * 0.5).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(preds, axis=0)


@torch.no_grad()
def _predict_dcnn(model, sequences, device, batch_size=512):
    from models.dream_cnn import one_hot_encode_batch

    model.eval()
    preds = []
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i : i + batch_size]
        x = torch.from_numpy(one_hot_encode_batch(batch_seqs)).float().to(device)
        x_rc = _reverse_complement(x)
        y_fwd = model(x).reshape(-1)
        y_rc = model(x_rc).reshape(-1)
        preds.append(((y_fwd + y_rc) * 0.5).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(preds, axis=0)


def _build_drnn(ckpt_path: Path, seq_len: int, device):
    from models.dream_rnn import DREAMRNN

    model = DREAMRNN(
        input_channels=6,
        sequence_length=seq_len,
        task_mode="yeast",
        hidden_dim=320,
        cnn_filters=160,
        dropout_cnn=0.2,
        dropout_lstm=0.3,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    sd = state.get("model_state_dict", state.get("state_dict", state))
    model.load_state_dict(sd)
    return model


def _build_dcnn(ckpt_path: Path, device):
    from models.dream_cnn import DREAMCNN

    model = DREAMCNN(
        in_channels=4,
        stem_channels=320,
        core_out_channels=64,
        head_hidden=256,
        dropout=0.2,
        task_mode="yeast",
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if "model_state_dicts" in state:
        model.load_state_dict(state["model_state_dicts"][0])
    elif "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    return model


def main():
    from data.yeast import YeastDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    drnn_ckpts = sorted(
        (REPO / "outputs" / "oracle_dream_rnn_yeast_kfold_v3").rglob(
            "fraction_1.0000/best_model.pt"
        )
    )
    dcnn_ckpts = sorted(
        (REPO / "outputs" / "exp0_yeast_dream_cnn_real" / "genomic" / "n6065324" / "hp0").glob(
            "seed*/best_model.pt"
        )
    )
    print(f"Found {len(drnn_ckpts)} DRNN + {len(dcnn_ckpts)} DCNN ckpts")

    ds = YeastDataset(data_path=str(REPO / "data" / "yeast"), split="test")
    sequences = list(ds.sequences)
    real_labels = ds.labels.astype(np.float32)
    seq_len = ds.get_sequence_length()
    n = len(sequences)
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    all_preds = np.zeros((len(drnn_ckpts) + len(dcnn_ckpts), n), dtype=np.float32)
    idx = 0
    for ckpt in drnn_ckpts:
        t0 = time.time()
        model = _build_drnn(ckpt, seq_len, device)
        all_preds[idx] = _predict_drnn(model, loader, device)
        del model
        torch.cuda.empty_cache()
        print(f"  DRNN {idx + 1}/{len(drnn_ckpts)}  {time.time() - t0:.1f}s")
        idx += 1
    for ckpt in dcnn_ckpts:
        t0 = time.time()
        model = _build_dcnn(ckpt, device)
        all_preds[idx] = _predict_dcnn(model, sequences, device)
        del model
        torch.cuda.empty_cache()
        print(f"  DCNN {idx - len(drnn_ckpts) + 1}/{len(dcnn_ckpts)}  {time.time() - t0:.1f}s")
        idx += 1

    ensemble_mean = all_preds.mean(axis=0)
    per_model_std = all_preds.std(axis=0)

    # Subset masks
    rnd_idx = (
        pd.read_csv(REPO / "data" / "yeast" / "test_subset_ids" / "all_random_seqs.csv")
        .iloc[:, 0]
        .values.astype(int)
    )
    ood_idx = (
        pd.read_csv(REPO / "data" / "yeast" / "test_subset_ids" / "yeast_seqs.csv")
        .iloc[:, 0]
        .values.astype(int)
    )

    print()
    print(f"=== Sanity checks (test split, n={n:,}) ===")
    print(f"\n[A] Ensemble pseudolabel distribution (DREAM scale [0,17]):")
    print(
        f"  Overall:  mean={ensemble_mean.mean():.3f}  std={ensemble_mean.std():.3f}  "
        f"range=[{ensemble_mean.min():.3f}, {ensemble_mean.max():.3f}]"
    )
    rnd_pred, ood_pred = ensemble_mean[rnd_idx], ensemble_mean[ood_idx]
    print(
        f"  In-dist:  mean={rnd_pred.mean():.3f}  std={rnd_pred.std():.3f}  "
        f"range=[{rnd_pred.min():.3f}, {rnd_pred.max():.3f}]  n={len(rnd_pred)}"
    )
    print(
        f"  OOD:      mean={ood_pred.mean():.3f}  std={ood_pred.std():.3f}  "
        f"range=[{ood_pred.min():.3f}, {ood_pred.max():.3f}]  n={len(ood_pred)}"
    )

    print(f"\n[B] Per-model uncertainty (std across {idx} models, per sequence):")
    print(
        f"  In-dist: median per-model_std = {np.median(per_model_std[rnd_idx]):.3f}  "
        f"mean = {per_model_std[rnd_idx].mean():.3f}"
    )
    print(
        f"  OOD:     median per-model_std = {np.median(per_model_std[ood_idx]):.3f}  "
        f"mean = {per_model_std[ood_idx].mean():.3f}"
    )

    print(f"\n[C] Cross-assay Pearson (ensemble pseudolabel vs MAUDE real label):")
    r_rnd = pearsonr(rnd_pred, real_labels[rnd_idx])[0]
    r_ood = pearsonr(ood_pred, real_labels[ood_idx])[0]
    print(f"  In-dist (random subset): r = {r_rnd:.4f}  (expect ~0.81 — DRNN/DCNN single-model)")
    print(f"  OOD (native genomic):    r = {r_ood:.4f}  (expect ~0.65 — DRNN/DCNN single-model)")

    print(f"\n[D] Per-model preds spread on a representative OOD sequence:")
    sample = ood_idx[0]
    print(f"  ood seq[0]: real_label={real_labels[sample]:.3f}")
    print(f"             ensemble_mean={ensemble_mean[sample]:.3f}")
    print(f"             per_model_preds={all_preds[:, sample]}")


if __name__ == "__main__":
    main()
