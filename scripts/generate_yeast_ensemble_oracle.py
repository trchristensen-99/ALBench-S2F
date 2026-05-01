"""Generate ensemble oracle pseudolabels for yeast train/val/test splits.

Loads 5 DREAM-RNN + 5 DREAM-CNN checkpoints (each trained on real DREAM
yeast labels at full data), runs each model on train/val/test sequences,
and saves the per-sequence mean across all 10 models as the ensemble
oracle pseudolabel.

Output:
    outputs/yeast_ensemble_oracle/{split}_pseudolabels.npz
        sequences: (N,) object array of original DNA sequences
        labels:    (N,) float32 ensemble mean prediction
        per_model_mean / per_model_std: (N,) statistics across the 10 models

Usage:
    uv run --no-sync python scripts/generate_yeast_ensemble_oracle.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]


def _reverse_complement(x: torch.Tensor) -> torch.Tensor:
    """RC for one-hot encoded tensors of shape (B, C, L), C >= 4 (ACGT order)."""
    out = x.flip(dims=[2]).clone()
    # Swap A<->T (channels 0 and 3) and C<->G (channels 1 and 2)
    out[:, [0, 1, 2, 3]] = out[:, [3, 2, 1, 0]]
    return out


@torch.no_grad()
def _predict_drnn(model, loader, device):
    """Predict with RC averaging on a DREAM-RNN (6-channel input)."""
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
    """Predict with RC averaging on a DREAM-CNN (4-channel input)."""
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
    """Construct a DREAM-RNN with default Prix Fixe HPs and load ckpt."""
    from models.dream_rnn import DREAMRNN

    # The kfold_v3 ensemble was trained with cnn_filters=256 (not 160).
    # Confirmed via outputs/oracle_dream_rnn_yeast_kfold_v3/fold_0/seed_42/config.json.
    model = DREAMRNN(
        input_channels=6,
        sequence_length=seq_len,
        task_mode="yeast",
        hidden_dim=320,
        cnn_filters=256,
        dropout_cnn=0.2,
        dropout_lstm=0.3,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    sd = state.get("model_state_dict", state.get("state_dict", state))
    model.load_state_dict(sd)
    return model


def _build_dcnn(ckpt_path: Path, device):
    """Construct a DREAM-CNN with default HPs and load ckpt.

    DCNN ckpts saved by exp1_1_scaling.py are dicts with 'model_state_dicts'
    holding a list of state_dicts (ensemble_size=1 here, so one entry).
    """
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


def _process_split(split_name: str, drnn_ckpts: list[Path], dcnn_ckpts: list[Path], out_dir: Path):
    from data.yeast import YeastDataset

    ds = YeastDataset(data_path=str(REPO / "data" / "yeast"), split=split_name)
    sequences = list(ds.sequences)
    real_labels = ds.labels.astype(np.float32)
    seq_len = ds.get_sequence_length()
    n = len(sequences)
    print(f"\n=== Split={split_name}  N={n:,}  seq_len={seq_len} ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    all_preds = np.zeros((len(drnn_ckpts) + len(dcnn_ckpts), n), dtype=np.float32)
    idx = 0

    for ckpt in drnn_ckpts:
        t0 = time.time()
        model = _build_drnn(ckpt, seq_len, device)
        all_preds[idx] = _predict_drnn(model, loader, device)
        del model
        torch.cuda.empty_cache()
        print(f"  DRNN {idx + 1}/{len(drnn_ckpts)}  {ckpt.parts[-3]}  {time.time() - t0:.1f}s")
        idx += 1

    for ckpt in dcnn_ckpts:
        t0 = time.time()
        model = _build_dcnn(ckpt, device)
        all_preds[idx] = _predict_dcnn(model, sequences, device)
        del model
        torch.cuda.empty_cache()
        print(
            f"  DCNN {idx - len(drnn_ckpts) + 1}/{len(dcnn_ckpts)}  "
            f"{ckpt.parts[-2]}  {time.time() - t0:.1f}s"
        )
        idx += 1

    mean_pred = all_preds.mean(axis=0)
    std_pred = all_preds.std(axis=0)
    out_path = out_dir / f"{split_name}_pseudolabels.npz"
    np.savez_compressed(
        out_path,
        sequences=np.array(sequences, dtype=object),
        labels=mean_pred,
        real_labels=real_labels,
        per_model_mean=mean_pred,
        per_model_std=std_pred,
        per_model_preds=all_preds,
    )
    print(
        f"  Saved {out_path}  (ensemble_mean range: [{mean_pred.min():.3f}, {mean_pred.max():.3f}])"
    )
    if split_name == "test":
        from scipy.stats import pearsonr

        r = pearsonr(mean_pred, real_labels)[0]
        print(f"  Sanity: ensemble vs real test labels Pearson R = {r:.4f}")


def main():
    out_dir = REPO / "outputs" / "yeast_ensemble_oracle"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 5 DRNN ckpts (real-label, full-data 5-fold ensemble)
    drnn_ckpts = sorted(
        (REPO / "outputs" / "oracle_dream_rnn_yeast_kfold_v3").rglob(
            "fraction_1.0000/best_model.pt"
        )
    )
    # 5 DCNN ckpts (real-label, full data, seeds 42/1042/2042/3042/4042)
    dcnn_ckpts = sorted(
        (REPO / "outputs" / "exp0_yeast_dream_cnn_real" / "genomic" / "n6065324" / "hp0").glob(
            "seed*/best_model.pt"
        )
    )
    print(f"Found {len(drnn_ckpts)} DRNN + {len(dcnn_ckpts)} DCNN checkpoints")
    if not drnn_ckpts or not dcnn_ckpts:
        raise SystemExit("Missing checkpoints — see paths above")

    for split in ("train", "val", "test"):
        _process_split(split, drnn_ckpts, dcnn_ckpts, out_dir)


if __name__ == "__main__":
    main()
