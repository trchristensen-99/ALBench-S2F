"""Score the DREAM eval sequences with one trained yeast oracle fold.

Writes per-fold predictions restricted to the sequences that fold HELD OUT, so the
aggregate assembled by eval_yeast_oracle_by_class.py is genuinely out-of-fold: no
sequence is ever scored by a model that trained on it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fold-id", type=int, required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--root", default="outputs/yeast_oracle_v2")
    ap.add_argument("--classes-npz", default="data/yeast/eval_classes_v2.npz")
    ap.add_argument("--batch-size", type=int, default=1024)
    args = ap.parse_args()

    from data.yeast import YeastDataset
    from models.dream_rnn import create_dream_rnn

    fold_dir = REPO / args.root / f"fold_{args.fold_id}{args.tag}"
    ckpt_path = fold_dir / "best_model.pt"
    if not ckpt_path.exists():
        print(f"no checkpoint at {ckpt_path}", file=sys.stderr)
        return 1

    z = np.load(REPO / args.classes_npz, allow_pickle=True)
    fold = z["fold"]
    # Only the sequences THIS fold held out, so the aggregate is out-of-fold.
    idx = np.flatnonzero(fold == args.fold_id)
    seqs = [str(s) for s in z["sequences"][idx]]
    print(f"fold {args.fold_id}{args.tag}: scoring {len(seqs):,} held-out eval sequences")

    ds = YeastDataset(
        data_path=str(REPO / "data" / "yeast"),
        split="train",
        subset_size=1000,
        context_mode="dream150",
    )
    processed = [str(x) for x in ds._add_plasmid_context(np.array(seqs))]

    ckpt = torch.load(ckpt_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_dream_rnn(
        input_channels=6,
        sequence_length=ds.get_sequence_length(),
        task_mode="yeast",
        hidden_dim=ckpt["args"]["hidden_dim"],
        cnn_filters=ckpt["args"]["cnn_filters"],
        dropout_cnn=ckpt["args"]["dropout_cnn"],
        dropout_lstm=ckpt["args"]["dropout_lstm"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(processed), args.batch_size):
            chunk = processed[i : i + args.batch_size]
            x = np.stack([ds.encode_sequence(s, {"is_singleton": 0.0}) for s in chunk])
            out = model(torch.from_numpy(x).float().to(device))
            preds.append(out.detach().cpu().numpy().ravel())
    pred = np.concatenate(preds)
    assert pred.shape[0] == len(seqs), f"{pred.shape} vs {len(seqs)}"

    out = fold_dir / "eval_class_predictions.npz"
    np.savez_compressed(out, idx=idx, pred=pred.astype(np.float32))
    print(f"  wrote {out}  mean={pred.mean():.3f} sd={pred.std():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
