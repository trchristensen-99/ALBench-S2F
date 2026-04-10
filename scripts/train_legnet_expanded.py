#!/usr/bin/env python
"""Train LegNet on expanded K562 dataset (Gosai + Agarwal negatives).

Trains LegNet with the existing Gosai chr-split data plus Agarwal intergenic
sequences and controls. Evaluates on:
  1. Standard chr-split test set (chr7+13)
  2. Held-out Agarwal shuffled controls
  3. Random DNA sequences

Usage:
    uv run --no-sync python scripts/train_legnet_expanded.py \
        --seed 42 --output-dir outputs/legnet_expanded/seed_42
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset, TensorDataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def one_hot_encode(seq: str) -> np.ndarray:
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    arr = np.zeros((4, len(seq)), dtype=np.float32)
    for i, c in enumerate(seq.upper()):
        if c in mapping:
            arr[mapping[c], i] = 1.0
    return arr


class ExpandedK562Dataset(Dataset):
    """K562 dataset combining Gosai + Agarwal elements."""

    def __init__(self, sequences, labels):
        self.sequences = sequences  # list of str
        self.labels = np.array(labels, dtype=np.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        ohe = one_hot_encode(seq)
        return torch.from_numpy(ohe), self.labels[idx]


def load_expanded_data(data_path: Path, expanded_path: Path, seed: int):
    """Load Gosai chr-split data + Agarwal expanded elements."""
    from data.k562 import K562Dataset

    # Standard chr-split train/val/test
    train_ds = K562Dataset(data_path=str(data_path), split="train")
    val_ds = K562Dataset(data_path=str(data_path), split="val")
    test_ds = K562Dataset(data_path=str(data_path), split="test")

    train_seqs = list(train_ds.sequences)
    train_labels = train_ds.labels.tolist()
    val_seqs = list(val_ds.sequences)
    val_labels = val_ds.labels.tolist()
    test_seqs = list(test_ds.sequences)
    test_labels = test_ds.labels.tolist()

    logger.info(
        "Gosai chr-split: train=%d, val=%d, test=%d",
        len(train_seqs),
        len(val_seqs),
        len(test_seqs),
    )

    # Load Agarwal expanded train elements (if available)
    agarwal_train_path = expanded_path / "agarwal_new_train.tsv"
    agarwal_test_path = expanded_path / "agarwal_new_test.tsv"

    new_train_seqs, new_train_labels, new_train_cats = [], [], []
    new_test_seqs, new_test_labels, new_test_cats = [], [], []

    if agarwal_train_path.exists():
        with open(agarwal_train_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                new_train_seqs.append(row["sequence"])
                new_train_labels.append(float(row["K562_log2FC"]))
                new_train_cats.append(row["category"])

    if agarwal_test_path.exists():
        with open(agarwal_test_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                new_test_seqs.append(row["sequence"])
                new_test_labels.append(float(row["K562_log2FC"]))
                new_test_cats.append(row["category"])
    else:
        # For baseline comparison, still evaluate on Agarwal controls
        # Load from the real expanded path
        real_test = REPO / "data" / "k562_expanded" / "agarwal_new_test.tsv"
        if real_test.exists():
            with open(real_test) as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    new_test_seqs.append(row["sequence"])
                    new_test_labels.append(float(row["K562_log2FC"]))
                    new_test_cats.append(row["category"])

    logger.info("Agarwal expanded: train=%d, test=%d", len(new_train_seqs), len(new_test_seqs))

    # Add Agarwal train to Gosai train
    # Put a fraction into validation too (10% of new train)
    rng = np.random.default_rng(seed)
    n_new_val = max(1, int(0.1 * len(new_train_seqs)))
    perm = rng.permutation(len(new_train_seqs))
    new_val_idx = perm[:n_new_val]
    new_train_idx = perm[n_new_val:]

    for i in new_train_idx:
        train_seqs.append(new_train_seqs[i])
        train_labels.append(new_train_labels[i])
    for i in new_val_idx:
        val_seqs.append(new_train_seqs[i])
        val_labels.append(new_train_labels[i])

    logger.info(
        "Expanded: train=%d (+%d new), val=%d (+%d new)",
        len(train_seqs),
        len(new_train_idx),
        len(val_seqs),
        len(new_val_idx),
    )

    return {
        "train": (train_seqs, train_labels),
        "val": (val_seqs, val_labels),
        "test": (test_seqs, test_labels),
        "agarwal_test": (new_test_seqs, new_test_labels, new_test_cats),
    }


def train_model(
    train_seqs,
    train_labels,
    val_seqs,
    val_labels,
    seed,
    lr=0.001,
    batch_size=512,
    epochs=80,
    patience=10,
    device="cuda",
):
    """Train LegNet and return best model."""
    from models.legnet import LegNet

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = LegNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    criterion = nn.MSELoss()

    train_ds = ExpandedK562Dataset(train_seqs, train_labels)
    val_ds = ExpandedK562Dataset(val_seqs, val_labels)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    best_val_r = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).squeeze()
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Validate
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                pred = model(x).squeeze()
                # RC average
                x_rc = x.flip(-1)[:, [3, 2, 1, 0], :]
                pred_rc = model(x_rc).squeeze()
                avg = (pred + pred_rc) / 2.0
                preds.append(avg.cpu().numpy())
                trues.append(y.numpy())

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        val_r = float(pearsonr(preds, trues)[0])
        train_loss = np.mean(losses)

        if epoch % 5 == 0 or val_r > best_val_r:
            logger.info(
                "  Epoch %d: loss=%.4f val_r=%.4f%s",
                epoch,
                train_loss,
                val_r,
                " *" if val_r > best_val_r else "",
            )

        if val_r > best_val_r:
            best_val_r = val_r
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("  Early stopping at epoch %d (best val_r=%.4f)", epoch, best_val_r)
                break

    model.load_state_dict(best_state)
    return model, best_val_r


def evaluate_on_sequences(model, sequences, device="cuda", batch_size=256):
    """Predict activity for a list of sequences using RC averaging."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            encoded = np.stack([one_hot_encode(s) for s in batch])
            x = torch.from_numpy(encoded).float().to(device)
            pred = model(x).squeeze()
            x_rc = x.flip(-1)[:, [3, 2, 1, 0], :]
            pred_rc = model(x_rc).squeeze()
            avg = (pred + pred_rc) / 2.0
            preds.append(avg.cpu().numpy())
    return np.concatenate(preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/legnet_expanded/seed_42"))
    parser.add_argument("--data-path", type=Path, default=REPO / "data" / "k562")
    parser.add_argument("--expanded-path", type=Path, default=REPO / "data" / "k562_expanded")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load data
    data = load_expanded_data(args.data_path, args.expanded_path, args.seed)
    train_seqs, train_labels = data["train"]
    val_seqs, val_labels = data["val"]
    test_seqs, test_labels = data["test"]
    ag_test_seqs, ag_test_labels, ag_test_cats = data["agarwal_test"]

    # Train
    logger.info("Training LegNet (expanded)...")
    model, best_val_r = train_model(
        train_seqs,
        train_labels,
        val_seqs,
        val_labels,
        seed=args.seed,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        device=device,
    )

    # Save model
    torch.save(
        {"model_state_dict": model.state_dict(), "best_val_r": best_val_r},
        args.output_dir / "best_model.pt",
    )

    # ═══════════════════════════════════════════════════════
    # Evaluate on all test sets
    # ═══════════════════════════════════════════════════════
    results = {"best_val_r": best_val_r, "seed": args.seed}

    # 1. Standard chr-split test
    test_preds = evaluate_on_sequences(model, test_seqs, device)
    test_r = float(pearsonr(test_preds, np.array(test_labels))[0])
    results["chr_split_test_r"] = test_r
    logger.info("Chr-split test r: %.4f", test_r)

    # 2. Agarwal held-out controls
    ag_preds = evaluate_on_sequences(model, ag_test_seqs, device)
    ag_labels_arr = np.array(ag_test_labels)

    for cat in sorted(set(ag_test_cats)):
        mask = [c == cat for c in ag_test_cats]
        cat_preds = ag_preds[mask]
        cat_labels = ag_labels_arr[mask]
        cat_key = cat.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
        results[f"agarwal_{cat_key}_mean_pred"] = float(np.mean(cat_preds))
        results[f"agarwal_{cat_key}_mean_label"] = float(np.mean(cat_labels))
        results[f"agarwal_{cat_key}_n"] = (
            int(mask.count(True)) if isinstance(mask, list) else int(sum(mask))
        )
        logger.info(
            "  Agarwal %s: pred=%.3f, label=%.3f (n=%d)",
            cat,
            np.mean(cat_preds),
            np.mean(cat_labels),
            sum(mask),
        )

    # 3. Random DNA
    rng = np.random.default_rng(42)
    random_seqs = ["".join(rng.choice(list("ACGT"), size=200)) for _ in range(1000)]
    random_preds = evaluate_on_sequences(model, random_seqs, device)
    results["random_dna_mean_pred"] = float(np.mean(random_preds))
    results["random_dna_std_pred"] = float(np.std(random_preds))
    results["random_dna_pct_positive"] = float(np.mean(random_preds > 0) * 100)
    logger.info(
        "Random DNA: mean=%.3f, std=%.3f, %%>0=%.1f%%",
        np.mean(random_preds),
        np.std(random_preds),
        100 * np.mean(random_preds > 0),
    )

    # Save results
    with open(args.output_dir / "result.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved: %s", args.output_dir / "result.json")

    # Print summary
    print("\n" + "=" * 60)
    print("EXPANDED LEGNET TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Best val r:        {best_val_r:.4f}")
    print(f"  Chr-split test r:  {test_r:.4f}")
    print(f"  Random DNA mean:   {np.mean(random_preds):.3f}")
    print(f"  Random DNA %>0:    {100 * np.mean(random_preds > 0):.1f}%")


if __name__ == "__main__":
    main()
