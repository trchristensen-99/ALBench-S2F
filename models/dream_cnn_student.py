"""DREAM-CNN student wrapper implementing the SequenceModel interface.

Analogous to DREAMRNNStudent but uses the faster DREAM-CNN architecture
(inverted residual blocks with SE attention instead of BiLSTM).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from albench.model import SequenceModel
from models.dream_cnn import DREAMCNN, one_hot_encode_batch
from models.loss_utils import YeastKLLoss
from models.training import train_model_optimized
from models.training_base import create_optimizer_and_scheduler


@dataclass
class TrainConfig:
    """Training hyperparameters for one ensemble member."""

    batch_size: int = 1024
    epochs: int = 80
    lr: float = 0.005
    weight_decay: float = 0.01
    pct_start: float = 0.3
    early_stopping_patience: int | None = None
    num_workers: int = 2


class _InMemorySequenceDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class DREAMCNNStudent(SequenceModel):
    """SequenceModel wrapper around an ensemble of DREAM-CNN models."""

    def __init__(
        self,
        in_channels: int = 4,
        sequence_length: int = 200,
        task_mode: str = "k562",
        ensemble_size: int = 3,
        device: str | None = None,
        train_config: TrainConfig | None = None,
        stem_channels: int = 320,
        core_out_channels: int = 64,
        head_hidden: int = 256,
        dropout: float = 0.2,
    ) -> None:
        self.in_channels = in_channels
        self.sequence_length = sequence_length
        self.task_mode = task_mode
        self.ensemble_size = ensemble_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.train_config = train_config or TrainConfig()

        self.models = [
            DREAMCNN(
                in_channels=in_channels,
                stem_channels=stem_channels,
                core_out_channels=core_out_channels,
                head_hidden=head_hidden,
                dropout=dropout,
                task_mode=task_mode,
            ).to(self.device)
            for _ in range(ensemble_size)
        ]

    def _encode_sequences(self, sequences: Sequence[str]) -> torch.Tensor:
        """Encode sequence strings to (N, 4, L) tensor."""
        target_len = self.sequence_length
        standardized: list[str] = []
        for seq in sequences:
            seq = seq.upper()
            if len(seq) < target_len:
                pad = target_len - len(seq)
                seq = "N" * (pad // 2) + seq + "N" * (pad - pad // 2)
            elif len(seq) > target_len:
                start = (len(seq) - target_len) // 2
                seq = seq[start : start + target_len]
            standardized.append(seq)
        arr = one_hot_encode_batch(standardized, seq_len=target_len)
        return torch.from_numpy(arr)

    def _predict_member(self, model: DREAMCNN, x: torch.Tensor) -> np.ndarray:
        model.eval()
        preds = []
        bs = 512
        with torch.no_grad():
            for i in range(0, len(x), bs):
                batch = x[i : i + bs].to(self.device)
                out = model(batch).detach().cpu().numpy()
                preds.append(out.reshape(-1))
        return np.concatenate(preds)

    def predict(self, sequences: list[str]) -> np.ndarray:
        """Predict as mean across ensemble members."""
        x = self._encode_sequences(sequences)
        preds = [self._predict_member(model, x) for model in self.models]
        return np.mean(np.stack(preds, axis=0), axis=0)

    def uncertainty(self, sequences: list[str]) -> np.ndarray:
        """MC dropout variance (30 passes per member)."""
        x = self._encode_sequences(sequences).to(self.device)
        all_vars: list[np.ndarray] = []
        for model in self.models:
            model.train()
            passes = []
            for _ in range(30):
                with torch.no_grad():
                    passes.append(model(x).detach().cpu().numpy().reshape(-1))
            all_vars.append(np.var(np.stack(passes, axis=0), axis=0))
        return np.mean(np.stack(all_vars, axis=0), axis=0)

    def embed(self, sequences: list[str]) -> np.ndarray:
        """Extract embeddings from the core block output (global avg pooled)."""
        x = self._encode_sequences(sequences)
        embeds = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                batch = x.to(self.device)
                h = model.stem(batch)
                h = model.core(h)
                # Global average pool → (B, core_out_channels)
                pooled = h.mean(dim=2)
                embeds.append(pooled.cpu().numpy())
        return np.mean(np.stack(embeds, axis=0), axis=0)

    def fit(
        self,
        sequences: list[str],
        labels: np.ndarray,
        val_sequences: list[str] | None = None,
        val_labels: np.ndarray | None = None,
    ) -> None:
        """Train all ensemble members.

        If val_sequences/val_labels are provided, they are used for early
        stopping and best-model selection. Otherwise a 10% random split of
        the training data is used.
        """
        x = self._encode_sequences(sequences)
        y = torch.from_numpy(labels.astype(np.float32))

        if val_sequences is not None and val_labels is not None:
            x_val = self._encode_sequences(val_sequences)
            y_val = torch.from_numpy(val_labels.astype(np.float32))
        else:
            # Internal 10% split
            n_val = max(50, int(0.1 * len(x)))
            perm = torch.randperm(len(x))
            val_idx, train_idx = perm[:n_val], perm[n_val:]
            x_val, y_val = x[val_idx], y[val_idx]
            x, y = x[train_idx], y[train_idx]

        train_dataset = _InMemorySequenceDataset(x, y)
        val_dataset = _InMemorySequenceDataset(x_val, y_val)
        nw = self.train_config.num_workers
        loader = DataLoader(
            train_dataset,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=nw,
            persistent_workers=nw > 0,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.train_config.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=nw,
            persistent_workers=nw > 0,
        )

        for model in self.models:
            # DREAM-CNN has no LSTM, so use uniform LR
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.train_config.lr,
                weight_decay=self.train_config.weight_decay,
            )
            steps_per_epoch = len(loader)
            total_steps = steps_per_epoch * self.train_config.epochs
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.train_config.lr,
                total_steps=total_steps,
                pct_start=self.train_config.pct_start,
            )
            criterion: nn.Module = YeastKLLoss() if self.task_mode == "yeast" else nn.MSELoss()
            train_model_optimized(
                model=model,
                train_loader=loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                num_epochs=self.train_config.epochs,
                device=self.device,
                scheduler=scheduler,
                checkpoint_dir=None,
                use_reverse_complement=False,  # DREAM-CNN uses 4-channel input, no RC flag
                early_stopping_patience=self.train_config.early_stopping_patience,
                metric_for_best="pearson_r",
                use_amp=True,
                use_compile=True,
            )
