"""DREAM-RNN student wrapper implementing the SequenceModel interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from albench.data.utils import one_hot_encode
from albench.model import SequenceModel
from albench.models.dream_rnn import DREAMRNN, create_dream_rnn
from albench.models.loss_utils import YeastKLLoss
from albench.models.training import train_model_optimized
from albench.models.training_base import create_optimizer_and_scheduler


@dataclass
class TrainConfig:
    """Training hyperparameters for one ensemble member."""

    batch_size: int = 1024
    epochs: int = 80
    lr: float = 0.005
    lr_lstm: float = 0.005
    weight_decay: float = 0.01
    pct_start: float = 0.3


class _InMemorySequenceDataset(Dataset):
    """Tensor-backed dataset used for wrapper-level fit calls."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class DREAMRNNStudent(SequenceModel):
    """SequenceModel wrapper around an ensemble of DREAMRNN models."""

    def __init__(
        self,
        input_channels: int,
        sequence_length: int,
        task_mode: str = "k562",
        ensemble_size: int = 3,
        device: str | None = None,
        train_config: TrainConfig | None = None,
    ) -> None:
        """Initialize the student and its ensemble.

        Args:
            input_channels: Number of model input channels.
            sequence_length: Sequence length expected by the model.
            task_mode: Task mode (``k562`` or ``yeast``).
            ensemble_size: Number of ensemble members.
            device: Torch device string; auto-detected when omitted.
            train_config: Training configuration used in ``fit``.
        """
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.task_mode = task_mode
        self.ensemble_size = ensemble_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.train_config = train_config or TrainConfig()
        self.models = [
            create_dream_rnn(
                input_channels=input_channels,
                sequence_length=sequence_length,
                task_mode=task_mode,
            ).to(self.device)
            for _ in range(ensemble_size)
        ]

    def _encode_sequences(
        self, sequences: Sequence[str], singleton_flag: float = 0.0
    ) -> torch.Tensor:
        """Encode sequence strings into model input tensors."""
        encoded: list[np.ndarray] = []
        for seq in sequences:
            base = one_hot_encode(seq, add_singleton_channel=False)
            rc = np.zeros((1, len(seq)), dtype=np.float32)
            if self.input_channels == 6:
                singleton = np.full((1, len(seq)), singleton_flag, dtype=np.float32)
                arr = np.concatenate([base, rc, singleton], axis=0)
            elif self.input_channels == 5:
                arr = np.concatenate([base, rc], axis=0)
            else:
                arr = base
            encoded.append(arr)
        array = np.stack(encoded, axis=0)
        return torch.from_numpy(array).float()

    def _predict_member(self, model: DREAMRNN, x: torch.Tensor) -> np.ndarray:
        """Run one model in eval mode and return numpy predictions."""
        model.eval()
        with torch.no_grad():
            out = model(x.to(self.device)).detach().cpu().numpy()
        return out.reshape(-1)

    def _embed_member(self, model: DREAMRNN, x: torch.Tensor) -> np.ndarray:
        """Extract 256-d pooled embedding before the final FC layer."""
        model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            conv1_short_out = model.conv1_short(x)
            conv1_long_out = model.conv1_long(x)
            conv1_out = torch.cat([conv1_short_out, conv1_long_out], dim=1)
            conv1_out = model.relu1(conv1_out)
            conv1_out = model.dropout1(conv1_out)
            lstm_in = conv1_out.permute(0, 2, 1)
            lstm_out, _ = model.lstm(lstm_in)
            lstm_out = lstm_out.permute(0, 2, 1)
            conv2_short_out = model.conv2_short(lstm_out)
            conv2_long_out = model.conv2_long(lstm_out)
            conv2_out = torch.cat([conv2_short_out, conv2_long_out], dim=1)
            conv2_out = model.relu2(conv2_out)
            conv2_out = model.dropout2(conv2_out)
            conv3_out = model.conv3(conv2_out)
            conv3_out = model.relu3(conv3_out)
            pooled = torch.mean(conv3_out, dim=2)
        return pooled.detach().cpu().numpy()

    def predict(self, sequences: list[str]) -> np.ndarray:
        """Predict activity as the mean across ensemble members."""
        x = self._encode_sequences(sequences)
        preds = [self._predict_member(model, x) for model in self.models]
        return np.mean(np.stack(preds, axis=0), axis=0)

    def uncertainty(self, sequences: list[str]) -> np.ndarray:
        """Estimate uncertainty via MC dropout variance over 30 passes."""
        x = self._encode_sequences(sequences).to(self.device)
        all_passes: list[np.ndarray] = []
        for model in self.models:
            member_passes: list[np.ndarray] = []
            model.train()
            for _ in range(30):
                with torch.no_grad():
                    member_passes.append(model(x).detach().cpu().numpy().reshape(-1))
            all_passes.append(np.var(np.stack(member_passes, axis=0), axis=0))
        return np.mean(np.stack(all_passes, axis=0), axis=0)

    def embed(self, sequences: list[str]) -> np.ndarray:
        """Return 256-d embeddings as ensemble mean pooled conv3 features."""
        x = self._encode_sequences(sequences)
        embeds = [self._embed_member(model, x) for model in self.models]
        return np.mean(np.stack(embeds, axis=0), axis=0)

    def fit(self, sequences: list[str], labels: np.ndarray) -> None:
        """Retrain all ensemble members with optimized training utilities."""
        x = self._encode_sequences(sequences)
        y = torch.from_numpy(labels.astype(np.float32))

        dataset = _InMemorySequenceDataset(x, y)
        loader = DataLoader(dataset, batch_size=self.train_config.batch_size, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=self.train_config.batch_size, shuffle=False)

        for model in self.models:
            optimizer, scheduler = create_optimizer_and_scheduler(
                model=model,
                train_loader=loader,
                num_epochs=self.train_config.epochs,
                lr=self.train_config.lr,
                lr_lstm=self.train_config.lr_lstm,
                weight_decay=self.train_config.weight_decay,
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
                use_reverse_complement=True,
                early_stopping_patience=None,
                metric_for_best="pearson_r",
                use_amp=True,
                use_compile=False,
            )
