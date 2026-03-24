"""DREAM-CNN model for sequence-to-function prediction.

Architecture from the DREAM Challenge 2022 benchmark (PrixFixe framework):
- BHI dual-kernel stem (Conv1d k=9 + k=15, concatenated)
- Autosome core: inverted residual blocks with squeeze-and-excitation
- Final: Conv1d → AdaptiveAvgPool → Linear

This is essentially the LegNet core architecture with BHI stem.
~1.9M params, much faster than DREAM-RNN (no LSTM).

Reference: https://github.com/de-novo/random-promoter-dream-challenge-2022
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ConvBlock(nn.Module):
    """Conv1d → ReLU → MaxPool → Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding="same")
        self.mp = nn.MaxPool1d(pool_size, stride=pool_size)
        self.do = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.do(self.mp(F.relu(self.conv(x))))


class SELayerSimple(nn.Module):
    """Simple squeeze-and-excitation layer."""

    def __init__(self, inp: int, oup: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(oup, inp // reduction),
            nn.SiLU(),
            nn.Linear(inp // reduction, oup),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc(y).view(b, c, 1)
        return x * y


# ---------------------------------------------------------------------------
# DREAM-CNN architecture
# ---------------------------------------------------------------------------


class DREAMCNNStem(nn.Module):
    """BHI dual-kernel stem: two parallel ConvBlocks with different kernel sizes."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 320,
        kernel_sizes: tuple[int, ...] = (9, 15),
        pool_size: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        assert out_channels % len(kernel_sizes) == 0
        each = out_channels // len(kernel_sizes)
        self.convs = nn.ModuleList(
            [ConvBlock(in_channels, each, k, pool_size, dropout) for k in kernel_sizes]
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([conv(x) for conv in self.convs], dim=1)


class AutosomeCoreBlock(nn.Module):
    """Inverted residual blocks with SE attention (LegNet-style)."""

    def __init__(
        self,
        in_channels: int = 320,
        out_channels: int = 64,
        block_sizes: tuple[int, ...] = (128, 128, 64, 64, 64),
        kernel_size: int = 7,
        resize_factor: int = 4,
        se_reduction: int = 4,
        filter_per_group: int = 2,
        bn_momentum: float = 0.1,
    ):
        super().__init__()
        activation = nn.SiLU
        sizes = [in_channels, *block_sizes, out_channels]
        self.n_blocks = len(sizes) - 1

        self.inv_res = nn.ModuleList()
        self.resize = nn.ModuleList()

        for prev_sz, sz in zip(sizes[:-1], sizes[1:]):
            # Inverted residual block
            self.inv_res.append(
                nn.Sequential(
                    nn.Conv1d(prev_sz, sz * resize_factor, 1, padding="same", bias=False),
                    nn.BatchNorm1d(sz * resize_factor, momentum=bn_momentum),
                    activation(),
                    nn.Conv1d(
                        sz * resize_factor,
                        sz * resize_factor,
                        kernel_size,
                        groups=sz * resize_factor // filter_per_group,
                        padding="same",
                        bias=False,
                    ),
                    nn.BatchNorm1d(sz * resize_factor, momentum=bn_momentum),
                    activation(),
                    SELayerSimple(prev_sz, sz * resize_factor, reduction=se_reduction),
                    nn.Conv1d(sz * resize_factor, prev_sz, 1, padding="same", bias=False),
                    nn.BatchNorm1d(prev_sz, momentum=bn_momentum),
                    activation(),
                )
            )
            # Resize block (merges residual + skip)
            self.resize.append(
                nn.Sequential(
                    nn.Conv1d(2 * prev_sz, sz, kernel_size, padding="same", bias=False),
                    nn.BatchNorm1d(sz, momentum=bn_momentum),
                    activation(),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for inv, res in zip(self.inv_res, self.resize):
            x = torch.cat([x, inv(x)], dim=1)
            x = res(x)
        return x


class DREAMCNNHead(nn.Module):
    """Final layers: Conv1d → AdaptiveAvgPool → Linear."""

    def __init__(self, in_channels: int = 64, hidden: int = 256, output_dim: int = 1):
        super().__init__()
        self.mapper = nn.Conv1d(in_channels, hidden, 1, padding="same")
        self.linear = nn.Linear(hidden, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mapper(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(2)
        return self.linear(x)


class DREAMCNN(nn.Module):
    """Full DREAM-CNN model.

    Parameters
    ----------
    in_channels : int
        Number of input channels (4 for one-hot DNA, 5 with RC flag).
    stem_channels : int
        Output channels from dual-kernel stem.
    core_out_channels : int
        Output channels from core block.
    head_hidden : int
        Hidden size in final head.
    dropout : float
        Dropout in stem.
    """

    def __init__(
        self,
        in_channels: int = 4,
        stem_channels: int = 320,
        core_out_channels: int = 64,
        head_hidden: int = 256,
        dropout: float = 0.2,
        task_mode: str = "k562",
    ):
        super().__init__()
        self.task_mode = task_mode
        output_dim = 18 if task_mode == "yeast" else 1
        self.output_dim = output_dim

        self.stem = DREAMCNNStem(
            in_channels=in_channels,
            out_channels=stem_channels,
            dropout=dropout,
        )
        self.core = AutosomeCoreBlock(
            in_channels=stem_channels,
            out_channels=core_out_channels,
        )
        self.head = DREAMCNNHead(
            in_channels=core_out_channels,
            hidden=head_hidden,
            output_dim=output_dim,
        )

        if task_mode == "yeast":
            bin_centers = torch.arange(18, dtype=torch.float32)
            self.register_buffer("bin_centers", bin_centers)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2 / n))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (B, C, L) one-hot encoded."""
        x = self.stem(x)
        x = self.core(x)
        logits = self.head(x)

        if self.task_mode == "yeast":
            probs = F.softmax(logits, dim=1)
            return (probs * self.bin_centers).sum(dim=1)
        return logits.squeeze(-1)

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get raw logits (before softmax for yeast, same as forward for K562)."""
        x = self.stem(x)
        x = self.core(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# One-hot encoding utility
# ---------------------------------------------------------------------------

_NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def one_hot_encode_batch(sequences: list[str], seq_len: int | None = None) -> np.ndarray:
    """One-hot encode DNA sequences to (N, 4, L) float32 array."""
    if seq_len is None:
        seq_len = max(len(s) for s in sequences)
    out = np.zeros((len(sequences), 4, seq_len), dtype=np.float32)
    for i, seq in enumerate(sequences):
        for j, nuc in enumerate(seq[:seq_len]):
            idx = _NUC_TO_IDX.get(nuc.upper())
            if idx is not None:
                out[i, idx, j] = 1.0
    return out
