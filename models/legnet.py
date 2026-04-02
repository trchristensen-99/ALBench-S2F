"""LegNet model for sequence-to-function prediction.

Standalone implementation of LegNet (NoGINet) from the DREAM Challenge 2022.
All components (LocalBlock, EffBlock, SELayer, Residual, ResidualConcat,
MappingBlock, LegNet) are included in this single file with no external
imports from the original dream_ablation source.

Supports two task modes:
- "yeast": 18-bin classification with KL loss (softmax output)
- "k562": 1-output regression with MSE loss (scalar output)

Reference: https://github.com/autosome-ru/LegNet
"""

from __future__ import annotations

import math
from typing import Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class SELayer(nn.Module):
    """Simple squeeze-and-excitation layer."""

    def __init__(self, inp: int, oup: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp // reduction)),
            nn.SiLU(),
            nn.Linear(int(inp // reduction), oup),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc(y).view(b, c, 1)
        return x * y


class Residual(nn.Module):
    """Additive residual wrapper."""

    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x) + x


class ResidualConcat(nn.Module):
    """Concatenation residual wrapper (doubles channel dim)."""

    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.fn(x), x], dim=1)


class LocalBlock(nn.Module):
    """Single Conv1d -> BatchNorm -> Activation block."""

    def __init__(
        self,
        in_ch: int,
        ks: int,
        activation: Type[nn.Module],
        out_ch: int | None = None,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = self.in_ch if out_ch is None else out_ch
        self.ks = ks
        self.block = nn.Sequential(
            nn.Conv1d(self.in_ch, self.out_ch, self.ks, padding="same", bias=False),
            nn.BatchNorm1d(self.out_ch),
            activation(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EffBlock(nn.Module):
    """Inverted residual block with squeeze-and-excitation."""

    def __init__(
        self,
        in_ch: int,
        ks: int,
        resize_factor: int,
        filter_per_group: int,
        activation: Type[nn.Module],
        out_ch: int | None = None,
        se_reduction: int | None = None,
        inner_dim_calculation: str = "out",
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = self.in_ch if out_ch is None else out_ch
        self.resize_factor = resize_factor
        self.se_reduction = resize_factor if se_reduction is None else se_reduction

        if inner_dim_calculation == "out":
            inner_dim = self.out_ch * self.resize_factor
        elif inner_dim_calculation == "in":
            inner_dim = self.in_ch * self.resize_factor
        else:
            raise ValueError(f"Wrong inner_dim_calculation: {inner_dim_calculation}")

        self.block = nn.Sequential(
            nn.Conv1d(self.in_ch, inner_dim, 1, padding="same", bias=False),
            nn.BatchNorm1d(inner_dim),
            activation(),
            nn.Conv1d(
                inner_dim,
                inner_dim,
                ks,
                groups=inner_dim // filter_per_group,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm1d(inner_dim),
            activation(),
            SELayer(self.in_ch, inner_dim, reduction=self.se_reduction),
            nn.Conv1d(inner_dim, self.in_ch, 1, padding="same", bias=False),
            nn.BatchNorm1d(self.in_ch),
            activation(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MappingBlock(nn.Module):
    """1x1 convolution with activation (channel mapping)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        activation: Type[nn.Module],
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, padding="same"),
            activation(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# LegNet
# ---------------------------------------------------------------------------


class LegNet(nn.Module):
    """LegNet (NoGINet) model.

    Parameters
    ----------
    in_channels : int
        Number of input channels. 4 for one-hot DNA, 5 with RC flag.
    block_sizes : list[int]
        Channel sizes for each block. Default: [256, 256, 128, 128, 64, 64, 32, 32].
    ks : int
        Kernel size. Default: 5.
    resize_factor : int
        Expansion factor in inverted residual blocks. Default: 4.
    activation : Type[nn.Module]
        Activation class. Default: nn.SiLU.
    filter_per_group : int
        Filters per group in depthwise conv. Default: 2.
    se_reduction : int
        SE reduction factor. Default: 4.
    task_mode : str
        "yeast" for 18-bin classification, "k562" for scalar regression.
    """

    def __init__(
        self,
        in_channels: int = 4,
        block_sizes: list[int] | None = None,
        ks: int = 5,
        resize_factor: int = 4,
        activation: Type[nn.Module] = nn.SiLU,
        final_activation: Type[nn.Module] = nn.SiLU,
        filter_per_group: int = 2,
        se_reduction: int = 4,
        inner_dim_calculation: str = "out",
        task_mode: str = "k562",
    ):
        super().__init__()
        if block_sizes is None:
            block_sizes = [256, 256, 128, 128, 64, 64, 32, 32]

        self.block_sizes = block_sizes
        self.task_mode = task_mode
        self.final_ch = 18 if task_mode == "yeast" else 1

        # Stem
        self.stem_block = LocalBlock(
            in_ch=in_channels,
            out_ch=block_sizes[0],
            ks=ks,
            activation=activation,
        )

        # Main body: ResidualConcat(EffBlock) + LocalBlock per stage
        blocks = []
        for prev_sz, sz in zip(block_sizes[:-1], block_sizes[1:]):
            block = nn.Sequential(
                ResidualConcat(
                    EffBlock(
                        in_ch=prev_sz,
                        out_ch=sz,
                        ks=ks,
                        resize_factor=resize_factor,
                        activation=activation,
                        filter_per_group=filter_per_group,
                        inner_dim_calculation=inner_dim_calculation,
                    )
                ),
                LocalBlock(
                    in_ch=2 * prev_sz,  # doubled by ResidualConcat
                    out_ch=sz,
                    ks=ks,
                    activation=activation,
                ),
            )
            blocks.append(block)
        self.main = nn.Sequential(*blocks)

        # Output mapping
        self.mapper = MappingBlock(
            in_ch=block_sizes[-1],
            out_ch=self.final_ch,
            activation=final_activation,
        )

        if task_mode == "yeast":
            self.register_buffer(
                "bin_centers",
                torch.arange(18, dtype=torch.float32),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (B, C, L).

        Returns
        -------
        torch.Tensor
            For "k562": scalar predictions of shape (B,).
            For "yeast": expected bin value of shape (B,).
        """
        x = self.stem_block(x)
        x = self.main(x)
        x = self.mapper(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(2)

        if self.task_mode == "yeast":
            probs = F.softmax(x, dim=1)
            return (probs * self.bin_centers).sum(dim=1)
        return x.squeeze(-1)

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get raw logits before softmax (yeast) or identity (k562)."""
        h = self.stem_block(x)
        h = self.main(h)
        h = self.mapper(h)
        h = F.adaptive_avg_pool1d(h, 1).squeeze(2)
        return h


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
