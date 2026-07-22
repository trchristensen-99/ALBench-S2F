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
# Alternative block classes (pluggable into LegNet via block_class kwarg)
# ---------------------------------------------------------------------------


class PlainConvBlock(nn.Module):
    """Vanilla 2-layer conv block (Conv → BN → SiLU)², preserves channels.

    Simplest baseline alternative to EffBlock — no inverted residual or SE.
    Drop-in replacement: matches EffBlock(in_ch, ks=...) signature; output is
    same shape as input.
    """

    def __init__(self, in_ch: int, ks: int = 5, **_unused):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, ks, padding="same", bias=False),
            nn.BatchNorm1d(in_ch),
            nn.SiLU(),
            nn.Conv1d(in_ch, in_ch, ks, padding="same", bias=False),
            nn.BatchNorm1d(in_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class QuickGELU(nn.Module):
    """sigmoid(1.702·x) · x — the GELU variant used by AlphaGenome."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(1.702 * x) * x


class RMSBatchNorm1d(nn.Module):
    """RMS batch norm: x * scale / sqrt(var + eps) + offset (no mean centering).

    Faithful port of `alphagenome_research.model.layers.RMSBatchNorm`: variance
    across (batch, seq) per channel, learned scale + offset, running EMA for eval.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.scale = nn.Parameter(torch.ones(1, num_features, 1))
        self.offset = nn.Parameter(torch.zeros(1, num_features, 1))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            var = x.pow(2).mean(dim=(0, 2))  # (C,)
            with torch.no_grad():
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * var.detach())
        else:
            var = self.running_var
        inv = self.scale / torch.sqrt(var.view(1, -1, 1) + self.eps)
        return x * inv + self.offset


class StandardizedConv1d(nn.Module):
    """Conv1d with weight standardization (AlphaGenome's `StandardizedConv1D`).

    Per output channel: zero-mean its kernel, divide by `sqrt(fan_in · var_w)` with
    a learned per-channel gain. Zero-init kernel matches AG's haiku init.
    """

    def __init__(self, in_ch: int, out_ch: int, ks: int):
        super().__init__()
        self.fan_in = ks * in_ch
        self.pad = (ks - 1) // 2
        self.weight = nn.Parameter(torch.zeros(out_ch, in_ch, ks))
        self.bias = nn.Parameter(torch.zeros(out_ch))
        self.gain = nn.Parameter(torch.ones(out_ch, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight - self.weight.mean(dim=(1, 2), keepdim=True)
        var = w.var(dim=(1, 2), keepdim=True, unbiased=False)
        scale = self.gain * torch.rsqrt(torch.clamp(self.fan_in * var, min=1e-4))
        return F.conv1d(x, w * scale, self.bias, padding=self.pad)


class AGBlock(nn.Module):
    """AlphaGenome-style stage: two pre-norm ConvBlocks with internal residuals.

    Each ConvBlock = RMSBatchNorm → QuickGELU → StandardizedConv1D, matching
    `alphagenome_research.model.convolutions.ConvBlock`. Internal residuals
    around each ConvBlock are essential because StandardizedConv1D weights
    init to zero (matching AG's haiku init) — without them, the second
    ConvBlock receives all-zero input and gradients never flow.

    Channels stay constant inside the block; channel growth happens outside
    in the parent LegNet stage.
    """

    def __init__(self, in_ch: int, ks: int = 5, **_unused):
        super().__init__()
        self.norm1 = RMSBatchNorm1d(in_ch)
        self.conv1 = StandardizedConv1d(in_ch, in_ch, ks)
        self.norm2 = RMSBatchNorm1d(in_ch)
        self.conv2 = StandardizedConv1d(in_ch, in_ch, ks)
        self.act = QuickGELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x))) + x
        return self.conv2(self.act(self.norm2(h))) + h


# Block class registry — used by hp_space.py + LegNet's block_class kwarg
BLOCK_CLASSES = {
    "eff": EffBlock,
    "plain": PlainConvBlock,
    "ag": AGBlock,
}


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
        multitask: bool = False,
        dropout: float = 0.0,
        conv_dropout: float | None = None,
        dense_dims: list[int] | None = None,
        dense_dropout: float = 0.0,
        block_class: str | Type[nn.Module] = "eff",
        pool_sizes: list | None = None,  # per-stage MaxPool1d factor (None=none; canonical MPRA-LegNet uses [2,2,2,2])
    ):
        super().__init__()
        if block_sizes is None:
            block_sizes = [256, 256, 128, 128, 64, 64, 32, 32]

        # Resolve string → class. Default "eff" keeps backward compat.
        block_cls: Type[nn.Module] = (
            BLOCK_CLASSES[block_class] if isinstance(block_class, str) else block_class
        )
        self.block_class = block_class

        self.block_sizes = block_sizes
        self.task_mode = task_mode
        self.multitask = multitask
        # Backward compat: `dropout` is the legacy single dropout. New code can pass
        # `conv_dropout` explicitly; falls back to `dropout` if not set. Peter's ask:
        # conv layers need less dropout than dense layers — so split the two.
        self.conv_dropout = conv_dropout if conv_dropout is not None else dropout
        self.dropout = self.conv_dropout  # alias for legacy callers
        self.dense_dims = list(dense_dims) if dense_dims else []
        self.dense_dropout = dense_dropout
        if task_mode == "yeast":
            self.final_ch = 18
        elif multitask:
            self.final_ch = 3
        else:
            self.final_ch = 1

        # Stem
        self.stem_block = LocalBlock(
            in_ch=in_channels,
            out_ch=block_sizes[0],
            ks=ks,
            activation=activation,
        )

        # Main body: ResidualConcat(block) + LocalBlock + optional Dropout per stage.
        # block is one of {EffBlock, PlainConvBlock, AGBlock}, all preserving in_ch.
        def _build_block(in_ch: int) -> nn.Module:
            if block_cls is EffBlock:
                return EffBlock(
                    in_ch=in_ch,
                    ks=ks,
                    resize_factor=resize_factor,
                    activation=activation,
                    filter_per_group=filter_per_group,
                    inner_dim_calculation=inner_dim_calculation,
                )
            # PlainConvBlock + AGBlock take only (in_ch, ks)
            return block_cls(in_ch=in_ch, ks=ks)

        self.pool_sizes = pool_sizes
        blocks = []
        for _si, (prev_sz, sz) in enumerate(zip(block_sizes[:-1], block_sizes[1:])):
            layers: list[nn.Module] = [
                ResidualConcat(_build_block(prev_sz)),
                LocalBlock(
                    in_ch=2 * prev_sz,  # doubled by ResidualConcat
                    out_ch=sz,
                    ks=ks,
                    activation=activation,
                ),
            ]
            if self.conv_dropout > 0:
                layers.append(nn.Dropout1d(self.conv_dropout))
            if pool_sizes is not None and _si < len(pool_sizes) and int(pool_sizes[_si]) > 1:
                layers.append(nn.MaxPool1d(int(pool_sizes[_si])))
            blocks.append(nn.Sequential(*layers))
        self.main = nn.Sequential(*blocks)

        # Output head. Two modes:
        #   (a) dense_dims is empty: original LegNet head — 1x1 conv mapper then GAP.
        #   (b) dense_dims non-empty: GAP → flatten → [Linear → activation → Dropout]*
        #       → final Linear(... → final_ch). Allows higher dropout for dense layers
        #       per Peter's "conv layers need less dropout than dense layers" guidance.
        if not self.dense_dims:
            self.mapper = MappingBlock(
                in_ch=block_sizes[-1],
                out_ch=self.final_ch,
                activation=final_activation,
            )
            self.dense_head = None
        else:
            self.mapper = None
            dense_layers: list[nn.Module] = []
            prev = block_sizes[-1]
            for d in self.dense_dims:
                dense_layers.append(nn.Linear(prev, d))
                dense_layers.append(activation())
                if self.dense_dropout > 0:
                    dense_layers.append(nn.Dropout(self.dense_dropout))
                prev = d
            dense_layers.append(nn.Linear(prev, self.final_ch))
            self.dense_head = nn.Sequential(*dense_layers)

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
        if self.dense_head is None:
            x = self.mapper(x)
            x = F.adaptive_avg_pool1d(x, 1).squeeze(2)
        else:
            x = F.adaptive_avg_pool1d(x, 1).squeeze(2)
            x = self.dense_head(x)

        if self.task_mode == "yeast":
            probs = F.softmax(x, dim=1)
            return (probs * self.bin_centers).sum(dim=1)
        if self.multitask:
            return x  # (B, 3) for multi-cell-type regression
        return x.squeeze(-1)

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get raw logits before softmax (yeast) or identity (k562)."""
        h = self.stem_block(x)
        h = self.main(h)
        if self.dense_head is None:
            h = self.mapper(h)
            h = F.adaptive_avg_pool1d(h, 1).squeeze(2)
        else:
            h = F.adaptive_avg_pool1d(h, 1).squeeze(2)
            h = self.dense_head(h)
        return h

    # --- Checkpoint helpers (matches SequenceModel.save/load_checkpoint) ---
    def save_checkpoint(self, path: str, **kwargs) -> None:
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "model_info": {
                    "model_type": "LegNet",
                    "block_sizes": self.block_sizes,
                    "task_mode": self.task_mode,
                    "multitask": self.multitask,
                    "dropout": self.dropout,
                    "conv_dropout": self.conv_dropout,
                    "dense_dims": self.dense_dims,
                    "dense_dropout": self.dense_dropout,
                    "block_class": self.block_class,
                },
                **kwargs,
            },
            path,
        )

    def load_checkpoint(self, path: str, strict: bool = True) -> dict:
        ckpt = torch.load(path, map_location="cpu")
        self.load_state_dict(ckpt["model_state_dict"], strict=strict)
        return ckpt


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
