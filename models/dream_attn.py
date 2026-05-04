"""DREAM-ATTN model for sequence-to-function prediction.

Standalone port of the UnlockDNA team's Conformer-style architecture from
the DREAM Challenge 2022 Prix-Fixe framework.  The original is at
``random-promoter-dream-challenge-2022/prixfixe/unlockdna/``.

Two task modes (matching the DREAM paper / pilot study split):
  - ``"yeast"``: 6-channel input (4 nt + RC orientation + singleton flag),
    150 bp sequence, 18-bin softmax head with bin-weighted-average output,
    trained with KL-on-soft-bins loss.
  - ``"k562"``: 5-channel input (4 nt + RC orientation flag), 200 bp
    sequence, scalar regression head trained with MSE.

Per the pilot paper, dropout=0.1 is applied in the first layers block to
enable MC-dropout uncertainty.

The architecture has three blocks following the Prix-Fixe contract:

  FirstLayers : kmer-Conv (k=3, stride=2) → Linear(embedding_dim) →
                + position embedding + strand embedding
  Core        : N × ConformerSASwiGLU (FFN + Conv + MultiHeadAttn + FFN)
                with positional embedding before each pass
  Head        : Linear → softmax→bin centers (yeast) or scalar (k562)

Reference: https://github.com/de-novo/random-promoter-dream-challenge-2022
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

_NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


# ---------------------------------------------------------------------------
# Building blocks (ported from UnlockDNA add_blocks.py)
# ---------------------------------------------------------------------------


class SwiGLU(nn.Module):
    """SwiGLU activation: split last-dim in half, apply silu to gate, multiply."""

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
        self.swish = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, gate = torch.chunk(x, 2, dim=self.dim)
        return out * self.swish(gate)


class FeedForwardSwiGLU(nn.Module):
    """LayerNorm → Linear(D, mult*D) → SwiGLU → Dropout → Linear(mult*D/2, D)."""

    def __init__(self, embedding_dim: int, mult: int = 4, rate: float = 0.0, use_bias: bool = True):
        super().__init__()
        out_dim = embedding_dim * mult // 2
        self.layernorm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.linear1 = nn.Linear(embedding_dim, embedding_dim * mult, bias=use_bias)
        self.swiglu = SwiGLU(dim=2)  # split along channel dim
        self.drop = nn.Dropout(rate)
        self.linear2 = nn.Linear(out_dim, embedding_dim, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, L)
        x = x.transpose(1, 2)  # (B, L, D)
        x = self.layernorm(x)
        x = self.linear1(x)  # (B, L, mult*D)
        x = self.swiglu(x)  # (B, L, mult*D/2)
        x = self.drop(x)
        x = self.linear2(x)  # (B, L, D)
        x = self.drop(x)
        return x.transpose(1, 2)


class ConformerSASwiGLU(nn.Module):
    """Conformer block: 1/2·FFN + Conv + MultiHeadAttn + 1/2·FFN.

    All residual; LayerNorm before each subblock. Uses SwiGLU FFN.
    """

    def __init__(
        self,
        embedding_dim: int,
        ff_mult: int = 4,
        kernel_size: int = 15,
        rate: float = 0.1,
        num_heads: int = 4,
        use_bias: bool = False,
    ):
        super().__init__()
        self.ff1 = FeedForwardSwiGLU(embedding_dim, ff_mult, rate, use_bias)
        self.layernorm1 = nn.LayerNorm(embedding_dim, eps=1e-6)
        # Depthwise + pointwise conv (Conformer convention)
        self.conv = nn.Sequential(
            nn.Conv1d(
                embedding_dim,
                embedding_dim,
                kernel_size,
                groups=embedding_dim,
                padding="same",
                bias=False,
            ),
            nn.Conv1d(embedding_dim, embedding_dim, 1, bias=True),
            nn.ReLU(),
            nn.Dropout(rate),
        )
        self.layernorm2 = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=num_heads, batch_first=True
        )
        self.ff2 = FeedForwardSwiGLU(embedding_dim, ff_mult, rate, use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, L)
        x = x + 0.5 * self.ff1(x)

        # Conv subblock with LayerNorm pre-conv
        h = x.transpose(1, 2)
        h = self.layernorm1(h)
        h = h.transpose(1, 2)
        h = h + self.conv(h)
        x = x + h

        # Self-attention subblock with LayerNorm pre-attn
        h = x.transpose(1, 2)
        h = self.layernorm2(h)
        h = h + self.attn(h, h, h, need_weights=False)[0]
        x = h.transpose(1, 2)

        x = x + 0.5 * self.ff2(x)
        return x


# ---------------------------------------------------------------------------
# DREAM-ATTN model
# ---------------------------------------------------------------------------


class DREAMATTNStem(nn.Module):
    """First-layers block: kmer conv (k=3, stride=2) → Linear(D)
    + learned position + (yeast only) strand embedding."""

    def __init__(
        self,
        in_channels: int,
        embedding_dim: int = 256,
        seqsize: int = 200,
        kmer: int = 3,
        stride: int = 2,
        dropout: float = 0.1,
        use_strand_embedding: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.kmer = kmer
        self.stride = stride
        # After kmer/stride: seq length is (seqsize / stride)
        self.seqsize_out = seqsize // stride
        # Each kmer slice flattens (in_channels * kmer) → embedding_dim
        self.kmer_dense = nn.Linear(in_channels * kmer, embedding_dim)
        self.pos_embedding = nn.Embedding(self.seqsize_out, embedding_dim)
        self.use_strand_embedding = use_strand_embedding
        if use_strand_embedding:
            self.strand_embedding = nn.Embedding(2, embedding_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, L)
        B, C, L = x.shape
        # Pad so unfold gives (L) windows
        pad_left = self.kmer // 2 - 1 if self.kmer % 2 == 0 else self.kmer // 2
        pad_right = self.kmer - 1 - pad_left
        x = F.pad(x, (pad_left, pad_right))
        # Unfold: (B, C, L, kmer)
        x = x.unfold(2, self.kmer, 1)  # (B, C, L, kmer)
        # Stride along sequence dim
        x = x[:, :, :: self.stride]
        # Flatten kmer * C → in_channels*kmer
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, L', C, kmer)
        x = x.view(x.size(0), x.size(1), -1)  # (B, L', C*kmer)
        x = self.kmer_dense(x)  # (B, L', embedding_dim)
        # Add positional embedding
        L_out = x.size(1)
        pos = torch.arange(L_out, device=x.device).unsqueeze(0)  # (1, L')
        x = x + self.pos_embedding(pos)
        if self.use_strand_embedding:
            # First half = forward (0), second half = reverse (1)
            half = L_out // 2
            strand = torch.cat(
                [
                    torch.zeros(half, dtype=torch.long, device=x.device),
                    torch.ones(L_out - half, dtype=torch.long, device=x.device),
                ]
            ).unsqueeze(0)
            x = x + self.strand_embedding(strand)
        x = self.drop(x)
        return x.transpose(1, 2)  # (B, embedding_dim, L')


class DREAMATTN(nn.Module):
    """DREAM-ATTN: Conformer-style sequence-to-expression model.

    Args:
        in_channels: 6 for yeast (4 nt + RC + singleton), 5 for K562
            (4 nt + RC), or 4 for legacy / no extra channels.
        sequence_length: 150 (yeast, padded promoter) or 200 (K562).
        embedding_dim: hidden dim of the conformer (default 256).
        num_blocks: number of stacked ConformerSASwiGLU layers (default 4).
        kernel_size: conv kernel size in the conv subblock (default 15).
        num_heads: heads in MultiHeadAttention (default 4).
        first_block_dropout: dropout in stem (default 0.1, pilot-paper spec).
        core_dropout: dropout in conformer core (default 0.1).
        head_dropout: dropout in final head (default 0.1).
        task_mode: ``"yeast"`` (18-bin classification) or ``"k562"``
            (scalar regression).
    """

    def __init__(
        self,
        in_channels: int = 6,
        sequence_length: int = 150,
        embedding_dim: int = 256,
        num_blocks: int = 4,
        kernel_size: int = 15,
        num_heads: int = 4,
        first_block_dropout: float = 0.1,
        core_dropout: float = 0.1,
        head_dropout: float = 0.1,
        task_mode: str = "yeast",
        multitask: bool = False,
    ):
        super().__init__()
        if task_mode == "yeast":
            output_dim = 18
        elif multitask:
            output_dim = 3
        else:
            output_dim = 1
        self.task_mode = task_mode
        self.multitask = multitask
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length

        self.stem = DREAMATTNStem(
            in_channels=in_channels,
            embedding_dim=embedding_dim,
            seqsize=sequence_length,
            kmer=3,
            stride=2,
            dropout=first_block_dropout,
            # Strand embedding only makes sense if the input is interleaved
            # forward+reverse. Our pipelines do not interleave at the input
            # (RC handling is at the channel or output-aggregation level), so
            # leave strand embedding off; it can be enabled later for an
            # interleaved-RC training strategy.
            use_strand_embedding=False,
        )
        self.blocks = nn.ModuleList(
            [
                ConformerSASwiGLU(
                    embedding_dim=embedding_dim,
                    kernel_size=kernel_size,
                    num_heads=num_heads,
                    rate=core_dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        # Final head: per-position linear → mean-pool → scalar/bin output
        self.head_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.head_drop = nn.Dropout(head_dropout)
        self.head = nn.Linear(embedding_dim, output_dim)

        if task_mode == "yeast":
            bin_centers = torch.arange(18, dtype=torch.float32)
            self.register_buffer("bin_centers", bin_centers)

        # Default weight init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, L)
        h = self.stem(x)  # (B, D, L')
        for block in self.blocks:
            h = block(h)
        # Mean-pool along sequence
        h = h.mean(dim=2)  # (B, D)
        h = self.head_norm(h)
        h = self.head_drop(h)
        logits = self.head(h)  # (B, output_dim)

        if self.task_mode == "yeast":
            probs = F.softmax(logits, dim=1)
            return (probs * self.bin_centers).sum(dim=1)
        if self.multitask:
            return logits  # (B, 3)
        return logits.squeeze(-1)

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Raw logits before softmax (yeast) or before scalar (k562)."""
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        h = h.mean(dim=2)
        h = self.head_norm(h)
        h = self.head_drop(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Encoding helper (mirrors models/dream_cnn.one_hot_encode_batch)
# ---------------------------------------------------------------------------


def one_hot_encode_batch(
    sequences: list[str],
    seq_len: Optional[int] = None,
    extra_channels: tuple[str, ...] = (),
    is_singleton: Optional[list[bool]] = None,
) -> np.ndarray:
    """One-hot encode DNA sequences to (N, 4 + len(extra_channels), L) float32 array.

    ``extra_channels`` may include ``"rc"`` (orientation, 0=fwd) and
    ``"singleton"`` (per-sample flag from ``is_singleton``). Caller is
    responsible for flipping ACGT channels and setting ``rc=1`` for an
    explicit RC view (matches DREAM-CNN convention).
    """
    if seq_len is None:
        seq_len = max(len(s) for s in sequences)
    n_extra = len(extra_channels)
    out = np.zeros((len(sequences), 4 + n_extra, seq_len), dtype=np.float32)
    for i, seq in enumerate(sequences):
        for j, nuc in enumerate(seq[:seq_len]):
            idx = _NUC_TO_IDX.get(nuc.upper())
            if idx is not None:
                out[i, idx, j] = 1.0
    for k, ch in enumerate(extra_channels):
        chan = 4 + k
        if ch == "rc":
            pass
        elif ch == "singleton":
            if is_singleton is None:
                continue
            for i, flag in enumerate(is_singleton):
                if flag:
                    out[i, chan, :] = 1.0
        else:
            raise ValueError(f"Unknown extra channel '{ch}'")
    return out


def create_dream_attn(
    task_mode: str,
    in_channels: Optional[int] = None,
    sequence_length: Optional[int] = None,
    **kwargs,
) -> DREAMATTN:
    """Factory mirroring ``create_dream_rnn`` / ``create_dream_cnn``.

    Defaults match the DREAM Challenge / pilot-paper spec:
      - yeast: in_channels=6, sequence_length=150
      - k562:  in_channels=5, sequence_length=200
    """
    if in_channels is None:
        in_channels = 6 if task_mode == "yeast" else 5
    if sequence_length is None:
        sequence_length = 150 if task_mode == "yeast" else 200
    return DREAMATTN(
        in_channels=in_channels,
        sequence_length=sequence_length,
        task_mode=task_mode,
        **kwargs,
    )
