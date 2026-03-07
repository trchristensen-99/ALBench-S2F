"""Enformer embedding extraction wrapper for K562 MPRA sequences.

Uses the lucidrains enformer-pytorch package with pretrained weights from
EleutherAI/enformer-official-rough.

Input: 200bp (or 600bp with flanks) DNA sequences
Output: Mean-pooled trunk embeddings of shape (batch, 3072)

The 200/600bp sequences are zero-padded to 196,608bp and centered.
Center bins (corresponding to the sequence region) are extracted and
mean-pooled to produce a single embedding vector per sequence.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from enformer_pytorch import Enformer

ENFORMER_SEQ_LEN = 196_608
ENFORMER_EMBED_DIM = 3072  # trunk output: dim * 2 = 1536 * 2
ENFORMER_TARGET_LEN = 896  # output bins after cropping
# Each output bin covers 128bp (196608 / 7 downsamples of 2x = 1536 positions, cropped to 896)
ENFORMER_BIN_SIZE = 128


class EnformerWrapper(nn.Module):
    """Frozen Enformer encoder for embedding extraction.

    Pads short sequences to 196,608bp (centered), runs through the trunk,
    extracts center bins corresponding to the input region, and mean-pools.
    """

    def __init__(
        self,
        model_name: str = "EleutherAI/enformer-official-rough",
        n_center_bins: int = 4,
    ):
        super().__init__()
        self.model = Enformer.from_pretrained(model_name)
        self.model.eval()
        self.n_center_bins = n_center_bins
        self.embed_dim = ENFORMER_EMBED_DIM

        # Freeze all parameters
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def extract_embeddings(self, one_hot_seqs: torch.Tensor) -> torch.Tensor:
        """Extract mean-pooled embeddings from short DNA sequences.

        Args:
            one_hot_seqs: (batch, 4, seq_len) one-hot encoded sequences.
                          seq_len can be 200 or 600 (with MPRA flanks).

        Returns:
            (batch, 3072) mean-pooled trunk embeddings.
        """
        B, C, L = one_hot_seqs.shape
        assert C == 4, f"Expected 4 channels, got {C}"

        # Enformer expects (batch, seq_len, 4) — channels last
        seqs_cl = one_hot_seqs.permute(0, 2, 1)  # (B, L, 4)

        # Pad to 196,608bp centered
        pad_total = ENFORMER_SEQ_LEN - L
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        # Pad sequence dimension (dim=1) with zeros
        padded = torch.nn.functional.pad(seqs_cl, (0, 0, pad_left, pad_right), value=0.0)
        # padded: (B, 196608, 4)

        # Forward pass through trunk
        emb = self.model(padded, return_only_embeddings=True)
        # emb: (B, 896, 3072)

        # Extract center bins
        center = ENFORMER_TARGET_LEN // 2  # 448
        half = self.n_center_bins // 2
        center_emb = emb[:, center - half : center + half, :]  # (B, n_center_bins, 3072)

        # Mean-pool over bins
        return center_emb.mean(dim=1)  # (B, 3072)

    def extract_embeddings_from_strings(
        self, sequences: list[str], device: torch.device | None = None
    ) -> torch.Tensor:
        """Extract embeddings from raw DNA strings.

        Args:
            sequences: List of DNA strings (A/C/G/T/N).
            device: Device to run on.

        Returns:
            (len(sequences), 3072) embeddings.
        """
        from data.utils import one_hot_encode

        tensors = []
        for seq in sequences:
            oh = one_hot_encode(seq, add_singleton_channel=False)  # (4, L)
            tensors.append(torch.from_numpy(oh).float())

        # Stack with padding to max length
        max_len = max(t.shape[1] for t in tensors)
        batch = torch.zeros(len(tensors), 4, max_len)
        for i, t in enumerate(tensors):
            batch[i, :, : t.shape[1]] = t

        if device is not None:
            batch = batch.to(device)
            self.model.to(device)

        return self.extract_embeddings(batch)


def get_enformer_embed_dim() -> int:
    """Return the embedding dimension for Enformer trunk output."""
    return ENFORMER_EMBED_DIM
