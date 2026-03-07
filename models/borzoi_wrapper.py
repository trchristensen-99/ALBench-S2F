"""Borzoi embedding extraction wrapper for K562 MPRA sequences.

Uses borzoi-pytorch with pretrained weights from johahi/borzoi-replicate-0.

Input: 200bp (or 600bp with flanks) DNA sequences
Output: Mean-pooled trunk embeddings of shape (batch, 1536)

The 200/600bp sequences are zero-padded to 196,608bp (channels-first) and centered.
Embeddings from get_embs_after_crop() are mean-pooled over the sequence dimension.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from borzoi_pytorch import Borzoi

BORZOI_SEQ_LEN = 196_608  # Minimum input: 6144 bins * 32 downsampling factor
BORZOI_EMBED_DIM = 1536
BORZOI_NUM_BINS = 6144


class BorzoiWrapper(nn.Module):
    """Frozen Borzoi encoder for embedding extraction.

    Pads short sequences to 196,608bp (centered, channels-first), runs through
    the encoder trunk, and mean-pools the output embeddings.
    """

    def __init__(
        self,
        model_name: str = "johahi/borzoi-replicate-0",
    ):
        super().__init__()
        self.model = Borzoi.from_pretrained(model_name)
        self.model.eval()
        self.embed_dim = BORZOI_EMBED_DIM

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
            (batch, 1536) mean-pooled embeddings.
        """
        B, C, L = one_hot_seqs.shape
        assert C == 4, f"Expected 4 channels, got {C}"

        # Borzoi expects (batch, 4, L) — channels first (already correct)
        # Pad to 196,608bp centered
        pad_total = BORZOI_SEQ_LEN - L
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        # Pad sequence dimension (dim=2) with zeros
        padded = torch.nn.functional.pad(one_hot_seqs, (pad_left, pad_right), value=0.0)
        # padded: (B, 4, 196608)

        # Forward pass through encoder trunk
        emb = self.model.get_embs_after_crop(padded)
        # emb: (B, 1536, 6144)

        # Mean-pool over sequence/bin dimension
        return emb.mean(dim=2)  # (B, 1536)

    def extract_embeddings_from_strings(
        self, sequences: list[str], device: torch.device | None = None
    ) -> torch.Tensor:
        """Extract embeddings from raw DNA strings.

        Args:
            sequences: List of DNA strings (A/C/G/T/N).
            device: Device to run on.

        Returns:
            (len(sequences), 1536) embeddings.
        """
        from data.utils import one_hot_encode

        tensors = []
        for seq in sequences:
            oh = one_hot_encode(seq, add_singleton_channel=False)  # (4, L)
            tensors.append(torch.from_numpy(oh).float())

        max_len = max(t.shape[1] for t in tensors)
        batch = torch.zeros(len(tensors), 4, max_len)
        for i, t in enumerate(tensors):
            batch[i, :, : t.shape[1]] = t

        if device is not None:
            batch = batch.to(device)
            self.model.to(device)

        return self.extract_embeddings(batch)


def get_borzoi_embed_dim() -> int:
    """Return the embedding dimension for Borzoi trunk output."""
    return BORZOI_EMBED_DIM
