"""Nucleotide Transformer v3 650M embedding extraction wrapper for K562 MPRA sequences.

Supports both pre-trained (NTv3_650M_pre) and post-trained (NTv3_650M_post) variants.
The post-trained model requires species conditioning via ``species_tokens`` in the
forward pass.

Input: DNA sequence strings (200bp or 600bp, N-padded to 640bp for U-Net divisibility)
Output: Mean-pooled embeddings of shape (batch, 1536)

Sequences are tokenized at single-base resolution, passed through the U-Net +
transformer architecture, and the final-layer embeddings are mean-pooled
(excluding padding tokens).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from nucleotide_transformer_v3.pretrained import (
    get_posttrained_ntv3_model,
    get_pretrained_ntv3_model,
)

NTV3_650M_EMBED_DIM = 1536
# U-Net uses 7 downsamples → sequences must be divisible by 2^7 = 128
NTV3_SEQ_DIVISOR = 128


def _pad_to_divisible(seq: str, divisor: int = NTV3_SEQ_DIVISOR) -> str:
    """Pad sequence with N to the next multiple of divisor."""
    remainder = len(seq) % divisor
    if remainder == 0:
        return seq
    pad_total = divisor - remainder
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return "N" * pad_left + seq + "N" * pad_right


class NTv3Wrapper:
    """Frozen Nucleotide Transformer v3 650M for embedding extraction.

    Uses JAX/Flax NNX (not nn.Module). All parameters are frozen by default
    since we only call the model without computing gradients.

    Args:
        model_name: Model identifier (``NTv3_650M_pre`` or ``NTv3_650M_post``).
        model_variant: ``"pre"`` for pre-trained, ``"post"`` for post-trained.
        species: Species name for post-trained conditioning (default ``"human"``).
        use_bfloat16: Use bfloat16 for inference.
    """

    def __init__(
        self,
        model_name: str = "NTv3_650M_pre",
        model_variant: str = "pre",
        species: str = "human",
        use_bfloat16: bool = True,
    ):
        self.model_name = model_name
        self.model_variant = model_variant
        self.species_token = None

        if model_variant == "post":
            model, tokenizer, config = get_posttrained_ntv3_model(
                model_name=model_name,
                use_bfloat16=use_bfloat16,
            )
            # Pre-compute species token once (shape (1,))
            self.species_token = model.encode_species(species)
        else:
            model, tokenizer, config = get_pretrained_ntv3_model(
                model_name=model_name,
                use_bfloat16=use_bfloat16,
            )

        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.embed_dim = config.embed_dim

    def extract_embeddings(self, sequences: list[str]) -> np.ndarray:
        """Extract mean-pooled embeddings from DNA sequences.

        Args:
            sequences: List of DNA strings (A/C/G/T/N). Sequences are
                N-padded to the next multiple of 128 for U-Net compatibility.

        Returns:
            (len(sequences), embed_dim) float32 numpy array of mean-pooled embeddings.
        """
        padded_seqs = [_pad_to_divisible(s) for s in sequences]
        tokens = self.tokenizer.batch_np_tokenize(padded_seqs)
        tokens_jax = jnp.asarray(tokens)

        if self.model_variant == "post":
            species_batch = jnp.tile(self.species_token, (tokens_jax.shape[0],))
            outs = self.model(tokens_jax, species_tokens=species_batch)
        else:
            outs = self.model(tokens_jax)
        embeddings = outs["embedding"]  # (B, T, D)

        # Create padding mask (exclude pad tokens from mean)
        pad_mask = jnp.expand_dims(tokens_jax != self.tokenizer.pad_token_id, axis=-1)  # (B, T, 1)

        masked_emb = embeddings * pad_mask
        seq_lens = jnp.sum(pad_mask, axis=1)  # (B, 1)
        mean_emb = jnp.sum(masked_emb, axis=1) / jnp.maximum(seq_lens, 1)  # (B, D)

        return np.asarray(mean_emb, dtype=np.float32)

    def extract_embeddings_batched(self, sequences: list[str], batch_size: int = 32) -> np.ndarray:
        """Extract embeddings in batches to avoid OOM.

        Args:
            sequences: List of DNA strings.
            batch_size: Number of sequences per batch.

        Returns:
            (len(sequences), embed_dim) float32 numpy array.
        """
        all_embs = []
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i : i + batch_size]
            embs = self.extract_embeddings(batch_seqs)
            all_embs.append(embs)
        return np.concatenate(all_embs, axis=0)


def get_ntv3_embed_dim() -> int:
    """Return the embedding dimension for NTv3 650M."""
    return NTV3_650M_EMBED_DIM
