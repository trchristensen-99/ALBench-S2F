"""Nucleotide Transformer v2 embedding extraction wrapper for K562 MPRA sequences.

Uses the InstaDeep nucleotide-transformer package (JAX/Haiku) with pretrained
250M_multi_species_v2 weights.

Input: DNA sequence strings (200bp or 600bp)
Output: Mean-pooled embeddings of shape (batch, 768)

Sequences are tokenized into 6-mers, passed through the transformer, and the
final layer embeddings are mean-pooled (excluding CLS token and padding).
"""

from __future__ import annotations

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from nucleotide_transformer.pretrained import get_pretrained_model

NT_V2_250M_EMBED_DIM = 768
NT_V2_250M_NUM_LAYERS = 24


class NTWrapper:
    """Frozen Nucleotide Transformer v2 250M for embedding extraction.

    Uses JAX/Haiku (not nn.Module). All parameters are frozen by default
    since we only call apply() without computing gradients.
    """

    def __init__(
        self,
        model_name: str = "250M_multi_species_v2",
        embedding_layer: int = 24,
        max_positions: int = 128,
    ):
        self.model_name = model_name
        self.embedding_layer = embedding_layer
        self.embed_dim = NT_V2_250M_EMBED_DIM

        parameters, forward_fn, tokenizer, config = get_pretrained_model(
            model_name=model_name,
            embeddings_layers_to_save=(embedding_layer,),
            max_positions=max_positions,
        )
        self.parameters = parameters
        self.forward_fn = hk.transform(forward_fn)
        self.tokenizer = tokenizer
        self.config = config
        self.embed_dim = config.embed_dim
        self._rng = jax.random.PRNGKey(0)

    def tokenize(self, sequences: list[str]) -> jnp.ndarray:
        """Tokenize DNA sequences into token IDs.

        Args:
            sequences: List of DNA strings.

        Returns:
            (batch, max_tokens) int32 array of token IDs.
        """
        tokens_batch = self.tokenizer.batch_tokenize(sequences)
        token_ids = jnp.asarray([t[1] for t in tokens_batch], dtype=jnp.int32)
        return token_ids

    def extract_embeddings(self, sequences: list[str]) -> np.ndarray:
        """Extract mean-pooled embeddings from DNA sequences.

        Args:
            sequences: List of DNA strings (A/C/G/T/N), typically 200bp or 600bp.

        Returns:
            (len(sequences), embed_dim) float32 numpy array of mean-pooled embeddings.
        """
        token_ids = self.tokenize(sequences)

        # Forward pass
        outs = self.forward_fn.apply(self.parameters, self._rng, token_ids)
        emb_key = f"embeddings_{self.embedding_layer}"
        embeddings = outs[emb_key]  # (B, T, D)

        # Remove CLS token (first position)
        emb_no_cls = embeddings[:, 1:, :]
        tokens_no_cls = token_ids[:, 1:]

        # Create padding mask
        pad_mask = jnp.expand_dims(
            tokens_no_cls != self.tokenizer.pad_token_id, axis=-1
        )  # (B, T-1, 1)

        # Masked mean pooling
        masked_emb = emb_no_cls * pad_mask
        seq_lens = jnp.sum(pad_mask, axis=1)  # (B, 1)
        mean_emb = jnp.sum(masked_emb, axis=1) / jnp.maximum(seq_lens, 1)  # (B, D)

        return np.asarray(mean_emb, dtype=np.float32)

    def extract_embeddings_batched(self, sequences: list[str], batch_size: int = 64) -> np.ndarray:
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


def get_nt_embed_dim() -> int:
    """Return the embedding dimension for NT v2 250M."""
    return NT_V2_250M_EMBED_DIM
