#!/usr/bin/env python
"""Verify that Enformer, Borzoi, and Nucleotide Transformer load and run correctly.

Run:
    uv run python scripts/verify_foundation_models.py
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch


def verify_enformer():
    """Verify Enformer loads and produces expected output shapes."""
    print("\n=== ENFORMER ===")
    from enformer_pytorch import Enformer

    t0 = time.time()
    model = Enformer.from_pretrained("EleutherAI/enformer-official-rough")
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Input: (batch, 196608, 4) one-hot, channels-last
    seq_len = 196_608
    x = torch.zeros(1, seq_len, 4)
    # Place a 600bp "sequence" in the center
    center = seq_len // 2
    for i in range(center - 300, center + 300):
        base = i % 4
        x[0, i, base] = 1.0

    with torch.no_grad():
        t0 = time.time()
        emb = model(x, return_only_embeddings=True)
        print(f"  Forward pass: {time.time() - t0:.1f}s")
        print(f"  Embedding shape: {emb.shape}")
        # Expected: (1, 896, 3072)
        assert emb.shape == (1, 896, 3072), f"Unexpected shape: {emb.shape}"

        # Center bin extraction: center of input maps to bin 448
        center_bins = emb[:, 446:450, :]  # 4 center bins
        mean_emb = center_bins.mean(dim=1)  # (1, 3072)
        print(f"  Center 4-bin mean embedding: {mean_emb.shape}")
        print(f"  Embedding range: [{emb.min():.4f}, {emb.max():.4f}]")

    print("  PASS")
    return True


def verify_borzoi():
    """Verify Borzoi loads and produces expected output shapes."""
    print("\n=== BORZOI ===")
    from borzoi_pytorch import Borzoi

    t0 = time.time()
    model = Borzoi.from_pretrained("johahi/borzoi-replicate-0")
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Input: (batch, 4, L) one-hot, channels-first
    # Minimum L such that after downsampling we have >= 6144 bins
    # Safe minimum: 131072 (2^17)
    seq_len = 131_072
    x = torch.zeros(1, 4, seq_len)
    # Place a 600bp "sequence" in the center
    center = seq_len // 2
    for i in range(center - 300, center + 300):
        base = i % 4
        x[0, base, i] = 1.0

    with torch.no_grad():
        t0 = time.time()
        emb = model.get_embs_after_crop(x)
        print(f"  Forward pass: {time.time() - t0:.1f}s")
        print(f"  Embedding shape: {emb.shape}")
        # Expected: (1, 1536, 6144)
        assert emb.shape[0] == 1 and emb.shape[1] == 1536, f"Unexpected shape: {emb.shape}"

        # Mean-pool over sequence dimension
        mean_emb = emb.mean(dim=2)  # (1, 1536)
        print(f"  Mean-pooled embedding: {mean_emb.shape}")
        print(f"  Embedding range: [{emb.min():.4f}, {emb.max():.4f}]")

    print("  PASS")
    return True


def verify_nt():
    """Verify Nucleotide Transformer v2 250M loads and produces expected output shapes."""
    print("\n=== NUCLEOTIDE TRANSFORMER v2 250M ===")
    import haiku as hk
    import jax
    import jax.numpy as jnp
    from nucleotide_transformer.pretrained import get_pretrained_model

    t0 = time.time()
    parameters, forward_fn, tokenizer, config = get_pretrained_model(
        model_name="250M_multi_species_v2",
        embeddings_layers_to_save=(24,),
        max_positions=128,
    )
    forward_fn = hk.transform(forward_fn)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Count parameters
    n_params = sum(x.size for x in jax.tree.leaves(parameters))
    print(f"  Parameters: {n_params:,}")
    print(f"  Embed dim: {config.embed_dim}")
    print(f"  Num layers: {config.num_layers}")

    # Tokenize a test sequence (200bp)
    test_seq = "ACGTACGTACGT" * 16 + "ACGTACGT"  # 200bp
    tokens_batch = tokenizer.batch_tokenize([test_seq])
    token_ids = jnp.asarray([t[1] for t in tokens_batch], dtype=jnp.int32)
    print(f"  Input tokens shape: {token_ids.shape}")

    # Forward pass
    rng = jax.random.PRNGKey(0)
    t0 = time.time()
    outs = forward_fn.apply(parameters, rng, token_ids)
    print(f"  Forward pass: {time.time() - t0:.1f}s")

    emb = outs["embeddings_24"]
    print(f"  Embedding shape: {emb.shape}")
    # Expected: (1, T, 1280) where T = num_tokens

    # Remove CLS token and apply padding mask
    emb_no_cls = emb[:, 1:, :]
    pad_mask = jnp.expand_dims(token_ids[:, 1:] != tokenizer.pad_token_id, axis=-1)
    masked_emb = emb_no_cls * pad_mask
    seq_lens = jnp.sum(pad_mask, axis=1)
    mean_emb = jnp.sum(masked_emb, axis=1) / seq_lens
    print(f"  Mean-pooled embedding: {mean_emb.shape}")
    print(f"  Embedding range: [{float(emb.min()):.4f}, {float(emb.max()):.4f}]")

    print("  PASS")
    return True


def main():
    results = {}

    for name, fn in [("Enformer", verify_enformer), ("Borzoi", verify_borzoi), ("NT", verify_nt)]:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  FAIL: {e}")
            results[name] = False

    print("\n=== SUMMARY ===")
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
