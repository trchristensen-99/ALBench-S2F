"""Pre-computed AlphaGenome encoder embedding cache for fast head-only training.

Since the encoder is frozen during training, we can run it once over the entire
dataset, cache the outputs, and then train only the small head network.  This
eliminates the dominant cost of each training step.

Three ``aug_mode`` settings are supported:

* ``"full"``     — No caching.  All augmentation (RC + shift) applied on-the-fly;
                   the encoder runs on every batch.  Original behaviour.
* ``"no_shift"`` — Cache canonical **and** RC embeddings at dataset build time.
                   RC augmentation is still applied per-sequence (50 % prob) by
                   looking up the appropriate cached embedding.  Shift augmentation
                   is disabled.  ~20–50× faster per epoch; ideal for architecture
                   search.
* ``"hybrid"``   — Cache canonical + RC embeddings.  At each training batch a
                   coin flip decides the path:
                     - Heads (50 %) → look up cached embeddings, apply per-seq RC.
                     - Tails (50 %) → run encoder with full shift + RC augmentation.
                   Reproduces the original shift-augmentation distribution in
                   expectation; ~2× faster than ``"full"``.

Storage estimate (K562, ~700k train seqs, T=5, D=1536, float16):
    canonical:  700k × 5 × 1536 × 2 B ≈ 10.75 GB
    rc:         same                   ≈ 10.75 GB
    total:                             ≈ 21.5 GB   ← fits in 96 GB H100 NVL VRAM
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import torch
from alphagenome_ft import custom_heads as custom_heads_module
from alphagenome_ft.embeddings_extended import ExtendedEmbeddings
from alphagenome_research.model import model as model_lib
from tqdm import tqdm

# ── Encoder-only JIT function ─────────────────────────────────────────────────


def build_encoder_fn(model: Any):
    """Return a JIT-compiled encoder-only function.

    Mirrors the mixed-precision setup inside ``create_model_with_heads`` exactly
    (params=float32, compute=bfloat16, output=bfloat16).

    Signature::

        encode(sequences_f32: (B, L, 4)) -> encoder_output_bf16: (B, T, D)

    Encoder params are drawn from ``model._params``; any extra head params
    present there are silently ignored by Haiku.
    """
    import jmp

    policy = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")

    @hk.transform_with_state
    def _encoder_fwd(dna_sequence, organism_index):
        with hk.mixed_precision.push_policy(model_lib.AlphaGenome, policy):
            with hk.mixed_precision.push_policy(model_lib.SequenceEncoder, policy):
                with hk.name_scope("alphagenome"):
                    trunk, _ = model_lib.SequenceEncoder()(dna_sequence)
        return trunk  # (B, T, D) bfloat16

    @jax.jit
    def _run(params, state, seqs, org_idx):
        output, _ = _encoder_fwd.apply(params, state, None, seqs, org_idx)
        return output

    return functools.partial(_run, model._params, model._state)


# ── Cache build / load ────────────────────────────────────────────────────────


def build_embedding_cache(
    model: Any,
    dataset: torch.utils.data.Dataset,
    cache_dir: Path,
    split: str,
    max_seq_len: int = 600,
    batch_size: int = 128,
    num_workers: int = 4,
) -> None:
    """Run the encoder once over ``dataset`` and write canonical + RC caches.

    Output files::

        {cache_dir}/{split}_canonical.npy   shape (N, T, D)  dtype float16
        {cache_dir}/{split}_rc.npy          shape (N, T, D)  dtype float16

    If both files already exist the function returns immediately (idempotent).

    **Cache build time (rough):** One encoder forward per sequence (×2 for canonical + RC).
    With N=627_660 train, batch_size=128 → ~4_902 batches × 2 ≈ 9_804 passes. On an H100,
    expect ~35–60 minutes for train and ~5–10 minutes for val (58k seqs). Total before
    head-only training starts: ~40–70 min.
    """
    out_canonical = cache_dir / f"{split}_canonical.npy"
    out_rc = cache_dir / f"{split}_rc.npy"

    if out_canonical.exists() and out_rc.exists():
        print(f"[EmbeddingCache] {split} cache already exists – skipping build.")
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    encoder_fn = build_encoder_fn(model)
    N = len(dataset)

    # Determine T and D from a single dummy forward pass
    dummy_seq = jnp.zeros((1, max_seq_len, 4), dtype=jnp.float32)
    dummy_org = jnp.zeros((1,), dtype=jnp.int32)
    sample_out = np.array(encoder_fn(dummy_seq, dummy_org))
    T, D = int(sample_out.shape[1]), int(sample_out.shape[2])
    print(
        f"[EmbeddingCache] Encoder output: T={T}, D={D}. "
        f"Building {split} cache for N={N} sequences …"
    )

    # Pre-allocate memory-mapped output files
    canonical_buf = np.lib.format.open_memmap(
        out_canonical, mode="w+", dtype=np.float16, shape=(N, T, D)
    )
    rc_buf = np.lib.format.open_memmap(out_rc, mode="w+", dtype=np.float16, shape=(N, T, D))

    def _collate_cache(batch_items):
        """Return (canonical_seqs, rc_seqs) for a list of (seq, label) pairs."""
        B = len(batch_items)
        x_can = np.zeros((B, max_seq_len, 4), dtype=np.float32)
        x_rc = np.zeros((B, max_seq_len, 4), dtype=np.float32)
        for i, (seq, _label) in enumerate(batch_items):
            seq_np = seq.numpy()[:4, :].T  # (L, 4)
            x_can[i] = seq_np
            x_rc[i] = seq_np[::-1, ::-1]
        return x_can, x_rc

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # preserve index alignment
        num_workers=num_workers,
        collate_fn=_collate_cache,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    ptr = 0
    for x_can, x_rc in tqdm(loader, desc=f"[EmbeddingCache] {split}"):
        B_actual = x_can.shape[0]
        org_idx = jnp.zeros((B_actual,), dtype=jnp.int32)

        emb_can = np.array(encoder_fn(jnp.array(x_can), org_idx))
        emb_rc = np.array(encoder_fn(jnp.array(x_rc), org_idx))

        # Clip to float16 range before casting (bfloat16 → float16 can overflow)
        canonical_buf[ptr : ptr + B_actual] = np.clip(emb_can, -65504, 65504).astype(np.float16)
        rc_buf[ptr : ptr + B_actual] = np.clip(emb_rc, -65504, 65504).astype(np.float16)
        ptr += B_actual

    # Flush to disk
    del canonical_buf, rc_buf
    print(f"[EmbeddingCache] {split} cache saved → {cache_dir}")


def load_embedding_cache(cache_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a pre-built cache as memory-mapped arrays (zero RAM until accessed).

    Returns:
        canonical: (N, T, D) float16 memmap
        rc:        (N, T, D) float16 memmap
    """
    canonical = np.load(cache_dir / f"{split}_canonical.npy", mmap_mode="r")
    rc = np.load(cache_dir / f"{split}_rc.npy", mmap_mode="r")
    return canonical, rc


# ── Head-only JIT functions ───────────────────────────────────────────────────


def build_head_only_predict_fn(
    model: Any,
    head_name: str,
    num_organisms: int = 2,
):
    """Build a JIT-compiled head-only predict function.

    Takes pre-computed encoder embeddings instead of raw DNA sequences,
    bypassing the (frozen) encoder entirely.

    Parameter path ``head/{head_name}/…`` in ``model._params`` is used; the
    encoder params also present in ``model._params`` are ignored by Haiku.

    Args:
        model:         The ``CustomAlphaGenomeModel`` returned by
                       ``create_model_with_heads``.
        head_name:     The registered head name (must match the original
                       ``register_s2f_head`` call).
        num_organisms: Passed to ``create_registered_head``; must match the
                       value used at model creation (typically 2 for all_folds).

    Returns:
        ``head_predict(params, encoder_output_f32, organism_indices)``
            → predictions of shape ``(B,)`` or ``(B, num_tracks)``.
    """

    @hk.transform_with_state
    def _head_fwd(encoder_output, organism_index):
        embeddings = ExtendedEmbeddings(
            embeddings_1bp=None,
            embeddings_128bp=None,
            encoder_output=encoder_output,
        )
        with hk.name_scope("head"):
            head = custom_heads_module.create_registered_head(
                head_name,
                metadata=None,
                num_organisms=num_organisms,
            )
            return head(embeddings, organism_index)

    state = model._state

    @jax.jit
    def head_predict(params, encoder_output, organism_index):
        output, _ = _head_fwd.apply(params, state, None, encoder_output, organism_index)
        return output

    return head_predict


def _get_head_subtree(params: dict) -> dict:
    """Extract the 'head' subtree from Haiku init params, handling transform-name wrapping."""
    if "_head_fwd" in params and isinstance(params.get("_head_fwd"), dict):
        inner = params["_head_fwd"]
        return inner.get("head", inner)
    return params.get("head", params)


def _set_nested(d: dict, keys: list, value: Any) -> dict:
    """Return a new nested dict with ``value`` set at ``keys`` path (immutable-style)."""
    if len(keys) == 1:
        return {k: (value if k == keys[0] else v) for k, v in d.items()}
    if keys[0] not in d or not isinstance(d[keys[0]], dict):
        return d  # path not found – leave unchanged
    return {k: (_set_nested(v, keys[1:], value) if k == keys[0] else v) for k, v in d.items()}


def reinit_head_params(
    model: Any,
    head_name: str,
    num_tokens: int = 5,
    dim: int = 1536,
    num_organisms: int = 2,
    rng: int | None = 0,
) -> None:
    """Re-initialize the head parameters to fix shape mismatch from stale checkpoints.

    ``create_model_with_heads`` initialises the head with a library-chosen dummy
    sequence length that may differ from the training length (e.g. T=128 giving
    ``hidden_0/w`` shape (196608, 512) instead of the correct (7680, 512) for T=5).
    This function re-inits with the correct ``(1, num_tokens, dim)`` encoder output
    and writes the fresh params back into ``model._params``, handling three common
    Haiku/alphagenome_ft parameter tree layouts:

    * **Nested top-level** ``{"head": {...}}``
    * **Nested inside alphagenome** ``{"alphagenome": {"head": {...}}}``
    * **Flat slash-string keys** ``{"head/layer/w": tensor, ...}``

    Call this after ``create_model_with_heads()`` and before ``freeze_except_head()``.
    """
    rng_key = jax.random.PRNGKey(rng) if isinstance(rng, int) else rng
    dummy_encoder_output = jnp.zeros((1, num_tokens, dim), dtype=jnp.float32)
    dummy_organism_index = jnp.zeros((1,), dtype=jnp.int32)

    @hk.transform_with_state
    def _head_fwd(encoder_output, organism_index):
        embeddings = ExtendedEmbeddings(
            embeddings_1bp=None,
            embeddings_128bp=None,
            encoder_output=encoder_output,
        )
        with hk.name_scope("head"):
            head = custom_heads_module.create_registered_head(
                head_name,
                metadata=None,
                num_organisms=num_organisms,
            )
            return head(embeddings, organism_index)

    fresh_params, fresh_state = _head_fwd.init(rng_key, dummy_encoder_output, dummy_organism_index)

    if not isinstance(model._params, dict):
        print("[EmbeddingCache] WARNING: model._params is not a dict; skipping reinit.")
        return

    head_subtree = _get_head_subtree(fresh_params)

    # ── Detect and handle params layout ──────────────────────────────────────
    layout = None

    # Layout 1: top-level "head" key  {"head": {...}, "alphagenome": {...}}
    if "head" in model._params:
        model._params = _set_nested(model._params, ["head"], head_subtree)
        layout = "top-level 'head'"

    # Layout 2: nested  {"alphagenome": {"head": {...}}}
    elif (
        "alphagenome" in model._params
        and isinstance(model._params.get("alphagenome"), dict)
        and "head" in model._params["alphagenome"]
    ):
        model._params = _set_nested(model._params, ["alphagenome", "head"], head_subtree)
        layout = "alphagenome/head nested"

    # Layout 3: flat slash-string keys  {"head/layer/w": tensor, ...}
    elif any(isinstance(k, str) and k.startswith("head/") for k in model._params):
        flat_fresh: dict = {}

        def _flatten(d: dict, prefix: str) -> None:
            for k, v in d.items():
                path = f"{prefix}/{k}"
                if isinstance(v, dict):
                    _flatten(v, path)
                else:
                    flat_fresh[path] = v

        _flatten(head_subtree, "head")
        model._params = {k: flat_fresh.get(k, v) for k, v in model._params.items()}
        layout = "flat slash-string keys"

    if layout is None:
        top_keys = list(model._params.keys())[:8]
        print(
            f"[EmbeddingCache] WARNING: could not locate 'head' in model._params "
            f"(top-level keys: {top_keys}). Head params NOT re-initialised; "
            "shape mismatch may occur."
        )
        return

    # ── Mirror for _state if present ─────────────────────────────────────────
    if getattr(model, "_state", None) and isinstance(model._state, dict):
        state_subtree = _get_head_subtree(fresh_state)
        if "head" in model._state:
            model._state = _set_nested(model._state, ["head"], state_subtree)
        elif (
            "alphagenome" in model._state
            and isinstance(model._state.get("alphagenome"), dict)
            and "head" in model._state["alphagenome"]
        ):
            model._state = _set_nested(model._state, ["alphagenome", "head"], state_subtree)

    print(f"[EmbeddingCache] Re-initialized head params (layout: {layout}).")


# ── Batch helpers ─────────────────────────────────────────────────────────────


def lookup_cached_batch(
    indices: np.ndarray,
    canonical: np.ndarray,
    rc: np.ndarray,
) -> np.ndarray:
    """Return a float32 embedding batch with per-sequence RC augmentation (50 %).

    For each sequence, independently draws a Bernoulli(0.5) sample to decide
    whether to use the canonical or RC cached embedding.

    Args:
        indices:   (B,) integer array of dataset indices.
        canonical: (N, T, D) float16 memmap.
        rc:        (N, T, D) float16 memmap.

    Returns:
        (B, T, D) float32 array ready for JAX.
    """
    B = len(indices)
    T, D = canonical.shape[1], canonical.shape[2]
    embeddings = np.empty((B, T, D), dtype=np.float32)
    use_rc = np.random.rand(B) > 0.5  # (B,) bool
    embeddings[use_rc] = rc[indices[use_rc]].astype(np.float32)
    embeddings[~use_rc] = canonical[indices[~use_rc]].astype(np.float32)
    return embeddings
